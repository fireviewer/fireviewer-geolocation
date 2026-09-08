from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from itertools import pairwise
from typing import Literal, Protocol

from pydantic import Field

from fireviewer_contracts.contracts import StrictModel
from fireviewer_contracts.mvp.contracts import (
    Detection,
    DetectionResultV1,
    EventEvidenceV1,
    GeographicAbstention,
    GeographicHypothesis,
    GeographicHypothesisResultV1,
    GeographicReference,
    GeographicScoreBreakdown,
    ProviderRun,
    UploadLocationEvidence,
    VisualObservation,
)
from fireviewer_geolocation.spatial_geometry import (
    SpatialGeometryError,
    TerrainSurface,
    wgs84_to_map,
)

_EARTH_RADIUS_M = 6_371_008.8


class TerrainElevationProvider(Protocol):
    reference_revision: str
    resolution_m: float

    def elevation_m(self, longitude: float, latitude: float) -> float | None: ...


class GeographicHypothesisConfig(StrictModel):
    maximum_camera_distance_m: float = Field(default=100_000, gt=100, le=200_000)
    minimum_bearing_tolerance_deg: float = Field(default=2, gt=0, le=45)
    bearing_gate_multiplier: float = Field(default=2, ge=1, le=5)
    camera_height_above_ground_m: float = Field(default=1.7, gt=0, le=20)
    target_height_above_ground_m: float = Field(default=1, ge=0, le=100)
    terrain_clearance_m: float = Field(default=1, ge=0, le=100)
    maximum_profile_samples: int = Field(default=512, ge=16, le=4_096)
    satellite_max_age_hours: float = Field(default=48, gt=0, le=720)
    base_history_tolerance_m: float = Field(default=500, ge=0, le=20_000)
    maximum_spread_rate_m_per_hour: float = Field(default=2_000, gt=0, le=50_000)
    history_reverse_limit_deg: float = Field(default=100, gt=45, le=180)
    minimum_progression_baseline_m: float = Field(default=50, ge=0, le=10_000)
    minimum_hypothesis_score: float = Field(default=0.4, ge=0, le=1)
    max_hypotheses_per_detection: int = Field(default=3, ge=1, le=16)


@dataclass(frozen=True, slots=True)
class TerrainVisibility:
    supported: bool
    score: float
    reason_code: str
    minimum_clearance_m: float | None
    camera_altitude_m: float | None
    target_altitude_m: float | None


@dataclass(frozen=True, slots=True)
class _DetectionContext:
    observation: VisualObservation
    result: DetectionResultV1
    detection: Detection


@dataclass(frozen=True, slots=True)
class _CandidateEvaluation:
    hypothesis: GeographicHypothesis | None
    reason_code: str


class TerrainSurfaceElevationProvider:
    """Read an immutable metric terrain surface without modifying its producer."""

    def __init__(
        self,
        surface: TerrainSurface,
        *,
        reference_revision: str,
    ) -> None:
        if not reference_revision:
            raise ValueError("terrain reference revision is required")
        self._surface = surface
        self.reference_revision = reference_revision
        self.resolution_m = surface.resolution_m

    def elevation_m(self, longitude: float, latitude: float) -> float | None:
        try:
            east, north = wgs84_to_map(longitude, latitude, map_crs=self._surface.crs)
        except SpatialGeometryError:
            return None
        return self._surface.sample(east, north)


def _stable_id(prefix: str, *parts: str) -> str:
    digest = sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _canonical_sha256(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _haversine_m(left: tuple[float, float], right: tuple[float, float]) -> float:
    left_lon, left_lat = (math.radians(value) for value in left)
    right_lon, right_lat = (math.radians(value) for value in right)
    delta_lon = right_lon - left_lon
    delta_lat = right_lat - left_lat
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(left_lat) * math.cos(right_lat) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(value))


def _initial_bearing_deg(origin: tuple[float, float], target: tuple[float, float]) -> float:
    origin_lon, origin_lat = (math.radians(value) for value in origin)
    target_lon, target_lat = (math.radians(value) for value in target)
    delta_lon = target_lon - origin_lon
    east = math.sin(delta_lon) * math.cos(target_lat)
    north = (
        math.cos(origin_lat) * math.sin(target_lat)
        - math.sin(origin_lat) * math.cos(target_lat) * math.cos(delta_lon)
    )
    return math.degrees(math.atan2(east, north)) % 360


def _angular_delta_deg(left: float, right: float) -> float:
    return abs((left - right + 180) % 360 - 180)


def _paths(geometry: dict[str, object]) -> tuple[tuple[tuple[float, float], ...], ...]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")

    def point(value: object) -> tuple[float, float]:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            raise ValueError("invalid geographic reference point")
        return float(value[0]), float(value[1])

    if geometry_type == "Point":
        return ((point(coordinates),),)
    if not isinstance(coordinates, (list, tuple)):
        raise ValueError("invalid geographic reference coordinates")
    if geometry_type in {"MultiPoint", "LineString"}:
        points = tuple(point(value) for value in coordinates)
        return tuple((value,) for value in points) if geometry_type == "MultiPoint" else (points,)
    if geometry_type in {"MultiLineString", "Polygon"}:
        return tuple(tuple(point(value) for value in path) for path in coordinates)
    if geometry_type == "MultiPolygon":
        return tuple(
            tuple(point(value) for value in path)
            for polygon in coordinates
            for path in polygon
        )
    raise ValueError("unsupported geographic reference geometry")


def _point_in_ring(point: tuple[float, float], ring: tuple[tuple[float, float], ...]) -> bool:
    if len(ring) < 3:
        return False
    x, y = point
    inside = False
    previous = ring[-1]
    for current in ring:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def _inside_reference(point: tuple[float, float], reference: GeographicReference) -> bool:
    geometry_type = reference.geometry_geojson["type"]
    if geometry_type not in {"Polygon", "MultiPolygon"}:
        return False
    return any(_point_in_ring(point, ring) for ring in _paths(reference.geometry_geojson))


def _reference_points(reference: GeographicReference) -> tuple[tuple[float, float], ...]:
    paths = _paths(reference.geometry_geojson)
    unique = {point for path in paths for point in path}
    geometry_type = reference.geometry_geojson["type"]
    if geometry_type in {"Polygon", "MultiPolygon"} and unique:
        centroid = (
            sum(point[0] for point in unique) / len(unique),
            sum(point[1] for point in unique) / len(unique),
        )
        if _inside_reference(centroid, reference):
            unique.add(centroid)
    return tuple(sorted(unique))


def _local_xy(origin: tuple[float, float], point: tuple[float, float]) -> tuple[float, float]:
    origin_lon, origin_lat = origin
    longitude, latitude = point
    x = math.radians(longitude - origin_lon) * _EARTH_RADIUS_M * math.cos(
        math.radians((origin_lat + latitude) / 2)
    )
    y = math.radians(latitude - origin_lat) * _EARTH_RADIUS_M
    return x, y


def _segment_distance_m(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    start_x, start_y = _local_xy(point, start)
    end_x, end_y = _local_xy(point, end)
    segment_x = end_x - start_x
    segment_y = end_y - start_y
    denominator = segment_x * segment_x + segment_y * segment_y
    if denominator <= 1e-12:
        return math.hypot(start_x, start_y)
    position = max(
        0.0,
        min(1.0, -(start_x * segment_x + start_y * segment_y) / denominator),
    )
    return math.hypot(start_x + position * segment_x, start_y + position * segment_y)


def _distance_to_reference_m(
    point: tuple[float, float],
    reference: GeographicReference,
) -> float:
    if _inside_reference(point, reference):
        return 0.0
    distances: list[float] = []
    for path in _paths(reference.geometry_geojson):
        if len(path) == 1:
            distances.append(_haversine_m(point, path[0]))
            continue
        distances.extend(
            _segment_distance_m(point, start, end)
            for start, end in pairwise(path)
        )
    return min(distances, default=math.inf)


def _reference_center(reference: GeographicReference) -> tuple[float, float]:
    points = _reference_points(reference)
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


class GeographicHypothesisEngine:
    provider_id = "deterministic-geographic-hypotheses"
    provider_version = "1.0.0"

    def __init__(
        self,
        terrain: TerrainElevationProvider | None,
        *,
        config: GeographicHypothesisConfig | None = None,
    ) -> None:
        if terrain is not None and (
            not terrain.reference_revision or terrain.resolution_m <= 0
        ):
            raise ValueError("terrain provider requires a revision and positive resolution")
        self.terrain = terrain
        self.config = config or GeographicHypothesisConfig()

    def locate(
        self,
        event: EventEvidenceV1,
        *,
        vision_artifacts: tuple[DetectionResultV1, ...],
        upload_locations: tuple[UploadLocationEvidence, ...],
        geographic_references: tuple[GeographicReference, ...],
        source_revision_sha256: str,
        generated_at: datetime,
    ) -> GeographicHypothesisResultV1:
        contexts = self._detection_contexts(event, vision_artifacts)
        input_hash = _canonical_sha256(
            {
                "event": event.model_dump(mode="json", by_alias=True),
                "vision_artifacts": [
                    item.model_dump(mode="json", by_alias=True) for item in vision_artifacts
                ],
                "upload_locations": [item.model_dump(mode="json") for item in upload_locations],
                "geographic_references": [
                    item.model_dump(mode="json") for item in geographic_references
                ],
                "source_revision_sha256": source_revision_sha256,
                "terrain_revision": (
                    self.terrain.reference_revision if self.terrain is not None else None
                ),
                "config": self.config.model_dump(mode="json"),
            }
        )
        hypotheses: list[GeographicHypothesis] = []
        abstentions: list[GeographicAbstention] = []
        if not contexts:
            abstentions.append(GeographicAbstention(reason_codes=("missing_visual_detection",)))
        locations_by_media = {item.media_id: item for item in upload_locations}
        satellite_references = tuple(
            item
            for item in geographic_references
            if item.reference_kind in {"satellite_hotspot", "satellite_active_area"}
        )
        history_references = tuple(
            item
            for item in geographic_references
            if item.reference_kind
            in {"prior_active_point", "prior_fire_front", "prior_perimeter"}
        )

        for context in contexts:
            location = locations_by_media.get(context.observation.media_id)
            base_reasons: set[str] = set()
            if location is None:
                base_reasons.add("missing_upload_location")
            elif location.heading_deg is None or location.horizontal_fov_deg is None:
                base_reasons.add("missing_camera_orientation")
            if not satellite_references:
                base_reasons.add("missing_satellite_reference")
            if self.terrain is None:
                base_reasons.add("missing_terrain_reference")
            if base_reasons:
                abstentions.append(
                    GeographicAbstention(
                        observation_id=context.observation.observation_id,
                        detection_id=context.detection.detection_id,
                        media_id=context.observation.media_id,
                        reason_codes=tuple(sorted(base_reasons)),
                    )
                )
                continue
            assert location is not None
            evaluations = tuple(
                self._evaluate(
                    context,
                    location=location,
                    satellite_reference=reference,
                    point=point,
                    history_references=history_references,
                    event_observed_at=event.time_window.to_at or event.time_window.from_at,
                )
                for reference in satellite_references
                for point in _reference_points(reference)
            )
            accepted = sorted(
                (item.hypothesis for item in evaluations if item.hypothesis is not None),
                key=lambda item: (-item.score, item.hypothesis_id),
            )[: self.config.max_hypotheses_per_detection]
            if accepted:
                hypotheses.extend(accepted)
                continue
            reasons = {item.reason_code for item in evaluations}
            abstentions.append(
                GeographicAbstention(
                    observation_id=context.observation.observation_id,
                    detection_id=context.detection.detection_id,
                    media_id=context.observation.media_id,
                    reason_codes=tuple(sorted(reasons or {"no_supported_geographic_hypothesis"})),
                )
            )

        provider_run = ProviderRun(
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            config={
                **self.config.model_dump(mode="json"),
                "terrain_reference_revision": (
                    self.terrain.reference_revision if self.terrain is not None else None
                ),
                "terrain_resolution_m": (
                    self.terrain.resolution_m if self.terrain is not None else None
                ),
                "coordinate_policy": "satellite_seed_camera_bearing_terrain_los_history_gate",
            },
            input_hash=input_hash,
            runtime_ms=0,
            cost_usd=0,
            generated_at=generated_at,
        )
        return GeographicHypothesisResultV1(
            event_id=event.event_id,
            source_event_evidence_sha256=source_revision_sha256,
            status="hypotheses" if hypotheses else "abstained",
            hypotheses=tuple(hypotheses),
            abstentions=tuple(abstentions),
            provider_run=provider_run,
        )

    @staticmethod
    def _detection_contexts(
        event: EventEvidenceV1,
        artifacts: tuple[DetectionResultV1, ...],
    ) -> tuple[_DetectionContext, ...]:
        media_by_id = {item.media_id: item for item in event.media}
        artifacts_by_media: dict[str, DetectionResultV1] = {}
        for artifact in artifacts:
            if artifact.media_id in artifacts_by_media:
                raise ValueError("multiple visual artifacts reference one media item")
            media = media_by_id.get(artifact.media_id)
            if media is None or artifact.provider_run.input_hash != media.sha256:
                raise ValueError("visual artifact does not match EventEvidence media")
            artifacts_by_media[artifact.media_id] = artifact
        contexts: list[_DetectionContext] = []
        for observation in event.visual_observations:
            if observation.observation_type != "detection":
                continue
            matched_artifact = artifacts_by_media.get(observation.media_id)
            if matched_artifact is None:
                raise ValueError("visual observation has no digest-qualified artifact")
            contexts.extend(
                _DetectionContext(
                    observation=observation,
                    result=matched_artifact,
                    detection=detection,
                )
                for detection in matched_artifact.detections
            )
        return tuple(contexts)

    def _evaluate(
        self,
        context: _DetectionContext,
        *,
        location: UploadLocationEvidence,
        satellite_reference: GeographicReference,
        point: tuple[float, float],
        history_references: tuple[GeographicReference, ...],
        event_observed_at: datetime | None,
    ) -> _CandidateEvaluation:
        assert location.heading_deg is not None
        assert location.horizontal_fov_deg is not None
        assert self.terrain is not None
        terrain_provider = self.terrain
        origin = (location.longitude, location.latitude)
        distance_m = _haversine_m(origin, point)
        if (
            distance_m <= max(location.accuracy_m, 1)
            or distance_m > self.config.maximum_camera_distance_m
        ):
            return _CandidateEvaluation(None, "camera_distance_out_of_range")
        left, _top, right, bottom = context.detection.bbox
        source_point = ((left + right) / 2, bottom)
        expected_bearing = (
            location.heading_deg
            + (source_point[0] - 0.5) * location.horizontal_fov_deg
        ) % 360
        actual_bearing = _initial_bearing_deg(origin, point)
        bearing_delta = _angular_delta_deg(expected_bearing, actual_bearing)
        bbox_half_width_deg = (right - left) * location.horizontal_fov_deg / 2
        bearing_tolerance = max(
            self.config.minimum_bearing_tolerance_deg,
            location.heading_uncertainty_deg or 0,
            bbox_half_width_deg,
        )
        if bearing_delta > bearing_tolerance * self.config.bearing_gate_multiplier:
            return _CandidateEvaluation(None, "camera_bearing_contradicted")
        bearing_score = max(
            0.0,
            1.0 - bearing_delta / (bearing_tolerance * self.config.bearing_gate_multiplier),
        )
        terrain = self._terrain_visibility(
            origin,
            point,
            camera_altitude_m=location.altitude_m,
        )
        if not terrain.supported:
            return _CandidateEvaluation(None, terrain.reason_code)

        temporal_score = self._temporal_score(
            satellite_reference.observed_at,
            event_observed_at,
        )
        if temporal_score == 0:
            return _CandidateEvaluation(None, "satellite_reference_outside_time_window")
        history_score, history_reason = self._history_score(
            point,
            event_observed_at=event_observed_at,
            references=history_references,
        )
        if history_reason == "history_progression_contradicted":
            return _CandidateEvaluation(None, history_reason)

        satellite_score = satellite_reference.confidence or 0.75
        values = {
            "visual": (context.detection.score, 0.20),
            "camera": (bearing_score, 0.30),
            "terrain": (terrain.score, 0.20),
            "satellite": (satellite_score, 0.20),
        }
        if temporal_score is not None:
            values["temporal"] = (temporal_score, 0.05)
        if history_score is not None:
            values["history"] = (history_score, 0.05)
        total_weight = sum(weight for _, weight in values.values())
        score = sum(value * weight for value, weight in values.values()) / total_weight
        if score < self.config.minimum_hypothesis_score:
            return _CandidateEvaluation(None, "geographic_score_below_review_gate")

        lateral_uncertainty_m = distance_m * math.sin(math.radians(bearing_tolerance))
        uncertainty_m = math.sqrt(
            location.accuracy_m**2
            + (location.altitude_uncertainty_m or 0) ** 2
            + (satellite_reference.horizontal_uncertainty_m or terrain_provider.resolution_m) ** 2
            + lateral_uncertainty_m**2
            + terrain_provider.resolution_m**2
        )
        supporting_ids = [satellite_reference.reference_id]
        if history_score is not None:
            supporting_ids.extend(item.reference_id for item in history_references)
        reason_codes = [
            "visual_box_bearing_supported",
            "terrain_line_of_sight_supported",
            "satellite_reference_supported",
            history_reason,
        ]
        if location.altitude_m is None:
            reason_codes.append("camera_altitude_derived_from_terrain")
        phenomenon: Literal["active_fire_point", "smoke_origin"] = (
            "active_fire_point"
            if context.detection.detection_class == "fire"
            else "smoke_origin"
        )
        hypothesis = GeographicHypothesis(
            hypothesis_id=_stable_id(
                "GEO",
                context.observation.observation_id,
                context.detection.detection_id,
                satellite_reference.reference_id,
                f"{point[0]:.8f}",
                f"{point[1]:.8f}",
            ),
            observation_id=context.observation.observation_id,
            detection_id=context.detection.detection_id,
            media_id=context.observation.media_id,
            phenomenon=phenomenon,
            longitude=point[0],
            latitude=point[1],
            geometry_geojson={"type": "Point", "coordinates": [point[0], point[1]]},
            horizontal_uncertainty_m=min(max(uncertainty_m, 1), 100_000),
            camera_bearing_deg=expected_bearing,
            camera_distance_m=distance_m,
            source_point_normalized=source_point,
            score=score,
            score_breakdown=GeographicScoreBreakdown(
                visual=context.detection.score,
                camera_bearing=bearing_score,
                terrain_visibility=terrain.score,
                satellite=satellite_score,
                temporal_alignment=temporal_score,
                history_progression=history_score,
            ),
            supporting_reference_ids=tuple(sorted(set(supporting_ids))),
            reason_codes=tuple(sorted(set(reason_codes))),
        )
        return _CandidateEvaluation(hypothesis, "geographic_hypothesis_supported")

    def _terrain_visibility(
        self,
        origin: tuple[float, float],
        target: tuple[float, float],
        *,
        camera_altitude_m: float | None,
    ) -> TerrainVisibility:
        assert self.terrain is not None
        terrain_provider = self.terrain
        distance_m = _haversine_m(origin, target)
        spacing_m = max(terrain_provider.resolution_m, 20)
        sample_count = min(
            self.config.maximum_profile_samples,
            max(16, math.ceil(distance_m / spacing_m) + 1),
        )
        samples: list[float] = []
        for index in range(sample_count):
            fraction = index / (sample_count - 1)
            longitude = origin[0] + (target[0] - origin[0]) * fraction
            latitude = origin[1] + (target[1] - origin[1]) * fraction
            elevation = terrain_provider.elevation_m(longitude, latitude)
            if elevation is None or not math.isfinite(elevation):
                return TerrainVisibility(
                    supported=False,
                    score=0,
                    reason_code="terrain_profile_unavailable",
                    minimum_clearance_m=None,
                    camera_altitude_m=None,
                    target_altitude_m=None,
                )
            samples.append(elevation)
        camera_z = (
            samples[0] + self.config.camera_height_above_ground_m
            if camera_altitude_m is None
            else camera_altitude_m
        )
        target_z = samples[-1] + self.config.target_height_above_ground_m
        clearances = [
            camera_z + (target_z - camera_z) * (index / (sample_count - 1)) - elevation
            for index, elevation in enumerate(samples[1:-1], start=1)
        ]
        minimum_clearance = min(clearances, default=math.inf)
        if minimum_clearance < self.config.terrain_clearance_m:
            return TerrainVisibility(
                supported=False,
                score=0,
                reason_code="terrain_line_of_sight_blocked",
                minimum_clearance_m=minimum_clearance,
                camera_altitude_m=camera_z,
                target_altitude_m=target_z,
            )
        score = min(1.0, 0.5 + minimum_clearance / max(20, distance_m * 0.01))
        return TerrainVisibility(
            supported=True,
            score=score,
            reason_code="terrain_line_of_sight_supported",
            minimum_clearance_m=minimum_clearance,
            camera_altitude_m=camera_z,
            target_altitude_m=target_z,
        )

    def _temporal_score(
        self,
        reference_time: datetime | None,
        event_time: datetime | None,
    ) -> float | None:
        if reference_time is None or event_time is None:
            return None
        age_hours = abs((event_time - reference_time).total_seconds()) / 3_600
        if age_hours > self.config.satellite_max_age_hours:
            return 0.0
        return 1.0 - age_hours / self.config.satellite_max_age_hours

    def _history_score(
        self,
        point: tuple[float, float],
        *,
        event_observed_at: datetime | None,
        references: tuple[GeographicReference, ...],
    ) -> tuple[float | None, str]:
        timed = tuple(
            sorted(
                (item for item in references if item.observed_at is not None),
                key=lambda item: (item.observed_at, item.reference_id),
            )
        )
        if not references or event_observed_at is None or not timed:
            return None, "history_unavailable"
        latest = timed[-1]
        assert latest.observed_at is not None
        elapsed_hours = max(
            0.0,
            (event_observed_at - latest.observed_at).total_seconds() / 3_600,
        )
        allowed_distance = (
            self.config.base_history_tolerance_m
            + elapsed_hours * self.config.maximum_spread_rate_m_per_hour
            + (latest.horizontal_uncertainty_m or 0)
        )
        distance_m = _distance_to_reference_m(point, latest)
        if distance_m > allowed_distance:
            return 0.0, "history_progression_contradicted"
        distance_score = max(0.0, 1.0 - distance_m / max(allowed_distance, 1))
        if len(timed) < 2:
            return distance_score, "history_distance_supported"
        previous = timed[-2]
        previous_center = _reference_center(previous)
        latest_center = _reference_center(latest)
        progression_distance = _haversine_m(previous_center, latest_center)
        candidate_distance = _haversine_m(latest_center, point)
        if (
            progression_distance < self.config.minimum_progression_baseline_m
            or candidate_distance < self.config.minimum_progression_baseline_m
        ):
            return distance_score, "history_distance_supported"
        progression_bearing = _initial_bearing_deg(previous_center, latest_center)
        candidate_bearing = _initial_bearing_deg(latest_center, point)
        delta = _angular_delta_deg(progression_bearing, candidate_bearing)
        if delta > self.config.history_reverse_limit_deg:
            return 0.0, "history_progression_contradicted"
        direction_score = max(0.0, 1.0 - delta / self.config.history_reverse_limit_deg)
        return (distance_score + direction_score) / 2, "history_direction_supported"


__all__ = [
    "GeographicHypothesisConfig",
    "GeographicHypothesisEngine",
    "TerrainElevationProvider",
    "TerrainSurfaceElevationProvider",
    "TerrainVisibility",
]
