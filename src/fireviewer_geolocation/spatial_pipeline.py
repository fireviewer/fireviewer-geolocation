from __future__ import annotations

import gc
import json
import math
import os
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal

from fireviewer_contracts.contracts import (
    SourceAnnotationV2,
    SpatialProposalV2,
    SpatialReferenceAssetV2,
    WorkerBatchItemV2,
    WorkerInputV2,
    WorkerModelRunV2,
    WorkerStageAttemptV2,
    WorkerStageGateV2,
    WorkerStageTraceV2,
)
from fireviewer_contracts.client.media_fetcher import MediaFetcher, MediaFetchError
from fireviewer_geolocation.roma_registration import (
    ROMA_SOURCE_REVISION,
    load_roma_model,
    match_pair,
    roma_root_from_environment,
)
from fireviewer_geolocation.spatial_geometry import (
    CameraPoseSolution,
    CrossViewMapCrop,
    FWTerrainSurface,
    SpatialGeometryError,
    annotation_ray_direction,
    crop_georeferenced_map,
    cross_view_search_radii,
    load_fwterrain,
    map_to_wgs84,
    metadata_camera_pose,
    select_consistent_cross_view_pose,
    solve_pnp_pose,
    wgs84_to_map,
)

ROMA_MODEL_ID = "AerialExtreMatch-RoMa"


def _now() -> datetime:
    return datetime.now(UTC)


def _stable_id(prefix: str, *parts: str) -> str:
    from hashlib import sha256

    digest = sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{digest}"


@dataclass(frozen=True, slots=True)
class SpatialPipelineExecution:
    proposals_by_input: Mapping[str, tuple[SpatialProposalV2, ...]]
    stage_traces: tuple[WorkerStageTraceV2, ...]
    model_runs: tuple[WorkerModelRunV2, ...]


@dataclass(frozen=True, slots=True)
class _ReferenceWorkspace:
    terrain: FWTerrainSurface
    orthophoto_path: Path | None
    orthophoto_bounds: tuple[float, float, float, float] | None


def _gate(
    *,
    phase: Literal["preflight", "postflight"],
    decision: Literal[
        "pass",
        "not_applicable",
        "abstain",
        "human_review",
        "failed_retryable",
        "failed_terminal",
    ],
    reason: str,
    available: tuple[str, ...] = (),
    missing: tuple[str, ...] = (),
    downstream_possible: bool = True,
) -> WorkerStageGateV2:
    return WorkerStageGateV2(
        phase=phase,
        decision=decision,
        reason_codes=(reason,),
        available_capabilities=available,
        missing_capabilities=missing,
        downstream_possible=downstream_possible,
    )


def _skipped_trace(
    *,
    role: Literal["cross_view_registration", "spatial_projection"],
    sequence: int,
    decision: Literal["not_applicable", "human_review", "abstain", "failed_terminal"],
    reason: str,
    available: tuple[str, ...] = (),
    missing: tuple[str, ...] = (),
) -> WorkerStageTraceV2:
    return WorkerStageTraceV2(
        stage_role=role,
        contract_id=f"stage.{role}.v1",
        sequence=sequence,
        status="skipped",
        retryable=False,
        preflight=_gate(
            phase="preflight",
            decision=decision,
            reason=reason,
            available=available,
            missing=missing,
            downstream_possible=decision != "failed_terminal",
        ),
    )


def _successful_trace(
    *,
    role: Literal["cross_view_registration", "spatial_projection"],
    sequence: int,
    started_at: datetime,
    finished_at: datetime,
    elapsed_ms: int,
    available_before: tuple[str, ...],
    available_after: tuple[str, ...],
) -> WorkerStageTraceV2:
    return WorkerStageTraceV2(
        stage_role=role,
        contract_id=f"stage.{role}.v1",
        sequence=sequence,
        status="succeeded",
        retryable=False,
        preflight=_gate(
            phase="preflight",
            decision="pass",
            reason="requirements_satisfied",
            available=available_before,
        ),
        postflight=_gate(
            phase="postflight",
            decision="pass",
            reason="minimum_output_satisfied",
            available=available_after,
        ),
        attempts=(
            WorkerStageAttemptV2(
                attempt=1,
                kind="initial",
                status="succeeded",
                started_at=started_at,
                finished_at=finished_at,
                inference_ms=elapsed_ms,
            ),
        ),
    )


def _failed_trace(
    *,
    role: Literal["cross_view_registration", "spatial_projection"],
    sequence: int,
    started_at: datetime,
    finished_at: datetime,
    elapsed_ms: int,
    error_code: str,
    retryable: bool,
) -> WorkerStageTraceV2:
    decision: Literal["failed_retryable", "failed_terminal"] = (
        "failed_retryable" if retryable else "failed_terminal"
    )
    return WorkerStageTraceV2(
        stage_role=role,
        contract_id=f"stage.{role}.v1",
        sequence=sequence,
        status="failed",
        retryable=retryable,
        preflight=_gate(
            phase="preflight",
            decision="pass",
            reason="requirements_satisfied",
        ),
        postflight=_gate(
            phase="postflight",
            decision=decision,
            reason=error_code,
            downstream_possible=False,
        ),
        attempts=(
            WorkerStageAttemptV2(
                attempt=1,
                kind="initial",
                status="failed",
                started_at=started_at,
                finished_at=finished_at,
                inference_ms=elapsed_ms,
                error_code=error_code,
            ),
        ),
    )


def _asset(batch: WorkerInputV2, kind: str) -> SpatialReferenceAssetV2 | None:
    if batch.reference_bundle is None:
        return None
    return next(
        (candidate for candidate in batch.reference_bundle.assets if candidate.kind == kind), None
    )


def _catalog_profile(
    path: Path,
    *,
    terrain_asset: SpatialReferenceAssetV2,
    orthophoto_asset: SpatialReferenceAssetV2,
) -> tuple[float, float, float, float]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpatialGeometryError("scene_catalog_invalid") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "fireviewer.remote-tile-catalog.v1"
        or payload.get("crs") != "EPSG:2154"
    ):
        raise SpatialGeometryError("scene_catalog_profile_invalid")
    try:
        far = payload["lod_policy"]["far"]
        bounds = far["bounds_l93_m"]
        catalog_terrain_sha = far["terrain"]["sha256"]
        catalog_imagery_sha = far["imagery"]["sha256"]
    except (KeyError, TypeError) as exc:
        raise SpatialGeometryError("scene_catalog_far_reference_missing") from exc
    if (
        not isinstance(bounds, list)
        or len(bounds) != 4
        or catalog_terrain_sha != terrain_asset.sha256
        or catalog_imagery_sha != orthophoto_asset.sha256
    ):
        raise SpatialGeometryError("scene_catalog_asset_mismatch")
    numeric_bounds = (
        float(bounds[0]),
        float(bounds[1]),
        float(bounds[2]),
        float(bounds[3]),
    )
    left, bottom, right, top = numeric_bounds
    if not all(math.isfinite(value) for value in numeric_bounds) or not (
        left < right and bottom < top
    ):
        raise SpatialGeometryError("scene_catalog_bounds_invalid")
    return numeric_bounds


def _camera_has_direct_pose(item: WorkerBatchItemV2) -> bool:
    camera = item.camera
    return (
        camera is not None
        and camera.pose_origin == "HUMAN_CONFIRMED"
        and all(
            value is not None
            for value in (
                camera.longitude,
                camera.latitude,
                camera.orthometric_height_m,
                camera.yaw_deg,
                camera.pitch_deg,
                camera.roll_deg,
                camera.horizontal_fov_deg,
                camera.image_width_px,
                camera.image_height_px,
            )
        )
    )


def _camera_can_cross_view(item: WorkerBatchItemV2) -> bool:
    camera = item.camera
    return camera is not None and all(
        value is not None
        for value in (
            camera.longitude,
            camera.latitude,
            camera.horizontal_fov_deg,
            camera.image_width_px,
            camera.image_height_px,
        )
    )


def _evidence_url(item: WorkerBatchItemV2, annotation: SourceAnnotationV2) -> str | None:
    if annotation.evidence_id == item.input_id and item.working_file_url is not None:
        return str(item.working_file_url)
    frame = next(
        (candidate for candidate in item.frames if candidate.frame_id == annotation.evidence_id),
        None,
    )
    return str(frame.working_file_url) if frame is not None else None


def _release_roma_model(model: object) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return


class DeterministicSpatialPipeline:
    def __init__(
        self,
        *,
        fetcher: MediaFetcher,
        enable_cross_view: bool | None = None,
        roma_root: Path | None = None,
    ) -> None:
        self.fetcher = fetcher
        self.enable_cross_view = (
            os.getenv("FW_ENABLE_ROMA_CROSS_VIEW", "false").strip().lower()
            in {"1", "true", "yes", "on"}
            if enable_cross_view is None
            else enable_cross_view
        )
        self.roma_root = roma_root or roma_root_from_environment()

    def _prepare_reference(
        self,
        batch: WorkerInputV2,
        stack: ExitStack,
        *,
        require_orthophoto: bool,
    ) -> _ReferenceWorkspace:
        terrain_asset = _asset(batch, "terrain_mnt")
        if terrain_asset is None:
            raise SpatialGeometryError("terrain_reference_missing")
        try:
            terrain_path = stack.enter_context(
                self.fetcher.download_verified(
                    str(terrain_asset.working_file_url), expected_sha256=terrain_asset.sha256
                )
            )
            terrain = load_fwterrain(terrain_path, declared_crs=terrain_asset.crs)
        except MediaFetchError as exc:
            raise SpatialGeometryError("terrain_reference_digest_mismatch") from exc
        if not require_orthophoto:
            return _ReferenceWorkspace(
                terrain=terrain,
                orthophoto_path=None,
                orthophoto_bounds=None,
            )
        orthophoto_asset = _asset(batch, "orthophoto")
        catalog_asset = _asset(batch, "scene_catalog")
        if orthophoto_asset is None:
            raise SpatialGeometryError("orthophoto_reference_missing")
        if catalog_asset is None:
            raise SpatialGeometryError("scene_catalog_missing")
        try:
            orthophoto_path = stack.enter_context(
                self.fetcher.download_verified(
                    str(orthophoto_asset.working_file_url), expected_sha256=orthophoto_asset.sha256
                )
            )
            catalog_path = stack.enter_context(
                self.fetcher.download_verified(
                    str(catalog_asset.working_file_url), expected_sha256=catalog_asset.sha256
                )
            )
        except MediaFetchError as exc:
            raise SpatialGeometryError("reference_asset_digest_mismatch") from exc
        bounds = _catalog_profile(
            catalog_path,
            terrain_asset=terrain_asset,
            orthophoto_asset=orthophoto_asset,
        )
        return _ReferenceWorkspace(
            terrain=terrain,
            orthophoto_path=orthophoto_path,
            orthophoto_bounds=bounds,
        )

    def _run_cross_view(
        self,
        *,
        batch: WorkerInputV2,
        annotations_by_input: Mapping[str, tuple[SourceAnnotationV2, ...]],
        reference: _ReferenceWorkspace,
        stack: ExitStack,
    ) -> tuple[dict[str, CameraPoseSolution], dict[str, str], WorkerModelRunV2]:
        from PIL import Image

        if reference.orthophoto_path is None or reference.orthophoto_bounds is None:
            raise SpatialGeometryError("orthophoto_reference_missing")
        load_started = perf_counter()
        started_at = _now()
        model = load_roma_model(self.roma_root)
        load_ms = round((perf_counter() - load_started) * 1_000)
        inference_started = perf_counter()
        poses: dict[str, CameraPoseSolution] = {}
        errors: dict[str, str] = {}
        try:
            with Image.open(reference.orthophoto_path) as map_handle:
                map_image = map_handle.convert("RGB")
            for item in batch.items:
                annotations = annotations_by_input.get(item.input_id, ())
                if (
                    not annotations
                    or _camera_has_direct_pose(item)
                    or not _camera_can_cross_view(item)
                ):
                    continue
                source_url = _evidence_url(item, annotations[0])
                if source_url is None:
                    errors[item.input_id] = "source_visual_missing"
                    continue
                try:
                    source_path = stack.enter_context(self.fetcher.download(source_url))
                    assert item.camera is not None
                    assert item.camera.longitude is not None
                    assert item.camera.latitude is not None
                    east, north = wgs84_to_map(
                        item.camera.longitude,
                        item.camera.latitude,
                        map_crs=reference.terrain.crs,
                    )
                    prior_height = item.camera.orthometric_height_m
                    if prior_height is None:
                        ground = reference.terrain.sample(east, north)
                        if ground is None:
                            raise SpatialGeometryError("camera_outside_terrain")
                        prior_height = ground + 2.0
                    prior_limit_m = max(
                        100.0,
                        min(3_000.0, (item.camera.horizontal_accuracy_m or 250.0) * 3.0),
                    )
                    candidates: list[tuple[float, CameraPoseSolution]] = []
                    scale_errors: list[str] = []
                    for radius_m in cross_view_search_radii(item.camera.horizontal_accuracy_m):
                        crop: CrossViewMapCrop | None = None
                        try:
                            crop = crop_georeferenced_map(
                                map_image,
                                map_bounds=reference.orthophoto_bounds,
                                centre_east_m=east,
                                centre_north_m=north,
                                radius_m=radius_m,
                            )
                            matches = match_pair(model, source_path, crop.image)
                            candidates.append(
                                (
                                    crop.scale_radius_m,
                                    solve_pnp_pose(
                                        source_pixels=matches.source_pixels,
                                        map_pixels=matches.map_pixels,
                                        certainties=matches.certainties,
                                        map_image_size=crop.image.size,  # type: ignore[attr-defined]
                                        map_bounds=crop.bounds_m,
                                        terrain=reference.terrain,
                                        camera=item.camera,
                                        prior_camera_center=(east, north, prior_height),
                                        maximum_prior_distance_m=prior_limit_m,
                                    ),
                                )
                            )
                        except SpatialGeometryError as exc:
                            scale_errors.append(exc.code)
                        finally:
                            if crop is not None:
                                crop.image.close()  # type: ignore[attr-defined]
                    if not candidates and scale_errors:
                        raise SpatialGeometryError(scale_errors[-1])
                    poses[item.input_id] = select_consistent_cross_view_pose(
                        candidates,
                        horizontal_accuracy_m=item.camera.horizontal_accuracy_m,
                    )
                except (MediaFetchError, SpatialGeometryError) as exc:
                    errors[item.input_id] = (
                        exc.code
                        if isinstance(exc, SpatialGeometryError)
                        else "source_visual_fetch_failed"
                    )
            map_image.close()
        finally:
            _release_roma_model(model)
        inference_ms = round((perf_counter() - inference_started) * 1_000)
        finished_at = _now()
        run = WorkerModelRunV2(
            model_role="cross_view_registration",
            model_id=ROMA_MODEL_ID,
            revision=ROMA_SOURCE_REVISION,
            status="succeeded",
            started_at=started_at,
            finished_at=finished_at,
            load_ms=load_ms,
            inference_ms=inference_ms,
        )
        return poses, errors, run

    @staticmethod
    def _accuracy_m(
        item: WorkerBatchItemV2,
        pose: CameraPoseSolution,
        hit_distance_m: float,
        terrain_resolution_m: float,
    ) -> float:
        camera = item.camera
        assert camera is not None
        base_accuracy = camera.horizontal_accuracy_m or 50.0
        orientation_error_deg = {
            "HUMAN_CONFIRMED": 2.0,
            "METADATA": 5.0,
            "USER_DECLARED": 8.0,
            "CROSS_VIEW_ESTIMATE": 3.0,
        }.get(camera.pose_origin or "", 8.0)
        if pose.origin == "CROSS_VIEW_RAYCAST" and pose.p95_reprojection_error_px is not None:
            orientation_error_deg = max(1.0, min(8.0, pose.p95_reprojection_error_px / 2.0))
        angular_component = hit_distance_m * math.tan(math.radians(orientation_error_deg))
        accuracy = math.sqrt(
            base_accuracy**2 + (terrain_resolution_m * 2.0) ** 2 + angular_component**2
        )
        return max(1.0, min(100_000.0, accuracy))

    def project(
        self,
        batch: WorkerInputV2,
        annotations_by_input: Mapping[str, tuple[SourceAnnotationV2, ...]],
        *,
        sequence_start: int,
    ) -> SpatialPipelineExecution:
        terrestrial = {
            item.input_id: annotations_by_input.get(item.input_id, ())
            for item in batch.items
            if item.media_type.value != "satellite_image"
            and annotations_by_input.get(item.input_id, ())
        }
        if not terrestrial:
            return SpatialPipelineExecution(
                proposals_by_input={},
                stage_traces=(
                    _skipped_trace(
                        role="cross_view_registration",
                        sequence=sequence_start,
                        decision="not_applicable",
                        reason="no_applicable_input",
                    ),
                    _skipped_trace(
                        role="spatial_projection",
                        sequence=sequence_start + 1,
                        decision="not_applicable",
                        reason="no_applicable_input",
                    ),
                ),
                model_runs=(),
            )

        needs_cross_view = any(
            not _camera_has_direct_pose(item)
            for item in batch.items
            if item.input_id in terrestrial
        )
        poses: dict[str, CameraPoseSolution] = {}
        pose_errors: dict[str, str] = {}
        traces: list[WorkerStageTraceV2] = []
        model_runs: list[WorkerModelRunV2] = []
        proposals: dict[str, tuple[SpatialProposalV2, ...]] = {}
        reference_error: str | None = None

        with ExitStack() as stack:
            try:
                reference = self._prepare_reference(
                    batch,
                    stack,
                    require_orthophoto=needs_cross_view and self.enable_cross_view,
                )
            except SpatialGeometryError as exc:
                reference = None
                reference_error = exc.code

            if not needs_cross_view:
                traces.append(
                    _skipped_trace(
                        role="cross_view_registration",
                        sequence=sequence_start,
                        decision="not_applicable",
                        reason="camera_pose_already_supplied",
                        available=("fire_point_pixel", "camera_pose", "reference_bundle"),
                    )
                )
            elif reference_error is not None:
                traces.append(
                    _skipped_trace(
                        role="cross_view_registration",
                        sequence=sequence_start,
                        decision="human_review",
                        reason=reference_error,
                        available=("fire_point_pixel",),
                        missing=("reference_bundle",),
                    )
                )
            elif not self.enable_cross_view:
                traces.append(
                    _skipped_trace(
                        role="cross_view_registration",
                        sequence=sequence_start,
                        decision="human_review",
                        reason="cross_view_benchmark_not_approved",
                        available=("fire_point_pixel", "reference_bundle"),
                        missing=("camera_pose",),
                    )
                )
                moment = _now()
                model_runs.append(
                    WorkerModelRunV2(
                        model_role="cross_view_registration",
                        model_id=ROMA_MODEL_ID,
                        revision=ROMA_SOURCE_REVISION,
                        status="skipped",
                        started_at=moment,
                        finished_at=moment,
                        load_ms=0,
                        inference_ms=0,
                        error_code="cross_view_benchmark_not_approved",
                    )
                )
            else:
                assert reference is not None
                cross_started_at = _now()
                cross_started = perf_counter()
                try:
                    poses, pose_errors, run = self._run_cross_view(
                        batch=batch,
                        annotations_by_input=annotations_by_input,
                        reference=reference,
                        stack=stack,
                    )
                    model_runs.append(run)
                    cross_finished_at = _now()
                    traces.append(
                        _successful_trace(
                            role="cross_view_registration",
                            sequence=sequence_start,
                            started_at=cross_started_at,
                            finished_at=cross_finished_at,
                            elapsed_ms=round((perf_counter() - cross_started) * 1_000),
                            available_before=("fire_point_pixel", "reference_bundle"),
                            available_after=(
                                "camera_pose" if poses else "explicit_abstention",
                                "fire_point_pixel",
                                "reference_bundle",
                                "spatial_matches",
                            ),
                        )
                    )
                except (MediaFetchError, SpatialGeometryError) as exc:
                    code = (
                        exc.code
                        if isinstance(exc, SpatialGeometryError)
                        else "reference_fetch_failed"
                    )
                    reference_error = code
                    cross_finished_at = _now()
                    traces.append(
                        _failed_trace(
                            role="cross_view_registration",
                            sequence=sequence_start,
                            started_at=cross_started_at,
                            finished_at=cross_finished_at,
                            elapsed_ms=round((perf_counter() - cross_started) * 1_000),
                            error_code=code,
                            retryable=isinstance(exc, MediaFetchError),
                        )
                    )

            projection_started_at = _now()
            projection_started = perf_counter()
            any_projection_attempted = False
            for item in batch.items:
                annotations = terrestrial.get(item.input_id)
                if not annotations:
                    continue
                item_proposals: list[SpatialProposalV2] = []
                pose: CameraPoseSolution | None = poses.get(item.input_id)
                if pose is None and _camera_has_direct_pose(item) and reference is not None:
                    assert item.camera is not None
                    try:
                        pose = metadata_camera_pose(item.camera, map_crs=reference.terrain.crs)
                    except SpatialGeometryError as exc:
                        pose_errors[item.input_id] = exc.code
                for annotation in annotations:
                    any_projection_attempted = True
                    if annotation.source_point_normalized is None:
                        item_proposals.append(
                            self._abstention(item, annotation, "source_point_missing")
                        )
                        continue
                    if reference is None:
                        code = reference_error or "terrain_reference_missing"
                        item_proposals.append(self._abstention(item, annotation, code))
                        continue
                    if pose is None:
                        code = pose_errors.get(
                            item.input_id,
                            "cross_view_benchmark_not_approved"
                            if needs_cross_view and not self.enable_cross_view
                            else "camera_pose_missing",
                        )
                        item_proposals.append(self._abstention(item, annotation, code))
                        continue
                    assert item.camera is not None
                    try:
                        direction = annotation_ray_direction(
                            annotation.source_point_normalized,
                            camera=item.camera,
                            pose=pose,
                        )
                        hit = reference.terrain.intersect_ray(pose.camera_center, direction)
                        longitude, latitude = map_to_wgs84(
                            hit.east_m,
                            hit.north_m,
                            map_crs=reference.terrain.crs,
                        )
                        accuracy = self._accuracy_m(
                            item,
                            pose,
                            hit.distance_m,
                            reference.terrain.resolution_m,
                        )
                        if accuracy > 5_000.0:
                            raise SpatialGeometryError("horizontal_accuracy_excessive")
                        assert batch.reference_bundle is not None
                        item_proposals.append(
                            SpatialProposalV2(
                                proposal_id=_stable_id("SP", annotation.annotation_id, pose.origin),
                                annotation_id=annotation.annotation_id,
                                status="projected_geometry",
                                proposal_kind=(
                                    "smoke_origin_point"
                                    if annotation.semantic_anchor
                                    in {"smoke_column_base", "smoke_origin_point"}
                                    else "active_fire_point"
                                ),
                                observed_at=item.captured_at,
                                geometry_origin=pose.origin,
                                longitude=longitude,
                                latitude=latitude,
                                altitude_m=hit.altitude_m,
                                geometry_geojson={
                                    "type": "Point",
                                    "coordinates": [longitude, latitude],
                                },
                                horizontal_accuracy_m=accuracy,
                                reference_bundle_sha256=batch.reference_bundle.manifest_sha256,
                                uncertainty_codes=(
                                    f"pose_{(item.camera.pose_origin or 'unknown').lower()}",
                                    "terrain_far_mnt",
                                ),
                            )
                        )
                    except SpatialGeometryError as exc:
                        item_proposals.append(self._abstention(item, annotation, exc.code))
                proposals[item.input_id] = tuple(item_proposals)

            projection_finished_at = _now()
            projection_elapsed_ms = round((perf_counter() - projection_started) * 1_000)
            if not any_projection_attempted:
                traces.append(
                    _skipped_trace(
                        role="spatial_projection",
                        sequence=sequence_start + 1,
                        decision="not_applicable",
                        reason="no_applicable_input",
                    )
                )
            elif reference is None:
                traces.append(
                    _skipped_trace(
                        role="spatial_projection",
                        sequence=sequence_start + 1,
                        decision="human_review",
                        reason=reference_error or "terrain_reference_missing",
                        available=("fire_point_pixel",),
                        missing=("camera_pose", "terrain_reference"),
                    )
                )
            else:
                traces.append(
                    _successful_trace(
                        role="spatial_projection",
                        sequence=sequence_start + 1,
                        started_at=projection_started_at,
                        finished_at=projection_finished_at,
                        elapsed_ms=projection_elapsed_ms,
                        available_before=("camera_pose", "fire_point_pixel", "terrain_reference"),
                        available_after=(
                            "explicit_abstention"
                            if not any(
                                proposal.status == "projected_geometry"
                                for group in proposals.values()
                                for proposal in group
                            )
                            else "spatial_proposals",
                            "terrain_reference",
                        ),
                    )
                )

        return SpatialPipelineExecution(
            proposals_by_input=proposals,
            stage_traces=tuple(traces),
            model_runs=tuple(model_runs),
        )

    @staticmethod
    def _abstention(
        item: WorkerBatchItemV2,
        annotation: SourceAnnotationV2,
        code: str,
    ) -> SpatialProposalV2:
        return SpatialProposalV2(
            proposal_id=_stable_id("SP", annotation.annotation_id, "abstain", code),
            annotation_id=annotation.annotation_id,
            status="insufficient_geometry",
            observed_at=item.captured_at,
            uncertainty_codes=(code,),
        )
