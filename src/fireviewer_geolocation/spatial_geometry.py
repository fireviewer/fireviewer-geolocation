from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from fireviewer_contracts.contracts import CameraMetadataV2

FWTILE_MAGIC = b"FWTILE1\0"
FWTILE_VERSION = 1
WGS84_CRS = "EPSG:4326"


class SpatialGeometryError(RuntimeError):
    """A deterministic geometry operation could not produce a defensible point."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class RaycastHit:
    east_m: float
    north_m: float
    altitude_m: float
    distance_m: float


@dataclass(frozen=True, slots=True)
class CameraPoseSolution:
    camera_center: Any
    camera_to_world: Any
    origin: Literal["CAMERA_RAYCAST", "CROSS_VIEW_RAYCAST"]
    inlier_count: int | None = None
    inlier_ratio: float | None = None
    median_reprojection_error_px: float | None = None
    p95_reprojection_error_px: float | None = None


class TerrainSurface(Protocol):
    """Minimal metric terrain contract shared by production and held-out evaluation."""

    @property
    def crs(self) -> str: ...

    @property
    def resolution_m(self) -> float: ...

    def sample(self, east_m: float, north_m: float) -> float | None: ...

    def sample_many(self, eastings: Any, northings: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class CrossViewMapCrop:
    image: object
    bounds_m: tuple[float, float, float, float]
    scale_radius_m: float


@dataclass(frozen=True, slots=True)
class FWTerrainSurface:
    elevations_m: Any
    valid_mask: Any
    bounds_l93_m: tuple[float, float, float, float]
    sample_spacing_m: tuple[float, float]
    crs: str
    sample_centres: bool

    @property
    def rows(self) -> int:
        return int(self.elevations_m.shape[0])

    @property
    def columns(self) -> int:
        return int(self.elevations_m.shape[1])

    @property
    def resolution_m(self) -> float:
        return max(self.sample_spacing_m)

    def sample_many(self, eastings: Any, northings: Any) -> Any:
        import numpy as np

        east = np.asarray(eastings, dtype=np.float64)
        north = np.asarray(northings, dtype=np.float64)
        left, _bottom, _right, top = self.bounds_l93_m
        spacing_x, spacing_y = self.sample_spacing_m
        offset = 0.5 if self.sample_centres else 0.0
        columns = np.rint((east - left) / spacing_x - offset).astype(np.int64)
        rows = np.rint((top - north) / spacing_y - offset).astype(np.int64)
        inside = (rows >= 0) & (rows < self.rows) & (columns >= 0) & (columns < self.columns)
        result = np.full(east.shape, np.nan, dtype=np.float64)
        if not bool(inside.any()):
            return result
        inside_indexes = np.flatnonzero(inside.reshape(-1))
        flat_rows = rows.reshape(-1)[inside_indexes]
        flat_columns = columns.reshape(-1)[inside_indexes]
        usable = self.valid_mask[flat_rows, flat_columns]
        usable_indexes = inside_indexes[np.asarray(usable, dtype=bool)]
        if usable_indexes.size:
            result.reshape(-1)[usable_indexes] = self.elevations_m[
                rows.reshape(-1)[usable_indexes], columns.reshape(-1)[usable_indexes]
            ]
        return result

    def sample(self, east_m: float, north_m: float) -> float | None:
        import numpy as np

        value = float(
            self.sample_many(
                np.asarray([east_m], dtype=np.float64),
                np.asarray([north_m], dtype=np.float64),
            )[0]
        )
        return value if math.isfinite(value) else None

    def intersect_ray(
        self,
        camera_center: Any,
        direction_world: Any,
        *,
        maximum_distance_m: float = 50_000.0,
    ) -> RaycastHit:
        import numpy as np

        center = np.asarray(camera_center, dtype=np.float64).reshape(3)
        direction = np.asarray(direction_world, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(direction))
        if not math.isfinite(norm) or norm <= 1e-9:
            raise SpatialGeometryError("camera_ray_invalid")
        direction /= norm
        camera_ground = self.sample(float(center[0]), float(center[1]))
        if camera_ground is None:
            raise SpatialGeometryError("camera_outside_terrain")
        if float(center[2]) <= camera_ground + 0.5:
            raise SpatialGeometryError("camera_below_terrain")

        step_m = max(2.0, min(20.0, self.resolution_m * 2.0))
        distances = np.arange(step_m, maximum_distance_m + step_m, step_m)
        points = center[None, :] + distances[:, None] * direction[None, :]
        elevations = self.sample_many(points[:, 0], points[:, 1])
        finite = np.isfinite(elevations)
        if not bool(finite.any()):
            raise SpatialGeometryError("ray_outside_terrain")
        first_invalid = int(np.flatnonzero(~finite)[0]) if bool((~finite).any()) else len(finite)
        usable_count = first_invalid
        if usable_count == 0:
            raise SpatialGeometryError("ray_outside_terrain")
        clearances = points[:usable_count, 2] - elevations[:usable_count]
        intersections = np.flatnonzero(clearances <= 0.0)
        if intersections.size == 0:
            code = (
                "ray_left_terrain"
                if first_invalid < len(finite)
                else "terrain_intersection_missing"
            )
            raise SpatialGeometryError(code)
        hit_index = int(intersections[0])
        lower_t = 0.0 if hit_index == 0 else float(distances[hit_index - 1])
        upper_t = float(distances[hit_index])
        for _ in range(24):
            midpoint = (lower_t + upper_t) / 2.0
            candidate = center + midpoint * direction
            terrain_height = self.sample(float(candidate[0]), float(candidate[1]))
            if terrain_height is None:
                raise SpatialGeometryError("terrain_nodata_on_ray")
            if float(candidate[2]) > terrain_height:
                lower_t = midpoint
            else:
                upper_t = midpoint
        distance_m = (lower_t + upper_t) / 2.0
        candidate = center + distance_m * direction
        terrain_height = self.sample(float(candidate[0]), float(candidate[1]))
        if terrain_height is None:
            raise SpatialGeometryError("terrain_nodata_on_ray")
        return RaycastHit(
            east_m=float(candidate[0]),
            north_m=float(candidate[1]),
            altitude_m=terrain_height,
            distance_m=distance_m,
        )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _decompress_exact(value: bytes, expected_size: int) -> bytes:
    decompressor = zlib.decompressobj()
    result = decompressor.decompress(value, expected_size + 1)
    if len(result) > expected_size or decompressor.unconsumed_tail:
        raise SpatialGeometryError("terrain_section_size_exceeded")
    result += decompressor.flush(expected_size - len(result) + 1)
    if len(result) != expected_size or decompressor.unconsumed_tail or not decompressor.eof:
        raise SpatialGeometryError("terrain_section_compression_invalid")
    return result


def load_fwterrain(path: Path, *, declared_crs: str | None = None) -> FWTerrainSurface:
    """Decode the immutable FAR MNT used by the real FireViewer Unity packages."""

    import numpy as np

    payload = path.read_bytes()
    if len(payload) < 16 or payload[:8] != FWTILE_MAGIC:
        raise SpatialGeometryError("terrain_container_invalid")
    version = struct.unpack_from("<H", payload, 8)[0]
    header_length = struct.unpack_from("<I", payload, 12)[0]
    if version != FWTILE_VERSION or header_length <= 0 or 16 + header_length > len(payload):
        raise SpatialGeometryError("terrain_container_version_invalid")
    try:
        header = json.loads(payload[16 : 16 + header_length])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpatialGeometryError("terrain_header_invalid") from exc
    if (
        not isinstance(header, dict)
        or header.get("kind") != "global_far_terrain"
        or header.get("crs") != "EPSG:2154"
        or header.get("linear_unit") != "metre"
    ):
        raise SpatialGeometryError("terrain_spatial_profile_invalid")
    if declared_crs is not None and horizontal_crs_identifier(declared_crs) != "EPSG:2154":
        raise SpatialGeometryError("terrain_declared_crs_mismatch")
    sections = header.get("sections")
    if not isinstance(sections, list):
        raise SpatialGeometryError("terrain_section_missing")
    terrain_section = next(
        (
            section
            for section in sections
            if isinstance(section, dict) and section.get("name") == "terrain"
        ),
        None,
    )
    if terrain_section is None or terrain_section.get("codec") != "zlib":
        raise SpatialGeometryError("terrain_section_missing")
    body_offset = 16 + header_length
    stored_offset = terrain_section.get("offset_bytes")
    stored_bytes = terrain_section.get("stored_bytes")
    raw_bytes = terrain_section.get("raw_bytes")
    if not all(isinstance(value, int) for value in (stored_offset, stored_bytes, raw_bytes)):
        raise SpatialGeometryError("terrain_section_invalid")
    assert isinstance(stored_offset, int)
    assert isinstance(stored_bytes, int)
    assert isinstance(raw_bytes, int)
    start = body_offset + stored_offset
    end = start + stored_bytes
    if stored_offset < 0 or stored_bytes <= 0 or end > len(payload):
        raise SpatialGeometryError("terrain_section_truncated")
    stored = payload[start:end]
    if _sha256_bytes(stored) != terrain_section.get("stored_sha256"):
        raise SpatialGeometryError("terrain_stored_digest_mismatch")
    try:
        raw = _decompress_exact(stored, raw_bytes)
    except zlib.error as exc:
        raise SpatialGeometryError("terrain_section_compression_invalid") from exc
    if len(raw) != raw_bytes or _sha256_bytes(raw) != terrain_section.get("raw_sha256"):
        raise SpatialGeometryError("terrain_raw_digest_mismatch")
    metadata = terrain_section.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("encoding") != "masked-regular-grid-z-u16.v1":
        raise SpatialGeometryError("terrain_encoding_unsupported")
    rows = metadata.get("rows")
    columns = metadata.get("columns")
    mask_offset = metadata.get("validity_mask_offset_bytes")
    valid_sample_count = metadata.get("valid_sample_count")
    if not all(
        isinstance(value, int) for value in (rows, columns, mask_offset, valid_sample_count)
    ):
        raise SpatialGeometryError("terrain_dimensions_invalid")
    assert isinstance(rows, int)
    assert isinstance(columns, int)
    assert isinstance(mask_offset, int)
    assert isinstance(valid_sample_count, int)
    count = rows * columns
    expected_raw_bytes = count * 2 + math.ceil(count / 8)
    if rows <= 0 or columns <= 0 or mask_offset != count * 2 or len(raw) != expected_raw_bytes:
        raise SpatialGeometryError("terrain_dimensions_invalid")
    quantization = metadata.get("elevation_quantization")
    spacing = metadata.get("sample_spacing_m")
    bounds = metadata.get("outer_bounds_l93_m")
    origin = header.get("origin_l93_m")
    if not (
        isinstance(quantization, dict)
        and isinstance(spacing, list)
        and len(spacing) == 2
        and isinstance(bounds, list)
        and len(bounds) == 4
        and isinstance(origin, list)
        and len(origin) == 3
    ):
        raise SpatialGeometryError("terrain_metadata_invalid")
    minimum_raw = quantization.get("minimum_m")
    step_raw = quantization.get("step_m")
    if not isinstance(minimum_raw, (int, float)) or not isinstance(step_raw, (int, float)):
        raise SpatialGeometryError("terrain_quantization_invalid")
    minimum = float(minimum_raw)
    step = float(step_raw)
    origin_z = float(origin[2])
    if not all(math.isfinite(value) for value in (minimum, step, origin_z)) or step < 0:
        raise SpatialGeometryError("terrain_quantization_invalid")
    numeric_spacing = (float(spacing[0]), float(spacing[1]))
    numeric_bounds = (
        float(bounds[0]),
        float(bounds[1]),
        float(bounds[2]),
        float(bounds[3]),
    )
    if (
        not all(math.isfinite(value) and value > 0 for value in numeric_spacing)
        or not math.isclose(
            numeric_bounds[2] - numeric_bounds[0],
            columns * numeric_spacing[0],
            rel_tol=1e-6,
            abs_tol=1e-3,
        )
        or not math.isclose(
            numeric_bounds[3] - numeric_bounds[1],
            rows * numeric_spacing[1],
            rel_tol=1e-6,
            abs_tol=1e-3,
        )
    ):
        raise SpatialGeometryError("terrain_grid_profile_invalid")
    encoded = np.frombuffer(raw, dtype="<u2", count=count).reshape(rows, columns)
    elevations = (minimum + encoded.astype(np.float32) * step + origin_z).astype(np.float32)
    mask_bytes = np.frombuffer(raw[mask_offset:], dtype=np.uint8)
    valid = np.unpackbits(mask_bytes, bitorder="little")[:count].reshape(rows, columns).astype(bool)
    if int(valid.sum()) != valid_sample_count:
        raise SpatialGeometryError("terrain_validity_mask_invalid")
    return FWTerrainSurface(
        elevations_m=elevations,
        valid_mask=valid,
        bounds_l93_m=numeric_bounds,
        sample_spacing_m=numeric_spacing,
        crs="EPSG:2154",
        sample_centres=bool(metadata.get("sample_centres")),
    )


def horizontal_crs_identifier(value: str) -> str:
    try:
        from pyproj import CRS
    except ImportError as exc:  # pragma: no cover - explicit image packaging failure
        raise SpatialGeometryError("crs_runtime_unavailable") from exc
    try:
        crs = CRS.from_user_input(value)
    except Exception as exc:
        raise SpatialGeometryError("crs_invalid") from exc
    if crs.is_compound:
        horizontal = next(
            (
                candidate
                for candidate in crs.sub_crs_list
                if candidate.is_projected or candidate.is_geographic
            ),
            None,
        )
        if horizontal is None:
            raise SpatialGeometryError("horizontal_crs_missing")
        crs = horizontal
    authority = crs.to_authority()
    return f"{authority[0]}:{authority[1]}" if authority is not None else crs.to_string()


def wgs84_to_map(longitude: float, latitude: float, *, map_crs: str) -> tuple[float, float]:
    try:
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover - explicit image packaging failure
        raise SpatialGeometryError("crs_runtime_unavailable") from exc
    horizontal = horizontal_crs_identifier(map_crs)
    try:
        east, north = Transformer.from_crs(WGS84_CRS, horizontal, always_xy=True).transform(
            longitude, latitude
        )
    except Exception as exc:
        raise SpatialGeometryError("camera_crs_transform_failed") from exc
    if not all(math.isfinite(value) for value in (east, north)):
        raise SpatialGeometryError("camera_crs_transform_failed")
    return float(east), float(north)


def map_to_wgs84(east_m: float, north_m: float, *, map_crs: str) -> tuple[float, float]:
    try:
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover - explicit image packaging failure
        raise SpatialGeometryError("crs_runtime_unavailable") from exc
    horizontal = horizontal_crs_identifier(map_crs)
    try:
        longitude, latitude = Transformer.from_crs(horizontal, WGS84_CRS, always_xy=True).transform(
            east_m, north_m
        )
    except Exception as exc:
        raise SpatialGeometryError("proposal_crs_transform_failed") from exc
    if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
        raise SpatialGeometryError("proposal_wgs84_out_of_bounds")
    return float(longitude), float(latitude)


def camera_intrinsics(camera: CameraMetadataV2) -> Any:
    import numpy as np

    if (
        camera.horizontal_fov_deg is None
        or camera.image_width_px is None
        or camera.image_height_px is None
    ):
        raise SpatialGeometryError("camera_intrinsics_missing")
    width = float(camera.image_width_px)
    height = float(camera.image_height_px)
    focal = width / (2.0 * math.tan(math.radians(camera.horizontal_fov_deg) / 2.0))
    if not math.isfinite(focal) or focal <= 0:
        raise SpatialGeometryError("camera_intrinsics_invalid")
    return np.asarray(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def metadata_camera_pose(camera: CameraMetadataV2, *, map_crs: str) -> CameraPoseSolution:
    """Build ENU pose: yaw clockwise from north, pitch up, roll clockwise in view."""

    import numpy as np

    required = (
        camera.longitude,
        camera.latitude,
        camera.orthometric_height_m,
        camera.yaw_deg,
        camera.pitch_deg,
        camera.roll_deg,
    )
    if any(value is None for value in required):
        raise SpatialGeometryError("camera_pose_incomplete")
    assert camera.longitude is not None
    assert camera.latitude is not None
    assert camera.orthometric_height_m is not None
    assert camera.yaw_deg is not None
    assert camera.pitch_deg is not None
    assert camera.roll_deg is not None
    east, north = wgs84_to_map(camera.longitude, camera.latitude, map_crs=map_crs)
    yaw = math.radians(camera.yaw_deg)
    pitch = math.radians(camera.pitch_deg)
    roll = math.radians(camera.roll_deg)
    forward = np.asarray(
        [math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)],
        dtype=np.float64,
    )
    right = np.asarray([math.cos(yaw), -math.sin(yaw), 0.0], dtype=np.float64)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    down = -up
    rolled_right = math.cos(roll) * right + math.sin(roll) * down
    rolled_down = -math.sin(roll) * right + math.cos(roll) * down
    camera_to_world = np.column_stack((rolled_right, rolled_down, forward))
    return CameraPoseSolution(
        camera_center=np.asarray([east, north, camera.orthometric_height_m], dtype=np.float64),
        camera_to_world=camera_to_world,
        origin="CAMERA_RAYCAST",
    )


def annotation_ray_direction(
    source_point_normalized: tuple[float, float],
    *,
    camera: CameraMetadataV2,
    pose: CameraPoseSolution,
) -> Any:
    import numpy as np

    intrinsics = camera_intrinsics(camera)
    assert camera.image_width_px is not None
    assert camera.image_height_px is not None
    pixel = np.asarray(
        [
            source_point_normalized[0] * camera.image_width_px,
            source_point_normalized[1] * camera.image_height_px,
            1.0,
        ],
        dtype=np.float64,
    )
    direction_camera = np.linalg.inv(intrinsics) @ pixel
    direction_world = pose.camera_to_world @ direction_camera
    norm = float(np.linalg.norm(direction_world))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise SpatialGeometryError("camera_ray_invalid")
    return direction_world / norm


def cross_view_search_radii(horizontal_accuracy_m: float | None) -> tuple[float, ...]:
    """Return bounded, materially distinct search radii around a camera position prior."""

    uncertainty = horizontal_accuracy_m or 250.0
    base = max(500.0, min(2_000.0, uncertainty * 3.0))
    return tuple(sorted({base, min(5_000.0, base * 2.0), min(5_000.0, base * 4.0)}))


def crop_georeferenced_map(
    map_image: object,
    *,
    map_bounds: tuple[float, float, float, float],
    centre_east_m: float,
    centre_north_m: float,
    radius_m: float,
) -> CrossViewMapCrop:
    """Crop an image while preserving the exact metric bounds represented by its pixels."""

    if radius_m <= 0 or not math.isfinite(radius_m):
        raise SpatialGeometryError("cross_view_crop_invalid")
    width, height = map_image.size  # type: ignore[attr-defined]
    left, bottom, right, top = map_bounds
    if width < 2 or height < 2 or not (left < right and bottom < top):
        raise SpatialGeometryError("cross_view_crop_invalid")
    desired_left = max(left, centre_east_m - radius_m)
    desired_right = min(right, centre_east_m + radius_m)
    desired_bottom = max(bottom, centre_north_m - radius_m)
    desired_top = min(top, centre_north_m + radius_m)
    pixel_left = max(0, math.floor((desired_left - left) / (right - left) * width))
    pixel_right = min(width, math.ceil((desired_right - left) / (right - left) * width))
    pixel_top = max(0, math.floor((top - desired_top) / (top - bottom) * height))
    pixel_bottom = min(height, math.ceil((top - desired_bottom) / (top - bottom) * height))
    if pixel_right - pixel_left < 128 or pixel_bottom - pixel_top < 128:
        raise SpatialGeometryError("cross_view_crop_too_small")
    actual_bounds = (
        left + pixel_left / width * (right - left),
        top - pixel_bottom / height * (top - bottom),
        left + pixel_right / width * (right - left),
        top - pixel_top / height * (top - bottom),
    )
    horizontal_span = actual_bounds[2] - actual_bounds[0]
    vertical_span = actual_bounds[3] - actual_bounds[1]
    return CrossViewMapCrop(
        image=map_image.crop((pixel_left, pixel_top, pixel_right, pixel_bottom)),  # type: ignore[attr-defined]
        bounds_m=actual_bounds,
        scale_radius_m=max(horizontal_span, vertical_span) / 2.0,
    )


def _pose_orientation_delta_degrees(left: CameraPoseSolution, right: CameraPoseSolution) -> float:
    import numpy as np

    left_forward = np.asarray(left.camera_to_world, dtype=np.float64)[:, 2]
    right_forward = np.asarray(right.camera_to_world, dtype=np.float64)[:, 2]
    denominator = float(np.linalg.norm(left_forward) * np.linalg.norm(right_forward))
    if denominator <= 1e-12:
        return math.inf
    cosine = float(np.clip(np.dot(left_forward, right_forward) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def select_consistent_cross_view_pose(
    candidates: list[tuple[float, CameraPoseSolution]],
    *,
    horizontal_accuracy_m: float | None,
) -> CameraPoseSolution:
    """Admit a pose only when two materially different map scales agree."""

    import numpy as np

    if len(candidates) < 2:
        raise SpatialGeometryError("cross_view_multiscale_confirmation_missing")
    centre_limit_m = max(75.0, min(750.0, (horizontal_accuracy_m or 250.0) * 2.0))
    agreeing_pairs: list[tuple[float, float, int, int]] = []
    for left_index in range(len(candidates)):
        for right_index in range(left_index + 1, len(candidates)):
            left_radius, left_pose = candidates[left_index]
            right_radius, right_pose = candidates[right_index]
            if min(left_radius, right_radius) <= 0:
                continue
            scale_separation = abs(
                math.log2(max(left_radius, right_radius) / min(left_radius, right_radius))
            )
            if scale_separation < 0.75:
                continue
            centre_delta = float(
                np.linalg.norm(
                    np.asarray(left_pose.camera_center, dtype=np.float64)[:2]
                    - np.asarray(right_pose.camera_center, dtype=np.float64)[:2]
                )
            )
            orientation_delta = _pose_orientation_delta_degrees(left_pose, right_pose)
            if centre_delta <= centre_limit_m and orientation_delta <= 10.0:
                agreeing_pairs.append((centre_delta, -scale_separation, left_index, right_index))
    if not agreeing_pairs:
        raise SpatialGeometryError("cross_view_multiscale_disagreement")
    _distance, _separation, left_index, right_index = min(agreeing_pairs)
    pair = (candidates[left_index][1], candidates[right_index][1])
    return min(
        pair,
        key=lambda pose: (
            pose.p95_reprojection_error_px or math.inf,
            pose.median_reprojection_error_px or math.inf,
            -(pose.inlier_ratio or 0.0),
        ),
    )


def solve_pnp_pose(
    *,
    source_pixels: Any,
    map_pixels: Any,
    certainties: Any,
    map_image_size: tuple[int, int],
    map_bounds: tuple[float, float, float, float],
    terrain: TerrainSurface,
    camera: CameraMetadataV2,
    prior_camera_center: Any | None = None,
    maximum_prior_distance_m: float = 3_000.0,
) -> CameraPoseSolution:
    """Solve and validate a map-to-camera pose; never returns an unchecked PnP result."""

    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - explicit image packaging failure
        raise SpatialGeometryError("pnp_runtime_unavailable") from exc
    source = np.asarray(source_pixels, dtype=np.float64).reshape(-1, 2)
    mapped = np.asarray(map_pixels, dtype=np.float64).reshape(-1, 2)
    confidence = np.asarray(certainties, dtype=np.float64).reshape(-1)
    if not (len(source) == len(mapped) == len(confidence)) or len(source) < 24:
        raise SpatialGeometryError("spatial_matches_insufficient")
    finite = (
        np.isfinite(source).all(axis=1) & np.isfinite(mapped).all(axis=1) & np.isfinite(confidence)
    )
    source, mapped, confidence = source[finite], mapped[finite], confidence[finite]
    if len(source) < 24:
        raise SpatialGeometryError("spatial_matches_insufficient")
    threshold = max(0.2, float(np.quantile(confidence, 0.5)))
    selected = confidence >= threshold
    source, mapped, confidence = source[selected], mapped[selected], confidence[selected]
    width, height = map_image_size
    left, bottom, right, top = map_bounds
    east = left + mapped[:, 0] / float(width) * (right - left)
    north = top - mapped[:, 1] / float(height) * (top - bottom)
    altitude = terrain.sample_many(east, north)
    valid = np.isfinite(altitude)
    source, east, north, altitude, confidence = (
        source[valid],
        east[valid],
        north[valid],
        altitude[valid],
        confidence[valid],
    )
    order = np.argsort(-confidence)
    kept: list[int] = []
    occupied: set[tuple[int, int, int, int]] = set()
    for index in order:
        key = (
            int(source[index, 0] // 8),
            int(source[index, 1] // 8),
            int(east[index] // 5),
            int(north[index] // 5),
        )
        if key in occupied:
            continue
        occupied.add(key)
        kept.append(int(index))
        if len(kept) >= 2_000:
            break
    if len(kept) < 24:
        raise SpatialGeometryError("spatial_matches_insufficient")
    object_points = np.column_stack((east[kept], north[kept], altitude[kept])).astype(np.float64)
    image_points = source[kept].astype(np.float64)
    horizontal_covariance = np.cov(object_points[:, :2], rowvar=False)
    eigenvalues = np.linalg.eigvalsh(horizontal_covariance)
    if (
        float(np.sqrt(max(eigenvalues[0], 0.0))) < 20.0
        or float(np.sqrt(max(eigenvalues[-1], 0.0))) < 80.0
    ):
        raise SpatialGeometryError("spatial_matches_degenerate")
    # PnP is badly conditioned when Lambert-93 coordinates around 7e5 / 6.6e6
    # are passed directly. Solve in a local ENU frame, then restore the global
    # origin when deriving the camera centre.
    object_origin = np.median(object_points, axis=0)
    local_object_points = object_points - object_origin
    intrinsics = camera_intrinsics(camera)
    success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
        local_object_points,
        image_points,
        intrinsics,
        None,
        flags=cv2.SOLVEPNP_EPNP,
        iterationsCount=1_000,
        reprojectionError=6.0,
        confidence=0.999,
    )
    if not success or inliers is None:
        raise SpatialGeometryError("pnp_solution_missing")
    inlier_indexes = inliers.reshape(-1)
    inlier_ratio = len(inlier_indexes) / len(local_object_points)
    if len(inlier_indexes) < 20 or inlier_ratio < 0.35:
        raise SpatialGeometryError("pnp_inliers_insufficient")
    rotation_vector, translation_vector = cv2.solvePnPRefineLM(
        local_object_points[inlier_indexes],
        image_points[inlier_indexes],
        intrinsics,
        None,
        rotation_vector,
        translation_vector,
    )
    projected, _ = cv2.projectPoints(
        local_object_points[inlier_indexes], rotation_vector, translation_vector, intrinsics, None
    )
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_points[inlier_indexes], axis=1)
    median_error = float(np.median(errors))
    p95_error = float(np.quantile(errors, 0.95))
    if median_error > 4.0 or p95_error > 8.0:
        raise SpatialGeometryError("pnp_reprojection_error_exceeded")
    world_to_camera, _ = cv2.Rodrigues(rotation_vector)
    camera_to_world = world_to_camera.T
    center = (-camera_to_world @ translation_vector).reshape(3) + object_origin
    if not np.isfinite(center).all():
        raise SpatialGeometryError("pnp_camera_center_invalid")
    ground = terrain.sample(float(center[0]), float(center[1]))
    if ground is None or not 1.0 <= float(center[2]) - ground <= 5_000.0:
        raise SpatialGeometryError("pnp_camera_height_invalid")
    if prior_camera_center is not None:
        prior = np.asarray(prior_camera_center, dtype=np.float64).reshape(3)
        if not math.isfinite(maximum_prior_distance_m) or maximum_prior_distance_m <= 0:
            raise SpatialGeometryError("pnp_camera_prior_invalid")
        if float(np.linalg.norm(center[:2] - prior[:2])) > maximum_prior_distance_m:
            raise SpatialGeometryError("pnp_camera_prior_mismatch")
    return CameraPoseSolution(
        camera_center=center,
        camera_to_world=camera_to_world,
        origin="CROSS_VIEW_RAYCAST",
        inlier_count=len(inlier_indexes),
        inlier_ratio=inlier_ratio,
        median_reprojection_error_px=median_error,
        p95_reprojection_error_px=p95_error,
    )
