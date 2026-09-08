from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from fireviewer_contracts.contracts import CameraMetadataV2
from fireviewer_geolocation.mvp.localization.durable_terrain import (
    AzureBackendTerrainResolver,
    DurableTerrainError,
    TerrainDownloadReceipt,
)
from fireviewer_contracts.mvp.supervision.backend_event_evidence import (
    AzureBackendEventEvidenceConfig,
    DurableTerrainReference,
)
from fireviewer_geolocation.spatial_geometry import (
    FWTerrainSurface,
    SpatialGeometryError,
    annotation_ray_direction,
    camera_intrinsics,
    load_fwterrain,
    map_to_wgs84,
    metadata_camera_pose,
    solve_pnp_pose,
)


def _far_container(
    *,
    rows: int = 100,
    columns: int = 100,
    spacing_m: float = 10.0,
    relative_height_m: float = 20.0,
    origin_z_m: float = 80.0,
) -> bytes:
    count = rows * columns
    encoded = np.zeros(count, dtype="<u2").tobytes()
    mask = bytes([0xFF]) * (count // 8) + (bytes([(1 << (count % 8)) - 1]) if count % 8 else b"")
    raw = encoded + mask
    compressed = zlib.compress(raw, level=9)
    bounds = [
        700_000.0,
        6_600_000.0,
        700_000.0 + columns * spacing_m,
        6_600_000.0 + rows * spacing_m,
    ]
    section = {
        "name": "terrain",
        "codec": "zlib",
        "offset_bytes": 0,
        "stored_bytes": len(compressed),
        "raw_bytes": len(raw),
        "stored_sha256": hashlib.sha256(compressed).hexdigest(),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "metadata": {
            "encoding": "masked-regular-grid-z-u16.v1",
            "rows": rows,
            "columns": columns,
            "sample_spacing_m": [spacing_m, spacing_m],
            "outer_bounds_l93_m": bounds,
            "sample_centres": True,
            "validity_mask_offset_bytes": len(encoded),
            "valid_sample_count": count,
            "elevation_quantization": {
                "minimum_m": relative_height_m,
                "maximum_m": relative_height_m,
                "step_m": 0.0,
            },
        },
    }
    header = {
        "schema": "fireviewer.fwtile.v1",
        "kind": "global_far_terrain",
        "tile_id": "test-far",
        "crs": "EPSG:2154",
        "linear_unit": "metre",
        "axis_convention": "X=east,Y=north,Z=up",
        "bounds_l93_m": bounds,
        "origin_l93_m": [700_500.0, 6_600_500.0, origin_z_m],
        "sections": [section],
        "metadata": {},
    }
    encoded_header = json.dumps(
        header,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return (
        struct.pack("<8sHHI", b"FWTILE1\0", 1, 0, len(encoded_header)) + encoded_header + compressed
    )


def _terrain() -> FWTerrainSurface:
    rows = columns = 241
    spacing = 10.0
    left = 700_000.0
    bottom = 6_600_000.0
    right = left + columns * spacing
    top = bottom + rows * spacing
    east = left + (np.arange(columns) + 0.5) * spacing
    north = top - (np.arange(rows) + 0.5) * spacing
    x, y = np.meshgrid(east, north)
    elevations = (
        105.0
        + 0.008 * (x - left)
        + 0.004 * (y - bottom)
        + 8.0 * np.sin((x - left) / 240.0) * np.cos((y - bottom) / 310.0)
    ).astype(np.float32)
    return FWTerrainSurface(
        elevations_m=elevations,
        valid_mask=np.ones((rows, columns), dtype=bool),
        bounds_l93_m=(left, bottom, right, top),
        sample_spacing_m=(spacing, spacing),
        crs="EPSG:2154",
        sample_centres=True,
    )


def test_fwterrain_decoder_restores_absolute_altitude_and_checks_digest(tmp_path: Path) -> None:
    path = tmp_path / "global.fwterrain"
    payload = _far_container()
    path.write_bytes(payload)

    terrain = load_fwterrain(path, declared_crs="EPSG:2154+EPSG:5720")

    assert terrain.sample(700_500.0, 6_600_500.0) == pytest.approx(100.0)
    assert terrain.resolution_m == 10.0

    tampered = bytearray(payload)
    tampered[-1] ^= 0x01
    path.write_bytes(tampered)
    with pytest.raises(SpatialGeometryError) as error:
        load_fwterrain(path)
    assert error.value.code == "terrain_stored_digest_mismatch"


class _TerrainTransport:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.materialized_path: Path | None = None

    def download(
        self,
        url: str,
        destination: Path,
        *,
        headers: dict[str, str],
        timeout_seconds: float,
        maximum_bytes: int,
    ) -> TerrainDownloadReceipt:
        assert url.endswith("/terrain/content")
        assert headers["Authorization"] == "Bearer " + ("t" * 32)
        assert timeout_seconds == 10
        assert maximum_bytes == len(self.payload)
        destination.write_bytes(self.payload)
        self.materialized_path = destination
        digest = hashlib.sha256(self.payload).hexdigest()
        return TerrainDownloadReceipt(
            size_bytes=len(self.payload),
            sha256=digest,
            headers={
                "etag": f'"{digest}"',
                "x-checksum-sha256": digest,
            },
        )


def _durable_terrain_reference(payload: bytes) -> DurableTerrainReference:
    return DurableTerrainReference(
        terrain_id="TERRAIN-1",
        package_id="PACKAGE-1",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        media_type="application/vnd.fireviewer.terrain",
        crs="EPSG:2154+EPSG:5720",
        resolution_m=10,
        content_url=(
            "https://api.example.test/api/v1/internal/event-evidence/"
            "EVENT-1/terrain/content"
        ),
    )


def test_durable_terrain_resolver_verifies_and_decodes_real_fwterrain() -> None:
    payload = _far_container()
    transport = _TerrainTransport(payload)
    resolver = AzureBackendTerrainResolver(
        AzureBackendEventEvidenceConfig(
            base_url="https://api.example.test",
            bearer_token="t" * 32,
        ),
        transport=transport,
    )

    provider = resolver.resolve(_durable_terrain_reference(payload))
    longitude, latitude = map_to_wgs84(700_500.0, 6_600_500.0, map_crs="EPSG:2154")

    assert provider.reference_revision == hashlib.sha256(payload).hexdigest()
    assert provider.resolution_m == 10.0
    assert provider.elevation_m(longitude, latitude) == pytest.approx(100.0)
    assert transport.materialized_path is not None
    assert not transport.materialized_path.exists()


def test_durable_terrain_resolver_rejects_a_revision_mismatch() -> None:
    payload = _far_container()
    transport = _TerrainTransport(payload)
    reference = _durable_terrain_reference(payload).model_copy(
        update={"sha256": "f" * 64}
    )
    resolver = AzureBackendTerrainResolver(
        AzureBackendEventEvidenceConfig(
            base_url="https://api.example.test",
            bearer_token="t" * 32,
        ),
        transport=transport,
    )

    with pytest.raises(DurableTerrainError, match="SHA-256"):
        resolver.resolve(reference)

    assert transport.materialized_path is not None
    assert not transport.materialized_path.exists()


def test_confirmed_camera_pose_projects_a_pixel_onto_the_mnt() -> None:
    terrain = _terrain()
    center_east = 701_100.0
    center_north = 6_601_000.0
    longitude, latitude = map_to_wgs84(center_east, center_north, map_crs=terrain.crs)
    camera = CameraMetadataV2(
        longitude=longitude,
        latitude=latitude,
        orthometric_height_m=300.0,
        horizontal_accuracy_m=5.0,
        yaw_deg=0.0,
        pitch_deg=-25.0,
        roll_deg=0.0,
        horizontal_fov_deg=70.0,
        image_width_px=1_920,
        image_height_px=1_080,
        pose_origin="HUMAN_CONFIRMED",
    )
    pose = metadata_camera_pose(camera, map_crs=terrain.crs)
    direction = annotation_ray_direction((0.5, 0.5), camera=camera, pose=pose)

    hit = terrain.intersect_ray(pose.camera_center, direction)

    assert hit.distance_m > 100.0
    assert hit.north_m > center_north
    assert hit.altitude_m == pytest.approx(terrain.sample(hit.east_m, hit.north_m), abs=0.01)


def test_pnp_is_stable_with_full_scale_lambert_93_coordinates() -> None:
    terrain = _terrain()
    width, height = 1_920, 1_080
    camera = CameraMetadataV2(
        horizontal_fov_deg=70.0,
        image_width_px=width,
        image_height_px=height,
    )
    camera_center = np.asarray([701_050.0, 6_600_500.0, 420.0], dtype=np.float64)
    target = np.asarray([701_200.0, 6_601_250.0, 120.0], dtype=np.float64)
    forward = target - camera_center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    camera_to_world = np.column_stack((right, down, forward))
    world_to_camera = camera_to_world.T
    intrinsics = camera_intrinsics(camera)

    left, bottom, right_bound, top = terrain.bounds_l93_m
    east_values = np.linspace(700_300.0, 701_900.0, 42)
    north_values = np.linspace(6_600_700.0, 6_602_000.0, 36)
    object_points = []
    source_pixels = []
    map_pixels = []
    for north in north_values:
        for east in east_values:
            altitude = terrain.sample(float(east), float(north))
            assert altitude is not None
            point = np.asarray([east, north, altitude], dtype=np.float64)
            camera_point = world_to_camera @ (point - camera_center)
            if camera_point[2] <= 0:
                continue
            projected = intrinsics @ camera_point
            u, v = projected[:2] / projected[2]
            if 16 <= u < width - 16 and 16 <= v < height - 16:
                object_points.append(point)
                source_pixels.append((u, v))
                map_pixels.append(
                    (
                        (east - left) / (right_bound - left) * 4_000.0,
                        (top - north) / (top - bottom) * 4_000.0,
                    )
                )
    assert len(object_points) > 100

    solution = solve_pnp_pose(
        source_pixels=np.asarray(source_pixels),
        map_pixels=np.asarray(map_pixels),
        certainties=np.ones(len(source_pixels)),
        map_image_size=(4_000, 4_000),
        map_bounds=terrain.bounds_l93_m,
        terrain=terrain,
        camera=camera,
        prior_camera_center=camera_center,
    )

    assert np.linalg.norm(solution.camera_center - camera_center) < 2.0
    assert solution.inlier_count is not None and solution.inlier_count > 100
    assert solution.p95_reprojection_error_px is not None
    assert solution.p95_reprojection_error_px < 1.0

    with pytest.raises(SpatialGeometryError) as prior_error:
        solve_pnp_pose(
            source_pixels=np.asarray(source_pixels),
            map_pixels=np.asarray(map_pixels),
            certainties=np.ones(len(source_pixels)),
            map_image_size=(4_000, 4_000),
            map_bounds=terrain.bounds_l93_m,
            terrain=terrain,
            camera=camera,
            prior_camera_center=camera_center + np.asarray([500.0, 0.0, 0.0]),
            maximum_prior_distance_m=100.0,
        )
    assert prior_error.value.code == "pnp_camera_prior_mismatch"


def test_pnp_abstains_when_matches_do_not_cover_two_dimensions() -> None:
    terrain = _terrain()
    source = np.column_stack((np.arange(30) * 10.0 + 100.0, np.full(30, 300.0)))
    mapped = np.column_stack((np.arange(30) * 20.0 + 100.0, np.full(30, 500.0)))
    camera = CameraMetadataV2(
        horizontal_fov_deg=70.0,
        image_width_px=1_920,
        image_height_px=1_080,
    )

    with pytest.raises(SpatialGeometryError) as error:
        solve_pnp_pose(
            source_pixels=source,
            map_pixels=mapped,
            certainties=np.ones(30),
            map_image_size=(4_000, 4_000),
            map_bounds=terrain.bounds_l93_m,
            terrain=terrain,
            camera=camera,
        )

    assert error.value.code == "spatial_matches_degenerate"
