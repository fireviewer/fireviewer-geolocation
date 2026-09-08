from __future__ import annotations

import hashlib
import json
import struct
import zlib
from contextlib import nullcontext
from datetime import UTC, datetime

import numpy as np
import pytest
from PIL import Image

from fireviewer_contracts.contracts import (
    AnalysisWindowV2,
    CameraMetadataV2,
    SourceAnnotationV2,
    SourceProvenanceV2,
    SpatialReferenceAssetV2,
    SpatialReferenceBundleV2,
    WorkerBatchItemV2,
    WorkerInputV2,
)
from fireviewer_contracts.client.media_fetcher import MediaFetcher
from fireviewer_geolocation.spatial_geometry import (
    CameraPoseSolution,
    SpatialGeometryError,
    crop_georeferenced_map,
    cross_view_search_radii,
    map_to_wgs84,
    select_consistent_cross_view_pose,
)
from fireviewer_geolocation.spatial_pipeline import DeterministicSpatialPipeline


class _Response:
    status_code = 200

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.headers = {"content-length": str(len(payload))}

    def iter_bytes(self, _chunk_size: int):
        yield self.payload


class _Client:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads

    def stream(self, method: str, url: str):
        assert method == "GET"
        return nullcontext(_Response(self.payloads[url]))

    def close(self) -> None:
        return None


def _far_container() -> bytes:
    rows = columns = 100
    spacing_m = 10.0
    count = rows * columns
    elevations = np.zeros(count, dtype="<u2").tobytes()
    mask = bytes([0xFF]) * (count // 8)
    raw = elevations + mask
    compressed = zlib.compress(raw, level=9)
    bounds = [700_000.0, 6_600_000.0, 701_000.0, 6_601_000.0]
    header = {
        "schema": "fireviewer.fwtile.v1",
        "kind": "global_far_terrain",
        "tile_id": "test-far",
        "crs": "EPSG:2154",
        "linear_unit": "metre",
        "axis_convention": "X=east,Y=north,Z=up",
        "bounds_l93_m": bounds,
        "origin_l93_m": [700_500.0, 6_600_500.0, 80.0],
        "sections": [
            {
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
                    "validity_mask_offset_bytes": len(elevations),
                    "valid_sample_count": count,
                    "elevation_quantization": {
                        "minimum_m": 20.0,
                        "maximum_m": 20.0,
                        "step_m": 0.0,
                    },
                },
            }
        ],
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


def _batch(
    terrain_url: str,
    terrain_sha256: str,
    *,
    pose_origin: str = "HUMAN_CONFIRMED",
) -> WorkerInputV2:
    longitude, latitude = map_to_wgs84(700_500.0, 6_600_250.0, map_crs="EPSG:2154")
    return WorkerInputV2(
        batch_id="BATCH-SPATIAL-1",
        batch_type="user_media",
        priority="scheduled_combined",
        analysis_window=AnalysisWindowV2(
            analysis_id="ANALYSIS-SPATIAL-1",
            fire_id="FR-99-00001",
            episode_id="E01",
            window_start_at=datetime(2026, 7, 10, tzinfo=UTC),
            window_end_at=datetime(2026, 7, 11, tzinfo=UTC),
            local_date="2026-07-10",
            timezone="Europe/Paris",
        ),
        reference_bundle=SpatialReferenceBundleV2(
            reference_id="SYNTHETIC-R1",
            manifest_sha256="a" * 64,
            assets=(
                SpatialReferenceAssetV2(
                    kind="terrain_mnt",
                    working_file_url=terrain_url,
                    sha256=terrain_sha256,
                    crs="EPSG:2154+EPSG:5720",
                    resolution_m=10.0,
                ),
            ),
        ),
        items=(
            WorkerBatchItemV2(
                input_id="INPUT-1",
                media_type="image",
                working_file_url="https://media.internal/private/source.jpg",
                provenance=SourceProvenanceV2(
                    source_key="PUBLIC-1",
                    license_identifier="private-analysis-only",
                    trust="operator",
                ),
                captured_at=datetime(2026, 7, 10, 9, tzinfo=UTC),
                camera=CameraMetadataV2(
                    longitude=longitude,
                    latitude=latitude,
                    orthometric_height_m=220.0,
                    horizontal_accuracy_m=5.0,
                    yaw_deg=0.0,
                    pitch_deg=-25.0,
                    roll_deg=0.0,
                    horizontal_fov_deg=70.0,
                    image_width_px=1_920,
                    image_height_px=1_080,
                    pose_origin=pose_origin,
                ),
            ),
        ),
    )


def _annotation() -> SourceAnnotationV2:
    return SourceAnnotationV2(
        annotation_id="ANN-1",
        evidence_id="INPUT-1",
        evidence_kind="image",
        semantic_anchor="active_fire_point",
        source_point_normalized=(0.5, 0.5),
        model_score=0.95,
    )


def _pipeline(monkeypatch, *, terrain_url: str, payload: bytes, enabled: bool = False):
    client = _Client({terrain_url: payload})
    monkeypatch.setattr("httpx.Client", lambda **_kwargs: client)
    fetcher = MediaFetcher(
        allowed_hosts=frozenset({"media.internal"}),
        max_bytes=2_000_000,
        max_cache_bytes=2_000_000,
    )
    return fetcher, DeterministicSpatialPipeline(
        fetcher=fetcher,
        enable_cross_view=enabled,
    )


def test_confirmed_pose_produces_a_native_private_active_fire_point(monkeypatch) -> None:
    terrain_url = "https://media.internal/private/global.fwterrain"
    payload = _far_container()
    fetcher, pipeline = _pipeline(monkeypatch, terrain_url=terrain_url, payload=payload)
    batch = _batch(terrain_url, hashlib.sha256(payload).hexdigest())

    with fetcher.batch_scope():
        result = pipeline.project(batch, {"INPUT-1": (_annotation(),)}, sequence_start=5)

    proposal = result.proposals_by_input["INPUT-1"][0]
    assert proposal.status == "projected_geometry"
    assert proposal.proposal_kind == "active_fire_point"
    assert proposal.geometry_geojson == {
        "type": "Point",
        "coordinates": [proposal.longitude, proposal.latitude],
    }
    assert proposal.geometry_origin == "CAMERA_RAYCAST"
    assert proposal.reference_bundle_sha256 == "a" * 64
    assert proposal.horizontal_accuracy_m is not None and proposal.horizontal_accuracy_m < 1_000
    assert [trace.stage_role for trace in result.stage_traces] == [
        "cross_view_registration",
        "spatial_projection",
    ]
    assert result.stage_traces[1].status == "succeeded"
    assert result.model_runs == ()


def test_unconfirmed_pose_abstains_while_roma_benchmark_is_closed(monkeypatch) -> None:
    terrain_url = "https://media.internal/private/global.fwterrain"
    payload = _far_container()
    fetcher, pipeline = _pipeline(monkeypatch, terrain_url=terrain_url, payload=payload)
    batch = _batch(
        terrain_url,
        hashlib.sha256(payload).hexdigest(),
        pose_origin="USER_DECLARED",
    )

    with fetcher.batch_scope():
        result = pipeline.project(batch, {"INPUT-1": (_annotation(),)}, sequence_start=5)

    proposal = result.proposals_by_input["INPUT-1"][0]
    assert proposal.status == "insufficient_geometry"
    assert proposal.uncertainty_codes == ("cross_view_benchmark_not_approved",)
    assert result.stage_traces[0].preflight.reason_codes == ("cross_view_benchmark_not_approved",)
    assert result.model_runs[0].status == "skipped"


def test_reference_digest_failure_is_explicit_and_never_projects(monkeypatch) -> None:
    terrain_url = "https://media.internal/private/global.fwterrain"
    payload = _far_container()
    fetcher, pipeline = _pipeline(monkeypatch, terrain_url=terrain_url, payload=payload)
    batch = _batch(terrain_url, "0" * 64)

    with fetcher.batch_scope():
        result = pipeline.project(batch, {"INPUT-1": (_annotation(),)}, sequence_start=5)

    proposal = result.proposals_by_input["INPUT-1"][0]
    assert proposal.status == "insufficient_geometry"
    assert proposal.uncertainty_codes == ("terrain_reference_digest_mismatch",)
    assert result.stage_traces[1].status == "skipped"


def test_cross_view_crop_preserves_exact_lambert_bounds() -> None:
    image = Image.new("RGB", (1_000, 500))
    crop = crop_georeferenced_map(
        image,
        map_bounds=(700_000.0, 6_600_000.0, 702_000.0, 6_601_000.0),
        centre_east_m=701_000.0,
        centre_north_m=6_600_500.0,
        radius_m=200.0,
    )
    try:
        assert crop.image.size == (200, 200)
        assert crop.bounds_m == (
            700_800.0,
            6_600_300.0,
            701_200.0,
            6_600_700.0,
        )
    finally:
        crop.image.close()
        image.close()


def _pose(
    centre: tuple[float, float, float],
    *,
    p95: float,
    inlier_ratio: float,
) -> CameraPoseSolution:
    return CameraPoseSolution(
        camera_center=np.asarray(centre, dtype=np.float64),
        camera_to_world=np.eye(3, dtype=np.float64),
        origin="CROSS_VIEW_RAYCAST",
        inlier_count=100,
        inlier_ratio=inlier_ratio,
        median_reprojection_error_px=p95 / 2.0,
        p95_reprojection_error_px=p95,
    )


def test_cross_view_requires_two_agreeing_scales_and_selects_best_pose() -> None:
    assert cross_view_search_radii(100.0) == (500.0, 1_000.0, 2_000.0)
    best = _pose((700_010.0, 6_600_010.0, 200.0), p95=2.0, inlier_ratio=0.8)
    agreeing = _pose((700_020.0, 6_600_015.0, 202.0), p95=3.0, inlier_ratio=0.7)
    outlier = _pose((702_000.0, 6_603_000.0, 250.0), p95=1.0, inlier_ratio=0.9)

    selected = select_consistent_cross_view_pose(
        [(500.0, best), (1_000.0, agreeing), (2_000.0, outlier)],
        horizontal_accuracy_m=100.0,
    )

    assert selected is best


def test_cross_view_abstains_on_single_scale_or_geographic_disagreement() -> None:
    first = _pose((700_000.0, 6_600_000.0, 200.0), p95=2.0, inlier_ratio=0.8)
    second = _pose((701_500.0, 6_602_000.0, 200.0), p95=2.0, inlier_ratio=0.8)

    with pytest.raises(SpatialGeometryError) as single:
        select_consistent_cross_view_pose([(500.0, first)], horizontal_accuracy_m=100.0)
    assert single.value.code == "cross_view_multiscale_confirmation_missing"

    with pytest.raises(SpatialGeometryError) as disagreement:
        select_consistent_cross_view_pose(
            [(500.0, first), (1_000.0, second)],
            horizontal_accuracy_m=100.0,
        )
    assert disagreement.value.code == "cross_view_multiscale_disagreement"
