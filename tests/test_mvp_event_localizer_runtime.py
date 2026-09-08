from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from fireviewer_contracts.mvp.contracts import EventEvidenceV1
from fireviewer_geolocation.mvp.localization import (
    DeterministicEvidenceFusion,
    EventLocalizationConfig,
    FusionConfig,
    MegaLocFaissEventLocalizer,
    MegaLocFaissRetriever,
    PanoramaxRegionalIndexBuilder,
    RegionalIndexConfig,
    RetrievalConfig,
    abstain_for_missing_reference_coverage,
    materialize_panoramax_cache,
)
from fireviewer_geolocation.mvp.localization.faiss_index import (
    FaissCosineIndex,
    IndexEntry,
    load_faiss_bundle,
    write_faiss_bundle,
)
from fireviewer_geolocation.mvp.localization.megaloc import MegaLocBatch, MegaLocEmbedding
from fireviewer_geolocation.mvp.localization.panoramax import (
    PanoramaxImage,
    PanoramaxSearchResult,
)
from fireviewer_geolocation.mvp.localization.panoramax_cache import CachedPanoramaxImageLoader
from fireviewer_geolocation.mvp.localization.perspective import PerspectiveConfig


class _FakeFlatIndex:
    def __init__(self, dimension: int) -> None:
        self.d = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return self.vectors.shape[0]

    def add(self, matrix: np.ndarray) -> None:
        self.vectors = np.asarray(matrix, dtype=np.float32).copy()

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = query @ self.vectors.T
        indices = np.argsort(-scores, axis=1)[:, :top_k]
        distances = np.take_along_axis(scores, indices, axis=1)
        return distances.astype(np.float32), indices.astype(np.int64)


class _FakeFaiss:
    IndexFlatIP = _FakeFlatIndex

    @staticmethod
    def serialize_index(index: _FakeFlatIndex) -> np.ndarray:
        stream = BytesIO()
        np.save(stream, index.vectors, allow_pickle=False)
        return np.frombuffer(stream.getvalue(), dtype=np.uint8)

    @staticmethod
    def deserialize_index(payload: np.ndarray) -> _FakeFlatIndex:
        vectors = np.load(BytesIO(bytes(payload)), allow_pickle=False)
        index = _FakeFlatIndex(vectors.shape[1])
        index.add(vectors)
        return index


def _batch(media: tuple[tuple[str, object], ...], vectors: np.ndarray) -> MegaLocBatch:
    return MegaLocBatch(
        model_id="gberton/MegaLoc",
        model_version="fixture-revision",
        embeddings=tuple(
            MegaLocEmbedding(
                embedding_id=f"EMBEDDING-{media_id}",
                media_id=media_id,
                dimension=vectors.shape[1],
                vector_sha256=sha256(vectors[index].tobytes()).hexdigest(),
            )
            for index, (media_id, _) in enumerate(media)
        ),
    )


class _RegionalEncoder:
    model_id = "gberton/MegaLoc"
    model_version = "fixture-revision"

    def encode(self, media: tuple[tuple[str, object], ...]) -> tuple[MegaLocBatch, np.ndarray]:
        vectors = np.asarray(
            [(1.0, 0.0) if media_id.endswith("h0") else (0.0, 1.0) for media_id, _ in media],
            dtype=np.float32,
        )
        return _batch(media, vectors), vectors


class _QueryEncoder:
    model_id = "gberton/MegaLoc"
    model_version = "fixture-revision"

    def encode(self, media: tuple[tuple[str, object], ...]) -> tuple[MegaLocBatch, np.ndarray]:
        if any(media_id == "FRAME-B" for media_id, _ in media):
            raise RuntimeError("fixture embedding failure")
        vectors = np.asarray([(1.0, 0.0) for _ in media], dtype=np.float32)
        return _batch(media, vectors), vectors


class _ImageLoader:
    def load(self, media: object) -> Image.Image:
        return Image.new("RGB", (64, 32), color=(40, 80, 120))


class _PanoramaLoader:
    def load(self, image: PanoramaxImage) -> Image.Image:
        return Image.new("RGB", (128, 64), color=(20, 40, 60))


def _panoramax_result() -> PanoramaxSearchResult:
    return PanoramaxSearchResult(
        zone_id="die-justin",
        api_url="https://panoramax.example/api",
        bbox_wgs84=(5.3, 44.7, 5.4, 44.8),
        query_sha256="a" * 64,
        retrieved_at=datetime(2026, 8, 21, 10, tzinfo=UTC),
        images=(
            PanoramaxImage(
                image_id="PANORAMA-1",
                sequence_id="SEQUENCE-1",
                longitude=5.37,
                latitude=44.75,
                heading_deg=10,
                field_of_view_deg=360,
                gps_accuracy_m=4,
                captured_at=datetime(2026, 7, 20, 10, tzinfo=UTC),
                image_url="https://panoramax.example/api/pictures/PANORAMA-1/hd.jpg",
                item_sha256="b" * 64,
            ),
        ),
    )


def test_regional_builder_projects_panoramax_and_builds_versioned_index() -> None:
    builder = PanoramaxRegionalIndexBuilder(
        image_loader=_PanoramaLoader(),
        encoder=_RegionalEncoder(),
        panoramax_revision="panoramax-snapshot-2026-08-21",
        config=RegionalIndexConfig(
            perspective=PerspectiveConfig(
                headings_deg=(0, 90),
                width_px=32,
                height_px=32,
            ),
            encode_batch_size=1,
        ),
        faiss_module=_FakeFaiss,
    )

    index = builder.build(_panoramax_result())
    manifest = index.manifest()

    assert manifest.vector_count == 2
    assert manifest.zone_id == "die-justin"
    assert manifest.panoramax_revision == "panoramax-snapshot-2026-08-21"
    assert [entry.crop_heading_deg for entry in manifest.entries] == [10, 100]
    assert (
        index.search(np.asarray(((1.0, 0.0),), dtype=np.float32), top_k=1)[0].entry.image_id
        == "PANORAMA-1"
    )


def test_regional_builder_keeps_regular_panoramax_photos_as_native_views() -> None:
    payload = _panoramax_result().model_dump(mode="python")
    payload["images"][0]["field_of_view_deg"] = 80
    result = PanoramaxSearchResult.model_validate(payload)
    builder = PanoramaxRegionalIndexBuilder(
        image_loader=_PanoramaLoader(),
        encoder=_RegionalEncoder(),
        panoramax_revision="panoramax-snapshot-2026-08-21",
        config=RegionalIndexConfig(
            perspective=PerspectiveConfig(headings_deg=(0, 90), width_px=32, height_px=32)
        ),
        faiss_module=_FakeFaiss,
    )

    manifest = builder.build(result).manifest()

    assert manifest.vector_count == 1
    assert manifest.entries[0].embedding_id.endswith("-native")
    assert manifest.entries[0].crop_heading_deg == 10


def test_regional_index_bundle_is_persisted_and_digest_checked(tmp_path: Path) -> None:
    directory = tmp_path / "zone-v1"
    builder = PanoramaxRegionalIndexBuilder(
        image_loader=_PanoramaLoader(),
        encoder=_RegionalEncoder(),
        panoramax_revision="panoramax-snapshot-2026-08-21",
        config=RegionalIndexConfig(
            perspective=PerspectiveConfig(headings_deg=(0,), width_px=32, height_px=32)
        ),
        faiss_module=_FakeFaiss,
    )
    index = builder.build(_panoramax_result())

    manifest = write_faiss_bundle(index, directory)
    restored = load_faiss_bundle(directory, faiss_module=_FakeFaiss)

    assert manifest.index_sha256 == restored.manifest().index_sha256
    assert (directory / "faiss.index").is_file()
    assert (directory / "index-manifest.json").is_file()


def test_panoramax_cache_downloads_once_and_replays_verified_image(tmp_path: Path) -> None:
    stream = BytesIO()
    Image.new("RGB", (80, 40), color=(10, 20, 30)).save(stream, format="JPEG")
    payload = stream.getvalue()

    class _Transport:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, url: str, *, max_bytes: int) -> bytes:
            self.calls += 1
            assert url.startswith("https://panoramax.example/api/")
            assert len(payload) < max_bytes
            return payload

    class _OfflineTransport:
        def get(self, url: str, *, max_bytes: int) -> bytes:
            raise AssertionError("qualified Panoramax cache unexpectedly used the network")

    transport = _Transport()
    directory = tmp_path / "panoramax-cache"

    first = materialize_panoramax_cache(
        _panoramax_result(),
        directory,
        transport=transport,
    )
    replay = materialize_panoramax_cache(
        _panoramax_result(),
        directory,
        transport=_OfflineTransport(),
    )
    image = CachedPanoramaxImageLoader.from_directory(directory).load(_panoramax_result().images[0])

    assert transport.calls == 1
    assert first.assets == replay.assets
    assert first.canonical_sha256() == replay.canonical_sha256()
    assert isinstance(image, Image.Image)
    assert image.size == (80, 40)


def test_panoramax_cache_records_empty_coverage_without_transport(tmp_path: Path) -> None:
    class _OfflineTransport:
        def get(self, url: str, *, max_bytes: int) -> bytes:
            raise AssertionError("an empty Panoramax result must not perform asset requests")

    result = _panoramax_result().model_copy(update={"images": ()})
    directory = tmp_path / "empty-panoramax-cache"

    manifest = materialize_panoramax_cache(
        result,
        directory,
        transport=_OfflineTransport(),
    )

    assert manifest.assets == ()
    assert manifest.search_result.images == ()
    assert (directory / "cache-manifest.json").is_file()


def _event() -> EventEvidenceV1:
    return EventEvidenceV1.model_validate(
        {
            "schema": "fireviewer.event-evidence.v1",
            "event_id": "EVENT-RUNTIME-1",
            "sources": [
                {
                    "source_id": "SOURCE-1",
                    "origin_id": "ORIGIN-1",
                    "publisher": "Witness",
                    "retrieved_at": "2026-08-21T10:00:00Z",
                    "source_type": "witness",
                    "independence_weight": 1,
                }
            ],
            "media": [
                {
                    "media_id": "VIDEO-1",
                    "source_id": "SOURCE-1",
                    "media_group_id": "GROUP-1",
                    "origin_id": "ORIGIN-1",
                    "kind": "video",
                    "sha256": "a" * 64,
                },
                {
                    "media_id": "FRAME-A",
                    "source_id": "SOURCE-1",
                    "media_group_id": "GROUP-1",
                    "origin_id": "ORIGIN-1",
                    "kind": "keyframe",
                    "sha256": "b" * 64,
                    "parent_media_id": "VIDEO-1",
                },
                {
                    "media_id": "FRAME-B",
                    "source_id": "SOURCE-1",
                    "media_group_id": "GROUP-1",
                    "origin_id": "ORIGIN-1",
                    "kind": "keyframe",
                    "sha256": "c" * 64,
                    "parent_media_id": "VIDEO-1",
                },
            ],
        }
    )


def test_event_localization_abstains_without_panoramax_coverage() -> None:
    first = abstain_for_missing_reference_coverage(_event())
    replay = abstain_for_missing_reference_coverage(first)

    assert first.location_candidates == ()
    assert first.candidate_clusters == ()
    assert first.needs_human_review is True
    assert [item.code for item in first.uncertainties] == [
        "panoramax_no_coverage",
        "no_location_candidates",
    ]
    assert replay == first


def _reference_index() -> FaissCosineIndex:
    vector = np.asarray(((1.0, 0.0),), dtype=np.float32)
    return FaissCosineIndex.build(
        vectors=vector,
        entries=(
            IndexEntry(
                embedding_id="REFERENCE-EMBEDDING-1",
                image_id="PANORAMA-1",
                sequence_id="SEQUENCE-1",
                longitude=5.37,
                latitude=44.75,
                horizontal_accuracy_m=4,
                captured_at=datetime(2026, 7, 20, 10, tzinfo=UTC),
                crop_heading_deg=10,
                vector_sha256=sha256(vector[0].tobytes()).hexdigest(),
            ),
        ),
        model_id="gberton/MegaLoc",
        model_version="fixture-revision",
        zone_id="die-justin",
        panoramax_revision="panoramax-snapshot-2026-08-21",
        faiss_module=_FakeFaiss,
    )


def test_event_localizer_isolates_failed_media_and_keeps_event_result_replayable() -> None:
    localizer = MegaLocFaissEventLocalizer(
        image_loader=_ImageLoader(),
        encoder=_QueryEncoder(),
        retriever=MegaLocFaissRetriever(
            _reference_index(),
            config=RetrievalConfig(top_k=1),
        ),
        fusion=DeterministicEvidenceFusion(
            FusionConfig(human_review_threshold=0, ambiguity_margin=0)
        ),
        config=EventLocalizationConfig(query_batch_size=2),
    )

    first = localizer.localize(_event())
    replay = localizer.localize(first)

    assert [candidate.media_id for candidate in first.location_candidates] == ["FRAME-A"]
    assert first.candidate_clusters[0].independent_media_count == 1
    assert first.needs_human_review is True
    assert [(item.code, item.scope_id) for item in first.uncertainties] == [
        ("visual_embedding_failed", "FRAME-B")
    ]
    assert replay.location_candidates == first.location_candidates
    assert replay.candidate_clusters == first.candidate_clusters
    assert replay.uncertainties == first.uncertainties


def test_event_localizer_rejects_an_index_from_another_model_revision() -> None:
    class _WrongRevisionEncoder(_QueryEncoder):
        model_version = "wrong-revision"

    try:
        MegaLocFaissEventLocalizer(
            image_loader=_ImageLoader(),
            encoder=_WrongRevisionEncoder(),
            retriever=MegaLocFaissRetriever(_reference_index()),
        )
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("model/index revision mismatch was accepted")
