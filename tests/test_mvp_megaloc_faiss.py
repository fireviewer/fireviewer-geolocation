from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image

from fireviewer_geolocation.mvp.localization.faiss_index import FaissCosineIndex, IndexEntry
from fireviewer_geolocation.mvp.localization.megaloc import (
    MegaLocConfig,
    MegaLocError,
    TorchMegaLocEncoder,
)
from fireviewer_geolocation.mvp.localization.retrieval import MegaLocFaissRetriever, RetrievalConfig


class _EmbeddingModel(torch.nn.Module):
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        mean = batch.mean(dim=(1, 2, 3))
        raw = torch.stack((mean, mean + 1, mean + 2, mean + 3), dim=1)
        return raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True)


class _InvalidEmbeddingModel(torch.nn.Module):
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.ones((batch.shape[0], 4), dtype=torch.float32)


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


def test_megaloc_encoder_uses_injected_model_and_emits_replayable_digests() -> None:
    encoder = TorchMegaLocEncoder(
        model_loader=_EmbeddingModel,
        model_version="fixture-revision",
        config=MegaLocConfig(image_size_px=32, expected_dimension=4, batch_size=1),
    )
    black = Image.new("RGB", (64, 32), color=(0, 0, 0))
    white = Image.new("RGB", (64, 32), color=(255, 255, 255))

    batch, vectors = encoder.encode((("MEDIA-1", black), ("MEDIA-2", white)))

    assert vectors.shape == (2, 4)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1)
    assert batch.model_version == "fixture-revision"
    assert batch.embeddings[0].vector_sha256 != batch.embeddings[1].vector_sha256


def test_megaloc_encoder_rejects_non_normalized_model_output() -> None:
    encoder = TorchMegaLocEncoder(
        model_loader=_InvalidEmbeddingModel,
        model_version="fixture-revision",
        config=MegaLocConfig(image_size_px=32, expected_dimension=4),
    )
    with pytest.raises(MegaLocError, match="not L2-normalized"):
        encoder.encode((("MEDIA-1", Image.new("RGB", (32, 32))),))


def _entries(vectors: np.ndarray) -> tuple[IndexEntry, ...]:
    from hashlib import sha256

    return tuple(
        IndexEntry(
            embedding_id=f"EMBEDDING-{index}",
            image_id=f"IMAGE-{index}",
            sequence_id="SEQUENCE-1",
            longitude=5.37 + index * 0.001,
            latitude=44.75 + index * 0.001,
            captured_at=datetime(2026, 8, 20, 10, index, tzinfo=UTC),
            crop_heading_deg=index * 90,
            vector_sha256=sha256(vector.tobytes()).hexdigest(),
        )
        for index, vector in enumerate(vectors)
    )


def test_faiss_cosine_index_is_versioned_searchable_and_digest_checked() -> None:
    vectors = np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32)
    index = FaissCosineIndex.build(
        vectors=vectors,
        entries=_entries(vectors),
        model_id="gberton/MegaLoc",
        model_version="fixture-revision",
        zone_id="die-justin",
        panoramax_revision="fixture-panoramax-revision",
        faiss_module=_FakeFaiss,
    )

    matches = index.search(np.asarray(((1.0, 0.0),), dtype=np.float32), top_k=2)
    manifest = index.manifest()
    payload = index.serialize()
    restored = FaissCosineIndex.restore(payload, manifest=manifest, faiss_module=_FakeFaiss)

    assert [match.entry.image_id for match in matches] == ["IMAGE-0", "IMAGE-1"]
    assert matches[0].score == 1
    assert manifest.vector_count == 2
    assert (
        restored.search(np.asarray(((0.0, 1.0),), dtype=np.float32), top_k=1)[0].entry.image_id
        == "IMAGE-1"
    )
    with pytest.raises(ValueError, match="digest"):
        FaissCosineIndex.restore(payload + b"tampered", manifest=manifest, faiss_module=_FakeFaiss)


def test_retriever_produces_ranked_candidates_for_every_media_descriptor() -> None:
    vectors = np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32)
    index = FaissCosineIndex.build(
        vectors=vectors,
        entries=_entries(vectors),
        model_id="gberton/MegaLoc",
        model_version="fixture-revision",
        zone_id="die-justin",
        panoramax_revision="fixture-panoramax-revision",
        faiss_module=_FakeFaiss,
    )
    retriever = MegaLocFaissRetriever(index, config=RetrievalConfig(top_k=2))

    candidates = retriever.retrieve_many(
        {
            "MEDIA-A": np.asarray(((1.0, 0.0),), dtype=np.float32),
            "MEDIA-B": np.asarray(((0.0, 1.0),), dtype=np.float32),
        }
    )

    assert len(candidates) == 4
    assert candidates[0].media_id == "MEDIA-A"
    assert candidates[0].reference_id == "IMAGE-0"
    assert candidates[0].score == 1
    assert candidates[0].raw_score == 1
    assert candidates[2].media_id == "MEDIA-B"
    assert candidates[2].reference_id == "IMAGE-1"


def test_real_faiss_binding_round_trips_when_localization_extra_is_installed() -> None:
    faiss = pytest.importorskip("faiss")
    vectors = np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32)
    index = FaissCosineIndex.build(
        vectors=vectors,
        entries=_entries(vectors),
        model_id="gberton/MegaLoc",
        model_version="fixture-revision",
        zone_id="die-justin",
        panoramax_revision="fixture-panoramax-revision",
        faiss_module=faiss,
    )
    manifest = index.manifest()
    restored = FaissCosineIndex.restore(
        index.serialize(),
        manifest=manifest,
        faiss_module=faiss,
    )

    assert (
        restored.search(np.asarray(((0.0, 1.0),), dtype=np.float32), top_k=1)[0].entry.image_id
        == "IMAGE-1"
    )
