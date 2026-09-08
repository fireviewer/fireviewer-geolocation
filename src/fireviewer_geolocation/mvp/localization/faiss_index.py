from __future__ import annotations

import json
import os
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

from pydantic import Field, model_validator

from fireviewer_contracts.contracts import SafeIdentifierV2, Sha256HexV2, StrictModel
from fireviewer_contracts.mvp.contracts.common import is_timezone_aware


class FaissUnavailableError(RuntimeError):
    """Raised when the optional FAISS runtime is absent."""


class IndexEntry(StrictModel):
    embedding_id: SafeIdentifierV2
    image_id: SafeIdentifierV2
    sequence_id: SafeIdentifierV2
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    altitude_m: float | None = Field(default=None, allow_inf_nan=False)
    horizontal_accuracy_m: float | None = Field(default=None, gt=0, le=100_000)
    captured_at: datetime
    crop_heading_deg: float = Field(ge=0, lt=360)
    vector_sha256: Sha256HexV2

    @model_validator(mode="after")
    def validate_capture(self) -> IndexEntry:
        if not is_timezone_aware(self.captured_at):
            raise ValueError("FAISS entry capture time must include a timezone")
        return self


class FaissIndexManifest(StrictModel):
    schema_name: Literal["fireviewer.panoramax-index.v1"] = "fireviewer.panoramax-index.v1"
    zone_id: SafeIdentifierV2
    panoramax_revision: str = Field(min_length=1, max_length=255)
    model_id: str = Field(min_length=1, max_length=500)
    model_version: str = Field(min_length=1, max_length=255)
    dimension: int = Field(gt=0)
    metric: Literal["cosine"] = "cosine"
    vector_count: int = Field(ge=1)
    vectors_sha256: Sha256HexV2
    index_sha256: Sha256HexV2
    entries: tuple[IndexEntry, ...] = Field(min_length=1, max_length=1_000_000)

    @model_validator(mode="after")
    def validate_entries(self) -> FaissIndexManifest:
        if self.vector_count != len(self.entries):
            raise ValueError("FAISS manifest vector count must match its entries")
        identifiers = [item.embedding_id for item in self.entries]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("FAISS manifest embedding identifiers must be unique")
        return self


class FaissMatch(StrictModel):
    rank: int = Field(ge=1)
    score: float = Field(ge=-1, le=1, allow_inf_nan=False)
    entry: IndexEntry


def _load_faiss() -> Any:
    try:
        import faiss  # type: ignore[import-not-found]
    except ImportError as exc:
        raise FaissUnavailableError(
            "faiss-cpu is required for the Panoramax regional index"
        ) from exc
    return faiss


def _qualified_vectors(vectors: Any, *, expected_rows: int | None = None) -> Any:
    import numpy as np

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("FAISS vectors must be a non-empty two-dimensional matrix")
    if expected_rows is not None and matrix.shape[0] != expected_rows:
        raise ValueError("FAISS vector count does not match metadata entries")
    if not np.isfinite(matrix).all():
        raise ValueError("FAISS vectors contain non-finite values")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms < 0.999) or np.any(norms > 1.001):
        raise ValueError("FAISS cosine vectors must be L2-normalized")
    return np.ascontiguousarray(matrix)


class FaissCosineIndex:
    def __init__(
        self,
        *,
        index: Any,
        entries: tuple[IndexEntry, ...],
        model_id: str,
        model_version: str,
        zone_id: str,
        panoramax_revision: str,
        vectors_sha256: str,
        faiss_module: Any,
    ) -> None:
        self.index = index
        self.entries = entries
        self.model_id = model_id
        self.model_version = model_version
        self.zone_id = zone_id
        self.panoramax_revision = panoramax_revision
        self.vectors_sha256 = vectors_sha256
        self.faiss = faiss_module

    @classmethod
    def build(
        cls,
        *,
        vectors: Any,
        entries: tuple[IndexEntry, ...],
        model_id: str,
        model_version: str,
        zone_id: str,
        panoramax_revision: str,
        faiss_module: Any | None = None,
    ) -> FaissCosineIndex:
        matrix = _qualified_vectors(vectors, expected_rows=len(entries))
        if len({entry.embedding_id for entry in entries}) != len(entries):
            raise ValueError("FAISS entries require unique embedding identifiers")
        runtime = faiss_module or _load_faiss()
        index = runtime.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return cls(
            index=index,
            entries=entries,
            model_id=model_id,
            model_version=model_version,
            zone_id=zone_id,
            panoramax_revision=panoramax_revision,
            vectors_sha256=sha256(matrix.tobytes()).hexdigest(),
            faiss_module=runtime,
        )

    @property
    def dimension(self) -> int:
        return int(self.index.d)

    def search(self, query_vector: Any, *, top_k: int) -> tuple[FaissMatch, ...]:
        if top_k <= 0:
            raise ValueError("FAISS top_k must be positive")
        matrix = _qualified_vectors(query_vector)
        if matrix.shape != (1, self.dimension):
            raise ValueError("FAISS query must contain exactly one compatible descriptor")
        distances, indices = self.index.search(matrix, min(top_k, len(self.entries)))
        matches: list[FaissMatch] = []
        for rank, (index, score) in enumerate(zip(indices[0], distances[0], strict=True), start=1):
            index_value = int(index)
            if index_value < 0:
                continue
            matches.append(
                FaissMatch(
                    rank=rank,
                    score=max(-1.0, min(1.0, float(score))),
                    entry=self.entries[index_value],
                )
            )
        return tuple(matches)

    def serialize(self) -> bytes:
        payload = self.faiss.serialize_index(self.index)
        if isinstance(payload, bytes):
            return payload
        return bytes(payload)

    def manifest(self) -> FaissIndexManifest:
        payload = self.serialize()
        return FaissIndexManifest(
            zone_id=self.zone_id,
            panoramax_revision=self.panoramax_revision,
            model_id=self.model_id,
            model_version=self.model_version,
            dimension=self.dimension,
            vector_count=len(self.entries),
            vectors_sha256=self.vectors_sha256,
            index_sha256=sha256(payload).hexdigest(),
            entries=self.entries,
        )

    @classmethod
    def restore(
        cls,
        payload: bytes,
        *,
        manifest: FaissIndexManifest,
        faiss_module: Any | None = None,
    ) -> FaissCosineIndex:
        import numpy as np

        if sha256(payload).hexdigest() != manifest.index_sha256:
            raise ValueError("FAISS index digest does not match its manifest")
        runtime = faiss_module or _load_faiss()
        index = runtime.deserialize_index(np.frombuffer(payload, dtype=np.uint8))
        if int(index.d) != manifest.dimension or int(index.ntotal) != manifest.vector_count:
            raise ValueError("FAISS index shape does not match its manifest")
        return cls(
            index=index,
            entries=manifest.entries,
            model_id=manifest.model_id,
            model_version=manifest.model_version,
            zone_id=manifest.zone_id,
            panoramax_revision=manifest.panoramax_revision,
            vectors_sha256=manifest.vectors_sha256,
            faiss_module=runtime,
        )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_name = stream.name
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def write_faiss_bundle(index: FaissCosineIndex, directory: Path) -> FaissIndexManifest:
    """Persist index bytes first and its digest-bearing manifest last."""

    if directory.exists() and not directory.is_dir():
        raise ValueError("FAISS bundle target must be a directory")
    payload = index.serialize()
    manifest = index.manifest()
    if sha256(payload).hexdigest() != manifest.index_sha256:
        raise ValueError("serialized FAISS index changed while creating its bundle")
    manifest_payload = json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    _atomic_write(directory / "faiss.index", payload)
    _atomic_write(directory / "index-manifest.json", manifest_payload)
    return manifest


def load_faiss_bundle(
    directory: Path,
    *,
    faiss_module: Any | None = None,
) -> FaissCosineIndex:
    manifest_path = directory / "index-manifest.json"
    index_path = directory / "faiss.index"
    manifest = FaissIndexManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    return FaissCosineIndex.restore(
        index_path.read_bytes(),
        manifest=manifest,
        faiss_module=faiss_module,
    )


__all__ = [
    "FaissCosineIndex",
    "FaissIndexManifest",
    "FaissMatch",
    "FaissUnavailableError",
    "IndexEntry",
    "load_faiss_bundle",
    "write_faiss_bundle",
]
