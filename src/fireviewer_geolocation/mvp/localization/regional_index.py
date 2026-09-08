from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from typing import Any, Protocol

from pydantic import Field

from fireviewer_contracts.contracts import StrictModel
from fireviewer_geolocation.mvp.localization.faiss_index import FaissCosineIndex, IndexEntry
from fireviewer_geolocation.mvp.localization.megaloc import MegaLocBatch
from fireviewer_geolocation.mvp.localization.panoramax import PanoramaxImage, PanoramaxSearchResult
from fireviewer_geolocation.mvp.localization.perspective import (
    PerspectiveConfig,
    PerspectiveCropManifest,
    generate_perspective_crops,
)


class PanoramaxImageLoader(Protocol):
    def load(self, image: PanoramaxImage) -> object: ...


class MegaLocBatchEncoder(Protocol):
    model_id: str
    model_version: str

    def encode(
        self,
        media: tuple[tuple[str, object], ...],
    ) -> tuple[MegaLocBatch, Any]: ...


class RegionalIndexConfig(StrictModel):
    perspective: PerspectiveConfig = Field(default_factory=PerspectiveConfig)
    encode_batch_size: int = Field(default=32, ge=1, le=1_024)
    panorama_fov_threshold_deg: float = Field(default=180, ge=180, le=360)


class PanoramaxRegionalIndexBuilder:
    """Build a qualified regional FAISS index from pinned Panoramax search results."""

    def __init__(
        self,
        *,
        image_loader: PanoramaxImageLoader,
        encoder: MegaLocBatchEncoder,
        panoramax_revision: str,
        config: RegionalIndexConfig | None = None,
        faiss_module: Any | None = None,
    ) -> None:
        if not panoramax_revision.strip():
            raise ValueError("Panoramax index construction requires an immutable revision")
        self.image_loader = image_loader
        self.encoder = encoder
        self.panoramax_revision = panoramax_revision
        self.config = config or RegionalIndexConfig()
        self.faiss_module = faiss_module

    def build(self, search_result: PanoramaxSearchResult) -> FaissCosineIndex:
        if not search_result.images:
            raise ValueError("Panoramax regional index requires at least one source image")

        vectors: list[Any] = []
        entries: list[IndexEntry] = []
        pending: list[tuple[PanoramaxImage, PerspectiveCropManifest, object]] = []

        def flush() -> None:
            if not pending:
                return
            media = tuple((crop.crop_id, image) for _, crop, image in pending)
            batch, matrix = self.encoder.encode(media)
            self._append_batch(batch, matrix, pending, vectors, entries)
            pending.clear()

        for source in sorted(search_result.images, key=lambda item: item.image_id):
            source_image = self.image_loader.load(source)
            for crop, image in self._views(source, source_image):
                pending.append((source, crop, image))
                if len(pending) >= self.config.encode_batch_size:
                    flush()
        flush()

        import numpy as np

        matrix = np.concatenate(vectors, axis=0)
        return FaissCosineIndex.build(
            vectors=matrix,
            entries=tuple(entries),
            model_id=self.encoder.model_id,
            model_version=self.encoder.model_version,
            zone_id=search_result.zone_id,
            panoramax_revision=self.panoramax_revision,
            faiss_module=self.faiss_module,
        )

    def _views(
        self,
        source: PanoramaxImage,
        source_image: object,
    ) -> tuple[tuple[PerspectiveCropManifest, object], ...]:
        if (
            source.field_of_view_deg is not None
            and source.field_of_view_deg >= self.config.panorama_fov_threshold_deg
        ):
            return generate_perspective_crops(
                source.image_id,
                source_image,
                config=self.config.perspective,
            )

        from PIL import Image

        if not isinstance(source_image, Image.Image):
            raise TypeError("Panoramax index inputs must be Pillow images")
        native = source_image.convert("RGB")
        pixel_payload = f"{native.mode}:{native.width}x{native.height}:".encode() + native.tobytes()
        view = PerspectiveCropManifest(
            crop_id=f"{source.image_id}-native",
            image_id=source.image_id,
            heading_deg=0,
            pitch_deg=0,
            horizontal_fov_deg=source.field_of_view_deg or 90,
            width_px=native.width,
            height_px=native.height,
            pixel_sha256=sha256(pixel_payload).hexdigest(),
        )
        return ((view, native),)

    def _append_batch(
        self,
        batch: MegaLocBatch,
        matrix: Any,
        pending: list[tuple[PanoramaxImage, PerspectiveCropManifest, object]],
        vectors: list[Any],
        entries: list[IndexEntry],
    ) -> None:
        import numpy as np

        if (
            batch.model_id != self.encoder.model_id
            or batch.model_version != self.encoder.model_version
        ):
            raise ValueError("MegaLoc batch identity changed during regional index construction")
        array = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))
        if array.ndim != 2 or array.shape[0] != len(pending):
            raise ValueError("MegaLoc batch shape does not match its perspective crops")
        if len(batch.embeddings) != len(pending):
            raise ValueError("MegaLoc metadata count does not match its perspective crops")

        for row_index, ((source, crop, _), embedding) in enumerate(
            zip(pending, batch.embeddings, strict=True)
        ):
            row = np.ascontiguousarray(array[row_index : row_index + 1])
            if embedding.media_id != crop.crop_id:
                raise ValueError("MegaLoc embedding order does not match its perspective crops")
            vector_digest = sha256(row[0].tobytes()).hexdigest()
            if embedding.vector_sha256 != vector_digest:
                raise ValueError("MegaLoc embedding digest does not match its descriptor vector")
            base_heading = source.heading_deg if source.heading_deg is not None else source.yaw_deg
            entries.append(
                IndexEntry(
                    embedding_id=embedding.embedding_id,
                    image_id=source.image_id,
                    sequence_id=source.sequence_id,
                    longitude=source.longitude,
                    latitude=source.latitude,
                    altitude_m=source.altitude_m,
                    horizontal_accuracy_m=source.gps_accuracy_m,
                    captured_at=source.captured_at,
                    crop_heading_deg=((base_heading or 0) + crop.heading_deg) % 360,
                    vector_sha256=vector_digest,
                )
            )
            vectors.append(row)


class CallablePanoramaxImageLoader:
    """Small adapter for an explicit cache or HTTP image-loading function."""

    def __init__(self, loader: Callable[[PanoramaxImage], object]) -> None:
        self.loader = loader

    def load(self, image: PanoramaxImage) -> object:
        return self.loader(image)


__all__ = [
    "CallablePanoramaxImageLoader",
    "MegaLocBatchEncoder",
    "PanoramaxImageLoader",
    "PanoramaxRegionalIndexBuilder",
    "RegionalIndexConfig",
]
