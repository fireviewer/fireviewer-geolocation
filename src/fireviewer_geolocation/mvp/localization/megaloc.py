from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from hashlib import sha256
from typing import Any

from pydantic import Field, model_validator

from fireviewer_contracts.contracts import SafeIdentifierV2, Sha256HexV2, StrictModel
from fireviewer_contracts.mvp.providers import ProviderDescriptor, ProviderHealth


class MegaLocError(RuntimeError):
    """Raised when a local MegaLoc inference cannot produce qualified descriptors."""


class MegaLocConfig(StrictModel):
    image_size_px: int = Field(default=322, ge=32, le=2_048)
    expected_dimension: int = Field(default=8_448, ge=1, le=100_000)
    batch_size: int = Field(default=8, ge=1, le=256)
    device: str = Field(default="cpu", min_length=3, max_length=64)


class MegaLocEmbedding(StrictModel):
    embedding_id: SafeIdentifierV2
    media_id: SafeIdentifierV2
    dimension: int = Field(gt=0)
    vector_sha256: Sha256HexV2


class MegaLocBatch(StrictModel):
    model_id: str = Field(min_length=1, max_length=500)
    model_version: str = Field(min_length=1, max_length=255)
    embeddings: tuple[MegaLocEmbedding, ...] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_embeddings(self) -> MegaLocBatch:
        identifiers = [item.embedding_id for item in self.embeddings]
        media_ids = [item.media_id for item in self.embeddings]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("MegaLoc embedding identifiers must be unique")
        if len(media_ids) != len(set(media_ids)):
            raise ValueError("MegaLoc batch may contain one descriptor per media item")
        dimensions = {item.dimension for item in self.embeddings}
        if len(dimensions) != 1:
            raise ValueError("MegaLoc batch embeddings must share one dimension")
        return self


class TorchMegaLocEncoder:
    """Local-only MegaLoc adapter using an explicitly supplied immutable model loader."""

    def __init__(
        self,
        *,
        model_loader: Callable[[], Any],
        model_id: str = "gberton/MegaLoc",
        model_version: str,
        config: MegaLocConfig | None = None,
    ) -> None:
        self.model_loader = model_loader
        self.model_id = model_id
        self.model_version = model_version
        self.config = config or MegaLocConfig()
        self.descriptor = ProviderDescriptor(
            provider_id="megaloc",
            provider_version="1.0.0",
            model_id=model_id,
            model_version=model_version,
            config=self.config.model_dump(mode="json"),
            capabilities=("image-retrieval-embedding",),
        )

    def healthcheck(self) -> ProviderHealth:
        from datetime import UTC, datetime

        return ProviderHealth(status="healthy", checked_at=datetime.now(UTC))

    def encode(
        self,
        media: tuple[tuple[SafeIdentifierV2, object], ...],
    ) -> tuple[MegaLocBatch, Any]:
        import numpy as np
        import torch
        from PIL import Image

        if not media:
            raise ValueError("MegaLoc encoding requires at least one image")
        media_ids = [media_id for media_id, _ in media]
        if len(media_ids) != len(set(media_ids)):
            raise ValueError("MegaLoc input media identifiers must be unique")
        tensors = []
        for _, raw_image in media:
            if not isinstance(raw_image, Image.Image):
                raise TypeError("MegaLoc inputs must be Pillow images")
            resized = raw_image.convert("RGB").resize(
                (self.config.image_size_px, self.config.image_size_px),
                resample=Image.Resampling.BILINEAR,
            )
            pixels = np.asarray(resized, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(pixels).permute(2, 0, 1)
            mean = torch.tensor((0.485, 0.456, 0.406), dtype=tensor.dtype)[:, None, None]
            std = torch.tensor((0.229, 0.224, 0.225), dtype=tensor.dtype)[:, None, None]
            tensors.append((tensor - mean) / std)

        model: Any | None = None
        batches: list[Any] = []
        try:
            model = self.model_loader()
            model = model.to(self.config.device)
            model.eval()
            with torch.inference_mode():
                for offset in range(0, len(tensors), self.config.batch_size):
                    batch = torch.stack(tensors[offset : offset + self.config.batch_size]).to(
                        self.config.device
                    )
                    output = model(batch)
                    if isinstance(output, tuple | list):
                        output = output[0]
                    if not isinstance(output, torch.Tensor):
                        raise MegaLocError("MegaLoc model returned a non-tensor output")
                    batches.append(output.detach().float().cpu())
        except MegaLocError:
            raise
        except Exception as exc:
            raise MegaLocError("MegaLoc local inference failed") from exc
        finally:
            if model is not None:
                with suppress(Exception):
                    model.to("cpu")

        vectors = torch.cat(batches, dim=0).numpy().astype(np.float32, copy=False)
        if vectors.shape != (len(media), self.config.expected_dimension):
            raise MegaLocError(
                "MegaLoc descriptor matrix does not match the configured count/dimension"
            )
        if not np.isfinite(vectors).all():
            raise MegaLocError("MegaLoc descriptor matrix contains non-finite values")
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms < 0.95) or np.any(norms > 1.05):
            raise MegaLocError("MegaLoc descriptors are not L2-normalized")
        vectors = vectors / norms[:, None]
        embeddings = tuple(
            MegaLocEmbedding(
                embedding_id=f"EMB-{sha256(f'{media_id}:{self.model_version}'.encode()).hexdigest()[:24]}",
                media_id=media_id,
                dimension=self.config.expected_dimension,
                vector_sha256=sha256(vectors[index].tobytes()).hexdigest(),
            )
            for index, media_id in enumerate(media_ids)
        )
        return (
            MegaLocBatch(
                model_id=self.model_id,
                model_version=self.model_version,
                embeddings=embeddings,
            ),
            vectors,
        )
