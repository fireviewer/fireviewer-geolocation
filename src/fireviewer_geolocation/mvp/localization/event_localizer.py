from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field

from fireviewer_contracts.contracts import SafeIdentifierV2, StrictModel
from fireviewer_contracts.mvp.contracts import (
    EventEvidenceV1,
    EvidenceMedia,
    LocationCandidate,
    Uncertainty,
)
from fireviewer_geolocation.mvp.localization.evidence_fusion import DeterministicEvidenceFusion
from fireviewer_geolocation.mvp.localization.megaloc import MegaLocBatch
from fireviewer_geolocation.mvp.localization.retrieval import MegaLocFaissRetriever
from fireviewer_contracts.mvp.providers import EvidenceFusionProvider


class EvidenceImageLoader(Protocol):
    def load(self, media: EvidenceMedia) -> object: ...


class EventMegaLocEncoder(Protocol):
    model_id: str
    model_version: str

    def encode(
        self,
        media: tuple[tuple[SafeIdentifierV2, object], ...],
    ) -> tuple[MegaLocBatch, Any]: ...


class LocalEvidenceImageLoader:
    """Load materialized event images from a bounded directory and verify their digests."""

    def __init__(
        self,
        *,
        root: Path,
        relative_paths_by_media_id: Mapping[str, str],
        max_bytes: int = 50_000_000,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("event media byte limit must be positive")
        self.root = root.resolve(strict=True)
        self.relative_paths_by_media_id = dict(relative_paths_by_media_id)
        self.max_bytes = max_bytes

    def load(self, media: EvidenceMedia) -> object:
        relative_path = self.relative_paths_by_media_id.get(media.media_id)
        if relative_path is None:
            raise FileNotFoundError(f"no materialized path for media {media.media_id}")
        candidate = (self.root / relative_path).resolve(strict=True)
        if not candidate.is_relative_to(self.root) or not candidate.is_file():
            raise ValueError("event media path leaves the configured materialization root")
        if candidate.stat().st_size > self.max_bytes:
            raise ValueError("event media exceeds the configured byte limit")
        payload = candidate.read_bytes()
        if sha256(payload).hexdigest() != media.sha256:
            raise ValueError("event media digest does not match its evidence contract")

        from PIL import Image

        with Image.open(BytesIO(payload)) as image:
            image.load()
            return image.convert("RGB")


class EventLocalizationConfig(StrictModel):
    eligible_kinds: tuple[Literal["photo", "keyframe"], ...] = ("photo", "keyframe")
    query_batch_size: int = Field(default=8, ge=1, le=256)
    max_media: int = Field(default=256, ge=1, le=512)


_FUSION_UNCERTAINTY_CODES = {
    "no_location_candidates",
    "candidate_clusters_below_threshold",
    "candidate_clusters_ambiguous",
}
_RUNTIME_UNCERTAINTY_CODES = {
    "media_load_failed",
    "visual_embedding_failed",
    "visual_retrieval_failed",
    "no_retrievable_media",
    "media_limit_applied",
}


def abstain_for_missing_reference_coverage(
    evidence: EventEvidenceV1,
    *,
    fusion: EvidenceFusionProvider | None = None,
) -> EventEvidenceV1:
    """Record a qualified no-coverage result without running image inference."""

    code = "panoramax_no_coverage"
    uncertainties = list(evidence.uncertainties)
    if not any(
        item.scope_type == "event" and item.scope_id == evidence.event_id and item.code == code
        for item in uncertainties
    ):
        uncertainties.append(
            Uncertainty(
                uncertainty_id=(
                    f"UNC-{sha256(f'{evidence.event_id}:{code}'.encode()).hexdigest()[:24]}"
                ),
                code=code,
                scope_type="event",
                scope_id=evidence.event_id,
                description=(
                    "The regional Panoramax query returned no reference imagery for this event."
                ),
            )
        )
    unresolved = EventEvidenceV1.model_validate(
        evidence.model_copy(
            update={
                "location_candidates": (),
                "candidate_clusters": (),
                "uncertainties": tuple(uncertainties),
                "needs_human_review": True,
            }
        )
    )
    return (fusion or DeterministicEvidenceFusion()).fuse(unresolved)


class MegaLocFaissEventLocalizer:
    """Run image loading, MegaLoc retrieval and deterministic fusion for one event."""

    def __init__(
        self,
        *,
        image_loader: EvidenceImageLoader,
        encoder: EventMegaLocEncoder,
        retriever: MegaLocFaissRetriever,
        fusion: EvidenceFusionProvider | None = None,
        config: EventLocalizationConfig | None = None,
    ) -> None:
        if (
            encoder.model_id != retriever.index.model_id
            or encoder.model_version != retriever.index.model_version
        ):
            raise ValueError("event encoder identity does not match the regional FAISS index")
        self.image_loader = image_loader
        self.encoder = encoder
        self.retriever = retriever
        self.fusion = fusion or DeterministicEvidenceFusion()
        self.config = config or EventLocalizationConfig()

    def localize(
        self,
        evidence: EventEvidenceV1,
        *,
        media_ids: tuple[SafeIdentifierV2, ...] | None = None,
    ) -> EventEvidenceV1:
        media_by_id = {item.media_id: item for item in evidence.media}
        if media_ids is not None:
            unknown = set(media_ids) - set(media_by_id)
            if unknown:
                raise ValueError("event localization requested unknown media identifiers")
            if len(media_ids) != len(set(media_ids)):
                raise ValueError("event localization media identifiers must be unique")
            requested = set(media_ids)
        else:
            requested = set(media_by_id)

        eligible = tuple(
            item
            for item in sorted(evidence.media, key=lambda value: value.media_id)
            if item.media_id in requested and item.kind in self.config.eligible_kinds
        )
        limited = len(eligible) > self.config.max_media
        selected = eligible[: self.config.max_media]
        selected_ids = {item.media_id for item in selected}
        uncertainties = self._clean_uncertainties(evidence, selected_ids)
        failures = False

        if limited:
            uncertainties.append(
                self._uncertainty(
                    event_id=evidence.event_id,
                    code="media_limit_applied",
                    scope_type="event",
                    scope_id=evidence.event_id,
                    description="The event media set exceeded the configured localization limit.",
                )
            )
            failures = True
        if not selected:
            uncertainties.append(
                self._uncertainty(
                    event_id=evidence.event_id,
                    code="no_retrievable_media",
                    scope_type="event",
                    scope_id=evidence.event_id,
                    description="The event contains no photo or keyframe eligible for retrieval.",
                )
            )
            failures = True

        loaded: list[tuple[SafeIdentifierV2, object]] = []
        for media in selected:
            try:
                loaded.append((media.media_id, self.image_loader.load(media)))
            except Exception:
                uncertainties.append(
                    self._uncertainty(
                        event_id=evidence.event_id,
                        code="media_load_failed",
                        scope_type="media",
                        scope_id=media.media_id,
                        description="The qualified local media payload could not be loaded.",
                    )
                )
                failures = True

        descriptors: dict[SafeIdentifierV2, Any] = {}
        for offset in range(0, len(loaded), self.config.query_batch_size):
            chunk = tuple(loaded[offset : offset + self.config.query_batch_size])
            chunk_vectors, chunk_failures = self._encode_with_isolation(
                evidence.event_id,
                chunk,
                uncertainties,
            )
            descriptors.update(chunk_vectors)
            failures = failures or chunk_failures

        candidates: list[LocationCandidate] = []
        for media_id in sorted(descriptors):
            try:
                candidates.extend(
                    self.retriever.retrieve_one(
                        media_id=media_id,
                        query_vector=descriptors[media_id],
                    )
                )
            except Exception:
                uncertainties.append(
                    self._uncertainty(
                        event_id=evidence.event_id,
                        code="visual_retrieval_failed",
                        scope_type="media",
                        scope_id=media_id,
                        description="The regional FAISS index could not retrieve this media.",
                    )
                )
                failures = True

        retained_candidates = tuple(
            item
            for item in evidence.location_candidates
            if not (
                item.provider_id == self.retriever.descriptor.provider_id
                and item.media_id in selected_ids
            )
        )
        prepared = EventEvidenceV1.model_validate(
            evidence.model_copy(
                update={
                    "location_candidates": (*retained_candidates, *candidates),
                    "candidate_clusters": (),
                    "uncertainties": tuple(uncertainties),
                    "needs_human_review": failures,
                }
            )
        )
        fused = self.fusion.fuse(prepared)
        if failures and not fused.needs_human_review:
            fused = EventEvidenceV1.model_validate(
                fused.model_copy(update={"needs_human_review": True})
            )
        return fused

    def _encode_with_isolation(
        self,
        event_id: SafeIdentifierV2,
        media: tuple[tuple[SafeIdentifierV2, object], ...],
        uncertainties: list[Uncertainty],
    ) -> tuple[dict[SafeIdentifierV2, Any], bool]:
        if not media:
            return {}, False
        try:
            batch, vectors = self.encoder.encode(media)
            return self._qualified_descriptors(batch, vectors, media), False
        except Exception:
            if len(media) > 1:
                midpoint = len(media) // 2
                left, left_failed = self._encode_with_isolation(
                    event_id, media[:midpoint], uncertainties
                )
                right, right_failed = self._encode_with_isolation(
                    event_id, media[midpoint:], uncertainties
                )
                return {**left, **right}, left_failed or right_failed
            media_id = media[0][0]
            uncertainties.append(
                self._uncertainty(
                    event_id=event_id,
                    code="visual_embedding_failed",
                    scope_type="media",
                    scope_id=media_id,
                    description="MegaLoc could not produce a qualified descriptor for this media.",
                )
            )
            return {}, True

    def _qualified_descriptors(
        self,
        batch: MegaLocBatch,
        vectors: Any,
        media: tuple[tuple[SafeIdentifierV2, object], ...],
    ) -> dict[SafeIdentifierV2, Any]:
        import numpy as np

        if (
            batch.model_id != self.encoder.model_id
            or batch.model_version != self.encoder.model_version
        ):
            raise ValueError("MegaLoc query batch identity changed during event localization")
        array = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        if array.ndim != 2 or array.shape[0] != len(media):
            raise ValueError("MegaLoc query batch shape does not match its media")
        if len(batch.embeddings) != len(media):
            raise ValueError("MegaLoc query metadata count does not match its media")
        descriptors: dict[SafeIdentifierV2, Any] = {}
        for index, ((media_id, _), embedding) in enumerate(
            zip(media, batch.embeddings, strict=True)
        ):
            descriptor = np.ascontiguousarray(array[index : index + 1])
            if embedding.media_id != media_id:
                raise ValueError("MegaLoc query embedding order does not match its media")
            if sha256(descriptor[0].tobytes()).hexdigest() != embedding.vector_sha256:
                raise ValueError("MegaLoc query embedding digest does not match its vector")
            descriptors[media_id] = descriptor
        return descriptors

    @staticmethod
    def _uncertainty(
        *,
        event_id: SafeIdentifierV2,
        code: SafeIdentifierV2,
        scope_type: Literal["event", "media"],
        scope_id: SafeIdentifierV2,
        description: str,
    ) -> Uncertainty:
        digest = sha256(f"{event_id}:{code}:{scope_type}:{scope_id}".encode()).hexdigest()[:24]
        return Uncertainty(
            uncertainty_id=f"UNC-{digest}",
            code=code,
            scope_type=scope_type,
            scope_id=scope_id,
            description=description,
        )

    @staticmethod
    def _clean_uncertainties(
        evidence: EventEvidenceV1,
        selected_ids: set[str],
    ) -> list[Uncertainty]:
        return [
            item
            for item in evidence.uncertainties
            if item.code not in _FUSION_UNCERTAINTY_CODES
            and not (
                item.code in _RUNTIME_UNCERTAINTY_CODES
                and (item.scope_type == "event" or item.scope_id in selected_ids)
            )
        ]


__all__ = [
    "EventLocalizationConfig",
    "EventMegaLocEncoder",
    "EvidenceImageLoader",
    "LocalEvidenceImageLoader",
    "MegaLocFaissEventLocalizer",
]
