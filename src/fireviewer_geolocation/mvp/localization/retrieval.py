from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from pydantic import Field

from fireviewer_contracts.contracts import SafeIdentifierV2, StrictModel
from fireviewer_contracts.mvp.contracts import LocationCandidate
from fireviewer_geolocation.mvp.localization.faiss_index import FaissCosineIndex
from fireviewer_contracts.mvp.providers import ProviderDescriptor, ProviderHealth


class RetrievalConfig(StrictModel):
    top_k: int = Field(default=20, ge=1, le=1_000)
    default_reference_radius_m: float = Field(default=25, gt=0, le=10_000)


class MegaLocFaissRetriever:
    """Convert MegaLoc cosine matches into per-media geographic candidates."""

    def __init__(
        self,
        index: FaissCosineIndex,
        *,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.index = index
        self.config = config or RetrievalConfig()
        self.descriptor = ProviderDescriptor(
            provider_id="megaloc-faiss",
            provider_version="1.0.0",
            model_id=index.model_id,
            model_version=index.model_version,
            config={
                **self.config.model_dump(mode="json"),
                "zone_id": index.zone_id,
                "panoramax_revision": index.panoramax_revision,
            },
            capabilities=("visual-place-candidates",),
        )

    def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(status="healthy", checked_at=datetime.now(UTC))

    def retrieve_one(
        self,
        *,
        media_id: SafeIdentifierV2,
        query_vector: Any,
    ) -> tuple[LocationCandidate, ...]:
        matches = self.index.search(query_vector, top_k=self.config.top_k)
        return tuple(
            LocationCandidate(
                candidate_id=f"CAND-{sha256(f'{media_id}:{match.entry.embedding_id}:{self.index.model_version}'.encode()).hexdigest()[:24]}",
                longitude=match.entry.longitude,
                latitude=match.entry.latitude,
                radius_m=(
                    match.entry.horizontal_accuracy_m or self.config.default_reference_radius_m
                ),
                score=(match.score + 1) / 2,
                raw_score=match.score,
                rank=match.rank,
                evidence_kind="visual_retrieval",
                provider_id=self.descriptor.provider_id,
                provider_version=self.descriptor.provider_version,
                media_id=media_id,
                reference_id=match.entry.image_id,
            )
            for match in matches
        )

    def retrieve_many(
        self,
        descriptors_by_media: dict[SafeIdentifierV2, Any],
    ) -> tuple[LocationCandidate, ...]:
        return tuple(
            candidate
            for media_id in sorted(descriptors_by_media)
            for candidate in self.retrieve_one(
                media_id=media_id,
                query_vector=descriptors_by_media[media_id],
            )
        )
