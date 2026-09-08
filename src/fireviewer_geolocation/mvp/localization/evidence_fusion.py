from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from hashlib import sha256
from math import asin, cos, radians, sin, sqrt

from pydantic import Field, model_validator

from fireviewer_contracts.contracts import StrictModel
from fireviewer_contracts.mvp.contracts import (
    CandidateCluster,
    EventEvidenceV1,
    LocationCandidate,
    ScoreBreakdown,
    Uncertainty,
)
from fireviewer_contracts.mvp.providers import ProviderDescriptor, ProviderHealth


def haversine_m(left: tuple[float, float], right: tuple[float, float]) -> float:
    """Return the great-circle distance for two (longitude, latitude) pairs."""

    left_lon, left_lat = (radians(value) for value in left)
    right_lon, right_lat = (radians(value) for value in right)
    delta_lon = right_lon - left_lon
    delta_lat = right_lat - left_lat
    value = sin(delta_lat / 2) ** 2 + cos(left_lat) * cos(right_lat) * sin(delta_lon / 2) ** 2
    return 2 * 6_371_008.8 * asin(sqrt(value))


class FusionWeights(StrictModel):
    retrieval: float = Field(default=0.30, ge=0, allow_inf_nan=False)
    source_independence: float = Field(default=0.15, ge=0, allow_inf_nan=False)
    geographic_prior: float = Field(default=0.20, ge=0, allow_inf_nan=False)
    metadata: float = Field(default=0.10, ge=0, allow_inf_nan=False)
    independent_media: float = Field(default=0.10, ge=0, allow_inf_nan=False)
    temporal_consistency: float = Field(default=0.05, ge=0, allow_inf_nan=False)
    geometric_verification: float = Field(default=0.10, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_nonzero(self) -> FusionWeights:
        if self.total <= 0:
            raise ValueError("at least one evidence fusion weight must be positive")
        return self

    @property
    def total(self) -> float:
        return sum(
            (
                self.retrieval,
                self.source_independence,
                self.geographic_prior,
                self.metadata,
                self.independent_media,
                self.temporal_consistency,
                self.geometric_verification,
            )
        )


class FusionConfig(StrictModel):
    cluster_distance_m: float = Field(default=2_000, gt=0, le=100_000)
    source_saturation: float = Field(default=3, gt=0, le=100)
    media_saturation: float = Field(default=3, gt=0, le=100)
    human_review_threshold: float = Field(default=0.55, ge=0, le=1)
    ambiguity_margin: float = Field(default=0.08, ge=0, le=1)
    weights: FusionWeights = Field(default_factory=FusionWeights)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parents = list(range(size))

    def find(self, index: int) -> int:
        parent = self.parents[index]
        if parent != index:
            self.parents[index] = self.find(parent)
        return self.parents[index]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parents[max(left_root, right_root)] = min(left_root, right_root)


def _candidate_groups(
    candidates: tuple[LocationCandidate, ...],
    *,
    cluster_distance_m: float,
) -> tuple[tuple[LocationCandidate, ...], ...]:
    ordered = tuple(sorted(candidates, key=lambda item: item.candidate_id))
    union_find = _UnionFind(len(ordered))
    for left_index, left in enumerate(ordered):
        for right_index in range(left_index + 1, len(ordered)):
            right = ordered[right_index]
            if (
                haversine_m(
                    (left.longitude, left.latitude),
                    (right.longitude, right.latitude),
                )
                <= cluster_distance_m
            ):
                union_find.union(left_index, right_index)
    groups: dict[int, list[LocationCandidate]] = defaultdict(list)
    for index, candidate in enumerate(ordered):
        groups[union_find.find(index)].append(candidate)
    return tuple(tuple(groups[key]) for key in sorted(groups))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _max(values: list[float]) -> float:
    return max(values, default=0.0)


class DeterministicEvidenceFusion:
    """Inspectable event-level candidate fusion with no trained fusion model."""

    descriptor = ProviderDescriptor(
        provider_id="deterministic-evidence-fusion",
        provider_version="1.0.0",
        config={},
        capabilities=("candidate-clustering", "source-deduplication", "event-ranking"),
    )

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()

    def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(status="healthy", checked_at=datetime.now(UTC))

    def fuse(self, evidence: EventEvidenceV1) -> EventEvidenceV1:
        groups = _candidate_groups(
            evidence.location_candidates,
            cluster_distance_m=self.config.cluster_distance_m,
        )
        clusters = tuple(self._cluster(evidence, group) for group in groups)
        clusters = tuple(sorted(clusters, key=lambda item: (-item.score, item.cluster_id)))
        uncertainty_code: str | None = None
        if not clusters:
            uncertainty_code = "no_location_candidates"
        elif clusters[0].score < self.config.human_review_threshold:
            uncertainty_code = "candidate_clusters_below_threshold"
        elif (
            len(clusters) > 1
            and clusters[0].score - clusters[1].score < self.config.ambiguity_margin
        ):
            uncertainty_code = "candidate_clusters_ambiguous"
        uncertainties = list(evidence.uncertainties)
        if uncertainty_code is not None and not any(
            item.scope_type == "event"
            and item.scope_id == evidence.event_id
            and item.code == uncertainty_code
            for item in uncertainties
        ):
            uncertainties.append(
                Uncertainty(
                    uncertainty_id=f"UNC-{sha256(f'{evidence.event_id}:{uncertainty_code}'.encode()).hexdigest()[:24]}",
                    code=uncertainty_code,
                    scope_type="event",
                    scope_id=evidence.event_id,
                    description={
                        "no_location_candidates": "No source produced a geographic candidate.",
                        "candidate_clusters_below_threshold": (
                            "The leading candidate cluster remains below the configured "
                            "review gate."
                        ),
                        "candidate_clusters_ambiguous": (
                            "The leading candidate clusters remain too close in score."
                        ),
                    }[uncertainty_code],
                )
            )
        return EventEvidenceV1.model_validate(
            evidence.model_copy(
                update={
                    "candidate_clusters": clusters,
                    "uncertainties": tuple(uncertainties),
                    "needs_human_review": uncertainty_code is not None,
                }
            )
        )

    def _cluster(
        self,
        evidence: EventEvidenceV1,
        candidates: tuple[LocationCandidate, ...],
    ) -> CandidateCluster:
        media_by_id = {item.media_id: item for item in evidence.media}
        source_by_id = {item.source_id: item for item in evidence.sources}
        candidate_weights = [max(candidate.score, 0.01) for candidate in candidates]
        total_candidate_weight = sum(candidate_weights)
        center = (
            sum(
                candidate.longitude * weight
                for candidate, weight in zip(candidates, candidate_weights, strict=True)
            )
            / total_candidate_weight,
            sum(
                candidate.latitude * weight
                for candidate, weight in zip(candidates, candidate_weights, strict=True)
            )
            / total_candidate_weight,
        )
        radius_m = max(
            haversine_m(center, (candidate.longitude, candidate.latitude)) + candidate.radius_m
            for candidate in candidates
        )

        source_ids: set[str] = set()
        media_ids: set[str] = set()
        for candidate in candidates:
            if candidate.source_id is not None:
                source_ids.add(candidate.source_id)
            if candidate.media_id is not None:
                media_ids.add(candidate.media_id)
                media = media_by_id[candidate.media_id]
                source_ids.add(media.source_id)

        origin_weights: dict[str, float] = {}
        for source_id in source_ids:
            source = source_by_id[source_id]
            origin_weights[source.origin_id] = max(
                origin_weights.get(source.origin_id, 0),
                source.independence_weight,
            )
        media_groups = {media_by_id[media_id].media_group_id for media_id in media_ids}

        retrieval_by_media_group: dict[str, float] = {}
        ungrouped_retrieval: list[float] = []
        for candidate in candidates:
            if candidate.evidence_kind != "visual_retrieval":
                continue
            if candidate.media_id is None:
                ungrouped_retrieval.append(candidate.score)
                continue
            group_id = media_by_id[candidate.media_id].media_group_id
            retrieval_by_media_group[group_id] = max(
                retrieval_by_media_group.get(group_id, 0), candidate.score
            )
        retrieval = _mean([*retrieval_by_media_group.values(), *ungrouped_retrieval])
        source_independence = min(
            sum(origin_weights.values()) / self.config.source_saturation,
            1.0,
        )
        independent_media = min(len(media_groups) / self.config.media_saturation, 1.0)
        metadata = _max(
            [candidate.score for candidate in candidates if candidate.evidence_kind == "metadata"]
        )
        temporal_consistency = _mean(
            [
                candidate.temporal_consistency
                for candidate in candidates
                if candidate.temporal_consistency is not None
            ]
        )
        geometric_verification = _max(
            [
                candidate.geometric_consistency
                if candidate.geometric_consistency is not None
                else candidate.score
                for candidate in candidates
                if candidate.evidence_kind == "geometric_verification"
                or candidate.geometric_consistency is not None
            ]
        )
        geographic_prior = 0.0
        if evidence.candidate_area is not None:
            prior_radius_m = evidence.candidate_area.radius_km * 1_000
            distance_m = haversine_m(center, evidence.candidate_area.center)
            geographic_prior = evidence.candidate_area.confidence * max(
                0.0,
                1.0 - distance_m / (2 * prior_radius_m),
            )
        breakdown = ScoreBreakdown(
            retrieval=retrieval,
            source_independence=source_independence,
            geographic_prior=geographic_prior,
            metadata=metadata,
            independent_media=independent_media,
            temporal_consistency=temporal_consistency,
            geometric_verification=geometric_verification,
        )
        weights = self.config.weights
        score = (
            breakdown.retrieval * weights.retrieval
            + breakdown.source_independence * weights.source_independence
            + breakdown.geographic_prior * weights.geographic_prior
            + breakdown.metadata * weights.metadata
            + breakdown.independent_media * weights.independent_media
            + breakdown.temporal_consistency * weights.temporal_consistency
            + breakdown.geometric_verification * weights.geometric_verification
        ) / weights.total
        candidate_ids = tuple(sorted(candidate.candidate_id for candidate in candidates))
        cluster_digest = sha256("\x1f".join(candidate_ids).encode()).hexdigest()[:24]
        return CandidateCluster(
            cluster_id=f"CLUSTER-{cluster_digest}",
            center=center,
            radius_m=radius_m,
            score=score,
            score_breakdown=breakdown,
            supporting_candidate_ids=candidate_ids,
            supporting_source_ids=tuple(sorted(source_ids)),
            supporting_media_ids=tuple(sorted(media_ids)),
            independent_source_count=len(origin_weights),
            independent_media_count=len(media_groups),
        )
