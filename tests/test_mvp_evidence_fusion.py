from __future__ import annotations

from copy import deepcopy

from fireviewer_contracts.mvp.contracts import EventEvidenceV1
from fireviewer_geolocation.mvp.localization import DeterministicEvidenceFusion, FusionConfig


def _payload() -> dict[str, object]:
    return {
        "schema": "fireviewer.event-evidence.v1",
        "event_id": "EVENT-FUSION-1",
        "candidate_area": {
            "center": [5.37, 44.75],
            "radius_km": 15,
            "confidence": 0.8,
            "supporting_source_ids": ["SOURCE-ARTICLE"],
        },
        "sources": [
            {
                "source_id": "SOURCE-ARTICLE",
                "origin_id": "ORIGIN-ARTICLE",
                "publisher": "Original article",
                "retrieved_at": "2026-08-21T10:00:00Z",
                "source_type": "press",
                "independence_weight": 1,
            },
            {
                "source_id": "SOURCE-REPOST",
                "origin_id": "ORIGIN-ARTICLE",
                "publisher": "Article aggregator",
                "retrieved_at": "2026-08-21T10:01:00Z",
                "source_type": "press",
                "independence_weight": 0.2,
            },
            {
                "source_id": "SOURCE-VIDEO",
                "origin_id": "ORIGIN-VIDEO",
                "publisher": "Witness",
                "retrieved_at": "2026-08-21T10:02:00Z",
                "source_type": "witness",
                "independence_weight": 1,
            },
        ],
        "media": [
            {
                "media_id": "VIDEO-1",
                "source_id": "SOURCE-VIDEO",
                "media_group_id": "GROUP-VIDEO-1",
                "origin_id": "ORIGIN-VIDEO",
                "kind": "video",
                "sha256": "a" * 64,
            },
            {
                "media_id": "FRAME-1",
                "source_id": "SOURCE-VIDEO",
                "media_group_id": "GROUP-VIDEO-1",
                "origin_id": "ORIGIN-VIDEO",
                "kind": "keyframe",
                "sha256": "b" * 64,
                "parent_media_id": "VIDEO-1",
            },
            {
                "media_id": "FRAME-2",
                "source_id": "SOURCE-VIDEO",
                "media_group_id": "GROUP-VIDEO-1",
                "origin_id": "ORIGIN-VIDEO",
                "kind": "keyframe",
                "sha256": "c" * 64,
                "parent_media_id": "VIDEO-1",
            },
        ],
        "location_candidates": [
            {
                "candidate_id": "CANDIDATE-FRAME-1",
                "longitude": 5.370,
                "latitude": 44.750,
                "radius_m": 100,
                "score": 0.9,
                "rank": 1,
                "evidence_kind": "visual_retrieval",
                "provider_id": "megaloc",
                "provider_version": "mvp-1",
                "media_id": "FRAME-1",
                "reference_id": "PANORAMAX-1",
                "temporal_consistency": 0.9,
            },
            {
                "candidate_id": "CANDIDATE-FRAME-2",
                "longitude": 5.371,
                "latitude": 44.751,
                "radius_m": 100,
                "score": 0.8,
                "rank": 1,
                "evidence_kind": "visual_retrieval",
                "provider_id": "megaloc",
                "provider_version": "mvp-1",
                "media_id": "FRAME-2",
                "reference_id": "PANORAMAX-2",
                "temporal_consistency": 0.9,
            },
            {
                "candidate_id": "CANDIDATE-ARTICLE",
                "longitude": 5.369,
                "latitude": 44.749,
                "radius_m": 1_000,
                "score": 0.8,
                "rank": 1,
                "evidence_kind": "research_prior",
                "provider_id": "research-fixture",
                "provider_version": "mvp-1",
                "source_id": "SOURCE-ARTICLE",
            },
            {
                "candidate_id": "CANDIDATE-REPOST",
                "longitude": 5.369,
                "latitude": 44.749,
                "radius_m": 1_000,
                "score": 0.7,
                "rank": 1,
                "evidence_kind": "research_prior",
                "provider_id": "research-fixture",
                "provider_version": "mvp-1",
                "source_id": "SOURCE-REPOST",
            },
            {
                "candidate_id": "CANDIDATE-FAR",
                "longitude": 5.50,
                "latitude": 44.90,
                "radius_m": 200,
                "score": 0.4,
                "rank": 2,
                "evidence_kind": "visual_retrieval",
                "provider_id": "megaloc",
                "provider_version": "mvp-1",
                "media_id": "FRAME-1",
                "reference_id": "PANORAMAX-FAR",
            },
        ],
    }


def test_fusion_counts_origins_and_media_groups_instead_of_raw_observations() -> None:
    result = DeterministicEvidenceFusion(
        FusionConfig(human_review_threshold=0.3, ambiguity_margin=0.02)
    ).fuse(EventEvidenceV1.model_validate(_payload()))

    assert len(result.candidate_clusters) == 2
    leading = result.candidate_clusters[0]
    assert leading.supporting_candidate_ids == (
        "CANDIDATE-ARTICLE",
        "CANDIDATE-FRAME-1",
        "CANDIDATE-FRAME-2",
        "CANDIDATE-REPOST",
    )
    assert leading.independent_source_count == 2
    assert leading.independent_media_count == 1
    assert leading.score_breakdown.retrieval == 0.9
    assert leading.score > result.candidate_clusters[1].score
    assert result.needs_human_review is False


def test_fusion_is_order_independent_and_cluster_ids_are_stable() -> None:
    payload = _payload()
    reversed_payload = deepcopy(payload)
    reversed_payload["location_candidates"] = list(reversed(payload["location_candidates"]))
    provider = DeterministicEvidenceFusion()

    first = provider.fuse(EventEvidenceV1.model_validate(payload))
    second = provider.fuse(EventEvidenceV1.model_validate(reversed_payload))

    assert first.candidate_clusters == second.candidate_clusters


def test_empty_fusion_abstains_and_does_not_duplicate_uncertainty_on_replay() -> None:
    payload = _payload()
    payload["location_candidates"] = []
    provider = DeterministicEvidenceFusion()

    first = provider.fuse(EventEvidenceV1.model_validate(payload))
    second = provider.fuse(first)

    assert first.needs_human_review is True
    assert first.candidate_clusters == ()
    assert first.uncertainties[0].code == "no_location_candidates"
    assert second.uncertainties == first.uncertainties
