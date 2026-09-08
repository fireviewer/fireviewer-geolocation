from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from fireviewer_contracts.mvp.contracts import EventEvidenceV1
from fireviewer_geolocation.mvp.localization.azure_maps import (
    AzureMapsConfig,
    AzureMapsError,
    AzureMapsGeoEnrichmentProvider,
    AzureMapsLocationQuery,
)

MAPS_CLIENT_ID = "11111111-1111-4111-8111-111111111111"
IDENTITY_CLIENT_ID = "22222222-2222-4222-8222-222222222222"
NOW = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)


def _evidence() -> EventEvidenceV1:
    return EventEvidenceV1.model_validate(
        {
            "schema": "fireviewer.event-evidence.v1",
            "event_id": "EVENT-DIE-1",
            "sources": [
                {
                    "source_id": "SOURCE-OFFICIAL-1",
                    "origin_id": "ORIGIN-SDIS-26",
                    "publisher": "SDIS 26",
                    "retrieved_at": "2026-08-22T10:00:00Z",
                    "source_type": "official",
                    "independence_weight": 1,
                }
            ],
            "claims": [
                {
                    "claim_id": "CLAIM-LOCATION-1",
                    "source_id": "SOURCE-OFFICIAL-1",
                    "claim_type": "incident_location",
                    "text": "Commune de Die, Drome, France",
                    "confidence": 0.8,
                }
            ],
        }
    )


def _query() -> AzureMapsLocationQuery:
    return AzureMapsLocationQuery(
        query_id="QUERY-DIE-1",
        source_id="SOURCE-OFFICIAL-1",
        claim_id="CLAIM-LOCATION-1",
        query="Die, Drome, France",
        view="FR",
    )


class _Transport:
    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, str | int], dict[str, str]]] = []

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, str | int],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        self.calls.append((url, params, headers))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _provider(transport: _Transport) -> AzureMapsGeoEnrichmentProvider:
    return AzureMapsGeoEnrichmentProvider(
        config=AzureMapsConfig(
            account_client_id=MAPS_CLIENT_ID,
            managed_identity_client_id=IDENTITY_CLIENT_ID,
        ),
        transport=transport,
        clock=lambda: NOW,
    )


def test_azure_maps_adds_only_sourced_review_candidates_with_provenance() -> None:
    transport = _Transport(
        [
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "id": "azure-feature-die",
                        "geometry": {"type": "Point", "coordinates": [5.3703, 44.7531]},
                        "bbox": [5.35, 44.73, 5.39, 44.78],
                        "properties": {
                            "type": "PopulatedPlace",
                            "confidence": "High",
                            "matchCodes": ["Good"],
                        },
                    }
                ],
            }
        ]
    )

    run = _provider(transport).enrich(_evidence(), (_query(),))

    assert run.evidence.candidate_area is None
    assert run.evidence.candidate_clusters == ()
    assert run.evidence.needs_human_review is True
    assert len(run.evidence.location_candidates) == 1
    candidate = run.evidence.location_candidates[0]
    assert (candidate.longitude, candidate.latitude) == (5.3703, 44.7531)
    assert candidate.score == pytest.approx(0.6)
    assert candidate.raw_score == pytest.approx(0.75)
    assert candidate.source_id == "SOURCE-OFFICIAL-1"
    assert candidate.evidence_kind == "research_prior"
    assert candidate.provider_id == "azure-maps-geocoding"
    assert candidate.radius_m > 2_000
    assert run.provider_run.cost_usd is None
    assert run.provider_run.input_hash
    assert len(run.response_hashes) == 1
    assert transport.calls == [
        (
            "https://atlas.microsoft.com/geocode",
            {
                "api-version": "2026-01-01",
                "query": "Die, Drome, France",
                "top": 3,
                "view": "FR",
            },
            {"x-ms-client-id": MAPS_CLIENT_ID},
        )
    ]


def test_azure_maps_caps_ambiguous_results_and_drops_malformed_features() -> None:
    transport = _Transport(
        [
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "id": "ambiguous-die",
                        "geometry": {"type": "Point", "coordinates": [5.37, 44.75]},
                        "properties": {
                            "type": "PopulatedPlace",
                            "confidence": "High",
                            "matchCodes": ["Ambiguous"],
                        },
                    },
                    {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": []}},
                ],
            }
        ]
    )

    evidence = _provider(transport).enrich(_evidence(), (_query(),)).evidence

    assert len(evidence.location_candidates) == 1
    assert evidence.location_candidates[0].score == pytest.approx(0.32)
    assert {item.code for item in evidence.uncertainties} == {
        "azure_maps_ambiguous_result",
        "azure_maps_results_dropped",
    }


def test_azure_maps_abstains_on_provider_error() -> None:
    transport = _Transport([AzureMapsError("fixture failure")])

    evidence = _provider(transport).enrich(_evidence(), (_query(),)).evidence

    assert evidence.location_candidates == ()
    assert [item.code for item in evidence.uncertainties] == ["azure_maps_provider_error"]
    assert evidence.needs_human_review is True


def test_azure_maps_rejects_a_query_without_matching_claim_provenance() -> None:
    query = _query().model_copy(update={"claim_id": "CLAIM-UNKNOWN"})

    with pytest.raises(ValueError, match="claim from the same source"):
        _provider(_Transport([])).enrich(_evidence(), (query,))
