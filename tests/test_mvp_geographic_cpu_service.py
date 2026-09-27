from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from fireviewer_contracts.mvp.contracts import EventEvidenceV1
from fireviewer_geolocation.mvp.localization.geographic_cpu_service import (
    GeographicCpuRunner,
    GeographicCpuService,
    GeographicCpuSettings,
    plan_azure_maps_queries,
)
from fireviewer_contracts.mvp.supervision.backend_event_evidence import (
    AzureBackendEventEvidenceConfig,
    BackendGeographicEvidenceReceipt,
    DurableEventEvidence,
)


def _durable(*, claim_type: str = "contributor_observation") -> DurableEventEvidence:
    event = EventEvidenceV1.model_validate(
        {
            "schema": "fireviewer.event-evidence.v1",
            "event_id": "EC-REAL-1",
            "sources": [
                {
                    "source_id": "SOURCE-1",
                    "origin_id": "ORIGIN-1",
                    "publisher": "SDIS",
                    "retrieved_at": "2026-08-23T10:00:00Z",
                    "source_type": "official",
                    "independence_weight": 1,
                }
            ],
            "claims": [
                {
                    "claim_id": "CLAIM-1",
                    "source_id": "SOURCE-1",
                    "claim_type": claim_type,
                    "text": "Die, Drome, France",
                    "confidence": 0.9,
                }
            ],
        }
    )
    return DurableEventEvidence(
        event=event,
        media_locations=(),
        vision_artifacts=(),
        upload_locations=(),
        prior_fire_states=(),
        geospatial_checks=(),
        geographic_references=(),
        source_revision_sha256="a" * 64,
    )


def _settings() -> GeographicCpuSettings:
    return GeographicCpuSettings(
        worker_token=SecretStr("w" * 32),
        backend=AzureBackendEventEvidenceConfig(
            base_url="https://api.example.test",
            bearer_token=SecretStr("b" * 32),
        ),
    )


def test_settings_start_without_azure_maps_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREVIEWER_GEO_WORKER_TOKEN", "w" * 32)
    monkeypatch.setenv("FIREVIEWER_BACKEND_BASE_URL", "https://api.example.test")
    monkeypatch.setenv("FIREVIEWER_BACKEND_TOKEN", "b" * 32)
    monkeypatch.delenv("FIREVIEWER_AZURE_MAPS_ENABLED", raising=False)
    monkeypatch.delenv("FIREVIEWER_AZURE_MAPS_ACCOUNT_CLIENT_ID", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)

    settings = GeographicCpuSettings.from_env()
    assert settings.azure_maps_enabled is False
    assert settings.azure_maps_account_client_id is None
    assert settings.managed_identity_client_id is None
    GeographicCpuService(settings=settings)


def test_enabling_azure_maps_requires_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREVIEWER_GEO_WORKER_TOKEN", "w" * 32)
    monkeypatch.setenv("FIREVIEWER_BACKEND_BASE_URL", "https://api.example.test")
    monkeypatch.setenv("FIREVIEWER_BACKEND_TOKEN", "b" * 32)
    monkeypatch.setenv("FIREVIEWER_AZURE_MAPS_ENABLED", "true")
    monkeypatch.delenv("FIREVIEWER_AZURE_MAPS_ACCOUNT_CLIENT_ID", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)

    with pytest.raises(ValidationError, match="Azure Maps identity is required"):
        GeographicCpuSettings.from_env()


def test_planner_accepts_only_explicitly_sourced_location_claims() -> None:
    assert plan_azure_maps_queries(_durable()) == ()
    planned = plan_azure_maps_queries(_durable(claim_type="incident_location"))
    assert len(planned) == 1
    assert planned[0].query == "Die, Drome, France"
    assert planned[0].claim_id == "CLAIM-1"


def test_runner_returns_abstention_without_mutation() -> None:
    durable = _durable()

    class Repository:
        def read(self, event_id: str) -> DurableEventEvidence:
            assert event_id == "EC-REAL-1"
            return durable

    class Geographic:
        def locate_payload(self, payload: dict[str, Any]) -> dict[str, object]:
            assert payload == {"event_id": "EC-REAL-1"}
            return {
                "schema": "fireviewer.geographic-hypotheses.v1",
                "status": "abstained",
                "hypotheses": [],
                "abstentions": [
                    {
                        "reason_codes": [
                            "missing_camera_orientation",
                            "missing_terrain_reference",
                        ]
                    }
                ],
            }

    class Publisher:
        payload: dict[str, Any] | None = None

        def publish(
            self,
            *,
            candidate_id: str,
            payload: dict[str, Any],
        ) -> BackendGeographicEvidenceReceipt:
            assert candidate_id == "EC-REAL-1"
            self.payload = payload
            return BackendGeographicEvidenceReceipt(
                candidate_id=candidate_id,
                source_revision_sha256="b" * 64,
                request_sha256="c" * 64,
                hypothesis_count=0,
                abstention_count=1,
                replayed=False,
            )

    publisher = Publisher()

    result = GeographicCpuRunner(
        repository=Repository(),
        geographic_service=Geographic(),  # type: ignore[arg-type]
        azure_maps=None,
        publisher=publisher,
    ).run_candidate("EC-REAL-1")

    assert result["source_event_evidence_sha256"] == "a" * 64
    hypotheses = result["geographic_hypotheses"]
    assert isinstance(hypotheses, dict)
    assert hypotheses["status"] == "abstained"
    assert publisher.payload is hypotheses
    assert result["persistence"]["abstention_count"] == 1
    assert result["coordinates_generated_by_visual_model"] is False
    assert result["map_mutation_allowed"] is False
    assert result["perimeter_mutation_allowed"] is False


def test_service_auth_is_constant_time_contract() -> None:
    class Runner:
        def run_candidate(self, candidate_id: str) -> dict[str, Any]:
            return {"candidate_id": candidate_id}

    service = GeographicCpuService(settings=_settings(), runner=Runner())
    assert service.authorize("Bearer " + "w" * 32)
    assert not service.authorize("Bearer invalid")
    assert service.run({"candidate_id": "EC-REAL-1"}) == {"candidate_id": "EC-REAL-1"}
