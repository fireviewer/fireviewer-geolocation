from __future__ import annotations

import json
import os
from hashlib import sha256
from hmac import compare_digest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol

from pydantic import Field, SecretStr, model_validator

from fireviewer_contracts.contracts import SafeIdentifierV2, StrictModel
from fireviewer_geolocation.mvp.localization.azure_maps import (
    AzureMapsConfig,
    AzureMapsGeoEnrichmentProvider,
    AzureMapsLocationQuery,
)
from fireviewer_geolocation.mvp.localization.durable_terrain import AzureBackendTerrainResolver
from fireviewer_geolocation.mvp.localization.geographic_endpoint import (
    DurableGeographicHypothesisService,
)
from fireviewer_contracts.mvp.supervision.backend_event_evidence import (
    AzureBackendEventEvidenceAdapter,
    AzureBackendEventEvidenceConfig,
    BackendGeographicEvidencePublisher,
    BackendGeographicEvidenceReceipt,
    DurableEventEvidence,
    EventEvidenceRepository,
)

_MAX_REQUEST_BYTES = 4 * 1_024
_LOCATION_CLAIM_TYPES = frozenset(
    {
        "incident_location",
        "location_name",
        "place_name",
        "reported_location",
    }
)


class GeographicCpuRequest(StrictModel):
    candidate_id: SafeIdentifierV2


class GeographicRunner(Protocol):
    def run_candidate(self, candidate_id: str) -> dict[str, Any]: ...


class GeographicEvidencePublisher(Protocol):
    def publish(
        self,
        *,
        candidate_id: str,
        payload: dict[str, Any],
    ) -> BackendGeographicEvidenceReceipt: ...


class GeographicCpuSettings(StrictModel):
    host: str = Field(default="0.0.0.0", min_length=1, max_length=255)  # noqa: S104
    port: int = Field(default=8080, ge=1, le=65_535)
    worker_token: SecretStr = Field(min_length=32, max_length=4_096)
    backend: AzureBackendEventEvidenceConfig
    azure_maps_enabled: bool = False
    azure_maps_account_client_id: str | None = Field(default=None, min_length=36, max_length=36)
    managed_identity_client_id: str | None = Field(default=None, min_length=36, max_length=36)

    @model_validator(mode="after")
    def require_azure_maps_identity_when_enabled(self) -> GeographicCpuSettings:
        if self.azure_maps_enabled and (
            self.azure_maps_account_client_id is None
            or self.managed_identity_client_id is None
        ):
            raise ValueError("Azure Maps identity is required only when Azure Maps is enabled")
        return self

    @classmethod
    def from_env(cls) -> GeographicCpuSettings:
        azure_maps_enabled = _env_bool("FIREVIEWER_AZURE_MAPS_ENABLED", False)
        return cls(
            host=os.getenv("FIREVIEWER_GEO_WORKER_HOST", "0.0.0.0"),  # noqa: S104
            port=int(os.getenv("PORT", "8080")),
            worker_token=SecretStr(os.environ["FIREVIEWER_GEO_WORKER_TOKEN"]),
            backend=AzureBackendEventEvidenceConfig(
                base_url=os.environ["FIREVIEWER_BACKEND_BASE_URL"],
                bearer_token=SecretStr(os.environ["FIREVIEWER_BACKEND_TOKEN"]),
                timeout_seconds=float(os.getenv("FIREVIEWER_BACKEND_TIMEOUT_SECONDS", "30")),
            ),
            azure_maps_enabled=azure_maps_enabled,
            azure_maps_account_client_id=(
                os.getenv("FIREVIEWER_AZURE_MAPS_ACCOUNT_CLIENT_ID")
                if azure_maps_enabled
                else None
            ),
            managed_identity_client_id=(
                os.getenv("AZURE_CLIENT_ID") if azure_maps_enabled else None
            ),
        )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _stable_query_id(candidate_id: str, claim_id: str) -> str:
    digest = sha256(f"{candidate_id}\x1f{claim_id}".encode()).hexdigest()[:24]
    return f"AZMAP-QUERY-{digest}"


def plan_azure_maps_queries(durable: DurableEventEvidence) -> tuple[AzureMapsLocationQuery, ...]:
    upload = durable.upload_locations[0] if durable.upload_locations else None
    bias = (upload.longitude, upload.latitude) if upload is not None else None
    planned: list[AzureMapsLocationQuery] = []
    for claim in durable.event.claims:
        if claim.claim_type not in _LOCATION_CLAIM_TYPES:
            continue
        query = claim.text.strip()
        if len(query) < 2:
            continue
        planned.append(
            AzureMapsLocationQuery(
                query_id=_stable_query_id(durable.event.event_id, claim.claim_id),
                source_id=claim.source_id,
                claim_id=claim.claim_id,
                query=query,
                bias_coordinates=bias,
            )
        )
    return tuple(planned[:20])


class GeographicCpuRunner:
    def __init__(
        self,
        *,
        repository: EventEvidenceRepository,
        geographic_service: DurableGeographicHypothesisService,
        azure_maps: AzureMapsGeoEnrichmentProvider | None,
        publisher: GeographicEvidencePublisher,
    ) -> None:
        self._repository = repository
        self._geographic_service = geographic_service
        self._azure_maps = azure_maps
        self._publisher = publisher

    def run_candidate(self, candidate_id: str) -> dict[str, Any]:
        durable = self._repository.read(candidate_id)
        queries = plan_azure_maps_queries(durable)
        maps_payload: dict[str, Any]
        if self._azure_maps is None:
            maps_payload = {"status": "disabled", "query_count": 0}
        elif not queries:
            maps_payload = {
                "status": "not_applicable",
                "query_count": 0,
                "reason_codes": ["no_sourced_location_claim"],
            }
        else:
            enrichment = self._azure_maps.enrich(durable.event, queries)
            maps_payload = {
                "status": "completed",
                "query_count": len(queries),
                "accepted_candidate_ids": list(enrichment.accepted_candidate_ids),
                "response_hashes": list(enrichment.response_hashes),
                "provider_run": enrichment.provider_run.model_dump(mode="json"),
                "location_candidates": [
                    item.model_dump(mode="json")
                    for item in enrichment.evidence.location_candidates
                    if item.candidate_id in enrichment.accepted_candidate_ids
                ],
                "uncertainties": [
                    item.model_dump(mode="json")
                    for item in enrichment.evidence.uncertainties
                    if item.code.startswith("azure_maps_")
                ],
            }
        geographic = self._geographic_service.locate_payload({"event_id": candidate_id})
        persistence = self._publisher.publish(
            candidate_id=candidate_id,
            payload=geographic,
        )
        return {
            "schema": "fireviewer.geographic-worker-response.v1",
            "candidate_id": candidate_id,
            "source_event_evidence_sha256": durable.source_revision_sha256,
            "azure_maps": maps_payload,
            "geographic_hypotheses": geographic,
            "persistence": persistence.model_dump(mode="json"),
            "coordinates_generated_by_visual_model": False,
            "map_mutation_allowed": False,
            "perimeter_mutation_allowed": False,
        }


class GeographicCpuService:
    def __init__(
        self,
        *,
        settings: GeographicCpuSettings,
        runner: GeographicRunner | None = None,
    ) -> None:
        self.settings = settings
        if runner is None:
            repository = AzureBackendEventEvidenceAdapter(settings.backend)
            azure_maps = None
            if settings.azure_maps_enabled:
                account_client_id = settings.azure_maps_account_client_id
                managed_identity_client_id = settings.managed_identity_client_id
                if account_client_id is None or managed_identity_client_id is None:
                    raise ValueError("Azure Maps identity is required when enabled")
                azure_maps = AzureMapsGeoEnrichmentProvider(
                    config=AzureMapsConfig(
                        account_client_id=account_client_id,
                        managed_identity_client_id=managed_identity_client_id,
                    )
                )
            runner = GeographicCpuRunner(
                repository=repository,
                geographic_service=DurableGeographicHypothesisService(
                    repository,
                    terrain_resolver=AzureBackendTerrainResolver(settings.backend),
                ),
                azure_maps=azure_maps,
                publisher=BackendGeographicEvidencePublisher(settings.backend),
            )
        self.runner = runner

    def authorize(self, header: str | None) -> bool:
        expected = f"Bearer {self.settings.worker_token.get_secret_value()}"
        return header is not None and compare_digest(header, expected)

    def run(self, payload: object) -> dict[str, Any]:
        request = GeographicCpuRequest.model_validate(payload)
        return self.runner.run_candidate(request.candidate_id)


def _handler_for(service: GeographicCpuService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "FireViewerGeographicCpu/1.0"

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/healthz":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            self._json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "runtime": "azure-cpu",
                    "azure_maps_enabled": service.settings.azure_maps_enabled,
                    "terrain_mode": "durable_event_reference",
                    "missing_orientation_policy": "abstain",
                    "missing_terrain_policy": "abstain",
                    "coordinates_generated_by_visual_model": False,
                    "map_mutation_allowed": False,
                    "perimeter_mutation_allowed": False,
                    "gpu": False,
                },
            )

        def do_POST(self) -> None:
            if self.path != "/v1/event-evidence/geographic-hypotheses":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            if not service.authorize(self.headers.get("Authorization")):
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= _MAX_REQUEST_BYTES:
                    raise ValueError("request_size_invalid")
                payload = json.loads(self.rfile.read(length))
                result = service.run(payload)
            except Exception as exc:
                self._json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": type(exc).__name__, "detail": str(exc)[:1_000]},
                )
                return
            self._json(HTTPStatus.OK, result)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


def create_geographic_cpu_server(service: GeographicCpuService) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(
        (service.settings.host, service.settings.port),
        _handler_for(service),
    )


def main() -> None:
    service = GeographicCpuService(settings=GeographicCpuSettings.from_env())
    server = create_geographic_cpu_server(service)
    print(f"FireViewer geographic CPU worker ready on port {service.settings.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()


__all__ = [
    "GeographicCpuRequest",
    "GeographicCpuRunner",
    "GeographicCpuService",
    "GeographicCpuSettings",
    "create_geographic_cpu_server",
    "main",
    "plan_azure_maps_queries",
]
