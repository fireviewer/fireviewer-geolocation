from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from typing import Any, Protocol

from pydantic import ValidationError

from fireviewer_contracts.mvp.contracts import GeographicHypothesisResultV1
from fireviewer_geolocation.mvp.localization.durable_terrain import DurableTerrainError
from fireviewer_geolocation.mvp.localization.geographic_hypotheses import (
    GeographicHypothesisConfig,
    GeographicHypothesisEngine,
    TerrainElevationProvider,
)
from fireviewer_contracts.mvp.supervision.backend_event_evidence import (
    BackendEventEvidenceError,
    BackendEventEvidenceNotFoundError,
    DurableEventEvidence,
    DurableTerrainReference,
    EventEvidenceRepository,
)

_MAX_REQUEST_BYTES = 64 * 1024


class TerrainResolver(Protocol):
    def resolve(
        self,
        reference: DurableTerrainReference,
    ) -> TerrainElevationProvider: ...


class DurableGeographicHypothesisService:
    """Run the geographic stage independently from visual detection and publication."""

    def __init__(
        self,
        repository: EventEvidenceRepository,
        engine: GeographicHypothesisEngine | None = None,
        *,
        terrain_resolver: TerrainResolver | None = None,
        engine_config: GeographicHypothesisConfig | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if (engine is None) == (terrain_resolver is None):
            raise ValueError("provide either a fixed engine or a durable terrain resolver")
        self._repository = repository
        self._engine = engine
        self._terrain_resolver = terrain_resolver
        self._engine_config = engine_config
        self._clock = clock or (lambda: datetime.now(UTC))

    def locate(
        self,
        event_id: str,
    ) -> tuple[DurableEventEvidence, GeographicHypothesisResultV1]:
        if not event_id or len(event_id) > 128:
            raise ValueError("event_id is invalid")
        durable = self._repository.read(event_id)
        engine = self._engine
        if engine is None:
            assert self._terrain_resolver is not None
            terrain = (
                None
                if durable.terrain_reference is None
                else self._terrain_resolver.resolve(durable.terrain_reference)
            )
            engine = GeographicHypothesisEngine(terrain, config=self._engine_config)
        result = engine.locate(
            durable.event,
            vision_artifacts=durable.vision_artifacts,
            upload_locations=durable.upload_locations,
            geographic_references=durable.geographic_references,
            source_revision_sha256=durable.source_revision_sha256,
            generated_at=self._clock(),
        )
        return durable, result

    def locate_payload(self, payload: dict[str, Any]) -> dict[str, object]:
        event_id = payload.get("event_id")
        if not isinstance(event_id, str):
            raise ValueError("event_id is invalid")
        _, result = self.locate(event_id)
        return result.model_dump(mode="json", by_alias=True)


def _handler_for(
    service: DurableGeographicHypothesisService,
) -> type[BaseHTTPRequestHandler]:
    class GeographicHypothesisHandler(BaseHTTPRequestHandler):
        server_version = "FireViewerGeographicHypotheses/1.0"

        def log_message(self, _format: str, *_args: object) -> None:
            return None

        def _write_json(self, status: HTTPStatus, payload: object) -> None:
            body = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            self.send_response(status.value)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            raw_length = self.headers.get("content-length")
            if raw_length is None:
                raise ValueError("content-length is required")
            length = int(raw_length)
            if length <= 0 or length > _MAX_REQUEST_BYTES:
                raise ValueError("request body size is invalid")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def do_GET(self) -> None:
            if self.path != "/health":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "route_not_found"})
                return
            self._write_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "stage": "geographic_hypotheses",
                    "visual_coordinates_allowed": False,
                    "map_mutation_allowed": False,
                    "perimeter_mutation_allowed": False,
                    "terrain_mode": (
                        "fixed_test_provider"
                        if service._engine is not None
                        else "durable_event_reference"
                    ),
                    "gpu": False,
                    "cost_usd": 0,
                },
            )

        def do_POST(self) -> None:
            if self.path != "/v1/events/geographic-hypotheses":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "route_not_found"})
                return
            try:
                response = service.locate_payload(self._read_json())
            except BackendEventEvidenceNotFoundError as exc:
                self._write_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": "event_not_found", "event_id": str(exc.args[0])},
                )
            except BackendEventEvidenceError as exc:
                self._write_json(
                    HTTPStatus.BAD_GATEWAY,
                    {"error": "backend_event_evidence_unavailable", "detail": str(exc)},
                )
            except DurableTerrainError as exc:
                self._write_json(
                    HTTPStatus.BAD_GATEWAY,
                    {"error": "durable_terrain_unavailable", "detail": str(exc)},
                )
            except (TypeError, ValueError, ValidationError, json.JSONDecodeError) as exc:
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "invalid_request", "detail": str(exc)},
                )
            else:
                self._write_json(HTTPStatus.OK, response)

    return GeographicHypothesisHandler


def create_geographic_hypothesis_server(
    repository: EventEvidenceRepository,
    engine: GeographicHypothesisEngine | None = None,
    *,
    terrain_resolver: TerrainResolver | None = None,
    engine_config: GeographicHypothesisConfig | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
    clock: Callable[[], datetime] | None = None,
) -> ThreadingHTTPServer:
    try:
        loopback = ip_address(host).is_loopback
    except ValueError as exc:
        raise ValueError("the geographic hypothesis host must be a loopback IP") from exc
    if not loopback:
        raise ValueError("the geographic hypothesis service must only listen on loopback")
    if not 0 <= port <= 65_535:
        raise ValueError("port must be between 0 and 65535")
    service = DurableGeographicHypothesisService(
        repository,
        engine,
        terrain_resolver=terrain_resolver,
        engine_config=engine_config,
        clock=clock,
    )
    return ThreadingHTTPServer((host, port), _handler_for(service))


__all__ = [
    "DurableGeographicHypothesisService",
    "TerrainResolver",
    "create_geographic_hypothesis_server",
]
