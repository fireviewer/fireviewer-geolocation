from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from math import isfinite
from time import perf_counter
from typing import Any, Protocol

from pydantic import Field, model_validator

from fireviewer_contracts.contracts import SafeIdentifierV2, StrictModel
from fireviewer_contracts.mvp.contracts import (
    EventEvidenceV1,
    LocationCandidate,
    ProviderRun,
    Uncertainty,
)
from fireviewer_geolocation.mvp.localization.evidence_fusion import haversine_m
from fireviewer_contracts.mvp.providers import ProviderDescriptor, ProviderHealth

AZURE_MAPS_SCOPE = "https://atlas.microsoft.com/.default"
AZURE_MAPS_ENDPOINT = "https://atlas.microsoft.com"
AZURE_MAPS_API_VERSION = "2026-01-01"


class AzureMapsError(RuntimeError):
    """A bounded Azure Maps request failed or returned an unusable payload."""


class AzureMapsTransport(Protocol):
    def get_json(
        self,
        url: str,
        *,
        params: dict[str, str | int],
        headers: dict[str, str],
    ) -> dict[str, Any]: ...


class AzureMapsConfig(StrictModel):
    account_client_id: str = Field(
        pattern=(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
            r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        )
    )
    managed_identity_client_id: str = Field(
        pattern=(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
            r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        )
    )
    endpoint: str = Field(default=AZURE_MAPS_ENDPOINT, pattern=r"^https://atlas\.microsoft\.com$")
    api_version: str = Field(default=AZURE_MAPS_API_VERSION, pattern=r"^2026-01-01$")
    top: int = Field(default=3, ge=1, le=5)
    max_queries: int = Field(default=20, ge=1, le=20)
    timeout_seconds: float = Field(default=10, ge=1, le=30)
    minimum_candidate_score: float = Field(default=0.20, ge=0, le=1)


class AzureMapsLocationQuery(StrictModel):
    query_id: SafeIdentifierV2
    source_id: SafeIdentifierV2
    claim_id: SafeIdentifierV2
    query: str = Field(min_length=2, max_length=500)
    view: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    bias_coordinates: tuple[float, float] | None = None
    bbox: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def validate_geographic_hints(self) -> AzureMapsLocationQuery:
        if self.bias_coordinates is not None:
            longitude, latitude = self.bias_coordinates
            if (
                not all(isfinite(value) for value in self.bias_coordinates)
                or not -180 <= longitude <= 180
                or not -90 <= latitude <= 90
            ):
                raise ValueError("Azure Maps bias must be a WGS84 longitude/latitude pair")
        if self.bbox is not None:
            west, south, east, north = self.bbox
            if (
                not all(isfinite(value) for value in self.bbox)
                or not -180 <= west < east <= 180
                or not -90 <= south < north <= 90
            ):
                raise ValueError("Azure Maps bbox must be ordered WGS84 west,south,east,north")
        return self


class AzureMapsEnrichmentRun(StrictModel):
    evidence: EventEvidenceV1
    provider_run: ProviderRun
    attempted_query_ids: tuple[SafeIdentifierV2, ...]
    response_hashes: tuple[str, ...]
    accepted_candidate_ids: tuple[SafeIdentifierV2, ...]


class AzureIdentityMapsTransport:
    """Authenticate Azure Maps REST calls with one explicit managed identity."""

    def __init__(
        self,
        *,
        managed_identity_client_id: str,
        timeout_seconds: float = 10,
    ) -> None:
        try:
            import httpx
            from azure.identity import ManagedIdentityCredential
        except ImportError as exc:  # pragma: no cover - exercised only in deployed runtime
            raise AzureMapsError(
                "Install the azure-maps optional dependencies to use managed identity"
            ) from exc
        self._httpx = httpx
        self._credential = ManagedIdentityCredential(client_id=managed_identity_client_id)
        self._timeout_seconds = timeout_seconds

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, str | int],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        token = self._credential.get_token(AZURE_MAPS_SCOPE).token
        request_headers = {
            **headers,
            "Authorization": f"Bearer {token}",
            "Accept": "application/geo+json, application/json",
        }
        try:
            response = self._httpx.get(
                url,
                params=params,
                headers=request_headers,
                timeout=self._timeout_seconds,
                follow_redirects=False,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise AzureMapsError("Azure Maps request failed") from exc
        if not isinstance(payload, dict):
            raise AzureMapsError("Azure Maps response must be a JSON object")
        return payload


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}-{sha256(value.encode('utf-8')).hexdigest()[:24]}"


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _feature_radius_m(feature: dict[str, Any], longitude: float, latitude: float) -> float:
    bbox = feature.get("bbox")
    if (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(value, (int, float)) and isfinite(value) for value in bbox)
    ):
        west, south, east, north = (float(value) for value in bbox)
        if -180 <= west <= east <= 180 and -90 <= south <= north <= 90:
            radius = max(
                haversine_m((longitude, latitude), (corner_lon, corner_lat))
                for corner_lon, corner_lat in (
                    (west, south),
                    (west, north),
                    (east, south),
                    (east, north),
                )
            )
            return min(max(radius, 25.0), 1_000_000.0)

    properties = feature.get("properties")
    raw_entity_type = properties.get("type") if isinstance(properties, dict) else None
    entity_type = raw_entity_type if isinstance(raw_entity_type, str) else ""
    return {
        "Address": 50.0,
        "RoadBlock": 250.0,
        "Road": 1_000.0,
        "Neighborhood": 2_000.0,
        "PopulatedPlace": 5_000.0,
        "Postcode1": 10_000.0,
        "AdminDivision2": 50_000.0,
        "AdminDivision1": 100_000.0,
        "CountryRegion": 500_000.0,
    }.get(entity_type, 1_000.0)


def _match_score(properties: dict[str, Any]) -> tuple[float, bool]:
    raw_confidence = properties.get("confidence")
    confidence = raw_confidence if isinstance(raw_confidence, str) else ""
    score = {"High": 0.75, "Medium": 0.55, "Low": 0.25}.get(confidence, 0.20)
    raw_codes = properties.get("matchCodes")
    match_codes = set(raw_codes) if isinstance(raw_codes, list) else set()
    ambiguous = "Ambiguous" in match_codes
    if ambiguous:
        score = min(score, 0.40)
    if "UpHierarchy" in match_codes:
        score = min(score, 0.45)
    return score, ambiguous


class AzureMapsGeoEnrichmentProvider:
    """Add sourced coarse geocoding candidates to EventEvidence without publishing geography."""

    def __init__(
        self,
        *,
        config: AzureMapsConfig,
        transport: AzureMapsTransport | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.transport = transport or AzureIdentityMapsTransport(
            managed_identity_client_id=config.managed_identity_client_id,
            timeout_seconds=config.timeout_seconds,
        )
        self.clock = clock or (lambda: datetime.now(UTC))
        self.descriptor = ProviderDescriptor(
            provider_id="azure-maps-geocoding",
            provider_version=config.api_version,
            config={
                "account_client_id": config.account_client_id,
                "endpoint": config.endpoint,
                "api_version": config.api_version,
                "top": config.top,
                "max_queries": config.max_queries,
                "minimum_candidate_score": config.minimum_candidate_score,
                "authentication": "managed-identity",
            },
            capabilities=("event-geocoding", "event-reverse-geocoding-context"),
        )

    def healthcheck(self) -> ProviderHealth:
        if not callable(getattr(self.transport, "get_json", None)):
            return ProviderHealth(
                status="unavailable",
                checked_at=self.clock(),
                reason_codes=("azure_maps_transport_unavailable",),
            )
        return ProviderHealth(status="healthy", checked_at=self.clock())

    def enrich(
        self,
        evidence: EventEvidenceV1,
        queries: tuple[AzureMapsLocationQuery, ...],
    ) -> AzureMapsEnrichmentRun:
        if not queries:
            raise ValueError("Azure Maps enrichment requires at least one sourced query")
        if len(queries) > self.config.max_queries:
            raise ValueError("Azure Maps enrichment query budget exceeded")
        if len({query.query_id for query in queries}) != len(queries):
            raise ValueError("Azure Maps enrichment query identifiers must be unique")

        source_ids = {source.source_id for source in evidence.sources}
        claims = {claim.claim_id: claim for claim in evidence.claims}
        for query in queries:
            claim = claims.get(query.claim_id)
            if query.source_id not in source_ids:
                raise ValueError("Azure Maps query references an unknown source")
            if claim is None or claim.source_id != query.source_id:
                raise ValueError("Azure Maps query must reference a claim from the same source")

        started = perf_counter()
        candidates = {item.candidate_id: item for item in evidence.location_candidates}
        uncertainties = {item.uncertainty_id: item for item in evidence.uncertainties}
        accepted_ids: list[str] = []
        response_hashes: list[str] = []

        for query in queries:
            claim = claims[query.claim_id]
            params: dict[str, str | int] = {
                "api-version": self.config.api_version,
                "query": query.query,
                "top": self.config.top,
            }
            if query.view is not None:
                params["view"] = query.view
            if query.bias_coordinates is not None:
                params["coordinates"] = ",".join(str(value) for value in query.bias_coordinates)
            if query.bbox is not None:
                params["bbox"] = ",".join(str(value) for value in query.bbox)

            try:
                response = self.transport.get_json(
                    f"{self.config.endpoint}/geocode",
                    params=params,
                    headers={"x-ms-client-id": self.config.account_client_id},
                )
            except AzureMapsError:
                self._add_uncertainty(
                    uncertainties,
                    query=query,
                    code="azure_maps_provider_error",
                    description="Azure Maps could not enrich this sourced location claim.",
                )
                continue

            response_hashes.append(_canonical_hash(response))
            raw_features = response.get("features")
            if response.get("type") != "FeatureCollection" or not isinstance(raw_features, list):
                self._add_uncertainty(
                    uncertainties,
                    query=query,
                    code="azure_maps_invalid_response",
                    description="Azure Maps returned an invalid GeoJSON feature collection.",
                )
                continue
            if not raw_features:
                self._add_uncertainty(
                    uncertainties,
                    query=query,
                    code="azure_maps_no_result",
                    description="Azure Maps returned no candidate for this sourced location claim.",
                )
                continue

            dropped = 0
            ambiguous = False
            accepted_for_query = 0
            for rank, raw_feature in enumerate(raw_features[: self.config.top], start=1):
                parsed = self._candidate_from_feature(
                    raw_feature,
                    query=query,
                    claim_confidence=claim.confidence,
                    rank=rank,
                )
                if parsed is None:
                    dropped += 1
                    continue
                candidate, candidate_ambiguous = parsed
                ambiguous = ambiguous or candidate_ambiguous
                if candidate.score < self.config.minimum_candidate_score:
                    dropped += 1
                    continue
                existing = candidates.get(candidate.candidate_id)
                if existing is not None and existing != candidate:
                    raise ValueError(f"conflicting Azure Maps candidate {candidate.candidate_id}")
                candidates[candidate.candidate_id] = candidate
                accepted_ids.append(candidate.candidate_id)
                accepted_for_query += 1

            if ambiguous or accepted_for_query > 1:
                self._add_uncertainty(
                    uncertainties,
                    query=query,
                    code="azure_maps_ambiguous_result",
                    description=(
                        "Azure Maps returned multiple or explicitly ambiguous candidates; "
                        "human review is required."
                    ),
                )
            if dropped:
                self._add_uncertainty(
                    uncertainties,
                    query=query,
                    code="azure_maps_results_dropped",
                    description=(
                        f"{dropped} malformed or low-confidence Azure Maps candidates were dropped."
                    ),
                )
            if accepted_for_query == 0:
                self._add_uncertainty(
                    uncertainties,
                    query=query,
                    code="azure_maps_no_accepted_candidate",
                    description=(
                        "Azure Maps produced no candidate above the configured quality floor."
                    ),
                )

        runtime_ms = max(0, round((perf_counter() - started) * 1_000))
        input_hash = _canonical_hash(
            {
                "event_id": evidence.event_id,
                "queries": [query.model_dump(mode="json") for query in queries],
                "provider": self.descriptor.model_dump(mode="json"),
            }
        )
        updated = EventEvidenceV1.model_validate(
            evidence.model_copy(
                update={
                    "location_candidates": tuple(
                        candidates[item_id] for item_id in sorted(candidates)
                    ),
                    "uncertainties": tuple(
                        uncertainties[item_id] for item_id in sorted(uncertainties)
                    ),
                    "needs_human_review": True,
                }
            )
        )
        return AzureMapsEnrichmentRun(
            evidence=updated,
            provider_run=ProviderRun(
                provider_id=self.descriptor.provider_id,
                provider_version=self.descriptor.provider_version,
                config=self.descriptor.config,
                input_hash=input_hash,
                runtime_ms=runtime_ms,
                cost_usd=None,
                generated_at=self.clock(),
            ),
            attempted_query_ids=tuple(query.query_id for query in queries),
            response_hashes=tuple(response_hashes),
            accepted_candidate_ids=tuple(dict.fromkeys(accepted_ids)),
        )

    def _candidate_from_feature(
        self,
        raw_feature: object,
        *,
        query: AzureMapsLocationQuery,
        claim_confidence: float,
        rank: int,
    ) -> tuple[LocationCandidate, bool] | None:
        if not isinstance(raw_feature, dict) or raw_feature.get("type") != "Feature":
            return None
        geometry = raw_feature.get("geometry")
        properties = raw_feature.get("properties")
        if (
            not isinstance(geometry, dict)
            or geometry.get("type") != "Point"
            or not isinstance(properties, dict)
        ):
            return None
        coordinates = geometry.get("coordinates")
        if (
            not isinstance(coordinates, list)
            or len(coordinates) < 2
            or not all(isinstance(value, (int, float)) for value in coordinates[:2])
        ):
            return None
        longitude, latitude = (float(value) for value in coordinates[:2])
        if (
            not isfinite(longitude)
            or not isfinite(latitude)
            or not -180 <= longitude <= 180
            or not -90 <= latitude <= 90
        ):
            return None

        match_score, ambiguous = _match_score(properties)
        raw_reference = raw_feature.get("id")
        if not isinstance(raw_reference, str) or not raw_reference.strip():
            raw_reference = _canonical_hash(raw_feature)
        reference_id = _stable_id("AZMAP", raw_reference)
        candidate_id = _stable_id(
            "CAND-AZMAP",
            f"{query.query_id}:{reference_id}:{self.config.api_version}",
        )
        return (
            LocationCandidate(
                candidate_id=candidate_id,
                longitude=longitude,
                latitude=latitude,
                radius_m=_feature_radius_m(raw_feature, longitude, latitude),
                score=match_score * claim_confidence,
                raw_score=match_score,
                rank=rank,
                evidence_kind="research_prior",
                provider_id=self.descriptor.provider_id,
                provider_version=self.descriptor.provider_version,
                source_id=query.source_id,
                reference_id=reference_id,
            ),
            ambiguous,
        )

    @staticmethod
    def _add_uncertainty(
        uncertainties: dict[str, Uncertainty],
        *,
        query: AzureMapsLocationQuery,
        code: str,
        description: str,
    ) -> None:
        uncertainty = Uncertainty(
            uncertainty_id=_stable_id("UNC-AZMAP", f"{query.query_id}:{code}"),
            code=code,
            scope_type="source",
            scope_id=query.source_id,
            description=description,
        )
        existing = uncertainties.get(uncertainty.uncertainty_id)
        if existing is not None and existing != uncertainty:
            raise ValueError(f"conflicting Azure Maps uncertainty {uncertainty.uncertainty_id}")
        uncertainties[uncertainty.uncertainty_id] = uncertainty


__all__ = [
    "AZURE_MAPS_API_VERSION",
    "AZURE_MAPS_ENDPOINT",
    "AZURE_MAPS_SCOPE",
    "AzureIdentityMapsTransport",
    "AzureMapsConfig",
    "AzureMapsEnrichmentRun",
    "AzureMapsError",
    "AzureMapsGeoEnrichmentProvider",
    "AzureMapsLocationQuery",
    "AzureMapsTransport",
]
