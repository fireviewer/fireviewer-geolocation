from __future__ import annotations

import json
from datetime import datetime
from hashlib import sha256
from typing import Any, Protocol
from urllib.parse import urljoin, urlsplit, urlunsplit

from pydantic import AnyHttpUrl, Field, model_validator

from fireviewer_contracts.contracts import SafeIdentifierV2, Sha256HexV2, StrictModel
from fireviewer_contracts.mvp.contracts.common import is_timezone_aware, validate_lon_lat


class PanoramaxError(RuntimeError):
    """Raised when a Panoramax response cannot be used safely."""


class PanoramaxQuery(StrictModel):
    zone_id: SafeIdentifierV2
    bbox_wgs84: tuple[float, float, float, float]
    limit: int = Field(default=1_000, ge=1, le=10_000)
    captured_after: datetime | None = None
    captured_before: datetime | None = None

    @model_validator(mode="after")
    def validate_query(self) -> PanoramaxQuery:
        min_lon, min_lat, max_lon, max_lat = self.bbox_wgs84
        validate_lon_lat((min_lon, min_lat), label="Panoramax minimum bbox corner")
        validate_lon_lat((max_lon, max_lat), label="Panoramax maximum bbox corner")
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError("Panoramax bbox must be ordered")
        if max_lon - min_lon > 2 or max_lat - min_lat > 2:
            raise ValueError("Panoramax MVP queries must remain regional")
        times = tuple(
            value for value in (self.captured_after, self.captured_before) if value is not None
        )
        if any(not is_timezone_aware(value) for value in times):
            raise ValueError("Panoramax query times must include a timezone")
        if (
            self.captured_after is not None
            and self.captured_before is not None
            and self.captured_before < self.captured_after
        ):
            raise ValueError("Panoramax capture end must not precede its start")
        return self


class PanoramaxImage(StrictModel):
    image_id: SafeIdentifierV2
    sequence_id: SafeIdentifierV2
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    altitude_m: float | None = Field(default=None, allow_inf_nan=False)
    heading_deg: float | None = Field(default=None, ge=0, lt=360)
    pitch_deg: float | None = Field(default=None, ge=-90, le=90)
    roll_deg: float | None = Field(default=None, ge=-180, le=180)
    yaw_deg: float | None = Field(default=None, ge=0, lt=360)
    field_of_view_deg: float | None = Field(default=None, gt=0, le=360)
    focal_length_mm: float | None = Field(default=None, gt=0)
    gps_accuracy_m: float | None = Field(default=None, gt=0, le=100_000)
    captured_at: datetime
    image_url: AnyHttpUrl | None = None
    item_url: AnyHttpUrl | None = None
    license_url: AnyHttpUrl | None = None
    item_sha256: Sha256HexV2

    @model_validator(mode="after")
    def validate_capture(self) -> PanoramaxImage:
        if not is_timezone_aware(self.captured_at):
            raise ValueError("Panoramax capture time must include a timezone")
        return self


class PanoramaxSearchResult(StrictModel):
    zone_id: SafeIdentifierV2
    api_url: AnyHttpUrl
    bbox_wgs84: tuple[float, float, float, float]
    query_sha256: Sha256HexV2
    retrieved_at: datetime
    images: tuple[PanoramaxImage, ...] = Field(max_length=10_000)

    @model_validator(mode="after")
    def validate_result(self) -> PanoramaxSearchResult:
        if not is_timezone_aware(self.retrieved_at):
            raise ValueError("Panoramax retrieval time must include a timezone")
        image_ids = [item.image_id for item in self.images]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("Panoramax search result contains duplicate image identifiers")
        return self


class JsonTransport(Protocol):
    def get_json(
        self,
        url: str,
        *,
        params: dict[str, str | int] | None = None,
    ) -> dict[str, Any]: ...


class HttpxJsonTransport:
    def __init__(self, *, timeout_seconds: float = 30) -> None:
        self.timeout_seconds = timeout_seconds

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, str | int] | None = None,
    ) -> dict[str, Any]:
        import httpx

        response = httpx.get(
            url,
            params=params,
            timeout=self.timeout_seconds,
            follow_redirects=False,
            headers={"Accept": "application/geo+json, application/json"},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise PanoramaxError("Panoramax returned a non-object JSON payload")
        return payload


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(payload.encode()).hexdigest()


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _link(items: object, relation: str) -> str | None:
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict) or item.get("rel") != relation:
            continue
        href = item.get("href")
        if isinstance(href, str):
            return href
    return None


def _image_asset(feature: dict[str, Any]) -> str | None:
    assets = feature.get("assets")
    if not isinstance(assets, dict):
        return _link(feature.get("links"), "item-preview")
    prioritized: list[tuple[int, str]] = []
    for key, raw_asset in assets.items():
        if not isinstance(raw_asset, dict):
            continue
        href = raw_asset.get("href")
        if not isinstance(href, str):
            continue
        roles = raw_asset.get("roles")
        role_values = roles if isinstance(roles, list) else []
        priority = 3
        if key in {"hd", "visual"}:
            priority = 0
        elif key in {"sd", "thumbnail"}:
            priority = 1
        elif "data" in role_values:
            priority = 2
        prioritized.append((priority, href))
    return min(prioritized, default=(99, ""))[1] or _link(feature.get("links"), "item-preview")


def parse_panoramax_feature(feature: dict[str, Any]) -> PanoramaxImage:
    if feature.get("type") != "Feature":
        raise PanoramaxError("Panoramax item is not a GeoJSON Feature")
    image_id = feature.get("id")
    sequence_id = feature.get("collection")
    geometry = feature.get("geometry")
    properties = feature.get("properties")
    if not isinstance(image_id, str) or not isinstance(sequence_id, str):
        raise PanoramaxError("Panoramax item identity is incomplete")
    if not isinstance(geometry, dict) or geometry.get("type") != "Point":
        raise PanoramaxError("Panoramax item requires Point geometry")
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list | tuple) or len(coordinates) < 2:
        raise PanoramaxError("Panoramax item coordinates are incomplete")
    if not isinstance(properties, dict):
        raise PanoramaxError("Panoramax item properties are missing")
    captured_raw = properties.get("datetime") or properties.get("datetimetz")
    if not isinstance(captured_raw, str):
        raise PanoramaxError("Panoramax item capture time is missing")
    try:
        captured_at = datetime.fromisoformat(captured_raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PanoramaxError("Panoramax item capture time is invalid") from exc
    interior = properties.get("pers:interior_orientation")
    interior = interior if isinstance(interior, dict) else {}
    longitude = _optional_float(coordinates[0])
    latitude = _optional_float(coordinates[1])
    if longitude is None or latitude is None:
        raise PanoramaxError("Panoramax item coordinates must be numeric")
    return PanoramaxImage.model_validate(
        {
            "image_id": image_id,
            "sequence_id": sequence_id,
            "longitude": longitude,
            "latitude": latitude,
            "altitude_m": _optional_float(coordinates[2]) if len(coordinates) >= 3 else None,
            "heading_deg": _optional_float(properties.get("view:azimuth")),
            "pitch_deg": _optional_float(properties.get("pers:pitch")),
            "roll_deg": _optional_float(properties.get("pers:roll")),
            "yaw_deg": _optional_float(properties.get("pers:yaw")),
            "field_of_view_deg": _optional_float(interior.get("field_of_view")),
            "focal_length_mm": _optional_float(interior.get("focal_length")),
            "gps_accuracy_m": _optional_float(properties.get("quality:horizontal_accuracy")),
            "captured_at": captured_at,
            "image_url": _image_asset(feature),
            "item_url": _link(feature.get("links"), "self"),
            "license_url": _link(feature.get("links"), "license"),
            "item_sha256": _canonical_digest(feature),
        }
    )


class PanoramaxClient:
    def __init__(
        self,
        *,
        api_url: str,
        transport: JsonTransport | None = None,
        max_pages: int = 100,
    ) -> None:
        parsed = urlsplit(api_url)
        if (
            parsed.scheme != "https"
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Panoramax API URL must be a plain HTTPS origin/path")
        self.api_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))
        self._origin = (parsed.scheme, parsed.hostname.lower(), parsed.port or 443)
        self.transport = transport or HttpxJsonTransport()
        self.max_pages = max_pages

    def _safe_page_url(self, href: str) -> str:
        absolute = urljoin(f"{self.api_url}/", href)
        parsed = urlsplit(absolute)
        origin = (parsed.scheme, (parsed.hostname or "").lower(), parsed.port or 443)
        if origin != self._origin or parsed.username is not None or parsed.password is not None:
            raise PanoramaxError("Panoramax pagination attempted to leave the configured origin")
        return absolute

    def search(self, query: PanoramaxQuery, *, retrieved_at: datetime) -> PanoramaxSearchResult:
        if not is_timezone_aware(retrieved_at):
            raise ValueError("Panoramax retrieved_at must include a timezone")
        params: dict[str, str | int] = {
            "bbox": ",".join(str(value) for value in query.bbox_wgs84),
            "limit": min(query.limit, 1_000),
        }
        if query.captured_after is not None or query.captured_before is not None:
            start = query.captured_after.isoformat() if query.captured_after else ".."
            end = query.captured_before.isoformat() if query.captured_before else ".."
            params["datetime"] = f"{start}/{end}"
        page_url = f"{self.api_url}/search"
        page_params: dict[str, str | int] | None = params
        images: dict[str, PanoramaxImage] = {}
        for _ in range(self.max_pages):
            payload = self.transport.get_json(page_url, params=page_params)
            features = payload.get("features")
            if not isinstance(features, list):
                raise PanoramaxError("Panoramax search response has no feature list")
            for raw_feature in features:
                if not isinstance(raw_feature, dict):
                    raise PanoramaxError("Panoramax search returned a non-object feature")
                image = parse_panoramax_feature(raw_feature)
                images.setdefault(image.image_id, image)
                if len(images) >= query.limit:
                    break
            if len(images) >= query.limit:
                break
            next_href = _link(payload.get("links"), "next")
            if next_href is None:
                break
            page_url = self._safe_page_url(next_href)
            page_params = None
        else:
            raise PanoramaxError("Panoramax pagination exceeded the configured page cap")
        query_payload = query.model_dump(mode="json")
        return PanoramaxSearchResult.model_validate(
            {
                "zone_id": query.zone_id,
                "api_url": self.api_url,
                "bbox_wgs84": query.bbox_wgs84,
                "query_sha256": _canonical_digest(query_payload),
                "retrieved_at": retrieved_at,
                "images": tuple(sorted(images.values(), key=lambda item: item.image_id)),
            }
        )
