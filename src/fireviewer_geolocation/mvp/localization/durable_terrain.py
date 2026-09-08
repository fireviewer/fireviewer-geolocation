"""Integrity-checked transient loading of durable backend FWTERRAIN objects."""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from fireviewer_geolocation.mvp.localization.geographic_hypotheses import (
    TerrainSurfaceElevationProvider,
)
from fireviewer_contracts.mvp.supervision.backend_event_evidence import (
    AzureBackendEventEvidenceConfig,
    DurableTerrainReference,
)
from fireviewer_geolocation.spatial_geometry import SpatialGeometryError, load_fwterrain


class DurableTerrainError(RuntimeError):
    """The selected durable terrain could not be verified or decoded."""


@dataclass(frozen=True, slots=True)
class TerrainDownloadReceipt:
    size_bytes: int
    sha256: str
    headers: Mapping[str, str]


class DurableTerrainTransport(Protocol):
    def download(
        self,
        url: str,
        destination: Path,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
        maximum_bytes: int,
    ) -> TerrainDownloadReceipt: ...


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


class UrllibDurableTerrainTransport:
    def download(
        self,
        url: str,
        destination: Path,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
        maximum_bytes: int,
    ) -> TerrainDownloadReceipt:
        request = Request(url, headers=dict(headers), method="GET")  # noqa: S310
        digest = hashlib.sha256()
        size = 0
        try:
            with build_opener(_NoRedirectHandler()).open(
                request,
                timeout=timeout_seconds,
            ) as response:
                if response.headers.get_content_type() != "application/vnd.fireviewer.terrain":
                    raise DurableTerrainError("backend terrain content type is invalid")
                raw_length = response.headers.get("content-length")
                if raw_length is not None and int(raw_length) > maximum_bytes:
                    raise DurableTerrainError("backend terrain exceeds its declared size")
                with destination.open("xb") as stream:
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > maximum_bytes:
                            raise DurableTerrainError("backend terrain exceeds its declared size")
                        digest.update(chunk)
                        stream.write(chunk)
                response_headers = {
                    key.casefold(): value for key, value in response.headers.items()
                }
        except HTTPError as exc:
            if exc.code == HTTPStatus.NOT_FOUND:
                raise DurableTerrainError("backend terrain was not found") from exc
            raise DurableTerrainError(f"backend terrain returned HTTP {exc.code}") from exc
        except (OSError, URLError, ValueError) as exc:
            raise DurableTerrainError("backend terrain download failed") from exc
        return TerrainDownloadReceipt(
            size_bytes=size,
            sha256=digest.hexdigest(),
            headers=response_headers,
        )


class AzureBackendTerrainResolver:
    """Load one checksum-qualified terrain into memory, leaving no durable local copy."""

    def __init__(
        self,
        config: AzureBackendEventEvidenceConfig,
        *,
        transport: DurableTerrainTransport | None = None,
    ) -> None:
        self._config = config
        self._transport = transport or UrllibDurableTerrainTransport()

    def resolve(
        self,
        reference: DurableTerrainReference,
    ) -> TerrainSurfaceElevationProvider:
        base = self._config.base_url.rstrip("/") + "/"
        if not reference.content_url.startswith(base):
            raise DurableTerrainError("terrain URL is outside the configured backend origin")
        with tempfile.TemporaryDirectory(prefix="fireviewer-terrain-") as directory:
            path = Path(directory) / f"{reference.terrain_id}.fwterrain"
            try:
                receipt = self._transport.download(
                    reference.content_url,
                    path,
                    headers={
                        "Accept": reference.media_type,
                        "Authorization": (
                            "Bearer "
                            + self._config.bearer_token.get_secret_value()
                        ),
                    },
                    timeout_seconds=self._config.timeout_seconds,
                    maximum_bytes=reference.size_bytes,
                )
                if receipt.size_bytes != reference.size_bytes:
                    raise DurableTerrainError("terrain size differs from durable metadata")
                if receipt.sha256 != reference.sha256:
                    raise DurableTerrainError("terrain SHA-256 differs from durable metadata")
                if (
                    receipt.headers.get("x-checksum-sha256") != reference.sha256
                    or receipt.headers.get("etag") != f'"{reference.sha256}"'
                ):
                    raise DurableTerrainError("terrain revision headers are inconsistent")
                surface = load_fwterrain(path, declared_crs=reference.crs)
            except (OSError, SpatialGeometryError) as exc:
                raise DurableTerrainError("durable terrain could not be decoded") from exc
        return TerrainSurfaceElevationProvider(
            surface,
            reference_revision=reference.sha256,
        )


__all__ = [
    "AzureBackendTerrainResolver",
    "DurableTerrainError",
    "DurableTerrainTransport",
    "TerrainDownloadReceipt",
    "UrllibDurableTerrainTransport",
]
