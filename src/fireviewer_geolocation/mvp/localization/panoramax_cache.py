from __future__ import annotations

import json
import os
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Literal, Protocol
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from fireviewer_contracts.contracts import SafeIdentifierV2, Sha256HexV2, StrictModel
from fireviewer_geolocation.mvp.localization.panoramax import PanoramaxImage, PanoramaxSearchResult


class PanoramaxAssetTransport(Protocol):
    def get(self, url: str, *, max_bytes: int) -> bytes: ...


class HttpxPanoramaxAssetTransport:
    def __init__(self, *, timeout_seconds: float = 90) -> None:
        self.timeout_seconds = timeout_seconds

    def get(self, url: str, *, max_bytes: int) -> bytes:
        import httpx

        payload = bytearray()
        with httpx.stream(
            "GET",
            url,
            timeout=self.timeout_seconds,
            follow_redirects=False,
            headers={"Accept": "image/jpeg,image/*;q=0.8"},
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if content_type and not content_type.startswith("image/"):
                raise ValueError("Panoramax asset response is not an image")
            for chunk in response.iter_bytes():
                payload.extend(chunk)
                if len(payload) > max_bytes:
                    raise ValueError("Panoramax image exceeds the configured byte limit")
        return bytes(payload)


class CachedPanoramaxAsset(StrictModel):
    image_id: SafeIdentifierV2
    relative_path: str = Field(min_length=1, max_length=500)
    content_sha256: Sha256HexV2
    byte_size: int = Field(gt=0, le=100_000_000)
    width_px: int = Field(gt=0, le=100_000)
    height_px: int = Field(gt=0, le=100_000)

    @model_validator(mode="after")
    def validate_path(self) -> CachedPanoramaxAsset:
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts or "\\" in self.relative_path:
            raise ValueError("Panoramax cache path must remain relative")
        return self


class PanoramaxCacheManifest(StrictModel):
    schema_name: Literal["fireviewer.panoramax-cache.v1"] = Field(
        default="fireviewer.panoramax-cache.v1",
        alias="schema",
        serialization_alias="schema",
    )
    search_result: PanoramaxSearchResult
    assets: tuple[CachedPanoramaxAsset, ...] = Field(default=(), max_length=10_000)

    @model_validator(mode="after")
    def validate_assets(self) -> PanoramaxCacheManifest:
        image_ids = {image.image_id for image in self.search_result.images}
        asset_ids = [asset.image_id for asset in self.assets]
        if len(asset_ids) != len(set(asset_ids)) or set(asset_ids) != image_ids:
            raise ValueError("Panoramax cache assets must exactly cover the search result")
        return self

    def canonical_sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json", by_alias=True),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return sha256(payload).hexdigest()


def _origin(url: str) -> tuple[str, str, int]:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("Panoramax cache URLs must use a plain HTTPS origin")
    return parsed.scheme, parsed.hostname.lower(), parsed.port or 443


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_name = stream.name
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _inspect_image(payload: bytes) -> tuple[int, int]:
    from PIL import Image

    with Image.open(BytesIO(payload)) as image:
        image.verify()
    with Image.open(BytesIO(payload)) as image:
        image.load()
        return image.width, image.height


def materialize_panoramax_cache(
    search_result: PanoramaxSearchResult,
    directory: Path,
    *,
    transport: PanoramaxAssetTransport | None = None,
    max_image_bytes: int = 64_000_000,
) -> PanoramaxCacheManifest:
    if max_image_bytes <= 0 or max_image_bytes > 100_000_000:
        raise ValueError("Panoramax image byte limit is outside the supported range")
    api_origin = _origin(str(search_result.api_url))
    selected_transport = transport or HttpxPanoramaxAssetTransport()
    assets: list[CachedPanoramaxAsset] = []

    for image in search_result.images:
        if image.image_url is None:
            raise ValueError("Panoramax cache requires a downloadable image URL")
        image_url = str(image.image_url)
        if _origin(image_url) != api_origin:
            raise ValueError("Panoramax image URL leaves the configured API origin")
        filename = f"{sha256(image.image_id.encode()).hexdigest()[:32]}.jpg"
        relative_path = PurePosixPath("images", filename)
        path = directory.joinpath(*relative_path.parts)
        if path.is_file():
            payload = path.read_bytes()
        else:
            payload = selected_transport.get(image_url, max_bytes=max_image_bytes)
            _atomic_write(path, payload)
        if not payload or len(payload) > max_image_bytes:
            raise ValueError("Panoramax cache payload violates the configured byte limit")
        width, height = _inspect_image(payload)
        assets.append(
            CachedPanoramaxAsset(
                image_id=image.image_id,
                relative_path=relative_path.as_posix(),
                content_sha256=sha256(payload).hexdigest(),
                byte_size=len(payload),
                width_px=width,
                height_px=height,
            )
        )

    manifest = PanoramaxCacheManifest(
        search_result=search_result,
        assets=tuple(assets),
    )
    manifest_payload = json.dumps(
        manifest.model_dump(mode="json", by_alias=True),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    _atomic_write(directory / "cache-manifest.json", manifest_payload)
    return manifest


class CachedPanoramaxImageLoader:
    def __init__(self, *, directory: Path, manifest: PanoramaxCacheManifest) -> None:
        self.directory = directory.resolve(strict=True)
        self.assets = {asset.image_id: asset for asset in manifest.assets}

    @classmethod
    def from_directory(cls, directory: Path) -> CachedPanoramaxImageLoader:
        manifest = PanoramaxCacheManifest.model_validate_json(
            (directory / "cache-manifest.json").read_text(encoding="utf-8")
        )
        return cls(directory=directory, manifest=manifest)

    def load(self, image: PanoramaxImage) -> object:
        asset = self.assets.get(image.image_id)
        if asset is None:
            raise FileNotFoundError("Panoramax image is absent from the cache manifest")
        path = (self.directory / asset.relative_path).resolve(strict=True)
        if not path.is_relative_to(self.directory) or not path.is_file():
            raise ValueError("Panoramax cache asset leaves its configured directory")
        payload = path.read_bytes()
        if len(payload) != asset.byte_size or sha256(payload).hexdigest() != asset.content_sha256:
            raise ValueError("Panoramax cache asset no longer matches its manifest")

        from PIL import Image

        with Image.open(BytesIO(payload)) as loaded:
            loaded.load()
            if (loaded.width, loaded.height) != (asset.width_px, asset.height_px):
                raise ValueError("Panoramax cache image dimensions changed")
            return loaded.convert("RGB")


__all__ = [
    "CachedPanoramaxAsset",
    "CachedPanoramaxImageLoader",
    "HttpxPanoramaxAssetTransport",
    "PanoramaxAssetTransport",
    "PanoramaxCacheManifest",
    "materialize_panoramax_cache",
]
