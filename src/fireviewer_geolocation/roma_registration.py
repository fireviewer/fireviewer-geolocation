"""Pinned, offline-only AerialExtreMatch-RoMa assets and registration runtime."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

ROMA_SOURCE_REVISION = "048ab96f84430f3e0f1144f05c94fe1e1f0bca8a"
ROMA_SOURCE_ARCHIVE_URL = (
    f"https://codeload.github.com/Xecades/AerialExtreMatch/tar.gz/{ROMA_SOURCE_REVISION}"
)
ROMA_SOURCE_ARCHIVE_SHA256 = "c95644abd917c62d7bbcad4ff057201aecf61daab282520603c4db606ecac5b4"
ROMA_LICENSE = "MIT"
DINO_LICENSE = "Apache-2.0"
DEFAULT_ROMA_ROOT = Path("/runpod-volume/firewarning-roma")
_DOWNLOAD_PROGRESS_STEP_BYTES = 128 * 1024 * 1024


class RomaAssetError(RuntimeError):
    """Raised when a pinned registration asset is missing or altered."""


@dataclass(frozen=True, slots=True)
class AssetSpec:
    filename: str
    url: str
    size: int
    sha256: str
    license: str


ROMA_CHECKPOINT = AssetSpec(
    filename="roma_extre.pth",
    url=("https://github.com/Xecades/AerialExtreMatch/releases/download/v1.0.0/roma_extre.pth"),
    size=445_618_304,
    sha256="368f9e0a59734fe293048496be201a57158278554f6f923f0b25a4424b30f6ae",
    license=ROMA_LICENSE,
)
DINO_CHECKPOINT = AssetSpec(
    filename="dinov2_vitl14_pretrain.pth",
    url=("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"),
    size=1_217_586_395,
    sha256="d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428",
    license=DINO_LICENSE,
)
ROMA_ASSETS = (ROMA_CHECKPOINT, DINO_CHECKPOINT)


@dataclass(frozen=True, slots=True)
class RomaAssetPaths:
    root: Path
    checkpoint: Path
    dinov2: Path


@dataclass(frozen=True, slots=True)
class DenseMatches:
    source_pixels: Any
    map_pixels: Any
    certainties: Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def roma_root_from_environment() -> Path:
    return Path(os.getenv("FW_ROMA_ROOT", str(DEFAULT_ROMA_ROOT))).resolve()


def asset_paths(root: Path) -> RomaAssetPaths:
    root = root.resolve()
    return RomaAssetPaths(
        root=root,
        checkpoint=root / "weights" / ROMA_CHECKPOINT.filename,
        dinov2=root / "weights" / DINO_CHECKPOINT.filename,
    )


def verify_asset(path: Path, spec: AssetSpec) -> None:
    if not path.is_file():
        raise RomaAssetError(f"pinned model asset is absent: {path}")
    actual_size = path.stat().st_size
    if actual_size != spec.size:
        raise RomaAssetError(f"model asset size mismatch for {path}: {actual_size} != {spec.size}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != spec.sha256:
        raise RomaAssetError(f"model asset SHA-256 mismatch for {path}")


def verify_roma_assets(root: Path) -> RomaAssetPaths:
    paths = asset_paths(root)
    verify_asset(paths.checkpoint, ROMA_CHECKPOINT)
    verify_asset(paths.dinov2, DINO_CHECKPOINT)
    return paths


def _download_asset(
    root: Path,
    spec: AssetSpec,
    *,
    opener: Callable[..., Any] = urlopen,
) -> Path:
    destination = root.resolve() / "weights" / spec.filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify_asset(destination, spec)
        print(
            f"firewarning bootstrap: spatial model cache hit asset={spec.filename}",
            flush=True,
        )
        return destination
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    try:
        print(
            "firewarning bootstrap: downloading spatial model "
            f"asset={spec.filename} bytes={spec.size}",
            flush=True,
        )
        downloaded = 0
        next_progress = _DOWNLOAD_PROGRESS_STEP_BYTES
        with opener(spec.url, timeout=120) as source, partial.open("wb") as output:
            while chunk := source.read(1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_progress or downloaded >= spec.size:
                    percent = min(100, round(downloaded * 100 / spec.size))
                    print(
                        "firewarning bootstrap: spatial model progress "
                        f"asset={spec.filename} downloaded={downloaded} "
                        f"total={spec.size} percent={percent}",
                        flush=True,
                    )
                    next_progress += _DOWNLOAD_PROGRESS_STEP_BYTES
        verify_asset(partial, spec)
        os.replace(partial, destination)
        print(
            f"firewarning bootstrap: spatial model ready asset={spec.filename}",
            flush=True,
        )
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return destination


def provision_roma_assets(root: Path) -> dict[str, Any]:
    root = root.resolve()
    for spec in ROMA_ASSETS:
        _download_asset(root, spec)
    paths = verify_roma_assets(root)
    manifest = {
        "assets": [
            {
                "filename": spec.filename,
                "license": spec.license,
                "sha256": spec.sha256,
                "size": spec.size,
                "source_url": spec.url,
            }
            for spec in ROMA_ASSETS
        ],
        "model": "AerialExtreMatch-RoMa",
        "runtime_network_required": False,
        "source_archive_sha256": ROMA_SOURCE_ARCHIVE_SHA256,
        "source_revision": ROMA_SOURCE_REVISION,
        "storage_policy": "external_volume_no_docker_image_no_git",
    }
    manifest_path = paths.root / "model-assets.json"
    partial = manifest_path.with_suffix(".json.partial")
    partial.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, manifest_path)
    return manifest


def load_roma_model(
    root: Path,
    *,
    device: str = "cuda",
    coarse_res: int = 560,
    upsample_res: int = 864,
) -> Any:
    """Load both state dicts from verified local files; this function never downloads."""
    paths = verify_roma_assets(root)
    try:
        import torch
        from romatch.models.model_zoo import roma_extre
    except ImportError as exc:  # pragma: no cover - explicit runtime packaging failure
        raise RomaAssetError("RoMa runtime dependencies are not installed") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RomaAssetError("CUDA is required for the production RoMa runtime")
    weights = torch.load(paths.checkpoint, map_location="cpu", weights_only=True)
    dinov2_weights = torch.load(paths.dinov2, map_location="cpu", weights_only=True)
    model = roma_extre(
        device=torch.device(device),
        weights=weights,
        dinov2_weights=dinov2_weights,
        coarse_res=coarse_res,
        upsample_res=upsample_res,
        amp_dtype=torch.float16,
    )
    del weights, dinov2_weights
    model.eval()
    return model


def match_pair(
    model: Any,
    source_image: Any,
    map_image: Any,
    *,
    sample_count: int = 5_000,
) -> DenseMatches:
    """Return dense source/map pixels; pose and terrain intersection stay deterministic."""
    try:
        import numpy as np
        import torch
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - explicit runtime packaging failure
        raise RomaAssetError("RoMa registration dependencies are not installed") from exc
    if isinstance(source_image, (str, os.PathLike)):
        with Image.open(source_image) as source_handle:
            source_input = source_handle.convert("RGB")
    else:
        source_input = source_image
    if isinstance(map_image, (str, os.PathLike)):
        with Image.open(map_image) as map_handle:
            map_input = map_handle.convert("RGB")
    else:
        map_input = map_image
    source_width, source_height = source_input.size
    map_width, map_height = map_input.size
    warp, certainty = model.match(source_input, map_input, device=torch.device("cuda"))
    matches, certainties = model.sample(warp, certainty, num=sample_count)
    source_points, map_points = model.to_pixel_coordinates(
        matches,
        source_height,
        source_width,
        map_height,
        map_width,
    )
    source_np = source_points.detach().float().cpu().numpy()
    map_np = map_points.detach().float().cpu().numpy()
    certainty_np = certainties.detach().float().cpu().numpy()
    finite = np.isfinite(source_np).all(axis=1) & np.isfinite(map_np).all(axis=1)
    return DenseMatches(
        source_pixels=source_np[finite],
        map_pixels=map_np[finite],
        certainties=certainty_np[finite],
    )
