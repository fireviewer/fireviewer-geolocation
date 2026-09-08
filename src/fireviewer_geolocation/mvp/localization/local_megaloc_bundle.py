from __future__ import annotations

import importlib.util
import json
import re
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from fireviewer_contracts.contracts import Sha256HexV2, StrictModel

MegaLocFilename = Literal["config.json", "megaloc_model.py", "model.safetensors"]
_REQUIRED_FILES: tuple[MegaLocFilename, ...] = (
    "config.json",
    "megaloc_model.py",
    "model.safetensors",
)


class LocalMegaLocFile(StrictModel):
    filename: MegaLocFilename
    byte_size: int = Field(gt=0)
    sha256: Sha256HexV2


class LocalMegaLocBundleManifest(StrictModel):
    schema_name: Literal["fireviewer.local-megaloc-bundle.v1"] = Field(
        default="fireviewer.local-megaloc-bundle.v1",
        alias="schema",
        serialization_alias="schema",
    )
    model_id: Literal["gberton/MegaLoc"] = "gberton/MegaLoc"
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    files: tuple[LocalMegaLocFile, ...] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_files(self) -> LocalMegaLocBundleManifest:
        filenames = [item.filename for item in self.files]
        if set(filenames) != set(_REQUIRED_FILES) or len(filenames) != len(set(filenames)):
            raise ValueError("local MegaLoc bundle must contain its three required files")
        return self


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_local_megaloc_bundle(
    directory: Path,
    *,
    revision: str,
) -> LocalMegaLocBundleManifest:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("MegaLoc revision must be an immutable 40-character commit")
    root = directory.resolve(strict=True)
    files: list[LocalMegaLocFile] = []
    for filename in _REQUIRED_FILES:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"local MegaLoc bundle is missing {filename}")
        metadata_path = root / ".cache" / "huggingface" / "download" / f"{filename}.metadata"
        if metadata_path.is_file():
            metadata_lines = metadata_path.read_text(encoding="utf-8").splitlines()
            if not metadata_lines or metadata_lines[0] != revision:
                raise ValueError("MegaLoc Hugging Face receipt revision does not match")
        files.append(
            LocalMegaLocFile(
                filename=filename,
                byte_size=path.stat().st_size,
                sha256=_file_digest(path),
            )
        )
    return LocalMegaLocBundleManifest(
        revision=revision,
        files=tuple(files),
    )


class LocalMegaLocModelLoader:
    """Load only a digest-qualified local MegaLoc source and safetensors bundle."""

    def __init__(
        self,
        *,
        directory: Path,
        manifest: LocalMegaLocBundleManifest,
    ) -> None:
        self.directory = directory.resolve(strict=True)
        self.manifest = manifest

    def __call__(self) -> Any:
        for expected in self.manifest.files:
            path = self.directory / expected.filename
            if (
                not path.is_file()
                or path.stat().st_size != expected.byte_size
                or _file_digest(path) != expected.sha256
            ):
                raise ValueError("local MegaLoc bundle no longer matches its manifest")

        config_payload = json.loads((self.directory / "config.json").read_text(encoding="utf-8"))
        if not isinstance(config_payload, dict):
            raise ValueError("local MegaLoc config must be a JSON object")
        architecture = config_payload.get("architectures")
        if architecture != ["MegaLoc"] or config_payload.get("model_type") != "megaloc":
            raise ValueError("local MegaLoc config declares an unexpected architecture")
        constructor_keys = ("feat_dim", "num_clusters", "cluster_dim", "token_dim", "mlp_dim")
        constructor_config: dict[str, int] = {}
        for key in constructor_keys:
            value = config_payload.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("local MegaLoc config contains an invalid dimension")
            constructor_config[key] = value

        module_name = f"_fireviewer_megaloc_{self.manifest.revision}"
        module_path = self.directory / "megaloc_model.py"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError("local MegaLoc module could not be imported")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            model_class = getattr(module, "MegaLoc", None)
            if model_class is None:
                raise ImportError("local MegaLoc module does not expose MegaLoc")
            model = model_class(**constructor_config)
            from safetensors.torch import load_file

            state = load_file(str(self.directory / "model.safetensors"), device="cpu")
            model.load_state_dict(state, strict=True)
            return model
        finally:
            sys.modules.pop(module_name, None)


__all__ = [
    "LocalMegaLocBundleManifest",
    "LocalMegaLocFile",
    "LocalMegaLocModelLoader",
    "inspect_local_megaloc_bundle",
]
