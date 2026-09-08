from __future__ import annotations

from hashlib import sha256
from math import isclose
from typing import Any

from pydantic import Field, model_validator

from fireviewer_contracts.contracts import SafeIdentifierV2, Sha256HexV2, StrictModel


class PerspectiveConfig(StrictModel):
    headings_deg: tuple[float, ...] = (0, 45, 90, 135, 180, 225, 270, 315)
    pitch_deg: float = Field(default=0, ge=-89, le=89)
    horizontal_fov_deg: float = Field(default=90, gt=0, lt=180)
    width_px: int = Field(default=512, ge=32, le=8_192)
    height_px: int = Field(default=512, ge=32, le=8_192)

    @model_validator(mode="after")
    def validate_headings(self) -> PerspectiveConfig:
        if not 1 <= len(self.headings_deg) <= 32:
            raise ValueError("perspective config requires between 1 and 32 headings")
        normalized = tuple(heading % 360 for heading in self.headings_deg)
        if len(normalized) != len(set(normalized)):
            raise ValueError("perspective headings must be unique modulo 360 degrees")
        object.__setattr__(self, "headings_deg", normalized)
        return self


class PerspectiveCropManifest(StrictModel):
    crop_id: SafeIdentifierV2
    image_id: SafeIdentifierV2
    heading_deg: float = Field(ge=0, lt=360)
    pitch_deg: float = Field(ge=-89, le=89)
    horizontal_fov_deg: float = Field(gt=0, lt=180)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    pixel_sha256: Sha256HexV2


def _pixel_digest(image: Any) -> str:
    mode = image.mode
    size = image.size
    payload = f"{mode}:{size[0]}x{size[1]}:".encode() + image.tobytes()
    return sha256(payload).hexdigest()


def equirectangular_to_perspective(
    image: object,
    *,
    heading_deg: float,
    pitch_deg: float,
    horizontal_fov_deg: float,
    output_size: tuple[int, int],
) -> Any:
    """Project an equirectangular panorama into one rectilinear RGB crop."""

    import numpy as np
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError("perspective projection requires a Pillow image")
    output_width, output_height = output_size
    if output_width <= 0 or output_height <= 0:
        raise ValueError("perspective output dimensions must be positive")
    if not 0 < horizontal_fov_deg < 180:
        raise ValueError("perspective horizontal FOV must lie between 0 and 180 degrees")
    if not -89 <= pitch_deg <= 89:
        raise ValueError("perspective pitch must lie between -89 and 89 degrees")

    source = np.asarray(image.convert("RGB"), dtype=np.float32)
    source_height, source_width = source.shape[:2]
    x_pixel = np.arange(output_width, dtype=np.float64) + 0.5
    y_pixel = np.arange(output_height, dtype=np.float64) + 0.5
    x_grid, y_grid = np.meshgrid(x_pixel, y_pixel)
    focal_px = (output_width / 2) / np.tan(np.deg2rad(horizontal_fov_deg) / 2)
    x_direction = (x_grid - output_width / 2) / focal_px
    y_direction = -(y_grid - output_height / 2) / focal_px
    z_direction = np.ones_like(x_direction)
    norm = np.sqrt(x_direction**2 + y_direction**2 + z_direction**2)
    x_direction /= norm
    y_direction /= norm
    z_direction /= norm

    pitch = np.deg2rad(pitch_deg)
    pitched_y = np.cos(pitch) * y_direction + np.sin(pitch) * z_direction
    pitched_z = -np.sin(pitch) * y_direction + np.cos(pitch) * z_direction
    heading = np.deg2rad(heading_deg % 360)
    world_x = np.cos(heading) * x_direction + np.sin(heading) * pitched_z
    world_y = pitched_y
    world_z = -np.sin(heading) * x_direction + np.cos(heading) * pitched_z

    longitude = np.arctan2(world_x, world_z)
    latitude = np.arcsin(np.clip(world_y, -1, 1))
    source_x = (longitude / (2 * np.pi) + 0.5) * source_width - 0.5
    source_y = (0.5 - latitude / np.pi) * source_height - 0.5

    left = np.floor(source_x).astype(np.int64) % source_width
    right = (left + 1) % source_width
    top = np.clip(np.floor(source_y).astype(np.int64), 0, source_height - 1)
    bottom = np.clip(top + 1, 0, source_height - 1)
    x_weight = (source_x - np.floor(source_x))[..., None]
    y_weight = (source_y - np.floor(source_y))[..., None]
    top_row = source[top, left] * (1 - x_weight) + source[top, right] * x_weight
    bottom_row = source[bottom, left] * (1 - x_weight) + source[bottom, right] * x_weight
    output = top_row * (1 - y_weight) + bottom_row * y_weight
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode="RGB")


def generate_perspective_crops(
    image_id: SafeIdentifierV2,
    image: object,
    *,
    config: PerspectiveConfig | None = None,
) -> tuple[tuple[PerspectiveCropManifest, Any], ...]:
    selected_config = config or PerspectiveConfig()
    crops: list[tuple[PerspectiveCropManifest, Any]] = []
    for heading in selected_config.headings_deg:
        crop = equirectangular_to_perspective(
            image,
            heading_deg=heading,
            pitch_deg=selected_config.pitch_deg,
            horizontal_fov_deg=selected_config.horizontal_fov_deg,
            output_size=(selected_config.width_px, selected_config.height_px),
        )
        heading_token = f"{heading:.6f}".rstrip("0").rstrip(".").replace(".", "p")
        crop_id = f"{image_id}-h{heading_token}"
        crops.append(
            (
                PerspectiveCropManifest(
                    crop_id=crop_id,
                    image_id=image_id,
                    heading_deg=heading,
                    pitch_deg=selected_config.pitch_deg,
                    horizontal_fov_deg=selected_config.horizontal_fov_deg,
                    width_px=selected_config.width_px,
                    height_px=selected_config.height_px,
                    pixel_sha256=_pixel_digest(crop),
                ),
                crop,
            )
        )
    return tuple(crops)


def is_four_view_subset(
    four_view: PerspectiveConfig,
    eight_view: PerspectiveConfig | None = None,
) -> bool:
    reference = eight_view or PerspectiveConfig()
    if len(four_view.headings_deg) != 4 or len(reference.headings_deg) != 8:
        return False
    if not set(four_view.headings_deg).issubset(reference.headings_deg):
        return False
    return (
        isclose(four_view.pitch_deg, reference.pitch_deg)
        and isclose(four_view.horizontal_fov_deg, reference.horizontal_fov_deg)
        and four_view.width_px == reference.width_px
        and four_view.height_px == reference.height_px
    )
