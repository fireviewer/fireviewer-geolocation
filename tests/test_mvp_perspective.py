from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from fireviewer_geolocation.mvp.localization.perspective import (
    PerspectiveConfig,
    generate_perspective_crops,
    is_four_view_subset,
)


def _longitude_panorama() -> Image.Image:
    longitude = np.linspace(0, 255, 360, endpoint=False, dtype=np.uint8)
    pixels = np.repeat(longitude[None, :, None], 180, axis=0)
    rgb = np.repeat(pixels, 3, axis=2)
    return Image.fromarray(rgb, mode="RGB")


def test_eight_perspective_crops_are_deterministic_and_follow_headings() -> None:
    config = PerspectiveConfig(width_px=64, height_px=64)
    first = generate_perspective_crops("PANORAMAX-1", _longitude_panorama(), config=config)
    second = generate_perspective_crops("PANORAMAX-1", _longitude_panorama(), config=config)

    assert len(first) == 8
    assert [manifest.pixel_sha256 for manifest, _ in first] == [
        manifest.pixel_sha256 for manifest, _ in second
    ]
    center_values = [crop.getpixel((32, 32))[0] for _, crop in first]
    assert 120 <= center_values[0] <= 135
    assert 184 <= center_values[2] <= 200
    assert center_values[4] <= 8 or center_values[4] >= 247
    assert 56 <= center_values[6] <= 72
    assert all(crop.size == (64, 64) for _, crop in first)


def test_four_view_profile_is_an_explicit_subset_of_eight_view_baseline() -> None:
    eight = PerspectiveConfig(width_px=64, height_px=64)
    four = PerspectiveConfig(
        headings_deg=(0, 90, 180, 270),
        width_px=64,
        height_px=64,
    )

    assert is_four_view_subset(four, eight)
    assert len(generate_perspective_crops("PANORAMAX-1", _longitude_panorama(), config=four)) == 4


def test_perspective_config_rejects_duplicate_modulo_headings() -> None:
    with pytest.raises(ValidationError, match="unique modulo"):
        PerspectiveConfig(headings_deg=(0, 360))
