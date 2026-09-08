from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from fireviewer_geolocation.mvp.localization.panoramax import (
    PanoramaxClient,
    PanoramaxError,
    PanoramaxQuery,
)


def _feature(image_id: str, longitude: float, latitude: float) -> dict[str, Any]:
    return {
        "type": "Feature",
        "id": image_id,
        "collection": "SEQUENCE-1",
        "geometry": {"type": "Point", "coordinates": [longitude, latitude, 510]},
        "properties": {
            "datetime": "2026-08-20T10:00:00Z",
            "view:azimuth": 214,
            "pers:pitch": -3,
            "pers:roll": 1,
            "pers:yaw": 2,
            "pers:interior_orientation": {
                "field_of_view": 90,
                "focal_length": 8.4,
            },
            "quality:horizontal_accuracy": 3.2,
        },
        "assets": {
            "hd": {
                "href": f"https://panoramax.example/api/pictures/{image_id}/hd.jpg",
                "roles": ["data"],
            }
        },
        "links": [
            {
                "rel": "self",
                "href": f"https://panoramax.example/api/pictures/{image_id}",
            },
            {"rel": "license", "href": "https://example.test/license"},
        ],
    }


class _Transport:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self.pages = pages
        self.calls: list[tuple[str, dict[str, str | int] | None]] = []

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, str | int] | None = None,
    ) -> dict[str, Any]:
        self.calls.append((url, params))
        return self.pages.pop(0)


def test_panoramax_client_reads_regional_stac_pages_and_preserves_metadata() -> None:
    transport = _Transport(
        [
            {
                "type": "FeatureCollection",
                "features": [_feature("IMAGE-2", 5.371, 44.751)],
                "links": [
                    {
                        "rel": "next",
                        "href": "https://panoramax.example/api/search?page=2",
                    }
                ],
            },
            {
                "type": "FeatureCollection",
                "features": [
                    _feature("IMAGE-1", 5.370, 44.750),
                    _feature("IMAGE-2", 5.371, 44.751),
                ],
                "links": [],
            },
        ]
    )
    client = PanoramaxClient(api_url="https://panoramax.example/api", transport=transport)
    result = client.search(
        PanoramaxQuery(
            zone_id="die-justin",
            bbox_wgs84=(5.36, 44.74, 5.38, 44.76),
            limit=10,
        ),
        retrieved_at=datetime(2026, 8, 21, 10, 0, tzinfo=UTC),
    )

    assert [item.image_id for item in result.images] == ["IMAGE-1", "IMAGE-2"]
    assert result.images[0].heading_deg == 214
    assert result.images[0].field_of_view_deg == 90
    assert result.images[0].gps_accuracy_m == 3.2
    assert str(result.images[0].image_url).endswith("/IMAGE-1/hd.jpg")
    assert transport.calls[0][1] == {"bbox": "5.36,44.74,5.38,44.76", "limit": 10}
    assert transport.calls[1][1] is None


def test_panoramax_client_rejects_national_queries_and_cross_origin_pagination() -> None:
    with pytest.raises(ValidationError, match="remain regional"):
        PanoramaxQuery(
            zone_id="france",
            bbox_wgs84=(-5, 41, 9, 51),
        )

    transport = _Transport(
        [
            {
                "type": "FeatureCollection",
                "features": [_feature("IMAGE-1", 5.37, 44.75)],
                "links": [{"rel": "next", "href": "https://attacker.example/items"}],
            }
        ]
    )
    client = PanoramaxClient(api_url="https://panoramax.example/api", transport=transport)
    with pytest.raises(PanoramaxError, match="configured origin"):
        client.search(
            PanoramaxQuery(
                zone_id="die-justin",
                bbox_wgs84=(5.36, 44.74, 5.38, 44.76),
            ),
            retrieved_at=datetime(2026, 8, 21, 10, 0, tzinfo=UTC),
        )
