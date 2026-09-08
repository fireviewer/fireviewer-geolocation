"""Deterministic observed-perimeter proposal from reviewed geographic points.

The function is deliberately conservative: it never creates a perimeter from
image pixels alone, requires explicit observed coordinates and evidence ids,
and returns a proposal with a human-review gate rather than a final incident
fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians
from typing import Literal


@dataclass(frozen=True)
class ObservedPoint:
    latitude: float
    longitude: float
    evidence_id: str
    confidence: float
    observation_kind: Literal["observed_hotspot", "observed_burned_perimeter"]


@dataclass(frozen=True)
class PerimeterProposal:
    geometry: dict[str, object] | None
    status: Literal["ready_for_human_review", "abstained"]
    reason: str | None
    evidence_ids: tuple[str, ...]
    uncertainty_m: float | None


def _valid(point: ObservedPoint) -> bool:
    return (
        -90.0 <= point.latitude <= 90.0
        and -180.0 <= point.longitude <= 180.0
        and bool(point.evidence_id.strip())
        and 0.0 <= point.confidence <= 1.0
    )


def _hull(
    points: list[tuple[float, float, ObservedPoint]],
) -> list[tuple[float, float, ObservedPoint]]:
    ordered = sorted(points, key=lambda item: (item[0], item[1], item[2].evidence_id))

    def cross(
        a: tuple[float, float, ObservedPoint],
        b: tuple[float, float, ObservedPoint],
        c: tuple[float, float, ObservedPoint],
    ) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower: list[tuple[float, float, ObservedPoint]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float, ObservedPoint]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def build_observed_perimeter(
    points: list[ObservedPoint],
    *,
    minimum_confidence: float = 0.75,
) -> PerimeterProposal:
    eligible = [
        point for point in points if _valid(point) and point.confidence >= minimum_confidence
    ]
    evidence_ids = tuple(sorted({point.evidence_id for point in eligible}))
    if len(eligible) < 3:
        return PerimeterProposal(
            None,
            "abstained",
            "insufficient_reviewed_points",
            evidence_ids,
            None,
        )
    latitude0 = sum(point.latitude for point in eligible) / len(eligible)
    scale = 111_320.0
    projected = [
        (
            point.longitude * scale * cos(radians(latitude0)),
            point.latitude * scale,
            point,
        )
        for point in eligible
    ]
    hull = _hull(projected)
    if len(hull) < 3:
        return PerimeterProposal(None, "abstained", "collinear_points", evidence_ids, None)
    coordinates = [[point.longitude, point.latitude] for _x, _y, point in hull]
    coordinates.append(coordinates[0])
    uncertainty = max(1.0, round((1.0 - min(point.confidence for point in eligible)) * 1000.0, 2))
    return PerimeterProposal(
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coordinates]},
            "properties": {
                "source": "deterministic_observed_points",
                "authoritative": False,
                "requires_human_review": True,
            },
        },
        "ready_for_human_review",
        None,
        evidence_ids,
        uncertainty,
    )
