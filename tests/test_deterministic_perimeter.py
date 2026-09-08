from fireviewer_geolocation.deterministic_perimeter import ObservedPoint, build_observed_perimeter
from fireviewer_contracts.geometry_contract import validate_geojson_geometry


def _point(lat: float, lon: float, evidence: str, confidence: float = 0.9) -> ObservedPoint:
    return ObservedPoint(lat, lon, evidence, confidence, "observed_burned_perimeter")


def test_perimeter_is_deterministic_and_requires_human_review() -> None:
    proposal = build_observed_perimeter(
        [_point(43.0, 1.0, "b"), _point(43.0, 1.01, "a"), _point(43.01, 1.0, "c")]
    )
    assert proposal.status == "ready_for_human_review"
    assert proposal.geometry is not None
    assert proposal.geometry["properties"] == {
        "source": "deterministic_observed_points",
        "authoritative": False,
        "requires_human_review": True,
    }
    assert proposal.evidence_ids == ("a", "b", "c")
    validate_geojson_geometry(proposal.geometry["geometry"])


def test_perimeter_abstains_with_too_few_points_or_low_confidence() -> None:
    proposal = build_observed_perimeter(
        [_point(43.0, 1.0, "a"), _point(43.01, 1.0, "b", confidence=0.2)]
    )
    assert proposal.status == "abstained"
    assert proposal.reason == "insufficient_reviewed_points"
