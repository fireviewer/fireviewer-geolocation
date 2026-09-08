from __future__ import annotations

from fireviewer_contracts.model_registry import enabled_public_models
from fireviewer_contracts.mvp_stack import load_mvp_stack
from fireviewer_geolocation.roma_registration import ROMA_SOURCE_REVISION


def test_public_model_provisioning_matches_the_frozen_manifest(monkeypatch) -> None:
    monkeypatch.setenv("FW_ENABLE_FIRE_DETECTOR_ENSEMBLE", "true")
    monkeypatch.delenv("FW_ENABLE_CONSENSUS_JUDGE", raising=False)
    stack = load_mvp_stack()

    declared = {
        (candidate.model_id, candidate.revision)
        for stage in stack.stages
        for candidate in stage.candidates
        if candidate.source == "huggingface" and candidate.provisioned_by_mvp
    }
    if stack.judge.candidate.source == "huggingface":
        declared.add((stack.judge.candidate.model_id, stack.judge.candidate.revision))
    provisioned = {(spec.model_id, spec.revision) for spec in enabled_public_models()}

    assert provisioned == declared


def test_provisioned_roma_baseline_uses_the_audited_source_revision() -> None:
    stack = load_mvp_stack()
    stage = next(stage for stage in stack.stages if stage.stage_id == "local_correspondence")
    baseline = next(candidate for candidate in stage.candidates if candidate.provisioned_by_mvp)

    assert baseline.candidate_id == "correspondence.aerial_extrematch_roma"
    assert baseline.revision == ROMA_SOURCE_REVISION
    assert stage.activation.value == "closed"
