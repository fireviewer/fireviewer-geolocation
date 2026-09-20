from __future__ import annotations

from fireviewer_contracts.model_registry import enabled_public_models
from fireviewer_contracts.mvp_stack import load_mvp_stack
from fireviewer_geolocation.roma_registration import ROMA_SOURCE_REVISION


def test_public_model_provisioning_matches_the_frozen_manifest(monkeypatch) -> None:
    monkeypatch.setenv("FW_ENABLE_FIRE_DETECTOR_ENSEMBLE", "true")
    monkeypatch.setenv("FW_ENABLE_CONSENSUS_JUDGE", "true")
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


def test_disabled_judge_is_not_downloaded_but_source_research_is_kept(monkeypatch) -> None:
    monkeypatch.setenv("FW_ENABLE_CONSENSUS_JUDGE", "false")
    models = enabled_public_models()
    assert all(spec.role != "consensus_judge" for spec in models)
    assert all(spec.model_id != "prism-ml/Ternary-Bonsai-2-27B-gguf" for spec in models)
    assert any(spec.role == "source_research" and spec.model_id == "Qwen/Qwen3-14B"
               for spec in models)
