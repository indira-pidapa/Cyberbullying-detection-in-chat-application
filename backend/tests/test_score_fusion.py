
from unittest.mock import MagicMock, patch

import pytest
import transformers


# Prevent large model weights from loading in unit tests.
def fake_model(*args, **kwargs):
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model
    model.parameters.return_value = []
    return model


with (
    patch.object(
        transformers.AutoTokenizer,
        "from_pretrained",
        return_value=MagicMock(),
    ),
    patch.object(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        side_effect=fake_model,
    ),
):
    from backend.pipeline import fuse_scores, THRESHOLD


BASE_INPUTS = {
    "muril_score": 0.4,
    "local_model_score": 0.4,
    "banter_signal": 0.0,
    "reciprocity_score": 0.0,
    "trajectory_score": 0.0,
    "history_length": 0,
}


def test_fusion_returns_expected_structure():
    result = fuse_scores(**BASE_INPUTS)

    expected_keys = {
        "base_score",
        "adjusted_score",
        "reduction_applied",
        "memory_bonus",
        "offensive_boost",
        "flagged",
        "label",
    }

    assert expected_keys.issubset(result.keys())


def test_base_score_combines_both_models():
    result = fuse_scores(**BASE_INPUTS)

    expected = 0.4 * 0.4 + 0.6 * 0.4

    assert result["base_score"] == pytest.approx(expected)
    assert result["adjusted_score"] == pytest.approx(expected)


def test_missing_global_score_uses_local_score():
    inputs = {
        **BASE_INPUTS,
        "muril_score": None,
        "local_model_score": 0.6,
    }

    result = fuse_scores(**inputs)

    assert result["base_score"] == pytest.approx(0.6)
    assert result["adjusted_score"] == pytest.approx(0.6)


def test_memory_scores_increase_adjusted_score():
    baseline = fuse_scores(**BASE_INPUTS)

    with_memory = fuse_scores(
        **BASE_INPUTS,
        local_memory_score=0.8,
        global_memory_score=0.6,
    )

    assert with_memory["memory_bonus"] == pytest.approx(0.11)
    assert (
        with_memory["adjusted_score"]
        > baseline["adjusted_score"]
    )


def test_offensive_signal_increases_score():
    baseline = fuse_scores(**BASE_INPUTS)

    with_offensive = fuse_scores(
        **BASE_INPUTS,
        offensive_signal=1.0,
    )

    assert with_offensive["offensive_boost"] == pytest.approx(0.2)
    assert (
        with_offensive["adjusted_score"]
        > baseline["adjusted_score"]
    )


def test_adjusted_score_is_capped_at_one():
    inputs = {
        **BASE_INPUTS,
        "muril_score": 1.0,
        "local_model_score": 1.0,
        "local_memory_score": 1.0,
        "global_memory_score": 1.0,
        "offensive_signal": 1.0,
    }

    result = fuse_scores(**inputs)

    assert result["adjusted_score"] <= 1.0


def test_adjusted_score_stays_within_probability_range():
    result = fuse_scores(**BASE_INPUTS)

    assert 0.0 <= result["adjusted_score"] <= 1.0


def test_custom_threshold_flags_high_score():
    inputs = {
        **BASE_INPUTS,
        "muril_score": 0.95,
        "local_model_score": 0.95,
        "threshold": 0.8,
    }

    result = fuse_scores(**inputs)

    assert result["flagged"] is True
    assert result["label"] == 1


def test_custom_threshold_does_not_flag_low_score():
    inputs = {
        **BASE_INPUTS,
        "muril_score": 0.1,
        "local_model_score": 0.1,
        "threshold": 0.9,
    }

    result = fuse_scores(**inputs)

    assert result["flagged"] is False
    assert result["label"] == 0


def test_default_threshold_is_valid():
    assert 0.0 <= THRESHOLD <= 1.0


def test_explicit_offensive_signal_limits_contextual_reduction():
    inputs = {
        **BASE_INPUTS,
        "banter_signal": 1.0,
        "reciprocity_score": 1.0,
        "trajectory_score": 1.0,
        "offensive_signal": 0.75,
        "history_length": 5,
    }

    result = fuse_scores(**inputs)

    assert result["reduction_applied"] <= 0.15


def test_strong_model_evidence_limits_contextual_reduction():
    inputs = {
        **BASE_INPUTS,
        "muril_score": 0.95,
        "local_model_score": 0.95,
        "banter_signal": 1.0,
        "reciprocity_score": 0.5,
        "trajectory_score": 1.0,
        "history_length": 5,
    }

    result = fuse_scores(**inputs)
    assert result["banter_override"] is False
    assert result["reduction_applied"] <= 0.15

# ---------------------------------------------------------
# STEP 5: evidence-gated banter override
# ---------------------------------------------------------

BANTER_CASE = {
    "muril_score": 0.97,
    "local_model_score": 0.99,
    "banter_signal": 0.65,
    "reciprocity_score": 1.0,
    "trajectory_score": 0.5,
    "history_length": 1,
    "offensive_signal": 0.35,
}


def test_mutual_explicit_banter_clears_overconfident_models():
    result = fuse_scores(**BANTER_CASE)

    assert result["banter_override"] is True
    assert result["reduction_applied"] >= 0.33
    assert result["adjusted_score"] < THRESHOLD
    assert result["flagged"] is False


@pytest.mark.parametrize(
    "change",
    [
        {"reciprocity_score": 0.0},        # one-sided
        {"reciprocity_score": 0.5},        # first message / no demonstrated mutuality
        {"banter_signal": 0.40},           # slang / elongation only
        {"offensive_signal": 0.75},        # real insult in current message
        {"history_length": 0},             # no exchange yet
        {"override_block_reason": "threat_language"},
        {"override_block_reason": "defensive_message"},
        {"override_block_reason": "strong_offensive_history"},
    ],
)
def test_override_requires_every_gate(change):
    result = fuse_scores(**{**BANTER_CASE, **change})

    assert result["banter_override"] is False
    assert result["flagged"] is True


def test_override_reduction_never_exceeds_cap():
    result = fuse_scores(**{**BANTER_CASE, "banter_signal": 1.0})

    assert result["banter_override"] is True
    assert result["reduction_applied"] <= 0.35


def test_override_reports_block_reason():
    result = fuse_scores(
        **{**BANTER_CASE, "override_block_reason": "threat_language"}
    )

    assert result["override_block_reason"] == "threat_language"
