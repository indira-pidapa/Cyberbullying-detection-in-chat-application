from unittest.mock import MagicMock, patch

import pytest
import transformers


# Mock model loading so tests never load the real MuRIL weights.
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
    from backend import lgs_pipeline as router


@pytest.fixture
def safe_local_route(monkeypatch):
    """Provide predictable signals for testing the routing decision."""
    monkeypatch.setattr(router, "get_offensive_signal", lambda text: 0.0)
    monkeypatch.setattr(router, "get_local_model_score", lambda text, history: 0.99)
    monkeypatch.setattr(router, "get_banter_signal", lambda text, history: 0.0)
    monkeypatch.setattr(
        router, "get_reciprocity_score", lambda history: (0.8, {"user": 2})
    )
    monkeypatch.setattr(
        router, "get_sentiment_trajectory", lambda history: (0.8, [0.2, 0.8])
    )
    monkeypatch.setattr(router, "get_message_embedding", lambda text: [0.1, 0.2])
    monkeypatch.setattr(router, "search_local_memory", lambda *a, **kw: [])
    monkeypatch.setattr(router, "search_global_memory", lambda *a, **kw: [])
    monkeypatch.setattr(
        router,
        "fuse_scores",
        lambda **kwargs: {
            "adjusted_score": 0.2,
            "flagged": False,
            "label": 0,
        },
    )


def test_high_confidence_and_low_contextual_risk_uses_local_route(safe_local_route):
    result = router.lgs_pipeline(
        conversation_id="unit-test",
        conversation_history=[],
        current_message={"sender": "user", "text": "Hello there"},
    )

    assert result["source"] == "local_only"
    assert result["router_reason"] == "high_local_confidence"
    assert result["confidence"] == pytest.approx(0.98)
    assert result["global_score"] is None
    assert result["final_score"] == pytest.approx(0.2)


def test_low_local_confidence_uses_global_fallback(safe_local_route, monkeypatch):
    monkeypatch.setattr(router, "get_local_model_score", lambda text, history: 0.55)
    monkeypatch.setattr(router, "get_muril_score", lambda text, history: 0.8)

    result = router.lgs_pipeline(
        conversation_id="unit-test",
        conversation_history=[],
        current_message={"sender": "user", "text": "Maybe harmful"},
    )

    assert result["source"] == "global_fallback"
    assert result["router_reason"] == "low_local_confidence"
    assert result["confidence"] == pytest.approx(0.1)
    assert result["global_score"] == pytest.approx(0.8)


def test_contextual_risk_triggers_global_fallback_even_with_high_confidence(
    safe_local_route, monkeypatch
):
    monkeypatch.setattr(router, "get_local_model_score", lambda text, history: 0.99)
    monkeypatch.setattr(router, "get_banter_signal", lambda text, history: 0.8)
    monkeypatch.setattr(router, "get_muril_score", lambda text, history: 0.7)

    result = router.lgs_pipeline(
        conversation_id="unit-test",
        conversation_history=[],
        current_message={"sender": "user", "text": "Test message"},
    )

    assert result["source"] == "global_fallback"
    assert result["router_reason"] == "contextual_risk"
    assert result["contextual_risk"] is True
    assert result["global_score"] == pytest.approx(0.7)


def test_local_memory_never_reinforces_score_from_prior_flags(safe_local_route, monkeypatch):
    monkeypatch.setattr(
        router,
        "search_local_memory",
        lambda *a, **kw: [
            {"flagged": True, "similarity": 0.7},
            {"flagged": False, "similarity": 0.99},
            {"flagged": True, "similarity": 0.85},
        ],
    )

    captured = {}

    def fake_fuse(**kwargs):
        captured.update(kwargs)
        return {"adjusted_score": 0.3, "flagged": False, "label": 0}

    monkeypatch.setattr(router, "fuse_scores", fake_fuse)

    result = router.lgs_pipeline(
        conversation_id="unit-test",
        conversation_history=[],
        current_message={"sender": "user", "text": "Test memory"},
    )

    # Earlier automated flags must NOT raise future scores (no feedback loop),
    # even when highly similar flagged memories exist.
    assert captured["local_memory_score"] is None
    assert result["local_memory_score"] is None
    # Memories are still retrieved and returned for context / auditing.
    assert len(result["local_memories"]) == 3


def test_global_memory_uses_close_harmful_neighbours(safe_local_route, monkeypatch):
    monkeypatch.setattr(router, "get_local_model_score", lambda text, history: 0.55)
    monkeypatch.setattr(router, "get_muril_score", lambda text, history: 0.8)
    monkeypatch.setattr(
        router,
        "search_global_memory",
        lambda *a, **kw: [
            {"label": 1, "similarity": 0.95},
            {"label": 0, "similarity": 0.99},
            {"label": "1", "similarity": 0.92},
            {"label": 1, "similarity": 0.60},   # too far away to count
        ],
    )

    captured = {}

    def fake_fuse(**kwargs):
        captured.update(kwargs)
        return {"adjusted_score": 0.7, "flagged": False, "label": 0}

    monkeypatch.setattr(router, "fuse_scores", fake_fuse)

    result = router.lgs_pipeline(
        conversation_id="unit-test",
        conversation_history=[],
        current_message={"sender": "user", "text": "Test global memory"},
    )

    assert captured["global_memory_score"] == pytest.approx(0.95)
    assert result["global_memory_score"] == pytest.approx(0.95)
    assert len(result["global_memories"]) == 4


def test_missing_conversation_id_skips_local_memory_search(
    safe_local_route, monkeypatch
):
    search_mock = MagicMock(return_value=[])
    monkeypatch.setattr(router, "search_local_memory", search_mock)

    result = router.lgs_pipeline(
        conversation_id=None,
        conversation_history=[],
        current_message={"sender": "user", "text": "Hello"},
    )

    search_mock.assert_not_called()
    assert result["local_memories"] == []


def test_result_contains_common_routing_fields(safe_local_route):
    result = router.lgs_pipeline(
        conversation_id="unit-test",
        conversation_history=[],
        current_message={"sender": "user", "text": "Hello"},
    )

    expected_keys = {
        "confidence",
        "source",
        "local_score",
        "global_score",
        "final_score",
        "threshold",
        "local_memories",
        "global_memories",
        "sender_counts",
        "history_length",
        "contextual_risk",
        "router_reason",
        "embedding",
    }

    assert expected_keys.issubset(result.keys())
def test_moderate_banter_still_gets_global_second_opinion(
    safe_local_route, monkeypatch
):
    # banter evidence (0.5+) means context may be masking harm or friendliness:
    # even a very confident local model must be cross-checked by the global model.
    monkeypatch.setattr(router, "get_local_model_score", lambda text, history: 0.99)
    monkeypatch.setattr(router, "get_banter_signal", lambda text, history: 0.55)
    monkeypatch.setattr(router, "get_muril_score", lambda text, history: 0.4)

    result = router.lgs_pipeline(
        conversation_id="unit-test",
        conversation_history=[],
        current_message={"sender": "user", "text": "Test message"},
    )

    assert result["source"] == "global_fallback"
    assert result["router_reason"] == "contextual_risk"
    assert result["global_score"] == pytest.approx(0.4)


def test_weak_banter_alone_does_not_force_global(safe_local_route, monkeypatch):
    monkeypatch.setattr(router, "get_local_model_score", lambda text, history: 0.99)
    monkeypatch.setattr(router, "get_banter_signal", lambda text, history: 0.45)

    result = router.lgs_pipeline(
        conversation_id="unit-test",
        conversation_history=[],
        current_message={"sender": "user", "text": "Test message"},
    )

    assert result["source"] == "local_only"
    assert result["global_score"] is None

def _capture_fuse(monkeypatch):
    captured = {}

    def fake_fuse(**kwargs):
        captured.update(kwargs)
        return {"adjusted_score": 0.3, "flagged": False, "label": 0}

    monkeypatch.setattr(router, "fuse_scores", fake_fuse)
    return captured


def test_router_passes_no_block_reason_for_ordinary_history(
    safe_local_route, monkeypatch
):
    captured = _capture_fuse(monkeypatch)

    router.lgs_pipeline(
        conversation_id="unit-test",
        conversation_history=[{"sender": "a", "text": "hey there"}],
        current_message={"sender": "b", "text": "hello"},
    )

    assert captured["override_block_reason"] is None


def test_router_blocks_override_when_history_has_threat(
    safe_local_route, monkeypatch
):
    captured = _capture_fuse(monkeypatch)

    router.lgs_pipeline(
        conversation_id="unit-test",
        conversation_history=[{"sender": "a", "text": "i will hurt you after school"}],
        current_message={"sender": "b", "text": "haha ok"},
    )

    assert captured["override_block_reason"] == "threat_language"


def test_router_blocks_override_when_a_sender_pushed_back(
    safe_local_route, monkeypatch
):
    captured = _capture_fuse(monkeypatch)
    monkeypatch.setattr(
        router, "get_reciprocity_score",
        lambda history: (0.9, {"a": {"offensive": 1, "defensive": 0, "laughter": 1},
                               "b": {"offensive": 1, "defensive": 1, "laughter": 1}}),
    )

    router.lgs_pipeline(
        conversation_id="unit-test",
        conversation_history=[{"sender": "a", "text": "lol"}],
        current_message={"sender": "b", "text": "lol ok"},
    )

    assert captured["override_block_reason"] == "defensive_message"


def test_router_blocks_override_after_strong_offensive_history(
    safe_local_route, monkeypatch
):
    captured = _capture_fuse(monkeypatch)
    monkeypatch.setattr(
        router, "get_offensive_signal",
        lambda text: 0.9 if "STRONG" in text else 0.0,
    )

    router.lgs_pipeline(
        conversation_id="unit-test",
        conversation_history=[{"sender": "a", "text": "STRONG insult"}],
        current_message={"sender": "b", "text": "lol"},
    )

    assert captured["override_block_reason"] == "strong_offensive_history"
def test_context_disagreement_forces_global_check(safe_local_route, monkeypatch):
    # contextual local is confident, but the message alone looks the opposite
    monkeypatch.setattr(
        router, "get_local_model_score",
        lambda text, history: 0.99 if history else 0.01,
    )
    monkeypatch.setattr(router, "get_muril_score", lambda text, history: 0.5)

    result = router.lgs_pipeline(
        conversation_id="unit-test",
        conversation_history=[{"sender": "a", "text": "hi"}],
        current_message={"sender": "b", "text": "Test message"},
    )

    assert result["source"] == "global_fallback"
    assert result["router_reason"] == "context_disagreement"


def test_router_applies_own_text_floor(safe_local_route, monkeypatch):
    captured = _capture_fuse(monkeypatch)
    monkeypatch.setattr(
        router, "get_local_model_score",
        lambda text, history: 0.83 if history else 0.93,
    )
    monkeypatch.setattr(
        router, "get_muril_score",
        lambda text, history: 0.01 if history else 0.96,
    )

    result = router.lgs_pipeline(
        conversation_id="unit-test",
        conversation_history=[{"sender": "a", "text": "lmao 😂"}],
        current_message={"sender": "b", "text": "i will hurt you after school"},
    )

    assert result["own_text_reason"] == "own_text_floor"
    assert captured["local_model_score"] == pytest.approx(0.93)
    assert captured["muril_score"] == pytest.approx(0.96)


def test_router_applies_victim_cap(safe_local_route, monkeypatch):
    captured = _capture_fuse(monkeypatch)
    monkeypatch.setattr(
        router, "get_local_model_score",
        lambda text, history: 0.97 if history else 0.01,
    )
    monkeypatch.setattr(
        router, "get_muril_score",
        lambda text, history: 0.98 if history else 0.05,
    )

    result = router.lgs_pipeline(
        conversation_id="unit-test",
        conversation_history=[{"sender": "a", "text": "you are a loser"}],
        current_message={"sender": "b", "text": "please stop, that really hurts"},
    )

    assert result["own_text_reason"] == "victim_own_text"
    assert captured["local_model_score"] == pytest.approx(0.01)
    assert captured["muril_score"] == pytest.approx(0.05)


def test_router_skips_victim_cap_if_defender_was_abusive(
    safe_local_route, monkeypatch
):
    captured = _capture_fuse(monkeypatch)
    monkeypatch.setattr(
        router, "get_local_model_score",
        lambda text, history: 0.97 if history else 0.01,
    )
    monkeypatch.setattr(
        router, "get_muril_score",
        lambda text, history: 0.98 if history else 0.05,
    )
    monkeypatch.setattr(
        router, "get_reciprocity_score",
        lambda history: (
            0.5,
            {"b": {"offensive": 1, "defensive": 1, "laughter": 0}},
        ),
    )

    result = router.lgs_pipeline(
        conversation_id="unit-test",
        conversation_history=[{"sender": "b", "text": "you idiot"}],
        current_message={"sender": "b", "text": "please stop, that really hurts"},
    )

    assert result["own_text_reason"] is None
    assert captured["local_model_score"] == pytest.approx(0.97)