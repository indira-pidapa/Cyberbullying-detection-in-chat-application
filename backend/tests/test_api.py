from unittest.mock import MagicMock, patch

import transformers
from fastapi.testclient import TestClient


# Mock expensive model loading and prevent database initialization at import.
def fake_model(*args, **kwargs):
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model
    model.parameters.return_value = []
    return model


from backend import chat as chat_module
from backend import auth as auth_module

with (
    patch.object(chat_module, "init_chat_db"),
    patch.object(auth_module, "init_auth_db"),
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
    from backend import main as api


client = TestClient(api.app)


def test_root_endpoint_returns_api_information():
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "CyberGuard Cyberbullying Detection API"
    assert data["docs"] == "http://localhost:8000/docs"
    assert "classify" in data["endpoints"]


def test_health_endpoint_returns_status():
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model"] == "MuRIL + LGS-MT"
    assert data["device"] in {"cpu", "cuda"}


def test_classify_rejects_blank_message():
    response = client.post(
        "/classify",
        json={
            "conversation_id": "unit-test",
            "sender_id": "tester",
            "message": "   ",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Message cannot be empty."


def test_classify_rejects_missing_required_fields():
    response = client.post(
        "/classify",
        json={"message": "Hello"},
    )

    assert response.status_code == 422


def test_classify_returns_expected_response_and_stores_message(monkeypatch):
    history_mock = MagicMock(return_value=[])
    pipeline_mock = MagicMock(
        return_value={
            "flagged": False,
            "label": 0,
            "final_score": 0.25,
            "adjusted_score": 0.25,
            "source": "local_only",
            "local_score": 0.8,
            "global_score": None,
            "local_memory_score": None,
            "global_memory_score": None,
            "banter_signal": 0.0,
            "reciprocity_score": 0.5,
            "trajectory_score": 0.5,
            "reduction_applied": 0.0,
            "confidence": 0.6,
            "history_length": 0,
            "sender_counts": {},
            "local_memories": [],
            "global_memories": [],
        }
    )
    embedding_mock = MagicMock(return_value=[0.1, 0.2])
    store_mock = MagicMock()

    monkeypatch.setattr(api, "fetch_conversation_history", history_mock)
    monkeypatch.setattr(api, "lgs_pipeline", pipeline_mock)
    monkeypatch.setattr(api, "get_message_embedding", embedding_mock)
    monkeypatch.setattr(api, "store_message", store_mock)

    response = client.post(
        "/classify",
        json={
            "conversation_id": "unit-test",
            "sender_id": "tester",
            "message": "  Hello there  ",
            "threshold": 0.7,
            "history_limit": 5,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["flagged"] is False
    assert data["label"] == 0
    assert data["adjusted_score"] == 0.25
    assert data["action"] == "allow"
    assert data["message"]["text"] == "Hello there"
    assert data["moderation"]["source"] == "local_only"
    assert data["moderation"]["threshold_used"] == 0.7

    history_mock.assert_called_once_with("unit-test", last_n=5)
    pipeline_mock.assert_called_once()
    embedding_mock.assert_called_once_with("Hello there")
    store_mock.assert_called_once()


def test_classify_does_not_run_pipeline_for_blank_message(monkeypatch):
    pipeline_mock = MagicMock()
    monkeypatch.setattr(api, "lgs_pipeline", pipeline_mock)

    response = client.post(
        "/classify",
        json={
            "conversation_id": "unit-test",
            "sender_id": "tester",
            "message": "   ",
        },
    )

    assert response.status_code == 400
    pipeline_mock.assert_not_called()


def test_current_user_endpoint_rejects_missing_token():
    response = client.get("/auth/me")

    assert response.status_code == 401


def test_current_user_endpoint_rejects_invalid_token(monkeypatch):
    monkeypatch.setattr(api, "decode_token", lambda token: None)

    response = client.get(
        "/auth/me",
        headers={"Authorization": "Bearer invalid-test-token"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or expired token."


def test_non_admin_cannot_access_admin_endpoint(monkeypatch):
    monkeypatch.setitem(
        api.app.dependency_overrides,
        api.require_user,
        lambda: {"id": "test-user", "role": "user"},
    )

    try:
        response = client.get("/db/conversations")
        assert response.status_code == 403
        assert response.json()["detail"] == "Admin access required."
    finally:
        api.app.dependency_overrides.pop(api.require_user, None)


def test_restricted_user_cannot_send_chat_message(monkeypatch):
    monkeypatch.setitem(
        api.app.dependency_overrides,
        api.require_user,
        lambda: {"id": "restricted-user", "strikes": 3, "role": "user"},
    )

    save_mock = MagicMock()
    monkeypatch.setattr(api, "save_chat_message", save_mock)

    try:
        response = client.post(
            "/chat/send",
            json={
                "conversation_id": "unit-test",
                "receiver_id": "another-test-user",
                "message": "Hello",
            },
        )

        assert response.status_code == 403
        assert "restricted" in response.json()["detail"].lower()
        save_mock.assert_not_called()
    finally:
        api.app.dependency_overrides.pop(api.require_user, None)


def test_chat_send_rejects_blank_message_without_classification(monkeypatch):
    monkeypatch.setitem(
        api.app.dependency_overrides,
        api.require_user,
        lambda: {"id": "test-user", "strikes": 0, "role": "user"},
    )
    monkeypatch.setattr(api, "are_friends", lambda *args: True)

    pipeline_mock = MagicMock()
    save_mock = MagicMock()
    monkeypatch.setattr(api, "lgs_pipeline", pipeline_mock)
    monkeypatch.setattr(api, "save_chat_message", save_mock)

    try:
        response = client.post(
            "/chat/send",
            json={
                "conversation_id": "unit-test",
                "receiver_id": "another-test-user",
                "message": "   ",
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "Message cannot be empty."
        pipeline_mock.assert_not_called()
        save_mock.assert_not_called()
    finally:
        api.app.dependency_overrides.pop(api.require_user, None)

def test_classify_reuses_embedding_from_pipeline(monkeypatch):
    pipeline_mock = MagicMock(
        return_value={
            "flagged": False,
            "label": 0,
            "final_score": 0.1,
            "adjusted_score": 0.1,
            "source": "local_only",
            "local_score": 0.1,
            "global_score": None,
            "local_memory_score": None,
            "global_memory_score": None,
            "banter_signal": 0.0,
            "reciprocity_score": 0.5,
            "trajectory_score": 0.5,
            "reduction_applied": 0.0,
            "confidence": 0.9,
            "history_length": 0,
            "sender_counts": {},
            "local_memories": [],
            "global_memories": [],
            "embedding": [0.3, 0.4],
        }
    )
    embedding_mock = MagicMock(return_value=[9.9])
    store_mock = MagicMock()

    monkeypatch.setattr(api, "fetch_conversation_history", MagicMock(return_value=[]))
    monkeypatch.setattr(api, "lgs_pipeline", pipeline_mock)
    monkeypatch.setattr(api, "get_message_embedding", embedding_mock)
    monkeypatch.setattr(api, "store_message", store_mock)

    response = client.post(
        "/classify",
        json={
            "conversation_id": "unit-test",
            "sender_id": "tester",
            "message": "Hello there",
            "threshold": 0.7,
            "history_limit": 5,
        },
    )

    assert response.status_code == 200
    embedding_mock.assert_not_called()
    assert store_mock.call_args.kwargs["embedding"] == [0.3, 0.4]

# ---------------------------------------------------------
# STEP 8b: real chat path (/chat/send, GET /chat/{id})
# ---------------------------------------------------------

SEND_BODY = {
    "conversation_id": "conv-1",
    "receiver_id": "bob",
    "message": "hello",
}
PAIR_CONVERSATION = {"id": "conv-1", "user1_id": "alice", "user2_id": "bob"}


def _prepare_chat_send(monkeypatch, conversation, flagged=False, history=None):
    monkeypatch.setitem(
        api.app.dependency_overrides,
        api.require_user,
        lambda: {"id": "alice", "strikes": 0, "role": "user"},
    )
    monkeypatch.setattr(api, "are_friends", lambda *args: True)
    monkeypatch.setattr(api, "get_user", lambda uid: {"id": uid, "strikes": 0})
    monkeypatch.setattr(api, "get_conversation", lambda cid: conversation)
    monkeypatch.setattr(api, "get_chat_messages", lambda cid: history or [])

    pipeline_mock = MagicMock(
        return_value={
            "flagged": flagged,
            "label": 1 if flagged else 0,
            "final_score": 0.9 if flagged else 0.1,
            "embedding": [0.1, 0.2],
        }
    )
    save_mock = MagicMock(return_value={"id": "m1"})
    strike_mock = MagicMock(return_value=1)
    monkeypatch.setattr(api, "lgs_pipeline", pipeline_mock)
    monkeypatch.setattr(api, "save_chat_message", save_mock)
    monkeypatch.setattr(api, "store_message", MagicMock())
    monkeypatch.setattr(api, "add_strike", strike_mock)
    return pipeline_mock, save_mock, strike_mock


def test_chat_send_rejects_conversation_of_other_users(monkeypatch):
    foreign = {"id": "conv-1", "user1_id": "carol", "user2_id": "dave"}
    pipeline_mock, save_mock, strike_mock = _prepare_chat_send(monkeypatch, foreign)

    try:
        response = client.post("/chat/send", json=SEND_BODY)

        assert response.status_code == 403
        pipeline_mock.assert_not_called()
        save_mock.assert_not_called()
        strike_mock.assert_not_called()
    finally:
        api.app.dependency_overrides.pop(api.require_user, None)


def test_chat_send_rejects_unknown_conversation(monkeypatch):
    pipeline_mock, save_mock, _ = _prepare_chat_send(monkeypatch, None)

    try:
        response = client.post("/chat/send", json=SEND_BODY)

        assert response.status_code == 403
        pipeline_mock.assert_not_called()
        save_mock.assert_not_called()
    finally:
        api.app.dependency_overrides.pop(api.require_user, None)


def test_chat_send_passes_last_five_messages_in_order_with_real_sender_ids(
    monkeypatch,
):
    history = [
        {"sender_id": "alice" if i % 2 == 0 else "bob", "message": f"m{i}"}
        for i in range(7)
    ]
    pipeline_mock, _, strike_mock = _prepare_chat_send(
        monkeypatch, PAIR_CONVERSATION, history=history
    )

    try:
        response = client.post("/chat/send", json=SEND_BODY)

        assert response.status_code == 200
        kwargs = pipeline_mock.call_args.kwargs
        assert kwargs["conversation_id"] == "conv-1"
        assert kwargs["current_message"] == {"sender": "alice", "text": "hello"}
        assert kwargs["conversation_history"] == [
            {"sender": "alice", "text": "m2"},
            {"sender": "bob", "text": "m3"},
            {"sender": "alice", "text": "m4"},
            {"sender": "bob", "text": "m5"},
            {"sender": "alice", "text": "m6"},
        ]
        strike_mock.assert_not_called()
        assert response.json()["moderation"]["strikes"] == 0
    finally:
        api.app.dependency_overrides.pop(api.require_user, None)


def test_chat_send_adds_a_strike_only_for_flagged_messages(monkeypatch):
    _, _, strike_mock = _prepare_chat_send(
        monkeypatch, PAIR_CONVERSATION, flagged=True
    )

    try:
        response = client.post("/chat/send", json=SEND_BODY)

        assert response.status_code == 200
        strike_mock.assert_called_once_with("alice")
        moderation = response.json()["moderation"]
        assert moderation["flagged"] is True
        assert moderation["action"] == "flag"
        assert moderation["strikes"] == 1
    finally:
        api.app.dependency_overrides.pop(api.require_user, None)


def test_get_chat_rejects_non_member(monkeypatch):
    monkeypatch.setitem(
        api.app.dependency_overrides,
        api.require_user,
        lambda: {"id": "mallory", "strikes": 0, "role": "user"},
    )
    messages_mock = MagicMock(return_value=[{"message": "private"}])
    monkeypatch.setattr(api, "get_conversation", lambda cid: PAIR_CONVERSATION)
    monkeypatch.setattr(api, "get_chat_messages", messages_mock)

    try:
        response = client.get("/chat/conv-1")

        assert response.status_code == 403
        messages_mock.assert_not_called()
    finally:
        api.app.dependency_overrides.pop(api.require_user, None)


def test_get_chat_allows_member(monkeypatch):
    monkeypatch.setitem(
        api.app.dependency_overrides,
        api.require_user,
        lambda: {"id": "bob", "strikes": 0, "role": "user"},
    )
    monkeypatch.setattr(api, "get_conversation", lambda cid: PAIR_CONVERSATION)
    monkeypatch.setattr(
        api, "get_chat_messages", lambda cid: [{"message": "hi", "sender_id": "alice"}]
    )

    try:
        response = client.get("/chat/conv-1")

        assert response.status_code == 200
        assert response.json()["messages"][0]["message"] == "hi"
    finally:
        api.app.dependency_overrides.pop(api.require_user, None)
