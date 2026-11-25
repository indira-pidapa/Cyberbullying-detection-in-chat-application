
import pytest
import chromadb

from backend import database


@pytest.fixture
def isolated_chroma(monkeypatch):
    """Use a fresh, isolated in-memory ChromaDB client."""
    client = chromadb.EphemeralClient()

    # Clear collections before each test to guarantee isolation.
    for collection in client.list_collections():
        client.delete_collection(collection.name)

    monkeypatch.setattr(database, "chroma_client", client)
    return client


@pytest.fixture
def sample_embedding():
    # Small embedding is sufficient for database unit tests.
    return [1.0, 0.0, 0.0]


def test_store_message_returns_id(
    isolated_chroma, sample_embedding
):
    message_id = database.store_message(
        conversation_id="test-conv",
        sender_id="user-a",
        message_text="Hello there",
        embedding=sample_embedding,
        muril_score=0.12,
        adjusted_score=0.15,
        flagged=False,
    )

    assert isinstance(message_id, str)
    assert len(message_id) > 0
    assert database.get_collection_if_exists("test-conv").count() == 1


def test_fetch_history_returns_expected_message(
    isolated_chroma, sample_embedding
):
    database.store_message(
        "history-test", "user-a", "Hello",
        sample_embedding, flagged=False
    )

    history = database.fetch_conversation_history("history-test")

    assert len(history) == 1
    assert history[0]["sender"] == "user-a"
    assert history[0]["text"] == "Hello"
    assert history[0]["flagged"] is False
    assert history[0]["timestamp"]


def test_fetch_history_returns_empty_for_missing_conversation(
    isolated_chroma
):
    assert database.fetch_conversation_history("missing") == []


def test_fetch_history_respects_last_n(
    isolated_chroma, sample_embedding
):
    for i in range(4):
        database.store_message(
            "limited-history", f"user-{i}",
            f"Message {i}", sample_embedding
        )

    history = database.fetch_conversation_history(
        "limited-history", last_n=2
    )

    assert len(history) == 2
    assert [message["text"] for message in history] == [
        "Message 2", "Message 3"
    ]


def test_get_all_messages_preserves_scores_and_flags(
    isolated_chroma, sample_embedding
):
    database.store_message(
        "all-messages", "user-a", "Potential bullying",
        sample_embedding,
        muril_score=0.91,
        adjusted_score=0.95,
        banter_signal=0.1,
        reciprocity_score=0.2,
        reduction_applied=0.05,
        flagged=True,
    )

    messages = database.get_all_messages("all-messages")

    assert len(messages) == 1
    message = messages[0]
    assert message["text"] == "Potential bullying"
    assert message["sender_id"] == "user-a"
    assert message["muril_score"] == pytest.approx(0.91)
    assert message["adjusted_score"] == pytest.approx(0.95)
    assert message["banter_signal"] == pytest.approx(0.1)
    assert message["reciprocity_score"] == pytest.approx(0.2)
    assert message["reduction_applied"] == pytest.approx(0.05)
    assert message["flagged"] is True
    assert message["message_number"] == 1


def test_get_all_messages_returns_empty_for_missing_conversation(
    isolated_chroma
):
    assert database.get_all_messages("missing") == []


def test_list_conversations_and_stats(
    isolated_chroma, sample_embedding
):
    database.store_message(
        "stats-a", "user-a", "Safe message",
        sample_embedding, flagged=False
    )
    database.store_message(
        "stats-a", "user-b", "Flagged message",
        sample_embedding, flagged=True
    )
    database.store_message(
        "stats-b", "user-c", "Another message",
        sample_embedding, flagged=False
    )

    conversations = database.list_all_conversations()
    by_id = {c["conversation_id"]: c for c in conversations}

    assert len(by_id) == 2
    assert by_id["stats-a"]["total_messages"] == 2
    assert by_id["stats-a"]["flagged_messages"] == 1
    assert by_id["stats-a"]["flag_rate"] == 0.5
    assert set(by_id["stats-a"]["participants"]) == {
        "user-a", "user-b"
    }

    stats = database.get_stats()

    assert stats["total_conversations"] == 2
    assert stats["total_messages"] == 3
    assert stats["total_flagged"] == 1
    assert stats["flag_rate"] == pytest.approx(1 / 3, abs=0.0001)


def test_delete_conversation_removes_collection(
    isolated_chroma, sample_embedding
):
    database.store_message(
        "delete-test", "user-a", "Temporary message",
        sample_embedding
    )

    assert database.delete_conversation("delete-test") is True
    assert database.get_collection_if_exists("delete-test") is None


def test_delete_missing_conversation_returns_false(isolated_chroma):
    assert database.delete_conversation("does-not-exist") is False


def test_global_memory_store_and_search(
    isolated_chroma, sample_embedding
):
    pattern_id = database.store_global_pattern(
        embedding=sample_embedding,
        pattern_text="Repeated targeted insults",
        label=1,
        confidence=0.97,
        language="English",
        pattern_type="bullying",
    )

    assert isinstance(pattern_id, str)
    assert pattern_id

    results = database.search_global_memory(
        embedding=sample_embedding,
        n_results=5,
    )

    assert len(results) == 1
    assert results[0]["pattern"] == "Repeated targeted insults"
    assert results[0]["metadata"]["label"] == "1"
    assert float(results[0]["metadata"]["confidence"]) == pytest.approx(0.97)
    assert results[0]["similarity"] == pytest.approx(1.0, abs=0.01)


def test_global_memory_search_returns_empty_when_no_patterns(
    isolated_chroma, sample_embedding
):
    results = database.search_global_memory(
        embedding=sample_embedding,
        n_results=5,
    )

    assert results == []
