import sqlite3
import uuid
from datetime import datetime, timezone

import pytest

from backend import chat


@pytest.fixture
def temp_chat_db(tmp_path, monkeypatch):
    """Use an isolated SQLite database for each test."""
    db_path = tmp_path / "test_users.db"

    def temporary_connection():
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn

    monkeypatch.setattr(chat, "get_connection", temporary_connection)
    chat.init_chat_db()

    # The stats query joins against the users table.
    conn = temporary_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

    return temporary_connection


def test_get_or_create_conversation_creates_record(temp_chat_db):
    conversation = chat.get_or_create_conversation("user-a", "user-b")

    assert conversation["id"]
    assert {conversation["user1_id"], conversation["user2_id"]} == {
        "user-a", "user-b"
    }
    assert conversation["created_at"]


def test_conversation_is_reused_regardless_of_user_order(temp_chat_db):
    first = chat.get_or_create_conversation("user-a", "user-b")
    second = chat.get_or_create_conversation("user-b", "user-a")

    assert first["id"] == second["id"]


def test_save_chat_message_persists_expected_fields(temp_chat_db):
    conversation = chat.get_or_create_conversation("user-a", "user-b")

    saved = chat.save_chat_message(
        conversation_id=conversation["id"],
        sender_id="user-a",
        receiver_id="user-b",
        message="Test message",
        flagged=True,
        score=0.91,
    )

    assert saved["id"]
    assert saved["conversation_id"] == conversation["id"]
    assert saved["sender_id"] == "user-a"
    assert saved["receiver_id"] == "user-b"
    assert saved["message"] == "Test message"
    assert saved["flagged"] is True
    assert saved["score"] == pytest.approx(0.91)

    rows = chat.get_chat_messages(conversation["id"])
    assert len(rows) == 1
    assert rows[0]["message"] == "Test message"
    assert rows[0]["flagged"] is True
    assert rows[0]["score"] == pytest.approx(0.91)


def test_get_chat_messages_returns_empty_for_unknown_conversation(temp_chat_db):
    assert chat.get_chat_messages("missing-conversation") == []


def test_messages_are_returned_in_chronological_order(temp_chat_db):
    conversation = chat.get_or_create_conversation("user-a", "user-b")

    chat.save_chat_message(
        conversation["id"], "user-a", "user-b", "First message"
    )
    chat.save_chat_message(
        conversation["id"], "user-b", "user-a", "Second message"
    )

    messages = chat.get_chat_messages(conversation["id"])

    assert [message["message"] for message in messages] == [
        "First message", "Second message"
    ]


def test_admin_stats_count_messages_and_flags(temp_chat_db):
    conversation = chat.get_or_create_conversation("user-a", "user-b")

    conn = temp_chat_db()
    conn.executemany(
        "INSERT INTO users (id, username) VALUES (?, ?)",
        [("user-a", "alice"), ("user-b", "bob")],
    )
    conn.commit()
    conn.close()

    chat.save_chat_message(
        conversation["id"], "user-a", "user-b", "Safe message",
        flagged=False, score=0.1,
    )
    chat.save_chat_message(
        conversation["id"], "user-b", "user-a", "Flagged message",
        flagged=True, score=0.95,
    )

    stats = chat.get_admin_stats()

    assert stats["total_conversations"] == 1
    assert stats["total_messages"] == 2
    assert stats["flagged_messages"] == 1
    assert len(stats["conversations"]) == 1
    assert stats["conversations"][0]["user1"] == "alice"
    assert stats["conversations"][0]["user2"] == "bob"

def test_get_conversation_returns_record_or_none(temp_chat_db):
    created = chat.get_or_create_conversation("user-a", "user-b")

    found = chat.get_conversation(created["id"])

    assert found["id"] == created["id"]
    assert {found["user1_id"], found["user2_id"]} == {"user-a", "user-b"}
    assert chat.get_conversation("missing-conversation") is None
