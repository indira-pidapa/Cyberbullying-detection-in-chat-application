from backend.lgs_pipeline import lgs_pipeline


print("=" * 60)
print("TESTING LGS-MT")
print("=" * 60)


# ------------------------------------------------------------
# TEST CONVERSATION
# ------------------------------------------------------------

conversation_id = "test_lgs_001"

history = [
    {
        "sender": "user_A",
        "text": "Hey, how are you?"
    },
    {
        "sender": "user_B",
        "text": "I'm good 😂"
    },
    {
        "sender": "user_A",
        "text": "Haha nice"
    }
]


current_message = {
    "sender": "user_B",
    "text": "wow you really did that huh"   # sarcastic/ambiguous, no clear signal
}


# ------------------------------------------------------------
# RUN LGS
# ------------------------------------------------------------

result = lgs_pipeline(
    conversation_id=conversation_id,
    conversation_history=history,
    current_message=current_message
)


# ------------------------------------------------------------
# PRINT RESULT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("LGS-MT RESULT")
print("=" * 60)

print(
    "Global score:",
    result["global_score"]
)

print(
    "Local score:",
    result["local_score"]
)

print(
    "Local memory score:",
    result["local_memory_score"]
)

print(
    "Global memory score:",
    result["global_memory_score"]
)

print(
    "Final score:",
    result["final_score"]
)

print(
    "Threshold:",
    result["threshold"]
)

print(
    "Flagged:",
    result["flagged"]
)

print(
    "Label:",
    result["label"]
)

print(
    "Local memories retrieved:",
    len(result["local_memories"])
)

print(
    "Global memories retrieved:",
    len(result["global_memories"])
)

print("=" * 60)