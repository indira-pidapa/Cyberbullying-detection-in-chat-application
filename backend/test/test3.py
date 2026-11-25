from backend.lgs_pipeline import lgs_pipeline

"""multi-message context test"""
conversation_id = "test_context_001"

history = []


messages = [
    ("user_A", "Hey, how are you?"),
    ("user_B", "I'm good 😂"),
    ("user_A", "Haha nice"),
    ("user_B", "You're so stupid 😂"),
    ("user_A", "Bro stop"),
    ("user_B", "Whatever loser"),
]


print("\n" + "=" * 70)
print("LGS-MT MULTI-MESSAGE CONTEXT TEST")
print("=" * 70)


for i, (sender, text) in enumerate(messages, start=1):

    current_message = {
        "sender": sender,
        "text": text,
        "conversation_id": conversation_id
    }

    result = lgs_pipeline(
        conversation_id=conversation_id,
        conversation_history=history,
        current_message=current_message
    )

    print("\n" + "-" * 70)
    print(f"MESSAGE {i}")
    print("-" * 70)

    print("Sender:", sender)
    print("Text:", text)

    print()
    print("Source:", result["source"])
    print("Confidence:", result["confidence"])

    print("Local score:", result["local_score"])
    print("Global score:", result["global_score"])

    print("Local memory score:", result["local_memory_score"])
    print("Global memory score:", result["global_memory_score"])

    print("Banter:", result["banter_signal"])
    print("Reciprocity:", result["reciprocity_score"])
    print("Trajectory:", result["trajectory_score"])

    print("Reduction:", result["reduction_applied"])
    print("Final score:", result["final_score"])

    print("Flagged:", result["flagged"])
    print("Label:", result["label"])

    print("History length:", result["history_length"])
    print("Local memories:", len(result["local_memories"]))
    print("Global memories:", len(result["global_memories"]))

    # Add current message to history
    history.append(current_message)


print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)