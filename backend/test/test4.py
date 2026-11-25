from backend.pipeline import get_message_embedding
from backend.memory import add_global_pattern
from backend.lgs_pipeline import lgs_pipeline


print("=" * 70)
print("LGS-MT GLOBAL MEMORY TEST")
print("=" * 70)


# ------------------------------------------------------------
# 1. CREATE A HARMFUL GLOBAL MEMORY
# ------------------------------------------------------------

pattern_text = "insulting someone by calling them stupid and worthless"

embedding = get_message_embedding(pattern_text)

memory_id = add_global_pattern(
    pattern_id="test_harmful_pattern_001",
    embedding=embedding,
    label=1,
    pattern_type="insult"
)

print("\nStored global memory:")
print("Memory ID:", memory_id)


# ------------------------------------------------------------
# 2. TEST CONVERSATION
# ------------------------------------------------------------

conversation_id = "memory_test_001"

history = [
    {
        "sender": "user_A",
        "text": "You keep making mistakes."
    },
    {
        "sender": "user_B",
        "text": "Whatever."
    }
]

current_message = {
    "sender": "user_A",
    "text": "You're stupid and worthless."
}

# ------------------------------------------------------------
# DEBUG: inspect raw global memory retrieval
# ------------------------------------------------------------
from .memory import search_global_memory

debug_embedding = get_message_embedding(current_message["text"])
raw_global = search_global_memory(query_embedding=debug_embedding, n_results=5)

print("\nRAW GLOBAL MEMORIES:")
for m in raw_global:
    print(m, "| label type:", type(m.get("label")))
# ------------------------------------------------------------
# 3. RUN LGS-MT
# ------------------------------------------------------------

result = lgs_pipeline(
    conversation_id=conversation_id,
    conversation_history=history,
    current_message=current_message
)


# ------------------------------------------------------------
# 4. PRINT RESULT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LGS-MT RESULT")
print("=" * 70)

print("Source:", result["source"])
print("Router reason:", result.get("router_reason"))

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
print("Threshold:", result["threshold"])

print("Flagged:", result["flagged"])
print("Label:", result["label"])

print("Local memories:", len(result["local_memories"]))
print("Global memories:", len(result["global_memories"]))

print("=" * 70)