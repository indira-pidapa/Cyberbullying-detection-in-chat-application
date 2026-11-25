import backend.pipeline as pipeline
#test1
#tests for both routing paths
"""                    MESSAGE
                       │
                       ▼
                LOCAL MODEL
                       │
              ┌────────┴────────┐
              │                 │
        confidence ≥ 0.75   confidence < 0.75
              │                 │
              ▼                 ▼
         LOCAL ONLY       GLOBAL FALLBACK
              │                 │
              │          Global MuRIL
              │          Global Memory
              │                 │
              └────────┬────────┘
                       ▼
                  SCORE FUSION
                       │
                       ▼
                 FINAL DECISION"""
# Force local model to be uncertain
pipeline.get_local_model_score = lambda *args, **kwargs: 0.55

from backend.lgs_pipeline import lgs_pipeline

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
    "text": "wow you really did that huh"
}

result = lgs_pipeline(
    conversation_id="test_global_fallback",
    conversation_history=history,
    current_message=current_message
)

print("=" * 60)
print("LGS-MT RESULT")
print("=" * 60)
print("Source:", result["source"])
print("Confidence:", result["confidence"])
print("Local score:", result["local_score"])
print("Global score:", result["global_score"])
print("Local memory score:", result["local_memory_score"])
print("Global memory score:", result["global_memory_score"])
print("Final score:", result["final_score"])
print("Flagged:", result["flagged"])
print("Label:", result["label"])
print("Local memories:", len(result["local_memories"]))
print("Global memories:", len(result["global_memories"]))
print("=" * 60)