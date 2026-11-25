from memory import (
    search_local_memory,
    search_global_memory,
    get_global_memory
)

from pipeline import get_message_embedding


# ============================================================
# TEST LOCAL MEMORY
# ============================================================

conversation_id = "test_lgsmt"

query = "you're stupid"

embedding = get_message_embedding(query)

local_results = search_local_memory(
    conversation_id=conversation_id,
    query_embedding=embedding,
    n_results=5
)

print("\n==============================")
print("LOCAL MEMORY")
print("==============================")

print(local_results)


# ============================================================
# TEST GLOBAL MEMORY
# ============================================================

global_results = search_global_memory(
    query_embedding=embedding,
    n_results=5
)

print("\n==============================")
print("GLOBAL MEMORY")
print("==============================")

print(global_results)