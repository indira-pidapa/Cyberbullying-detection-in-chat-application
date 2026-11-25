from database import (
    get_global_memory_collection,
    store_global_pattern,
    search_global_memory
)

print("\n--- GLOBAL MEMORY TEST ---")

# 1. Get/create collection
collection = get_global_memory_collection()

print("Collection:", collection.name)
print("Initial count:", collection.count())


# 2. Fake embedding for testing
# MuRIL embeddings should eventually be 768-dimensional.
test_embedding = [0.01] * 768


# 3. Store a test generalized pattern
pattern_id = store_global_pattern(
    embedding=test_embedding,
    pattern_text="general insulting language pattern",
    label=1,
    confidence=0.92,
    language="english",
    pattern_type="insult"
)

print("Stored pattern:", pattern_id)


# 4. Search Global Memory
results = search_global_memory(
    embedding=test_embedding,
    n_results=3
)

print("\nSearch results:")

for result in results:
    print(result)