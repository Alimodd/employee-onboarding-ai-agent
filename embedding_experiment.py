"""Standalone educational experiment: text embeddings and semantic similarity.

This script is intentionally separate from the FastAPI chat flow. It:

    1. Loads the API key from the existing environment configuration.
    2. Defines ~10 short sentences.
    3. Creates an embedding for each sentence using the installed google-genai SDK.
    4. Picks one sentence as the query.
    5. Computes cosine similarity between the query and every other sentence.
    6. Sorts the results from most to least similar.
    7. Prints an aligned rank / score / sentence table.

It does NOT build a vector database, store embeddings, or implement RAG. It is a
demonstration of what embeddings are and how semantic similarity behaves,
including the classic limitation around negation.

Run with:  python embedding_experiment.py
"""

from __future__ import annotations

import model_client
from similarity import cosine_similarity

# Current, non-deprecated embedding model in the installed google-genai SDK.
EMBED_MODEL = "gemini-embedding-001"

# ~10 short sentences. Several share keywords but differ in meaning, and one is a
# negation ("cannot work remotely") that shares most words with a positive one.
SENTENCES = [
    "Employees receive twenty annual leave days.",        # 0 (query)
    "Workers have twenty vacation days each year.",        # 1 same meaning, few shared words
    "Employees can work remotely two days per week.",      # 2 different topic
    "Remote work is permitted twice a week.",              # 3 paraphrase of 2
    "Employees must report sickness before 9 AM.",         # 4 different topic
    "The company provides free lunch.",                    # 5 unrelated
    "The weather is sunny today.",                         # 6 unrelated
    "Python is a programming language.",                   # 7 unrelated
    "Employees cannot work remotely.",                     # 8 NEGATION example
    "How many vacation days do employees receive?",        # 9 question about leave
]

QUERY_INDEX = 0  # "Employees receive twenty annual leave days."


def embed_sentences(sentences: list[str]) -> list[list[float]]:
    """Return one embedding vector per sentence using the Gemini embedding model.

    Args:
        sentences: Non-empty list of non-empty strings.

    Returns:
        A list of float vectors, aligned by index with ``sentences``.

    Raises:
        ValueError: If input is empty/invalid, or the response shape is unexpected
            (e.g. wrong count, empty vectors, or mismatched dimensions).
        model_client.ConfigurationError: If the API key is missing.
        model_client.ProviderRequestError: If the embedding API call fails.
        model_client.EmptyModelResponseError: If the API returns no embeddings.
    """
    if not sentences or any(not s or not s.strip() for s in sentences):
        raise ValueError("All sentences must be non-empty, non-whitespace strings.")

    # Reuse the model client's key loading and client construction (no duplication).
    api_key = model_client._get_api_key()
    client = model_client._build_client(api_key)

    try:
        response = client.models.embed_content(model=EMBED_MODEL, contents=sentences)
    except Exception as exc:  # noqa: BLE001 - normalise into the known hierarchy
        raise model_client.ProviderRequestError("Embedding request failed.") from exc

    embeddings = getattr(response, "embeddings", None)
    if not embeddings:
        raise model_client.EmptyModelResponseError("Embedding API returned no data.")
    if len(embeddings) != len(sentences):
        raise ValueError(
            f"Unexpected embedding count: got {len(embeddings)}, "
            f"expected {len(sentences)}."
        )

    vectors: list[list[float]] = []
    dimension: int | None = None
    for item in embeddings:
        values = getattr(item, "values", None)
        if not values:
            raise ValueError("Unexpected embedding format: missing values.")
        if dimension is None:
            dimension = len(values)
        elif len(values) != dimension:
            raise ValueError(
                f"Embedding dimension mismatch: {len(values)} vs {dimension}."
            )
        vectors.append(list(values))

    return vectors


def rank_by_similarity(
    vectors: list[list[float]], query_index: int
) -> list[tuple[int, float]]:
    """Rank every sentence (except the query) by cosine similarity to the query.

    Args:
        vectors: One embedding vector per sentence.
        query_index: Index of the sentence used as the query.

    Returns:
        A list of ``(sentence_index, similarity)`` tuples, sorted from highest to
        lowest similarity. The query itself is excluded from the results.
    """
    query_vec = vectors[query_index]
    scored = [
        (i, cosine_similarity(query_vec, vec))
        for i, vec in enumerate(vectors)
        if i != query_index
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def main() -> None:
    try:
        vectors = embed_sentences(SENTENCES)
    except model_client.ConfigurationError as exc:
        print(f"[configuration error] {exc}")
        return
    except model_client.ModelClientError as exc:
        print(f"[embedding error] {exc}")
        return
    except ValueError as exc:
        print(f"[validation error] {exc}")
        return

    ranked = rank_by_similarity(vectors, QUERY_INDEX)

    print(f"Query sentence: \"{SENTENCES[QUERY_INDEX]}\"")
    print(f"Embedding model: {EMBED_MODEL}  |  vector dimension: {len(vectors[0])}")
    print()
    print(f"{'Rank':<5}{'Score':<9}Sentence")
    print("-" * 70)
    for rank, (idx, score) in enumerate(ranked, start=1):
        print(f"{rank:<5}{score:<9.4f}{SENTENCES[idx]}")

    print()
    print("Note: the negation sentence (\"Employees cannot work remotely.\") often")
    print("scores high against remote-work sentences because it shares most words,")
    print("even though it means the opposite. High similarity != same meaning.")


if __name__ == "__main__":
    main()
