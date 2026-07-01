"""
retriever.py — Local Semantic Search Engine for SHL Assessment Catalog

Uses sentence-transformers (all-MiniLM-L6-v2) to embed every assessment in
catalog.json and builds a FAISS index for instant similarity search.

The model and index are loaded exactly ONCE via a module-level singleton,
so all subsequent searches are sub-millisecond.
"""

import json
import os
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Path to the catalog file (sits in the project root alongside this file)
# ---------------------------------------------------------------------------
_CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catalog.json")

# ---------------------------------------------------------------------------
# Embedding model — small, fast, and accurate for semantic similarity
# ---------------------------------------------------------------------------
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Module-level singleton (lazy init via get_retriever())
# ---------------------------------------------------------------------------
_retriever_instance = None


class AssessmentRetriever:
    """Retriever that loads the catalog, embeds it, and
    exposes a fast FAISS-backed search function.

    Usage:
        retriever = get_retriever()                # loads once
        results   = retriever.search("leadership") # instant lookup
    """

    def __init__(self) -> None:
        # 1. Load catalog -------------------------------------------------
        self.catalog: List[Dict[str, Any]] = self._load_catalog()

        # 2. Load embedding model -----------------------------------------
        print(f"[retriever] Loading model '{_MODEL_NAME}'...")
        self.model: SentenceTransformer = SentenceTransformer(_MODEL_NAME)

        # 3. Build FAISS index --------------------------------------------
        self.index: faiss.IndexFlatIP = self._build_index()

        # 4. Build lookup caches for validation ----------------------------
        self._name_lookup: Dict[str, Dict] = {
            item["name"].lower(): item for item in self.catalog
        }
        self._link_lookup: Dict[str, Dict] = {
            item.get("link", "").lower(): item for item in self.catalog
        }

        print(
            f"[retriever] Ready — {len(self.catalog)} assessments indexed "
            f"with model '{_MODEL_NAME}'"
        )

    # ---- Public API ----------------------------------------------------------
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Return the *top_k* catalog entries most similar to *query*.

        Each returned item is the full original JSON dictionary from
        catalog.json, enriched with a ``_score`` key (cosine similarity).
        """
        query_vector = self.model.encode([query], normalize_embeddings=True)
        query_vector = np.array(query_vector, dtype="float32")

        scores, indices = self.index.search(query_vector, min(top_k, len(self.catalog)))

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self.catalog[idx])
            entry["_score"] = round(float(score), 4)
            results.append(entry)

        return results

    def get_by_name(self, name: str) -> Dict[str, Any] | None:
        """Exact (case-insensitive) lookup by assessment name."""
        return self._name_lookup.get(name.lower())

    def get_by_link(self, link: str) -> Dict[str, Any] | None:
        """Exact (case-insensitive) lookup by assessment link/URL."""
        return self._link_lookup.get(link.lower())

    def get_all_keys(self) -> List[str]:
        """Return all unique assessment category keys in the catalog."""
        keys_set: set[str] = set()
        for item in self.catalog:
            for k in item.get("keys", []):
                keys_set.add(k)
        return sorted(keys_set)

    # ---- Private helpers -----------------------------------------------------
    @staticmethod
    def _load_catalog() -> List[Dict[str, Any]]:
        """Read and parse catalog.json."""
        if not os.path.exists(_CATALOG_PATH):
            raise FileNotFoundError(
                f"Catalog file not found at {_CATALOG_PATH}. "
                "Please place catalog.json in the project root."
            )
        with open(_CATALOG_PATH, "r", encoding="utf-8") as fh:
            raw = fh.read()
        # strict=False handles control characters (\r\n, \t) inside JSON strings
        data = json.loads(raw, strict=False)

        if not isinstance(data, list):
            raise ValueError("catalog.json must contain a JSON array of assessment objects.")

        print(f"[retriever] Loaded {len(data)} assessments from catalog.json")
        return data

    def _build_index(self) -> faiss.IndexFlatIP:
        """Embed every catalog entry and add vectors to a FAISS index."""
        texts = [self._assessment_to_text(item) for item in self.catalog]
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype="float32")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        print(f"[retriever] FAISS index built — {index.ntotal} vectors, dim={dimension}")
        return index

    @staticmethod
    def _assessment_to_text(item: Dict[str, Any]) -> str:
        """Combine an assessment's fields into a single string for embedding."""
        parts: List[str] = []

        if name := item.get("name"):
            parts.append(str(name))

        if description := item.get("description"):
            parts.append(str(description))

        if job_levels := item.get("job_levels"):
            if isinstance(job_levels, list) and job_levels:
                parts.append("Job Levels: " + ", ".join(str(jl) for jl in job_levels))

        if keys := item.get("keys"):
            if isinstance(keys, list) and keys:
                parts.append("Test Type: " + ", ".join(str(k) for k in keys))

        if languages := item.get("languages"):
            if isinstance(languages, list) and languages:
                parts.append("Languages: " + ", ".join(str(l) for l in languages))

        if duration := item.get("duration"):
            parts.append("Duration: " + str(duration))

        if item.get("remote") == "yes":
            parts.append("Remote: Yes")
        if item.get("adaptive") == "yes":
            parts.append("Adaptive: Yes")

        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------
def get_retriever() -> AssessmentRetriever:
    """Get or create the singleton AssessmentRetriever instance.

    Thread-safe enough for a single-process ASGI server (uvicorn).
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = AssessmentRetriever()
    return _retriever_instance


def search_assessments(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Convenience wrapper — gets the singleton retriever and searches."""
    return get_retriever().search(query, top_k)


# ---------------------------------------------------------------------------
# Quick smoke test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running smoke test...\n")

    results = search_assessments("Java developer mid-level", top_k=5)
    print("Query: 'Java developer mid-level'")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['_score']:.4f}]  {r.get('name', 'N/A')}")
        print(f"     URL:  {r.get('link', 'N/A')}")
        print(f"     Type: {r.get('keys', [])}")
    print(f"\nReturned {len(results)} results.")

    retriever = get_retriever()
    print(f"\nAll assessment categories: {retriever.get_all_keys()}")
