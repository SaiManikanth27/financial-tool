"""
embedder.py
-----------
Builds a FAISS vector index from text chunks using sentence-transformers.
"""

from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None   # lazy-load so startup is fast


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def build_index(chunks: List[str]) -> faiss.IndexFlatL2:
    """Embed all chunks and build a FAISS index."""
    model = _get_model()
    embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def search_index(
    query: str, index: faiss.IndexFlatL2, chunks: List[str], top_k: int = 8
) -> List[str]:
    """Return the top-k most relevant chunks for a query."""
    model = _get_model()
    q_emb = model.encode([query], convert_to_numpy=True).astype(np.float32)
    _, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]
