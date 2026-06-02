"""
answerer.py
-----------
Takes a user question, retrieves relevant stock data chunks via FAISS,
and generates a natural language answer using the Groq LLM (free tier).
"""

import os
from typing import List, Tuple
import faiss
from groq import Groq
from rag.embedder import search_index

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _client


_SYSTEM_PROMPT = """You are a helpful stock market analyst assistant.

You are given real historical stock data for a specific ticker as context.
Answer the user's question based ONLY on the provided data.

Rules:
- Be concise and specific. Use exact numbers from the data when relevant.
- If the data doesn't contain enough info to answer precisely, say so.
- Use plain English — no jargon unless the user asks for it.
- Format numbers cleanly: prices as $X.XX, percentages as X.X%, volumes as XM."""


def answer_question(
    question: str,
    index: faiss.IndexFlatL2,
    chunks: List[str],
    ticker: str,
    top_k: int = 8,
) -> Tuple[str, List[str]]:
    """
    Full RAG pipeline:
      1. Retrieve top-k relevant stock data chunks via FAISS
      2. Pass to Groq LLM with the question
      3. Return (answer, source_snippets)
    """
    relevant_chunks = search_index(question, index, chunks, top_k=top_k)
    context = "\n\n".join(relevant_chunks)

    user_msg = (
        f"Stock: {ticker}\n\n"
        f"Historical data context:\n{context}\n\n"
        f"Question: {question}"
    )

    response = _get_client().chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=400,
    )

    answer = response.choices[0].message.content.strip()
    sources = relevant_chunks[:3]   # Show top 3 sources in the UI
    return answer, sources
