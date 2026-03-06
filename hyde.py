"""
VINE-Agent: HyDE (Hypothetical Document Embeddings) Query Expander
Bridges the semantic gap between lay farmer language and technical agronomic docs.

Reference: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al., 2023)
"""

from __future__ import annotations
import logging
from typing import List
import numpy as np

from prompts import HYDE_PROMPT

logger = logging.getLogger(__name__)


class HyDEExpander:
    """
    HyDE Query Expander for the VINE agricultural knowledge base.

    Instead of embedding the raw farmer query, we:
      1. Ask an LLM to write a *hypothetical* agronomic answer
      2. Embed that hypothetical answer
      3. Use the resulting vector for semantic retrieval

    This shifts retrieval from query→document to answer→answer similarity,
    which dramatically improves recall for domain-specific queries.

    Example:
      Raw query:  "my grapes look pale and drooping"
      Hypothetical: "Pale, drooping grapevine leaves are frequently caused by
                     iron chlorosis, high soil pH (>7.5), waterlogging of the
                     root zone, or boron toxicity. In drip-irrigated Vitis
                     vinifera vineyards, check soil EC and pH first …"
    """

    def __init__(self, llm, embed_model, n_hypotheticals: int = 1):
        """
        Args:
            llm:              Any LangChain-compatible LLM (Groq, vLLM, OpenAI, etc.)
            embed_model:      LangChain embedding model (HuggingFaceEmbeddings, etc.)
            n_hypotheticals:  Number of hypotheticals to generate & average.
                              1 is fast; 3–5 gives more robust coverage.
        """
        self.llm = llm
        self.embed_model = embed_model
        self.n_hypotheticals = n_hypotheticals

    def _generate_hypothetical(self, query: str) -> str:
        """Generate one hypothetical agronomic answer for the query."""
        prompt = HYDE_PROMPT.format_messages(query=query)
        response = self.llm.invoke(prompt)
        # Handle both string returns and AIMessage objects
        if hasattr(response, "content"):
            return response.content.strip()
        return str(response).strip()

    def expand(self, query: str) -> np.ndarray:
        """
        Returns the HyDE embedding vector for the query.

        If n_hypotheticals > 1, generates multiple hypothetical documents
        and returns their mean embedding (ensemble HyDE).
        """
        hypotheticals = []
        for _ in range(self.n_hypotheticals):
            hyp = self._generate_hypothetical(query)
            hypotheticals.append(hyp)
            logger.debug(f"HyDE hypothetical: {hyp[:120]}…")

        # Embed all hypothetical docs
        vectors = self.embed_model.embed_documents(hypotheticals)
        mean_vec = np.mean(np.array(vectors), axis=0)
        return mean_vec

    def expand_and_retrieve_texts(
        self, query: str, faiss_store, k: int = 100
    ) -> List[str]:
        """
        Convenience wrapper: expand query → retrieve k documents from a
        LangChain FAISS store, returned as plain text strings.
        """
        hyde_vec = self.expand(query)
        # FAISS similarity_search_by_vector returns LangChain Documents
        docs = faiss_store.similarity_search_by_vector(
            embedding=hyde_vec.tolist(), k=k
        )
        return [doc.page_content for doc in docs]

    def get_hypothetical_text(self, query: str) -> str:
        """Return the raw hypothetical answer text (useful for debugging/logging)."""
        return self._generate_hypothetical(query)
