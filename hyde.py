from __future__ import annotations
import logging
from typing import List
import numpy as np

from prompts import HYDE_PROMPT

logger = logging.getLogger(__name__)


class HyDEExpander:

    def __init__(self, llm, embed_model, n_hypotheticals: int = 1):
        self.llm = llm
        self.embed_model = embed_model
        self.n_hypotheticals = n_hypotheticals

    def _generate_hypothetical(self, query: str) -> str:
        
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
        hyde_vec = self.expand(query)
        # FAISS similarity_search_by_vector returns LangChain Documents
        docs = faiss_store.similarity_search_by_vector(
            embedding=hyde_vec.tolist(), k=k
        )
        return [doc.page_content for doc in docs]

    def get_hypothetical_text(self, query: str) -> str:
        return self._generate_hypothetical(query)
