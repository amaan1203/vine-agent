"""
VINE-Agent: Hybrid Retriever (ColBERT + FAISS + HyDE)

Pipeline:
  1. HyDE expands the query → hypothetical embedding
  2. FAISS retrieves top-1000 candidates using that embedding
  3. ColBERT (via RAGatouille) re-ranks to top-k
  4. Final docs passed to Raptor constructor in agent.py

Ported from ReCOR-RAG (make_raptor node + colbert_reranker function),
extended with HyDE integration.
"""

from __future__ import annotations
import logging
import sys
import types
from typing import Any, Dict, List, Optional, Union

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from hyde import HyDEExpander

logger = logging.getLogger(__name__)


def _patch_ragatouille():
    """
    Compatibility shim required by RAGatouille for older LangChain versions.
    Directly ported from ReCOR-RAG answer-generation.ipynb.
    """
    from langchain_core.documents.compressor import BaseDocumentCompressor
    fake_module = types.ModuleType("langchain.retrievers.document_compressors.base")
    fake_module.BaseDocumentCompressor = BaseDocumentCompressor
    sys.modules["langchain.retrievers.document_compressors.base"] = fake_module


def load_colbert(checkpoint: str = "colbert-ir/colbertv2.0"):
    """
    Load ColBERT reranker via RAGatouille.
    Falls back gracefully if ragatouille is not installed (CPU-only mode).
    """
    _patch_ragatouille()
    try:
        from ragatouille import RAGPretrainedModel
        rag = RAGPretrainedModel.from_pretrained(checkpoint)
        logger.info(f"[RETRIEVER] ColBERT loaded: {checkpoint}")
        return rag
    except ImportError:
        logger.warning("[RETRIEVER] ragatouille not installed; ColBERT reranking disabled.")
        return None


def colbert_reranker(
    query: str,
    documents: List[Union[str, Dict]],
    k: int = 100,
    reranker_model=None,
) -> List[Dict]:
    """
    Re-rank documents using ColBERT late-interaction scoring.
    Returns top-k dicts with 'content' and 'score'.

    If reranker_model is None, returns documents unchanged (fallback mode).
    """
    if reranker_model is None:
        logger.warning("[RETRIEVER] ColBERT model not available — returning documents unranked.")
        texts = [d if isinstance(d, str) else d.get("content", str(d)) for d in documents]
        return [{"content": t, "score": 1.0} for t in texts[:k]]

    # Normalise to list of strings
    if documents and not isinstance(documents[0], str):
        doc_texts = [d.get("content", str(d)) for d in documents]
    else:
        doc_texts = documents

    reranked = reranker_model.rerank(query=query, documents=doc_texts, k=k)
    return reranked


class VINERetriever:
    """
    Hybrid retriever for VINE-Agent.

    Combines:
      - HyDE: bridging lay language → technical agronomic embedding space
      - FAISS: fast approximate NN search over the knowledge base
      - ColBERT: late-interaction token-level re-ranking for precision
    """

    def __init__(
        self,
        faiss_store: FAISS,
        embed_model,
        llm,
        colbert_model=None,
        faiss_k: int = 200,
        colbert_k: int = 100,
        use_hyde: bool = True,
        n_hyde_hypotheticals: int = 1,
    ):
        """
        Args:
            faiss_store:          Pre-built LangChain FAISS store (from data_loader.py).
            embed_model:          Embedding model (same used to build the store).
            llm:                  LLM used by HyDE to generate hypotheticals.
            colbert_model:        RAGatouille ColBERT instance (or None to skip).
            faiss_k:              Initial FAISS candidate count.
            colbert_k:            Final documents after ColBERT re-ranking.
            use_hyde:             Enable HyDE expansion (default True).
            n_hyde_hypotheticals: Number of hypotheticals to average in HyDE.
        """
        self.faiss_store = faiss_store
        self.embed_model = embed_model
        self.colbert_model = colbert_model
        self.faiss_k = faiss_k
        self.colbert_k = colbert_k
        self.use_hyde = use_hyde

        if use_hyde:
            self.hyde = HyDEExpander(
                llm=llm,
                embed_model=embed_model,
                n_hypotheticals=n_hyde_hypotheticals,
            )
        else:
            self.hyde = None

    def retrieve(self, query: str) -> List[str]:
        """
        Full retrieval pipeline:
          1. (Optional) HyDE expansion
          2. FAISS candidate retrieval
          3. ColBERT re-ranking
          Returns list of text strings for RAPTOR to index.
        """
        # Step 1: Query → embedding (HyDE or standard)
        logger.info(f"[RETRIEVER] Starting retrieval for query: {query[:50]}...")
        if self.use_hyde and self.hyde is not None:
            hyde_vec = self.hyde.expand(query)
            raw_docs = self.faiss_store.similarity_search_by_vector(
                embedding=hyde_vec.tolist(), k=self.faiss_k
            )
            logger.info(f"[RETRIEVER] HyDE + FAISS: retrieved {len(raw_docs)} candidates.")
        else:
            raw_docs = self.faiss_store.similarity_search(query, k=self.faiss_k)
            logger.info(f"[RETRIEVER] Standard FAISS: retrieved {len(raw_docs)} candidates.")

        # Normalise to strings
        raw_texts = [d.page_content for d in raw_docs]

        # Step 2: ColBERT re-ranking
        reranked = colbert_reranker(
            query=query,
            documents=raw_texts,
            k=self.colbert_k,
            reranker_model=self.colbert_model,
        )
        final_texts = [item["content"] for item in reranked]
        logger.info(f"[RETRIEVER] ColBERT → {len(final_texts)} final documents selected.")
        return final_texts

    @classmethod
    def build_faiss_from_texts(
        cls,
        texts: List[str],
        embed_model,
        save_path: Optional[str] = None,
    ) -> FAISS:
        """
        Utility: build a FAISS store from a list of text strings.
        Optionally save to disk.
        """
        docs = [Document(page_content=t) for t in texts]
        store = FAISS.from_documents(docs, embed_model)
        if save_path:
            store.save_local(save_path)
            logger.info(f"FAISS store saved to {save_path}")
        return store

    @classmethod
    def load_faiss(
        cls, path: str, embed_model, allow_dangerous: bool = True
    ) -> FAISS:
        """Load a pre-saved FAISS store."""
        store = FAISS.load_local(
            path, embed_model, allow_dangerous_deserialization=allow_dangerous
        )
        logger.info(f"FAISS store loaded from {path}")
        return store
