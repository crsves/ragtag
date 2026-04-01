"""
Cross-encoder reranker for RAG pipeline.
Reranks candidates using a cross-encoder (far more accurate than bi-encoder scoring).

Pipeline position: FAISS + BM25 candidates → reranker → top-k final results
"""
from typing import List, Dict
from sentence_transformers import CrossEncoder


class Reranker:
    """Cross-encoder reranker. Much slower than bi-encoder but far more accurate."""

    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        print(f"Loading reranker: {model_name}")
        self.model = CrossEncoder(model_name)
        print("Reranker ready")

    def rerank(self, query: str, results: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Rerank results using cross-encoder.

        Args:
            query: The search query
            results: List of result dicts (must have 'chunk' key with 'text')
            top_k: How many to return after reranking

        Returns:
            Reranked list, sorted best-first, with 'rerank_score' and updated 'rank'.
        """
        if not results:
            return results

        pairs = [(query, r['chunk']['text']) for r in results]
        scores = self.model.predict(pairs)

        for result, score in zip(results, scores):
            result['rerank_score'] = float(score)

        reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)

        for i, r in enumerate(reranked):
            r['rank'] = i + 1

        return reranked[:top_k]
