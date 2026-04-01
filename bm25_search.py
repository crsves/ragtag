"""
BM25 keyword search — exact/lexical matching safety net.

Complements FAISS semantic search: catches exact word matches that embeddings miss
(e.g. "birthday", specific names, rare terms).

Requires: rank_bm25
"""
import json
import re
from typing import List, Dict
from pathlib import Path
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    """Lowercase + split on non-alphanumeric characters."""
    return re.findall(r'\w+', text.lower())


class BM25Search:
    """BM25 index over chunks for keyword retrieval."""

    def __init__(self, chunks: List[Dict]):
        """
        Build BM25 index from chunks.

        Args:
            chunks: List of chunk dicts, each with a 'text' field.
        """
        self.chunks = chunks
        print(f"Building BM25 index over {len(chunks)} chunks...")
        tokenized = [_tokenize(c['text']) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        print("BM25 index ready")

    def search(self, query: str, k: int = 20) -> List[Dict]:
        """
        BM25 keyword search.

        Args:
            query: Search query
            k: Max results to return

        Returns:
            List of result dicts (same shape as RAGRetriever results).
            Only returns chunks with score > 0 (actual keyword matches).
        """
        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                break  # rest will also be 0, BM25 scores are monotone after sort
            results.append({
                'rank': rank + 1,
                'score': float(scores[idx]),
                'bm25_score': float(scores[idx]),
                'chunk': self.chunks[idx],
            })

        return results

    @classmethod
    def from_file(cls, chunks_path: str) -> 'BM25Search':
        """Load chunks from JSON and build index."""
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        return cls(chunks)


if __name__ == '__main__':
    import sys

    index = BM25Search.from_file('processed/chunks.json')

    query = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else 'birthday'
    results = index.search(query, k=10)

    print(f"\nBM25 results for: '{query}'\n" + '=' * 60)
    for r in results:
        chunk = r['chunk']
        print(f"[{r['rank']}] score={r['score']:.3f}  {chunk.get('timestamp_start', '')}  {chunk['text'][:120]}")
