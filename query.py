"""
Step 5: Query flow - Retrieve relevant chunks

Two retrievers are available:
  RAGRetriever    — vanilla FAISS semantic search (top-k, no reranking)
  HybridRetriever — full pipeline:
                      query type detection
                      → FAISS + BM25 (weights adjusted per query type)
                      → keyword boost
                      → weighted candidate pool
                      → neighbor expansion
                      → cross-encoder rerank
                    Use this for production queries.
"""
import re
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from embed import EmbeddingGenerator
from store import VectorStore

# Keywords that signal a fact/lookup query (dates, names, specific values).
# These queries benefit from heavy BM25 weighting.
_FACT_SIGNALS = {
    'when', 'date', 'birthday', 'how old', 'what time', 'year', 'month',
    'day', 'age', 'born', 'anniversary', 'since', 'until', 'start', 'end',
    'number', 'address', 'phone', 'name', 'who is', 'what is her', 'what is his',
}

# Words not worth boosting for keyword matching.
_STOP_WORDS = {
    'the', 'a', 'an', 'is', 'in', 'of', 'to', 'and', 'or', 'what', 'when',
    'where', 'who', 'how', 'did', 'was', 'were', 'her', 'his', 'she', 'he',
    'they', 'it', 'at', 'on', 'for', 'with', 'from', 'that', 'this', 'are',
    'do', 'does', 'i', 'me', 'my', 'we', 'you', 'be', 'been', 'have', 'had',
}


def _detect_query_type(query: str) -> str:
    """
    Detect whether a query is fact-like (specific lookup) or semantic (open-ended).

    Returns:
        'fact'     — exact match / date / name lookup → boost BM25
        'semantic' — open-ended / narrative → boost FAISS
    """
    q = query.lower()
    if any(signal in q for signal in _FACT_SIGNALS):
        return 'fact'
    return 'semantic'


def _boost_keywords(results: List[Dict], query: str, boost: float = 0.2) -> List[Dict]:
    """
    Add a score bonus to chunks that contain exact query keywords.
    Applied before reranking so keyword hits survive the cut.

    Args:
        results: List of result dicts (modified in place)
        query:   The raw query string
        boost:   Score added per chunk that matches any keyword

    Returns:
        Same list with 'score' bumped and 'keyword_boosted' flag set.
    """
    keywords = set(re.findall(r'\w+', query.lower())) - _STOP_WORDS
    if not keywords:
        return results

    for r in results:
        text_lower = r['chunk']['text'].lower()
        if any(kw in text_lower for kw in keywords):
            r['score'] = r['score'] + boost
            r['keyword_boosted'] = True
        else:
            r.setdefault('keyword_boosted', False)

    return results


class RAGRetriever:
    """Retrieve relevant chunks for queries."""
    
    def __init__(self, store_dir: str = 'processed/vector_store', model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize retriever.
        
        Args:
            store_dir: Path to vector store
            model_name: Embedding model name (must match the one used for indexing)
        """
        print("Initializing RAG retriever...")
        self.vector_store = VectorStore.load(store_dir)
        self.embedding_generator = EmbeddingGenerator(model_name)
        print("Retriever ready")
    
    def retrieve(self, query: str, k: int = 50, min_score: Optional[float] = None) -> List[Dict]:
        """
        Retrieve top-k relevant chunks for a query.
        
        Args:
            query: Question or search query
            k: Number of results to return
            min_score: Minimum similarity score (optional filter)
        
        Returns:
            List of dicts with 'chunk', 'score', and 'rank'
        """
        # Embed query
        query_embedding = self.embedding_generator.embed_query(query)
        
        # Search
        scores, chunks = self.vector_store.search(query_embedding, k=k)
        
        # Format results
        results = []
        for rank, (score, chunk) in enumerate(zip(scores, chunks)):
            # Filter by minimum score if specified
            if min_score is not None and score < min_score:
                continue
            
            results.append({
                'rank': rank + 1,
                'score': float(score),
                'chunk': chunk
            })
        
        return results
    
    def retrieve_with_temporal_context(self, query: str, k: int = 5, 
                                      time_window: Optional[str] = None) -> List[Dict]:
        """
        Retrieve chunks with optional temporal filtering.
        
        Args:
            query: Question
            k: Number of results
            time_window: Optional timestamp filter (e.g., "2024")
        
        Returns:
            List of results sorted by relevance (and optionally by time)
        """
        results = self.retrieve(query, k=k * 2)  # Get more results for filtering
        
        # Filter by time if specified
        if time_window:
            results = [
                r for r in results 
                if time_window in r['chunk'].get('timestamp_start', '') or
                   time_window in r['chunk'].get('timestamp_end', '')
            ]
        
        # Return top k
        return results[:k]
    
    def print_results(self, results: List[Dict], show_full_text: bool = False):
        """Pretty print results."""
        if not results:
            print("No results found.")
            return
        
        print(f"\nFound {len(results)} results:\n")
        print("=" * 80)
        
        for result in results:
            chunk = result['chunk']
            score = result['score']
            rank = result['rank']
            
            print(f"\n[{rank}] Score: {score:.4f}")
            print(f"Sender: {chunk.get('sender', 'unknown')}")
            print(f"Timestamp: {chunk.get('timestamp_start', 'unknown')}")
            
            text = chunk['text']
            if show_full_text or len(text) < 200:
                print(f"Text: {text}")
            else:
                print(f"Text: {text[:200]}...")
            
            print("-" * 80)


class HybridRetriever:
    """
    Full retrieval pipeline:
      1. FAISS semantic search  (top faiss_k)
      2. BM25 keyword search    (top bm25_k)
      3. Merge + dedupe by chunk_id
      4. Neighbor expansion     (±expand_n adjacent chunks for context)
      5. Cross-encoder rerank   (final_k results)

    This replaces bare RAGRetriever for production use.
    """

    def __init__(
        self,
        store_dir: str = 'processed/vector_store',
        chunks_path: str = 'processed/chunks.json',
        model_name: str = 'all-MiniLM-L6-v2',
        reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    ):
        from bm25_search import BM25Search
        from rerank import Reranker

        self.retriever = RAGRetriever(store_dir, model_name)
        self.bm25 = BM25Search.from_file(chunks_path)
        self.reranker = Reranker(reranker_model)

        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.all_chunks = json.load(f)

        print("HybridRetriever ready")

    def retrieve(
        self,
        query: str,
        faiss_k: int = 50,
        bm25_k: int = 20,
        expand_n: int = 2,
        final_k: int = 10,
        debug: bool = False,
    ) -> List[Dict]:
        """
        Run the full hybrid pipeline:
          1. Detect query type → adjust faiss_k / bm25_k
          2. FAISS semantic search
          3. BM25 keyword search
          4. Keyword score boosting
          5. Weighted candidate pool (capped per source)
          6. Neighbor expansion
          7. Cross-encoder rerank

        Args:
            query:    Search query
            faiss_k:  Base FAISS candidate count (adjusted by query type)
            bm25_k:   Base BM25 candidate count (adjusted by query type)
            expand_n: Neighbor window (±expand_n chunks around each hit)
            final_k:  Results to return after reranking
            debug:    Print pipeline stats when True

        Returns:
            Reranked list of result dicts, best first.
            Each dict has: rank, score, rerank_score, chunk,
                           is_neighbor, keyword_boosted, source (faiss/bm25/neighbor)
        """
        # --- 1. Query type detection → adjust weights ---
        query_type = _detect_query_type(query)
        if query_type == 'fact':
            faiss_k = max(faiss_k - 20, 20)   # 30
            bm25_k  = bm25_k + 20              # 40
        # semantic: keep defaults (faiss_k=50, bm25_k=20)

        if debug:
            print(f"\n[DEBUG] query_type={query_type}  faiss_k={faiss_k}  bm25_k={bm25_k}")

        # --- 2. FAISS ---
        faiss_results = self.retriever.retrieve(query, k=faiss_k)
        for r in faiss_results:
            r['source'] = 'faiss'
            r.setdefault('is_neighbor', False)

        # --- 3. BM25 ---
        bm25_results = self.bm25.search(query, k=bm25_k)
        for r in bm25_results:
            r['source'] = 'bm25'
            r.setdefault('is_neighbor', False)

        if debug:
            print(f"[DEBUG] FAISS hits={len(faiss_results)}  BM25 hits={len(bm25_results)}")

        # --- 4. Keyword boost (before pool selection) ---
        faiss_results = _boost_keywords(faiss_results, query)
        bm25_results  = _boost_keywords(bm25_results,  query)

        # --- 5. Weighted candidate pool (capped per source) ---
        faiss_ids = {r['chunk']['chunk_id'] for r in faiss_results[:30]}
        seen_ids: set = set()
        merged: List[Dict] = []

        # FAISS: top 30
        for r in faiss_results[:30]:
            cid = r['chunk']['chunk_id']
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(r)

        # BM25: top 15 unique (not already in FAISS pool)
        bm25_added = 0
        for r in bm25_results:
            if bm25_added >= 15:
                break
            cid = r['chunk']['chunk_id']
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(r)
                bm25_added += 1

        if debug:
            n_bm25_only = sum(1 for r in merged if r['source'] == 'bm25')
            print(f"[DEBUG] merged pool={len(merged)}  bm25_unique={n_bm25_only}")

        # --- 6. Neighbor expansion (capped at 30 new chunks) ---
        n_total = len(self.all_chunks)
        expanded_ids = set(seen_ids)
        neighbors: List[Dict] = []

        for r in merged:
            if len(neighbors) >= 30:
                break
            cid = r['chunk']['chunk_id']
            for offset in range(-expand_n, expand_n + 1):
                if offset == 0:
                    continue
                neighbor_id = cid + offset
                if 0 <= neighbor_id < n_total and neighbor_id not in expanded_ids:
                    expanded_ids.add(neighbor_id)
                    neighbors.append({
                        'rank': 0,
                        'score': 0.0,
                        'chunk': self.all_chunks[neighbor_id],
                        'is_neighbor': True,
                        'keyword_boosted': False,
                        'source': 'neighbor',
                    })

        candidates = merged + neighbors

        if debug:
            print(f"[DEBUG] neighbors added={len(neighbors)}  total candidates={len(candidates)}")

        # --- 7. Rerank ---
        reranked = self.reranker.rerank(query, candidates, top_k=final_k)

        if debug:
            print(f"[DEBUG] reranked → top {len(reranked)} results")
            for r in reranked[:5]:
                boosted = '★' if r.get('keyword_boosted') else ' '
                neighbor = '[N]' if r.get('is_neighbor') else '   '
                print(
                    f"  [{r['rank']}] {boosted} {neighbor} "
                    f"rerank={r.get('rerank_score', 0):.4f} "
                    f"src={r.get('source','?'):8s}  "
                    f"{r['chunk']['text'][:60]!r}"
                )

        return reranked

    def get_context_window(self, chunk_id: int, window: int = 5) -> List[Dict]:
        """
        Return the chunk at chunk_id plus up to `window` chunks on each side.

        Args:
            chunk_id: Integer index of the anchor chunk in all_chunks
            window:   Number of messages to include before and after the anchor

        Returns:
            Ordered list of chunk dicts spanning [chunk_id-window, chunk_id+window]
        """
        n_total = len(self.all_chunks)
        start = max(0, chunk_id - window)
        end = min(n_total, chunk_id + window + 1)
        return self.all_chunks[start:end]

    def print_results(self, results: List[Dict], show_full_text: bool = False):
        """Pretty-print reranked results."""
        if not results:
            print("No results found.")
            return

        print(f"\nFound {len(results)} results:\n" + "=" * 80)
        for r in results:
            chunk = r['chunk']
            rerank_score = r.get('rerank_score', r['score'])
            tags = []
            if r.get('is_neighbor'):
                tags.append('neighbor')
            if r.get('keyword_boosted'):
                tags.append('keyword-hit')
            src = r.get('source', '?')
            tag_str = f"  [{', '.join(tags)}]" if tags else ""
            print(f"\n[{r['rank']}] rerank={rerank_score:.4f}  score={r['score']:.4f}  src={src}{tag_str}")
            print(f"Sender: {chunk.get('sender', 'unknown')}")
            print(f"Time:   {chunk.get('timestamp_start', 'unknown')}")
            text = chunk['text']
            if show_full_text or len(text) < 200:
                print(f"Text:   {text}")
            else:
                print(f"Text:   {text[:200]}...")
            print("-" * 80)


if __name__ == '__main__':
    import sys

    # Use --debug flag to enable pipeline stats
    args = sys.argv[1:]
    debug = '--debug' in args
    args = [a for a in args if a != '--debug']

    retriever = HybridRetriever()

    if args:
        query = ' '.join(args)
        results = retriever.retrieve(query, debug=debug)
        retriever.print_results(results, show_full_text=True)
    else:
        print("\n" + "=" * 80)
        print("RAG Hybrid Query Interface  (FAISS + BM25 + rerank)")
        print("Tip: start query with 'debug:' to see pipeline stats")
        print("=" * 80)
        print("\nType your questions. Type 'quit' to exit.\n")

        while True:
            query = input("Query: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                break
            if not query:
                continue

            _debug = query.startswith('debug:')
            if _debug:
                query = query[len('debug:'):].strip()

            results = retriever.retrieve(query, debug=_debug)
            retriever.print_results(results)
