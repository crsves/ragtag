"""
Step 6: Feed retrieved chunks to LLM for answer generation

Prompt format instructs the LLM to return structured output:
  Answer:  [direct answer — one phrase or sentence]
  Quote:   [exact text from messages]
  When:    [timestamp of the source message]
  Context: [one sentence of surrounding context if needed]
"""
import json
from typing import List, Dict, Optional
from query import HybridRetriever, _detect_query_type


class RAGAnswerer:
    """Generate answers using retrieved context."""

    def __init__(
        self,
        store_dir: str = 'processed/vector_store',
        chunks_path: str = 'processed/chunks.json',
    ):
        """Initialize answerer with hybrid retriever."""
        self.retriever = HybridRetriever(store_dir, chunks_path)
    
    def create_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Create a structured extraction prompt for the LLM.

        The prompt instructs the LLM to return:
          Answer:  direct answer (one phrase/sentence)
          Quote:   verbatim text from the messages
          When:    timestamp of the source
          Context: one sentence of surrounding info (if needed)

        Args:
            query: User's question
            context_chunks: Retrieved and reranked chunks

        Returns:
            Formatted prompt string
        """
        query_type = _detect_query_type(query)

        # Build context — ordered best-first (already reranked)
        context_parts = []
        for i, result in enumerate(context_chunks):
            chunk = result['chunk']
            score = result.get('rerank_score', result['score'])
            tags = []
            if result.get('keyword_boosted'):
                tags.append('keyword-match')
            if result.get('is_neighbor'):
                tags.append('context')
            tag_str = f"  [{', '.join(tags)}]" if tags else ""
            context_parts.append(
                f"[{i+1}] relevance={score:.2f}{tag_str}\n"
                f"  {chunk.get('timestamp_start', 'unknown')} | {chunk.get('sender', 'unknown')}\n"
                f"  {chunk['text']}"
            )

        context = "\n\n".join(context_parts)

        # Fact queries get tighter extraction instructions
        if query_type == 'fact':
            instructions = (
                "- Extract the EXACT answer (a date, name, number, or short phrase)\n"
                "- Copy the EXACT quote from the message that contains the answer\n"
                "- Include the timestamp of that message\n"
                "- If the answer is not in the context, say: Answer: unknown\n"
                "- Do NOT infer or guess — only use what is literally in the messages"
            )
        else:
            instructions = (
                "- Summarize what the messages say about the question\n"
                "- Quote the most relevant line(s) exactly as written\n"
                "- Include the timestamps of quoted messages\n"
                "- If context is insufficient, say so directly\n"
                "- Be concise and factual — no speculation"
            )

        prompt = f"""You are an assistant extracting answers from chat message history.

MESSAGES (ordered by relevance, best first):
{context}

INSTRUCTIONS:
{instructions}

QUESTION: {query}

Respond in exactly this format:
Answer:  [direct answer]
Quote:   "[exact text from one of the messages above]"
When:    [timestamp]
Context: [one sentence of surrounding context, or "n/a"]"""

        return prompt
    
    def answer(self, query: str, k: int = 10, show_sources: bool = True, debug: bool = False) -> Dict:
        """
        Answer a question using RAG.

        Args:
            query: User's question
            k: Number of final chunks after reranking (default 10)
            show_sources: Whether to include source chunks in response
            debug: Print pipeline stats

        Returns:
            Dict with 'prompt', 'sources', 'source_ids', 'query_type'
        """
        query_type = _detect_query_type(query)

        # Retrieve relevant chunks via full hybrid pipeline
        results = self.retriever.retrieve(query, final_k=k, debug=debug)
        
        if not results:
            return {
                'prompt': f"Question: {query}\n\nNo relevant messages found.",
                'sources': [],
                'source_ids': []
            }
        
        # Create prompt
        prompt = self.create_prompt(query, results)
        
        # Track which chunks were used
        source_ids = [r['chunk']['chunk_id'] for r in results]
        
        response = {
            'prompt': prompt,
            'sources': results if show_sources else None,
            'source_ids': source_ids,
            'query': query,
            'query_type': query_type,
        }
        
        return response
    
    def print_answer_prompt(self, query: str, k: int = 10, debug: bool = False):
        """Print the prompt that would be sent to an LLM."""
        response = self.answer(query, k=k, debug=debug)

        print("\n" + "=" * 80)
        print(f"PROMPT FOR LLM  [query_type={response['query_type']}]")
        print("=" * 80)
        print(response['prompt'])
        print("\n" + "=" * 80)
        print(f"Sources used: {len(response['source_ids'])} chunks  |  IDs: {response['source_ids']}")
        print("=" * 80)
    
    def save_answer_session(self, query: str, output_file: str, k: int = 5):
        """Save a query session with prompt and sources."""
        response = self.answer(query, k=k)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
        
        print(f"Saved answer session to {output_file}")


if __name__ == '__main__':
    import sys

    args = sys.argv[1:]
    debug = '--debug' in args
    args = [a for a in args if a != '--debug']

    answerer = RAGAnswerer()

    if args:
        query = ' '.join(args)
        answerer.print_answer_prompt(query, k=10, debug=debug)
    else:
        print("\n" + "=" * 80)
        print("RAG Answer Generator  (FAISS + BM25 + rerank + structured extraction)")
        print("Tip: start query with 'debug:' to see pipeline stats")
        print("=" * 80)
        print()

        while True:
            query = input("Question: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                break
            if not query:
                continue

            _debug = query.startswith('debug:')
            if _debug:
                query = query[len('debug:'):].strip()

            answerer.print_answer_prompt(query, k=10, debug=_debug)
