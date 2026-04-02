"""
End-to-end RAG: HybridRetriever → Llama 3.3 70B via NIM

Usage:
    python ask.py "When did she seem upset?"
    python ask.py --debug "When is her birthday?"
    python ask.py          # interactive mode

Or import:
    from ask import ask
    result = ask("When did she seem upset?")
    print(result['answer'])
"""
import sys
import textwrap
from typing import Dict, List, Optional

from openai import OpenAI

from query import HybridRetriever, _detect_query_type

# ---------------------------------------------------------------------------
# Lazy-loaded retriever — initialized once per process
# ---------------------------------------------------------------------------
_retriever: Optional[HybridRetriever] = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _expand_with_window(
    results: List[Dict],
    retriever,
    window: int,
) -> List[Dict]:
    """
    Attach a 'context_window' list to each result containing the surrounding
    chunks from the original corpus.  Results are deduplicated by chunk_id so
    that overlapping windows from nearby hits don't cause double-counting.

    Args:
        results:   Reranked result dicts from HybridRetriever
        retriever: HybridRetriever instance (provides get_context_window)
        window:    Messages to include before/after each matched chunk

    Returns:
        Same list with 'context_window' key added to every dict.
    """
    seen_window_ids: set = set()
    expanded = []
    for r in results:
        chunk_id = r['chunk'].get('chunk_id')
        if chunk_id is None or window == 0:
            r = dict(r)
            r['context_window'] = [r['chunk']]
            expanded.append(r)
            continue

        window_chunks = retriever.get_context_window(chunk_id, window)

        # Deduplicate at the window level: skip chunks already shown in a
        # previous result's window, but always keep the matched chunk itself.
        deduped = []
        for wc in window_chunks:
            wid = wc.get('chunk_id')
            if wid == chunk_id or wid not in seen_window_ids:
                deduped.append(wc)
                if wid is not None:
                    seen_window_ids.add(wid)

        r = dict(r)
        r['context_window'] = deduped
        expanded.append(r)
    return expanded


def _build_context(results: List[Dict]) -> str:
    """Format reranked chunks (with optional context windows) into the prompt block."""
    parts = []
    for i, r in enumerate(results):
        chunk = r['chunk']
        anchor_id = chunk.get('chunk_id')
        tags = []
        if r.get('keyword_boosted'):
            tags.append('keyword-match')
        if r.get('is_neighbor'):
            tags.append('context-neighbor')
        tag_str = f"  [{', '.join(tags)}]" if tags else ""

        header = (
            f"[Chunk {i+1}]{tag_str}\n"
            f"  Time: {chunk.get('timestamp_start', 'unknown')} | "
            f"From: {chunk.get('sender', 'unknown')}"
        )

        window = r.get('context_window')
        if window and len(window) > 1:
            lines = []
            for wc in window:
                ts = wc.get('timestamp_start', '')
                sender = wc.get('sender', '')
                text = wc.get('text', '')
                prefix = ">>> " if wc.get('chunk_id') == anchor_id else "    "
                lines.append(f"{prefix}[{ts}] {sender}: {text}")
            parts.append(header + "\n" + "\n".join(lines))
        else:
            parts.append(header + f"\n  {chunk['text']}")

    return "\n\n".join(parts)


def _build_prompt(query: str, results: List[Dict]) -> str:
    """Fill the NIM prompt template with query + retrieved context."""
    import nim_config
    context_text = _build_context(results)

    # Warn if context is suspiciously large (shouldn't happen with k=10)
    if len(context_text) > nim_config.MAX_CONTEXT_CHARS:
        print(f"[warn] context is {len(context_text):,} chars — consider reducing final_k")

    return f"""You are an expert assistant. Answer the user's question using only the context provided below. Do not make assumptions beyond the context.

Question:
{query}

Context chunks:
{context_text}

Instructions:
- Provide a clear and concise answer.
- Use information only from the context.
- Include references to the chunks where relevant (e.g. "according to Chunk 3").
- If the answer is not in the context, say: "The information is not available in the provided context."
- Maintain clarity and correct pronoun references.

Answer:"""


# ---------------------------------------------------------------------------
# NIM call
# ---------------------------------------------------------------------------

def _call_nim(prompt: str) -> str:
    """Send prompt to Llama 3.3 70B via NIM and return the answer text."""
    import nim_config
    from openai import OpenAI

    client = OpenAI(
        base_url=nim_config.NIM_BASE_URL,
        api_key=nim_config.NIM_API_KEY,
    )

    # Only pass thinking mode for models that support it
    thinking_models = {"deepseek-ai/deepseek-v3.2"}
    use_thinking = (
        getattr(nim_config, "THINKING_MODE", False)
        and nim_config.NIM_MODEL in thinking_models
    )

    kwargs = dict(
        model=nim_config.NIM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=nim_config.TEMPERATURE,
        top_p=nim_config.TOP_P,
        max_tokens=nim_config.MAX_TOKENS,
        stream=False,
    )

    if use_thinking:
        kwargs["extra_body"] = {"chat_template_kwargs": {"thinking": True}}

    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask(
    query: str,
    k: Optional[int] = None,
    window: int = 5,
    debug: bool = False,
) -> Dict:
    """
    Ask a question. Retrieves context from the chat history and answers
    using Llama 3.3 70B via NIM.

    Args:
        query:  Natural language question
        k:      Number of chunks to pass to the LLM (post-rerank)
        window: Surrounding messages to include around each retrieved chunk
                (0 = no context expansion, default 5)
        debug:  Print retrieval pipeline stats

    Returns:
        Dict with keys:
          'answer'      — LLM response string
          'query'       — original query
          'query_type'  — 'fact' or 'semantic'
          'sources'     — list of result dicts from HybridRetriever
          'prompt'      — the prompt that was sent to the LLM
    """
    import nim_config
    if k is None:
        k = nim_config.FINAL_K
    retriever = _get_retriever()
    query_type = _detect_query_type(query)

    if debug:
        print(f"\n[ask] query_type={query_type}  k={k}  window={window}")

    results = retriever.retrieve(query, final_k=k, debug=debug)

    if not results:
        return {
            'answer': "The information is not available in the provided context.",
            'query': query,
            'query_type': query_type,
            'sources': [],
            'prompt': '',
        }

    results = _expand_with_window(results, retriever, window)
    prompt = _build_prompt(query, results)
    answer = _call_nim(prompt)

    return {
        'answer': answer,
        'query': query,
        'query_type': query_type,
        'sources': results,
        'prompt': prompt,
    }


def _print_result(result: Dict, show_sources: bool = False):
    """Pretty-print an ask() result."""
    print("\n" + "=" * 80)
    print(f"Q [{result['query_type']}]: {result['query']}")
    print("=" * 80)
    # Wrap answer for readability
    for line in result['answer'].splitlines():
        print(textwrap.fill(line, width=80) if line.strip() else "")

    if show_sources and result['sources']:
        print("\n" + "-" * 40)
        print(f"Sources ({len(result['sources'])} chunks):")
        for r in result['sources']:
            chunk = r['chunk']
            score = r.get('rerank_score', r['score'])
            boosted = '★' if r.get('keyword_boosted') else ' '
            neighbor = '[N]' if r.get('is_neighbor') else '   '
            print(
                f"  {boosted}{neighbor} rerank={score:.3f}  "
                f"{chunk.get('timestamp_start','?')}  "
                f"{chunk['text'][:80]!r}"
            )
    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_window_flag(args: list) -> tuple:
    """Extract --window N from args. Returns (window_int, remaining_args)."""
    window = 5
    clean = []
    i = 0
    while i < len(args):
        if args[i] == '--window' and i + 1 < len(args):
            try:
                window = int(args[i + 1])
            except ValueError:
                pass
            i += 2
        else:
            clean.append(args[i])
            i += 1
    return window, clean


if __name__ == '__main__':
    args = sys.argv[1:]
    debug = '--debug' in args
    sources = '--sources' in args
    args = [a for a in args if a not in ('--debug', '--sources')]
    window, args = _parse_window_flag(args)

    if args:
        query = ' '.join(args)
        result = ask(query, window=window, debug=debug)
        _print_result(result, show_sources=sources)
    else:
        print("\n" + "=" * 80)
        print("RAG  ×  Llama 3.3 70B (NIM)")
        print("Flags: --debug  --sources  --window N")
        print("Prefix 'debug:' or 'sources:' or 'window:N:' in interactive mode")
        print("=" * 80)
        print()

        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not query or query.lower() in ('quit', 'exit', 'q'):
                break

            _debug = 'debug:' in query
            _sources = 'sources:' in query
            _window = window  # default from CLI arg
            # parse inline window:N: prefix
            import re as _re
            wm = _re.match(r'window:(\d+):', query)
            if wm:
                _window = int(wm.group(1))
                query = query[wm.end():].strip()
            query = query.replace('debug:', '').replace('sources:', '').strip()

            if not query:
                continue

            result = ask(query, window=_window, debug=_debug)
            _print_result(result, show_sources=_sources)
