#!/usr/bin/env python3
"""
RAG retrieval bridge — long-running subprocess for the Go Bubbletea TUI.
Reads JSON queries from stdin, returns JSON results on stdout.
Keeps the retriever loaded in memory for fast repeated queries.
"""
import io
import json
import sys
import os
import traceback
from pathlib import Path

# On Windows, Python opens stdout/stdin in text mode which translates \n to
# \r\n on output and strips \r on input.  The Go side reads with bufio.Scanner
# which strips only \n, leaving a stray \r that breaks JSON parsing.
# Force Unix newlines on both ends — safe on all platforms.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, newline='\n', line_buffering=True)
sys.stdin  = io.TextIOWrapper(sys.stdin.buffer,  newline='',  line_buffering=True)

if getattr(sys, 'frozen', False):
    # Frozen binary: __file__ points to _MEIPASS temp dir, not the rag directory.
    # The TUI launcher sets cmd.Dir = ragDir before exec, so CWD is correct.
    RAG_DIR = Path.cwd()
else:
    RAG_DIR = Path(__file__).parent
    sys.path.insert(0, str(RAG_DIR))
    os.chdir(RAG_DIR)

# Lazy imports (heavy, avoid at module level)
ChatManager = None
HybridRetriever = None
_expand_with_window = None
_build_context = None

_retriever = None
_retriever_chat = None
_chat_manager = None


def ensure_imports():
    global ChatManager, HybridRetriever, _expand_with_window, _build_context
    if ChatManager is None:
        from chat_manager import ChatManager as CM
        ChatManager = CM
    if HybridRetriever is None:
        from query import HybridRetriever as HR
        HybridRetriever = HR
    if _expand_with_window is None:
        from ask import _expand_with_window as ew, _build_context as bc
        _expand_with_window = ew
        _build_context = bc


def get_chat_manager():
    global _chat_manager
    if _chat_manager is None:
        ensure_imports()
        _chat_manager = ChatManager(RAG_DIR)
    return _chat_manager


def get_retriever(slug=None):
    global _retriever, _retriever_chat
    ensure_imports()
    cm = get_chat_manager()
    active = slug or cm.get_active_slug()
    if _retriever is None or _retriever_chat != active:
        store_dir = cm.get_store_dir(active)
        chunks_file = cm.get_chunks_file(active)
        if not store_dir or not Path(store_dir).exists():
            raise RuntimeError(f"No index for chat '{active}'. Run Pipeline → Rebuild first.")
        _retriever = HybridRetriever(store_dir=store_dir, chunks_path=chunks_file)
        _retriever_chat = active
    return _retriever


def serialize_result(r):
    chunk = r["chunk"]
    return {
        "rank": r.get("rank", 0),
        "score": float(r.get("score", 0)),
        "rerank_score": float(r.get("rerank_score", r.get("score", 0))),
        "keyword_boosted": bool(r.get("keyword_boosted", False)),
        "is_neighbor": bool(r.get("is_neighbor", False)),
        "source": str(r.get("source", "?")),
        "chunk": {
            "chunk_id": str(chunk.get("chunk_id", "")),
            "text": str(chunk.get("text", "")),
            "timestamp_start": str(chunk.get("timestamp_start", "")),
            "sender": str(chunk.get("sender", "")),
        },
    }


def handle(req):
    cmd = req.get("cmd", "retrieve")

    if cmd == "retrieve":
        retriever = get_retriever(req.get("chat"))
        results = retriever.retrieve(
            req["query"],
            final_k=req.get("k", 10),
            debug=req.get("debug", False),
        )
        if not results:
            return {"results": [], "context": ""}

        window = req.get("window", 5)
        results = _expand_with_window(results, retriever, window)
        context = _build_context(results)

        return {
            "results": [serialize_result(r) for r in results],
            "context": context,
            "debug_stats": getattr(retriever, "_last_stats", None),
        }

    elif cmd == "list_chats":
        cm = get_chat_manager()
        chats = cm.list_chats()
        active = cm.get_active_slug()
        # Convert to JSON-safe dict
        safe_chats = {}
        for slug, info in chats.items():
            safe_chats[slug] = {
                "display_name": info.get("display_name", slug),
                "store_dir": str(info.get("store_dir", "")),
                "created_at": str(info.get("created_at", "")),
                "last_updated": str(info.get("last_updated", "")),
            }
        return {"chats": safe_chats, "active": active or ""}

    elif cmd == "set_chat":
        slug = req["slug"]
        cm = get_chat_manager()
        cm.set_active(slug)
        global _retriever, _retriever_chat
        _retriever = None
        _retriever_chat = None
        return {"ok": True}

    elif cmd == "stats":
        cm = get_chat_manager()
        active = cm.get_active_slug()
        chat = cm.get_active_chat()
        if not chat:
            return {"error": "no active chat"}
        store_dir = Path(chat["store_dir"])
        info_file = store_dir / "info.json"
        info = {}
        if info_file.exists():
            try:
                info = json.loads(info_file.read_text())
            except Exception:
                pass
        # Count chunks file
        chunks_file = chat.get("chunks_file", "")
        chunk_count = 0
        if chunks_file and Path(chunks_file).exists():
            try:
                data = json.loads(Path(chunks_file).read_text())
                chunk_count = len(data) if isinstance(data, list) else 0
            except Exception:
                pass
        return {
            "slug": active,
            "display_name": chat["display_name"],
            "store_dir": str(store_dir),
            "total_vectors": info.get("total_vectors", chunk_count),
            "created_at": chat.get("created_at", "?"),
            "last_updated": chat.get("last_updated", "?"),
        }

    else:
        return {"error": f"unknown cmd: {cmd}"}


# Signal readiness (heavy imports happen here on first use, bridge signals ready immediately)
print(json.dumps({"ready": True}), flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        result = handle(req)
    except Exception as e:
        result = {"error": str(e), "traceback": traceback.format_exc()}
    print(json.dumps(result), flush=True)
