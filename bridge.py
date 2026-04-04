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
    context_window = []
    for wc in r.get("context_window", []):
        context_window.append({
            "chunk_id": str(wc.get("chunk_id", "")),
            "text": str(wc.get("text", "")),
            "timestamp_start": str(wc.get("timestamp_start", "")),
            "sender": str(wc.get("sender", "")),
        })
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
        "context_window": context_window,
    }


def _parse_chat_messages(data):
    """Normalize various chat JSON formats to [{sender, text, timestamp}]."""
    if isinstance(data, dict):
        items = None
        for key in ("messages", "msgs", "chats", "data"):
            if key in data and isinstance(data[key], list):
                items = data[key]
                break
        if items is None:
            items = next((v for v in data.values() if isinstance(v, list)), [])
    elif isinstance(data, list):
        items = data
    else:
        return []

    result = []
    for m in items:
        if not isinstance(m, dict):
            continue
        sender = (m.get("sender") or m.get("from") or m.get("author") or
                  m.get("name") or m.get("from_name") or m.get("username") or "?")
        if isinstance(sender, dict):
            sender = sender.get("nickname") or sender.get("username") or sender.get("name") or "?"
        text = (m.get("text") or m.get("content") or m.get("message") or m.get("body") or "")
        ts = (m.get("timestamp") or m.get("date") or m.get("time") or
              m.get("created_at") or m.get("ts") or m.get("date_unixtime") or "")
        if text:
            result.append({"sender": str(sender)[:32], "text": str(text), "timestamp": str(ts)})
    return result


def handle(req):
    global _retriever, _retriever_chat
    cmd = req.get("cmd", "retrieve")

    if cmd == "retrieve":
        retriever = get_retriever(req.get("chat"))
        results, debug_stats = retriever.retrieve(
            req["query"],
            final_k=req.get("k", 10),
            debug=req.get("debug", False),
            min_results=req.get("min_results", 0),
            score_threshold=req.get("score_threshold", 0.0),
        )
        if not results:
            return {"results": [], "context": "", "debug_stats": debug_stats}

        window = req.get("window", 5)
        results = _expand_with_window(results, retriever, window)
        context = _build_context(results)

        return {
            "results": [serialize_result(r) for r in results],
            "context": context,
            "debug_stats": debug_stats,
        }

    elif cmd == "rebuild":
        cm = get_chat_manager()
        active = cm.get_active_slug()
        chat = cm.get_active_chat()
        if not chat:
            return {"error": "no active chat"}
        try:
            ensure_imports()
            from pipeline import build_rag_system
            store_dir = str(chat.get("store_dir", ""))
            # Find raw files for this chat
            raw_files = chat.get("raw_files", [])
            if not raw_files:
                return {"error": "no raw files configured for this chat"}
            build_rag_system(input_file=raw_files[0], output_dir=store_dir)
            # Invalidate cached retriever
            _retriever = None
            _retriever_chat = None
            return {"ok": True, "message": f"Rebuilt index for '{active}'"}
        except Exception as e:
            return {"error": f"rebuild failed: {e}"}

    elif cmd == "ingest":
        file_path = req.get("file", "")
        if not file_path:
            return {"error": "file path required"}
        cm = get_chat_manager()
        chat = cm.get_active_chat()
        if not chat:
            return {"error": "no active chat"}
        try:
            from update import RAGUpdater
            store_dir = str(chat.get("store_dir", ""))
            updater = RAGUpdater(store_dir=store_dir + "/vector_store")
            updater.update_from_new_file(file_path)
            _retriever = None
            _retriever_chat = None
            return {"ok": True, "message": f"Ingested data from '{file_path}'"}
        except Exception as e:
            return {"error": f"ingest failed: {e}"}

    elif cmd == "test_retrieve":
        retriever = get_retriever(req.get("chat"))
        results, debug_stats = retriever.retrieve(
            req["query"],
            final_k=req.get("k", 10),
            debug=True,
            min_results=req.get("min_results", 0),
            score_threshold=req.get("score_threshold", 0.0),
        )
        return {
            "results": [serialize_result(r) for r in results],
            "context": "",
            "debug_stats": debug_stats,
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

    elif cmd == "check":
        from update import get_latest_timestamp
        from datetime import datetime, timezone
        cm = get_chat_manager()
        chat = cm.get_active_chat()
        if not chat:
            return {"error": "no active chat"}
        store_dir = str(chat.get("store_dir", "")) + "/vector_store"
        latest = get_latest_timestamp(store_dir)
        if not latest:
            return {"ok": True, "latest": "", "export_after": "", "hours_behind": -1}
        dt = datetime.fromisoformat(latest)
        dt_utc = dt.astimezone(timezone.utc)
        now_utc = datetime.now(timezone.utc)
        hours = int((now_utc - dt_utc).total_seconds() / 3600)
        export_after = dt_utc.strftime('%Y-%m-%d %H:%M:%S')
        return {
            "ok": True,
            "latest": dt_utc.strftime('%Y-%m-%d %H:%M UTC'),
            "export_after": export_after,
            "hours_behind": hours,
        }

    elif cmd == "list_raw_files":
        raw_dir = RAG_DIR / "raw"
        if raw_dir.is_dir():
            files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".json"))
        else:
            files = []
        return {"files": files}

    elif cmd == "read_raw_file":
        filename = req.get("file", "")
        if not filename:
            return {"error": "file required"}
        filename = os.path.basename(filename)  # sanitize
        raw_path = RAG_DIR / "raw" / filename
        if not raw_path.is_file():
            return {"error": f"file not found: {filename}"}
        try:
            data = json.loads(raw_path.read_text(encoding="utf-8"))
            messages = _parse_chat_messages(data)
            return {"messages": messages}
        except Exception as e:
            return {"error": f"read failed: {e}"}

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
