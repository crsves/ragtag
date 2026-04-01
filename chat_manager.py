"""
Multi-chat registry — tracks all indexed chats and the active selection.

Layout on disk:
    processed/chats/registry.json          ← index of all chats
    processed/chats/<slug>/vector_store/   ← FAISS index + metadata
    processed/chats/<slug>/chunks.json
    processed/chats/<slug>/normalized.json
    processed/chats/<slug>/embeddings/

Legacy single-chat stores at processed/vector_store/ are auto-detected
on first run and registered as the "default" chat.
"""
import json
import re
from datetime import datetime
from pathlib import Path


def slugify(name: str) -> str:
    """Convert a display name to a filesystem-safe slug."""
    slug = name.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '_', slug)
    return slug.strip('_') or 'chat'


class ChatManager:
    """Manages a registry of chats, each with its own vector store."""

    def __init__(self, rag_dir: Path):
        self.rag_dir = Path(rag_dir)
        self.chats_dir = self.rag_dir / "processed" / "chats"
        self.registry_path = self.chats_dir / "registry.json"
        self._reg = self._load_or_init()

    # ── init ──────────────────────────────────────────────────────────────────

    def _load_or_init(self) -> dict:
        self.chats_dir.mkdir(parents=True, exist_ok=True)
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text())

        reg = {"active": None, "chats": {}}

        # Auto-register legacy single store if present
        legacy_store = self.rag_dir / "processed" / "vector_store"
        legacy_chunks = self.rag_dir / "processed" / "chunks.json"
        if legacy_store.exists():
            reg["chats"]["default"] = {
                "display_name": "Default",
                "store_dir": str(legacy_store),
                "chunks_file": str(legacy_chunks),
                "raw_file": None,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
            }
            reg["active"] = "default"

        self._write(reg)
        return reg

    def _write(self, reg: dict = None):
        r = reg if reg is not None else self._reg
        self.registry_path.write_text(json.dumps(r, indent=2, ensure_ascii=False))

    # ── read ──────────────────────────────────────────────────────────────────

    def list_chats(self) -> dict:
        return dict(self._reg["chats"])

    def get_active_slug(self) -> str | None:
        return self._reg.get("active")

    def get_active_chat(self) -> dict | None:
        slug = self.get_active_slug()
        return self._reg["chats"].get(slug) if slug else None

    def get_chat(self, slug: str) -> dict | None:
        return self._reg["chats"].get(slug)

    def get_store_dir(self, slug: str = None) -> str | None:
        chat = self.get_chat(slug or self.get_active_slug())
        return chat["store_dir"] if chat else None

    def get_chunks_file(self, slug: str = None) -> str | None:
        chat = self.get_chat(slug or self.get_active_slug())
        return chat.get("chunks_file") if chat else None

    def get_chunk_count(self, slug: str) -> int:
        chat = self.get_chat(slug)
        if not chat:
            return 0
        info = Path(chat["store_dir"]) / "info.json"
        if info.exists():
            try:
                return json.loads(info.read_text()).get("total_vectors", 0)
            except Exception:
                pass
        return 0

    def has_chats(self) -> bool:
        return bool(self._reg["chats"])

    def unique_slug(self, base: str) -> str:
        """Return a slug that doesn't collide with existing slugs."""
        slug = slugify(base)
        if slug not in self._reg["chats"]:
            return slug
        i = 2
        while f"{slug}_{i}" in self._reg["chats"]:
            i += 1
        return f"{slug}_{i}"

    # ── write ─────────────────────────────────────────────────────────────────

    def set_active(self, slug: str):
        self._reg["active"] = slug
        self._write()

    def register(self, slug: str, display_name: str,
                 store_dir: str, chunks_file: str, raw_file: str = None):
        self._reg["chats"][slug] = {
            "display_name": display_name,
            "store_dir": str(store_dir),
            "chunks_file": str(chunks_file),
            "raw_file": str(raw_file) if raw_file else None,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }
        self._write()

    def touch(self, slug: str):
        if slug in self._reg["chats"]:
            self._reg["chats"][slug]["last_updated"] = datetime.now().isoformat()
            self._write()

    def delete(self, slug: str):
        if slug not in self._reg["chats"]:
            return
        del self._reg["chats"][slug]
        if self._reg.get("active") == slug:
            remaining = list(self._reg["chats"].keys())
            self._reg["active"] = remaining[0] if remaining else None
        self._write()
