# ragtag

A local RAG system for searching and chatting with your chat message history. Point it at a Discord/chat export, build an index, and ask anything — all on your machine, no cloud required.

```
"When did she go to Japan?"
"What food does she like?"
"What were the most stressful periods?"
```

Answers come from real retrieved messages with source attribution — no hallucinations.

---

## Quick Start

### 1. Prerequisites

- **Python 3.9+** — [python.org](https://www.python.org/downloads/)
- **An OpenAI API key** — used only for the chat answer step

### 2. Install Python dependencies

**Linux / macOS**
```bash
bash setup.sh
```

**Windows** (PowerShell)
```powershell
.\setup.ps1
```

This installs `sentence-transformers`, `faiss-cpu`, `numpy`, `tqdm`, and `rank_bm25` locally — no GPU needed.

### 3. Add your data

Drop your chat export into `raw/`. The pipeline expects JSON in Discord export format (via [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter)).

```
raw/
└── yourexport.min.json
```

Edit `pipeline.py` line 1 to point at your file if needed (default: `raw/sania.min.json`).

### 4. Build the index

```bash
python3 pipeline.py
```

This runs the full pipeline: normalize → chunk → embed → store. Embedding ~250k messages takes 5–15 minutes on CPU. Only needs to run once.

```
processed/
├── normalized.json
├── chunks.json
├── embeddings/
└── vector_store/        ← FAISS index lives here
```

### 5. Configure your OpenAI key

```bash
# Linux / macOS
export OPENAI_API_KEY="sk-..."

# Windows (PowerShell)
$env:OPENAI_API_KEY = "sk-..."
```

### 6. Launch the TUI

Download the binary for your platform from the table below, or use the one in `rag-tui/`.

```bash
# Linux
export RAG_DIR="$(pwd)"
./rag-tui/rag-tui-linux-amd64

# macOS (Apple Silicon)
export RAG_DIR="$(pwd)"
./rag-tui/rag-tui-mac-arm64

# macOS (Intel)
export RAG_DIR="$(pwd)"
./rag-tui/rag-tui-mac-intel

# Windows (PowerShell)
$env:RAG_DIR = (Get-Location).Path
.\rag-tui\rag-tui-windows.exe
```

---

## Platform Binaries

Pre-built binaries are in `rag-tui/`. No Go installation needed to run them.

| Platform | Binary |
|---|---|
| Linux x86-64 | `rag-tui/rag-tui-linux-amd64` |
| Linux ARM64 | `rag-tui/rag-tui-linux-arm64` |
| macOS Apple Silicon | `rag-tui/rag-tui-mac-arm64` |
| macOS Intel | `rag-tui/rag-tui-mac-intel` |
| Windows x64 | `rag-tui/rag-tui-windows.exe` |

On macOS you may need to allow the binary in **System Settings → Privacy & Security** the first time you run it.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `RAG_DIR` | Yes | Absolute path to the repo root (where `bridge.py` lives) |
| `OPENAI_API_KEY` | Yes | Your OpenAI key for answer generation |
| `RAGTAG_PYTHON` | No | Override the Python executable (e.g. `/usr/bin/python3.11`) |

`RAGTAG_PYTHON` is useful when you have multiple Python versions or a virtual environment. Without it ragtag auto-detects `python3` → `python` → `py`.

---

## TUI Controls

| Key | Action |
|---|---|
| Type + Enter | Send a message / ask a question |
| `/help` | Show all commands and keybindings |
| `/model` | Switch LLM model |
| `Ctrl+C` / `Ctrl+D` | Quit |

---

## Updating the Index with New Messages

When you have new exports to add:

```bash
python3 update.py raw/new_export.json
```

Only new messages are embedded and appended — the existing index is not rebuilt.

---

## Building from Source (optional)

Requires Go 1.21+.

```bash
cd rag-tui
make all          # build all 5 platform binaries
make linux-amd64  # build one target
make clean        # remove all built binaries
```

---

## Architecture

```
raw/*.json
    │
    ▼
normalize.py   →  processed/normalized.json
chunk.py       →  processed/chunks.json
embed.py       →  processed/embeddings/
store.py       →  processed/vector_store/  (FAISS)
    │
    └── bridge.py  ←  rag-tui (Go TUI, spawns bridge as subprocess)
            │
            ▼
        query.py   hybrid BM25 + semantic search
        answer.py  OpenAI answer generation
```

All retrieval and embedding runs locally. Only the final answer generation step calls OpenAI.

---

## Privacy

- Embeddings and the FAISS index never leave your machine
- The original message data stays in `raw/` (never sent anywhere)
- Only the assembled context for each question is sent to OpenAI

---

## Troubleshooting

**TUI opens but queries fail / "no index for chat"**
- Make sure `RAG_DIR` points to the repo root, not the `rag-tui/` subfolder
- Run `python3 pipeline.py` first to build the index

**`python3` not found on Windows**
- Set `RAGTAG_PYTHON` to the full path: `$env:RAGTAG_PYTHON = "C:\Python312\python.exe"`

**Slow first query**
- The first query loads the sentence-transformer model into RAM (~150 MB). Subsequent queries are fast.

**macOS: "cannot be opened because the developer cannot be verified"**
- Run: `xattr -d com.apple.quarantine ./rag-tui/rag-tui-mac-arm64`

**Out of memory during pipeline**
- Reduce batch size in `embed.py`: set `batch_size=16`

**Poor search results:**
- Try different embedding model (see Customization)
- Increase k (number of results)
- Adjust chunking strategy (bundle more messages)

## 📈 Next Steps

1. **Integrate with LLM**: Pipe `answer.py` output to Claude/GPT API
2. **Add filters**: Filter by sender, date range, sentiment
3. **Build UI**: Streamlit or Gradio interface
4. **Export results**: Save answer sessions with sources
5. **Analytics**: Track common query themes, response quality

## 📚 References

- **Sentence Transformers**: https://www.sbert.net/
- **FAISS**: https://github.com/facebookresearch/faiss
- **RAG Papers**: https://arxiv.org/abs/2005.11401

---

**Built with**: Python, sentence-transformers, FAISS, NumPy  
**License**: Mozilla Public License Version 2.0  
**Author**: devin
