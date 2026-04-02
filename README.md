# ragtag

A local RAG system for searching and chatting with your chat message history. Point it at a Slack or Discord export, build an index, and ask anything — all on your machine, no cloud required.

```
"When did she go to Japan?"
"What food does she like?"
"What were the most stressful periods?"
```

Answers come from real retrieved messages with source attribution — no hallucinations.

---

## Installation

The easiest way to install ragtag is with the one-line installer. It handles everything: Python environment, dependencies, TUI binary, and optionally indexes a demo dataset so you can start querying immediately.

```bash
curl -sL https://ragtag.crsv.es | bash
```

The installer presents a menu:

- **Release** *(recommended)* — downloads prebuilt binaries, no compiler needed
- **Clone** — clones the full repo, for developers who want to modify the code
- **Exit** — quit without installing anything

It will prompt for a NIM API key (used for answer generation), offer to auto-index the included demo dataset, and place a `ragtag` command in `~/.local/bin`.

### Non-interactive / CI install

```bash
curl -sL https://ragtag.crsv.es | bash -s -- --release --yes --nim-key "nvapi-..."
```

| Flag | Description |
|---|---|
| `--release` | Use prebuilt binaries (recommended) |
| `--clone` | Clone source instead |
| `--yes` / `-y` | Accept all defaults, no prompts |
| `--nim-key <key>` | Supply API key non-interactively |
| `--skip-nim-key` | Skip key setup (configure later) |
| `--dir <path>` | Install to a custom directory |
| `--bin-dir <path>` | Custom directory for the `ragtag` symlink |

---

## Manual Setup

If you prefer not to use the installer:

### 1. Prerequisites

- **Python 3.9+** — [python.org](https://www.python.org/downloads/)
- **A NIM API key** — [build.nvidia.com](https://build.nvidia.com) — used only for the answer generation step

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs `sentence-transformers`, `faiss-cpu`, `numpy`, `tqdm`, `rank_bm25`, and `openai`. No GPU needed.

### 3. Add your data

Drop your JSON export into `raw/`. The pipeline accepts Slack exports and Discord exports (via [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter)).

```
raw/
└── yourexport.json
```

### 4. Build the index

```bash
python3 pipeline.py raw/yourexport.json
```

This runs the full pipeline: normalize → chunk → embed → store. Embedding ~250k messages takes 5–15 minutes on CPU. Only needs to run once.

```
processed/
├── normalized.json
├── chunks.json
├── embeddings/
└── vector_store/        ← FAISS index lives here
```

### 5. Configure your API key

```bash
# Linux / macOS
export NIM_API_KEY="nvapi-..."

# Windows (PowerShell)
$env:NIM_API_KEY = "nvapi-..."
```

### 6. Launch the TUI

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

Pre-built binaries are attached to each GitHub release. The installer downloads the right one automatically. No Go installation needed to run them.

| Platform | Binary |
|---|---|
| Linux x86-64 | `pipeline-linux-amd64` · `rag-tui-linux-amd64` |
| Linux ARM64 | `pipeline-linux-arm64` · `rag-tui-linux-arm64` |
| macOS Apple Silicon | `pipeline-mac-arm64` · `rag-tui-mac-arm64` |
| macOS Intel | `pipeline-mac-intel` · `rag-tui-mac-intel` |
| Windows x64 | `pipeline-windows.exe` · `rag-tui-windows.exe` |

On macOS you may need to allow the binary in **System Settings → Privacy & Security** the first time you run it.

```bash
# Remove quarantine attribute if Gatekeeper blocks the binary
xattr -d com.apple.quarantine ./rag-tui/rag-tui-mac-arm64
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `RAG_DIR` | Yes | Absolute path to the repo root (where `bridge.py` lives) |
| `NIM_API_KEY` | Yes | Your NIM API key for answer generation |
| `RAGTAG_PYTHON` | No | Override the Python executable (e.g. `/usr/bin/python3.11`) |

`RAGTAG_PYTHON` is useful when you have multiple Python versions or a virtual environment. Without it ragtag auto-detects `python3.12` → `python3.11` → … → `python`.

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

The pipeline and bridge Python binaries are built with PyInstaller. See [docs.crsv.es/ragtag](https://docs.crsv.es/ragtag) for a detailed write-up of the packaging process and the compatibility problems encountered with modern ML libraries.

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
        answer.py  NIM answer generation
```

All retrieval and embedding runs locally. Only the final answer generation step calls the NIM API.

---

## Privacy

- Embeddings and the FAISS index never leave your machine
- The original message data stays in `raw/` (never sent anywhere)
- Only the assembled context for each question is sent to the NIM API

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

**Poor search results**
- Try a different embedding model (see Customization)
- Increase k (number of retrieved results)
- Adjust chunking strategy (bundle more messages per chunk)

---

## References

- **Sentence Transformers**: https://www.sbert.net/
- **FAISS**: https://github.com/facebookresearch/faiss
- **RAG paper**: https://arxiv.org/abs/2005.11401

---

**Built with**: Python, sentence-transformers, FAISS, NumPy, Go  
**License**: Mozilla Public License Version 2.0  
**Author**: devin
