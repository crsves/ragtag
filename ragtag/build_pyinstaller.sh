#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

os_raw="$(uname -s)"
arch_raw="$(uname -m)"

case "$os_raw" in
	Linux) target_os="linux" ;;
	Darwin) target_os="mac" ;;
	*)
		echo "Unsupported host OS for artifact naming: $os_raw" >&2
		exit 1
		;;
esac

case "$target_os/$arch_raw" in
	linux/x86_64|linux/amd64) target_suffix="linux-amd64" ;;
	linux/aarch64|linux/arm64) target_suffix="linux-arm64" ;;
	mac/x86_64) target_suffix="mac-intel" ;;
	mac/arm64|mac/aarch64) target_suffix="mac-arm64" ;;
	*)
		echo "Unsupported host architecture for artifact naming: $os_raw/$arch_raw" >&2
		exit 1
		;;
esac

echo "Installing pyinstaller..."
for _py in python3.12 python3.11 python3.10 python3.9 python3; do
	if command -v "$_py" >/dev/null 2>&1; then
		PY_BIN="$_py"
		break
	fi
done
if [[ -z "${PY_BIN:-}" ]]; then
	echo "Python 3.9+ is required to build PyInstaller artifacts." >&2
	exit 1
fi

rm -rf .venv
"$PY_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Keep torch CPU-only to avoid gigantic CUDA runtime downloads in CI/server builds.
export PIP_INDEX_URL="https://download.pytorch.org/whl/cpu"
export PIP_EXTRA_INDEX_URL="https://pypi.org/simple"

python -m pip install --only-binary=:all: pyinstaller

# Preferred path: use repository-pinned requirements when compatible.
if ! python -m pip install --only-binary=:all: -r requirements.txt; then
	echo "Pinned requirements are incompatible with $PY_BIN; using build-compatible deps..."
	# torch 2.2.x has a frozen-exe incompatibility in torch._inductor.config
	# (inspect.getsource fails at import time). Use >=2.4.0 to avoid it.
	# transformers >=4.44 introduced create_import_structure_from_path() which
	# scans .py source files at runtime — unavailable in frozen exes. Pin <4.44.
	# torchvision is not a pipeline dependency — skip it to keep binary size down.
	python -m pip install --only-binary=:all: \
		"torch>=2.4.0" \
		"numpy" \
		"tqdm" \
		"rank_bm25>=0.2.2" \
		"openai>=1.0.0" \
		"faiss-cpu" \
		"huggingface_hub>=0.23.0" \
		"transformers>=4.41.0,<4.44.0" \
		"sentence-transformers>=3.0.0,<4.0.0"
fi

echo "Building standalone binaries..."
# Use spec files so the runtime hook is included automatically.
pyinstaller bridge.spec
cp dist/bridge "dist/bridge-${target_suffix}"
pyinstaller pipeline.spec
cp dist/pipeline "dist/pipeline-${target_suffix}"

echo "Done! The executables are located in the dist/ directory."
cp "dist/bridge-${target_suffix}" ragtag/
cp "dist/pipeline-${target_suffix}" ragtag/
