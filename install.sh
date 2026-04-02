#!/usr/bin/env bash
# install.sh — ragtag installer
# https://github.com/crsves/ragtag

set -euo pipefail

# ─── Colors (24-bit ANSI) ─────────────────────────────────────────────────────
C1='\033[38;2;184;216;186m'   # #b8d8ba  mint green
C2='\033[38;2;217;219;188m'   # #d9dbbc  lime
C3='\033[38;2;252;221;188m'   # #fcddbc  peach
C4='\033[38;2;239;149;157m'   # #ef959d  pink
C5='\033[38;2;105;88;95m'     # #69585f  mauve
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

# ─── Constants ────────────────────────────────────────────────────────────────
VERSION="0.1.2"
REPO="crsves/ragtag"
INSTALL_DIR="${HOME}/.local/bin"
INSTALL_METHOD=""
RAGTAG_DIR=""
PYTHON_CMD=""
SELECTED=0
NON_INTERACTIVE=false
AUTO_YES=false
SKIP_NIM_KEY=false
SKIP_BANNER=false
NIM_API_KEY_ARG=""
RELEASE_PY_FALLBACK=false
RAGTAG_BIN_PATH=""
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
if [[ "$SCRIPT_PATH" == "-" || "$SCRIPT_PATH" == "bash" ]]; then
    SCRIPT_DIR="$(pwd)"
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
fi
TTY="/dev/tty"

RABBIT_FRAMES=()

# ─── Logging helpers ──────────────────────────────────────────────────────────
info()    { echo -e "  ${C1}•${RESET} $*"; }
success() { echo -e "  ${C2}✓${RESET} $*"; }
warn()    { echo -e "  ${C3}⚠${RESET} $*"; }
error()   { echo -e "  ${C4}✗${RESET} ${BOLD}$*${RESET}" >&2; }
fatal()   { error "$*"; exit 1; }
step()    { echo -e "\n${C4}${BOLD}▸ $*${RESET}"; }

show_cursor() { printf '\033[?25h'; }
hide_cursor() { printf '\033[?25l'; }

cleanup() {
    local exit_code=$?
    # Kill any running spinner
    if [[ -n "${SPINNER_PID:-}" ]]; then
        kill "$SPINNER_PID" 2>/dev/null || true
        wait "$SPINNER_PID" 2>/dev/null || true
    fi
    show_cursor
    tput cnorm > "$TTY" 2>/dev/null || true
    printf "\r\033[K"  # clear spinner line
    if [[ $exit_code -eq 130 ]]; then
        echo -e "\n  ${C5}Installation cancelled.${RESET}\n"
    fi
}

trap 'cleanup' EXIT
trap 'exit 130' INT TERM
ensure_tty() {
    [[ -r "$TTY" && -w "$TTY" ]] || fatal "An interactive terminal is required to run this installer."
}

prompt_line() {
    local __var_name="$1"
    local prompt_text="$2"
    local input

    printf '%b' "$prompt_text" > "$TTY"
    IFS= read -r input < "$TTY"
    printf -v "$__var_name" '%s' "$input"
}

prompt_secret() {
    local __var_name="$1"
    local prompt_text="$2"
    local input

    printf '%b' "$prompt_text" > "$TTY"
    IFS= read -rs input < "$TTY"
    printf '\n' > "$TTY"
    printf -v "$__var_name" '%s' "$input"
}

print_usage() {
    cat <<'EOF'
Usage: install.sh [options]

Options:
  --clone                   Install from git clone (developer mode)
  --release                 Install from prebuilt release assets
  --method <clone|release>  Explicit install method
  --dir <path>              Install target directory (default: ~/ragtag)
  --bin-dir <path>          Binary install directory (default: ~/.local/bin)
  --nim-key <key>           Write NIM_API_KEY without prompting
  --skip-nim-key            Skip NIM key configuration
  --yes, -y                 Non-interactive mode with defaults
  --non-interactive         Non-interactive mode (defaults to release)
  --no-banner               Skip splash banner
  --help, -h                Show this help message

Examples:
  curl -sL https://ragtag.crsv.es | bash -s -- --release --yes --skip-nim-key
  curl -sL https://ragtag.crsv.es | bash -s -- --clone --dir "$HOME/src/ragtag"
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --clone)
                INSTALL_METHOD=0
                ;;
            --release)
                INSTALL_METHOD=1
                ;;
            --method)
                shift
                [[ $# -gt 0 ]] || fatal "--method requires a value: clone or release"
                case "$1" in
                    clone) INSTALL_METHOD=0 ;;
                    release) INSTALL_METHOD=1 ;;
                    *) fatal "Invalid --method value: $1 (expected clone or release)" ;;
                esac
                ;;
            --dir|--install-dir)
                shift
                [[ $# -gt 0 ]] || fatal "--dir requires a directory path"
                RAGTAG_DIR="$1"
                ;;
            --bin-dir)
                shift
                [[ $# -gt 0 ]] || fatal "--bin-dir requires a directory path"
                INSTALL_DIR="$1"
                ;;
            --nim-key)
                shift
                [[ $# -gt 0 ]] || fatal "--nim-key requires a value"
                NIM_API_KEY_ARG="$1"
                ;;
            --skip-nim-key)
                SKIP_NIM_KEY=true
                ;;
            --yes|-y)
                AUTO_YES=true
                NON_INTERACTIVE=true
                ;;
            --non-interactive)
                NON_INTERACTIVE=true
                ;;
            --no-banner)
                SKIP_BANNER=true
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                fatal "Unknown option: $1 (use --help for valid options)"
                ;;
        esac
        shift
    done

    if [[ "$NON_INTERACTIVE" == true && -z "$INSTALL_METHOD" ]]; then
        INSTALL_METHOD=1
    fi
}

download_release_asset() {
    local asset_name="$1"
    local output_path="$2"
    local github_latest="https://github.com/${REPO}/releases/latest/download"
    local mirror_url="https://ragtag.crsv.es/releases/latest/download"

    # 1. GitHub /latest/download direct redirect
    if curl -fsSL --progress-bar "${github_latest}/${asset_name}" \
            -o "$output_path" 2>/dev/null; then
        return 0
    fi

    # 2. GitHub API: resolve the exact asset URL from releases/latest
    #    (handles cases where /latest redirect is stale or rate-limited)
    local api_url
    api_url="$(
        curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" 2>/dev/null \
        | grep -o '"browser_download_url":"[^"]*'"${asset_name}"'"' \
        | grep -o 'https://[^"]*'
    )"
    if [[ -n "$api_url" ]]; then
        if curl -fsSL --progress-bar "$api_url" -o "$output_path" 2>/dev/null; then
            return 0
        fi
    fi

    # 3. Mirror fallback
    curl -fsSL --progress-bar "${mirror_url}/${asset_name}" \
        -o "$output_path" 2>/dev/null
}

download_release_python_sources() {
    local base_raw="https://raw.githubusercontent.com/${REPO}/main"
    local files=(
        answer.py
        ask.py
        bm25_search.py
        bridge.py
        chat_manager.py
        chunk.py
        embed.py
        normalize.py
        pipeline.py
        query.py
        rerank.py
        store.py
        update.py
        requirements.txt
    )

    local file
    for file in "${files[@]}"; do
        if ! curl -fsSL "${base_raw}/${file}" -o "${RAGTAG_DIR}/${file}"; then
            fatal "Failed to download ${file} for Python fallback mode."
        fi
    done
}

write_default_demo_data() {
        local demo_path="$1"
        cat > "$demo_path" <<'JSON'
[
    {
        "sender": "alice",
        "content": "Welcome to ragtag demo data.",
        "timestamp": "2026-01-01T12:00:00Z"
    },
    {
        "sender": "bob",
        "content": "This sample file is intentionally tiny so installs stay fast.",
        "timestamp": "2026-01-01T12:01:00Z"
    },
    {
        "sender": "alice",
        "content": "Run pipeline.py on your own export to build a real index.",
        "timestamp": "2026-01-01T12:02:00Z"
    }
]
JSON
}

ensure_demo_data() {
        local demo_dir="${RAGTAG_DIR}/raw"
        local demo_file="${demo_dir}/slack.json"

        mkdir -p "$demo_dir"
        if [ -f "$demo_file" ]; then
                return
        fi

        start_spinner "Downloading demo data (slack.json) …"
        if curl -fsSL "https://ragtag.crsv.es/raw/slack.json" -o "$demo_file" 2>/dev/null; then
                stop_spinner
                success "Demo data ready"
                return
        fi

        if curl -fsSL "https://raw.githubusercontent.com/${REPO}/main/raw/slack.json" -o "$demo_file" 2>/dev/null; then
                stop_spinner
                success "Demo data ready"
                return
        fi

        stop_spinner
        warn "Demo data download failed; creating a built-in sample file instead."
        write_default_demo_data "$demo_file"
}

# ─── Pretty banner ────────────────────────────────────────────────────────────
load_rabbit_frames() {
    local anim_source="${SCRIPT_DIR}/rabbit_anim.js"
    local parser_python=""
    local candidate

    [[ -f "$anim_source" ]] || return 1

    for candidate in python3.12 python3.11 python3.10 python3.9 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' >/dev/null 2>&1; then
                parser_python="$candidate"
                break
            fi
        fi
    done

    [[ -n "$parser_python" ]] || return 1

    mapfile -d '' -t RABBIT_FRAMES < <("$parser_python" - "$anim_source" <<'PY'
import ast
import pathlib
import re
import sys

source = pathlib.Path(sys.argv[1]).read_text()
match = re.search(r'const frames = \[(.*)\];\n\nprocess\.stdout', source, re.S)
if not match:
    raise SystemExit(1)

frames = ast.literal_eval('[' + match.group(1) + ']')
for frame in frames:
    sys.stdout.write(frame)
    sys.stdout.write('\0')
PY
)

    ((${#RABBIT_FRAMES[@]} > 0))
}

print_wordmark() {
    echo -e "  ${C1}${BOLD}  _ __ __ _  __ _| |_ __ _  __ _ ${RESET}"
    echo -e "  ${C2}${BOLD} | '__/ _\` |/ _\` | __/ _\` |/ _\` |${RESET}"
    echo -e "  ${C3}${BOLD} | | | (_| | (_| | || (_| | (_| |${RESET}"
    echo -e "  ${C4}${BOLD} |_|  \\__,_|\\__, |\\__\\__,_|\\__, |${RESET}"
    echo -e "  ${C5}${BOLD}             __/ |          __/ | ${RESET}"
    echo -e "  ${C5}${BOLD}            |___/          |___/  ${RESET}"
}

render_banner_frame() {
    local frame="$1"
    local colors=("$C1" "$C2" "$C3" "$C4" "$C5")
    local i=0

    clear
    echo
    while IFS= read -r line; do
        printf '  %b%s%b\n' "${colors[$((i % ${#colors[@]}))]}" "$line" "$RESET"
        i=$((i + 1))
    done <<< "$frame"
    echo
    print_wordmark
    echo
    echo -e "  ${DIM}v${VERSION}  ·  retrieval-augmented generation for your own data${RESET}"
    echo -e "  ${DIM}https://github.com/${REPO}${RESET}"
    echo
    echo -e "  ${C5}────────────────────────────────────────────────────${RESET}"
    echo
}

animate_banner() {
    local frame

    hide_cursor
    for frame in "${RABBIT_FRAMES[@]}"; do
        render_banner_frame "$frame"
        sleep 0.06
    done
    show_cursor
}

print_banner() {
    if load_rabbit_frames; then
        local last_index=$(( ${#RABBIT_FRAMES[@]} - 1 ))
        animate_banner
        render_banner_frame "${RABBIT_FRAMES[$last_index]}"
        return
    fi

    clear
    echo
    print_wordmark
    echo
    echo -e "  ${DIM}v${VERSION}  ·  retrieval-augmented generation for your own data${RESET}"
    echo -e "  ${DIM}https://github.com/${REPO}${RESET}"
    echo
    echo -e "  ${C5}────────────────────────────────────────────────────${RESET}"
    echo
}

# ─── Arrow-key interactive selector ──────────────────────────────────────────
# Usage: select_option "Prompt text" "Option 1" "Option 2" ...
# Sets global $SELECTED to the 0-based index of the chosen option.
select_option() {
    local prompt="$1"; shift
    local options=("$@")
    local num=${#options[@]}
    SELECTED=0

    ensure_tty

    # Save cursor position, hide it
    tput civis > "$TTY" 2>/dev/null || true

    echo -e "  ${C3}${BOLD}${prompt}${RESET}" > "$TTY"
    echo > "$TTY"

    # Initial draw
    for i in "${!options[@]}"; do
        if [ "$i" -eq "$SELECTED" ]; then
            echo -e "  ${C4}${BOLD} ▶  ${options[$i]}${RESET}" > "$TTY"
        else
            echo -e "  ${DIM}    ${options[$i]}${RESET}" > "$TTY"
        fi
    done

    while true; do
        # Read a keypress (may be escape sequence)
        IFS= read -rsn1 key < "$TTY"
        if [[ "$key" == $'\x1b' ]]; then
            # Read the next two chars of the escape sequence
            IFS= read -rsn2 -t 0.1 seq < "$TTY" || seq=""
            case "$seq" in
                '[A'|'OA') # Up arrow
                    SELECTED=$(( (SELECTED - 1 + num) % num ))
                    ;;
                '[B'|'OB') # Down arrow
                    SELECTED=$(( (SELECTED + 1) % num ))
                    ;;
            esac
        elif [[ "$key" == "" || "$key" == $'\n' ]]; then
            # Enter pressed
            break
        elif [[ "$key" == "k" ]]; then
            SELECTED=$(( (SELECTED - 1 + num) % num ))
        elif [[ "$key" == "j" ]]; then
            SELECTED=$(( (SELECTED + 1) % num ))
        fi

        # Redraw: move cursor up (num lines) and reprint
        tput cuu "$num" > "$TTY" 2>/dev/null || true
        for i in "${!options[@]}"; do
            if [ "$i" -eq "$SELECTED" ]; then
                echo -e "  ${C4}${BOLD} ▶  ${options[$i]}${RESET}" > "$TTY"
            else
                echo -e "  ${DIM}    ${options[$i]}${RESET}" > "$TTY"
            fi
        done
    done

    tput cnorm > "$TTY" 2>/dev/null || true
    echo > "$TTY"
}

# ─── Spinner ──────────────────────────────────────────────────────────────────
# Usage: start_spinner "message" — stores PID in $SPINNER_PID
SPINNER_PID=""
start_spinner() {
    local msg="$1"
    local frames=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
    (
        local i=0
        while true; do
            printf "\r  ${C3}%s${RESET} %s  " "${frames[$i]}" "$msg"
            i=$(( (i + 1) % ${#frames[@]} ))
            sleep 0.1
        done
    ) &
    SPINNER_PID=$!
    disown "$SPINNER_PID" 2>/dev/null || true
}

stop_spinner() {
    if [[ -n "${SPINNER_PID:-}" ]]; then
        kill "$SPINNER_PID" 2>/dev/null || true
        wait "$SPINNER_PID" 2>/dev/null || true
        SPINNER_PID=""
    fi
    printf "\r\033[K"  # clear the spinner line
}

# ─── OS install hints ─────────────────────────────────────────────────────────
python_install_hint() {
    local os_type
    os_type="$(uname -s)"
    echo
    echo -e "  ${C3}${BOLD}Install Python 3.9+:${RESET}"
    case "$os_type" in
        Linux)
            echo -e "  ${DIM}Ubuntu/Debian:  sudo apt install python3.11 python3.11-venv${RESET}"
            echo -e "  ${DIM}Fedora/RHEL:    sudo dnf install python3.11${RESET}"
            echo -e "  ${DIM}Arch:           sudo pacman -S python${RESET}"
            echo -e "  ${DIM}Or via pyenv:   https://github.com/pyenv/pyenv${RESET}"
            ;;
        Darwin)
            echo -e "  ${DIM}Homebrew:   brew install python@3.12${RESET}"
            echo -e "  ${DIM}Official:   https://www.python.org/downloads/${RESET}"
            ;;
        *)
            echo -e "  ${DIM}https://www.python.org/downloads/${RESET}"
            ;;
    esac
    echo
}

# ─── 1. Dependency checks ─────────────────────────────────────────────────────
check_deps() {
    step "Checking dependencies"

    if [ "$INSTALL_METHOD" -eq 0 ]; then
        # Python 3.9+
        local py_found=false
        for cmd in python3.12 python3.11 python3.10 python3.9 python3 python; do
            if command -v "$cmd" &>/dev/null; then
                local ver
                ver="$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
                local major minor
                major="${ver%%.*}"
                minor="${ver#*.}"
                if [[ "$major" -gt 3 ]] || { [[ "$major" -eq 3 ]] && [[ "$minor" -ge 9 ]]; }; then
                    PYTHON_CMD="$cmd"
                    py_found=true
                    success "Python ${ver} (${cmd})"
                    break
                fi
            fi
        done
        if ! $py_found; then
            error "Python 3.9+ not found."
            python_install_hint
            fatal "Please install Python 3.9 or newer and re-run this script."
        fi

        if ! "$PYTHON_CMD" -m venv --help &>/dev/null; then
            error "Python venv module not available for ${PYTHON_CMD}."
            echo -e "  ${DIM}Ubuntu/Debian: sudo apt install python3-venv${RESET}"
            fatal "python3-venv is required for clone installs."
        fi
        success "python venv"
    fi

    # curl
    if ! command -v curl &>/dev/null; then
        error "curl not found."
        echo -e "  ${DIM}Ubuntu/Debian: sudo apt install curl${RESET}"
        echo -e "  ${DIM}macOS:         brew install curl${RESET}"
        fatal "curl is required. Please install it and re-run."
    fi
    success "curl"
}

check_git() {
    if ! command -v git &>/dev/null; then
        error "git not found."
        echo -e "  ${DIM}Ubuntu/Debian: sudo apt install git${RESET}"
        echo -e "  ${DIM}macOS:         brew install git  (or install Xcode CLI tools)${RESET}"
        fatal "git is required for the clone install method."
    fi
    success "git"
}

# ─── 2. Install method prompt ─────────────────────────────────────────────────
choose_install_method() {
    if [[ -n "$INSTALL_METHOD" ]]; then
        if [[ "$INSTALL_METHOD" -eq 0 ]]; then
            info "Install method: clone"
        else
            info "Install method: release"
        fi
        return
    fi

    if [[ "$NON_INTERACTIVE" == true ]]; then
        INSTALL_METHOD=1
        info "Non-interactive mode detected: defaulting to release install"
        return
    fi

    if [[ ! -r "$TTY" || ! -w "$TTY" ]]; then
        INSTALL_METHOD=1
        info "No interactive TTY found: defaulting to release install"
        return
    fi

    step "Installation method"

    echo -e "  ${DIM}Choose how to get ragtag:${RESET}"
    echo
    echo -e "  ${C2}${BOLD}Release${RESET}  — pre-built executables, nothing else to install"
    echo -e "            ${DIM}No Python or git required  |  Best for: everyone${RESET}"
    echo
    echo -e "  ${C1}Clone${RESET}    — full source code, easy to edit and hack on"
    echo -e "            ${DIM}Requires: git + Python 3.9+  |  Best for: developers${RESET}"
    echo

    select_option "Select install method  (↑/↓ to move, Enter to confirm)" \
        "Release  — binary only  ★ recommended" \
        "Clone    — full source, for developers" \
        "Exit"

    case $SELECTED in
        0) INSTALL_METHOD=1 ;;  # release
        1) INSTALL_METHOD=0 ;;  # clone
        2)
            echo
            echo -e "  ${C5}No changes made. See you next time!${RESET}"
            echo
            exit 0
            ;;
    esac
}

# ─── 3. Platform detection ───────────────────────────────────────────────────
detect_platform() {
    local os_raw arch_raw
    os_raw="$(uname -s)"
    arch_raw="$(uname -m)"

    case "$os_raw" in
        Linux)
            PLATFORM_OS="linux"
            case "$arch_raw" in
                x86_64|amd64)   PLATFORM_ARCH="amd64" ;;
                aarch64|arm64)  PLATFORM_ARCH="arm64" ;;
                *) fatal "Unsupported architecture: ${arch_raw}" ;;
            esac
            ;;
        Darwin)
            case "$arch_raw" in
                arm64|aarch64)
                    PLATFORM_OS="mac"
                    PLATFORM_ARCH="arm64"
                    ;;
                x86_64)
                    PLATFORM_OS="mac"
                    PLATFORM_ARCH="intel"
                    ;;
                *) fatal "Unsupported macOS architecture: ${arch_raw}" ;;
            esac
            ;;
        *) fatal "Unsupported operating system: ${os_raw}" ;;
    esac

    BINARY_NAME="ragtag-${PLATFORM_OS}-${PLATFORM_ARCH}"
    info "Detected platform: ${PLATFORM_OS}/${PLATFORM_ARCH} → ${BOLD}${BINARY_NAME}${RESET}"
}

# ─── 4. Download ──────────────────────────────────────────────────────────────
do_clone() {
    step "Cloning repository"
    check_git

    local default_dir="${HOME}/ragtag"
    local user_dir="${RAGTAG_DIR:-}"

    if [[ -z "$user_dir" ]]; then
        if [[ "$NON_INTERACTIVE" == true ]]; then
            user_dir="$default_dir"
            info "Using install directory: ${user_dir}"
        else
            prompt_line user_dir "  ${C3}Install directory${RESET} [${default_dir}]: "
            user_dir="${user_dir:-$default_dir}"
        fi
    fi

    # Expand tilde if typed literally in the prompt
    user_dir="${user_dir/#\~/$HOME}"

    RAGTAG_DIR="$user_dir"

    if [ -d "$RAGTAG_DIR" ]; then
        if [ -d "${RAGTAG_DIR}/.git" ]; then
            info "Existing git checkout found. Pulling latest changes…"
            git -C "$RAGTAG_DIR" pull --ff-only
            success "Repository updated at ${RAGTAG_DIR}"
            return
        fi

        if [[ -n "$(ls -A "$RAGTAG_DIR" 2>/dev/null)" ]]; then
            warn "Directory ${RAGTAG_DIR} already exists and is not empty."
            if [[ "$AUTO_YES" == true || "$NON_INTERACTIVE" == true ]]; then
                info "Using existing directory contents because non-interactive mode is enabled."
                return
            fi
            local confirm
            prompt_line confirm "  ${C3}Use existing directory without cloning? (y/N)${RESET} "
            [[ "${confirm,,}" != "y" ]] && fatal "Aborted."
            return
        fi
    fi

    info "Cloning into ${RAGTAG_DIR} …"
    git clone "https://github.com/${REPO}.git" "$RAGTAG_DIR"
    success "Repository cloned to ${RAGTAG_DIR}"

    ensure_demo_data
}

do_release() {
    step "Downloading latest release"

    local default_dir="${HOME}/ragtag"
    local user_dir="${RAGTAG_DIR:-}"

    if [[ -z "$user_dir" ]]; then
        if [[ "$NON_INTERACTIVE" == true ]]; then
            user_dir="$default_dir"
            info "Using install directory: ${user_dir}"
        else
            prompt_line user_dir "  ${C3}Install directory${RESET} [${default_dir}]: "
            user_dir="${user_dir:-$default_dir}"
        fi
    fi

    # Expand tilde if typed literally in the prompt
    user_dir="${user_dir/#\~/$HOME}"

    RAGTAG_DIR="$user_dir"
    mkdir -p "$RAGTAG_DIR"

    local dl_total=4

    # ── [1/4] TUI binary ──────────────────────────────────────────────────────
    # Try the current naming (ragtag-*) then the legacy naming (rag-tui-*) so
    # that older cached installers still work against newer releases and vice-versa.
    local tui_dest="${RAGTAG_DIR}/${BINARY_NAME}"
    local tui_downloaded=false
    local tui_try
    for tui_try in \
        "ragtag-${PLATFORM_OS}-${PLATFORM_ARCH}" \
        "rag-tui-${PLATFORM_OS}-${PLATFORM_ARCH}"; do
        echo -e "  ${DIM}[1/${dl_total}]${RESET} Downloading ${tui_try} …"
        if download_release_asset "$tui_try" "$tui_dest"; then
            chmod +x "$tui_dest"
            local sz; sz="$(du -sh "$tui_dest" 2>/dev/null | cut -f1)"
            success "${tui_try}  ${DIM}(${sz})${RESET}"
            tui_downloaded=true
            break
        fi
        warn "Not found as ${tui_try}, trying alternate name …"
    done
    if [[ "$tui_downloaded" != true ]]; then
        fatal "Could not download TUI binary for ${PLATFORM_OS}/${PLATFORM_ARCH}. Check https://github.com/${REPO}/releases for available assets."
    fi

    # ── [2/4] pipeline ────────────────────────────────────────────────────────
    echo -e "  ${DIM}[2/${dl_total}]${RESET} Downloading pipeline …"
    if download_release_asset "pipeline-${PLATFORM_OS}-${PLATFORM_ARCH}" "${RAGTAG_DIR}/pipeline"; then
        chmod +x "${RAGTAG_DIR}/pipeline"
        local sz; sz="$(du -sh "${RAGTAG_DIR}/pipeline" 2>/dev/null | cut -f1)"
        success "pipeline  ${DIM}(${sz})${RESET}"
    else
        warn "No prebuilt pipeline for this platform — will use Python fallback."
        RELEASE_PY_FALLBACK=true
    fi

    # ── [3/4] bridge ──────────────────────────────────────────────────────────
    echo -e "  ${DIM}[3/${dl_total}]${RESET} Downloading bridge …"
    if download_release_asset "bridge-${PLATFORM_OS}-${PLATFORM_ARCH}" "${RAGTAG_DIR}/bridge"; then
        chmod +x "${RAGTAG_DIR}/bridge"
        local sz; sz="$(du -sh "${RAGTAG_DIR}/bridge" 2>/dev/null | cut -f1)"
        success "bridge  ${DIM}(${sz})${RESET}"
    else
        warn "No prebuilt bridge for this platform — will use Python fallback."
        RELEASE_PY_FALLBACK=true
    fi

    # ── [4/4] nim_config.py ───────────────────────────────────────────────────
    start_spinner "[4/${dl_total}] Fetching nim_config.py …"
    if curl -fsSL \
            "https://ragtag.crsv.es/nim_config.py" \
            -o "${RAGTAG_DIR}/nim_config.py" 2>/dev/null; then
        stop_spinner
        success "nim_config.py"
    else
        stop_spinner
        warn "Could not fetch nim_config.py — you can create it manually later."
    fi

    ensure_demo_data

    if [[ "$RELEASE_PY_FALLBACK" == true ]]; then
        warn "Falling back to Python runtime for retrieval/pipeline on this platform."
        download_release_python_sources
    fi

    success "Download complete → ${RAGTAG_DIR}"
}

# ─── 5. Install binary ────────────────────────────────────────────────────────
install_binary() {
    step "Installing TUI binary"

    local binary_src
    # For clone: binary lives in ragtag/ subdir
    if [ "$INSTALL_METHOD" -eq 0 ]; then
        binary_src="${RAGTAG_DIR}/ragtag/${BINARY_NAME}"
    else
        binary_src="${RAGTAG_DIR}/${BINARY_NAME}"
    fi

    if [ ! -f "$binary_src" ]; then
        if [ "$INSTALL_METHOD" -eq 0 ]; then
            warn "Pre-built binary not found at ${binary_src}"
            info "Attempting to download ${BINARY_NAME} from release assets …"
            mkdir -p "$(dirname "$binary_src")"
            if curl -fsSL "https://github.com/${REPO}/releases/latest/download/${BINARY_NAME}" -o "$binary_src"; then
                success "Downloaded ${BINARY_NAME}"
            else
                warn "Could not download ${BINARY_NAME}."
                warn "You can build it yourself: cd ${RAGTAG_DIR}/ragtag && make"
                return
            fi
        else
            warn "Pre-built binary not found at ${binary_src}"
            warn "You can build it yourself: cd ${RAGTAG_DIR}/ragtag && make"
            return
        fi
    fi

    chmod +x "$binary_src"
    RAGTAG_BIN_PATH="$binary_src"

    mkdir -p "$INSTALL_DIR"
    local dest="${INSTALL_DIR}/ragtag"

    # Write a launcher script that embeds RAG_DIR so the TUI finds bridge/pipeline
    # regardless of what directory the user is in when they run it.
    printf '#!/bin/sh\nexport RAG_DIR="%s"\nexec "%s" "$@"\n' "$RAGTAG_DIR" "$binary_src" > "$dest"
    chmod +x "$dest"
    success "Installed launcher → ${dest}"

    if [ "$INSTALL_METHOD" -eq 1 ] && [ -f "${RAGTAG_DIR}/pipeline" ]; then
        local pipeline_dest="${INSTALL_DIR}/ragtag-pipeline"
        if ln -sf "${RAGTAG_DIR}/pipeline" "$pipeline_dest" 2>/dev/null; then
            success "Symlinked → ${pipeline_dest}"
        else
            cp "${RAGTAG_DIR}/pipeline" "$pipeline_dest"
            success "Copied → ${pipeline_dest}"
        fi
    fi

    # Remind about PATH if needed
    if ! echo ":${PATH}:" | grep -q ":${INSTALL_DIR}:"; then
        echo
        warn "${INSTALL_DIR} is not in your PATH."
        echo -e "  ${DIM}Add this to your shell profile (~/.bashrc, ~/.zshrc, etc.):${RESET}"
        echo -e "  ${C2}    export PATH=\"${INSTALL_DIR}:\$PATH\"${RESET}"
    fi
}

# ─── 6. Python dependencies ───────────────────────────────────────────────────
install_python_deps() {
    if [ "$INSTALL_METHOD" -eq 1 ] && [[ "$RELEASE_PY_FALLBACK" != true ]]; then
        return
    fi

    if [ "$INSTALL_METHOD" -eq 1 ]; then
        step "Installing Python runtime fallback"
    else
        step "Installing Python dependencies"
    fi

    if [[ -z "$PYTHON_CMD" ]]; then
        local py_found=false
        for cmd in python3.12 python3.11 python3.10 python3.9 python3 python; do
            if command -v "$cmd" &>/dev/null; then
                local ver
                ver="$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
                local major minor
                major="${ver%%.*}"
                minor="${ver#*.}"
                if [[ "$major" -gt 3 ]] || { [[ "$major" -eq 3 ]] && [[ "$minor" -ge 9 ]]; }; then
                    PYTHON_CMD="$cmd"
                    py_found=true
                    success "Python ${ver} (${cmd})"
                    break
                fi
            fi
        done
        if ! $py_found; then
            error "Python 3.9+ not found."
            python_install_hint
            fatal "Python is required for fallback mode on this platform."
        fi

        if ! "$PYTHON_CMD" -m venv --help &>/dev/null; then
            error "Python venv module not available for ${PYTHON_CMD}."
            echo -e "  ${DIM}Ubuntu/Debian: sudo apt install python3-venv${RESET}"
            fatal "python3-venv is required for fallback mode."
        fi
    fi

    local req="${RAGTAG_DIR}/requirements.txt"
    if [ ! -f "$req" ]; then
        warn "requirements.txt not found at ${req} — skipping."
        return
    fi

    local venv_dir="${RAGTAG_DIR}/.venv"
    if [ ! -d "$venv_dir" ]; then
        start_spinner "Creating virtual environment …"
        "$PYTHON_CMD" -m venv "$venv_dir"
        stop_spinner
        success "Virtual environment created"
    fi

    PYTHON_CMD="${venv_dir}/bin/python"

    start_spinner "Upgrading pip …"
    "$PYTHON_CMD" -m pip install --upgrade pip setuptools wheel --quiet >/dev/null 2>&1 || true
    stop_spinner

    start_spinner "Installing packages — first run may take 1–2 minutes …"
    if "$PYTHON_CMD" -m pip install -r "$req" --quiet 2>&1 | tee /tmp/ragtag_pip.log >/dev/null; then
        stop_spinner
        success "Python packages installed"
    else
        stop_spinner
        warn "Pinned requirements failed — trying compatible wheels …"
        start_spinner "Installing compatible packages …"
        if PIP_INDEX_URL="https://download.pytorch.org/whl/cpu" \
           PIP_EXTRA_INDEX_URL="https://pypi.org/simple" \
           "$PYTHON_CMD" -m pip install --only-binary=:all: \
                "torch==2.2.2" \
                "torchvision==0.17.2" \
                "numpy" \
                "tqdm" \
                "rank_bm25>=0.2.2" \
                "openai>=1.0.0" \
                "faiss-cpu" \
                "transformers>=4.41.0" \
                "sentence-transformers>=3.0.0" \
                --quiet 2>&1 | tee -a /tmp/ragtag_pip.log >/dev/null; then
            stop_spinner
            success "Python packages installed (compatible set)"
        else
            stop_spinner
            error "pip install failed. See /tmp/ragtag_pip.log for details."
            cat /tmp/ragtag_pip.log >&2
            fatal "Dependency install failed."
        fi
    fi

    if [ "$INSTALL_METHOD" -eq 1 ] && [[ "$RELEASE_PY_FALLBACK" == true ]]; then
        local pipeline_wrapper="${INSTALL_DIR}/ragtag-pipeline"
        cat > "$pipeline_wrapper" <<EOF
#!/usr/bin/env bash
exec "${RAGTAG_DIR}/.venv/bin/python" "${RAGTAG_DIR}/pipeline.py" "\$@"
EOF
        chmod +x "$pipeline_wrapper"
        success "Installed wrapper → ${pipeline_wrapper}"

        if [[ -n "$RAGTAG_BIN_PATH" ]]; then
            local ragtag_wrapper="${INSTALL_DIR}/ragtag"
            cat > "$ragtag_wrapper" <<EOF
#!/usr/bin/env bash
if [ -x "${RAGTAG_DIR}/.venv/bin/python" ]; then
    export RAGTAG_PYTHON="${RAGTAG_DIR}/.venv/bin/python"
fi
exec "${RAGTAG_BIN_PATH}" "\$@"
EOF
            chmod +x "$ragtag_wrapper"
            success "Installed wrapper → ${ragtag_wrapper}"
        fi
    fi
}

# ─── 7. NIM API key ───────────────────────────────────────────────────────────
configure_nim_key() {
    step "NIM API key (NVIDIA)"

    local config="${RAGTAG_DIR}/nim_config.py"
    if [ ! -f "$config" ]; then
        warn "nim_config.py not found at ${config}; attempting to download it."
        if curl -fsSL "https://ragtag.crsv.es/nim_config.py" -o "$config"; then
            success "Downloaded nim_config.py"
        else
            warn "Could not download nim_config.py — cannot write key."
            return
        fi
    fi

    if [[ -n "$NIM_API_KEY_ARG" ]]; then
        local tmp
        tmp="$(mktemp)"
        while IFS= read -r line; do
            if [[ "$line" == NIM_API_KEY* ]]; then
                printf 'NIM_API_KEY = "%s"\n' "$NIM_API_KEY_ARG"
            else
                printf '%s\n' "$line"
            fi
        done < "$config" > "$tmp"
        mv "$tmp" "$config"
        success "NIM API key written to nim_config.py"
        return
    fi

    if [[ "$SKIP_NIM_KEY" == true || "$NON_INTERACTIVE" == true ]]; then
        info "Skipped — edit ${RAGTAG_DIR}/nim_config.py later."
        return
    fi

    echo -e "  ${DIM}ragtag uses NVIDIA NIM for LLM inference.${RESET}"
    echo -e "  ${DIM}Get a free key at: ${C3}https://build.nvidia.com${RESET}"
    echo
    local yn
    prompt_line yn "  ${C3}Set your NIM API key now? (y/N)${RESET} "
    [[ "${yn,,}" != "y" ]] && { info "Skipped — edit ${RAGTAG_DIR}/nim_config.py later."; return; }

    # Read without echo for security
    local key
    prompt_secret key "  ${C4}${BOLD}NIM_API_KEY${RESET}: "

    if [[ -z "$key" ]]; then
        warn "Empty key entered — skipping."
        return
    fi

    # Replace the NIM_API_KEY line safely (no sed -i portability issues)
    local tmp
    tmp="$(mktemp)"
    while IFS= read -r line; do
        if [[ "$line" == NIM_API_KEY* ]]; then
            printf 'NIM_API_KEY = "%s"\n' "$key"
        else
            printf '%s\n' "$line"
        fi
    done < "$config" > "$tmp"
    mv "$tmp" "$config"

    success "NIM API key written to nim_config.py"
}

# ─── 8. Auto-index demo data ──────────────────────────────────────────────────
auto_index_demo() {
    local demo_file="${RAGTAG_DIR}/raw/slack.json"
    local processed_dir="${RAGTAG_DIR}/processed"

    # Skip silently if demo data isn't there
    [ -f "$demo_file" ] || return

    # Skip if already indexed
    if [ -d "$processed_dir" ] && [ -n "$(ls -A "$processed_dir" 2>/dev/null)" ]; then
        info "Demo data already indexed — skipping."
        return
    fi

    # Skip in non-interactive mode
    if [[ "$NON_INTERACTIVE" == true ]]; then
        return
    fi

    step "Index demo data"

    echo -e "  ${DIM}ragtag needs to embed and index your data before you can search it.${RESET}"
    echo -e "  ${DIM}This downloads a small (~90 MB) embedding model on first run and${RESET}"
    echo -e "  ${DIM}processes the demo Slack export into a local vector store.${RESET}"
    echo -e "  ${DIM}It typically takes about 1 minute on a modern laptop.${RESET}"
    echo

    local yn
    prompt_line yn "  ${C3}Index the demo data now? (Y/n)${RESET} "
    if [[ "${yn,,}" == "n" ]]; then
        info "Skipped — run the pipeline yourself when ready."
        return
    fi

    echo

    # Determine how to invoke the pipeline
    local pipeline_cmd=()
    if [ "$INSTALL_METHOD" -eq 0 ] || [[ "$RELEASE_PY_FALLBACK" == true ]]; then
        local venv_python="${RAGTAG_DIR}/.venv/bin/python"
        if [ -x "$venv_python" ]; then
            pipeline_cmd=("$venv_python" "${RAGTAG_DIR}/pipeline.py")
        elif [[ -n "$PYTHON_CMD" ]]; then
            pipeline_cmd=("$PYTHON_CMD" "${RAGTAG_DIR}/pipeline.py")
        else
            warn "No Python interpreter found — cannot auto-index."
            return
        fi
    else
        local pipeline_bin="${RAGTAG_DIR}/pipeline"
        if [ ! -x "$pipeline_bin" ]; then
            warn "Pipeline binary not found — cannot auto-index."
            return
        fi
        pipeline_cmd=("$pipeline_bin")
    fi

    info "Running: ${pipeline_cmd[*]} raw/slack.json"
    echo

    if (cd "$RAGTAG_DIR" && "${pipeline_cmd[@]}" raw/slack.json); then
        echo
        success "Demo data indexed successfully"
    else
        echo
        warn "Indexing failed. You can retry manually:"
        echo -e "  ${DIM}cd ${RAGTAG_DIR} && ${pipeline_cmd[*]} raw/slack.json${RESET}"
    fi
}

# ─── Main ────────────────────────────────────────────────────────────────────
main() {
    parse_args "$@"

    if [[ "$SKIP_BANNER" != true ]]; then
        print_banner
    fi

    choose_install_method
    check_deps
    detect_platform

    if [ "$INSTALL_METHOD" -eq 0 ]; then
        do_clone
    else
        do_release
    fi

    install_binary
    install_python_deps
    configure_nim_key
    auto_index_demo

    # Auto-launch the TUI once installation is done.
    if [[ "$NON_INTERACTIVE" != true && -n "${RAGTAG_BIN_PATH:-}" && -x "$RAGTAG_BIN_PATH" ]]; then
        echo -e "  ${C1}${BOLD}Launching ragtag…${RESET}"
        echo
        cd "$RAGTAG_DIR"
        exec "$RAGTAG_BIN_PATH"
    fi
}

main "$@"