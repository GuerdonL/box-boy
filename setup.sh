#!/usr/bin/env bash
# Set up a control machine (a small Ubuntu/Debian server; no GPU needed). Safe to run again.
#   git clone git@github.com:<you>/box-boy.git && cd box-boy && ./setup.sh
# Installs box-boy into its own virtualenv and puts `bb`, `vastai` and `hf` on your PATH.
# It does not log you in anywhere; it ends by telling you which logins are still missing.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
if [ "$(id -u)" = "0" ] && [ -n "${SUDO_USER:-}" ]; then
    echo "run ./setup.sh as yourself, not with sudo: it asks for sudo itself only to install packages." >&2
    echo "(under sudo, the virtualenv ends up owned by root and \`bb\` goes on root's PATH, not yours)" >&2
    exit 1
fi
SUDO=""; [ "$(id -u)" != "0" ] && SUDO="sudo"
for d in "$HERE/.venv" "$HERE/boxboy.egg-info"; do
    if [ -e "$d" ] && [ ! -O "$d" ]; then
        echo "$d is owned by $(stat -c %U "$d") (probably an earlier run under sudo). Remove it, then run this again:" >&2
        echo "  sudo rm -rf $HERE/.venv $HERE/boxboy.egg-info" >&2
        exit 1
    fi
done

need=()
command -v git >/dev/null || need+=(git)
command -v curl >/dev/null || need+=(curl)
# Use the newest python3.x >= 3.11 on PATH; plain `python3` may be older (3.10 on Ubuntu 22.04).
PY=""
for c in python3.13 python3.12 python3.11 python3; do
    command -v "$c" >/dev/null && "$c" -c "import sys; sys.exit(sys.version_info < (3, 11))" 2>/dev/null && { PY=$(command -v "$c"); break; }
done
if [ -z "$PY" ]; then
    need+=(python3.11 python3.11-venv); PY=python3.11
elif ! "$PY" -c "import venv, ensurepip" 2>/dev/null; then
    need+=("$(basename "$PY")-venv")
fi
command -v gh >/dev/null || need+=(gh)
command -v tmux >/dev/null || need+=(tmux)
if [ ${#need[@]} -gt 0 ]; then
    echo "installing: ${need[*]}"
    $SUDO apt-get update -qq && $SUDO apt-get install -y -qq "${need[@]}"
fi
"$PY" -c "import sys; assert sys.version_info >= (3, 11), 'box-boy needs Python 3.11 or newer'"

# Rebuild a venv left over from an older Python (e.g. an earlier run that used 3.10).
if [ -x "$HERE/.venv/bin/python" ] && ! "$HERE/.venv/bin/python" -c "import sys; sys.exit(sys.version_info < (3, 11))" 2>/dev/null; then
    echo "removing old .venv (Python < 3.11)"; rm -rf "$HERE/.venv"
fi
[ -d "$HERE/.venv" ] || "$PY" -m venv "$HERE/.venv"
"$HERE/.venv/bin/pip" install -q -U pip
"$HERE/.venv/bin/pip" install -q -e "$HERE[dev]"

mkdir -p "$HOME/.local/bin"
for tool in bb vastai hf; do ln -sf "$HERE/.venv/bin/$tool" "$HOME/.local/bin/$tool"; done
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *)
    grep -qs 'HOME/.local/bin' "$HOME/.bashrc" || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    export PATH="$HOME/.local/bin:$PATH"
    echo "added ~/.local/bin to PATH in ~/.bashrc (open a new shell, or: source ~/.bashrc)";;
esac
# fish does not read ~/.bashrc; fish_add_path is a no-op if the directory is already there
command -v fish >/dev/null && fish -c 'fish_add_path -U ~/.local/bin' 2>/dev/null || true
[ -f "$HOME/.ssh/id_ed25519.pub" ] || ls "$HOME"/.ssh/id_*.pub >/dev/null 2>&1 \
    || { mkdir -p "$HOME/.ssh" && ssh-keygen -q -t ed25519 -N "" -f "$HOME/.ssh/id_ed25519" && echo "created ~/.ssh/id_ed25519"; }

echo
echo "installed. Log in to the three services (each once; skip the ones already done):"
echo "  vastai set api-key <key>     # console.vast.ai > Account > API Keys; needs instance-create permission"
echo "  hf auth login                # a Hugging Face token with write access"
echo "  gh auth login                # GitHub, so you can clone and push the project"
echo "then, inside the project checkout:"
echo "  bb doctor"
