#!/usr/bin/env bash
# Set up a control machine (a small Ubuntu/Debian server; no GPU needed). Safe to run again.
#   git clone git@github.com:<you>/box-boy.git && cd box-boy && ./setup.sh
# Installs box-boy into its own virtualenv and puts `bb`, `vastai` and `hf` on your PATH.
# It does not log you in anywhere; it ends by telling you which logins are still missing.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SUDO=""; [ "$(id -u)" != "0" ] && SUDO="sudo"

need=()
command -v git >/dev/null || need+=(git)
command -v curl >/dev/null || need+=(curl)
python3 -c "import venv, ensurepip" 2>/dev/null || need+=(python3-venv)
command -v gh >/dev/null || need+=(gh)
command -v tmux >/dev/null || need+=(tmux)
if [ ${#need[@]} -gt 0 ]; then
    echo "installing: ${need[*]}"
    $SUDO apt-get update -qq && $SUDO apt-get install -y -qq "${need[@]}"
fi
python3 -c "import sys; assert sys.version_info >= (3, 11), 'box-boy needs Python 3.11 or newer'"

[ -d "$HERE/.venv" ] || python3 -m venv "$HERE/.venv"
"$HERE/.venv/bin/pip" install -q -U pip
"$HERE/.venv/bin/pip" install -q -e "$HERE[dev]"

mkdir -p "$HOME/.local/bin"
for tool in bb vastai hf; do ln -sf "$HERE/.venv/bin/$tool" "$HOME/.local/bin/$tool"; done
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *)
    grep -qs 'HOME/.local/bin' "$HOME/.bashrc" || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    export PATH="$HOME/.local/bin:$PATH"
    echo "added ~/.local/bin to PATH in ~/.bashrc (open a new shell, or: source ~/.bashrc)";;
esac
[ -f "$HOME/.ssh/id_ed25519.pub" ] || ls "$HOME"/.ssh/id_*.pub >/dev/null 2>&1 \
    || { mkdir -p "$HOME/.ssh" && ssh-keygen -q -t ed25519 -N "" -f "$HOME/.ssh/id_ed25519" && echo "created ~/.ssh/id_ed25519"; }

echo
echo "installed. Log in to the three services (each once; skip the ones already done):"
echo "  vastai set api-key <key>     # console.vast.ai > Account > API Keys; needs instance-create permission"
echo "  hf auth login                # a Hugging Face token with write access"
echo "  gh auth login                # GitHub, so you can clone and push the project"
echo "then, inside the project checkout:"
echo "  bb doctor"
