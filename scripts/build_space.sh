#!/usr/bin/env bash
# Assemble the Hugging Face Space in build/space: the app, the source it
# imports, the Space card, and a flat requirements.txt (Spaces cannot follow
# "-r" includes). Torch comes from the CPU-only index: Spaces' free hardware
# has no GPU, and the CUDA build is several GB larger.
set -euo pipefail
cd "$(dirname "$0")/.."

out=build/space
rm -rf "$out"
mkdir -p "$out"

cp app.py "$out/"
cp -r src "$out/src"
cp space/README.md "$out/README.md"
find "$out" -name "__pycache__" -type d -prune -exec rm -rf {} +

{
  echo "--extra-index-url https://download.pytorch.org/whl/cpu"
  grep -hvE '^\s*(#|-r|$)' requirements.txt requirements-app.txt
} > "$out/requirements.txt"

echo "Built $out:"
find "$out" -type f | sort
