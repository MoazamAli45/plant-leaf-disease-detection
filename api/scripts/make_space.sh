#!/usr/bin/env bash
# Assemble a Hugging Face Docker Space in api/space/build/ (see docs/deployment.md §3).
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f model/inference.keras || ! -f model/classes.json ]]; then
  echo "api/model/ is missing. Run: (cd ml && .venv/bin/python -m leafdx.train.export_web api-model)" >&2
  exit 1
fi

out=space/build
rm -rf "$out"
mkdir -p "$out"
cp Dockerfile .dockerignore pyproject.toml "$out"/
cp space/README.md "$out"/README.md
cp -R src model "$out"/
find "$out" -name __pycache__ -type d -prune -exec rm -rf {} +
echo "Space assembled in api/$out ($(du -sh "$out" | cut -f1))."
echo "Upload: hf upload <user>/leafdx-api api/$out . --repo-type=space"
