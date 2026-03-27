#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"
uv run python scripts/build_report_assets.py

cd "$ROOT_DIR/reports"
export TEXINPUTS="../Formatting_Instructions_For_NeurIPS_2026:${TEXINPUTS:-}"
latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error final_report.tex
