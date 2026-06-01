#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Build ConductorBlindSpot.pdf from its markdown source (POSIX).
# Mirrors build.ps1.  Requires: pandoc, xelatex, latexmk.
#   Debian/Ubuntu: sudo apt install pandoc texlive-xetex latexmk \
#                                   texlive-fonts-recommended texlive-latex-extra
#   macOS (brew):  brew install pandoc && brew install --cask mactex-no-gui
# -----------------------------------------------------------------------------
set -euo pipefail

SOURCE="ConductorBlindSpot.md"
CLEAN=0
SKIP_PDF=0
for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN=1 ;;
        --skip-pdf) SKIP_PDF=1 ;;
        *) SOURCE="$arg" ;;
    esac
done

ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/$SOURCE"
BUILD_DIR="$ROOT/build"
LATEX_DIR="$ROOT/latex"
PREAMBLE="$LATEX_DIR/preamble.tex"
PDF_NAME="${SOURCE%.md}.pdf"
FINAL_PDF="$ROOT/$PDF_NAME"

[[ -f "$SRC" ]]      || { echo "Source not found: $SRC"      >&2; exit 1; }
[[ -f "$PREAMBLE" ]] || { echo "Preamble not found: $PREAMBLE" >&2; exit 1; }
command -v pandoc  >/dev/null || { echo "pandoc not found on PATH"  >&2; exit 1; }
command -v latexmk >/dev/null || { echo "latexmk not found on PATH" >&2; exit 1; }
command -v xelatex >/dev/null || { echo "xelatex not found on PATH" >&2; exit 1; }

# Preserve build/utils (shared maintenance scripts); wipe the rest.
[[ "$CLEAN" -eq 1 && -d "$BUILD_DIR" ]] && find "$BUILD_DIR" -mindepth 1 -maxdepth 1 ! -name utils -exec rm -rf {} + || true
mkdir -p "$BUILD_DIR"

# --- Parse title / author / date --------------------------------------------
TITLE="$(grep -m1 -E '^# '       "$SRC" | sed 's/^# //')"
AUTHOR_RAW="$(grep -m1 -E '^\*\*[^*]+\*\*$' "$SRC")"
AUTHOR="${AUTHOR_RAW#\*\*}"; AUTHOR="${AUTHOR%\*\*}"
DATE_RAW="$(grep -m1 -E '^\*[A-Z][a-z]+ [0-9]{1,2}, [0-9]{4}\*$' "$SRC")"
DATE="${DATE_RAW#\*}"; DATE="${DATE%\*}"

[[ -n "$TITLE"  ]] || { echo "Could not parse title from $SOURCE"  >&2; exit 1; }
[[ -n "$AUTHOR" ]] || { echo "Could not parse author from $SOURCE" >&2; exit 1; }
[[ -n "$DATE"   ]] || { echo "Could not parse date from $SOURCE"   >&2; exit 1; }

# --- Strip the title block; extract abstract --------------------------------
HR1=$(grep -n -m1 -E '^---[[:space:]]*$' "$SRC" | head -n1 | cut -d: -f1)
[[ -n "$HR1" ]] || { echo "No '---' found to close title block" >&2; exit 1; }

BODY_TMP="$BUILD_DIR/_body.md"
tail -n +$((HR1 + 1)) "$SRC" > "$BODY_TMP"

ABSTRACT="$(awk '
    /^## Abstract[[:space:]]*$/ { inside=1; next }
    inside && /^---[[:space:]]*$/ { inside=0; next }
    inside { print }
' "$BODY_TMP")"

BODY_STRIPPED="$(awk '
    /^## Abstract[[:space:]]*$/ { inside=1; next }
    inside && /^---[[:space:]]*$/ { inside=0; next }
    !inside { print }
' "$BODY_TMP")"

# --- Emit prepared markdown with YAML header -------------------------------
PREPPED="$BUILD_DIR/paper.md"
{
    printf -- '---\n'
    printf 'title: "%s"\n'  "${TITLE//\"/\\\"}"
    printf 'author: "%s"\n' "${AUTHOR//\"/\\\"}"
    printf 'date: "%s"\n'   "${DATE//\"/\\\"}"
    printf 'abstract: |\n'
    printf '%s\n' "$ABSTRACT" | sed 's/^/  /'
    printf -- '---\n\n'
    printf '%s\n' "$BODY_STRIPPED"
} > "$PREPPED"

# --- Pandoc: markdown -> standalone .tex -----------------------------------
TEX="$BUILD_DIR/paper.tex"
pandoc "$PREPPED" \
    --from=markdown+tex_math_dollars+tex_math_single_backslash+pipe_tables+smart+implicit_figures+autolink_bare_uris \
    --to=latex \
    --output="$TEX" \
    --standalone \
    --wrap=preserve \
    --pdf-engine=xelatex \
    --top-level-division=section \
    --table-of-contents --toc-depth=3 \
    --include-in-header="$PREAMBLE" \
    -V documentclass=article \
    -V classoption=11pt \
    -V papersize=letter \
    -V geometry:margin=1in \
    -V colorlinks=true

echo "Wrote TeX: $TEX"

[[ "$SKIP_PDF" -eq 1 ]] && { echo "--skip-pdf specified; stopping."; exit 0; }

# --- latexmk: .tex -> .pdf -------------------------------------------------
( cd "$BUILD_DIR" && latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error paper.tex )

cp "$BUILD_DIR/paper.pdf" "$FINAL_PDF"
printf 'OK   %s  (%s bytes)\n' "$FINAL_PDF" "$(wc -c < "$FINAL_PDF")"
