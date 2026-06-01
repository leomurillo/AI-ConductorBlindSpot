# Building `ConductorBlindSpot.pdf`

This document describes how to reproduce the PDF from the canonical markdown
source (`ConductorBlindSpot.md`). The build is deliberately minimal and fully
offline after the one-time toolchain install.

## What the build does

1. Parses `ConductorBlindSpot.md` and extracts:
   - the H1 title,
   - the `**Author**` line,
   - the `*Month DD, YYYY*` date line,
   - the `## Abstract` block (everything between that heading and the next
     horizontal rule `---`).
2. Writes a preprocessed markdown file with a YAML front matter block
   containing `title`, `author`, `date`, `abstract`; the body that follows
   it is the manuscript minus the front-matter title block.
3. Invokes `pandoc` to convert the preprocessed markdown to a standalone
   LaTeX document, injecting `latex/preamble.tex` as extra header.
4. Invokes `latexmk -xelatex` to compile the LaTeX to PDF (2 passes for
   table of contents / cross-references).
5. Copies the final PDF next to the source as `ConductorBlindSpot.pdf`.

Intermediate artifacts live under `build/` and are safe to delete at any
time. Pass `-Clean` (PowerShell) or `--clean` (bash) to force a fresh build.
For the full repository validation checklist, see
[`docs/validation.md`](docs/validation.md).

## Directory layout

```
.
├── ConductorBlindSpot.md   # canonical source
├── ConductorBlindSpot.pdf  # build output (gitignored OK)
├── latex/
│   └── preamble.tex        # xelatex preamble injected via pandoc
├── build/                  # pandoc + latexmk intermediates
├── build.ps1               # Windows / PowerShell build script
├── build.sh                # POSIX build script
└── BUILD.md                # this file
```

## Windows (PowerShell)

```powershell
# From this folder:
pwsh ./build.ps1             # incremental
pwsh ./build.ps1 -Clean      # wipe build/ first (recommended after edits)
pwsh ./build.ps1 -SkipPdf    # stop after generating build/paper.tex
```

## Linux / macOS

```bash
bash build.sh                # incremental
bash build.sh --clean        # wipe build/ first
bash build.sh --skip-pdf     # stop after generating build/paper.tex
```

One-time toolchain installation requires Pandoc plus a XeLaTeX-capable TeX
distribution (`latexmk` and `xelatex`). On macOS, `brew install pandoc` plus
MacTeX or BasicTeX with the standard LaTeX packages is sufficient; on Linux,
install `pandoc`, `texlive-xetex`, `latexmk`, `texlive-fonts-recommended`, and
`texlive-latex-extra` or the distribution equivalents.

## Expected output (smoke test)

A successful run ends with a line of the form

```
OK   .../ConductorBlindSpot.pdf  (XXX KB)
```

and produces a PDF containing:

- Title, author, date (from the top of the markdown);
- Abstract (native LaTeX `abstract` environment);
- A clickable table of contents (sections and subsections);
- Body sections, Acknowledgments, References, Appendices.

## Notes

- **Working filename.** The source filename `ConductorBlindSpot.md` is stable
  through revision; the published title is finalized as a late editorial pass
  and changes only the H1 inside the file, not the filename.
- **Front-matter parsing.** `build.ps1` / `build.sh` identify the title /
  author / date by regex. If the top of the markdown is restructured (author
  line no longer `**...**`, etc.), the scripts will fail loudly with a
  specific "Could not parse ..." message.
