<#
.SYNOPSIS
  Build the ConductorBlindSpot.pdf from its markdown source.

.DESCRIPTION
  Converts `ConductorBlindSpot.md` to LaTeX via pandoc, then compiles
  a PDF with XeLaTeX (via latexmk). Outputs `ConductorBlindSpot.pdf`
  next to the source. Intermediate files live under `build/`.

.PARAMETER Source
  Markdown source file (default: ConductorBlindSpot.md).

.PARAMETER Clean
  Wipe the `build/` directory before building (forces a fresh run).

.PARAMETER SkipPdf
  Only generate the `.tex`; do not invoke latexmk. Useful for debugging.

.EXAMPLE
  pwsh ./build.ps1

.EXAMPLE
  pwsh ./build.ps1 -Clean
#>
[CmdletBinding()]
param(
    [string]$Source  = 'ConductorBlindSpot.md',
    [switch]$Clean,
    [switch]$SkipPdf
)

$ErrorActionPreference = 'Stop'

# --- Paths -------------------------------------------------------------------
$root     = Split-Path -Parent $PSCommandPath
$srcPath  = Join-Path $root $Source
if (-not (Test-Path $srcPath)) { throw "Source not found: $srcPath" }

$buildDir = Join-Path $root 'build'
$latexDir = Join-Path $root 'latex'
$preamble = Join-Path $latexDir 'preamble.tex'
if (-not (Test-Path $preamble)) { throw "Preamble not found: $preamble" }

$pdfName  = [System.IO.Path]::ChangeExtension($Source, '.pdf')
$finalPdf = Join-Path $root $pdfName

if ($Clean -and (Test-Path $buildDir)) { Remove-Item $buildDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

# --- Locate tools ------------------------------------------------------------
# Refresh PATH from Machine + User so a freshly-installed pandoc/miktex is found
# without opening a new shell.
$env:PATH = [System.Environment]::GetEnvironmentVariable('PATH', 'Machine') + ';' +
            [System.Environment]::GetEnvironmentVariable('PATH', 'User')   + ';' +
            $env:PATH

function Get-Tool {
    param([string]$Name, [string[]]$Fallback = @())
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in $Fallback) {
        if (Test-Path $p) { return $p }
    }
    throw "Tool not found on PATH: $Name (install via: winget install JohnMacFarlane.Pandoc; winget install MiKTeX.MiKTeX)"
}

$pandoc  = Get-Tool 'pandoc'  @("$env:LOCALAPPDATA\Pandoc\pandoc.exe")
$latexmk = Get-Tool 'latexmk' @("$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64\latexmk.exe")

# Make sure xelatex is resolvable by latexmk (it is, if miktex is on PATH)
Get-Tool 'xelatex' @("$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64\xelatex.exe") | Out-Null

Write-Host ("Using pandoc : {0}" -f $pandoc)
Write-Host ("Using latexmk: {0}" -f $latexmk)

# --- Pre-process: strip the markdown title block ----------------------------
# The MD source carries a human-readable title block (lines 1..N up to the
# first `---` horizontal rule). Pandoc gets a pruned body plus explicit
# --metadata title/author/date so the PDF has a native \maketitle + abstract.
$lines = Get-Content $srcPath -Encoding utf8

# Parse title / author / date from the source
$title  = ($lines | Where-Object { $_ -match '^# \S' } | Select-Object -First 1) -replace '^#\s+',''
if (-not $title) { throw "Could not parse title (H1) from $Source" }

$authorLine = ($lines | Where-Object { $_ -match '^\*\*[^*]+\*\*$' } | Select-Object -First 1)
if (-not $authorLine) { throw "Could not parse author (**...**) from $Source" }
$author = $authorLine -replace '^\*\*|\*\*$',''

# Date: take the date field we wrote during authoring (matches "*Month DD, YYYY*")
$dateLine = ($lines | Where-Object { $_ -match '^\*[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\*$' } | Select-Object -First 1)
if (-not $dateLine) { throw "Could not parse date (*Month DD, YYYY*) from $Source" }
$date = $dateLine -replace '^\*|\*$',''

# Find the first `---` horizontal rule that closes the title block.
$hrIdx = $null
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match '^---\s*$') { $hrIdx = $i; break }
}
if ($null -eq $hrIdx) { throw "Could not find the title-block terminator (first '---' line)" }

# Body = everything after the first '---' horizontal rule
$body = ($lines[($hrIdx + 1)..($lines.Count - 1)] -join "`n")

# Extract abstract (between '## Abstract' and the next '---')
$abstract = $null
$absStart = $null; $absEnd = $null
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($null -eq $absStart -and $lines[$i] -match '^##\s+Abstract\s*$') { $absStart = $i + 1; continue }
    if ($null -ne $absStart -and $lines[$i] -match '^---\s*$')           { $absEnd   = $i - 1; break }
}
if ($null -ne $absStart -and $null -ne $absEnd -and $absEnd -ge $absStart) {
    $abstract = ($lines[$absStart..$absEnd] -join "`n").Trim()
}

# Strip the original Abstract block from the body so it isn't duplicated
if ($null -ne $absStart) {
    $keep = @()
    $skipping = $false
    for ($i = $hrIdx + 1; $i -lt $lines.Count; $i++) {
        if (-not $skipping -and $lines[$i] -match '^##\s+Abstract\s*$') { $skipping = $true; continue }
        if ($skipping -and $lines[$i] -match '^---\s*$')                { $skipping = $false; continue }
        if ($skipping) { continue }
        $keep += $lines[$i]
    }
    $body = ($keep -join "`n").TrimStart("`r","`n")
}

# YAML front matter for pandoc
function Esc-Yaml([string]$s) { return $s -replace '"','\"' }

$yaml = @"
---
title: "$(Esc-Yaml $title)"
author: "$(Esc-Yaml $author)"
date: "$(Esc-Yaml $date)"
abstract: |
$(
    if ($abstract) {
        ($abstract -split "`n" | ForEach-Object { '  ' + $_ }) -join "`n"
    } else { '  ' }
)
---

"@

$preppedMd = Join-Path $buildDir 'paper.md'
Set-Content -Path $preppedMd -Value ($yaml + $body) -Encoding utf8
Write-Host ("Wrote prepped markdown: {0}" -f $preppedMd)

# --- Pandoc: markdown -> standalone .tex ------------------------------------
$texFile = Join-Path $buildDir 'paper.tex'

$pandocArgs = @(
    $preppedMd,
    '--from=markdown+tex_math_dollars+tex_math_single_backslash+pipe_tables+smart+implicit_figures+autolink_bare_uris',
    '--to=latex',
    "--output=$texFile",
    '--standalone',
    '--wrap=preserve',
    '--pdf-engine=xelatex',
    '--top-level-division=section',
    '--table-of-contents',
    '--toc-depth=3',
    "--include-in-header=$preamble",
    '-V', 'documentclass=article',
    '-V', 'classoption=11pt',
    '-V', 'papersize=letter',
    '-V', 'geometry:margin=1in',
    '-V', 'colorlinks=true',
    '-V', 'mainfont=Latin Modern Roman',
    '-V', 'mathfont=Latin Modern Math',
    '-V', 'monofont=Latin Modern Mono'
)

Write-Host 'Running pandoc...'
& $pandoc @pandocArgs
if ($LASTEXITCODE -ne 0) { throw "pandoc failed with exit code $LASTEXITCODE" }
Write-Host ("Wrote TeX: {0}" -f $texFile)

if ($SkipPdf) {
    Write-Host '-SkipPdf specified; stopping before PDF build.'
    return
}

# --- latexmk: .tex -> .pdf --------------------------------------------------
Write-Host 'Running latexmk (xelatex)...'
Push-Location $buildDir
try {
    & $latexmk '-xelatex' '-interaction=nonstopmode' '-halt-on-error' '-file-line-error' 'paper.tex'
    if ($LASTEXITCODE -ne 0) {
        throw "latexmk failed with exit code $LASTEXITCODE (see $buildDir\paper.log)"
    }
} finally {
    Pop-Location
}

# --- Publish ---------------------------------------------------------------
$builtPdf = Join-Path $buildDir 'paper.pdf'
if (-not (Test-Path $builtPdf)) { throw "Expected PDF not produced: $builtPdf" }
Copy-Item $builtPdf $finalPdf -Force

$info = Get-Item $finalPdf
Write-Host ''
Write-Host ('OK   {0}  ({1:N0} KB)' -f $finalPdf, ($info.Length / 1KB))
