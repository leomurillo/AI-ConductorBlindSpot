# -*- coding: utf-8 -*-
"""add_internal_links.py — T-series (ל-series) internal-hyperlink generator.

Brings a manuscript's internal cross-references up to the PDF-clickable house
style shared by the ל-series and the Conductor Blind Spot paper. Two LaTeX
mechanisms (raw TeX passed through by pandoc's `raw_tex`), mirroring T7:

  * Sections   — every heading gets a `{#sec:N}` / `{#sec:N-M}` id
                 (appendices `{#sec:app-a}`, etc.). References in the text
                 `§N`, `§N.M`, `Section N`, `Appendix X[-Y]` become
                 `\\hyperref[sec:...]{...}`.
  * Statements — every numbered Theorem/Lemma/Proposition/Corollary/
                 Conjecture/Definition/Remark/Example is preceded by a
                 `\\hypertarget{stmt:N-M}{}` line; references `Theorem N.M`
                 (incl. plurals/ranges, e.g. "Theorems 5.2 and 5.6") become
                 `\\hyperlink{stmt:N-M}{...}`. Conjecture 5.8' -> stmt:5-8-prime,
                 Appendix A.2 -> stmt:a-2.

What it deliberately does NOT touch:
  * HTML comment blocks `<!-- ... -->` (editorial notes, dropped by pandoc).
  * The `## Abstract` heading (build.ps1/build.sh match it by exact regex to
    extract the abstract — adding an id there would break the build).
  * Statement *declaration* lines (the `**Theorem N.M ...**` line itself is
    given a hypertarget but is never wrapped in a self-link; a negative
    look-behind for `*` skips the bold declaration).

Idempotent: skips headings that already carry `{#`, and a single re.sub pass
never re-wraps the links it just created. Safe to re-run after a renumber.

USAGE
  python build/utils/add_internal_links.py <paper.md>
The file is edited IN PLACE after writing a `<paper.md>.bak` backup. A summary
(targets found, links created, and any reference whose target is missing) is
printed to stdout. Always rebuild and confirm the LaTeX log reports
0 "undefined references" and 0 "multiply defined" labels.
"""
import re
import shutil
import sys

TYPES = ["Theorem", "Lemma", "Proposition", "Corollary",
         "Conjecture", "Definition", "Remark", "Example"]
TYPE_ALT = (r"(?:Theorems?|Lemmas?|Propositions?|Corollar(?:y|ies)"
            r"|Conjectures?|Definitions?|Remarks?|Examples?)")
NUM = r"(?:[A-D]\.\d+|\d+(?:\.\d+){1,2})(?:\$'\$)?"


def normid(num):
    s = num
    prime = "$'$" in s
    s = s.replace("$'$", "").strip().lower().replace(".", "-")
    return s + ("-prime" if prime else "")


def heading_id(title):
    t = title.strip()
    m = re.match(r"Appendix ([A-D])\.", t)
    if m:
        return "sec:app-" + m.group(1).lower()
    m = re.match(r"([A-D])\.(\d+)\b", t)
    if m:
        return "sec:app-%s-%s" % (m.group(1).lower(), m.group(2))
    m = re.match(r"(\d+)\.(\d+)\b", t)
    if m:
        return "sec:%s-%s" % (m.group(1), m.group(2))
    m = re.match(r"(\d+)\.(?:\s|$)", t)
    if m:
        return "sec:%s" % m.group(1)
    return "sec:" + re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")


def main(path):
    shutil.copy(path, path + ".bak")
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"(<!--.*?-->)", text, flags=re.DOTALL)

    decl_re = re.compile(r"^\*\*(?:" + "|".join(TYPES) + r")\s+(" + NUM + r")", re.M)
    targets = set()
    for i, seg in enumerate(parts):
        if i % 2 == 0:
            for m in decl_re.finditer(seg):
                targets.add(normid(m.group(1)))

    stats = {"headings": 0, "hypertargets": 0, "sec_refs": 0, "stmt_refs": 0}
    missing = []

    def process_lines(seg):
        out = []
        for line in seg.split("\n"):
            hm = re.match(r"^(#{2,3})\s+(.*\S)\s*$", line)
            if hm and not line.startswith("## Abstract") and "{#" not in line:
                out.append("%s %s {#%s}" % (hm.group(1), hm.group(2),
                                            heading_id(hm.group(2))))
                stats["headings"] += 1
                continue
            dm = re.match(r"^\*\*(?:" + "|".join(TYPES) + r")\s+(" + NUM + r")", line)
            if dm:
                out.append("\\hypertarget{stmt:%s}{}" % normid(dm.group(1)))
                stats["hypertargets"] += 1
            out.append(line)
        return "\n".join(out)

    sec_re = re.compile(
        r"(§§?)(\d+(?:\.\d+)?)"
        r"|\b(Sections?)[ ~](\d+(?:\.\d+)?)"
        r"|\b(Appendices|Appendix)[ ~]([A-D])(\s*[–-]\s*[A-D])?")

    def sec_repl(m):
        if m.group(2):
            stats["sec_refs"] += 1
            return "\\hyperref[sec:%s]{%s%s}" % (m.group(2).replace(".", "-"),
                                                 m.group(1), m.group(2))
        if m.group(4):
            stats["sec_refs"] += 1
            return "\\hyperref[sec:%s]{%s~%s}" % (m.group(4).replace(".", "-"),
                                                  m.group(3), m.group(4))
        if m.group(6):
            stats["sec_refs"] += 1
            rng = m.group(7) or ""
            return "\\hyperref[sec:app-%s]{%s~%s%s}" % (m.group(6).lower(),
                                                        m.group(5), m.group(6), rng)
        return m.group(0)

    s1_re = re.compile(r"(?<!\*)\b(" + TYPE_ALT + r")([ ~])(" + NUM + r")")

    def s1_repl(m):
        nid = normid(m.group(3))
        if nid in targets:
            stats["stmt_refs"] += 1
            return "\\hyperlink{stmt:%s}{%s~%s}" % (nid, m.group(1), m.group(3))
        missing.append("%s %s" % (m.group(1), m.group(3)))
        return m.group(0)

    s2_re = re.compile(
        r"(\\hyperlink\{stmt:[^}]+\}\{[^}]+\})"
        r"(\s*(?:,? (?:and|&|/|to) |[–-]|, )\s*)(" + NUM + r")")

    def s2_repl(m):
        nid = normid(m.group(3))
        if nid in targets:
            stats["stmt_refs"] += 1
            return "%s%s\\hyperlink{stmt:%s}{%s}" % (m.group(1), m.group(2),
                                                     nid, m.group(3))
        return m.group(0)

    for i, seg in enumerate(parts):
        if i % 2 == 0:
            seg = process_lines(seg)
            seg = sec_re.sub(sec_repl, seg)
            seg = s1_re.sub(s1_repl, seg)
            for _ in range(6):
                new = s2_re.sub(s2_repl, seg)
                if new == seg:
                    break
                seg = new
            parts[i] = seg

    open(path, "w", encoding="utf-8").write("".join(parts))
    print("statement targets (%d): %s" % (len(targets), " ".join(sorted(targets))))
    print("stats:", stats)
    print("UNLINKED stmt refs (no target):", sorted(set(missing)) or "none")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python add_internal_links.py <paper.md>")
    main(sys.argv[1])
