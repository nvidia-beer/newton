#!/usr/bin/env python3
"""
Build full documentation PDF from Markdown files.

Converts the main documentation MD files (paper-centric narrative + appendices)
to LaTeX and compiles a single PDF for the authors and future extension work.
Requires: pandoc (for md->latex), pdflatex (or latexmk).
Output: documentation/build/Newton_Inchworm_Implementation.pdf
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

DOC_DIR = Path(__file__).resolve().parent.parent


def check_pdflatex() -> bool:
    """Return True if pdflatex is available."""
    return shutil.which("pdflatex") is not None


def print_install_help(build_dir: Path) -> None:
    print("pdflatex not found. Install a TeX distribution, then re-run this script.", file=sys.stderr)
    print("", file=sys.stderr)
    print("  Ubuntu/Debian:  sudo apt install texlive-latex-base texlive-latex-extra", file=sys.stderr)
    print("  Fedora:         sudo dnf install texlive-scheme-basic texlive-latex", file=sys.stderr)
    print("  macOS:          brew install --cask mactex-no-gui   (or full MacTeX)", file=sys.stderr)
    print("", file=sys.stderr)
    print(f"LaTeX sources written to: {build_dir.resolve()}", file=sys.stderr)
    print(f"Compile later:  cd {build_dir} && pdflatex main.tex", file=sys.stderr)
BUILD_DIR = DOC_DIR / "build"
MD_FILES = [
    "01_introduction.md",
    "02_paper_model.md",
    "03_dynamic_extension.md",
    "04_solver_layers.md",
    "06_example_inchworm.md",
    "A_equations_reference.md",
    "B_code_implementation.md",
]
OUTPUT_PDF = BUILD_DIR / "Newton_Inchworm_Implementation.pdf"


def md_to_latex_simple(md_path: Path) -> str:
    """Convert Markdown to LaTeX without pandoc: basic sections and code blocks."""
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    out = []
    in_code = False
    code_lang = ""
    for line in lines:
        if line.strip().startswith("```"):
            if in_code:
                out.append("\\end{lstlisting}")
                in_code = False
            else:
                code_lang = line.strip()[3:].strip() or "text"
                out.append("\\begin{lstlisting}[language=" + code_lang + "]")
                in_code = True
            continue
        if in_code:
            out.append(line.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}"))
            continue
        # Headers
        if line.startswith("# "):
            out.append("\\section{" + line[2:].strip() + "}")
        elif line.startswith("## "):
            out.append("\\subsection{" + line[3:].strip() + "}")
        elif line.startswith("### "):
            out.append("\\subsubsection{" + line[4:].strip() + "}")
        else:
            # Inline code (do not replace backslash in \( \) or \[ \])
            line = re.sub(r"`([^`]+)`", r"\\texttt{\1}", line)
            out.append(line)
    if in_code:
        out.append("\\end{lstlisting}")
    return "\n".join(out)


# Unicode -> ASCII/LaTeX for use inside lstlisting (verbatim) and to avoid encoding issues
LSTLISTING_UNICODE_REPLACES = [
    ("→", "->"),
    ("↓", "|"),
    ("–", "-"),
    ("−", "-"),
    ("Δ", "Delta "),  # space so "Δd" -> "Delta d"
    ("±", "+/-"),
    ("φ", "phi"),
    ("₁", "1"),
    ("₂", "2"),
]


def markdown_bold_to_tex(content: str) -> str:
    """Convert **bold** to \\textbf{bold} (non-greedy, per pair)."""
    return re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", content)


def markdown_images_to_tex(content: str) -> str:
    """Convert markdown image ![...](path) to LaTeX \\includegraphics (pandoc sometimes leaves these raw)."""
    # Match ![alt text](path) - path is group 1; alt can contain \), so match ]( then path
    return re.sub(
        r"!\[.*?\]\(([^)]+)\)",
        r"\\begin{center}\n\\includegraphics[width=0.5\\textwidth]{\1}\n\\end{center}",
        content,
    )


def tabular_to_tabularx(content: str) -> str:
    """Convert \\begin{tabular}{l...} to tabularx with \\textwidth so tables fit in page."""
    def repl(m: re.Match) -> str:
        n = len(m.group(1))
        col_spec = "l" + "X" * (n - 1) if n > 1 else "l"
        return r"\begin{tabularx}{\textwidth}{" + col_spec + "}"
    content = re.sub(r"\\begin\{tabular\}\{(l+)\}", repl, content)
    content = content.replace(r"\end{tabular}", r"\end{tabularx}")
    # Slightly smaller font so table fits and stays readable
    content = re.sub(
        r"(\\begin\{center\}\s*)\n(\s*\\begin\{tabularx\})",
        r"\1\n\\small\n\2",
        content,
    )
    return content


def markdown_tables_to_tex(content: str) -> str:
    """Convert markdown pipe tables to LaTeX tabular (split on ' | ' to preserve \\texttt)."""
    lines = content.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not re.match(r"^\|.+\|$", line.strip()):
            out.append(line)
            i += 1
            continue
        rows: list[list[str]] = []
        j = i
        while j < len(lines) and re.match(r"^\|.+\|$", lines[j].strip()):
            raw = lines[j].strip()
            # Split by " | " (space-pipe-space) to avoid breaking \texttt{...}
            parts = [p.strip() for p in raw.strip("|").split(" | ")]
            # Skip separator row (all dashes/pipes) or single-cell separator
            if parts and (
                all(re.match(r"^[\-\s:]+$", p) for p in parts)
                or (len(parts) == 1 and re.match(r"^[\|\-\s:]+$", parts[0]))
            ):
                j += 1
                continue
            if parts:
                rows.append(parts)
            j += 1
        if len(rows) < 2:
            out.append(line)
            i += 1
            continue
        ncol = max(len(r) for r in rows)
        # Use tabularx so table fits within \textwidth; first column 'l', rest 'X' (wrapping)
        col_spec = "l" + "X" * (ncol - 1) if ncol > 1 else "l"
        out.append(r"\begin{center}")
        out.append(r"\small")
        out.append(r"\begin{tabularx}{\textwidth}{" + col_spec + r"}")
        for idx, r in enumerate(rows):
            row = (r + [""] * ncol)[:ncol]
            out.append(" & ".join(row) + r" \\")
            if idx == 0:
                out.append(r"\hline")
        out.append(r"\end{tabularx}")
        out.append(r"\end{center}")
        i = j
    return "\n".join(out)


def replace_lstlisting_unicode(content: str) -> str:
    """Replace Unicode inside lstlisting blocks so pdflatex does not choke."""
    import re as re_inner
    pattern = re_inner.compile(r"\\begin\{lstlisting\}.*?\\end\{lstlisting\}", re_inner.DOTALL)
    def repl(m: re_inner.Match) -> str:
        block = m.group(0)
        for u, a in LSTLISTING_UNICODE_REPLACES:
            block = block.replace(u, a)
        return block
    return pattern.sub(repl, content)


def escape_tex_underscores(content: str) -> str:
    """Escape _ inside \\texttt{...} and in section headings so LaTeX does not treat them as subscripts."""
    def repl_texttt(m: re.Match) -> str:
        return r"\texttt{" + m.group(1).replace("_", r"\_") + "}"
    content = re.sub(r"\\texttt\{([^}]*)\}", repl_texttt, content)
    # Escape _ in \section, \subsection, \subsubsection titles
    def repl_heading(m: re.Match) -> str:
        return m.group(1) + m.group(2).replace("_", r"\_") + m.group(3)
    content = re.sub(
        r"(\\(?:section|subsection|subsubsection)\{)([^}]*)(\})",
        repl_heading,
        content,
    )
    return content


def run_pandoc(md_path: Path, tex_path: Path) -> bool:
    """Convert MD to LaTeX using pandoc. Returns True on success."""
    try:
        subprocess.run(
            [
                "pandoc",
                str(md_path),
                "-f", "markdown",
                "-t", "latex",
                "--wrap=preserve",
                "-o", str(tex_path),
            ],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def main() -> int:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    use_pandoc = False
    try:
        subprocess.run(["pandoc", "--version"], capture_output=True, check=True)
        use_pandoc = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    tex_parts = []
    for name in MD_FILES:
        md_path = DOC_DIR / name
        if not md_path.exists():
            print(f"Warning: {md_path} not found", file=sys.stderr)
            continue
        tex_path = BUILD_DIR / (name.replace(".md", "") + ".tex")
        if use_pandoc:
            if run_pandoc(md_path, tex_path):
                tex_parts.append(tex_path)
            else:
                content = md_to_latex_simple(md_path)
                tex_path.write_text(content, encoding="utf-8")
                tex_parts.append(tex_path)
        else:
            content = md_to_latex_simple(md_path)
            tex_path.write_text(content, encoding="utf-8")
            tex_parts.append(tex_path)

    main_tex = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage[hidelinks]{hyperref}
\usepackage{listings}
\usepackage{parskip}
\usepackage[margin=2.5cm]{geometry}
\usepackage{graphicx}
\graphicspath{{../}}
\usepackage{tabularx}
\usepackage{newunicodechar}
\newunicodechar{Δ}{\ensuremath{\Delta}}
\newunicodechar{φ}{\ensuremath{\phi}}
\newunicodechar{₁}{\ensuremath{_1}}
\newunicodechar{₂}{\ensuremath{_2}}
\newunicodechar{±}{\ensuremath{\pm}}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{⇒}{\ensuremath{\Rightarrow}}
\newunicodechar{↓}{\ensuremath{\downarrow}}
\newunicodechar{–}{-}
\newunicodechar{−}{\ensuremath{-}}
\newunicodechar{ω}{\ensuremath{\omega}}
\newunicodechar{π}{\ensuremath{\pi}}
\newunicodechar{≈}{\ensuremath{\approx}}
\title{Soft Inchworm Crawling: From Paper to Simulation}
\author{Based on Understanding Inchworm Crawling for Soft-Robotics.\\[0.5em]\normalsize Gamus et al., arXiv:1911.05227}
\date{}
\begin{document}
\maketitle
\tableofcontents
\newpage
"""
    for tex_path in tex_parts:
        stem = tex_path.stem
        main_tex += f"\\input{{{tex_path.name}}}\n"

    main_tex += "\\end{document}\n"
    main_path = BUILD_DIR / "main.tex"
    main_path.write_text(main_tex, encoding="utf-8")

    # Fix raw markdown in pandoc output: bold, tables; escape underscores; lstlisting
    for tex_path in tex_parts:
        content = tex_path.read_text(encoding="utf-8")
        content = replace_lstlisting_unicode(content)
        content = markdown_bold_to_tex(content)
        content = markdown_images_to_tex(content)
        content = markdown_tables_to_tex(content)
        content = tabular_to_tabularx(content)
        content = escape_tex_underscores(content)
        content = content.replace(r"\begin{lstlisting}[language=text]", r"\begin{lstlisting}")
        tex_path.write_text(content, encoding="utf-8")

    if not check_pdflatex():
        print_install_help(BUILD_DIR)
        return 1

    # Compile with pdflatex (run twice for TOC/refs; exit code 1 is often just "rerun" warnings)
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(BUILD_DIR), str(main_path)],
            cwd=str(BUILD_DIR),
            capture_output=True,  # do not decode as text (log may contain non-UTF-8)
        )
    out = BUILD_DIR / "main.pdf"
    if out.exists():
        out.rename(OUTPUT_PDF)
        print(f"Built: {OUTPUT_PDF}")
        # Clean build folder: keep only the PDF
        for f in BUILD_DIR.iterdir():
            if f.is_file() and f != OUTPUT_PDF:
                try:
                    f.unlink()
                except OSError as e:
                    print(f"Warning: could not remove {f}: {e}", file=sys.stderr)
        return 0
    print("pdflatex did not produce main.pdf (check build/main.log for errors)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
