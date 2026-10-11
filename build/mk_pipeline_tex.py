# -*- coding: utf-8 -*-
"""Sinh tang-a-pipeline.tex tu .md: chuyen fence -> dia, ** -> textbf, backtick -> texttt.
Giu ascii trong dia (font mono khong co ky tu ke khung)."""
import io, re, sys

STEM  = sys.argv[1] if len(sys.argv) > 1 else "tang-a-pipeline"
TITLE = sys.argv[2] if len(sys.argv) > 2 else "Tầng A --- pipeline chi tiết"
SUB   = sys.argv[3] if len(sys.argv) > 3 else "công thức và ví dụ chạy tay"

SRC = "D:\\ragproj\\%s.md" % STEM
DST = "D:\\ragproj\\%s.tex" % STEM
md = io.open(SRC, encoding="utf-8").read()

PRE = r"""% =====================================================================
%  Tang A -- pipeline chi tiet, cong thuc va vi du chay tay
%  bien dich:  tectonic tang-a-pipeline.tex
%  (sinh tu tang-a-pipeline.md boi build/mk_pipeline_tex.py -- dung sua tay)
% =====================================================================
\documentclass[11pt,a4paper]{article}

\usepackage{fontspec}
\setmainfont{texgyrepagella}[
  Extension      = .otf,
  UprightFont    = *-regular,
  BoldFont       = *-bold,
  ItalicFont     = *-italic,
  BoldItalicFont = *-bolditalic ]
\setmonofont{lmmono10-regular}[Extension=.otf, Scale=0.86]

\renewcommand{\contentsname}{Mục lục}
\usepackage[a4paper,margin=2.2cm]{geometry}
\usepackage{fancyvrb}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=black!70, urlcolor=blue!60!black,
            pdftitle={__STEM__}}
\usepackage{titlesec}
\titleformat{\section}{\large\bfseries}{\thesection.}{0.6em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{0.6em}{}
\setlength{\parskip}{0.45em}
\setlength{\parindent}{0pt}

\DefineVerbatimEnvironment{dia}{Verbatim}%
  {fontsize=\small, frame=single, framesep=4pt, rulecolor=\color{black!25}}

\title{\bfseries __TITLE__\\[3pt]
       \large\mdseries __SUB__}
\author{}
\date{}

\begin{document}
\maketitle
\tableofcontents
\vspace{1em}

"""

# ── tach khoi fence va khoi van ban ──────────────────────────────
parts = re.split(r"\n```\n", "\n" + md + "\n")
out = []
# parts: text, code, text, code, ...
ASCII = {"─": "-", "│": "|", "┌": "+", "┐": "+", "└": "+", "┘": "+",
         "├": "+", "┤": "+", "┬": "+", "┴": "+", "┼": "+",
         "→": "->", "←": "<-", "⟺": "<=>", "⟹": "=>", "↓": "v",
         "≥": ">=", "≤": "<=", "≠": "!=", "⊆": "c=", "⊂": "c",
         "∈": "thuộc", "∉": "không thuộc", "∅": "rỗng", "∖": "bỏ",
         "∃": "có", "∪": "+", "∩": "^", "σ": "sigma", "θ": "theta",
         "τ": "tau", "β": "beta", "π": "pi", "λ": "lambda", "Σ": "tổng",
         "≈": "~", "·": "*", "×": "x", "⌊": "floor(", "⌋": ")",
         # KHONG map "â" -> "a^": â co trong CHU TIENG VIET thuong
         # (câu · cân · lâu · tâng), map se pha chu.  Font mono phu duoc no.
         "«": '"', "»": '"', "ĝ": "g^", "ṽ": "v~", "ŝ": "s^",
         "⁺": "+", "⁻": "-", "√": "sqrt", "φ": "phi", "Π": "Pi",
         "₀": "_0", "₁": "_1", "₂": "_2", "ᵦ": "_beta", "₃": "_3",
         "ₜ": "_t", "ⱼ": "_j", "ₖ": "_k", "ₛ": "_s", "ₙ": "_n"}


def to_ascii(t):
    for a, b in ASCII.items():
        t = t.replace(a, b)
    return t


def inline(t):
    """markdown inline -> latex, cho khoi van ban."""
    t = t.replace("\\", r"\textbackslash{}")
    t = t.replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
    t = t.replace("#", r"\#").replace("_", r"\_")
    t = t.replace("{", r"\{").replace("}", r"\}")
    t = t.replace("^", r"\textasciicircum{}").replace("~", r"\textasciitilde{}")
    # link [text](url)
    t = re.sub(r"\\\[([^\]]+)\\\]\(([^)]+)\)", r"\\href{\2}{\1}", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\\href{\2}{\1}", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t, flags=re.S)
    t = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"\\emph{\1}", t)
    t = re.sub(r"`([^`]+)`", lambda m: r"\texttt{%s}" % m.group(1), t)
    t = t.replace("§", r"\S{}")
    return t


def block(t):
    """khoi van ban: heading, bullet, hr, doan."""
    res = []
    for raw in t.split("\n"):
        ln = raw.rstrip()
        if not ln.strip():
            res.append(""); continue
        if ln.strip() == "---":
            res.append(r"\vspace{0.3em}\hrule\vspace{0.6em}"); continue
        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            lvl, txt = len(m.group(1)), inline(m.group(2))
            txt = re.sub(r"^\d+(b)?\.?\s*", "", txt)
            txt = re.sub(r"^\\?\d+\\?\.\\?\d+\s*", "", txt)
            if lvl == 1:   continue           # tieu de da o \maketitle
            elif lvl == 2: res.append(r"\section{%s}" % txt)
            elif lvl == 3: res.append(r"\subsection{%s}" % txt)
            else:          res.append(r"\paragraph{%s}" % txt)
            continue
        m = re.match(r"^(\d+)\.\s+(.*)$", ln)
        if m:
            res.append(r"\textbf{%s.} %s\\" % (m.group(1), inline(m.group(2)))); continue
        if ln.startswith("- ") or ln.startswith("* "):
            res.append(r"$\bullet$\, %s\\" % inline(ln[2:])); continue
        if ln.startswith("|"):
            res.append(inline(ln.strip("|").replace("|", " \\quad ")) + r"\\"); continue
        res.append(inline(ln))
    return "\n".join(res)


for i, p in enumerate(parts):
    if i % 2 == 0:
        out.append(block(p))
    else:
        body = to_ascii(p.strip("\n"))
        out.append("\\begin{dia}\n%s\n\\end{dia}" % body)

PRE2 = (PRE.replace("__TITLE__", TITLE)
           .replace("__SUB__", SUB)
           .replace("__STEM__", STEM))
tex = PRE2 + "\n".join(out) + "\n\n\\end{document}\n"
tex = re.sub(r"\n{4,}", "\n\n\n", tex)
io.open(DST, "w", encoding="utf-8").write(tex)
print("  -> %s  (%d ky tu)" % (DST, len(tex)))
