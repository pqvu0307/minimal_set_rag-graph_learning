# -*- coding: utf-8 -*-
"""Ghep build/sl_*.tex -> slides.tex, va tu them [fragile] cho frame co verbatim.

Beamer khong xu ly duoc verbatim trong frame thuong (no doc truoc noi dung
frame), nen MOI frame chua \\begin{dia} phai la \\begin{frame}[fragile].
Lam tu dong o day de khong phai theo doi tay moi lan them slide.
"""
import io
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "slides.tex"

parts = sorted(HERE.glob("sl_*.tex"))
if not parts:
    print("khong thay build/sl_*.tex"); sys.exit(1)

tex = "".join(p.read_text(encoding="utf-8") for p in parts)
if "\\end{document}" not in tex:
    tex = tex.rstrip() + "\n\n\\end{document}\n"

# ── them [fragile] cho frame co verbatim ────────────────────────────
FRAME = re.compile(r"\\begin\{frame\}(\[[^\]]*\])?(\{.*?)?(?=\n)", re.S)


def fix(tex: str) -> tuple:
    out = []
    i = 0
    n_fix = 0
    for m in re.finditer(r"\\begin\{frame\}", tex):
        pass
    # tach theo frame roi xu ly tung cai
    chunks = re.split(r"(\\begin\{frame\})", tex)
    res = [chunks[0]]
    k = 1
    while k < len(chunks) - 1:
        head, body = chunks[k], chunks[k + 1]
        end = body.find("\\end{frame}")
        inner = body[:end] if end >= 0 else body
        has_vb = "\\begin{dia}" in inner or "\\begin{Verbatim}" in inner
        already = inner.lstrip().startswith("[")
        if has_vb and not already:
            head = "\\begin{frame}[fragile]"
            n_fix += 1
        elif has_vb and already and "fragile" not in inner.split("]")[0]:
            # da co option khac -> chen fragile vao
            opt_end = inner.find("]")
            body = inner[:opt_end] + ",fragile" + inner[opt_end:] + body[end:]
            n_fix += 1
        res.append(head); res.append(body)
        k += 2
    if k == len(chunks) - 1:
        res.append(chunks[k])
    return "".join(res), n_fix


tex, n_fix = fix(tex)
OUT.write_text(tex, encoding="utf-8")

n_frame = tex.count("\\begin{frame}")
n_dia = tex.count("\\begin{dia}")
print("  ghep %d phan: %s" % (len(parts), ", ".join(p.name for p in parts)))
print("  -> %s   %d ky tu" % (OUT, len(tex)))
print("  %d frame · %d khoi dia · them [fragile] cho %d frame"
      % (n_frame, n_dia, n_fix))


# ── kiem tran: gan vao build de khong tai dien ─────────────────────
def check_overflow(pdf: str = None) -> int:
    """Doc tran doc/ngang tren PDF da bien dich.

    Beamer KHONG bao loi khi noi dung chay qua le trang — no chi in
    "Overfull vbox" vao log, de bo qua.  Nen phai do tren PDF.
    """
    try:
        import fitz
    except ImportError:
        print("  (bo qua kiem tran: thieu pymupdf)")
        return 0
    from pathlib import Path as _P
    f = _P(pdf or (HERE.parent / "slides.pdf"))
    if not f.exists():
        print("  (chua co %s — bien dich truoc)" % f.name)
        return 0
    d = fitz.open(str(f))
    H, W = d[0].rect.height, d[0].rect.width
    bad = []
    for i, p in enumerate(d, 1):
        b = p.get_text("blocks")
        if not b:
            continue
        lo, ri = max(x[3] for x in b), max(x[2] for x in b)
        t = p.get_text().strip().split("\n")[0][:46]
        if lo > H + 0.5:
            bad.append((i, "doc  +%.1fpt" % (lo - H), t))
        if ri > W + 0.5:
            bad.append((i, "ngang +%.1fpt" % (ri - W), t))
    if bad:
        print("  ! %d cho TRAN:" % len(bad))
        for i, how, t in bad:
            print("     slide %-3d %-14s %s" % (i, how, t))
    else:
        print("  %d slide, khong cho nao tran" % d.page_count)
    return len(bad)


if __name__ == "__main__":
    # check_overflow() o day doc PDF cua LAN BIEN DICH TRUOC, nen no bao ve
    # mot ban khac voi slides.tex vua ghi — lech pha, rat de hieu nham.
    # Kiem that nam o build/checklayout.py, chay SAU tectonic:
    #     python build/mk_slides.py
    #     build/tools/tectonic.exe slides.tex
    #     python build/checklayout.py
    print("  -> bien dich roi chay: python build/checklayout.py")
    sys.exit(0)
