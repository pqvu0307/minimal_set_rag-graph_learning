# -*- coding: utf-8 -*-
"""Tim chu tieng Viet thieu dau trong .tex va .pptx.
   KHONG bo qua o da co san mot vai dau -- do chinh la loi cua ban truoc,
   no de lot moi o SUA DO."""
import io, re, sys
from pptx import Presentation

# tu tieng Viet mà LUON phai co dau; thay dang khong dau la chac chan thieu
W = (r"duong|canh|khong|cua|dich|tang|mot|nhieu|diem|duoc|neu|cum|doan|tap|tim|"
     r"luot|vong|chon|thuoc|rong|deu|gia tri|bang chung|truy van|do thi|nhan|"
     r"toi tieu|don dieu|thang|phan tu|thu tu|cham|dung|hoan vi|mat na|"
     r"bat bien|giu|cong|nuot|chuan hoa|cuoi|khop|cung|mem|quay ve|chua")
HINT = re.compile(r"\b(%s)\b" % W)

# Vai tu co CA HAI dang dung trong tieng Viet, nen chung la hint TOI.
# Khong bo khoi W (vi chung VAN bat duoc loi that), ma loai theo NGU CANH:
#     dung    "noi dung" · "dung luong"      dung  ->  "dung/dung/dung" thi SAI
#     thang   "thang 1-5" · "thang do"       thang ->  "thang/thang" thi SAI
#     cong    "cong thuc" · "cong tinh"      cong  ->  "cong" thi SAI
# Khong co ngoai le nay thi bo kiem bao dong gia va mat tin cay — dung cai
# loi nguoc voi lan truoc (bo qua o da co dau nen bao "0 thieu" sai).
SAFE = [
    r"nội dung", r"dung lượng", r"dung sai",
    r"thang\s*[0-9]", r"thang đo", r"thang điểm",
    r"công thức", r"cộng tính", r"công cụ", r"công suất",
]
SAFE_RE = [re.compile(s) for s in SAFE]


def hits(t):
    low = t.lower()
    for r in SAFE_RE:                 # xoa ngu canh an toan truoc khi do
        low = r.sub(" ", low)
    return len(set(HINT.findall(low)))

print("=" * 68)
bad = []
for path, tag in ((r"D:\ragproj\pipeline-shapley-ig-graph.tex", "chinh"),
                  (r"D:\ragproj\tang-a-pipeline.tex",          "tangA"),
                  (r"D:\ragproj\tang-a-shapley.tex",           "shply"),
                  (r"D:\ragproj\ke-hoach.tex",                 "kehoa")):
    tex = io.open(path, encoding="utf-8").read()
    blocks = re.findall(r"\\begin\{dia\}(.*?)\\end\{dia\}", tex, re.S)
    b2 = [(tag, i, b, hits(b)) for i, b in enumerate(blocks, 1) if hits(b) >= 2]
    print("  %-6s %3d khoi dia,  %d khoi con tu THIEU DAU" % (tag, len(blocks), len(b2)))
    for ch in "\u2500\u2502\u250c\u2510\u2514\u2518\u251c\u2524\u252c\u2534\u253c":
        if ch in tex:
            print("     ! ky tu ke khung %r trong %s" % (ch, tag))
    bad += b2
for tag, i, b, h in bad[:8]:
    print("     [%s] #%-3d (%d tu) %s" % (tag, i, h, b.strip().split("\n")[0][:46]))
if len(bad) > 8: print("     ... con %d khoi" % (len(bad) - 8))

print("-" * 68)
prs = Presentation(r"D:\ragproj\quy-gan-rag-hai-tang.pptx")
rows = []
for n, sl in enumerate(prs.slides, 1):
    for sh in sl.shapes:
        if not sh.has_text_frame: continue
        h = hits(sh.text_frame.text)
        if h >= 2: rows.append((n, h, sh.text_frame.text.strip().split("\n")[0][:46]))
print("  .pptx:  %d o text con tu THIEU DAU, o %d slide"
      % (len(rows), len({r[0] for r in rows})))
for n, h, t in rows[:10]: print("     slide %-3d (%d tu) %s" % (n, h, t))
if len(rows) > 10: print("     ... con %d o" % (len(rows) - 10))
print("=" * 68)
sys.exit(1 if (bad or rows) else 0)
