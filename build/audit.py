# -*- coding: utf-8 -*-
"""Ra soat nhat quan tai lieu: tham chieu muc, ky hieu chua khai bao, md<->tex lech."""
import io, re, sys

MD   = r"D:\ragproj\pipeline-shapley-ig-graph.md"
TEX  = r"D:\ragproj\pipeline-shapley-ig-graph.tex"
AMD  = r"D:\ragproj\cho-de-sai.md"
rd   = lambda p: io.open(p, encoding='utf-8').read()
md, tex, amd = rd(MD), rd(TEX), rd(AMD)
issues = []

# ── 1. tham chieu §x.y tro toi muc khong ton tai ─────────────────
have = set(re.findall(r'^#{2,3}\s+(\d+(?:\.\d+)?)\.?\s', md, re.M))
tops = {h.split('.')[0] for h in have}
app  = set(re.findall(r'^#{2,3}\s+(\d+)\.?\s', amd, re.M))
for m in re.finditer(r'§(\d+(?:\.\d+)?)', md):
    ref, lead = m.group(1), md[max(0, m.start() - 70):m.start()]
    if 'cho-de-sai' in lead or 'phụ lục' in lead.lower():   # tro sang phu luc
        if ref.split('.')[0] not in app:
            issues.append(f"tham chieu phu luc §{ref} khong co trong cho-de-sai")
        continue
    if ref not in have and ref not in tops:
        issues.append(f"tham chieu §{ref} khong co muc tuong ung")

# ── 2. ky hieu dung ma chua khai bao o bang "Ky hieu muc nay" ────
declared = set()
for blk in re.findall(r'\*\*Ký hiệu mục này giới thiệu\.?\*\*\s*\n+```\n(.*?)```', md, re.S):
    for line in blk.splitlines():
        if line.strip():
            declared |= set(re.split(r'[,\s]+', line.strip())[:3])
if not declared:
    issues.append("khong doc duoc bang ky hieu nao -- regex hong")
for sym in ("n_max", "eps", "E_reach", "suff_max", "S_k", "k_top", "rank_GFM",
            "τ_q", "f_θ", "β_i", "PGI_i", "S*"):
    if not any(d == sym or d.startswith(sym + "(") for d in declared):
        issues.append(f"ky hieu {sym} dung ma chua khai bao")

# ── 3. md <-> tex: cac khang dinh then chot phai co o CA HAI ─────
pairs = [
    ("n_max = 5",            r"n_{\max}=5",                    "n_max hang 5"),
    ("suff_max",             r"\suff_{\max}",                  "tien kiem suff_max"),
    ("credit mồ côi",        "credit mồ côi",                  "credit mo coi"),
    ("ngân sách đọc hiểu",   "ngân sách đọc hiểu",             "ngan sach doc hieu"),
    ("đường tắt",            "đường tắt",                      "o duong tat"),
    ("gain biên",            "gain biên",                      "dung theo gain bien"),
    ("entity linking",       "entity linking",                 "nguyen nhan entity linking"),
    # hai pha -- phai co o CA HAI file
    ("3.983.000",            "3.983.000",                      "so du lieu 4.0M"),
        ("tỉa xuống",            "tỉa xuống",                       "tia roi quet"),
    ("hàng cứu",             "hàng cứu",                        "hang cuu"),
    ("pool không chứa tập đủ", "pool không chứa tập đủ",        "co pool thieu"),
    ("Dấu xuất xứ",          "Dấu xuất xứ",                     "dau xuat xu"),
    ("sàn cụm thắng",        "sàn cụm thắng",                   "san cum thang"),
    ("C  ≤  ⌊ m / 2 ⌋",      r"C\;\le\;\lfloor m/2\rfloor",     "tran C"),
    ("CỔNG p",               "CỔNG p",                          "cong p"),
    ("2^|S*| − 2",           "2^{|S^*|}-2",                     "quet het tap con"),
    ("Chạy theo lô",         "Chạy theo lô",                    "chay theo lo"),
    ("Rò rỉ ngữ cảnh pool",  "Rò rỉ ngữ cảnh pool",             "ro ri pool"),
    ("mốc GBM",              "mốc GBM",                         "moc GBM"),
    ("ba khuôn",             "ba khuôn",                        "ba khuon che"),
]
for a, b, name in pairs:
    if (a in md) != (b in tex):
        issues.append(f"md/tex lech: {name}  (md={a in md}, tex={b in tex})")

# ── 4. cach noi da bi thay the, khong duoc con sot ───────────────
# "|V_q| + 1" van duoc nhac co y o muc "Hai chan da thu va loai" -> chi bat khi
# no con duoc DUNG lam dinh nghia.
stale = ["n_max = |V_q|", "n_max lần", "gộp hai nguyên nhân", "ba ca còn lại",
         "trần số vòng greedy", "4 ca", "§9.4", "hằng số. KHÔNG phải hàm"]
for s in stale:
    for name, txt in (("md", md), ("tex", tex), ("phu luc", amd)):
        if s in txt:
            issues.append(f"con sot cach noi cu trong {name}: {s!r}")

# ── 5. khoi ma trong .tex chi duoc dung ASCII ke khung ───────────
for ch in "─│┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣":
    if ch in tex:
        issues.append(f"ky tu ke khung {ch!r} trong .tex (font mono khong co)")

print("=" * 62)
if issues:
    for i in issues: print("  !", i)
    print(f"  -> {len(issues)} van de")
else:
    print("  sach: tham chieu, ky hieu, md<->tex, cach noi cu, ky tu")
print("=" * 62)
sys.exit(1 if issues else 0)
