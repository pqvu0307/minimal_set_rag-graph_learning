# -*- coding: utf-8 -*-
"""Kiem ngoai le SAFE khong lam HONG bo do thieu dau.

checkdau.py co sys.exit o cuoi nen khong import duoc -> lay lai logic o day.
Neu sua W hoac SAFE trong checkdau.py thi sua ca day.
"""
import re
import sys

W = (r"duong|canh|khong|cua|dich|tang|mot|nhieu|diem|duoc|neu|cum|doan|tap|tim|"
     r"luot|vong|chon|thuoc|rong|deu|gia tri|bang chung|truy van|do thi|nhan|"
     r"toi tieu|don dieu|thang|phan tu|thu tu|cham|dung|hoan vi|mat na|"
     r"bat bien|giu|cong|nuot|chuan hoa|cuoi|khop|cung|mem|quay ve|chua")
HINT = re.compile(r"\b(%s)\b" % W)
SAFE = [r"nội dung", r"dung lượng", r"dung sai",
        r"thang\s*[0-9]", r"thang đo", r"thang điểm",
        r"công thức", r"cộng tính", r"công cụ", r"công suất"]
SAFE_RE = [re.compile(s) for s in SAFE]


def hits(t):
    low = t.lower()
    for r in SAFE_RE:
        low = r.sub(" ", low)
    return len(set(HINT.findall(low)))


# Nguong la >= 2 TU PHAN BIET, nen phai thu tren khoi co co that
# (bo kiem chay tren ca khoi dia, hang chuc dong), khong phai mau 4 chu.
CASES = [
    # (van ban, co phai THIEU DAU that khong, ghi chu)
    ("cache   khoa = hash( model, template, noi dung chunk da sap )\n"
     "        bam NOI DUNG chu khong phai ten — chunk lai corpus la ten va nhau",
     True, "khoi ASCII tran -> PHAI bat"),
    ("cache   khóa = hash( model, template, nội dung chunk đã sắp )\n"
     "        băm NỘI DUNG chứ không phải tên — chunk lại corpus là tên va nhau",
     False, "co dau, 'dung' trong 'noi dung' -> khong bao"),

    ("cham    v(S) = P_judge  doc logprob token dau\n"
     "        KHONG dung thang 1-5 — 5 muc lam hang chuc tap con hoa nhau",
     True, "khoi ASCII tran -> PHAI bat"),
    ("chấm    v(S) = P_judge  đọc logprob token ĐẦU\n"
     "        KHÔNG dùng thang 1–5 — 5 mức làm hàng chục tập con hòa nhau",
     False, "co dau, 'thang' la thang do -> khong bao"),

    ("cong thuc tinh duoc, cong tinh la mot phan cua gia tri",
     True, "khoi ASCII tran -> PHAI bat"),
    ("công thức tính được, cộng tính là một phần của giá trị",
     False, "co dau -> khong bao"),

    ("duong canh khong cua dich", True, "nhieu tu ASCII -> PHAI bat"),
    ("đường cạnh không của đích", False, "day du dau -> khong bao"),

    # ngoai le KHONG duoc che loi that nam canh no
    ("nội dung nhung canh thi duong", True,
     "ngoai le che 'noi dung' nhung 'canh/duong' van phai bat"),

    # nguong >= 2: mot tu hint khong du de bao -> CO Y, tranh duong tinh gia
    ("noi dung", False, "1 tu hint duoi nguong 2 -> KHONG bao, co y"),
]

fails = 0
for t, should_flag, why in CASES:
    h = hits(t)
    flagged = h >= 2
    ok = flagged == should_flag
    print("  %s hits=%d flag=%-5s muon=%-5s  %-30s %s"
          % ("ok  " if ok else "FAIL", h, flagged, should_flag, repr(t)[:32], why))
    fails += not ok

print("=" * 70)
if fails:
    print("  %d FAIL" % fails); sys.exit(1)
print("  ngoai le SAFE khong lam hong detector")
