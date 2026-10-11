# -*- coding: utf-8 -*-
"""Do bo cuc slides.pdf: co chu, chu tran hop, cot lech, va cell lech dong.

Vi sao phai do tren PDF: beamer KHONG bao loi khi noi dung chay qua le hay
tran ra ngoai tcolorbox — no chi ghi "Overfull vbox" vao log, rat de bo qua.
Va loi "lech chu" trong bang thi khong co ca mot dong log nao.
"""
import sys
from collections import Counter, defaultdict

try:
    import fitz
except ImportError:
    print("thieu pymupdf"); sys.exit(0)

PDF = "D:/ragproj/slides.pdf"
MIN_SIZE = 6.8          # duoi nguong nay coi la khong doc noi khi chieu
CELL_TOL = 1.5          # lech baseline > nguong nay giua hai o cung hang = loi


def spans(page):
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                if s["text"].strip():
                    yield s


def boxes(page):
    """Khung tcolorbox: hinh chu nhat to, khong phai ca trang."""
    return [d["rect"] for d in page.get_drawings()
            if d["rect"].width > 55 and d["rect"].height > 16
            and d["rect"].width < page.rect.width - 20]


def main():
    doc = fitz.open(PDF)
    W, H = doc[0].rect.width, doc[0].rect.height
    nbad = 0

    # ── 1. co chu ────────────────────────────────────────────────────
    census, small = Counter(), defaultdict(set)
    for i, p in enumerate(doc, 1):
        for b in p.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                ss = [s for s in l["spans"] if s["text"].strip()]
                big = max((s["size"] for s in ss), default=0)
                for s in ss:
                    sz = round(s["size"], 1)
                    census[sz] += len(s["text"])
                    if sz >= MIN_SIZE or len(s["text"].strip()) <= 2:
                        continue
                    # chi so duoi/tren: cung dong co span lon hon han ->
                    # nho la DUNG kieu chu, khong phai van ban than bi thu nho
                    if big > s["size"] * 1.18:
                        continue
                    small[sz].add(i)
    print("=" * 70)
    print("1. co chu")
    for sz, n in sorted(census.items()):
        print("     %5.1fpt  %6d ky tu" % (sz, n))
    if small:
        nbad += 1
        print("   ! duoi %.1fpt:" % MIN_SIZE)
        for sz in sorted(small):
            print("     %.1fpt  slide %s" % (sz, sorted(small[sz])))
    else:
        print("   khong co chu nao duoi %.1fpt (tru chi so cong thuc)" % MIN_SIZE)

    # ── 2. tran khoi trang ───────────────────────────────────────────
    print("-" * 70)
    print("2. tran khoi vung than trang")
    #  Khong du khi chi so voi H: noi dung dung truoc me trang van co the
    #  de len dai footer va trong nhu bi cat.  Than trang ket thuc o
    #  FOOT_TOP, do chinh la cai mat nhin thay.
    #  LAY MODE, khong lay min: footer xuat hien o MOI trang nen baseline cua
    #  no la gia tri lap nhieu nhat.  Dung min thi chinh phan noi dung dang
    #  tran xuong se keo moc di, va nguong tu that chat theo loi — cang hong
    #  cang bao nang, ca o nhung slide khong he doi.
    band = Counter(round(s["origin"][1], 1) for p in doc for s in spans(p)
                   if s["origin"][1] > H - 20)
    fb = band.most_common(1)[0][0] if band else H
    foot = fb - 8.0
    hit = []
    for i, p in enumerate(doc, 1):
        # Loai footer theo BASELINE cua no, khong theo vung y: noi dung bi tran
        # cung nam trong dai do, loai ca dai thi chinh cai can bat lai lot.
        body = [s for s in spans(p) if abs(s["origin"][1] - fb) > 1.0]
        if not body:
            continue
        lo = max(s["bbox"][3] for s in body)
        ri = max(s["bbox"][2] for s in body)
        if lo > foot:
            hit.append((i, "doc  +%.1fpt (de len footer)" % (lo - foot)))
        if ri > W + 0.5:
            hit.append((i, "ngang +%.1fpt" % (ri - W)))
    for i, how in hit:
        print("   ! slide %-3d %s" % (i, how))
    if hit:
        nbad += len(hit)
    else:
        print("   %d slide, khong cho nao tran trang" % doc.page_count)

    # ── 3. chu tran ra ngoai hop ─────────────────────────────────────
    print("-" * 70)
    print("3. chu tran ra ngoai tcolorbox")
    hit = []
    for i, p in enumerate(doc, 1):
        rs = boxes(p)
        if not rs:
            continue
        for s in spans(p):
            r = fitz.Rect(s["bbox"])
            # hop nho nhat ma o DAU dong nam trong
            # o DAU dong phai nam THUC SU ben trong hop, khong chi "ben phai
            # canh trai cua no" — thieu dieu kien x1 thi mot span o cot trai
            # bi ghep voi hop o cot phai va bao tran 200pt.
            cand = [R for R in rs
                    if R.x0 - 1 <= r.x0 < R.x1 - 2
                    and R.y0 - 2 <= r.y0 and R.y1 + 2 >= r.y1]
            if not cand:
                continue
            box = min(cand, key=lambda R: R.get_area())
            over = r.x1 - (box.x1 - 3.0)
            if over > 0.5:
                hit.append((i, over, s["text"][:40]))
    seen = set()
    for i, over, t in sorted(hit, key=lambda x: -x[1]):
        if i in seen:
            continue
        seen.add(i)
        print("   ! slide %-3d tran %5.1fpt   %r" % (i, over, t))
    if hit:
        nbad += len(seen)
    else:
        print("   khong khoi nao tran ra ngoai hop")

    # ── 4. cell lech dong trong bang ────────────────────────────────
    #  Gom span theo baseline; hai cum x roi nhau ma baseline lech
    #  trong khoang (TOL, mot dong) la dau hieu o p{} bi tut.
    print("-" * 70)
    print("4. o bang lech dong (loi \\color dau \\vtop)")
    #  Dau hieu RIENG cua loi nay: cung mot do lech dy lap lai o NHIEU hang
    #  cua cung mot bang.  Hai cot doc lap canh nhau cung lech, nhung lech
    #  moi hang mot khac — nen doi hoi "dy giong nhau >= 3 lan" moi loai
    #  duoc bao dong gia, con loi that thi no lap lai dung tung hang.
    hit = []
    for i, p in enumerate(doc, 1):
        rows = defaultdict(list)
        for s in spans(p):
            rows[round(s["origin"][1], 1)].append(s)
        ys = sorted(rows)
        pat = Counter()
        ex = {}
        for a in range(len(ys)):
            for b in range(a + 1, len(ys)):
                dy = round(ys[b] - ys[a], 1)
                if dy <= CELL_TOL or dy > 8.0:
                    continue
                ra, rb = rows[ys[a]], rows[ys[b]]
                if abs(ra[0]["size"] - rb[0]["size"]) > 0.3:
                    continue
                ax0 = min(s["bbox"][0] for s in ra)
                ax1 = max(s["bbox"][2] for s in ra)
                bx0 = min(s["bbox"][0] for s in rb)
                bx1 = max(s["bbox"][2] for s in rb)
                gap = bx0 - ax1 if bx0 > ax1 else ax0 - bx1
                # hai o cua CUNG mot hang bang: ke nhau, cach nhau it
                if not (0 < gap < 32):
                    continue
                pat[dy] += 1
                ex.setdefault(dy, (ra[0]["text"][:22], rb[0]["text"][:22]))
        for dy, n in pat.items():
            if n >= 3:
                hit.append((i, dy, n, ex[dy]))
    for i, dy, n, (t1, t2) in hit:
        print("   ! slide %-3d lech %.1fpt lap %d hang   %r | %r"
              % (i, dy, n, t1, t2))
    if hit:
        nbad += len(hit)
    else:
        print("   khong bang nao co lech lap lai")

    # ── 5. hai cot khong cung mot moc tren ──────────────────────────
    print("-" * 70)
    print("5. hai cot khong bat dau cung do cao")
    hit = []
    for i, p in enumerate(doc, 1):
        # bo qua frame [plain] (khong co dai tieu de): no khong dung columns,
        # nen phep do "nua trai / nua phai" khong co nghia o do
        if not any(s["origin"][1] < 25 for s in spans(p)):
            continue
        body = [s for s in spans(p) if s["origin"][1] > 30]
        if not body:
            continue
        mid = W / 2
        L = [s["bbox"][1] for s in body if s["bbox"][2] < mid]
        R = [s["bbox"][1] for s in body if s["bbox"][0] > mid]
        if not L or not R:
            continue
        d = abs(min(L) - min(R))
        if d > 22:
            hit.append((i, d, min(L), min(R)))
    for i, d, l, r in hit:
        print("   ! slide %-3d lech %5.1fpt  (trai y=%.0f, phai y=%.0f)" % (i, d, l, r))
    if hit:
        nbad += len(hit)
    else:
        print("   moi slide hai cot cung moc")

    print("=" * 70)
    print("TONG: %d nhom van de" % nbad)
    return 1 if nbad else 0


if __name__ == "__main__":
    sys.exit(main())
