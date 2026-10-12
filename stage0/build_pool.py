# -*- coding: utf-8 -*-
"""Dung POOL THAT bang BM25 tren TOAN corpus, thay vi dung doan kem san.

Vi sao can: moi con so hom nay do tren 10-20 doan KEM SAN cua bo du lieu,
tuc doan vang CHAC CHAN co mat.  Do la pool thien vi, va no lam mo mot cau
hoi quan trong: trong so ca `tran_thap` / `moi_cover_truot`, bao nhieu la
POOL THIEU THAT va bao nhieu la TA DO SAI?

Pool that can truy hoi tren toan corpus.  Dung BM25 + ngau nhien thi lam
duoc MIEN PHI — khong can OpenIE (~$800), khong can do thi, khong can GPU.
Nguon `gfm` phai doi do thi, nen o day chi co hai trong ba nguon; da ghi ro.

Chi so quan trong nhat la RECALL@pool: ti le truy van ma MOI doan vang deu
duoc lay ve.  No chinh la tran cua tang A — khong doan nao lay ve thi khong
co cach nao tim ra no.

    python stage0/build_pool.py --n=1000 --corpus=musique --k_bm25=20 --k_rand=5
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import random
import re
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

RAW = pathlib.Path("artifacts/raw")
_STOP = set(("the a an of in on at to is was were are and or for by with as "
             "from that this it its his her their be been has have had not").split())


def toks(s: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", s.lower())
            if w not in _STOP and len(w) > 1]


class BM25:
    """BM25 thuan numpy — corpus vai chuc nghin doan thi khong can thu vien."""

    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf: List[Dict[str, int]] = []
        self.len: List[int] = []
        df: Dict[str, int] = {}
        for d in docs:
            t = toks(d)
            c = collections.Counter(t)
            self.tf.append(c); self.len.append(len(t))
            for w in c:
                df[w] = df.get(w, 0) + 1
        self.N = len(docs)
        self.avg = sum(self.len) / max(1, self.N)
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
                    for w, n in df.items()}
        # chi muc nguoc: tu -> cac doan chua no.  Khong co no thi moi truy van
        # phai quet ca corpus.
        self.inv: Dict[str, List[int]] = collections.defaultdict(list)
        for i, c in enumerate(self.tf):
            for w in c:
                self.inv[w].append(i)

    def top(self, q: str, k: int) -> List[int]:
        sc: Dict[int, float] = collections.defaultdict(float)
        for w in set(toks(q)):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i in self.inv[w]:
                f = self.tf[i][w]
                sc[i] += idf * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return sorted(sc, key=lambda i: -sc[i])[:k]


def load_musique(n: int):
    f = RAW / "musique" / "dev.jsonl"
    rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()]
    corpus: Dict[str, str] = {}          # khoa = title|text, de khu trung lap
    items = []
    for r in rows:
        if not (r.get("question_decomposition") and r.get("paragraphs")):
            continue
        gold_keys = []
        for p in r["paragraphs"]:
            t = (p.get("paragraph_text") or "").strip()
            if not t:
                continue
            key = ("%s. %s" % (p.get("title", ""), t)).strip(". ")
            corpus[key] = key
            if p.get("idx") in {s.get("paragraph_support_idx")
                                for s in r["question_decomposition"]}:
                gold_keys.append(key)
        if gold_keys:
            items.append({"q": r["question"], "y": str(r["answer"]),
                          "al": r.get("answer_aliases") or [],
                          "gold_keys": gold_keys, "row": r})
        if len(items) >= n:
            break
    return items, sorted(corpus)


def main() -> int:
    arg = {k.lstrip("-"): v for k, v in
           (a.split("=", 1) for a in sys.argv[1:] if "=" in a)}
    n = int(arg.get("n", 1000))
    k_bm = int(arg.get("k_bm25", 20))
    k_rd = int(arg.get("k_rand", 5))
    rng = random.Random(0)

    t0 = time.time()
    items, corpus = load_musique(n)
    print("== %d truy van · corpus %d doan (da khu trung lap) ==" % (len(items), len(corpus)))
    bm = BM25(corpus)
    print("   dung chi muc %.0fs" % (time.time() - t0))

    # Truy hoi bang MANH, khong chi bang cau hoi.
    #
    # BM25 khop voi CAU HOI nen no tim duoc chang 1 gan nhu luon luon, va
    # truot chang 2 — vi doan chang 2 thuong khong chia se tu nao voi cau hoi.
    # Do la ban chat multi-hop, va la ly do nguon gfm ton tai.
    # Nhung manh o c0 CO nhac thuc the bac cau ("Forrest Gump do Zemeckis dao
    # dien"), nen BM25 tren MANH co the voi toi chang 2 ma khong can do thi.
    # O day dung manh VANG lam tran tren: no tra loi "neu phan ra dung thi
    # truy hoi bang manh co du khong?"
    from s0f_decompose import from_annotation
    hit_all = hit_any = hit_claim = 0
    rank_of_gold: List[int] = []
    pools = []
    t1 = time.time()
    for i, it in enumerate(items, 1):
        cl = from_annotation(it["row"]) or []
        if cl:
            per = max(2, k_bm // max(1, len(cl)))
            seen, cpool = set(), []
            for c in cl:
                for j in bm.top(c, per):
                    if j not in seen:
                        seen.add(j); cpool.append(j)
            for j in bm.top(it["q"], k_bm - min(len(cpool), k_bm // 2)):
                if j not in seen:
                    seen.add(j); cpool.append(j)
            ctexts = {corpus[j] for j in cpool[:k_bm + k_rd]}
            hit_claim += int(all(g in ctexts for g in it["gold_keys"]))
        top = bm.top(it["q"], k_bm)
        rest = [j for j in range(len(corpus)) if j not in set(top)]
        pool = top + rng.sample(rest, min(k_rd, len(rest)))
        texts = [corpus[j] for j in pool]
        gold_in = [g for g in it["gold_keys"] if g in set(texts)]
        hit_all += int(len(gold_in) == len(it["gold_keys"]))
        hit_any += int(bool(gold_in))
        for g in it["gold_keys"]:
            full = bm.top(it["q"], 200)
            rank_of_gold.append(next((r + 1 for r, j in enumerate(full)
                                      if corpus[j] == g), 999))
        pools.append({"q": it["q"], "y": it["y"], "al": it["al"],
                      "pool": texts,
                      "gold_idx": sorted(texts.index(g) for g in gold_in)})
        if i % 100 == 0 or i == len(items):
            print("   %4d/%d  %.0fs" % (i, len(items), time.time() - t1), flush=True)

    N = len(items)
    print("\n" + "=" * 60)
    print("   pool = %d bm25 + %d ngau nhien = %d doan" % (k_bm, k_rd, k_bm + k_rd))
    print("   lay duoc TOAN BO doan vang : %5.1f%%   <- tran cua tang A" % (100 * hit_all / N))
    print("   lay duoc IT NHAT MOT       : %5.1f%%" % (100 * hit_any / N))
    print("   -- truy hoi bang MANH (tran tren, dung manh vang) --")
    print("   lay duoc TOAN BO doan vang : %5.1f%%" % (100 * hit_claim / N))
    r = sorted(rank_of_gold)
    for k in (1, 5, 10, 20, 50, 100):
        print("   doan vang trong top-%-3d    : %5.1f%%"
              % (k, 100 * sum(1 for x in r if x <= k) / len(r)))
    print("=" * 60)
    out = RAW.parent / "reports" / "pool_bm25.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"k_bm25": k_bm, "k_rand": k_rd, "n": N,
                               "recall_all": hit_all / N, "recall_any": hit_any / N,
                               "pools": pools[:400]}, ensure_ascii=False),
                   encoding="utf-8")
    print("-> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
