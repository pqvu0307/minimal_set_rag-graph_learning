# -*- coding: utf-8 -*-
"""Pool ba nguon — bm25 + MO RONG THEO LIEN KET + ngau nhien.

Do duoc o build_pool.py: bm25 lay duoc TOAN BO doan vang chi 32.5%, nhung
lay duoc it nhat mot toi 96.2%.  Khoang cach do la ban chat multi-hop —
bm25 khop voi CAU HOI nen bat duoc chang 1, va truot chang 2 vi doan chang
2 thuong khong chia se tu nao voi cau hoi.

Nguon gfm sinh ra de vá cho dung cho do, nhung no can OpenIE (~$800).
O day thu mot nguon thu ba RE HON NHIEU, nham dung lo do:

    cac doan nay la bai Wikipedia CO TIEU DE.
    noi A -> B neu TIEU DE cua B xuat hien trong than cua A.

Do chinh la cau truc ma HotpotQA / MuSiQue / 2Wiki duoc xay dung tu do:
chang 2 la bai ma chang 1 TRO TOI.  Khong can LLM, khong can nhung, khong
can GPU — chi can mot bang tra tieu de.

    python stage0/pool_sources.py --n=400 --k_bm25=15 --k_link=5 --k_rand=5
"""
from __future__ import annotations

import collections
import json
import pathlib
import random
import re
import sys
import time
from typing import Any, Dict, List, Sequence, Set, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_pool import BM25, RAW, toks                 # noqa: E402


def load(n: int) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """Tra ve (items, titles, texts) — giu tieu de TACH RIENG khoi than."""
    f = RAW / "musique" / "dev.jsonl"
    rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()]
    key2i: Dict[str, int] = {}
    titles: List[str] = []
    texts: List[str] = []
    items = []
    for r in rows:
        if not (r.get("question_decomposition") and r.get("paragraphs")):
            continue
        gold = {s.get("paragraph_support_idx") for s in r["question_decomposition"]}
        gk = []
        for p in r["paragraphs"]:
            t = (p.get("paragraph_text") or "").strip()
            ti = (p.get("title") or "").strip()
            if not t:
                continue
            k = ti + " | " + t
            if k not in key2i:
                key2i[k] = len(titles); titles.append(ti); texts.append(t)
            if p.get("idx") in gold:
                gk.append(key2i[k])
        if gk:
            items.append({"q": r["question"], "y": str(r["answer"]),
                          "gold": set(gk), "row": r})
        if len(items) >= n:
            break
    return items, titles, texts


def link_graph(titles: Sequence[str], texts: Sequence[str]) -> Dict[int, List[int]]:
    """A -> B neu tieu de cua B xuat hien trong than cua A.

    Quet n-gram cua than roi tra bang tieu de: O(so tu), khong phai O(so doan^2).
    """
    by_title: Dict[str, List[int]] = collections.defaultdict(list)
    maxw = 1
    for i, t in enumerate(titles):
        w = tuple(re.findall(r"[a-z0-9]+", t.lower()))
        if w:
            by_title[" ".join(w)].append(i)
            maxw = max(maxw, len(w))
    maxw = min(maxw, 8)
    out: Dict[int, List[int]] = {}
    for i, body in enumerate(texts):
        w = re.findall(r"[a-z0-9]+", body.lower())
        hit: Set[int] = set()
        for a in range(len(w)):
            for L in range(1, min(maxw, len(w) - a) + 1):
                g = " ".join(w[a:a + L])
                if g in by_title:
                    hit.update(by_title[g])
        hit.discard(i)
        out[i] = sorted(hit)
    return out


def main() -> int:
    arg = {k.lstrip("-"): v for k, v in
           (a.split("=", 1) for a in sys.argv[1:] if "=" in a)}
    n = int(arg.get("n", 400))
    k_bm = int(arg.get("k_bm25", 15))
    k_lk = int(arg.get("k_link", 5))
    k_rd = int(arg.get("k_rand", 5))
    rng = random.Random(0)

    t0 = time.time()
    items, titles, texts = load(n)
    docs = ["%s. %s" % (a, b) for a, b in zip(titles, texts)]
    print("== %d truy van · corpus %d doan ==" % (len(items), len(docs)))
    bm = BM25(docs)
    g = link_graph(titles, texts)
    deg = sum(len(v) for v in g.values()) / max(1, len(g))
    print("   bm25 + do thi lien ket xong %.0fs · bac trung binh %.1f"
          % (time.time() - t0, deg))

    def recall(pick) -> Tuple[float, float]:
        allh = anyh = 0
        for it in items:
            P = pick(it)
            inter = it["gold"] & P
            allh += int(len(inter) == len(it["gold"]))
            anyh += int(bool(inter))
        return 100 * allh / len(items), 100 * anyh / len(items)

    def bm_only(it):
        return set(bm.top(it["q"], k_bm + k_lk))

    def bm_rand(it):
        s = set(bm.top(it["q"], k_bm + k_lk))
        rest = [j for j in range(len(docs)) if j not in s]
        return s | set(rng.sample(rest, min(k_rd, len(rest))))

    def bm_link(it):
        seed = bm.top(it["q"], k_bm)
        s = set(seed)
        # mo rong mot buoc: uu tien doan duoc NHIEU seed tro toi
        cnt: Dict[int, int] = collections.Counter()
        for j in seed:
            for kk in g.get(j, []):
                if kk not in s:
                    cnt[kk] += 1
        for kk, _ in cnt.most_common(k_lk):
            s.add(kk)
        return s

    def bm_link_rand(it):
        s = bm_link(it)
        rest = [j for j in range(len(docs)) if j not in s]
        return s | set(rng.sample(rest, min(k_rd, len(rest))))

    print("\n" + "=" * 62)
    print("%-34s %9s %9s" % ("nguon", "TOAN BO", "it nhat 1"))
    for name, fn in (("bm25 %d" % (k_bm + k_lk), bm_only),
                     ("bm25 %d + ngau nhien %d" % (k_bm + k_lk, k_rd), bm_rand),
                     ("bm25 %d + lien ket %d" % (k_bm, k_lk), bm_link),
                     ("bm25 %d + lien ket %d + ngau nhien %d"
                      % (k_bm, k_lk, k_rd), bm_link_rand)):
        a, b = recall(fn)
        print("%-34s %8.1f%% %8.1f%%" % (name, a, b))
    print("=" * 62)
    out = RAW.parent / "reports" / "pool_sources.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": len(items), "k_bm25": k_bm, "k_link": k_lk,
                               "k_rand": k_rd, "avg_degree": deg},
                              indent=1), encoding="utf-8")
    print("-> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
