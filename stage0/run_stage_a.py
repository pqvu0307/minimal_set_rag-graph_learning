# -*- coding: utf-8 -*-
"""Chay TANG A dau-cuoi tren mot tap nho, voi LLM that.

Duong chay dung thu tu pipeline:
    pool -> c0 (che + bang phu) -> c1/c3/c4 (ba moc -> tau_q)
         -> c5 (liet ke nghiem phu) -> c6 (do v) -> c7 (chon S*)

Khac voi probe_covers.py: o kia KHONG goi LLM, chi do bang phu.  O day goi
that, nen tra loi duoc hai cau con treo:
    1  ti le MOI nghiem phu deu truot v  (c7 nhanh 2) — chua ai do
    2  v(C) co con can khong, khi c6 da do vai cover

HAI CHO PHAI DOC KY TRUOC KHI TIN SO:
    pool o day la doan KEM SAN cua bo du lieu, khong phai pool that 40 doan
    tu 3 nguon.  Doan vang chac chan co mat -> moi ti le deu LAC QUAN.
    R (moc 12 doan) xap xi bang trung khop tu voi cau hoi, vi chua co rerank.

    python stage0/run_stage_a.py --n=30 --corpus=musique
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from s0_common import load_config                                  # noqa: E402
from s0e_judge import Judge, OpenAICompatBackend    # noqa: E402
from s0f_decompose import LLMDecomposer, passages_of               # noqa: E402
from s0g_nli import (NLI, _words, coverage_table, minimal_covers,  # noqa: E402
                     should_exit)
from probe_covers import LOADERS, LARGE                            # noqa: E402


def rank_by_question(q: str, ps: List[str]) -> List[int]:
    """Xap xi rerank bang trung khop tu co trong so idf.  Chua co rerank that."""
    toks = [_words(p) for p in ps]
    df: Dict[str, int] = {}
    for t in toks:
        for w in t:
            df[w] = df.get(w, 0) + 1
    c = _words(q)
    sc = [sum(math.log(1 + len(ps) / (1 + df.get(w, 0))) for w in c & t) for t in toks]
    return sorted(range(len(ps)), key=lambda i: -sc[i])


def main() -> int:
    arg = {k.lstrip("-"): v for k, v in
           (a.split("=", 1) for a in sys.argv[1:] if "=" in a)}
    n = int(arg.get("n", 30))
    corpus = arg.get("corpus", "musique")
    cfg = load_config()
    cfg["nli"]["max_length"] = 256
    cfg["nli"]["model"] = arg.get("nli", LARGE)
    theta = float(arg.get("theta", 0.95))
    maxcov = int(arg.get("maxcov", 8))      # 98% truy van co <= 8 nghiem phu

    print("== TANG A dau-cuoi · %s · n=%d ==" % (corpus, n))
    print("   sinh  %s" % cfg["gen"]["model"])
    print("   cham  %s" % cfg["judge"]["judge_model"])
    print("   che   %s" % cfg["decompose"]["model"])
    data = LOADERS[corpus](n)

    nli = NLI(cfg)
    if nli.model is None:
        print("NLI hong:", nli.error); return 1
    judge = Judge(cfg)          # Judge tu dung backend; tham so 2 la telemetry
    dec = LLMDecomposer(cfg)
    if dec.cli is None:
        print("LLM che hong:", dec.error); return 1

    st = collections.Counter()
    sizes = collections.Counter()
    rows: List[Dict[str, Any]] = []
    t0 = time.time()

    for i, (ref_claims, ps, gold, meta) in enumerate(data, 1):
        q = meta.get("question", "")
        y = str(meta.get("answer", ""))
        al = meta.get("answer_aliases") or []
        rec: Dict[str, Any] = {"q": q[:90], "y": y, "gold": sorted(gold)}

        # ── c0 ──────────────────────────────────────────────────────
        claims, usage = dec(q, y, ps)
        if not claims:
            st["c0_che_hong"] += 1; rec["ket"] = "c0_che_hong"; rows.append(rec); continue
        ids = ["p%d" % k for k in range(len(ps))]
        tb = coverage_table(nli, ids, ps, claims, cfg)
        rec.update(J=tb["J"], claims=claims,
                   rescued=len(tb.get("rescued_claims") or {}))
        if should_exit(tb, cfg)[0] and tb["orphan_claims"]:
            st["c0_thoat"] += 1; rec["ket"] = "c0_thoat"; rows.append(rec); continue

        # ── c1 / c3 / c4 ────────────────────────────────────────────
        order = rank_by_question(q, ps)
        R = [ids[k] for k in order[:12]]
        txt = {ids[k]: ps[k] for k in range(len(ps))}
        r0 = judge.v(q, y, [], txt, aliases=al)
        rR = judge.v(q, y, R, txt, aliases=al)
        rC = judge.v(q, y, ids, txt, aliases=al)
        v0, vR, vC = r0.v, rR.v, rC.v
        rec["a0"] = r0.answer[:90]
        rec["aC"] = rC.answer[:90]
        v_ref = max(v0, vR, vC)
        tau = theta * (v_ref - v0) + v0
        rec.update(v0=round(v0, 3), vR=round(vR, 3), vC=round(vC, 3),
                   tau=round(tau, 3))
        if v_ref - v0 < 0.05:
            st["mau_so_nho"] += 1; rec["ket"] = "mau_so_nho"; rows.append(rec); continue

        # ── c5 / c6 ─────────────────────────────────────────────────
        covers = sorted(minimal_covers(tb, max_size=4, cap=64), key=len)
        rec["n_cover"] = len(covers)
        covers = covers[:maxcov]
        if not covers:
            st["khong_cover"] += 1; rec["ket"] = "khong_cover"; rows.append(rec); continue
        best = None
        measured = []
        for c in covers:
            rv = judge.v(q, y, list(c), txt, aliases=al)
            measured.append((sorted(c), round(rv.v, 3), rv.answer[:60]))
            if rv.v >= tau and best is None:
                best = (list(c), rv.v)
        rec["do"] = measured

        # ── c7 ──────────────────────────────────────────────────────
        if best is None:
            st["c7_moi_cover_truot"] += 1; rec["ket"] = "c7_moi_cover_truot"
        else:
            T, vT = best
            st["xong"] += 1
            sizes[len(T)] += 1
            rec.update(ket="xong", T=sorted(T), vT=round(vT, 3))
            g = {"p%d" % x for x in gold}
            if g:
                st["T_dung_tap_vang" if set(T) == g else "T_khac_tap_vang"] += 1
        rows.append(rec)
        if i % 5 == 0 or i == len(data):
            print("   %3d/%d  %5.0fs  %s" % (i, len(data), time.time() - t0,
                                             dict(st)))

    # ── bao cao ─────────────────────────────────────────────────────
    tot = len(rows)
    print("\n" + "=" * 62)
    for k, v in st.most_common():
        print("   %-22s %4d  %5.1f%%" % (k, v, 100 * v / tot))
    print("\n   |T|: %s" % dict(sorted(sizes.items())))
    if OpenAICompatBackend.degraded:
        print("   ! judge chay o CHE DO XAP XI MOT TOKEN — DeepInfra khong tra")
        print("     top_logprobs.  v(S) van lien tuc va don dieu, nhung do")
        print("     phan giai thap hon ban co top_logprobs day du.")
    print("   %.0fs tong, %.1fs/truy van" % (time.time() - t0, (time.time() - t0) / tot))
    print("=" * 62)
    out = pathlib.Path("artifacts/reports/run_stage_a_%s.json" % corpus)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stats": dict(st), "sizes": dict(sizes),
                               "rows": rows}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print("-> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
