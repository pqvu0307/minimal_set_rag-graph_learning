# -*- coding: utf-8 -*-
"""TANG A ban 2 — bo model sinh, cham bang suff(S) = B(S) x C(S).

Vi sao doi (cmp_scorers.py, 3200 tap con, hai corpus):

    bo cham        gold>gold_m1  gold>rand  tb     gia
    A  sinh+cham      66.7%        75.3%   71.0%  2 luot + GPU
    B  hoi thang      87.3%        99.3%   93.3%  1 luot
    C  nli gop        90.7%        89.3%   90.0%  0 luot
    B x C             95.4%        98.6%   97.0%  1 luot

A la v(S) cu.  No te nhat vi 31% so lan, model sinh doc DUNG tap vang roi
noi "INSUFFICIENT" — v(S) do nang luc cua Qwen, khong do bang chung.

HE QUA CAU TRUC: C(rong) = 0 theo dinh nghia, nen suff(rong) = 0 luon.
Khong con san v(rong), va hai chot chan cu gop lam mot:
    cu   v_ref - v(rong) < 0.05   (va THIEU chot tran thap -> nhan rac lot)
    moi  suff(C) < suff_min,  tau_q = theta * suff(C)
Nho the ca hong `mau_so_nho` (44% o ban 1) bien mat — no von la hieu ung
phu cua viec cham bang model sinh.

CHAY THEO HAI DOT, khong phai vong for theo truy van.  Ban dau toi viet
tuan tu tung truy van roi chi song song BEN TRONG moi truy van: 2.5s x 1000
= 42 phut, va A100 ngoi choi vi nut that la DO TRE API chu khong phai GPU.
Cac truy van doc lap hoan toan, nen:
    dot 0  bang phu + liet ke cover     GPU, tuan tu, nhanh
    dot 1  tinh TRAN cho MOI truy van   song song toan bo
    dot 2  cham MOI cover con lai       song song toan bo

    python stage0/run_stage_a2.py --n=1000 --corpus=musique --workers=48
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from s0_common import load_config                       # noqa: E402
from s0e_judge import _aliases_line                     # noqa: E402
from s0g_nli import NLI, coverage_table, minimal_covers  # noqa: E402
from cmp_scorers import SUFF_TMPL, block, p_yes         # noqa: E402
from probe_covers import LOADERS, LARGE                 # noqa: E402


def c_score(tb: Dict[str, Any], S: Sequence[str]) -> float:
    """C = mat xich yeu nhat: min qua cac manh cua max entail trong S."""
    s = set(S)
    per = [max((tb["grid"][k][j] for k in s), default=0.0)
           for j in range(tb["J"])]
    return min(per) if per else 0.0


def main() -> int:
    arg = {k.lstrip("-"): v for k, v in
           (a.split("=", 1) for a in sys.argv[1:] if "=" in a)}
    n = int(arg.get("n", 150))
    corpus = arg.get("corpus", "musique")
    theta = float(arg.get("theta", 0.95))
    suff_min = float(arg.get("suff_min", 0.30))
    maxcov = int(arg.get("maxcov", 8))
    W = int(arg.get("workers", 48))

    cfg = load_config()
    cfg["nli"]["max_length"] = 256
    cfg["nli"]["model"] = LARGE
    from openai import OpenAI
    cli = OpenAI(api_key=os.environ["DEEPINFRA_API_KEY"],
                 base_url=cfg["judge"]["base_url"], max_retries=4)
    jm = cfg["judge"]["judge_model"]
    nli = NLI(cfg)

    print("== TANG A ban 2 · %s · n=%d · suff_min=%.2f · %d luong =="
          % (corpus, n, suff_min, W))
    data = LOADERS[corpus](n)

    def B(job):
        q, y, al, S, txt = job
        return p_yes(cli, jm, SUFF_TMPL.format(
            passages=block(S, txt), q=q, y=y, aliases=_aliases_line(al, True)))

    # ── dot 0: bang phu + cover (GPU) ───────────────────────────────
    t0 = time.time()
    recs: List[Dict[str, Any]] = []
    for i, (claims, ps, gold, meta) in enumerate(data, 1):
        ids = ["p%d" % k for k in range(len(ps))]
        tb = coverage_table(nli, ids, ps, claims, cfg)
        covers = sorted(minimal_covers(tb, max_size=4, cap=64), key=len)
        recs.append({"q": meta["question"], "y": str(meta["answer"]),
                     "al": meta.get("answer_aliases") or [],
                     "ids": ids, "txt": dict(zip(ids, ps)), "tb": tb,
                     "gold": sorted(gold), "J": tb["J"],
                     "gold_in_pool": meta.get("gold_in_pool"),
                     "gold_total": meta.get("gold_total"),
                     "n_cover": len(covers), "covers": covers[:maxcov]})
        if i % 200 == 0 or i == len(data):
            print("   dot0 %4d/%d  %.0fs" % (i, len(data), time.time() - t0), flush=True)

    # ── dot 1: TRAN cho moi truy van, song song ─────────────────────
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=W) as ex:
        ceil_b = list(ex.map(B, [(r["q"], r["y"], r["al"], r["ids"], r["txt"])
                                 for r in recs]))
    calls = len(recs)
    for r, b in zip(recs, ceil_b):
        r["suff_C"] = (b or 0.0) * c_score(r["tb"], r["ids"])
        r["tau"] = theta * r["suff_C"]
    print("   dot1 tran: %d luot, %.0fs" % (calls, time.time() - t1))

    # ── dot 2: cham moi cover cua cac truy van qua tran ─────────────
    jobs, owner = [], []
    for r in recs:
        if r["suff_C"] < suff_min or not r["covers"]:
            continue
        for c in r["covers"]:
            jobs.append((r["q"], r["y"], r["al"], list(c), r["txt"]))
            owner.append((r, c))
    t2 = time.time()
    with ThreadPoolExecutor(max_workers=W) as ex:
        bs = list(ex.map(B, jobs))
    calls += len(jobs)
    print("   dot2 cover: %d luot, %.0fs" % (len(jobs), time.time() - t2))
    for (r, c), b in zip(owner, bs):
        r.setdefault("scored", []).append((list(c), (b or 0.0) * c_score(r["tb"], c)))

    # ── ket luan ────────────────────────────────────────────────────
    st = collections.Counter()
    sizes = collections.Counter()
    rows = []
    for r in recs:
        out = {k: r[k] for k in ("q", "y", "gold", "J", "n_cover")}
        out["q"] = out["q"][:90]
        out["suff_C"] = round(r["suff_C"], 3); out["tau"] = round(r["tau"], 3)
        if r["suff_C"] < suff_min:
            st["tran_thap"] += 1; out["ket"] = "tran_thap"
        elif not r["covers"]:
            st["khong_cover"] += 1; out["ket"] = "khong_cover"
        else:
            sc = r.get("scored", [])
            out["do"] = [(sorted(c), round(s, 3)) for c, s in sc]
            ok = [(c, s) for c, s in sc if s >= r["tau"]]
            # thua man NGUONG thi uu tien tap NHO nhat, roi diem cao nhat
            if not ok:
                st["moi_cover_truot"] += 1; out["ket"] = "moi_cover_truot"
                out["best"] = round(max((s for _, s in sc), default=0.0), 3)
            else:
                T, sT = min(ok, key=lambda x: (len(x[0]), -x[1]))
                st["xong"] += 1; sizes[len(T)] += 1
                out.update(ket="xong", T=sorted(T), suff_T=round(sT, 3))
                g = {"p%d" % x for x in r["gold"]}
                if g:
                    st["T_dung" if set(T) == g else "T_khac"] += 1
        rows.append(out)

    tot = len(rows)
    # Tren pool THAT, doan vang co the khong duoc lay ve.  Phai tach ro
    # "ta tim truot" voi "pool khong co" — khong thi hai loi doi lap nhau
    # bi gop thanh mot con so vo nghia.
    gp = [r for r in recs if r.get("gold_total")]
    if gp:
        full = sum(1 for r in gp if r["gold_in_pool"] == r["gold_total"])
        print("\n   POOL: lay duoc toan bo doan vang %d/%d = %.1f%%"
              % (full, len(gp), 100 * full / len(gp)))
    print("\n" + "=" * 62)
    for k, v in st.most_common():
        print("   %-18s %4d  %5.1f%%" % (k, v, 100 * v / tot))
    if st["xong"]:
        print("\n   trong so ca XONG: T dung tap vang %.0f%%"
              % (100 * st["T_dung"] / st["xong"]))
    print("   |T|: %s" % dict(sorted(sizes.items())))
    print("   %d luot llm / %d truy van = %.1f moi truy van"
          % (calls, tot, calls / max(1, tot)))
    print("   tong %.0fs  (dot0 %.0fs · dot1 %.0fs · dot2 %.0fs)"
          % (time.time() - t0, t1 - t0, t2 - t1, time.time() - t2))
    print("=" * 62)
    out = pathlib.Path("artifacts/reports/run_stage_a2_%s.json" % corpus)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stats": dict(st), "sizes": dict(sizes),
                               "calls": calls, "rows": rows},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print("-> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
