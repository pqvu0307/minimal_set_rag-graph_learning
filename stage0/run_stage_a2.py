# -*- coding: utf-8 -*-
"""TANG A ban 2 — bo model sinh, cham bang suff(S) = B(S) x C(S).

Vi sao doi (so tren 150 truy van / 600 tap con, cmp_scorers.py):

    bo cham        gold>gold_m1  gold>rand  tb     gia
    A  sinh+cham      66.7%        75.3%   71.0%  2 luot + GPU
    B  hoi thang      87.3%        99.3%   93.3%  1 luot
    C  nli gop        90.7%        89.3%   90.0%  0 luot
    B x C             94.7%        98.0%   96.3%  1 luot

A la v(S) cu.  No te nhat vi 31% so lan, model sinh doc DUNG tap vang roi
noi "INSUFFICIENT" — v(S) do nang luc cua Qwen, khong do bang chung.

MOT HE QUA CAU TRUC: C(rong) = 0 theo dinh nghia, nen suff(rong) = 0 luon.
Khong con san v(rong) nua, va hai chot chan cu gop lam mot:

    cu   v_ref - v(rong) < 0.05      "truy hoi khong dong gop"
         (va THIEU chot tran thap -> tau_q vo nghia, nhan rac lot)
    moi  suff(C) < suff_min          "pool khong do duoc dap an"
         tau_q = theta * suff(C)

Nho the, ca hong `mau_so_nho` (44% o ban 1) bien mat — no von la hieu ung
phu cua viec cham bang model sinh.

    python stage0/run_stage_a2.py --n=150 --corpus=musique
"""
from __future__ import annotations

import collections
import json
import math
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

    cfg = load_config()
    cfg["nli"]["max_length"] = 256
    cfg["nli"]["model"] = LARGE
    from openai import OpenAI
    cli = OpenAI(api_key=os.environ["DEEPINFRA_API_KEY"],
                 base_url=cfg["judge"]["base_url"])
    jm = cfg["judge"]["judge_model"]
    nli = NLI(cfg)

    print("== TANG A ban 2 · %s · n=%d · suff_min=%.2f ==" % (corpus, n, suff_min))
    print("   cham  %s   (khong dung model sinh)" % jm)
    data = LOADERS[corpus](n)

    def B(q, y, al, S, txt):
        return p_yes(cli, jm, SUFF_TMPL.format(
            passages=block(S, txt), q=q, y=y, aliases=_aliases_line(al, True)))

    st = collections.Counter()
    sizes = collections.Counter()
    rows: List[Dict[str, Any]] = []
    calls = 0
    t0 = time.time()

    for i, (claims, ps, gold, meta) in enumerate(data, 1):
        q, y = meta["question"], str(meta["answer"])
        al = meta.get("answer_aliases") or []
        ids = ["p%d" % k for k in range(len(ps))]
        txt = dict(zip(ids, ps))
        rec: Dict[str, Any] = {"q": q[:90], "y": y, "gold": sorted(gold)}

        tb = coverage_table(nli, ids, ps, claims, cfg)
        rec["J"] = tb["J"]

        # tran: ca pool.  C(toan pool) ~ 1 neu moi manh co nguoi do.
        ceil = B(q, y, al, ids, txt) * c_score(tb, ids)
        calls += 1
        rec["suff_C"] = round(ceil, 3)
        if ceil < suff_min:
            st["tran_thap"] += 1; rec["ket"] = "tran_thap"; rows.append(rec); continue
        tau = theta * ceil
        rec["tau"] = round(tau, 3)

        covers = sorted(minimal_covers(tb, max_size=4, cap=64), key=len)
        rec["n_cover"] = len(covers)
        covers = covers[:maxcov]
        if not covers:
            st["khong_cover"] += 1; rec["ket"] = "khong_cover"; rows.append(rec); continue

        with ThreadPoolExecutor(max_workers=8) as ex:
            bs = list(ex.map(lambda c: B(q, y, al, list(c), txt), covers))
        calls += len(covers)
        scored = [(list(c), b * c_score(tb, c)) for c, b in zip(covers, bs)]
        rec["do"] = [(sorted(c), round(s, 3)) for c, s in scored]

        ok = [(c, s) for c, s in scored if s >= tau]
        if not ok:
            st["moi_cover_truot"] += 1; rec["ket"] = "moi_cover_truot"
        else:
            T, sT = min(ok, key=lambda x: (len(x[0]), -x[1]))
            st["xong"] += 1; sizes[len(T)] += 1
            rec.update(ket="xong", T=sorted(T), suff_T=round(sT, 3))
            g = {"p%d" % x for x in gold}
            if g:
                st["T_dung" if set(T) == g else "T_khac"] += 1
        rows.append(rec)
        if i % 25 == 0 or i == len(data):
            print("   %4d/%d  %5.0fs  %s" % (i, len(data), time.time() - t0, dict(st)))

    tot = len(rows)
    print("\n" + "=" * 62)
    for k, v in st.most_common():
        print("   %-18s %4d  %5.1f%%" % (k, v, 100 * v / tot))
    xong = st["xong"]
    if xong:
        print("\n   trong so ca XONG: T dung tap vang %.0f%%" % (100 * st["T_dung"] / xong))
    print("   |T|: %s" % dict(sorted(sizes.items())))
    print("   %d luot llm / %d truy van = %.1f luot moi truy van"
          % (calls, tot, calls / max(1, tot)))
    print("   %.0fs, %.1fs/truy van" % (time.time() - t0, (time.time() - t0) / tot))
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
