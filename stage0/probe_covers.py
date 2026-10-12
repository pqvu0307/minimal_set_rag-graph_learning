# -*- coding: utf-8 -*-
"""Do PHAN BO SO NGHIEM PHU tren corpus that.  Khong goi mot luot LLM nao.

Vi sao phep do nay quyet dinh: g_theta ton tai de khoi goi LLM cho moi tap
con TRONG KHONG GIAN TIM KIEM.  Nhung bang phu da thu khong gian do ve dung
danh sach nghiem phu.  Neu danh sach do thuong nho thi do v het bang LLM
cung re — va ca pha 1 (surrogate, 4 vong, ~$251) la thua.

CANH BAO khi doc ket qua: pool o day la doan KEM SAN cua bo du lieu, tuc
doan vang CHAC CHAN co mat.  Do la pool THIEN VI.  Tren pool that (40 doan
tu 3 nguon) thi manh mo coi co HAI nguon goc — NLI truot VA truy hoi truot —
va con so se khac.

    python stage0/probe_covers.py --n=200 --corpus=musique
    python stage0/probe_covers.py --n=200 --corpus=2wiki --thr=0.5
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from s0_common import load_config                                  # noqa: E402
from s0f_decompose import (from_annotation, from_evidences,        # noqa: E402
                           passages_of)
from s0g_nli import NLI, coverage_table, minimal_covers, should_exit  # noqa: E402

RAW = pathlib.Path("artifacts/raw")
LARGE = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


# ── nap corpus: moi cai tra ve (claims, passages, chi so doan vang) ──
def load_musique(n: int) -> List[Tuple[List[str], List[str], set, Dict]]:
    f = RAW / "musique" / "dev.jsonl"
    rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()]
    out = []
    for r in rows:
        if not (r.get("question_decomposition") and r.get("paragraphs")):
            continue
        cl, ps = from_annotation(r), passages_of(r)
        if not cl or not ps:
            continue
        gold = {s["paragraph_support_idx"] for s in r["question_decomposition"]
                if s.get("paragraph_support_idx") is not None}
        out.append((cl, ps, gold, r))
        if len(out) >= n:
            break
    return out


def load_2wiki(n: int) -> List[Tuple[List[str], List[str], set, Dict]]:
    import pandas as pd
    d = pd.read_parquet("hf://datasets/xanhho/2WikiMultihopQA/dev.parquet")
    J = lambda s: json.loads(s) if isinstance(s, str) else s
    out = []
    for i in range(len(d)):
        r = d.iloc[i]
        ctx = J(r["context"])
        row = {"evidences": J(r["evidences"])}
        cl = from_evidences(row)
        if not cl or not ctx:
            continue
        ps = ["%s. %s" % (c[0], " ".join(c[1])) for c in ctx]
        titles = {c[0]: k for k, c in enumerate(ctx)}
        gold = {titles[t] for t, _ in J(r["supporting_facts"]) if t in titles}
        out.append((cl, ps, gold,
                    {"type": r["type"], "question": r["question"],
                     "answer": r["answer"], "answer_aliases": []}))
        if len(out) >= n:
            break
    return out


LOADERS = {"musique": load_musique, "2wiki": load_2wiki}


def main() -> int:
    arg = {k.lstrip("-"): v for k, v in
           (a.split("=", 1) for a in sys.argv[1:] if "=" in a)}
    n = int(arg.get("n", 200))
    corpus = arg.get("corpus", "musique")
    cfg = load_config()
    cfg["nli"]["max_length"] = 256
    cfg["nli"]["model"] = arg.get("model", LARGE)
    if "thr" in arg:
        cfg["nli"]["entail_threshold"] = float(arg["thr"])
    if "rescue" in arg:
        cfg["nli"]["rescue_orphan_lexical"] = arg["rescue"] not in ("0", "false")

    print("== %s · n=%d · nguong %.2f · cuu-tu=%s =="
          % (corpus, n, cfg["nli"]["entail_threshold"],
             cfg["nli"].get("rescue_orphan_lexical", True)))
    data = LOADERS[corpus](n)
    print("   nap %d truy van, %d doan/truy van (trung binh)"
          % (len(data), sum(len(p) for _, p, _, _ in data) // max(1, len(data))))
    print("   J: %s" % dict(sorted(collections.Counter(
        len(c) for c, _, _, _ in data).items())))

    nli = NLI(cfg)
    if nli.model is None:
        print("khong nap duoc NLI:", nli.error); return 1
    print("   NLI %s (%s)\n" % (cfg["nli"]["model"].split("/")[-1], nli.device))

    ncov = collections.Counter()
    exits = orph = vac = resc = 0
    gold_in = gold_eq = gold_tot = 0
    t0 = time.time()
    for i, (claims, ps, gold, meta) in enumerate(data, 1):
        ids = ["p%d" % k for k in range(len(ps))]
        tb = coverage_table(nli, ids, ps, claims, cfg)
        exits += int(should_exit(tb, cfg)[0])
        orph += int(bool(tb["orphan_claims"]))
        vac += int(bool(tb["vacuous_claims"]))
        resc += int(bool(tb.get("rescued_claims")))
        covers = minimal_covers(tb, max_size=4, cap=256)
        ncov[len(covers)] += 1
        if gold:
            gold_tot += 1
            g = {"p%d" % x for x in gold}
            cs = [set(c) for c in covers]
            gold_in += int(any(g <= c for c in cs))
            gold_eq += int(any(g == c for c in cs))
        if i % 50 == 0 or i == len(data):
            el = time.time() - t0
            print("   %4d/%d  %5.1fs  (%.2fs/truy van)" % (i, len(data), el, el / i))

    tot = sum(ncov.values()) or 1
    nz = sum(v for k, v in ncov.items() if k > 0)
    mean = sum(k * v for k, v in ncov.items() if k > 0) / max(1, nz)
    print("\n" + "=" * 60)
    run = 0
    for k in sorted(ncov):
        run += ncov[k]
        print("   %3d nghiem phu : %4d  %5.1f%%  (tich luy %5.1f%%)"
              % (k, ncov[k], 100 * ncov[k] / tot, 100 * run / tot))
    print("\n   co >=1 nghiem phu : %.0f%%   trung binh %.2f   <=8: %.0f%%"
          % (100 * nz / tot, mean,
             100 * sum(v for k, v in ncov.items() if 0 < k <= 8) / max(1, nz)))
    print("   mo coi %.0f%% · rong %.0f%% · duoc cuu %.0f%% · thoat som %.0f%%"
          % (100 * orph / tot, 100 * vac / tot, 100 * resc / tot, 100 * exits / tot))
    if gold_tot:
        print("   tap vang trong mot nghiem phu: %.0f%%   bang dung: %.0f%%"
              % (100 * gold_in / gold_tot, 100 * gold_eq / gold_tot))
    print("=" * 60)

    out = RAW.parent / "reports" / ("probe_covers_%s.json" % corpus)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "corpus": corpus, "n": tot, "covers": dict(ncov), "mean_covers": round(mean, 2),
        "orphan": orph, "vacuous": vac, "rescued": resc, "exit": exits,
        "gold_in_cover": gold_in, "gold_eq_cover": gold_eq, "gold_tot": gold_tot,
        "threshold": cfg["nli"]["entail_threshold"], "model": cfg["nli"]["model"],
        "seconds": round(time.time() - t0, 1)}, indent=2), encoding="utf-8")
    print("-> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
