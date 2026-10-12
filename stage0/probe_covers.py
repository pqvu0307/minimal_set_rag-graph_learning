# -*- coding: utf-8 -*-
"""Do PHAN BO SO NGHIEM PHU tren MuSiQue that.

Vi sao phep do nay quyet dinh: g_theta ton tai de khoi goi LLM cho moi tap
con TRONG KHONG GIAN TIM KIEM.  Nhung bang phu da thu khong gian do ve dung
danh sach nghiem phu.  Neu danh sach do thuong nho thi do v het bang LLM
cung re — va ca pha 1 (surrogate, 4 vong active learning, ~$177 + gio GPU)
la thua.

Chay:  python stage0/probe_covers.py [N]
Khong can GPU.  Khong goi mot luot LLM nao — chi NLI chay local.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from s0_common import load_config                      # noqa: E402
from s0f_decompose import from_annotation, passages_of  # noqa: E402
from s0g_nli import NLI, coverage_table, minimal_covers, should_exit  # noqa: E402

DEV = pathlib.Path("artifacts/raw/musique/dev.jsonl")


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    cfg = load_config()
    cfg["nli"]["max_length"] = 256          # doan ~100 tu, manh 1 cau — 256 du

    rows = [json.loads(l) for l in DEV.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r.get("question_decomposition") and r.get("paragraphs")]
    rows = rows[:n]
    print("== %d truy van MuSiQue dev ==" % len(rows))

    # dang cau hoi con: ">> quan he" hay cau hoi tu nhien?
    sub = [s.get("question", "") for r in rows for s in r["question_decomposition"]]
    arrow = sum(1 for s in sub if ">>" in s)
    print("   cau hoi con: %d, trong do %d (%.0f%%) dang 'X >> quan he'"
          % (len(sub), arrow, 100 * arrow / max(1, len(sub))))

    dstat: Dict[str, int] = {}
    prepared = []
    for r in rows:
        cl = from_annotation(r, dstat)
        ps = passages_of(r)
        if cl and ps:
            prepared.append((r, cl, ps))
    npat, nfb = dstat.get("pattern", 0), dstat.get("fallback", 0)
    print("   viet lai thanh menh de: %d theo mau, %d theo khuon  (mau %.0f%%)"
          % (npat, nfb, 100 * npat / max(1, npat + nfb)))
    print("   dung duoc %d truy van\n" % len(prepared))

    nli = NLI(cfg)
    if nli.model is None:
        print("khong nap duoc NLI:", nli.error)
        return 1
    print("   NLI = %s  (%s)\n" % (cfg["nli"]["model"], nli.device))

    ncov = collections.Counter()
    nJ = collections.Counter()
    exits = orph = vac = 0
    gold_in_covers = gold_eq_some = 0
    gold_tot = 0
    t0 = time.time()

    for i, (r, claims, ps) in enumerate(prepared, 1):
        ids = ["p%d" % k for k in range(len(ps))]
        tb = coverage_table(nli, ids, ps, claims, cfg)
        ex, _ = should_exit(tb, cfg)
        exits += int(ex)
        orph += int(bool(tb["orphan_claims"]))
        vac += int(bool(tb["vacuous_claims"]))
        covers = minimal_covers(tb, max_size=4, cap=256)
        ncov[len(covers)] += 1
        nJ[tb["J"]] += 1

        gold = {s.get("paragraph_support_idx")
                for s in r["question_decomposition"]
                if s.get("paragraph_support_idx") is not None}
        if gold:
            gold_tot += 1
            gids = {"p%d" % g for g in gold}
            csets = [set(c) for c in covers]
            gold_in_covers += int(any(gids <= c for c in csets))
            gold_eq_some += int(any(gids == c for c in csets))

        if i % 25 == 0 or i == len(prepared):
            el = time.time() - t0
            print("   %4d/%d   %5.1fs   (%.2fs/truy van, con ~%.0fs)"
                  % (i, len(prepared), el, el / i, el / i * (len(prepared) - i)))

    print("\n" + "=" * 64)
    print("PHAN BO SO NGHIEM PHU TOI TIEU  (tang 0 cua c5)")
    tot = sum(ncov.values())
    run = 0
    for k in sorted(ncov):
        run += ncov[k]
        print("   %3d nghiem phu : %4d truy van  %5.1f%%   (tich luy %5.1f%%)"
              % (k, ncov[k], 100 * ncov[k] / tot, 100 * run / tot))
    le8 = sum(v for k, v in ncov.items() if 0 < k <= 8)
    nz = sum(v for k, v in ncov.items() if k > 0)
    print("\n   J (so manh)      : %s" % dict(sorted(nJ.items())))
    print("   co >=1 nghiem phu: %d / %d = %.0f%%" % (nz, tot, 100 * nz / tot))
    if nz:
        print("   trong so do, <=8 : %d / %d = %.0f%%" % (le8, nz, 100 * le8 / nz))
    print("\n   thoat som (should_exit): %4d  %.0f%%" % (exits, 100 * exits / tot))
    print("   co manh MO COI         : %4d  %.0f%%" % (orph, 100 * orph / tot))
    print("   co manh RONG           : %4d  %.0f%%" % (vac, 100 * vac / tot))
    if gold_tot:
        print("\n   tap vang NAM TRONG mot nghiem phu : %.0f%%"
              % (100 * gold_in_covers / gold_tot))
        print("   tap vang BANG DUNG mot nghiem phu : %.0f%%"
              % (100 * gold_eq_some / gold_tot))
    print("=" * 64)

    out = pathlib.Path("artifacts/reports/probe_covers.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n": tot, "covers": dict(ncov), "J": dict(nJ),
        "exit": exits, "orphan": orph, "vacuous": vac,
        "arrow_form_pct": round(100 * arrow / max(1, len(sub)), 1),
        "declarativize": {"pattern": npat, "fallback": nfb},
        "gold_subset_of_a_cover": gold_in_covers, "gold_equals_a_cover": gold_eq_some,
        "gold_tot": gold_tot,
        "seconds": round(time.time() - t0, 1),
    }, indent=2), encoding="utf-8")
    print("-> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
