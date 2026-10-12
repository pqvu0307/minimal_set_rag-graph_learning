# -*- coding: utf-8 -*-
"""Phan ra co doan (grounded) vs khong co doan (bare) — LLM dang DOC hay NHO?

Cau hoi: voi corpus khong co phan ra san, LLM phai tu cung thuc the bac cau
("Forrest Gump" khong nam trong cau hoi lan dap an).  Dua pool vao prompt thi
no DOC duoc; khong dua thi no phai NHO.  Khac biet co that khong?

Do tren MuSiQue, noi co dap so: question_decomposition cho dap an TRUNG GIAN
cua tung chang, tuc chinh la thuc the bac cau.

    bare      q + y                  thuan tri nho
    grounded  q + y + pool           doc duoc

Ba chi so:
    bat duoc thuc the bac cau   co nhac dap an trung gian khong
    cham toi dap an             co nhac y khong
    so manh J khop              co rung chang nao khong

    python stage0/cmp_decompose.py --n=120
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from s0_common import load_config                              # noqa: E402
from s0f_decompose import (LLMDecomposer, _norm, from_annotation,  # noqa: E402
                           passages_of, reaches_answer)

RAW = pathlib.Path("artifacts/raw")


def main() -> int:
    arg = {k.lstrip("-"): v for k, v in
           (a.split("=", 1) for a in sys.argv[1:] if "=" in a)}
    n = int(arg.get("n", 120))
    cfg = load_config()
    dec = LLMDecomposer(cfg)
    if dec.cli is None:
        print("LLM chua san sang:", dec.error); return 1

    f = RAW / "musique" / "dev.jsonl"
    rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()]
    items = []
    for r in rows:
        if not (r.get("question_decomposition") and r.get("paragraphs")):
            continue
        # thuc the bac cau = dap an TRUNG GIAN, bo chang cuoi (do chinh la y)
        br = [str(s.get("answer", "")).strip()
              for s in r["question_decomposition"][:-1]
              if str(s.get("answer", "")).strip()]
        if not br:
            continue
        items.append((r, br, passages_of(r), len(from_annotation(r) or [])))
        if len(items) >= n:
            break
    print("== %d truy van, moi cai chay 2 che do ==" % len(items))

    def run(mode):
        def one(it):
            r, br, ps, Jref = it
            q, y = r["question"], str(r["answer"])
            cl, u = dec(q, y, ps if mode == "grounded" else None)
            blob = _norm(" ".join(cl))
            return {"J": len(cl), "Jref": Jref,
                    "bridge": all(_norm(b) in blob for b in br),
                    "reach": reaches_answer(cl, y, r.get("answer_aliases")),
                    "ungrounded": u.get("ungrounded", 0),
                    "tok": u.get("prompt_tokens", 0) + u.get("output_tokens", 0),
                    "claims": cl}
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=10) as ex:
            out = list(ex.map(one, items))
        print("   %-9s xong %.0fs" % (mode, time.time() - t0))
        return out

    res = {m: run(m) for m in ("grounded", "bare")}

    print("\n" + "=" * 62)
    print("%-26s %10s %10s" % ("", "grounded", "bare"))
    N = len(items)
    for lab, f_ in (("bat duoc thuc the bac cau", lambda x: x["bridge"]),
                    ("cham toi dap an", lambda x: x["reach"]),
                    ("so manh J khop ban goc", lambda x: x["J"] == x["Jref"]),
                    ("J nho hon goc (rung chang)", lambda x: x["J"] < x["Jref"])):
        print("%-26s %9.0f%% %9.0f%%"
              % (lab, 100 * sum(f_(x) for x in res["grounded"]) / N,
                 100 * sum(f_(x) for x in res["bare"]) / N))
    for lab, f_ in (("token moi truy van", lambda x: x["tok"]),
                    ("so manh trung binh", lambda x: x["J"])):
        print("%-26s %10.1f %10.1f"
              % (lab, sum(f_(x) for x in res["grounded"]) / N,
                 sum(f_(x) for x in res["bare"]) / N))
    print("%-26s %10d %10s"
          % ("manh bi loai vi trich bia", sum(x["ungrounded"] for x in res["grounded"]), "-"))
    print("=" * 62)
    out = RAW.parent / "reports" / "cmp_decompose.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {m: [{k: v for k, v in x.items()} for x in r] for m, r in res.items()},
        indent=1, ensure_ascii=False), encoding="utf-8")
    print("-> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
