# -*- coding: utf-8 -*-
"""Dung lai RINSE theo mo ta, va so voi B / C / BxC tren CUNG tap con.

KHONG phai RINSE that: bai (arXiv:2609.37469) khong phat hanh weights, va
huong chieu cua doc gia thu ba hoc tu du lieu cua ho.  Day la ban DUNG LAI
theo spec, nen goi la RINSE-like.  Ba tin hieu:

  R1 phu noi dung    reranker cross-encoder, min qua cac CUA SO cua cau hoi
  R2 bang chung span QA trich xuat co lua chon "khong co dap an"
                     -> tin hieu TOI CHUA CO; khac han entailment
  R3 doc cheo doan   model nho CHAY LOCAL cham "du hay chua"
                     -> ban mien phi cua B; neu no bang B thi bo duoc API

Va mot y cua RINSE ma ta thieu: HIEU CHINH CO.  Tap to thi diem cao mot cach
tam thuong, nen tru di xu huong tuyen tinh theo log(1+|S|) truoc khi so.
Do la dung cho cap gold (|S|=2) vs gold_p1 (|S|=3) cua ta.

    python stage0/cmp_rinse.py --n=150
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import sys
import time
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

RINSE_LM = "Qwen/Qwen2.5-1.5B-Instruct"
RERANK = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QA_MODEL = "deepset/roberta-base-squad2"


class Rerank:
    """R1 — phu noi dung.  Cat cau hoi thanh cua so, moi cua so phai duoc MOT
    doan nao do do.  Lay min qua cac cua so: mat xich yeu nhat."""

    def __init__(self):
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
        self.t = AutoTokenizer.from_pretrained(RERANK)
        self.m = AutoModelForSequenceClassification.from_pretrained(
            RERANK).eval().to("cuda")
        self.torch = torch

    def windows(self, q: str, w: int = 4) -> List[str]:
        ws = self.t.tokenize(q)
        if len(ws) <= w:
            return [q]
        return [self.t.convert_tokens_to_string(ws[i:i + w])
                for i in range(0, len(ws) - w + 1, max(1, w // 2))]

    def score(self, q: str, passages: Sequence[str]) -> float:
        win = self.windows(q)
        pairs = [(x, p) for x in win for p in passages]
        if not pairs:
            return 0.0
        out = []
        for i in range(0, len(pairs), 128):
            b = pairs[i:i + 128]
            enc = self.t([x for x, _ in b], [p for _, p in b], padding=True,
                         truncation=True, max_length=256,
                         return_tensors="pt").to("cuda")
            with self.torch.no_grad():
                out += self.m(**enc).logits[:, 0].tolist()
        per = [max(out[k * len(passages):(k + 1) * len(passages)])
               for k in range(len(win))]
        return min(per)


class AnswerSpan:
    """R2 — co doan nao chua span ma model QA thich hon 'khong co dap an'?"""

    def __init__(self):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        self.t = AutoTokenizer.from_pretrained(QA_MODEL)
        self.m = AutoModelForQuestionAnswering.from_pretrained(
            QA_MODEL).eval().to("cuda")
        self.torch = torch

    def score(self, q: str, passages: Sequence[str]) -> float:
        best = -1e9
        for i in range(0, len(passages), 16):
            b = list(passages[i:i + 16])
            enc = self.t([q] * len(b), b, padding=True, truncation=True,
                         max_length=384, return_tensors="pt").to("cuda")
            with self.torch.no_grad():
                o = self.m(**enc)
            s, e = o.start_logits, o.end_logits
            # diem "co dap an" = max span, tru diem cua vi tri CLS (null)
            null = s[:, 0] + e[:, 0]
            span = s.max(dim=1).values + e.max(dim=1).values
            best = max(best, float((span - null).max()))
        return best


class SmallLM:
    """R3 — model nho CHAY LOCAL tra loi 'du hay chua'.  Doc logprob Yes/No
    tu chinh logits, nen co DU phan phoi, khong phai xap xi nhu qua API."""

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.t = AutoTokenizer.from_pretrained(RINSE_LM)
        self.m = AutoModelForCausalLM.from_pretrained(
            RINSE_LM, dtype=torch.float16, device_map="cuda").eval()
        self.torch = torch
        self.yes = [self.t.encode(x, add_special_tokens=False)[0]
                    for x in ("Yes", " Yes", "yes")]
        self.no = [self.t.encode(x, add_special_tokens=False)[0]
                   for x in ("No", " No", "no")]

    def score(self, q: str, y: str, passages: Sequence[str]) -> float:
        body = "".join("[%d] %s\n" % (i + 1, p) for i, p in enumerate(passages))
        p = ("%s\nQuestion: %s\nProposed answer: %s\n\nDo the passages contain "
             "enough information to establish the proposed answer? "
             "Reply Yes or No.\nAnswer:" % (body, q, y))
        s = self.t.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)
        enc = self.t(s, return_tensors="pt", truncation=True,
                     max_length=6000).to("cuda")
        with self.torch.no_grad():
            lg = self.m(**enc).logits[0, -1]
        pr = self.torch.softmax(lg.float(), dim=-1)
        a = float(sum(pr[i] for i in self.yes))
        b = float(sum(pr[i] for i in self.no))
        return a / (a + b) if (a + b) > 0 else 0.0


def zfix(vals: List[float], sizes: List[int]) -> List[float]:
    """Chuan hoa z, SAU khi tru xu huong tuyen tinh theo log(1+|S|).

    Khong co buoc nay thi tap to an diem mot cach tam thuong — dung chuyen
    lam hong phep so gold (|S|=2) voi gold_p1 (|S|=3)."""
    x = [math.log(1 + s) for s in sizes]
    n = len(vals)
    mx, mv = sum(x) / n, sum(vals) / n
    den = sum((a - mx) ** 2 for a in x) or 1.0
    k = sum((a - mx) * (b - mv) for a, b in zip(x, vals)) / den
    res = [v - k * (a - mx) for v, a in zip(vals, x)]
    m = sum(res) / n
    sd = (sum((r - m) ** 2 for r in res) / n) ** 0.5 or 1.0
    return [(r - m) / sd for r in res]


def main() -> int:
    arg = {k.lstrip("-"): v for k, v in
           (a.split("=", 1) for a in sys.argv[1:] if "=" in a)}
    src = pathlib.Path(arg.get("src", "artifacts/reports/cmp_scorers_musique.json"))
    rows = json.load(src.open(encoding="utf-8"))
    print("== %d tap con tu %s ==" % (len(rows), src.name))

    # can lai van ban doan: file cmp_scorers chi luu id, nen nap lai corpus
    from probe_covers import LOADERS
    data = LOADERS[arg.get("corpus", "musique")](int(arg.get("n", 500)))
    txts = {}
    for claims, ps, gold, meta in data:
        txts[meta["question"]] = ps
    rows = [r for r in rows if r["q"] in txts]
    print("   khop duoc %d tap con" % len(rows))

    r1, r2, r3 = Rerank(), AnswerSpan(), SmallLM()
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        ps = [txts[r["q"]][int(d[1:])] for d in r["S"]]
        r["R1"] = r1.score(r["q"], ps)
        r["R2"] = r2.score(r["q"], ps)
        r["R3"] = r3.score(r["q"], r["y"], ps)
        if i % 100 == 0 or i == len(rows):
            print("   %4d/%d  %.0fs" % (i, len(rows), time.time() - t0), flush=True)

    sizes = [len(r["S"]) for r in rows]
    for k in ("R1", "R2", "R3"):
        for r, z in zip(rows, zfix([r[k] for r in rows], sizes)):
            r[k + "z"] = z
    for r in rows:
        r["RINSE"] = r["R1z"] + r["R2z"] + r["R3z"]

    byq: Dict[str, Dict[str, Dict]] = collections.defaultdict(dict)
    for r in rows:
        byq[r["q"]][r["kind"]] = r

    def acc(f, pos, neg):
        ok = tot = 0
        for g in byq.values():
            if pos in g and neg in g:
                x, z = f(g[pos]), f(g[neg])
                if x == x and z == z:
                    tot += 1; ok += int(x > z)
        return 100 * ok / max(1, tot)

    SC = {"A sinh": lambda r: r["A"], "B hoi": lambda r: r["B"],
          "C nli": lambda r: r["C"], "BxC": lambda r: r["B"] * r["C"],
          "R1 phu": lambda r: r["R1"], "R2 span": lambda r: r["R2"],
          "R3 lm nho": lambda r: r["R3"], "RINSE-like": lambda r: r["RINSE"],
          "R3xC": lambda r: r["R3"] * r["C"],
          "RINSE+C": lambda r: r["RINSE"] + 2 * r["C"]}
    print("\n" + "=" * 66)
    print("%-12s %13s %13s %10s" % ("bo cham", "gold>gold_m1", "gold>rand", "tb"))
    for k, f in SC.items():
        xs = [acc(f, "gold", "gold_m1"), acc(f, "gold", "rand")]
        print("%-12s %12.1f%% %12.1f%% %9.1f%%" % (k, xs[0], xs[1], sum(xs) / 2))
    print("=" * 66)
    out = pathlib.Path("artifacts/reports/cmp_rinse.json")
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print("-> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
