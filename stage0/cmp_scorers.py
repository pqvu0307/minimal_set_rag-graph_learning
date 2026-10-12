# -*- coding: utf-8 -*-
"""So ba cach cham mot tap con S, tren cung mot tap con.

    A  v(S) hien tai    sinh (Qwen local) + cham (DeepSeek)   2 luot
    B  hoi thang        "S co du de ket luan y?"              1 luot
    C  NLI gop          min_j max_{k in S} entail(k, j)       0 luot

Cau hoi: cai nao tach duoc TAP CHUA BANG CHUNG khoi tap khong chua, TRONG
CUNG MOT TRUY VAN?  Do la phep do dung — so diem giua cac truy van khac nhau
thi vo nghia, vi moi cau mot do kho.

Moi truy van bon tap con, nhan biet truoc tu doan vang G:
    gold     = G                    du
    gold_m1  = G bo mot doan        THIEU mot chang
    rand     = |G| doan khong vang  thieu
    gold_p1  = G them mot doan      du, nhung KHONG toi tieu

Diem phai: gold > gold_m1, gold > rand, va gold ~ gold_p1.

Sinh chay THEO LO tren GPU — 400 lan sinh tuan tu mat ~23 phut, theo lo
32 thi con ~2 phut.

    python stage0/cmp_scorers.py --n=100
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from s0_common import load_config                       # noqa: E402
from s0e_judge import GEN_TMPL, JUDGE_TMPL, _aliases_line  # noqa: E402
from s0g_nli import NLI, coverage_table                 # noqa: E402
from probe_covers import LOADERS, LARGE                 # noqa: E402

SUFF_TMPL = (
    "{passages}\nQuestion: {q}\nProposed answer: {y}{aliases}\n\n"
    "Do the passages above contain enough information to establish that the "
    "proposed answer is correct? Judge ONLY from the passages; do not use "
    "outside knowledge.\nReply with exactly one word: Yes or No.\nAnswer:")


def block(ids: Sequence[str], txt: Dict[str, str]) -> str:
    return "".join("[%d] %s\n" % (i + 1, txt[d]) for i, d in enumerate(ids))


class Local:
    """Sinh theo LO tren GPU, co chat template va TAT thinking."""

    def __init__(self, cfg):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        mid = cfg["gen"]["model"]
        self.tok = AutoTokenizer.from_pretrained(mid)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.mdl = AutoModelForCausalLM.from_pretrained(
            mid, dtype=torch.float16, device_map="cuda").eval()
        self.torch = torch
        self.max_new = cfg["gen"]["max_tokens"]

    def _wrap(self, p: str) -> str:
        for kw in ({"enable_thinking": False}, {}):
            try:
                return self.tok.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False,
                    add_generation_prompt=True, **kw)
            except TypeError:
                continue
        return p

    def gen(self, prompts: List[str], bs: int = 16) -> List[str]:
        out: List[str] = []
        for i in range(0, len(prompts), bs):
            enc = self.tok([self._wrap(p) for p in prompts[i:i + bs]],
                           return_tensors="pt", padding=True,
                           truncation=True, max_length=7000).to("cuda")
            with self.torch.no_grad():
                o = self.mdl.generate(**enc, max_new_tokens=self.max_new,
                                      do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
            for j in range(o.shape[0]):
                t = self.tok.decode(o[j][enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
                out.append(t.rsplit("</think>", 1)[-1].strip())
            print("      sinh %d/%d" % (len(out), len(prompts)), flush=True)
        return out


def p_yes(cli, model: str, prompt: str) -> float:
    """Mot luot, doc logprob token dau.  DeepInfra khong tra top_logprobs nen
    phai don khoi luong sang phia doi lap — xap xi, da ghi ro o s0e_judge."""
    import math
    for attempt in range(4):
        try:
            r = cli.chat.completions.create(
                model=model, max_tokens=1, temperature=0.0, logprobs=True,
                messages=[{"role": "user", "content": prompt}])
            c = r.choices[0].logprobs.content[0]
            p = min(max(math.exp(c.logprob), 1e-6), 1 - 1e-6)
            return p if (c.token or "").strip() in ("Yes", "yes", "YES") else 1 - p
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return float("nan")


def main() -> int:
    arg = {k.lstrip("-"): v for k, v in
           (a.split("=", 1) for a in sys.argv[1:] if "=" in a)}
    n = int(arg.get("n", 100))
    cfg = load_config()
    cfg["nli"]["max_length"] = 256
    cfg["nli"]["model"] = LARGE
    rng = random.Random(0)

    from openai import OpenAI
    cli = OpenAI(api_key=os.environ["DEEPINFRA_API_KEY"],
                 base_url=cfg["judge"]["base_url"])
    jm = cfg["judge"]["judge_model"]
    nli = NLI(cfg)

    data = LOADERS[arg.get("corpus", "musique")](n)
    jobs: List[Dict[str, Any]] = []
    for claims, ps, gold, meta in data:
        if len(gold) < 2 or len(ps) < len(gold) + 2:
            continue
        ids = ["p%d" % k for k in range(len(ps))]
        txt = dict(zip(ids, ps))
        G = ["p%d" % g for g in sorted(gold)]
        other = [i for i in ids if i not in G]
        tb = coverage_table(nli, ids, ps, claims, cfg)
        cand = {
            "gold": G,
            "gold_m1": G[:-1],
            "rand": rng.sample(other, len(G)),
            "gold_p1": G + [rng.choice(other)],
        }
        for name, S in cand.items():
            jobs.append({"qi": len(jobs), "q": meta["question"],
                         "y": str(meta["answer"]),
                         "al": meta.get("answer_aliases") or [],
                         "kind": name, "S": S, "txt": txt, "tb": tb,
                         "key": meta["question"]})
    print("== %d truy van -> %d tap con can cham ==" % (len(data), len(jobs)))

    # ── C: NLI gop — mien phi, lam truoc ────────────────────────────
    for j in jobs:
        tb, S = j["tb"], set(j["S"])
        per = [max((tb["grid"][k][c] for k in S), default=0.0)
               for c in range(tb["J"])]
        j["C"] = min(per) if per else 0.0

    # ── A: sinh theo lo, roi cham ───────────────────────────────────
    loc = Local(cfg)
    t0 = time.time()
    gens = loc.gen([GEN_TMPL.format(passages=block(j["S"], j["txt"]), q=j["q"])
                    for j in jobs], bs=int(arg.get("bs", 16)))
    print("   sinh xong %.0fs" % (time.time() - t0))
    for j, a in zip(jobs, gens):
        j["ans"] = a

    def score_A(j):
        al = _aliases_line(j["al"], True)
        return p_yes(cli, jm, JUDGE_TMPL.format(q=j["q"], y=j["y"], aliases=al,
                                                a=j["ans"] or "(empty)"))

    def score_B(j):
        al = _aliases_line(j["al"], True)
        return p_yes(cli, jm, SUFF_TMPL.format(
            passages=block(j["S"], j["txt"]), q=j["q"], y=j["y"], aliases=al))

    for tag, fn in (("A", score_A), ("B", score_B)):
        t1 = time.time()
        with ThreadPoolExecutor(max_workers=int(arg.get("workers", 48))) as ex:
            for j, v in zip(jobs, ex.map(fn, jobs)):
                j[tag] = v
        print("   %s xong %.0fs" % (tag, time.time() - t1))

    # ── bao cao: xep hang TRONG CUNG truy van ───────────────────────
    byq: Dict[str, Dict[str, Dict]] = collections.defaultdict(dict)
    for j in jobs:
        byq[j["key"]][j["kind"]] = j
    print("\n" + "=" * 64)
    SC = {"A": lambda r: r["A"], "B": lambda r: r["B"], "C": lambda r: r["C"],
          "BxC": lambda r: r["B"] * r["C"],
          "(B+C)/2": lambda r: (r["B"] + r["C"]) / 2,
          "min(B,C)": lambda r: min(r["B"], r["C"])}
    pairs = [("gold", "gold_m1"), ("gold", "rand")]

    def acc(f, pos, neg):
        ok = tot = 0
        for g in byq.values():
            if pos in g and neg in g:
                x, z = f(g[pos]), f(g[neg])
                if x == x and z == z:
                    tot += 1; ok += int(x > z)
        return 100 * ok / max(1, tot)

    print("%-10s %13s %13s %10s"
          % ("bo cham", "gold>gold_m1", "gold>rand", "trung binh"))
    for k, f in SC.items():
        xs = [acc(f, pos, neg) for pos, neg in pairs]
        print("%-10s %12.1f%% %12.1f%% %9.1f%%" % (k, xs[0], xs[1], sum(xs) / 2))
    print()
    for tag in ("A", "B", "C"):
        m = {k: [g[k][tag] for g in byq.values() if k in g] for k in
             ("gold", "gold_m1", "rand", "gold_p1")}
        print("   %s  trung binh: %s" % (tag, {k: round(sum(v) / max(1, len(v)), 3)
                                               for k, v in m.items()}))
    print("=" * 64)
    out = pathlib.Path("artifacts/reports/cmp_scorers_%s.json"
                       % arg.get("corpus", "musique"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([{k: j[k] for k in
                                ("q", "y", "kind", "S", "ans", "A", "B", "C")}
                               for j in jobs], indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print("-> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
