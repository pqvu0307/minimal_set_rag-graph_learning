import os, sys, json
os.chdir("/content/ragproj"); sys.path.insert(0, "stage0")
from s0_common import load_config
from s0e_judge import Judge, GEN_TMPL, JUDGE_TMPL
from probe_covers import LOADERS
cfg = load_config(); j = Judge(cfg)
cl, ps, gold, meta = LOADERS["musique"](1)[0]
q, y = meta["question"], str(meta["answer"])
ids = ["p%d" % k for k in range(len(ps))]; txt = {i: p for i, p in zip(ids, ps)}
res = j.v(q, y, ids, txt, aliases=meta.get("answer_aliases") or [])
print("cau hoi:", q)
print("dap an :", y, "| aliases:", meta.get("answer_aliases"))
print("v(C)   :", round(res.v, 4))
print("truong JudgeResult:", [f for f in dir(res) if not f.startswith("_")])
for f in ("answer", "ans", "gen", "text", "a"):
    if hasattr(res, f):
        print("  %s = %r" % (f, str(getattr(res, f))[:400]))
