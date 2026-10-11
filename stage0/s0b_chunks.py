# -*- coding: utf-8 -*-
"""0.B — DO cach cat chunk, khong quy dinh no.

Don vi chunk da bi cong cu quyet dinh: moi entry cua documents.json la MOT
document node, framework khong cat ben trong.  Va `T` phai la tap NODE vi tang B
dinh tuyen toi node — nen khong co lua chon cat lai.

Viec cua buoc nay la do hai thu ma moi bang chi phi dua vao:
    1  phan bo do dai, TACH RIENG document va summary
       -> cost(S) = overhead + tong len(k),  khong phai 250*|S|
    2  cac doan co CHONG LAP khong
       -> neu co thi phai khu trung lap TRUOC khi lap pool, vi chunk chong nhau
          tao du thua nhan tao, dung thu §3.6 sinh ra de phat hien
"""
from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from s0_common import (ASK, MISSING, OK, TokenCounter, banner, cost_tokens,
                       describe, load_config, resolve, verdict, write_report)

WORD = re.compile(r"\w+", re.UNICODE)


def iter_nodes(graph_dir: Path) -> Iterator[Dict[str, str]]:
    """Duyet nodes.csv, ho tro ca bo cuc phang va theo tung bo."""
    cands = [graph_dir / "nodes.csv"]
    cands += sorted(graph_dir.glob("*/nodes.csv"))
    for p in cands:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row["_src"] = p.parent.name
                yield row


def node_text(row: Dict[str, str]) -> str:
    """nodes.csv khong co schema co dinh -> thu cac ten cot hay gap."""
    for c in ("text", "content", "body", "passage", "summary", "title_text"):
        if row.get(c):
            return row[c]
    # mot so ban dat noi dung trong `attributes` dang JSON
    a = row.get("attributes") or row.get("attribute") or ""
    if a.strip().startswith("{"):
        try:
            d = json.loads(a)
            for c in ("text", "content", "body", "summary"):
                if d.get(c):
                    return str(d[c])
        except Exception:
            pass
    return row.get("name") or row.get("id") or ""


def node_type(row: Dict[str, str]) -> str:
    for c in ("type", "node_type", "label"):
        if row.get(c):
            return row[c].strip().lower()
    return "?"


def shingles(text: str, k: int) -> set:
    w = WORD.findall(text.lower())
    if len(w) < k:
        return {" ".join(w)} if w else set()
    return {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)}


def overlap_scan(texts: List[str], k: int, thr: float,
                 sample: int = 1500, seed: int = 0) -> Dict[str, Any]:
    """Lay mau roi so tung cap — 3M node thi khong so het duoc.

    Dung chi so nguoc tren shingle de chi so cac cap CO kha nang trung,
    thay vi O(n^2) tren mau.
    """
    rnd = random.Random(seed)
    if len(texts) > sample:
        texts = rnd.sample(texts, sample)
    sh = [shingles(t, k) for t in texts]
    inv: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(sh):
        for g in s:
            inv[g].append(i)
    pairs: Dict[Tuple[int, int], int] = defaultdict(int)
    for g, idxs in inv.items():
        if len(idxs) < 2 or len(idxs) > 40:       # bo shingle qua pho bien
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                pairs[(idxs[a], idxs[b])] += 1
    hits = []
    for (i, j), _ in pairs.items():
        u = len(sh[i] | sh[j])
        if not u:
            continue
        jac = len(sh[i] & sh[j]) / u
        if jac >= thr:
            hits.append(round(jac, 3))
    return {
        "sampled": len(texts),
        "shingle_k": k,
        "threshold": thr,
        "pairs_examined": len(pairs),
        "overlapping_pairs": len(hits),
        "overlap_rate": round(len(hits) / max(1, len(texts)), 4),
        "jaccard_examples": sorted(hits, reverse=True)[:10],
    }


def run(cfg: Dict[str, Any]) -> Dict[str, Any]:
    banner("0.B  Do cach cat chunk  (khong quy dinh no)")

    gdir = resolve(cfg, "graph_dir")
    tc = TokenCounter()
    want = set(cfg["chunks"]["node_types_in_pool"])

    lens: Dict[str, List[int]] = defaultdict(list)
    texts: Dict[str, List[str]] = defaultdict(list)
    n_rows = 0
    for row in iter_nodes(gdir):
        n_rows += 1
        t = node_type(row)
        if t not in want and t != "entity":
            t = t or "?"
        if t == "entity":
            continue
        txt = node_text(row)
        if not txt:
            continue
        lens[t].append(tc.count(txt))
        if len(texts[t]) < 4000:
            texts[t].append(txt)

    rep: Dict[str, Any] = {
        "graph_dir": str(gdir),
        "rows_seen": n_rows,
        "token_counter_exact": tc.exact,
        "length_by_type": {t: describe(v) for t, v in lens.items()},
    }

    if not n_rows:
        verdict(MISSING, "khong doc duoc nodes.csv  ->  chay 0.A truoc")
        write_report(cfg, "s0b_chunks", rep)
        return rep

    if not tc.exact:
        verdict(ASK, "khong co tiktoken — do dai la XAP XI 1,3 token/tu.\n"
                     "           Du de doc phan bo, nhung bang chi phi cuoi phai\n"
                     "           dem bang tokenizer cua dung model judge.\n"
                     "           pip install tiktoken")

    # ── 1. phan bo do dai ─────────────────────────────────────────
    assumed = cfg["chunks"]["assumed_tokens_per_doc"]
    for t, d in rep["length_by_type"].items():
        print("\n  type = %-9s n=%-9d p50=%-6.0f mean=%-7.1f p90=%-6.0f p99=%-7.0f max=%.0f"
              % (t, d["n"], d["p50"], d["mean"], d["p90"], d["p99"], d["max"]))

    doc = rep["length_by_type"].get("document")
    if doc:
        if doc["mean"] <= assumed * 0.8:
            verdict(OK, "doan ngan hon gia dinh %d token (mean %.0f)  ->  bang chi phi "
                        "hien tai la BAO THU" % (assumed, doc["mean"]))
        elif doc["mean"] >= assumed * 1.25:
            verdict(ASK, "doan DAI hon gia dinh %d token (mean %.0f)  ->  moi bang chi phi "
                         "trong ca ba tai lieu dang THAP hon thuc te" % (assumed, doc["mean"]))
        else:
            verdict(OK, "mean %.0f ~ gia dinh %d  ->  bang chi phi dung duoc"
                    % (doc["mean"], assumed))
        if doc["p99"] > 4 * max(1.0, doc["p50"]):
            verdict(ASK, "co DUOI DAI: p99=%.0f gap %.1f lan p50=%.0f\n"
                         "           mot node dai chiem cho cua nhieu node khac trong prompt\n"
                         "           -> can dat tran, hoac xu rieng nhom node dai"
                    % (doc["p99"], doc["p99"] / max(1.0, doc["p50"]), doc["p50"]))

    smy = rep["length_by_type"].get("summary")
    if smy and doc:
        ratio = smy["mean"] / max(1.0, doc["mean"])
        rep["summary_over_document_length_ratio"] = round(ratio, 2)
        verdict(ASK if ratio >= 2 else OK,
                "summary dai gap %.1f lan document (mean %.0f vs %.0f)\n"
                "           -> cost(S) PHAI la overhead + tong len(k), khong phai 250*|S|"
                % (ratio, smy["mean"], doc["mean"]))

    # ── 2. chong lap ──────────────────────────────────────────────
    ov = {}
    for t in ("document", "summary"):
        if texts.get(t):
            ov[t] = overlap_scan(texts[t],
                                 cfg["chunks"]["overlap_shingle"],
                                 cfg["chunks"]["overlap_threshold"])
    rep["overlap"] = ov
    for t, o in ov.items():
        if o["overlapping_pairs"] == 0:
            verdict(OK, "%s: khong thay chong lap tren mau %d"
                    % (t, o["sampled"]))
        else:
            verdict(ASK, "%s: %d cap chong lap tren mau %d (ty le %.3f)\n"
                         "           -> PHAI khu trung lap TRUOC khi lap pool,\n"
                         "              khong thi du thua nhan tao lam ban §3.6"
                    % (t, o["overlapping_pairs"], o["sampled"], o["overlap_rate"]))

    # ── 3. bang chi phi tinh LAI theo do dai that ─────────────────
    oh = cfg["judge"]["prompt_overhead_tokens"]
    if doc:
        m = doc["mean"]
        rep["recomputed_cost"] = {
            "overhead": oh,
            "mean_doc_tokens": m,
            "v_empty": cost_tokens([], oh),
            "v_singleton": cost_tokens([m], oh),
            "v_R_12": cost_tokens([m] * 12, oh),
            "v_C_40": cost_tokens([m] * 40, oh),
            "note": "so nay thay cho 500/750/3.500/10.500 trong tai lieu",
        }
        print("\n  chi phi tinh LAI theo do dai that (mean %.0f token/doan):" % m)
        for k in ("v_empty", "v_singleton", "v_R_12", "v_C_40"):
            print("     %-14s %8d token" % (k, rep["recomputed_cost"][k]))
        print("     (tai lieu dang ghi:   500 /    750 /  3.500 / 10.500)")

    p = write_report(cfg, "s0b_chunks", rep)
    print("\n  -> %s" % p)
    return rep


if __name__ == "__main__":
    run(load_config())
