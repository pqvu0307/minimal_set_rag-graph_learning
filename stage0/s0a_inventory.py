# -*- coding: utf-8 -*-
"""0.A — Lay hien vat va kiem ke.

Tra loi BON cau hoi cua ke-hoach truoc khi viet dong code nao khac:
  1  do thi da dung san co duoc PHAT HANH khong, hay phai tu dung?
  2  paper noi co BA constructor  ->  dung cai nao?
  3  MOT do thi cho ca ba bo, hay MOT MOI BO?
  4  dem: truy van moi bo · bao nhieu co supporting_documents
          · bao nhieu co phan ra (MuSiQue)
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from s0_common import (ASK, MISSING, OK, active_datasets, banner,
                       load_config, openie_cost, resolve, verdict,
                       write_report)

CSV_FILES = ("nodes.csv", "relations.csv", "edges.csv")


def _sniff_csv(path: Path, limit: int = 200_000) -> Dict[str, Any]:
    """Doc header + dem theo cot `type` neu co, khong nap het vao RAM."""
    out: Dict[str, Any] = {"exists": path.exists(), "path": str(path)}
    if not path.exists():
        return out
    out["size_bytes"] = path.stat().st_size
    types = Counter()
    n = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        out["columns"] = rd.fieldnames or []
        tcol = next((c for c in (rd.fieldnames or []) if c.lower() in
                     ("type", "node_type", "label")), None)
        out["type_column"] = tcol
        for row in rd:
            n += 1
            if tcol:
                types[(row.get(tcol) or "").strip()] += 1
            if n >= limit:
                out["truncated_at"] = limit
                break
    out["rows_scanned"] = n
    if types:
        out["type_counts"] = dict(types.most_common())
    return out


def inventory_graph(cfg: Dict[str, Any]) -> Dict[str, Any]:
    gdir = resolve(cfg, "graph_dir")
    res: Dict[str, Any] = {"graph_dir": str(gdir), "per_dataset": {}}

    # hai bo cuc co the: graph_dir/<file>.csv  hoac  graph_dir/<dataset>/<file>.csv
    flat = all((gdir / f).exists() for f in CSV_FILES)
    res["layout"] = "flat" if flat else "per_dataset_or_missing"

    if flat:
        res["flat"] = {f: _sniff_csv(gdir / f) for f in CSV_FILES}
    for ds in active_datasets(cfg):
        d = gdir / ds
        if d.is_dir():
            res["per_dataset"][ds] = {f: _sniff_csv(d / f) for f in CSV_FILES}
    return res


def inventory_raw(cfg: Dict[str, Any]) -> Dict[str, Any]:
    rdir = resolve(cfg, "raw_dir")
    res: Dict[str, Any] = {"raw_dir": str(rdir), "datasets": {}}

    for ds in active_datasets(cfg):
        for base in (rdir / ds, rdir):
            docs = base / "documents.json"
            if not docs.exists():
                continue
            info: Dict[str, Any] = {"base": str(base)}
            # documents.json co the rat lon -> chi doc dau de xac nhan dang
            with docs.open("r", encoding="utf-8") as f:
                head = f.read(4096)
            info["documents_json_head"] = head[:400]
            info["documents_json_bytes"] = docs.stat().st_size
            info["looks_like_dict_of_title_to_text"] = head.lstrip().startswith("{")

            for split in ("train.json", "dev.json", "test.json", "valid.json"):
                p = base / split
                if not p.exists():
                    continue
                s: Dict[str, Any] = {"bytes": p.stat().st_size}
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    rows: List[Dict[str, Any]] = (
                        data if isinstance(data, list) else list(data.values()))
                    s["n"] = len(rows)
                    s["with_answer"] = sum(1 for r in rows if r.get("answer"))
                    s["with_supporting_documents"] = sum(
                        1 for r in rows if r.get("supporting_documents"))
                    s["with_decomposition"] = sum(
                        1 for r in rows
                        if r.get("question_decomposition") or r.get("decomposition"))
                    if rows:
                        s["example_keys"] = sorted(rows[0].keys())
                except Exception as e:                       # file lon / jsonl
                    s["parse_error"] = repr(e)[:160]
                info[split] = s
            res["datasets"][ds] = info
            break
    return res


def inventory_nodes_in_train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """start_nodes / target_nodes nam o processed/stage1/<ds>/train.json.

    Chung la DAN XUAT cua buoc dung do thi, khong phai thu dataset cho san:
        raw/train.json       chi co supporting_documents
        processed/.../train.json   + start_nodes + target_nodes (nhom theo LOAI)

    Hai thu phai dem:
      1  target_nodes gom nhung LOAI nao
         -> neu co `document` thi ro ri supporting_documents la TRUC TIEP;
            neu chi co `entity` thi gian tiep, nhe hon han
      2  bao nhieu truy van co start_nodes RONG
         -> entity linking truot  ->  GFM chet san voi truy van do
         -> va no se DOI LOT "pool thieu" o cua thoat C0, tron hai nguyen nhan
            ma §6.1 von tach rieng
    """
    gdir = resolve(cfg, "graph_dir")
    out: Dict[str, Any] = {}
    for ds in active_datasets(cfg):
        p = gdir / ds / "train.json"
        if not p.exists():
            p = gdir / ds / "test.json"
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            out[ds] = {"parse_error": repr(e)[:140]}; continue
        rows = data if isinstance(data, list) else list(data.values())
        st_types, tg_types = Counter(), Counter()
        empty_start = 0
        for r in rows:
            sn = r.get("start_nodes") or {}
            tn = r.get("target_nodes") or {}
            if isinstance(sn, dict):
                for t, v in sn.items():
                    st_types[t] += len(v or [])
                if not any(v for v in sn.values()):
                    empty_start += 1
            if isinstance(tn, dict):
                for t, v in tn.items():
                    tg_types[t] += len(v or [])
        out[ds] = {
            "file": str(p), "n": len(rows),
            "start_node_types": dict(st_types.most_common()),
            "target_node_types": dict(tg_types.most_common()),
            "queries_with_empty_start_nodes": empty_start,
            "empty_start_rate": round(empty_start / max(1, len(rows)), 4),
        }
    return out


def run(cfg: Dict[str, Any]) -> Dict[str, Any]:
    banner("0.A  Lay hien vat va kiem ke")

    repo = resolve(cfg, "gfmrag_repo")
    rep = {
        "checkpoint": cfg["artifacts"]["checkpoint"],
        "run_sft": cfg["artifacts"]["run_sft"],
        "repo_present": repo.exists(),
        "repo_path": str(repo),
        "mode": cfg["artifacts"].get("mode", "test"),
        "active_datasets": active_datasets(cfg),
        "graph": inventory_graph(cfg),
        "raw": inventory_raw(cfg),
        "nodes_in_train": inventory_nodes_in_train(cfg),
    }
    print("  mode = %s   dataset = %s" % (rep["mode"], rep["active_datasets"]))

    # ── cau hoi 1: do thi co san chua ──────────────────────────────
    g = rep["graph"]
    have_flat = g["layout"] == "flat" and all(
        g.get("flat", {}).get(f, {}).get("exists") for f in CSV_FILES)
    have_per_ds = bool(g["per_dataset"]) and all(
        all(g["per_dataset"][ds][f]["exists"] for f in CSV_FILES)
        for ds in g["per_dataset"])
    rep["q1_graph_available"] = bool(have_flat or have_per_ds)
    mode = rep["mode"]
    if rep["q1_graph_available"]:
        verdict(OK, "do thi co san cho mode=%s  ->  KHONG phai tu dung" % mode)
    elif mode == "test":
        verdict(MISSING,
                "mode=test ma chua thay nodes/relations/edges.csv.\n"
                "           Index cho test set DUOC PHAT HANH — tai ve, dung dung:\n"
                "           %s\n"
                "           (ban phat hanh gom testing split VA full training data raw,\n"
                "            nhung index dung san CHI co cho test set)"
                % cfg["artifacts"]["onedrive_url"])
    else:
        verdict(ASK,
                "mode=train: index KHONG duoc phat hanh.  Doc gfm-rag ghi ro\n"
                "           train splits 'must be indexed locally using Stage 1 scripts'.\n"
                "           Day la DONG CHI PHI LON NHAT cua chuong trinh — xem uoc tien duoi.")

    # ── cau hoi 2: constructor nao ─────────────────────────────────
    cfgs = sorted(p.name for p in (repo / "gfmrag" / "workflow" / "config").rglob("*.yaml")) \
        if repo.exists() else []
    rep["q2_constructor_configs_found"] = cfgs[:40]
    if cfgs:
        verdict(ASK, "thay %d file config — PHAI chot MOT constructor, "
                     "ba cach se cho ba cach cat khac nhau" % len(cfgs))
    else:
        verdict(MISSING, "chua clone repo:  git clone https://github.com/RManLuo/gfm-rag")

    # ── cau hoi 3: mot do thi hay mot moi bo ───────────────────────
    if have_per_ds:
        rep["q3_scope"] = "per_dataset"
        verdict(OK, "MOT do thi MOI BO  ->  truy hoi trong dung corpus cua bo do")
    elif have_flat:
        rep["q3_scope"] = "single_shared"
        verdict(ASK,
                "chi thay MOT do thi phang.  Neu no tron ca ba bo thi SAI:\n"
                "           truy van HotpotQA se lôi ve doan tu MuSiQue.\n"
                "           Kiem nodes.csv xem ten node co tien to theo bo khong.")
    else:
        rep["q3_scope"] = "unknown"

    # ── cau hoi 4: dem ────────────────────────────────────────────
    counts = {}
    for ds, info in rep["raw"]["datasets"].items():
        for split in ("train.json", "dev.json", "valid.json", "test.json"):
            if split in info and "n" in info[split]:
                counts[f"{ds}/{split}"] = {
                    "n": info[split]["n"],
                    "with_answer": info[split].get("with_answer"),
                    "with_supporting_documents":
                        info[split].get("with_supporting_documents"),
                    "with_decomposition": info[split].get("with_decomposition"),
                }
    rep["q4_counts"] = counts
    if counts:
        verdict(OK, "dem duoc %d split" % len(counts))
        for k, v in counts.items():
            print("           %-24s n=%-8s answer=%-8s support=%-8s decomp=%s"
                  % (k, v["n"], v["with_answer"],
                     v["with_supporting_documents"], v["with_decomposition"]))
    else:
        verdict(MISSING, "chua thay train.json / documents.json trong %s"
                % resolve(cfg, "raw_dir"))

    # ── kiem chung con so 10,7 cua ke-hoach ────────────────────────
    tot_q = sum(v["n"] for k, v in counts.items() if k.endswith("train.json"))
    rep["q4_total_train_queries"] = tot_q
    if tot_q:
        print("\n           tong truy van train = %d  (ke-hoach ghi 277.839)" % tot_q)
        print("           ti le document/truy van = 10,7 la chu ky distractor setting;")
        print("           0.B se xac nhan bang do dai doan that.")

    # ── start_nodes / target_nodes ────────────────────────────────
    nt = rep["nodes_in_train"]
    if nt:
        print("\n  start_nodes / target_nodes  (DAN XUAT tu buoc dung do thi):")
        for ds, info in nt.items():
            if "n" not in info:
                continue
            print("     %-22s n=%-7d start=%s  target=%s"
                  % (ds, info["n"], info["start_node_types"],
                     info["target_node_types"]))
            print("     %-22s start_nodes RONG: %d (%.2f%%)"
                  % ("", info["queries_with_empty_start_nodes"],
                     100 * info["empty_start_rate"]))
        # ro ri truc tiep hay gian tiep?
        has_doc_target = any("document" in (i.get("target_node_types") or {})
                             for i in nt.values() if "n" in i)
        rep["leak_is_direct"] = has_doc_target
        if has_doc_target:
            verdict(ASK, "target_nodes CO loai `document`  ->  ro ri "
                         "supporting_documents la TRUC TIEP.\n"
                         "           GD 1 PHAI chay tren test set, va bao cao moi con so\n"
                         "           HAI LAN: pool chi-BM25 vs pool day du.")
        else:
            verdict(OK, "target_nodes khong co loai `document`  ->  ro ri GIAN TIEP, "
                        "nhe hon han canh bao o 0.J")
        worst = max((i["empty_start_rate"] for i in nt.values() if "n" in i),
                    default=0.0)
        if worst > 0.02:
            verdict(ASK, "toi %.1f%% truy van co start_nodes RONG  ->  entity linking "
                         "truot.\n"
                         "           Ca nay se DOI LOT 'pool thieu' o cua thoat C0.\n"
                         "           Co `pool thieu` PHAI tach lam hai, khong thi\n"
                         "           o «loi do thi» §6.1 tron hai nguyen nhan."
                    % (100 * worst))
        else:
            verdict(OK, "start_nodes rong chi %.2f%%  ->  entity linking khong phai "
                        "nguon nhieu dang ke" % (100 * worst))

    # ── uoc tien OpenIE cho train split ──────────────────────────
    nq = rep.get("q4_total_train_queries") or 0
    sp = cfg["splits"]
    est = {}
    for name, nquery in (("pha 1", sp["phase1_n"]),
                         ("pha 1 + pha 2", sp["phase1_n"] + sp["phase2_n"])):
        ndocs = int(nquery * cfg["openie"]["docs_per_query"])
        est[name] = openie_cost(cfg, ndocs)
    if nq:
        est["toan bo train da thay"] = openie_cost(
            cfg, int(nq * cfg["openie"]["docs_per_query"]))
    rep["openie_estimate"] = est
    print("\n  Uoc tien OpenIE neu phai tu dung index (mode=train):")
    for k, v in est.items():
        print("     %-24s %9d doc  ->  $%s" % (k, v["n_docs"], f"{v['usd']:,.0f}"))
    if cfg["openie"]["model"] == "PHAI_DIEN":
        verdict(ASK, "config.openie.model con la PHAI_DIEN.  Model RE bot 5-10 lan\n"
                     "           nhung triple xau hon -> nhanh GFM yeu them, ma ta von\n"
                     "           da co y dung checkpoint chua fine-tune.")

    p = write_report(cfg, "s0a_inventory", rep)
    print("\n  -> %s" % p)
    return rep


if __name__ == "__main__":
    run(load_config())
