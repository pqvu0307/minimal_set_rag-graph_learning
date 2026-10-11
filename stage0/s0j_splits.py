# -*- coding: utf-8 -*-
"""0.J — Chia tap va KHOA no lai.

Hai he qua cua viec dao nhan tren CHINH tap train cua G-reasoner:

  1  RO RI nhan vang vao nhanh GFM
     loss giam sat cua G-reasoner train tren "labeled relevant nodes",
     ma do chinh la supporting_documents — chinh chuan ngoai ta dinh dung o GD 1.
     =>  GD 1 PHAI chay tren split checkpoint CHUA train.
     =>  va bao cao chi so TACH THEO NHANH NGUON: chenh lech giua
         pool chi-BM25 va pool day du CHINH LA co cua ro ri.

  2  TRAIN-ON-TRAIN khi danh gia tang B
     =>  phai khoa held-out TRUOC khi GD 2 chay.

File khoa ghi ra reports/splits.lock.json va KHONG duoc ghi lai — moi lan chay
sau chi doc va kiem.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from s0_common import (ASK, MISSING, OK, active_datasets, banner,
                       load_config, resolve, verdict, write_report)

LOCK = "splits.lock.json"


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else list(data.values())


def _qid(row: Dict[str, Any], i: int) -> str:
    for k in ("id", "_id", "qid", "question_id"):
        if row.get(k):
            return str(row[k])
    q = str(row.get("question", ""))
    return "h" + hashlib.sha1(q.encode()).hexdigest()[:16] if q else "i%d" % i


def discover(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Tim cac split co san cho tung bo."""
    rdir = resolve(cfg, "raw_dir")
    found: Dict[str, Dict[str, Any]] = {}
    for ds in active_datasets(cfg):
        for base in (rdir / ds, rdir):
            hit: Dict[str, Any] = {}
            for name in ("train", "dev", "valid", "test"):
                p = base / ("%s.json" % name)
                if p.exists():
                    try:
                        rows = _load_rows(p)
                    except Exception as e:
                        hit[name] = {"error": repr(e)[:120]}; continue
                    hit[name] = {
                        "path": str(p), "n": len(rows),
                        "with_support": sum(
                            1 for r in rows if r.get("supporting_documents")),
                        "with_answer": sum(1 for r in rows if r.get("answer")),
                    }
            if hit:
                found[ds] = hit; break
    return found


def run(cfg: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
    banner("0.J  Chia tap — va khoa lai")

    lock_path = resolve(cfg, "reports") / LOCK
    rep: Dict[str, Any] = {"lock_path": str(lock_path)}

    if lock_path.exists() and not force:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        rep["lock"] = lock
        verdict(OK, "da co file khoa — KHONG ghi lai.  Chi doc va kiem.")
        print("     calib=%s  phase1=%d  phase2=%d  holdout=%d"
              % (lock["calib"]["split"], len(lock["phase1_ids"]),
                 len(lock["phase2_ids"]), len(lock["holdout_ids"])))
        write_report(cfg, "s0j_splits", rep)
        return rep

    found = discover(cfg)
    rep["discovered"] = found
    if not found:
        verdict(MISSING, "khong thay train/dev json nao trong %s  ->  chay 0.A truoc"
                % resolve(cfg, "raw_dir"))
        write_report(cfg, "s0j_splits", rep)
        return rep

    for ds, splits in found.items():
        print("\n     %s" % ds)
        for name, info in splits.items():
            if "n" in info:
                print("        %-6s n=%-8d support=%-8d answer=%d"
                      % (name, info["n"], info["with_support"], info["with_answer"]))

    # ── 1. chon split hieu chuan: PHAI khong phai train ───────────
    # test set la split checkpoint CHUA train, VA no co index dung san
    # -> khop hoan hao cho GD 1, $0 indexing.
    calib_ds, calib_split = None, None
    for ds, splits in found.items():
        for name in ("test", "dev", "valid"):
            if name in splits and splits[name].get("with_support"):
                calib_ds, calib_split = ds, name; break
        if calib_split:
            break

    if not calib_split:
        verdict(ASK,
                "KHONG tim duoc split ngoai train co supporting_documents.\n"
                "           Day la chan tren cua GD 1: hieu chuan tren train split\n"
                "           thi nhanh GFM da thay san nhan vang, va CA BA con so\n"
                "           (moi tham so · adq dang tin · doi chieu T) PHONG GIA.\n"
                "           Index cho test set DA DUOC PHAT HANH — tai ve roi dat\n"
                "           artifacts.mode = test.")
        rep["calib"] = None
    else:
        rep["calib"] = {"dataset": calib_ds, "split": calib_split,
                        "n_available": found[calib_ds][calib_split]["n"],
                        "n_to_use": cfg["splits"]["calib_n"]}
        verdict(OK, "hieu chuan GD 1:  %s / %s  (%d co san, dung %d)"
                % (calib_ds, calib_split,
                   found[calib_ds][calib_split]["n"], cfg["splits"]["calib_n"]))

    # ── 2. chia train thanh phase1 / phase2 / holdout ─────────────
    train_rows: List[Dict[str, Any]] = []
    for ds, splits in found.items():
        if "train" not in splits or "path" not in splits["train"]:
            continue
        for i, r in enumerate(_load_rows(Path(splits["train"]["path"]))):
            train_rows.append({"ds": ds, "qid": _qid(r, i)})

    rep["train_total"] = len(train_rows)
    rep["mode"] = cfg["artifacts"].get("mode", "test")
    if not train_rows:
        if rep["mode"] == "test":
            verdict(OK, "mode=test: khong co train split — DUNG nhu vay.\n"
                        "           GD 0.5 chay ca bon cong tren 3.000 truy van test:\n"
                        "              %d hieu chuan  ·  %d pha 1 thu nho  ·  %d cong p\n"
                        "           Chia tap cho pha 1/pha 2 that chi lam khi mode=train."
                    % (cfg["splits"]["calib_n"],
                       cfg["splits"]["dryrun"]["phase1_n"],
                       cfg["splits"]["dryrun"]["pgate_n"]))
        else:
            verdict(MISSING, "mode=train ma khong doc duoc train split")
        write_report(cfg, "s0j_splits", rep)
        return rep

    n1 = cfg["splits"]["phase1_n"]; n2 = cfg["splits"]["phase2_n"]
    hmin = cfg["splits"]["holdout_min"]
    need = n1 + n2 + hmin
    print("\n     train tong = %d" % len(train_rows))
    print("     can        = %d (pha1) + %d (pha2) + %d (holdout toi thieu) = %d"
          % (n1, n2, hmin, need))

    if len(train_rows) < need:
        # co lai theo ti le, nhung GIU NGUYEN holdout toi thieu
        avail = len(train_rows) - hmin
        if avail <= 0:
            verdict(ASK, "train chi %d, khong du cho ca holdout toi thieu %d.\n"
                         "           Phai giam holdout_min, hoac dung them bo khac."
                    % (len(train_rows), hmin))
            write_report(cfg, "s0j_splits", rep)
            return rep
        scale = avail / (n1 + n2)
        n1s, n2s = int(n1 * scale), int(n2 * scale)
        verdict(ASK, "train %d < can %d  ->  co lai theo ti le %.3f:\n"
                     "           pha1 %d -> %d   ·   pha2 %d -> %d   ·   holdout giu %d\n"
                     "           Ngan sach phai tinh lai theo con so moi."
                % (len(train_rows), need, scale, n1, n1s, n2, n2s, hmin))
        n1, n2 = n1s, n2s
    else:
        verdict(OK, "train du: con lai %d cho holdout (toi thieu %d)"
                % (len(train_rows) - n1 - n2, hmin))

    rnd = random.Random(cfg["splits"]["seed"])
    idx = list(range(len(train_rows)))
    rnd.shuffle(idx)
    p1 = [train_rows[i] for i in idx[:n1]]
    p2 = [train_rows[i] for i in idx[n1:n1 + n2]]
    ho = [train_rows[i] for i in idx[n1 + n2:]]

    lock = {
        "seed": cfg["splits"]["seed"],
        "calib": rep["calib"] or {"split": None},
        "phase1_ids": ["%s/%s" % (r["ds"], r["qid"]) for r in p1],
        "phase2_ids": ["%s/%s" % (r["ds"], r["qid"]) for r in p2],
        "holdout_ids": ["%s/%s" % (r["ds"], r["qid"]) for r in ho],
        "note": ("KHOA. Khong ghi lai file nay. holdout la tap DUY NHAT chua bi "
                 "dao nhan, nen moi con so cua tang B phai do tren no."),
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, ensure_ascii=False), encoding="utf-8")
    rep["lock_written"] = True
    rep["sizes"] = {"phase1": len(p1), "phase2": len(p2), "holdout": len(ho)}
    verdict(OK, "da khoa:  pha1=%d  pha2=%d  holdout=%d  ->  %s"
            % (len(p1), len(p2), len(ho), lock_path))

    # ── 3. nhac lai cach doc chi so o GD 1 ────────────────────────
    print("\n     GD 1 phai bao cao MOI con so HAI LAN:")
    print("        pool chi-BM25   doc lap voi nhan vang")
    print("        pool day du     nhanh GFM da thay nhan vang")
    print("        chenh lech giua hai lan CHINH LA co cua ro ri")

    p = write_report(cfg, "s0j_splits", rep)
    print("\n  -> %s" % p)
    return rep


if __name__ == "__main__":
    run(load_config())
