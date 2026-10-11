# -*- coding: utf-8 -*-
"""Chay toan bo giai doan 0 theo thu tu, roi tong hop thanh MOT bang.

Khong buoc nao crash vi thieu hien vat — no bao THIEU va noi phai lam gi.
Cuoi cung in ra danh sach nhung gi CAN NGUOI quyet dinh.

    python run_stage0.py                 # chay het
    python run_stage0.py 0.A 0.B         # chay vai buoc
    python run_stage0.py --config x.yaml
"""
from __future__ import annotations

import sys
import traceback
from typing import Any, Callable, Dict, List, Tuple

from s0_common import OK, ASK, MISSING, banner, load_config, write_report

import s0a_inventory
import s0b_chunks
import s0c_pool
import s0e_judge
import s0f_decompose
import s0g_nli
import s0j_splits

STEPS: List[Tuple[str, str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = [
    ("0.A", "Lay hien vat va kiem ke",            s0a_inventory.run),
    ("0.B", "Do cach cat chunk",                  s0b_chunks.run),
    ("0.C", "Pool da nguon + rerank theo nguon",  s0c_pool.run),
    ("0.E", "Khung judge + do prompt caching",    s0e_judge.run),
    ("0.F", "Che cau hoi thanh manh",             s0f_decompose.run),
    ("0.G", "Bo NLI va bang phu",                 s0g_nli.run),
    ("0.J", "Chia tap va khoa lai",               s0j_splits.run),
]

# Nhung gi khong tu quyet duoc — in lai o cuoi
NEEDS_HUMAN = [
    ("#2  model judge",
     "config.judge.gen_model / judge_model dang la PHAI_DIEN. "
     "Quyet dinh nay dinh ra cost(S) va toan bo bang chi phi."),
    ("#3  model NLI",
     "config.nli.model dang mac dinh DeBERTa-v3-base-mnli-fever-anli. "
     "Giu hay doi? 0.G in ra do chinh xac tren vi du co doan bay."),
    ("#5  ba trich dan",
     "Deegan-Packel 1978 · Banzhaf 1965 · Holler-Packel 1983 — chua xac thuc."),
    ("#6  model OpenIE",
     "config.openie.model — CHI can khi mode=train. Model RE bot 5-10 lan nhung "
     "triple xau hon -> nhanh GFM yeu them, ma ta von da co y dung checkpoint "
     "chua fine-tune. 0.A in uoc tien."),
    ("#7  index may bo",
     "Ca ba bo (HotpotQA + MuSiQue + 2Wiki) hay chi mot? Chi HotpotQA thi re ~3 lan "
     "nhung mat ca 3-4 hop cua MuSiQue — dung ca du an nham."),
    ("0.A-1  INDEXING",
     "Index dung san CHI co cho TEST set (tai OneDrive). Train set phai tu dung: "
     "~$820 cho phan pha 1, ~$9.800 cho ca pha 1+2. Day la DONG CHI PHI LON NHAT, "
     "va no la quyet dinh SAU cong, khong phai chi phi dau vao."),
    ("0.A-2  constructor",
     "Paper noi co BA constructor. Phai chot MOT, khong thi ba cach cat khac nhau."),
    ("0.A-3  pham vi do thi",
     "MOT do thi cho ca ba bo, hay MOT MOI BO? Tron ba bo la sai."),
    ("§3.8  parsimony",
     "summary o lai trong pool -> mot summary co the phu nhieu manh, va "
     "parsimony -|S| cho no THANG tap document tuong duong. Dem SO NGUON BI BAO, "
     "hay chap nhan T chi gom summary va gan co? 0.G in ra co gap ca nay khong."),
]


def main(argv: List[str]) -> int:
    cfg_path = None
    want: List[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--config":
            cfg_path = argv[i + 1]; i += 2; continue
        want.append(argv[i].upper()); i += 1

    cfg = load_config(cfg_path)
    results: Dict[str, Any] = {}
    status: Dict[str, str] = {}

    for code, title, fn in STEPS:
        if want and code not in want:
            continue
        try:
            results[code] = fn(cfg)
            status[code] = "chay xong"
        except Exception as e:
            status[code] = "LOI: %s" % repr(e)[:160]
            banner("%s  %s" % (code, title))
            print("  [LOI] %s" % repr(e)[:200])
            traceback.print_exc(limit=3)

    # ── tong hop ──────────────────────────────────────────────────
    banner("TONG HOP GIAI DOAN 0")
    for code, title, _ in STEPS:
        if code in status:
            print("  %-5s %-34s %s" % (code, title, status[code]))

    # cac con so ma GD 0 PHAI tra ve cho ke-hoach
    numbers: Dict[str, Any] = {}
    b = results.get("0.B") or {}
    if b.get("length_by_type", {}).get("document"):
        d = b["length_by_type"]["document"]
        numbers["token_moi_doan_p50"] = d["p50"]
        numbers["token_moi_doan_mean"] = d["mean"]
        numbers["gia_dinh_cu"] = cfg["chunks"]["assumed_tokens_per_doc"]
    if b.get("recomputed_cost"):
        numbers["chi_phi_tinh_lai"] = b["recomputed_cost"]
    if b.get("overlap"):
        numbers["chong_lap"] = {k: v["overlapping_pairs"]
                                for k, v in b["overlap"].items()}
    e = results.get("0.E") or {}
    if "cached_prompt_tokens_total" in e:
        numbers["prompt_caching_an"] = e["cached_prompt_tokens_total"] > 0
        numbers["real_over_est"] = e.get("real_over_est")
    f = results.get("0.F") or {}
    if "J_match_rate" in f:
        numbers["che_cau_hoi_khop"] = f["J_match_rate"]
        numbers["J_phan_bo"] = f.get("J_distribution_annotated")
    j = results.get("0.J") or {}
    if j.get("sizes"):
        numbers["chia_tap"] = j["sizes"]
    if (j.get("calib") or {}).get("split"):
        numbers["split_hieu_chuan"] = "%s/%s" % (j["calib"]["dataset"],
                                                 j["calib"]["split"])

    if numbers:
        print("\n  Con so GD 0 tra ve cho ke-hoach:")
        for k, v in numbers.items():
            print("     %-26s %s" % (k, v))

    print("\n  CAN NGUOI QUYET DINH:")
    for tag, msg in NEEDS_HUMAN:
        print("     %-22s %s" % (tag, msg))

    write_report(cfg, "stage0_summary",
                 {"status": status, "numbers": numbers,
                  "needs_human": [{"tag": t, "note": m} for t, m in NEEDS_HUMAN]})
    print("\n  -> reports/stage0_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
