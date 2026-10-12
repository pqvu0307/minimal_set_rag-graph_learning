# -*- coding: utf-8 -*-
"""0.G — Bo NLI va BANG PHU.

adq(S) = phu het   <=>   moi manh co it nhat mot doan trong S entail no.

Bang phu tinh xong voi 0 LUOT LLM, va no cho ba thu:
  1  CUA THOAT    manh nao khong doan nao chong do  ->  pool khong tra loi duoc
  2  UNG VIEN     bai toan phu tap nho, giai CHINH XAC
  3  PRIOR |T|    so manh noi |T| nen bang bao nhieu

Cua thoat phai dat THAN TRONG: chi thoat khi diem entail cua MOI doan deu rat
thap, khong phai chi khi nhan la neutral.  NLI hay bo sot bang chung dien dat lai.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from s0_common import (ASK, MISSING, OK, banner, load_config, verdict,
                       write_report)


class NLI:
    """3 lop: contradiction / neutral / entailment."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg["nli"]
        self.model = None
        self.tok = None
        self.error: Optional[str] = None
        self.entail_idx: Optional[int] = None
        try:
            import torch
            from transformers import (AutoModelForSequenceClassification,
                                      AutoTokenizer)
            name = self.cfg["model"]
            self.tok = AutoTokenizer.from_pretrained(name)
            self.model = AutoModelForSequenceClassification.from_pretrained(name)
            self.model.eval()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.torch = torch
            # tim chi so lop entailment tu id2label, dung doan theo thu tu
            id2l = {int(k): str(v).lower()
                    for k, v in self.model.config.id2label.items()}
            for i, lab in id2l.items():
                if "entail" in lab:
                    self.entail_idx = i
            if self.entail_idx is None:
                self.error = ("khong tim duoc lop 'entailment' trong id2label=%s"
                              % id2l)
            self.id2label = id2l
        except Exception as e:
            self.error = repr(e)[:220]

    def entail_probs(self, premises: Sequence[str],
                     hypotheses: Sequence[str]) -> List[float]:
        """P(entailment) cho tung cap (premise, hypothesis)."""
        if self.model is None or self.entail_idx is None or not premises:
            return [0.0] * len(premises)
        out: List[float] = []
        bs = self.cfg["batch_size"]
        for i in range(0, len(premises), bs):
            enc = self.tok(list(premises[i:i + bs]), list(hypotheses[i:i + bs]),
                           truncation=True, padding=True,
                           max_length=self.cfg["max_length"],
                           return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                p = self.torch.softmax(self.model(**enc).logits, dim=-1)
            out += [float(x[self.entail_idx]) for x in p]
        return out


# ── bang phu ────────────────────────────────────────────────────────
def coverage_table(nli: NLI, doc_ids: Sequence[str], doc_texts: Sequence[str],
                   claims: Sequence[str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """|doc| x |claim| diem entail.  0 luot LLM."""
    thr = cfg["nli"]["entail_threshold"]
    prem: List[str] = []; hyp: List[str] = []
    for d in doc_texts:
        for c in claims:
            prem.append(d); hyp.append(c)
    probs = nli.entail_probs(prem, hyp)

    J = len(claims)
    grid: Dict[str, List[float]] = {}
    for i, did in enumerate(doc_ids):
        grid[did] = [round(probs[i * J + j], 4) for j in range(J)]

    # manh j <- tap doan entail no
    covered_by: List[List[str]] = []
    for j in range(J):
        covered_by.append([did for did in doc_ids if grid[did][j] >= thr])

    # ── suc khoe cua MANH — doc mien phi tu chinh bang vua tinh ──────
    #
    # Mot manh hong co hai dang, va hai dang HONG NGUOC NHAU:
    #
    #   mo coi   khong doan nao phu.  Co the pool thieu THAT, cung co the
    #            LLM bia sai thuc the bac cau.  Hai nguyen nhan khac han
    #            nhau ma cung mot bieu hien -> phai bao ca hai kha nang.
    #   rong     qua nua pool phu duoc.  Manh qua chung chung nen khong
    #            loc duoc gi — adq thanh luon-dung, va ca ca moi tham so
    #            cung lot.  Day la kieu hong AM THAM hon, vi no khong
    #            lam truy van thoat, no chi lam bo loc ngung lam viec.
    n = max(1, len(doc_ids))
    vac = cfg["nli"].get("vacuous_fraction", 0.5)
    orphan = [j for j in range(J) if not covered_by[j]]
    vacuous = [j for j in range(J) if len(covered_by[j]) > vac * n]
    return {"claims": list(claims), "J": J, "threshold": thr,
            "grid": grid, "covered_by": covered_by,
            "orphan_claims": orphan, "vacuous_claims": vacuous,
            "coverage_count": [len(covered_by[j]) for j in range(J)],
            "max_entail_per_claim": [
                round(max([grid[d][j] for d in doc_ids], default=0.0), 4)
                for j in range(J)]}


def adq(table: Dict[str, Any], S: Sequence[str]) -> Tuple[bool, Set[int]]:
    """cov(S) = { j : co k thuoc S voi e(k,j) = entail }."""
    s = set(S)
    cov = {j for j, ks in enumerate(table["covered_by"]) if s & set(ks)}
    return len(cov) == table["J"], cov


def should_exit(table: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[bool, List[int]]:
    """Cua thoat THAN TRONG: chi thoat khi MOI doan deu rat thap voi mot manh."""
    cap = cfg["nli"]["exit_max_entail"]
    bad = [j for j, m in enumerate(table["max_entail_per_claim"]) if m < cap]
    return bool(bad), bad


def minimal_covers(table: Dict[str, Any], max_size: int = 4,
                   cap: int = 64) -> List[List[str]]:
    """Nghiem phu TOI TIEU — giai chinh xac, vi bai toan nho.

    Day la tang 0 cua phep tim theo tang o §3.6.
    """
    sets = [set(ks) for ks in table["covered_by"]]
    if any(not s for s in sets):
        return []
    universe = set(range(table["J"]))
    best: List[List[str]] = []
    pool = sorted(set().union(*sets))
    # PHAI bat dau tu co 1, khong phai co J: mot node co the phu NHIEU manh
    # (dung ca summary bao nhieu document) -> nghiem toi tieu co the nho hon J.
    # Bat dau tu J thi bo sot dung ca do.
    max_size = max(max_size, table["J"])
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(pool, size):
            cov = {j for j, s in enumerate(sets) if s & set(combo)}
            if cov != universe:
                continue
            # bo qua neu chua mot nghiem da tim duoc (dam bao TOI TIEU)
            if any(set(b) <= set(combo) for b in best):
                continue
            best.append(sorted(combo))
            if len(best) >= cap:
                return best
        if best:
            break
    return best


# ── kiem ────────────────────────────────────────────────────────────
def run(cfg: Dict[str, Any]) -> Dict[str, Any]:
    banner("0.G  Bo NLI va bang phu")

    rep: Dict[str, Any] = {"model": cfg["nli"]["model"]}
    nli = NLI(cfg)
    rep["error"] = nli.error
    if nli.model is None:
        verdict(MISSING, "khong nap duoc NLI: %s\n"
                         "           pip install transformers torch" % nli.error)
        write_report(cfg, "s0g_nli", rep)
        return rep
    rep["id2label"] = nli.id2label
    rep["entail_index"] = nli.entail_idx
    rep["device"] = nli.device
    verdict(OK, "NLI = %s  (%s, entail=%d)"
            % (cfg["nli"]["model"], nli.device, nli.entail_idx))

    # dung chinh vi du cua tai lieu — co ca doan BAY
    docs = {
        "d3":  "Forrest Gump won the Academy Award for Best Picture for 1994.",
        "d7":  "The 1994 Best Picture award went to Forrest Gump.",
        "d12": "Forrest Gump was directed by Robert Zemeckis.",
        "d23": "Robert Zemeckis directed Forrest Gump and Back to the Future.",
        "d31": "Zemeckis directed Forrest Gump in 1994.",
        "d19": "Forrest Gump was a cultural phenomenon of the 1990s.",
        "d40": "The Shawshank Redemption was released in 1994.",
    }
    claims = ["The film that won Best Picture for 1994 is Forrest Gump.",
              "Forrest Gump was directed by Robert Zemeckis."]

    ids = list(docs); txts = [docs[d] for d in ids]
    tb = coverage_table(nli, ids, txts, claims, cfg)
    rep["table"] = tb

    print("\n     %-6s %s" % ("doan", "  ".join("manh%d" % (j + 1)
                                                for j in range(tb["J"]))))
    for d in ids:
        print("     %-6s %s   %s" % (
            d, "  ".join("%.3f" % x for x in tb["grid"][d]),
            docs[d][:46]))
    print("\n     manh <- doan chong do:")
    for j, ks in enumerate(tb["covered_by"]):
        print("        manh%d <- %s" % (j + 1, ks or "RONG"))

    # ── 1. bay moi tham so co bi loai khong ───────────────────────
    ok19, cov19 = adq(tb, ["d19"])
    rep["adq_bait"] = {"phu_het": ok19, "cov": sorted(cov19)}
    if not ok19 and not cov19:
        verdict(OK, "adq({d19}) khong phu manh nao  ->  BAY MOI THAM SO BI LOAI. "
                    "Day la ca §3.2 dung adq de bat.")
    else:
        verdict(ASK, "adq({d19}) phu %s — doan khong co du kien van entail.\n"
                     "           Nguong entail_threshold=%.2f co the qua thap,\n"
                     "           hoac model NLI nay khong tach duoc 'nhac ten' vs 'khang dinh'."
                % (sorted(cov19), cfg["nli"]["entail_threshold"]))

    # ── 2. hai doan trung y van thieu manh ────────────────────────
    ok37, cov37 = adq(tb, ["d3", "d7"])
    rep["adq_redundant_pair"] = {"phu_het": ok37, "cov": sorted(cov37)}
    verdict(OK if not ok37 else ASK,
            "adq({d3,d7}) = %s, cov=%s  ->  hai doan trung y VAN thieu manh 2"
            % ("phu het" if ok37 else "KHONG phu", sorted(cov37)))

    # ── 3. tap dung phu het ───────────────────────────────────────
    okT, covT = adq(tb, ["d3", "d12"])
    rep["adq_true_pair"] = {"phu_het": okT, "cov": sorted(covT)}
    verdict(OK if okT else ASK,
            "adq({d3,d12}) = %s  ->  day moi la tap du ve bang chung"
            % ("phu het" if okT else "KHONG phu — NLI bo sot bang chung"))

    # ── 4. nghiem phu = tang 0 cua §3.6 ───────────────────────────
    covers = minimal_covers(tb)
    rep["minimal_covers"] = covers
    print("\n     nghiem phu toi tieu (tang 0 cua §3.6): %s" % covers)
    if covers:
        verdict(OK, "%d nghiem phu, co %d  ->  beam khoi tu day, khong mo tu rong"
                % (len(covers), len(covers[0])))
    else:
        verdict(ASK, "khong co nghiem phu  ->  cua thoat se kich. Kiem lai nguong.")

    # Mot node co the phu NHIEU manh -> nghiem co the nho hon J.
    # Voi summary o lai trong pool, do la ca "toi tieu uu tien summary":
    # parsimony -|S| se cho {summary} co 1 THANG {d3,d12} co 2.
    # GD 0 chi DO va gan co; quyet dinh thuoc §3.8.
    smaller = [c for c in covers if len(c) < tb["J"]]
    rep["covers_smaller_than_J"] = smaller
    if smaller:
        verdict(ASK,
                "co %d nghiem phu NHO HON so manh J=%d: %s\n"
                "           mot node phu nhieu manh — dung ca summary bao nhieu document.\n"
                "           parsimony -|S| se cho no THANG tap document tuong duong.\n"
                "           CAN QUYET DINH (§3.8): parsimony dem SO NGUON BI BAO,\n"
                "           hay chap nhan T chi gom summary va gan co cho tang B loc?"
                % (len(smaller), tb["J"], smaller[:4]))
    else:
        verdict(OK, "moi nghiem phu co co = J = %d  ->  chua gap ca summary bao trum"
                % tb["J"])

    # ── 5. cua thoat ──────────────────────────────────────────────
    ex, bad = should_exit(tb, cfg)
    rep["exit"] = {"would_exit": ex, "uncovered_claims": bad}
    verdict(OK if not ex else ASK,
            "cua thoat: %s%s" % ("KHONG thoat" if not ex else "THOAT",
                                 "" if not bad else "  manh thieu %s" % bad))

    # ── 6. ca pool thieu: bo d12/d23/d31 ──────────────────────────
    ids2 = ["d3", "d7", "d19", "d40"]
    tb2 = coverage_table(nli, ids2, [docs[d] for d in ids2], claims, cfg)
    ex2, bad2 = should_exit(tb2, cfg)
    rep["exit_when_pool_short"] = {"would_exit": ex2, "uncovered": bad2,
                                   "max_entail": tb2["max_entail_per_claim"]}
    verdict(OK if ex2 else ASK,
            "pool bo het doan noi dao dien  ->  %s  (max entail %s)"
            % ("THOAT, dung" if ex2 else "KHONG thoat — cua thoat qua long",
               tb2["max_entail_per_claim"]))

    p = write_report(cfg, "s0g_nli", rep)
    print("\n  -> %s" % p)
    return rep


if __name__ == "__main__":
    run(load_config())
