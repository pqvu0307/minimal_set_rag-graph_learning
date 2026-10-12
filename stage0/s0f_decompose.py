# -*- coding: utf-8 -*-
"""0.F — Che cau hoi thanh manh (sub-claim).

Hai duong:
  MuSiQue           phan ra 1-hop + dap an trung gian GAN SAN  ->  0 luot LLM
  HotpotQA / 2Wiki  LLM, 1 luot MOI TRUY VAN (khong phai moi tap con)

Va dung chinh nhanh MuSiQue de DO xem LLM che co sot hop khong — cach duy nhat
kiem buoc nay ma khong can nguoi gan.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from s0_common import (ASK, MISSING, OK, banner, load_config, resolve,
                       verdict, write_report)

# Khong co doan kem theo: LLM phai TU NHO thuc the bac cau.  Giu lam duong lui,
# khong phai duong chinh.
PROMPT = (
    "Decompose the question into the minimal set of atomic factual claims that "
    "together entail the gold answer. Each claim must be a single standalone "
    "sentence stating one fact, verifiable against one passage.\n"
    "Return ONLY a JSON array of strings, no prose.\n\n"
    "Question: {q}\nGold answer: {y}\n\nClaims:")

# Co doan kem theo: viec cua LLM tu NHO thanh DOC.
#
# Vi sao quan trong: voi "ai dao dien phim thang oscar 1994?" + "Zemeckis",
# thuc the bac cau "Forrest Gump" KHONG nam trong q lan y.  Khong dua doan vao
# thi LLM phai lay tu tri nho tham so — dung thi may, sai thi manh vo nghia,
# khong doan nao phu, va truy van thoat o c0 NHU THE pool thieu.
# Dua doan vao thi "Forrest Gump" nam san trong doan 3: chi can doc ra.
#
# Bat trich nguyen van de KIEM DUOC: quote nao khong tim thay trong doan la
# LLM dang bia, va ta loai manh do bang mot phep so chuoi, khong can model.
PROMPT_GROUNDED = (
    "Below are candidate passages, then a question and its gold answer.\n"
    "Decompose into the minimal set of atomic factual claims that together "
    "entail the gold answer. Each claim must be one standalone sentence "
    "stating one fact, verifiable against ONE passage.\n"
    "Every claim MUST be supported by a verbatim quote from the passages. "
    "Do not use outside knowledge.\n"
    'Return ONLY a JSON array of objects: [{{"claim": "...", "quote": "..."}}]\n\n'
    "Passages:\n{passages}\n\nQuestion: {q}\nGold answer: {y}\n\nClaims:")


def _norm(s: str) -> str:
    """Chuan hoa de so trich dan: thuong hoa, gom khoang trang."""
    return re.sub(r"\s+", " ", s.lower()).strip()


def passages_of(row: Dict[str, Any]) -> List[str]:
    """Doan ung vien kem theo mot row — MuSiQue `paragraphs`, HotpotQA `context`."""
    out: List[str] = []
    for p in row.get("paragraphs") or []:
        if isinstance(p, dict):
            t = (p.get("paragraph_text") or p.get("text") or "").strip()
            if t:
                out.append(("%s. %s" % (p.get("title", ""), t)).strip(". "))
    for c in row.get("context") or []:            # HotpotQA: [title, [sents]]
        if isinstance(c, (list, tuple)) and len(c) == 2:
            out.append(("%s. %s" % (c[0], " ".join(c[1]))).strip())
    return out


# ── nhanh 1: phan ra gan san (MuSiQue) ──────────────────────────────
#
# MuSiQue KHONG cho san "manh".  No cho (cau hoi con, dap an con).  Model NLI
# duoc huan luyen tren CAU KHANG DINH, nen dua thang "Who directed X? -> Y"
# vao lam hypothesis la ngoai phan phoi — bang phu se la rac, va khong co
# canh bao nao.  Phai chuyen ve menh de truoc.
#
# Phu bang mau cho cac dang hay gap cua MuSiQue (sinh tu quan he Wikidata),
# con lai dung khuon bao phu.  Ti le dung mau duoc DEM va bao o 0.F — thap
# thi nen chuyen han sang nhanh LLM.
_PAT = [
    # "Who is the director of X?"  ·  "What is the capital of X?"
    (re.compile(r"^(?:who|what|which)\s+(?:is|was|are|were)\s+the\s+"
                r"(?P<rel>[\w\s\-]+?)\s+(?:of|for|in)\s+(?P<ent>.+?)\s*\?*$", re.I),
     "{ans} is the {rel} of {ent}."),
    # "Who directed X?"  ·  "Who founded X?"  ·  "Who wrote X?"
    (re.compile(r"^who\s+(?P<verb>\w+ed|wrote|led|won)\s+(?P<ent>.+?)\s*\?*$", re.I),
     "{ans} {verb} {ent}."),
    # "When was X born / released / founded?" — chi bat dang BI DONG ("was"),
    # vi "When did X die?" doi chia lai dong tu ("die" -> "died") ma lam bang
    # tay thi sai nhieu hon dung.  De no roi xuong khuon bao phu.
    (re.compile(r"^when\s+was\s+(?P<ent>.+?)\s+(?P<verb>\w+)\s*\?*$", re.I),
     "{ent} was {verb} in {ans}."),
    # "Where is X located?"
    (re.compile(r"^where\s+(?:is|was)\s+(?P<ent>.+?)\s+located\s*\?*$", re.I),
     "{ent} is located in {ans}."),
]
_FALLBACK = 'The answer to "{q}" is {ans}.'


def declarativize(sub_q: str, sub_a: str) -> Tuple[str, bool]:
    """(cau hoi con, dap an con) -> mot MENH DE, kem co da khop mau hay chua."""
    q = sub_q.strip().rstrip("?").strip() + "?"
    for rx, tmpl in _PAT:
        m = rx.match(q)
        if m:
            g = {k: v.strip() for k, v in m.groupdict().items() if v}
            out = tmpl.format(ans=sub_a, **g)
            return (out[:1].upper() + out[1:]), True
    return _FALLBACK.format(q=q, ans=sub_a), False


def from_annotation(row: Dict[str, Any],
                    stats: Optional[Dict[str, int]] = None) -> Optional[List[str]]:
    """MuSiQue: question_decomposition = [{question, answer, paragraph_support_idx}]."""
    dec = row.get("question_decomposition") or row.get("decomposition")
    if not dec:
        return None
    out: List[str] = []
    answers: List[str] = []
    for step in dec:
        if not isinstance(step, dict):
            return None
        sq = (step.get("question") or "").strip()
        sa = (step.get("answer") or "").strip()
        if not sq or not sa:
            continue
        # giai tham chieu kieu "#1" ve DAP AN con truoc do (khong phai ve ca
        # menh de — menh de da duoc viet lai nen khong con tach bang " -> ")
        sq = re.sub(r"#(\d+)",
                    lambda m: answers[int(m.group(1)) - 1]
                    if 0 < int(m.group(1)) <= len(answers) else m.group(0), sq)
        claim, matched = declarativize(sq, sa)
        if stats is not None:
            stats["pattern" if matched else "fallback"] = \
                stats.get("pattern" if matched else "fallback", 0) + 1
        answers.append(sa)
        out.append(claim)
    return out or None


# ── nhanh 2: LLM ────────────────────────────────────────────────────
class LLMDecomposer:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        # mac dinh theo judge (DeepSeek) — che cau hoi la 1 luot MOI TRUY VAN,
        # nen re, khong can chay local
        self.model = (cfg["decompose"].get("model")
                      or cfg["judge"]["judge_model"])
        self.cli = None
        self.error: Optional[str] = None
        if cfg["decompose"]["provider"] == "dummy" or self.model == "PHAI_DIEN":
            self.error = "provider/model chua dien"
            return
        try:
            from openai import OpenAI
            key = os.environ.get(cfg["judge"]["api_key_env"], "")
            if not key:
                raise RuntimeError("thieu %s" % cfg["judge"]["api_key_env"])
            self.cli = OpenAI(api_key=key, base_url=cfg["judge"]["base_url"])
        except Exception as e:
            self.error = repr(e)[:200]

    def __call__(self, q: str, y: str,
                 passages: Optional[Sequence[str]] = None
                 ) -> Tuple[List[str], Dict[str, int]]:
        if self.cli is None:
            return [], {"prompt_tokens": 0, "output_tokens": 0, "ungrounded": 0}
        if passages:
            k = self.cfg["decompose"].get("grounding_passages", 12)
            body = "\n".join("[%d] %s" % (i + 1, p.strip().replace("\n", " ")[:600])
                             for i, p in enumerate(passages[:k]))
            prompt = PROMPT_GROUNDED.format(passages=body, q=q, y=y)
        else:
            prompt = PROMPT.format(q=q, y=y)
        r = self.cli.chat.completions.create(
            model=self.model, temperature=0.0, max_tokens=600,
            messages=[{"role": "user", "content": prompt}])
        txt = (r.choices[0].message.content or "").strip()
        m = re.search(r"\[.*\]", txt, re.S)
        claims: List[str] = []
        quotes: List[str] = []
        if m:
            try:
                for c in json.loads(m.group(0)):
                    if isinstance(c, dict):
                        s = str(c.get("claim", "")).strip()
                        if s:
                            claims.append(s)
                            quotes.append(str(c.get("quote", "")).strip())
                    elif str(c).strip():
                        claims.append(str(c).strip()); quotes.append("")
            except Exception:
                pass
        if not claims:
            claims = [ln.strip("-* ").strip() for ln in txt.splitlines()
                      if len(ln.strip()) > 10]
            quotes = [""] * len(claims)

        # Loai manh khong neo duoc vao doan nao — do la LLM dang bia.  Phep so
        # chuoi thuan tuy, 0 model, nen khong them mot nguon sai nao.
        ungrounded = 0
        if passages:
            blob = _norm(" ".join(passages))
            keep_c, keep_q = [], []
            for c, qt in zip(claims, quotes):
                if qt and _norm(qt) not in blob:
                    ungrounded += 1
                    continue
                keep_c.append(c); keep_q.append(qt)
            if keep_c:                      # khong bao gio tra ve rong vi loc
                claims = keep_c
        u = r.usage
        return claims[: self.cfg["decompose"]["max_claims"]], {
            "prompt_tokens": getattr(u, "prompt_tokens", 0),
            "output_tokens": getattr(u, "completion_tokens", 0),
            "ungrounded": ungrounded}


def decompose(row: Dict[str, Any], llm: Optional[LLMDecomposer],
              cfg: Dict[str, Any],
              passages: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    if cfg["decompose"]["use_annotated_when_available"]:
        ann = from_annotation(row)
        if ann:
            return {"claims": ann, "source": "annotated", "usage": {}}
    if llm is None or llm.cli is None:
        return {"claims": [], "source": "none", "usage": {}}
    # uu tien doan that (pool cua c0); khong co thi lay doan kem theo row
    ps = list(passages) if passages else passages_of(row)
    claims, usage = llm(row.get("question", ""), str(row.get("answer", "")), ps)
    return {"claims": claims, "source": "llm_grounded" if ps else "llm",
            "usage": usage}


# ── kiem: LLM che co sot hop khong (do tren MuSiQue) ────────────────
def run(cfg: Dict[str, Any], n_probe: int = 20) -> Dict[str, Any]:
    banner("0.F  Che cau hoi thanh manh")

    rep: Dict[str, Any] = {}
    rdir = resolve(cfg, "raw_dir")

    # tim mot file co phan ra gan san
    rows: List[Dict[str, Any]] = []
    src_file = None
    for base in (rdir / "musique", rdir):
        for split in ("train.json", "dev.json", "valid.json"):
            p = base / split
            if not p.exists():
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            cand = data if isinstance(data, list) else list(data.values())
            if any(r.get("question_decomposition") or r.get("decomposition")
                   for r in cand[:200]):
                rows = cand; src_file = str(p); break
        if rows:
            break

    rep["annotated_source"] = src_file
    if not rows:
        verdict(MISSING,
                "khong thay file nao co question_decomposition.\n"
                "           -> khong do duoc 'LLM che co sot hop khong',\n"
                "              tuc mat lop kiem duy nhat khong can nguoi gan.\n"
                "           Can file MuSiQue goc (khong chi ban da xu ly cua gfm-rag).")
        write_report(cfg, "s0f_decompose", rep)
        return rep

    have = [r for r in rows
            if (r.get("question_decomposition") or r.get("decomposition"))
            and r.get("question") and r.get("answer")][:n_probe]
    verdict(OK, "co %d truy van co phan ra gan san (dung %d de do)"
            % (sum(1 for r in rows if r.get("question_decomposition")
                   or r.get("decomposition")), len(have)))

    # phan bo so manh J — day la PRIOR cho |T|
    dstat: Dict[str, int] = {}
    js = [len(from_annotation(r, dstat) or []) for r in rows[:5000]]
    js = [x for x in js if x]
    from collections import Counter
    rep["J_distribution_annotated"] = dict(sorted(Counter(js).items()))
    print("     phan bo so manh J (gan san): %s" % rep["J_distribution_annotated"])
    print("     -> J chinh la PRIOR cho |T|.  |T| lech xa J  =>  phinh, chac chan.")

    # MuSiQue cho (cau hoi con, dap an con), KHONG cho menh de.  Ta phai viet
    # lai thanh cau khang dinh truoc khi dua vao NLI.  Ti le dung mau sach cao
    # thi nhanh nay dung duoc mien phi; thap thi nen chuyen han sang LLM.
    npat, nfb = dstat.get("pattern", 0), dstat.get("fallback", 0)
    rate = npat / max(1, npat + nfb)
    rep["declarativize"] = {"pattern": npat, "fallback": nfb,
                            "pattern_rate": round(rate, 3)}
    print("     viet lai thanh menh de: %d theo mau sach, %d theo khuon bao phu"
          % (npat, nfb))
    if rate < 0.6:
        verdict(ASK, "chi %.0f%% cau hoi con khop mau sach; phan con lai dung\n"
                     "           khuon 'The answer to \"...\" is X.' — NLI doc duoc\n"
                     "           nhung yeu hon.  Can thi them mau, hoac cho ca\n"
                     "           MuSiQue chay nhanh LLM." % (100 * rate))
    else:
        verdict(OK, "%.0f%% cau hoi con khop mau sach" % (100 * rate))

    llm = LLMDecomposer(cfg)
    rep["llm_error"] = llm.error
    if llm.cli is None:
        verdict(ASK, "LLM chua cau hinh (%s)  ->  chua do duoc do khop.\n"
                     "           Dien config.decompose.model roi chay lai buoc nay."
                % llm.error)
        write_report(cfg, "s0f_decompose", rep)
        return rep

    # Do HAI thu, tren CUNG mot bo co dap so:
    #   1. so manh co khop khong          (sot hop)
    #   2. co bat duoc THUC THE BAC CAU khong
    # (2) moi la rui ro that cua corpus khong co phan ra san: thuc the bac cau
    # khong nam trong q lan y, nen LLM phai doc ra tu doan — hoac bia.
    # Chay CA HAI che do de thay dua doan vao co an thua gi khong.
    def bridges(row: Dict[str, Any]) -> List[str]:
        """Dap an TRUNG GIAN — khong tinh dap an cuoi, vi no da nam trong y."""
        dec = row.get("question_decomposition") or []
        return [str(s.get("answer", "")).strip() for s in dec[:-1]
                if isinstance(s, dict) and str(s.get("answer", "")).strip()]

    def hit(claims: List[str], bs: List[str]) -> Optional[bool]:
        if not bs:
            return None
        blob = _norm(" ".join(claims))
        return all(_norm(b) in blob for b in bs)

    rows_out = []
    agree = 0
    br = {"grounded_hit": 0, "grounded_tot": 0, "bare_hit": 0, "bare_tot": 0}
    for r in have:
        ann = from_annotation(r) or []
        bs = bridges(r)
        ps = passages_of(r)
        got, usage = llm(r["question"], str(r["answer"]), ps)
        bare, _ = llm(r["question"], str(r["answer"]))      # khong dua doan
        ok = len(got) == len(ann)
        agree += int(ok)
        hg, hb = hit(got, bs), hit(bare, bs)
        for k, h in (("grounded", hg), ("bare", hb)):
            if h is not None:
                br[k + "_tot"] += 1
                br[k + "_hit"] += int(h)
        rows_out.append({"question": r["question"][:90],
                         "J_annotated": len(ann), "J_llm": len(got),
                         "match": ok, "llm_claims": got,
                         "bridges": bs, "bridge_ok_grounded": hg,
                         "bridge_ok_bare": hb,
                         "n_passages": len(ps), "usage": usage})
    rep["probe"] = rows_out
    rep["J_match_rate"] = round(agree / max(1, len(have)), 3)
    rep["bridge_entity"] = br
    rep["ungrounded_claims"] = sum(x["usage"].get("ungrounded", 0) for x in rows_out)
    tok = sum(x["usage"].get("prompt_tokens", 0) + x["usage"].get("output_tokens", 0)
              for x in rows_out)
    rep["mean_tokens_per_query"] = round(tok / max(1, len(rows_out)), 1)

    print("\n     ty le khop so manh = %.1f%%   (%d/%d)"
          % (100 * rep["J_match_rate"], agree, len(have)))
    print("     token moi truy van   = %.0f   (config gia dinh ~400)"
          % rep["mean_tokens_per_query"])

    if rep["J_match_rate"] >= 0.85:
        verdict(OK, "LLM che khop tot  ->  dung duoc cho nhanh HotpotQA / 2Wiki")
    elif rep["J_match_rate"] >= 0.6:
        verdict(ASK, "khop %.0f%% — co kha nang SOT HOP.  Hau qua: bang phu thieu cot,\n"
                     "           va T duoc nhan khi CON THIEU bang chung.\n"
                     "           -> sua prompt, hoac gan co khi J khong khop so hop du kien"
                % (100 * rep["J_match_rate"]))
    else:
        verdict(ASK, "khop chi %.0f%% — chua dung duoc.  Phai sua buoc nay TRUOC GD 1,\n"
                     "           khong thi bang phu sai o goc va moi thu sau sai theo"
                % (100 * rep["J_match_rate"]))

    # ── thuc the bac cau: phep do quan trong nhat cua buoc nay ──────
    if br["grounded_tot"]:
        rg = br["grounded_hit"] / br["grounded_tot"]
        rb = br["bare_hit"] / max(1, br["bare_tot"])
        print("\n     bat duoc THUC THE BAC CAU:")
        print("       co dua doan vao    %3d/%3d = %.0f%%"
              % (br["grounded_hit"], br["grounded_tot"], 100 * rg))
        print("       chi q + y          %3d/%3d = %.0f%%"
              % (br["bare_hit"], br["bare_tot"], 100 * rb))
        print("     manh bi loai vi trich dan khong co trong doan: %d"
              % rep["ungrounded_claims"])
        if rg >= 0.9:
            verdict(OK, "neo vao doan thi LLM doc duoc thuc the bac cau\n"
                        "           ->  dung duoc cho corpus khong co phan ra san")
        else:
            verdict(ASK, "chi %.0f%% bat duoc thuc the bac cau DU DA dua doan vao.\n"
                         "           Hau qua: manh vo nghia -> khong doan nao phu ->\n"
                         "           truy van thoat o c0 NHU THE pool thieu, khong\n"
                         "           phan biet duoc.  Phai sua truoc GD 1." % (100 * rg))
        if rb < rg - 0.05:
            print("     -> dua doan vao an them %.0f diem.  Giu che do grounded."
                  % (100 * (rg - rb)))

    p = write_report(cfg, "s0f_decompose", rep)
    print("\n  -> %s" % p)
    return rep


if __name__ == "__main__":
    run(load_config())
