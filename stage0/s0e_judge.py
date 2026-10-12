# -*- coding: utf-8 -*-
"""0.E — Khung judge:  v(S) = P_judge("dung" | q, a_S, y).

Ba chi tiet khong duoc lam sai, vi ca mo hinh chi phi treo vao chung:

  1  judge tra SO LIEN TUC — doc logprob token DAU cua Yes/No.
     Thang 1-5 lam hang chuc tap con hoa nhau, S va S\\{c} khong phan biet
     duoc, va "toi tieu" mat nghia.
  2  cache khoa bam NOI DUNG chunk da sap, khong bam ten.
     Bam ten thi chunk lai corpus se lam ten va nhau va cache tra sai.
  3  gom lo theo TIEN TO CHUNG.  Vong 1 greedy sinh dung dang do: moi ung vien
     la S + {k} voi cung S.  Neu prompt caching khong an thi cost(S) sai.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from s0_common import (ASK, MISSING, OK, DiskCache, Telemetry, TokenCounter,
                       banner, cache_key, cost_tokens, load_config, resolve,
                       verdict, write_report)

GEN_TMPL = (
    "Answer the question using ONLY the passages below. "
    "If the passages are insufficient, say exactly: INSUFFICIENT.\n\n"
    "{passages}\nQuestion: {q}\nAnswer:")

JUDGE_TMPL = (
    "Question: {q}\nGold answer: {y}{aliases}\nCandidate answer: {a}\n\n"
    "Is the candidate answer correct, i.e. semantically equivalent to the gold "
    "answer (or to any of the also-acceptable answers)? "
    "Reply with exactly one word: Yes or No.\nAnswer:")


def _aliases_line(aliases, enabled: bool) -> str:
    """answer_aliases vao prompt — khong co thi "Zemeckis" vs "Robert Zemeckis"
    bi cham No, v(S) thap gia, va tap DUNG bi loai."""
    if not enabled or not aliases:
        return ""
    xs = [str(a).strip() for a in aliases if str(a).strip()]
    return "\nAlso acceptable: " + " · ".join(xs[:8]) if xs else ""


# ── adapter nha cung cap ────────────────────────────────────────────
class DummyBackend:
    """Chay duoc khong can API — de kiem duong ong, KHONG de lay so that."""
    name = "dummy"

    def generate(self, prompt: str, max_tokens: int, temperature: float):
        return "INSUFFICIENT", {"prompt_tokens": 0, "output_tokens": 0}

    def first_token_logprobs(self, prompt: str) -> Dict[str, float]:
        return {"No": math.log(0.9), "Yes": math.log(0.1)}


class OpenAICompatBackend:
    """Bat ky endpoint tuong thich OpenAI: OpenAI, vLLM, Together, ...

    Judge BAT BUOC endpoint ho tro logprobs + top_logprobs.
    """
    name = "openai_compat"

    def __init__(self, cfg: Dict[str, Any]):
        from openai import OpenAI            # pip install openai
        key = os.environ.get(cfg["judge"]["api_key_env"], "")
        if not key:
            raise RuntimeError("thieu bien moi truong %s"
                               % cfg["judge"]["api_key_env"])
        self.cli = OpenAI(api_key=key, base_url=cfg["judge"]["base_url"])
        # backend nay gio CHI lam judge khi provider = split_local_gen;
        # giu gen_model cho che do openai_compat (sinh qua API) neu can
        self.gen_model = cfg["judge"].get("gen_model") or cfg.get("gen", {}).get("model")
        self.judge_model = cfg["judge"]["judge_model"]

    def generate(self, prompt: str, max_tokens: int, temperature: float):
        r = self.cli.chat.completions.create(
            model=self.gen_model, max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}])
        u = r.usage
        return (r.choices[0].message.content or "").strip(), {
            "prompt_tokens": getattr(u, "prompt_tokens", 0),
            "output_tokens": getattr(u, "completion_tokens", 0),
            "cached_prompt_tokens": getattr(
                getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0,
        }

    # Co endpoint tra DU top_logprobs, co endpoint chi tra logprob cua token
    # DA CHON.  DeepInfra (10/2026) thuoc loai thu hai: no NHAN top_logprobs
    # ma khong bao gio tra — dung kieu hong im lang ma check_logprobs_sane
    # duoc viet ra de bat.  Nen o day ghi ro dang dung che do nao.
    degraded = False

    def first_token_logprobs(self, prompt: str) -> Dict[str, float]:
        r = self.cli.chat.completions.create(
            model=self.judge_model, max_tokens=1, temperature=0.0,
            logprobs=True, top_logprobs=20,
            messages=[{"role": "user", "content": prompt}])
        lp = r.choices[0].logprobs
        if not lp or not lp.content:
            raise RuntimeError("endpoint khong tra logprobs — judge khong dung duoc")
        c = lp.content[0]
        top = getattr(c, "top_logprobs", None)
        if top:
            return {t.token: t.logprob for t in top}

        # XAP XI MOT TOKEN.  Chi biet token da chon va xac suat cua no, nen
        # don phan con lai sang phia doi lap:  p(Yes) = p  hoac  1 - p.
        # Dung duoc vi prompt ep tra loi MOT TU trong {Yes, No}, nen khoi
        # luong roi ra ngoai hai tu do rat nho.  Van la xap xi — v(S) o che
        # do nay co do phan giai thap hon, nhat la khi model rat chac.
        import math as _m
        type(self).degraded = True
        p = min(max(_m.exp(c.logprob), 1e-6), 1 - 1e-6)
        tok = (c.token or "").strip()
        yes = tok in ("Yes", "yes", "YES")
        return {"Yes": _m.log(p if yes else 1 - p),
                "No": _m.log(1 - p if yes else p)}


class LocalHFBackend:
    """Sinh cau tra loi bang model chay TRONG MAY (Colab GPU).

    vLLM neu co (nhanh hon nhieu, co prefix caching); khong thi transformers.
    Chi lam viec `generate` — judge van qua API, vi can logprobs on dinh.

    Doi voi mo hinh chi phi: sinh o day KHONG ton tien, ton GIO GPU.
    Nen rang buoc chuyen tu TIEN sang THONG LUONG + gioi han session Colab,
    va DiskCache tro thanh bat buoc de chay lai duoc nhieu phien.
    """
    name = "local_hf"

    def __init__(self, cfg: Dict[str, Any]):
        g = cfg["gen"]
        self.model_id = g["model"]
        self.impl = None
        self.error: Optional[str] = None
        self.max_tokens = g["max_tokens"]
        self.temperature = g["temperature"]
        # Chat template PHAI co.  Dua prompt tho vao model instruct la sai tu
        # goc; voi ho Qwen3/3.5 con te hon vi chung MAC DINH BAT thinking —
        # toan bo dau ra chay vao <think> va `content` ve RONG.  Da gap dung
        # loi do khi goi Qwen3.5-9B qua API: 400 token van chua nghi xong.
        self.chat_tok = None
        try:
            from transformers import AutoTokenizer
            self.chat_tok = AutoTokenizer.from_pretrained(self.model_id)
        except Exception as e:
            self.error = "tokenizer: %s" % repr(e)[:120]
        try:
            from vllm import LLM, SamplingParams        # type: ignore
            self._LLM = LLM(model=self.model_id,
                            dtype=g.get("dtype", "auto"),
                            gpu_memory_utilization=g.get("gpu_mem_util", 0.90),
                            max_model_len=g.get("max_model_len", 8192),
                            enable_prefix_caching=True)
            self._SP = SamplingParams
            self.impl = "vllm"
            return
        except Exception as e:
            self.error = "vllm: %s" % repr(e)[:120]
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(self.model_id)
            self.mdl = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype="auto", device_map="auto")
            self.mdl.eval()
            self.torch = torch
            self.impl = "transformers"
        except Exception as e:
            self.error = (self.error or "") + " | transformers: %s" % repr(e)[:120]

    def _chat(self, prompt: str) -> str:
        """Boc prompt vao chat template, va TAT thinking neu model co."""
        if self.chat_tok is None:
            return prompt
        msgs = [{"role": "user", "content": prompt}]
        for kw in ({"enable_thinking": False}, {}):
            try:
                return self.chat_tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, **kw)
            except TypeError:
                continue          # template khong nhan enable_thinking
        return prompt

    @staticmethod
    def _strip_think(t: str) -> str:
        """Phong khi thinking van lot ra: bo phan <think>...</think>."""
        if "</think>" in t:
            t = t.rsplit("</think>", 1)[1]
        return t.strip()

    def generate(self, prompt: str, max_tokens: int, temperature: float):
        prompt = self._chat(prompt)
        if self.impl == "vllm":
            sp = self._SP(temperature=temperature, max_tokens=max_tokens)
            out = self._LLM.generate([prompt], sp, use_tqdm=False)
            txt = self._strip_think(out[0].outputs[0].text)
            return txt, {"prompt_tokens": len(out[0].prompt_token_ids),
                         "output_tokens": len(out[0].outputs[0].token_ids),
                         "cached_prompt_tokens": 0, "local": True}
        if self.impl == "transformers":
            enc = self.tok(prompt, return_tensors="pt").to(self.mdl.device)
            with self.torch.no_grad():
                out = self.mdl.generate(
                    **enc, max_new_tokens=max_tokens,
                    do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                    pad_token_id=self.tok.eos_token_id)
            new = out[0][enc["input_ids"].shape[1]:]
            return self._strip_think(
                self.tok.decode(new, skip_special_tokens=True)), {
                "prompt_tokens": int(enc["input_ids"].shape[1]),
                "output_tokens": int(new.shape[0]),
                "cached_prompt_tokens": 0, "local": True}
        raise RuntimeError("khong nap duoc model sinh local: %s" % self.error)

    def first_token_logprobs(self, prompt: str) -> Dict[str, float]:
        raise RuntimeError("LocalHFBackend chi sinh; judge phai qua API")


class SplitBackend:
    """Sinh = backend A (local)  ·  judge = backend B (API).

    Hai nua cua v(S) chay o hai cho khac nhau, va do la co y:
    sinh can thong luong (GPU), judge can LOGPROBS ON DINH (API).
    """
    name = "split"

    def __init__(self, gen, judge):
        self.gen, self.judge = gen, judge

    def generate(self, prompt: str, max_tokens: int, temperature: float):
        return self.gen.generate(prompt, max_tokens, temperature)

    def first_token_logprobs(self, prompt: str) -> Dict[str, float]:
        return self.judge.first_token_logprobs(prompt)


def make_backend(cfg: Dict[str, Any]):
    p = cfg["judge"]["provider"]
    if p == "dummy":
        return DummyBackend()
    if p == "openai_compat":
        return OpenAICompatBackend(cfg)
    if p == "split_local_gen":
        return SplitBackend(LocalHFBackend(cfg), OpenAICompatBackend(cfg))
    raise ValueError("provider la khong biet: %s" % p)


def check_logprobs_sane(lp: Dict[str, float]) -> Optional[str]:
    """DeepSeek V3.2 co bao cao tra logprobs toan 0 / -9999.

    Loi nay IM LANG: tham so duoc nhan, response co truong logprobs, nhung so
    vo nghia -> v(S) thanh nhi phan -> "toi tieu" mat nghia (dung thu §3.1
    canh bao o vu thang 1-5).  Phai kiem GIA TRI, khong chi kiem ton tai.
    """
    if not lp:
        return "logprobs rong"
    vals = list(lp.values())
    if all(v == 0 for v in vals):
        return "moi logprob = 0 -> vo nghia"
    if any(v <= -9000 for v in vals) and len({round(v) for v in vals}) <= 2:
        return "logprobs chi co 0 va -9999 -> suy bien (loi da bao o DeepSeek V3.2)"
    if len({round(v, 3) for v in vals}) <= 1:
        return "moi logprob bang nhau -> vo nghia"
    return None


# ── v(S) ────────────────────────────────────────────────────────────
@dataclass
class JudgeResult:
    v: float
    answer: str
    cost_est: int
    cost_real: int
    cached: bool
    p_yes: float


class Judge:
    def __init__(self, cfg: Dict[str, Any], telemetry: Optional[Telemetry] = None):
        self.cfg = cfg
        self.jc = cfg["judge"]
        self.backend = make_backend(cfg)
        self.cache = DiskCache(resolve(cfg, "cache") / "judge")
        self.tc = TokenCounter()
        self.tel = telemetry or Telemetry()
        self.overhead = self.jc["prompt_overhead_tokens"]

    def _model_tag(self) -> str:
        """Dinh danh CA hai model trong khoa cache.

        Doi model sinh HOAC model judge la doi ket qua, nen cache phai tach.
        """
        gen = self.cfg.get("gen", {}).get("model") or self.jc.get("gen_model") or "?"
        return "%s|%s" % (gen, self.jc["judge_model"])

    # thu tu CHUAN de tien to chung dung duoc o prompt caching
    @staticmethod
    def canonical(doc_ids: Sequence[str]) -> List[str]:
        return sorted(doc_ids)

    def _passages(self, doc_ids: Sequence[str], texts: Dict[str, str]) -> str:
        return "\n\n".join("[%d] %s" % (i + 1, texts[d])
                           for i, d in enumerate(doc_ids))

    def p_yes(self, logprobs: Dict[str, float]) -> float:
        ys = [v for k, v in logprobs.items() if k in self.jc["yes_tokens"]]
        ns = [v for k, v in logprobs.items() if k in self.jc["no_tokens"]]
        if not ys and not ns:
            return float("nan")
        sy = sum(math.exp(v) for v in ys)
        sn = sum(math.exp(v) for v in ns)
        return sy / (sy + sn) if (sy + sn) > 0 else float("nan")

    def v(self, q: str, y: str, doc_ids: Sequence[str],
          texts: Dict[str, str], step: str = "A1",
          aliases: Optional[Sequence[str]] = None) -> JudgeResult:
        ids = self.canonical(doc_ids)
        chunk_texts = [texts[d] for d in ids]
        al = _aliases_line(aliases, self.jc.get("use_answer_aliases", True))
        # aliases vao KHOA cache: doi aliases la doi phep cham
        key = cache_key(self._model_tag(),
                        GEN_TMPL + JUDGE_TMPL + q + y + al, chunk_texts)

        est = cost_tokens(self.tc.count_many(chunk_texts), self.overhead)

        hit = self.cache.get(key)
        if hit is not None:
            self.tel.tally(step).add(0, 0, cached=True)
            return JudgeResult(hit["v"], hit["answer"], est,
                               hit.get("cost_real", 0), True, hit["p_yes"])

        gen_prompt = GEN_TMPL.format(passages=self._passages(ids, texts), q=q)
        ans, u1 = self.backend.generate(
            gen_prompt, self.jc["gen_max_tokens"], self.jc["temperature"])

        lp = self.backend.first_token_logprobs(
            JUDGE_TMPL.format(q=q, y=y, aliases=al, a=ans or "(empty)"))
        bad = check_logprobs_sane(lp)
        if bad:
            raise RuntimeError(
                "logprobs khong dung duoc: %s\n"
                "  Dung deepseek-chat, KHONG dung deepseek-reasoner — thinking mode\n"
                "  nhan tham so logprobs nhung KHONG co tac dung (loi im lang)." % bad)
        py = self.p_yes(lp)

        real = (u1.get("prompt_tokens", 0) + u1.get("output_tokens", 0))
        self.tel.tally(step).add(u1.get("prompt_tokens", 0),
                                 u1.get("output_tokens", 0))
        res = {"v": py, "p_yes": py, "answer": ans, "cost_real": real,
               "cached_prompt_tokens": u1.get("cached_prompt_tokens", 0)}
        self.cache.put(key, res)
        return JudgeResult(py, ans, est, real, False, py)


# ── kiem duong ong + do prompt caching co AN khong ──────────────────
def run(cfg: Dict[str, Any], texts: Optional[Dict[str, str]] = None,
        q: str = "Who directed the film that won Best Picture for 1994?",
        y: str = "Robert Zemeckis",
        aliases: Optional[Sequence[str]] = ("Zemeckis", "Robert Lee Zemeckis")
        ) -> Dict[str, Any]:
    banner("0.E  Khung judge  —  va do prompt caching co an khong")

    rep: Dict[str, Any] = {
        "provider": cfg["judge"]["provider"],
        "gen_model": cfg.get("gen", {}).get("model"),
        "judge_model": cfg["judge"]["judge_model"],
    }
    if cfg["judge"]["provider"] == "split_local_gen":
        print("  sinh  = %s   (LOCAL, GPU — khong ton tien, ton GIO)"
              % cfg["gen"]["model"])
        print("  judge = %s   (API — can logprobs that)"
              % cfg["judge"]["judge_model"])
        if "reasoner" in str(cfg["judge"]["judge_model"]).lower():
            verdict(ASK, "judge_model co chu 'reasoner' -> thinking mode KHONG\n"
                         "           ho tro logprobs, va no KHONG BAO LOI.\n"
                         "           Doi sang deepseek-chat.")

    if cfg["judge"]["judge_model"] == "PHAI_DIEN":
        verdict(ASK, "config.judge.gen_model / judge_model con la PHAI_DIEN.\n"
                     "           Day la quyet dinh #2 cua ke-hoach, va no quyet\n"
                     "           cost(S) cung toan bo bang chi phi.  Dien roi chay lai.")
        rep["status"] = "cho quyet dinh #2"
        write_report(cfg, "s0e_judge", rep)
        return rep

    texts = texts or {
        "d_award": "Forrest Gump won the Academy Award for Best Picture for 1994.",
        "d_dir":   "Forrest Gump was directed by Robert Zemeckis.",
        "d_dup":   "The 1994 Best Picture award went to Forrest Gump.",
        "d_bait":  "Forrest Gump was a cultural phenomenon of the 1990s.",
    }
    tel = Telemetry()
    try:
        j = Judge(cfg, tel)
    except Exception as e:
        verdict(MISSING, "khong khoi tao duoc judge: %s" % repr(e)[:180])
        rep["error"] = repr(e)[:300]
        write_report(cfg, "s0e_judge", rep)
        return rep

    # cac tap con dung chung TIEN TO — dung dang vong 1 greedy sinh ra
    base = ["d_award"]
    probes = [[], base, base + ["d_dir"], base + ["d_dup"], base + ["d_bait"],
              ["d_bait"]]

    rep["aliases_used"] = list(aliases or [])
    rep["use_answer_aliases"] = cfg["judge"].get("use_answer_aliases", True)
    rows = []
    t0 = time.time()
    for S in probes:
        r = j.v(q, y, S, texts, step="A1", aliases=aliases)
        rows.append({"S": sorted(S), "v": round(r.v, 4),
                     "answer": r.answer[:80],
                     "cost_est": r.cost_est, "cost_real": r.cost_real,
                     "cached": r.cached})
        print("     v(%-28s) = %-7.4f  est=%-6d real=%-6d  %s"
              % (",".join(sorted(S)) or "rong", r.v, r.cost_est, r.cost_real,
                 "cache" if r.cached else ""))
    rep["probes"] = rows
    rep["wall_seconds"] = round(time.time() - t0, 1)
    rep["telemetry"] = tel.as_dict()

    # ── 1. v co lien tuc khong ────────────────────────────────────
    vals = [r["v"] for r in rows if r["v"] == r["v"]]
    uniq = len(set(round(v, 3) for v in vals))
    rep["distinct_v_values"] = uniq
    if uniq >= max(3, len(vals) - 1):
        verdict(OK, "v lien tuc: %d gia tri phan biet tren %d tap con"
                % (uniq, len(vals)))
    else:
        verdict(ASK, "v chi co %d gia tri phan biet tren %d tap con.\n"
                     "           Neu judge dang tra thang roi rac thi 'toi tieu'\n"
                     "           mat nghia — kiem lai endpoint co tra logprobs that."
                % (uniq, len(vals)))

    # ── 2. bay moi tham so co bi bat khong ────────────────────────
    v_bait = next((r["v"] for r in rows if r["S"] == ["d_bait"]), None)
    v_pair = next((r["v"] for r in rows
                   if r["S"] == sorted(["d_award", "d_dir"])), None)
    rep["bait_vs_pair"] = {"v_bait": v_bait, "v_pair": v_pair}
    if v_bait is not None and v_pair is not None:
        if v_bait > 0.5:
            verdict(ASK, "v({d_bait}) = %.3f CAO du doan nay khong chua du kien.\n"
                         "           Day dung la ca MOI THAM SO ma §3.2 noi — va no\n"
                         "           xac nhan adq la bat buoc, khong phai tuy chon."
                    % v_bait)
        else:
            verdict(OK, "v({d_bait}) = %.3f thap  ->  mau nay khong kich moi tham so"
                    % v_bait)

    # ── 3. prompt caching co AN khong ─────────────────────────────
    _al = _aliases_line(aliases, cfg["judge"].get("use_answer_aliases", True))
    cached_tok = sum(
        (j.cache.get(cache_key(
            "%s|%s" % (cfg.get("gen", {}).get("model") or "?",
                       cfg["judge"]["judge_model"]),
            GEN_TMPL + JUDGE_TMPL + q + y + _al,
            [texts[d] for d in sorted(S)])) or {}).get("cached_prompt_tokens", 0)
        for S in probes)
    rep["cached_prompt_tokens_total"] = cached_tok
    est = sum(r["cost_est"] for r in rows)
    real = sum(r["cost_real"] for r in rows)
    rep["cost_est_total"] = est
    rep["cost_real_total"] = real
    if real:
        ratio = real / max(1, est)
        rep["real_over_est"] = round(ratio, 3)
        print("\n     tong est = %d   tong real = %d   real/est = %.2f"
              % (est, real, ratio))
        if cfg["judge"]["provider"] == "split_local_gen":
            verdict(OK,
                    "sinh chay LOCAL  ->  prompt caching cua API khong ap dung.\n"
                    "           Phan tiet kiem tien to chung gio do vLLM prefix\n"
                    "           caching lo, va no la THONG LUONG chu khong phai tien.")
        elif cached_tok == 0:
            verdict(ASK,
                    "cached_prompt_tokens = 0  ->  prompt caching CHUA an.\n"
                    "           cost(S) = overhead + tong len(k) van dung cho ke hoach,\n"
                    "           nhung phan tiet kiem nho tien to chung thi KHONG co.\n"
                    "           Kiem: endpoint co ho tro caching? tien to co du dai?\n"
                    "           (nhieu endpoint doi >=1024 token moi cache)")
        else:
            verdict(OK, "prompt caching AN: %d token duoc tai dung" % cached_tok)
        if ratio > 1.3 or ratio < 0.7:
            verdict(ASK, "real/est = %.2f lech >30%%  ->  cap nhat lai "
                         "prompt_overhead_tokens trong config" % ratio)

    # ── 4. aliases co thuc su can khong — do, dung doan ──────────
    if aliases:
        base = ["d_award", "d_dir"]
        r_with = j.v(q, y, base, texts, step="A1", aliases=aliases)
        r_without = j.v(q, y, base, texts, step="A1", aliases=None)
        rep["aliases_effect"] = {"v_with": round(r_with.v, 4),
                                 "v_without": round(r_without.v, 4),
                                 "delta": round(r_with.v - r_without.v, 4)}
        d = rep["aliases_effect"]["delta"]
        print("\n     v voi aliases = %.4f   ·   khong aliases = %.4f   ·   lech %+.4f"
              % (r_with.v, r_without.v, d))
        if abs(d) >= 0.05:
            verdict(ASK, "aliases doi v %+.3f  ->  BAT BUOC phai nhet aliases vao\n"
                         "           prompt judge, khong thi tap DUNG bi loai oan." % d)
        else:
            verdict(OK, "aliases doi v khong dang ke (%+.3f) tren mau nay — nhung\n"
                        "           van giu, vi mau nay model tra dung dang gold san." % d)

    p = write_report(cfg, "s0e_judge", rep)
    print("\n  -> %s" % p)
    return rep


if __name__ == "__main__":
    run(load_config())
