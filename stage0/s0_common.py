# -*- coding: utf-8 -*-
"""Giai doan 0 — tien ich dung chung: cau hinh, duong dan, dem token, bao cao.

Moi module s0*.py ghi mot file JSON vao reports/.  Khong module nao im lang:
thieu hien vat thi bao THIEU va noi phai lam gi, khong crash.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import yaml
except ImportError:                                    # pragma: no cover
    yaml = None

HERE = Path(__file__).resolve().parent


# ── cau hinh ────────────────────────────────────────────────────────
def load_config(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    p = Path(path) if path else HERE / "config.yaml"
    if yaml is None:
        raise RuntimeError("thieu pyyaml:  pip install pyyaml")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(p)
    return cfg


def resolve(cfg: Dict[str, Any], key: str) -> Path:
    """Duong dan tuyet doi theo paths.root."""
    root = Path(cfg["paths"]["root"]).expanduser().resolve()
    return (root / cfg["paths"][key]).resolve()


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    for k in ("reports", "cache"):
        resolve(cfg, k).mkdir(parents=True, exist_ok=True)


def active_datasets(cfg: Dict[str, Any]) -> List[str]:
    """Dataset dang dung, theo artifacts.mode.

    mode = test   ->  index CO SAN (tai tu OneDrive), $0 indexing
    mode = train  ->  index KHONG co, phai tu dung bang Stage 1 scripts
    """
    a = cfg["artifacts"]
    return list(a["test_datasets"] if a.get("mode", "test") == "test"
                else a["train_datasets"])


def openie_cost(cfg: Dict[str, Any], n_docs: int) -> Dict[str, Any]:
    """Uoc tien OpenIE neu phai tu dung index.

    GFM-RAG dung LLM lam cong cu OpenIE tren TUNG document, nen chi phi ty le
    voi so document — day la dong chi phi lon nhat cua chuong trinh khi mode=train.
    """
    o = cfg["openie"]
    usd = (n_docs * o["prompt_tokens_per_doc"] / 1e6 * o["price_in_per_m"]
           + n_docs * o["output_tokens_per_doc"] / 1e6 * o["price_out_per_m"])
    return {"n_docs": n_docs, "usd": round(usd, 0),
            "usd_per_doc": round(usd / max(1, n_docs), 6),
            "assumes": {k: o[k] for k in
                        ("model", "prompt_tokens_per_doc",
                         "output_tokens_per_doc", "price_in_per_m",
                         "price_out_per_m")}}


# ── dem token ───────────────────────────────────────────────────────
class TokenCounter:
    """tiktoken neu co, khong thi xap xi theo tu.

    Xap xi la 1.3 token moi tu — du cho 0.B do PHAN BO, nhung bang chi phi
    cuoi cung phai dung tiktoken cua dung model judge.
    """

    def __init__(self, encoding: str = "cl100k_base"):
        self.exact = False
        self._enc = None
        try:
            import tiktoken
            self._enc = tiktoken.get_encoding(encoding)
            self.exact = True
        except Exception:
            pass

    def count(self, text: str) -> int:
        if self._enc is not None:
            return len(self._enc.encode(text))
        return int(len(text.split()) * 1.3) + 1

    def count_many(self, texts: Iterable[str]) -> List[int]:
        return [self.count(t) for t in texts]


# ── chi phi mot luot judge ──────────────────────────────────────────
def cost_tokens(chunk_lens: Iterable[int], overhead: int = 500) -> int:
    """cost(S) = overhead + tong do dai THAT cua cac chunk trong S.

    Doi tu `500 + 250*|S|` vi document (~100) va summary (~600) la hai lop
    do dai khac nhau — xem ke-hoach 0.B.
    """
    return overhead + int(sum(chunk_lens))


# ── telemetry: dem token theo TUNG BUOC, khong dem tong ─────────────
@dataclass
class StepTally:
    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    cached_hits: int = 0

    def add(self, prompt: int, output: int, cached: bool = False) -> None:
        self.calls += 1
        self.prompt_tokens += prompt
        self.output_tokens += output
        if cached:
            self.cached_hits += 1


@dataclass
class Telemetry:
    steps: Dict[str, StepTally] = field(default_factory=dict)

    def tally(self, step: str) -> StepTally:
        return self.steps.setdefault(step, StepTally())

    def as_dict(self) -> Dict[str, Any]:
        out = {k: asdict(v) for k, v in sorted(self.steps.items())}
        out["_total"] = {
            "calls": sum(v.calls for v in self.steps.values()),
            "prompt_tokens": sum(v.prompt_tokens for v in self.steps.values()),
            "output_tokens": sum(v.output_tokens for v in self.steps.values()),
            "cached_hits": sum(v.cached_hits for v in self.steps.values()),
        }
        return out


# ── cache: khoa bam NOI DUNG, khong bam ten ─────────────────────────
def cache_key(model: str, template: str, chunk_texts: List[str]) -> str:
    """Bam noi dung chunk da sap, khong bam ID.

    Bam ten thi chunk lai corpus se lam ten va nhau va cache tra sai.
    """
    h = hashlib.sha256()
    h.update(model.encode()); h.update(b"\x00")
    h.update(template.encode()); h.update(b"\x00")
    for t in sorted(chunk_texts):
        h.update(hashlib.sha256(t.encode()).digest())
    return h.hexdigest()


class DiskCache:
    def __init__(self, root: Path):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[Any]:
        p = self._p(key)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def put(self, key: str, value: Any) -> None:
        p = self._p(key); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


# ── bao cao ─────────────────────────────────────────────────────────
def write_report(cfg: Dict[str, Any], name: str, payload: Dict[str, Any]) -> Path:
    ensure_dirs(cfg)
    payload = dict(payload)
    payload["_meta"] = {
        "step": name,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": cfg.get("_config_path"),
    }
    p = resolve(cfg, "reports") / f"{name}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def read_report(cfg: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    p = resolve(cfg, "reports") / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# ── in ket qua: moi buoc tu bao PASS / THIEU / CAN NGUOI ────────────
OK, MISSING, ASK = "PASS", "THIEU", "CAN NGUOI"


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print("  " + title)
    print("=" * 72)


def verdict(status: str, msg: str) -> None:
    mark = {OK: "  [PASS] ", MISSING: "  [THIEU]", ASK: "  [HOI]  "}[status]
    print(mark + " " + msg)


def pct(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return float(s[i])


def describe(xs: List[int]) -> Dict[str, float]:
    if not xs:
        return {}
    return {
        "n": len(xs),
        "min": float(min(xs)),
        "p10": pct(xs, 0.10),
        "p50": pct(xs, 0.50),
        "mean": round(sum(xs) / len(xs), 1),
        "p90": pct(xs, 0.90),
        "p99": pct(xs, 0.99),
        "max": float(max(xs)),
    }
