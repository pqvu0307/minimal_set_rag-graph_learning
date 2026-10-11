# -*- coding: utf-8 -*-
"""0.C / 0.D — Pool ung vien, va rerank THEO TUNG NGUON.

Ba nguon tren DUNG mot tap node, nen tinh dong nhat cach cat la dung theo
CAU TRUC — moi doan trong pool deu la mot node vi no lay tu do thi ra.
Tien kiem §6.1 ("doan lay ve khong co node nao trong do thi") khong the fail.

Han ngach RIENG moi nguon, khong gop chung: reranker chia se diem mu voi
retriever, gop chung thi doan ma BM25 loi ve dung VI GFM khong thay se bi loai.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from s0_common import (ASK, MISSING, OK, TokenCounter, banner, load_config,
                       resolve, verdict, write_report)
from s0b_chunks import iter_nodes, node_text, node_type


# ── kho van ban: chi node o lop van ban ─────────────────────────────
@dataclass
class DocStore:
    ids: List[str]
    texts: List[str]
    types: List[str]
    lens: List[int]
    index_of: Dict[str, int]

    def __len__(self) -> int:
        return len(self.ids)

    def text(self, doc_id: str) -> str:
        return self.texts[self.index_of[doc_id]]

    def length(self, doc_id: str) -> int:
        return self.lens[self.index_of[doc_id]]


def build_docstore(cfg: Dict[str, Any], limit: Optional[int] = None) -> DocStore:
    want = set(cfg["chunks"]["node_types_in_pool"])
    tc = TokenCounter()
    ids: List[str] = []; texts: List[str] = []; types: List[str] = []
    for row in iter_nodes(resolve(cfg, "graph_dir")):
        t = node_type(row)
        if t not in want:
            continue
        txt = node_text(row)
        if not txt:
            continue
        nid = row.get("name") or row.get("id") or f"n{len(ids)}"
        ids.append(nid); texts.append(txt); types.append(t)
        if limit and len(ids) >= limit:
            break
    return DocStore(ids=ids, texts=texts, types=types,
                    lens=tc.count_many(texts),
                    index_of={d: i for i, d in enumerate(ids)})


# ── nguon 1: BM25 ───────────────────────────────────────────────────
class BM25Source:
    """bm25s neu co (nhanh, chay duoc o 3M doc), khong thi rank_bm25."""

    name = "bm25"

    def __init__(self, store: DocStore, cfg: Dict[str, Any]):
        self.store = store
        self.impl = None
        k1 = cfg["pool"]["bm25"]["k1"]; b = cfg["pool"]["bm25"]["b"]
        try:
            import bm25s
            self._tok = bm25s.tokenize(store.texts, show_progress=False)
            self._r = bm25s.BM25(k1=k1, b=b)
            self._r.index(self._tok, show_progress=False)
            self.impl = "bm25s"
        except Exception:
            try:
                from rank_bm25 import BM25Okapi
                self._r = BM25Okapi([t.lower().split() for t in store.texts],
                                    k1=k1, b=b)
                self.impl = "rank_bm25"
            except Exception:
                self.impl = None

    def retrieve(self, query: str, k: int) -> List[Tuple[str, float]]:
        if self.impl == "bm25s":
            import bm25s
            q = bm25s.tokenize([query], show_progress=False)
            idx, sc = self._r.retrieve(q, k=min(k, len(self.store)),
                                       show_progress=False)
            return [(self.store.ids[int(i)], float(s))
                    for i, s in zip(idx[0], sc[0])]
        if self.impl == "rank_bm25":
            sc = self._r.get_scores(query.lower().split())
            order = sorted(range(len(sc)), key=lambda i: -sc[i])[:k]
            return [(self.store.ids[i], float(sc[i])) for i in order]
        return []


# ── nguon 2: GFM (G-reasoner), checkpoint pretrain, KHONG sft ───────
class GFMSource:
    name = "gfm"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.retriever = None
        self.error: Optional[str] = None
        if cfg["artifacts"].get("run_sft"):
            self.error = ("config dat run_sft=true — ke-hoach §Da chot noi KHONG "
                          "fine-tune, vi pool phai doc lap voi tin hieu dang dao")
            return
        try:
            from gfmrag import GFMRetriever          # type: ignore
            self.retriever = GFMRetriever.from_index(
                index_path=str(resolve(cfg, "graph_dir")),
                model_path=cfg["artifacts"]["checkpoint"])
        except Exception as e:
            self.error = repr(e)[:200]

    def retrieve(self, query: str, k: int) -> List[Tuple[str, float]]:
        if self.retriever is None:
            return []
        docs = self.retriever.retrieve(query, top_k=k)
        out = []
        for d in docs:
            did = getattr(d, "title", None) or getattr(d, "id", None) or str(d)
            sc = float(getattr(d, "score", 0.0))
            out.append((did, sc))
        return out


# ── nguon 3: boc ngau nhien tren DUNG tap node do ───────────────────
class RandomSource:
    name = "random"

    def __init__(self, store: DocStore, seed: int):
        self.store = store; self.rnd = random.Random(seed)

    def retrieve(self, query: str, k: int) -> List[Tuple[str, float]]:
        n = len(self.store)
        if n == 0:
            return []
        picks = self.rnd.sample(range(n), min(k, n))
        return [(self.store.ids[i], 0.0) for i in picks]


# ── 0.D rerank theo tung nguon, han ngach rieng ─────────────────────
class Reranker:
    def __init__(self, cfg: Dict[str, Any]):
        self.model = None
        self.error: Optional[str] = None
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(cfg["pool"]["rerank"]["model"])
            self.batch = cfg["pool"]["rerank"]["batch_size"]
        except Exception as e:
            self.error = repr(e)[:200]

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        if self.model is None or not texts:
            return [0.0] * len(texts)
        pairs = [(query, t) for t in texts]
        return [float(s) for s in self.model.predict(
            pairs, batch_size=self.batch, show_progress_bar=False)]


def build_pool(query: str, store: DocStore, sources: Dict[str, Any],
               reranker: Reranker, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """K <= 60 tho -> khu trung lap -> rerank THEO NGUON -> n = 40."""
    quotas: Dict[str, int] = dict(cfg["pool"]["quotas"])
    k_raw = cfg["pool"]["k_raw_max"]
    per_src_raw = max(quotas.values()) * 2

    raw: Dict[str, List[Tuple[str, float]]] = {}
    for name, src in sources.items():
        raw[name] = src.retrieve(query, min(per_src_raw, k_raw))

    # khu trung lap: mot doan chi thuoc nguon ĐAU TIEN tim ra no,
    # theo thu tu uu tien gfm -> bm25 -> random, de han ngach khong bi
    # mot nguon an sang nguon khac
    seen: set = set()
    kept: Dict[str, List[str]] = {k: [] for k in sources}
    for name in ("gfm", "bm25", "random"):
        for did, _ in raw.get(name, []):
            if did in seen or did not in store.index_of:
                continue
            seen.add(did); kept[name].append(did)

    pool: List[Dict[str, Any]] = []
    for name, quota in quotas.items():
        cand = kept.get(name, [])
        if not cand:
            continue
        sc = reranker.score(query, [store.text(d) for d in cand])
        order = sorted(range(len(cand)), key=lambda i: -sc[i])[:quota]
        for rank, i in enumerate(order):
            pool.append({"id": cand[i], "source": name, "rerank": sc[i],
                         "rank_in_source": rank,
                         "type": store.types[store.index_of[cand[i]]],
                         "tokens": store.length(cand[i])})

    return {
        "query": query,
        "n": len(pool),
        "pool": pool,
        "raw_counts": {k: len(v) for k, v in raw.items()},
        "kept_after_dedup": {k: len(v) for k, v in kept.items()},
        "quota_filled": {q: sum(1 for p in pool if p["source"] == q)
                         for q in quotas},
        "pool_tokens": sum(p["tokens"] for p in pool),
    }


def run(cfg: Dict[str, Any], probe_queries: Optional[List[str]] = None,
        store_limit: Optional[int] = None) -> Dict[str, Any]:
    banner("0.C / 0.D  Pool da nguon va rerank theo nguon")

    store = build_docstore(cfg, limit=store_limit)
    rep: Dict[str, Any] = {
        "docstore_size": len(store),
        "store_limit": store_limit,
        "types": {t: store.types.count(t) for t in set(store.types)},
    }
    if not len(store):
        verdict(MISSING, "docstore rong  ->  chay 0.A roi 0.B truoc")
        write_report(cfg, "s0c_pool", rep)
        return rep
    verdict(OK, "docstore %d node o lop van ban  %s" % (len(store), rep["types"]))

    bm = BM25Source(store, cfg); rep["bm25_impl"] = bm.impl
    if bm.impl is None:
        verdict(MISSING, "khong co bm25s lan rank_bm25:  pip install bm25s")
    else:
        verdict(OK, "BM25 = %s" % bm.impl)

    gfm = GFMSource(cfg); rep["gfm_error"] = gfm.error
    if gfm.retriever is None:
        verdict(MISSING, "GFM chua nap duoc: %s\n"
                         "           pip install gfmrag  (hoac chay trong repo da clone)"
                % gfm.error)
    else:
        verdict(OK, "GFM = %s, KHONG sft" % cfg["artifacts"]["checkpoint"])

    rr = Reranker(cfg); rep["rerank_error"] = rr.error
    if rr.model is None:
        verdict(MISSING, "khong nap duoc reranker: %s\n"
                         "           pip install sentence-transformers" % rr.error)
    else:
        verdict(OK, "reranker = %s" % cfg["pool"]["rerank"]["model"])

    sources: Dict[str, Any] = {}
    if bm.impl:
        sources["bm25"] = bm
    if gfm.retriever is not None:
        sources["gfm"] = gfm
    sources["random"] = RandomSource(store, cfg["splits"]["seed"])

    probes = probe_queries or [
        "Who directed the film that won Best Picture for 1994?",
        "What is the capital of the country where the Eiffel Tower is located?",
    ]
    samples = []
    for q in probes:
        samples.append(build_pool(q, store, sources, rr, cfg))
    rep["samples"] = samples

    for s in samples:
        print("\n  q: %s" % s["query"][:66])
        print("     tho       %s" % s["raw_counts"])
        print("     sau dedup %s" % s["kept_after_dedup"])
        print("     han ngach %s   n=%d   %d token"
              % (s["quota_filled"], s["n"], s["pool_tokens"]))

    want_n = cfg["pool"]["n_final"]
    short = [s for s in samples if s["n"] < want_n]
    if short:
        verdict(ASK, "pool chua day n=%d o %d/%d truy van thu.\n"
                     "           Thieu nguon nao thi han ngach cua no bi bo trong —\n"
                     "           §3.1 noi khong duoc don cho du, nen day la BAO CAO,\n"
                     "           khong phai loi."
                % (want_n, len(short), len(samples)))
    else:
        verdict(OK, "pool day n=%d o moi truy van thu" % want_n)

    p = write_report(cfg, "s0c_pool", rep)
    print("\n  -> %s" % p)
    return rep


if __name__ == "__main__":
    run(load_config())
