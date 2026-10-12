# -*- coding: utf-8 -*-
"""QuadGraph bon tang theo paper G-Reasoner (arXiv:2509.24276).

    G = (V, E, R, T, S)      T = {attribute, entity, document, community}
    canh xuyen tang: has_attribute · included_in · belongs_to

VI SAO PHAI TU DUNG: repo RManLuo/gfm-rag da phat hanh KHONG co file nao
cho attribute / community.  kg_constructor.py chi sinh document + entity +
is_mentioned_in.  Paper noi ro QuadGraph la LOP CHUAN HOA tren do thi dung
bang cong cu co san (HippoRAG 2, LightRAG, Youtu-GraphRAG) — tang community
von den tu kieu LightRAG, tang attribute tu bo trich xuat giau hon.  Voi
corpus cua ta thi khong co san, nen dung theo dinh nghia trong paper.

Phan tang tu OpenIE:
    document    moi doan van
    entity      chu the, va doi tuong NAO CO XUAT HIEN LAM CHU THE o dau do
                (tuc no duoc noi toi nhu mot thuc the, khong phai mot gia tri)
    attribute   doi tuong CHI xuat hien o vi tri doi tuong — ngay thang, so,
                cum mo ta.  Day la "thuoc tinh chung cua node" cua paper.
    community   lan truyen nhan tren do thi entity-entity.  Paper: "gom node
                theo tuong dong ngu nghia hoac lien thong cau truc de cung
                cap thong tin TOAN CUC".

    python stage0/build_quadgraph.py --queries=400
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import random
import re
import sys
from typing import Any, Dict, List, Sequence, Set, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

KG = pathlib.Path("artifacts/kg")
VALUE = re.compile(r"^(\d{1,4}([-/]\d{1,2})*|\d+(\.\d+)?\s*\w*|"
                   r"(january|february|march|april|may|june|july|august|"
                   r"september|october|november|december)\b.*)$", re.I)


def norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def load_openie() -> Dict[int, Dict[str, Any]]:
    f = KG / "musique_dev_openie.jsonl"
    out: Dict[int, Dict[str, Any]] = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        out[r["i"]] = r
    return out


def communities(adj: Dict[str, Set[str]], rounds: int = 8) -> Dict[str, int]:
    """Lan truyen nhan — moi node lay nhan pho bien nhat quanh no.

    Chon cach nay vi no khong can thu vien ngoai va khong can chon truoc so
    cum; paper chi noi "gom theo lien thong cau truc".
    """
    lab = {v: i for i, v in enumerate(adj)}
    order = list(adj)
    rng = random.Random(0)
    for _ in range(rounds):
        rng.shuffle(order)
        moved = 0
        for v in order:
            if not adj[v]:
                continue
            c = collections.Counter(lab[u] for u in adj[v] if u in lab)
            if not c:
                continue
            best = max(c.items(), key=lambda kv: (kv[1], -kv[0]))[0]
            if best != lab[v]:
                lab[v] = best; moved += 1
        if not moved:
            break
    return lab


def build(recs: Dict[int, Dict[str, Any]], titles: Sequence[str],
          texts: Sequence[str]) -> Dict[str, Any]:
    # vong 1: xem chuoi nao tung dung lam CHU THE -> do la thuc the
    subj: Set[str] = set()
    for r in recs.values():
        for h, _, _ in r.get("t", []):
            if norm(h):
                subj.add(norm(h))

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, str]] = []
    rels: Set[str] = set()
    doc2ent: Dict[int, Set[str]] = {}
    adj: Dict[str, Set[str]] = collections.defaultdict(set)

    for i, t in enumerate(titles):
        nodes["doc::%d" % i] = {"name": t or "doc%d" % i, "type": "document"}

    def kind(x: str) -> str:
        if x in subj:
            return "entity"
        return "attribute" if (VALUE.match(x) or len(x.split()) > 4) else "entity"

    for i, r in recs.items():
        if i >= len(titles):
            continue
        ents: Set[str] = set()
        for h, rel, ta in r.get("t", []):
            h, ta, rel = norm(h), norm(ta), norm(rel)
            if not (h and ta and rel):
                continue
            kh, kt = kind(h), kind(ta)
            for x, k in ((h, kh), (ta, kt)):
                nodes.setdefault(x, {"name": x, "type": k})
            if kt == "attribute":
                edges.append({"s": h, "r": "has_attribute", "t": ta})
                rels.add("has_attribute")
            else:
                edges.append({"s": h, "r": rel, "t": ta})
                rels.add(rel)
                adj[h].add(ta); adj[ta].add(h)
            if kh == "entity":
                ents.add(h)
            if kt == "entity":
                ents.add(ta)
        for e in (norm(x) for x in r.get("ents", [])):
            if e:
                nodes.setdefault(e, {"name": e, "type": kind(e)})
                if nodes[e]["type"] == "entity":
                    ents.add(e)
        doc2ent[i] = ents
        for e in ents:
            edges.append({"s": e, "r": "included_in", "t": "doc::%d" % i})
        rels.add("included_in")

    # tang community tren do thi entity-entity
    for v in list(nodes):
        if nodes[v]["type"] == "entity":
            adj.setdefault(v, set())
    lab = communities(adj)
    size = collections.Counter(lab.values())
    cid: Dict[int, str] = {}
    for v, l in lab.items():
        if size[l] < 2:
            continue
        c = cid.setdefault(l, "comm::%d" % len(cid))
        nodes.setdefault(c, {"name": c, "type": "community"})
        edges.append({"s": v, "r": "belongs_to", "t": c})
    rels.add("belongs_to")
    # tai lieu thuoc cong dong ma phan lon thuc the cua no thuoc ve
    for i, es in doc2ent.items():
        cc = collections.Counter(cid[lab[e]] for e in es
                                 if e in lab and lab[e] in cid)
        if cc:
            edges.append({"s": "doc::%d" % i, "r": "belongs_to",
                          "t": cc.most_common(1)[0][0]})
    return {"nodes": nodes, "edges": edges, "rels": rels,
            "doc2ent": doc2ent, "lab": lab, "cid": cid}


def main() -> int:
    arg = {k.lstrip("-"): v for k, v in
           (a.split("=", 1) for a in sys.argv[1:] if "=" in a)}
    nq = int(arg.get("queries", 400))
    from pool_sources import load as load_titles
    items, titles, texts = load_titles(nq)
    recs = load_openie()
    print("== OpenIE: %d / %d doan ==" % (len(recs), len(titles)))
    if len(recs) < 50:
        print("   qua it de dung do thi, doi OpenIE chay them"); return 0
    g = build(recs, titles, texts)
    c = collections.Counter(v["type"] for v in g["nodes"].values())
    print("   node theo TANG: %s" % dict(c))
    er = collections.Counter(e["r"] for e in g["edges"])
    print("   canh xuyen tang: has_attribute=%d included_in=%d belongs_to=%d"
          % (er["has_attribute"], er["included_in"], er["belongs_to"]))
    print("   quan he mien: %d loai, %d canh"
          % (len(g["rels"]) - 3,
             sum(v for k, v in er.items()
                 if k not in ("has_attribute", "included_in", "belongs_to"))))
    print("   cong dong: %d cum" % len(g["cid"]))
    out = KG / "quadgraph_stats.json"
    out.write_text(json.dumps({"nodes": dict(c), "edges": dict(er),
                               "n_comm": len(g["cid"]),
                               "openie_docs": len(recs)}, indent=1),
                   encoding="utf-8")
    print("-> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
