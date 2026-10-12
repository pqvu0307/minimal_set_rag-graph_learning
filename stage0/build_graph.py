# -*- coding: utf-8 -*-
"""Dung do thi DUNG cau truc kg_constructor cua G-Reasoner, tu OpenIE da chay.

Doc tu RManLuo/gfm-rag, gfmrag/graph_index_construction/graph_constructors/
kg_constructor.py.  Cau truc that chi co HAI loai node:

    node  document   name = tieu de, attributes.content = noi dung
    node  entity     tu bo ba OpenIE (ca head lan tail)
    edge  head --quan he--> tail            tu bo ba
    edge  entity --is_mentioned_in--> doc   tu document2entities

KHONG co node type "summary" — ghi chu truoc do cua toi sai.
Cai goi la "nhieu tang" that ra la HAI LOP CANH tren hai loai node.

Truy hoi dung do thi nay lam NGUON THU BA cho pool:
    seed  = doan tu bm25
    -> thuc the cua seed
    -> doan khac cung nhac nhung thuc the do
    xep theo so thuc the chung, co trong so idf cua thuc the
Day la bieu dien "mot buoc" cua do thi, de so SONG SONG voi mo rong theo
lien ket tieu de (51.0%) va voi bm25 thuan (32.5%).

    python stage0/build_graph.py --queries=400
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import sys
from typing import Any, Dict, List, Sequence, Set, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

KG = pathlib.Path("artifacts/kg")


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
        out[r["i"]] = r          # ban ghi sau de len ban ghi truoc
    return out


def build(recs: Dict[int, Dict[str, Any]], titles: Sequence[str],
          texts: Sequence[str]) -> Dict[str, Any]:
    nodes: Dict[str, Dict[str, Any]] = {}
    rels: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, str]] = []
    doc2ent: Dict[int, Set[str]] = {}

    for i, t in enumerate(titles):
        key = "doc::%d" % i
        nodes[key] = {"name": t or ("doc%d" % i), "type": "document",
                      "attributes": {"content": texts[i][:400]}}

    for i, r in recs.items():
        if i >= len(titles):
            continue
        ents = set()
        for h, rel, ta in r.get("t", []):
            for e in (h, ta):
                e = norm(e)
                if e and e not in nodes:
                    nodes[e] = {"name": e, "type": "entity", "attributes": {}}
                if e:
                    ents.add(e)
            rl = norm(rel)
            if rl and rl not in rels:
                rels[rl] = {"name": rl, "attributes": {}}
            if norm(h) and norm(ta) and rl:
                edges.append({"source": norm(h), "relation": rl,
                              "target": norm(ta)})
        for e in (norm(x) for x in r.get("ents", [])):
            if e:
                if e not in nodes:
                    nodes[e] = {"name": e, "type": "entity", "attributes": {}}
                ents.add(e)
        doc2ent[i] = ents
        rels.setdefault("is_mentioned_in", {"name": "is_mentioned_in",
                                            "attributes": {}})
        for e in ents:
            edges.append({"source": e, "relation": "is_mentioned_in",
                          "target": "doc::%d" % i})
    return {"nodes": nodes, "relations": rels, "edges": edges,
            "doc2ent": doc2ent}


def expand(g: Dict[str, Any], seed: Sequence[int], k: int) -> List[int]:
    """Mot buoc tren do thi: doan -> thuc the -> doan khac.

    Trong so idf theo thuc the: thuc the xuat hien o moi doan (vd 'is')
    khong noi len dieu gi, nen phai ha trong so xuong.
    """
    d2e: Dict[int, Set[str]] = g["doc2ent"]
    e2d: Dict[str, List[int]] = collections.defaultdict(list)
    for d, es in d2e.items():
        for e in es:
            e2d[e].append(d)
    N = max(1, len(d2e))
    sc: Dict[int, float] = collections.defaultdict(float)
    sset = set(seed)
    for d in seed:
        for e in d2e.get(d, ()):
            w = math.log(1 + N / (1 + len(e2d[e])))
            for d2 in e2d[e]:
                if d2 not in sset:
                    sc[d2] += w
    return sorted(sc, key=lambda d: -sc[d])[:k]


def main() -> int:
    arg = {k.lstrip("-"): v for k, v in
           (a.split("=", 1) for a in sys.argv[1:] if "=" in a)}
    nq = int(arg.get("queries", 400))
    from pool_sources import load as load_titles, link_graph
    from build_pool import BM25
    items, titles, texts = load_titles(nq)
    recs = load_openie()
    print("== OpenIE co %d/%d doan ==" % (len(recs), len(titles)))
    g = build(recs, titles, texts)
    nd = sum(1 for v in g["nodes"].values() if v["type"] == "document")
    ne = sum(1 for v in g["nodes"].values() if v["type"] == "entity")
    print("   node: %d document + %d entity · %d quan he · %d canh"
          % (nd, ne, len(g["relations"]), len(g["edges"])))

    docs = ["%s. %s" % (a, b) for a, b in zip(titles, texts)]
    bm = BM25(docs)
    lg = link_graph(titles, texts)

    def rec(fn) -> float:
        ok = 0
        for it in items:
            if it["gold"] <= fn(it):
                ok += 1
        return 100 * ok / max(1, len(items))

    def f_bm(it, k=20):
        return set(bm.top(it["q"], k))

    def f_link(it, kb=12, kl=8):
        seed = bm.top(it["q"], kb)
        cnt: Dict[int, int] = collections.Counter()
        for j in seed:
            for kk in lg.get(j, []):
                if kk not in set(seed):
                    cnt[kk] += 1
        return set(seed) | {kk for kk, _ in cnt.most_common(kl)}

    def f_kg(it, kb=12, kl=8):
        seed = bm.top(it["q"], kb)
        return set(seed) | set(expand(g, seed, kl))

    def f_both(it, kb=12, kl=4, kk=4):
        seed = bm.top(it["q"], kb)
        s = set(seed)
        cnt: Dict[int, int] = collections.Counter()
        for j in seed:
            for x in lg.get(j, []):
                if x not in s:
                    cnt[x] += 1
        s |= {x for x, _ in cnt.most_common(kl)}
        s |= set(expand(g, seed, kk))
        return s

    print("\n" + "=" * 56)
    print("%-34s %10s" % ("nguon (pool 20 doan)", "TOAN BO"))
    for name, fn in (("bm25 20", f_bm),
                     ("bm25 12 + lien ket tieu de 8", f_link),
                     ("bm25 12 + DO THI KG 8", f_kg),
                     ("bm25 12 + lien ket 4 + KG 4", f_both)):
        print("%-34s %9.1f%%" % (name, rec(fn)))
    print("=" * 56)
    out = KG / "graph_musique_dev.json"
    out.write_text(json.dumps(
        {"n_doc": nd, "n_ent": ne, "n_rel": len(g["relations"]),
         "n_edge": len(g["edges"]), "openie_docs": len(recs)}, indent=1),
        encoding="utf-8")
    print("-> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
