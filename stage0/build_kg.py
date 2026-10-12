# -*- coding: utf-8 -*-
"""Dung KG bang DUNG pipeline OpenIE cua G-Reasoner, chay qua DeepInfra.

Lay nguyen prompt tu RManLuo/gfm-rag:
    gfmrag/graph_index_construction/openie_extraction_instructions.py
Dung prompt cua rieng minh thi KG se KHONG tuong thich voi checkpoint cua ho.

HAI BUOC moi tai lieu, moi buoc mot one-shot:
    1  NER        -> {"named_entities": [...]}
    2  post-NER   -> {"triples": [[chu the, quan he, doi tuong], ...]}
G-Reasoner mac dinh dung gpt-4o-mini; o day doi sang DeepSeek-V4-Flash qua
DeepInfra, va GHI RO vi do la mot khac biet voi ban goc.

Gia thuc do: $0.10/M vao, $0.15/M ra.  MuSiQue dev 21.100 doan x 2 luot
~ $2-4, khong phai $800 nhu toi uoc luc lap ke hoach.

CHAY QUA DEM nen phai chiu duoc dut:
  * ghi NOI TIEP ra .jsonl, khong giu trong RAM
  * chay lai thi bo qua doan da xong
  * 429 thi lui dan — tran do duoc ~4.7 luot/giay

    python stage0/build_kg.py --n=0 --workers=12
"""
from __future__ import annotations

import json
import os
import pathlib
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

RAW = pathlib.Path("artifacts/raw")
OUT = pathlib.Path("artifacts/kg")

# ── nguyen van tu openie_extraction_instructions.py ────────────────
ONE_SHOT_PASSAGE = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""

ONE_SHOT_ENTITIES = """{"named_entities":
    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}
"""

ONE_SHOT_TRIPLES = """{"triples": [
            ["Radio City", "located in", "India"],
            ["Radio City", "is", "private FM radio station"],
            ["Radio City", "started on", "3 July 2001"],
            ["Radio City", "plays songs in", "Hindi"],
            ["Radio City", "plays songs in", "English"]
            ["Radio City", "forayed into", "New Media"],
            ["Radio City", "launched", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "launched in", "May 2008"],
            ["PlanetRadiocity.com", "is", "music portal"],
            ["PlanetRadiocity.com", "offers", "news"],
            ["PlanetRadiocity.com", "offers", "videos"],
            ["PlanetRadiocity.com", "offers", "songs"]
    ]
}
"""

NER_SYS = """Your task is to extract named entities from the given paragraph.
Respond with a JSON list of entities.
Strictly follow the required JSON format.
"""
NER_FRAME = "Paragraph:```\n{p}\n```"

OIE_SYS = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists.
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph.

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.

"""
OIE_FRAME = """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{p}
```

{ents}
"""


def ner_msgs(p: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": NER_SYS},
            {"role": "user", "content": NER_FRAME.format(p=ONE_SHOT_PASSAGE)},
            {"role": "assistant", "content": ONE_SHOT_ENTITIES},
            {"role": "user", "content": NER_FRAME.format(p=p)}]


def oie_msgs(p: str, ents_json: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": OIE_SYS},
            {"role": "user", "content": OIE_FRAME.format(p=ONE_SHOT_PASSAGE,
                                                         ents=ONE_SHOT_ENTITIES)},
            {"role": "assistant", "content": ONE_SHOT_TRIPLES},
            {"role": "user", "content": OIE_FRAME.format(p=p, ents=ents_json)}]


class Limiter:
    def __init__(self):
        self.lock = threading.Lock()
        self.delay = 0.0
        self.n429 = 0

    def wait(self):
        if self.delay:
            time.sleep(self.delay * (0.5 + random.random()))

    def hit(self):
        with self.lock:
            self.n429 += 1
            self.delay = min(8.0, (self.delay or 0.25) * 1.6)

    def ok(self):
        with self.lock:
            self.delay = max(0.0, self.delay * 0.93)


def corpus_musique(nq: int = 0) -> List[Tuple[str, str]]:
    """Corpus = doan cua nq truy van dau (0 = het).

    Index ca bo dev la 21.100 doan; ta chi danh gia tren vai tram truy van,
    nen chi can corpus cua chung — it hon ~4 lan, va la DUNG corpus ma pool
    rut ra tu do."""
    f = RAW / "musique" / "dev.jsonl"
    seen: Dict[str, str] = {}
    lines = f.read_text(encoding="utf-8").splitlines()
    if nq:
        lines = lines[:nq]
    for line in lines:
        r = json.loads(line)
        for p in r.get("paragraphs", []):
            t = (p.get("paragraph_text") or "").strip()
            if t and t not in seen:
                seen[t] = (p.get("title") or "").strip()
    return [(v, k) for k, v in seen.items()]


def jload(txt: str, key: str) -> List[Any]:
    a, b = txt.find("{"), txt.rfind("}")
    if a < 0 or b <= a:
        return []
    try:
        return json.loads(txt[a:b + 1]).get(key, []) or []
    except Exception:
        # mo hinh hay bo quen dau phay giua cac phan tu (chinh one-shot cua
        # ho cung thieu mot dau phay) -> vot bang cach vá tho roi thu lai
        try:
            fixed = txt[a:b + 1].replace("]\n", "],\n").replace("],,", "],")
            return json.loads(fixed).get(key, []) or []
        except Exception:
            return []


def main() -> int:
    arg = {k.lstrip("-"): v for k, v in
           (a.split("=", 1) for a in sys.argv[1:] if "=" in a)}
    n = int(arg.get("n", 0))
    W = int(arg.get("workers", 12))
    MODEL = arg.get("model", "deepseek-ai/DeepSeek-V4-Flash")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "musique_dev_openie.jsonl"

    docs = corpus_musique(int(arg.get("queries", 0)))
    if n:
        docs = docs[:n]
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["i"])
            except Exception:
                pass
    todo = [(i, t, b) for i, (t, b) in enumerate(docs) if i not in done]
    print("== OpenIE kieu G-Reasoner · %d doan, xong %d, con %d · %d luong =="
          % (len(docs), len(done), len(todo), W), flush=True)
    print("   model %s  (ban goc dung gpt-4o-mini)" % MODEL, flush=True)
    if not todo:
        print("khong con gi de lam"); return 0

    from openai import OpenAI
    cli = OpenAI(api_key=os.environ["DEEPINFRA_API_KEY"],
                 base_url="https://api.deepinfra.com/v1/openai",
                 max_retries=0, timeout=120)
    lim = Limiter()
    lock = threading.Lock()
    fh = path.open("a", encoding="utf-8")
    st = {"ok": 0, "fail": 0, "ent": 0, "trip": 0, "cost": 0.0}
    t0 = time.time()

    def ask(msgs, maxtok):
        for attempt in range(6):
            lim.wait()
            try:
                r = cli.chat.completions.create(model=MODEL, temperature=0.0,
                                                max_tokens=maxtok, messages=msgs)
                lim.ok()
                return ((r.choices[0].message.content or ""),
                        getattr(r.usage, "estimated_cost", 0.0) or 0.0)
            except Exception as e:
                s = str(e)
                if "429" in s or "rate" in s.lower():
                    lim.hit(); time.sleep(1.5 * (attempt + 1)); continue
                time.sleep(1.0 * (attempt + 1))
        return "", 0.0

    def one(job):
        i, title, body = job
        p = ("%s\n%s" % (title, body)) if title else body
        p = p[:4000]
        t1, c1 = ask(ner_msgs(p), 300)
        ents = [str(x).strip() for x in jload(t1, "named_entities") if str(x).strip()]
        ej = json.dumps({"named_entities": ents})
        t2, c2 = ask(oie_msgs(p, ej), 1200)
        trips = [[str(x).strip()[:160] for x in t]
                 for t in jload(t2, "triples")
                 if isinstance(t, (list, tuple)) and len(t) == 3]
        return {"i": i, "title": title, "ents": ents, "t": trips,
                "cost": c1 + c2, "fail": 0 if (ents or trips) else 1}

    with ThreadPoolExecutor(max_workers=W) as ex:
        for k, res in enumerate(ex.map(one, todo), 1):
            with lock:
                fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                if k % 50 == 0:
                    fh.flush()
                st["fail" if res["fail"] else "ok"] += 1
                st["ent"] += len(res["ents"]); st["trip"] += len(res["t"])
                st["cost"] += res["cost"]
            if k % 250 == 0 or k == len(todo):
                el = time.time() - t0
                print("   %6d/%d  %5.0fs  %.1f doan/s  %d ent  %d bo ba  "
                      "$%.3f  429=%d delay=%.2f  loi=%d"
                      % (k, len(todo), el, k / max(1e-9, el), st["ent"],
                         st["trip"], st["cost"], lim.n429, lim.delay, st["fail"]),
                      flush=True)
    fh.close()
    print("\nxong: ok=%d fail=%d · %d thuc the · %d bo ba · $%.3f · %.0fs"
          % (st["ok"], st["fail"], st["ent"], st["trip"], st["cost"],
             time.time() - t0))
    print("-> %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
