# -*- coding: utf-8 -*-
"""Day file tu may nay sang Colab qua daemon — khong can clone repo.

Repo dang private nen `git clone` tren Colab tra ve 128.  Thay vi cap token
hay de repo public, chi day DUNG nhung file can cho phep do.

    python build/colab_sync.py stage0/s0_common.py stage0/config.yaml ...
"""
from __future__ import annotations

import base64
import json
import pathlib
import socket
import sys
from typing import List

HOST, PORT = "127.0.0.1", 8787
ROOT = "/content/ragproj"


def send(code: str) -> str:
    c = socket.create_connection((HOST, PORT), timeout=1800)
    c.sendall((json.dumps({"op": "run", "code": code}) + "\n").encode("utf-8"))
    buf = []
    while True:
        b = c.recv(65536)
        if not b:
            break
        buf.append(b)
        if b.endswith(b"\n"):
            break
    c.close()
    return json.loads(b"".join(buf).decode("utf-8")).get("out", "")


def pull(paths: List[str]) -> int:
    """Keo file TU Colab ve may — ket qua chay phai giu lai, khong thi mat
    het khi runtime tat."""
    code = (
        "import base64, json, pathlib\n"
        "out = {}\n"
        "for rel in %r:\n"
        "    p = pathlib.Path(%r) / rel\n"
        "    out[rel] = base64.b64encode(p.read_bytes()).decode() if p.exists() else None\n"
        "print('@@' + base64.b64encode(json.dumps(out).encode()).decode())\n"
        % (paths, ROOT))
    res = send(code)
    tag = res.find("@@")
    if tag < 0:
        print(res[:500]); return 1
    blob = json.loads(base64.b64decode(res[tag + 2:].split()[0]).decode())
    for rel, b64 in blob.items():
        if b64 is None:
            print("  (khong co tren Colab) %s" % rel); continue
        p = pathlib.Path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(base64.b64decode(b64))
        print("  ve may  %7d B  %s" % (p.stat().st_size, rel))
    return 0


def main() -> int:
    if sys.argv[1:2] == ["pull"]:
        return pull(sys.argv[2:])
    files = sys.argv[1:]
    if not files:
        print("can it nhat mot file"); return 1
    blob = {}
    for f in files:
        p = pathlib.Path(f)
        blob[str(p).replace("\\", "/")] = base64.b64encode(
            p.read_bytes()).decode("ascii")
    payload = base64.b64encode(json.dumps(blob).encode("utf-8")).decode("ascii")

    code = (
        "import base64, json, pathlib, os\n"
        "ROOT = pathlib.Path(%r)\n"
        "blob = json.loads(base64.b64decode(%r).decode())\n"
        "for rel, b64 in blob.items():\n"
        "    p = ROOT / rel\n"
        "    p.parent.mkdir(parents=True, exist_ok=True)\n"
        "    p.write_bytes(base64.b64decode(b64))\n"
        "    print('  %%7d B  %%s' %% (p.stat().st_size, rel))\n"
        "os.chdir(ROOT)\n"
        "print('cwd', os.getcwd())\n" % (ROOT, payload))
    print(send(code))
    return 0


if __name__ == "__main__":
    sys.exit(main())
