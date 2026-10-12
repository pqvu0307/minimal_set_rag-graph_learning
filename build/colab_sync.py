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


def main() -> int:
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
