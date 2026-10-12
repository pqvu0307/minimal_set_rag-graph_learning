# -*- coding: utf-8 -*-
"""Chay viec DAI tren Colab ma khong de cell nao chay lau.

Vi sao: RPC cua ColabMCP co timeout (dat 1800s).  Mot cell chay 90 phut thi
RPC bo cuoc va mat ca me — da mat hai lan vi dung chuyen do.

Cach dung dung: cell chi PHONG mot tien trinh roi tra ve ngay; tien trinh
ghi ra log; sau do doc log bang nhung cell ngan.  Khong cell nao chay qua
vai giay, nen khong co gi de timeout.

    python build/colab_bg.py start <file.py> [ten]
    python build/colab_bg.py tail  <ten> [so_dong]
    python build/colab_bg.py ps
"""
from __future__ import annotations

import json
import pathlib
import socket
import sys

HOST, PORT = "127.0.0.1", 8787
LOGDIR = "/content/logs"


def send(code: str, timeout: float = 300) -> str:
    c = socket.create_connection((HOST, PORT), timeout=timeout)
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


def start(path: str, name: str) -> int:
    src = pathlib.Path(path).read_text(encoding="utf-8")
    code = (
        "import base64, os, pathlib, subprocess, sys\n"
        "os.makedirs(%r, exist_ok=True)\n"
        "p = pathlib.Path(%r) / (%r + '.py')\n"
        "p.write_bytes(base64.b64decode(%r))\n"
        "log = pathlib.Path(%r) / (%r + '.log')\n"
        "f = open(log, 'wb')\n"
        # start_new_session: tien trinh song doc lap voi cell da sinh ra no
        "pr = subprocess.Popen([sys.executable, '-u', str(p)], stdout=f,\n"
        "                      stderr=subprocess.STDOUT, cwd='/content/ragproj',\n"
        "                      start_new_session=True)\n"
        "print('pid', pr.pid, '-> ', log)\n"
        % (LOGDIR, LOGDIR, name,
           __import__("base64").b64encode(src.encode()).decode(), LOGDIR, name))
    print(send(code))
    return 0


def tail(name: str, n: int) -> int:
    code = ("import pathlib\n"
            "p = pathlib.Path(%r) / (%r + '.log')\n"
            "print(p.read_text(errors='replace')[-6000:] if p.exists() else 'chua co log')\n"
            % (LOGDIR, name))
    out = send(code)
    print("\n".join(out.splitlines()[-n:]))
    return 0


def ps() -> int:
    print(send(
        "import subprocess, pathlib\n"
        "print(subprocess.run(['bash','-lc',"
        "'ps -eo pid,etime,cmd | grep -v grep | grep \"python -u\" || true'],"
        " capture_output=True, text=True).stdout)\n"
        "d = pathlib.Path(%r)\n"
        "print('logs:', sorted(f.name for f in d.glob('*.log')) if d.exists() else [])\n"
        % LOGDIR))
    return 0


def main() -> int:
    a = sys.argv[1] if len(sys.argv) > 1 else "ps"
    if a == "start":
        return start(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "job")
    if a == "tail":
        return tail(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 30)
    return ps()


if __name__ == "__main__":
    sys.exit(main())
