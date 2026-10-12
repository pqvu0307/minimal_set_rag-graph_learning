# -*- coding: utf-8 -*-
"""Client MCP stdio toi gian — de goi mot MCP server NGAY TRONG phien nay.

Vi sao can: Claude Code nap cong cu MCP luc KHOI DONG phien.  Server dang ky
giua chung thi phai restart moi dung duoc.  Nhung MCP stdio chi la JSON-RPC
tren stdin/stdout, nen noi chuyen thang voi no duoc, khong phai doi.

Dung:
    python build/mcp_stdio.py list
    python build/mcp_stdio.py call <ten_tool> '<json args>'
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from typing import Any, Dict, Optional

UVX = ("C:/Users/ADMIN/AppData/Local/Packages/"
       "PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0/LocalCache/"
       "local-packages/Python313/Scripts/uvx.exe")
CMD = [UVX, "--from", "git+https://github.com/googlecolab/colab-mcp", "colab-mcp"]


class MCP:
    def __init__(self, cmd=None, timeout: float = 120.0):
        self.timeout = timeout
        self.p = subprocess.Popen(
            cmd or CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8",
            errors="replace", bufsize=1)
        self._id = 0
        self.err: list = []
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        for line in self.p.stderr:
            self.err.append(line.rstrip())

    def _send(self, obj: Dict[str, Any]):
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()

    def rpc(self, method: str, params: Optional[Dict[str, Any]] = None):
        self._id += 1
        mid = self._id
        self._send({"jsonrpc": "2.0", "id": mid, "method": method,
                    "params": params or {}})
        box: Dict[str, Any] = {}

        def rd():
            for line in self.p.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    m = json.loads(line)
                except Exception:
                    continue
                if m.get("id") == mid:
                    box["r"] = m
                    return
        t = threading.Thread(target=rd, daemon=True)
        t.start()
        t.join(self.timeout)
        if "r" not in box:
            raise TimeoutError("khong co phan hoi cho %s sau %.0fs.  stderr:\n%s"
                               % (method, self.timeout, "\n".join(self.err[-12:])))
        r = box["r"]
        if "error" in r:
            raise RuntimeError("%s -> %s" % (method, r["error"]))
        return r.get("result")

    def start(self):
        out = self.rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "ragproj-probe", "version": "0.1"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return out

    def close(self):
        try:
            self.p.terminate()
        except Exception:
            pass


def main() -> int:
    what = sys.argv[1] if len(sys.argv) > 1 else "list"
    m = MCP()
    try:
        info = m.start()
        si = (info or {}).get("serverInfo", {})
        print("server: %s %s" % (si.get("name"), si.get("version")))
        if what == "list":
            tools = (m.rpc("tools/list") or {}).get("tools", [])
            print("%d cong cu:" % len(tools))
            for t in tools:
                d = (t.get("description") or "").split("\n")[0][:96]
                req = (t.get("inputSchema") or {}).get("required", [])
                print("  %-32s %s" % (t["name"], d))
                if req:
                    print("      bat buoc: %s" % req)
        elif what == "call":
            name = sys.argv[2]
            args = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
            r = m.rpc("tools/call", {"name": name, "arguments": args})
            print(json.dumps(r, indent=2, ensure_ascii=False)[:4000])
        return 0
    finally:
        m.close()


if __name__ == "__main__":
    sys.exit(main())
