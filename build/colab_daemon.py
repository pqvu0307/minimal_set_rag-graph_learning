# -*- coding: utf-8 -*-
"""Giu MOT tien trinh ColabMCP song lau, va nhan lenh qua socket cuc bo.

Vi sao can: ban dau moi lenh sinh mot server MCP roi giet no khi xong.  Phia
trinh duyet thay server cuc bo lien tuc chet nen bao "disconnected from MCP
server", va moi lan lai phai Accept dialog.  Mot tien trinh, mot lan Accept,
nhieu lenh.

    python build/colab_daemon.py serve          # chay nen, giu ket noi
    python build/colab_daemon.py run "code"     # gui lenh toi daemon
    python build/colab_daemon.py runf file.py
    python build/colab_daemon.py cells
    python build/colab_daemon.py stop
"""
from __future__ import annotations

import json
import pathlib
import socket
import sys
import threading
import time
from typing import Any, Dict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mcp_stdio import MCP  # noqa: E402

HOST, PORT = "127.0.0.1", 8787


def _recv_all(conn) -> str:
    buf = []
    while True:
        b = conn.recv(65536)
        if not b:
            break
        buf.append(b)
        if b.endswith(b"\n"):
            break
    return b"".join(buf).decode("utf-8")


class Server:
    def __init__(self):
        self.m = MCP(timeout=1800)
        self.m.start()
        self.lock = threading.Lock()
        self.ready = False
        self.note = "chua bat tay"
        # Bat tay CHAY NEN: no doi nguoi bam Accept trong trinh duyet, co the
        # lau.  Ban cu chan o day roi moi mo socket, nen trong luc doi thi
        # khong hoi duoc trang thai — tuong nhu daemon chet.
        threading.Thread(target=self._handshake, daemon=True).start()

    def _handshake(self):
        try:
            ok = self.connect()
            self.ready = bool(ok)
            self.note = "da bat tay" if ok else "Accept bi tu choi / het han"
        except Exception as e:
            self.note = "%s: %s" % (type(e).__name__, str(e)[:160])

    def connect(self):
        r = self.m.rpc("tools/call",
                       {"name": "open_colab_browser_connection", "arguments": {}})
        return (r.get("structuredContent") or {}).get("result")

    def call(self, name, args, _retry=True):
        r = self.m.rpc("tools/call", {"name": name, "arguments": args})
        txt = "".join(c.get("text", "") for c in r.get("content", []))
        if r.get("isError"):
            # Doi runtime Colab (vd T4 -> A100) la KHOI DONG LAI, va cau noi
            # rung — nhung `ready` cu van con True nen daemon bao trang thai
            # OI.  Dau hieu la cong cu notebook bien mat khoi danh sach.
            # Tu bat tay lai mot lan roi thu lai, thay vi bao loi la lam.
            if _retry and ("Unknown tool" in txt or "not found" in txt.lower()):
                self.ready = bool(self.connect())
                self.note = "da noi lai" if self.ready else "noi lai that bai"
                if self.ready:
                    return self.call(name, args, _retry=False)
            raise RuntimeError(txt or str(r))
        return txt

    def cells(self):
        return json.loads(self.call("get_cells", {})).get("cells", [])

    def run(self, code: str) -> str:
        cs = self.cells()
        if cs and not "".join(cs[-1].get("source") or []).strip():
            cid = cs[-1]["id"]
        else:
            self.call("add_code_cell",
                      {"cellIndex": len(cs), "language": "python", "code": ""})
            cid = self.cells()[-1]["id"]
        self.call("update_cell", {"cellId": cid, "content": code})
        t0 = time.time()
        out = self.call("run_code_cell", {"cellId": cid})
        return self.render(out) + "\n--- %.1fs ---" % (time.time() - t0)

    @staticmethod
    def render(raw: str) -> str:
        """Gom stdout/err/ket qua thanh van ban doc duoc."""
        try:
            outs = json.loads(raw).get("outputs", [])
        except Exception:
            return raw
        parts = []
        for o in outs:
            if o.get("output_type") == "stream":
                parts.append("".join(o.get("text") or []))
            elif o.get("output_type") == "error":
                # Traceback cua Colab day ma mau ANSI — de nguyen thi mot loi
                # nho cung chiem ca man hinh.  Boc ra truoc khi in.
                import re as _re
                strip = lambda s: _re.sub(r"\x1b\[[0-9;]*m", "", s)
                tb = [strip(l).rstrip() for l in (o.get("traceback") or [])]
                tb = [l for l in tb if l.strip()]
                parts.append("%s: %s\n%s" % (o.get("ename"), o.get("evalue"),
                                             "\n".join(tb[-5:])))
            elif o.get("output_type") in ("execute_result", "display_data"):
                parts.append("".join((o.get("data") or {}).get("text/plain") or []))
        return "".join(parts).strip() or "(khong co output)"

    # Viec DAI khong duoc buoc client phai om socket ca tieng.  Truoc day
    # mot client bi timeout cat la daemon sap theo, keo chet luon job dang
    # chay — da mat mot me 90 phut vi dung chuyen do.  Gio: submit tra ve
    # ma job ngay, poll hoi ket qua sau.
    jobs: Dict[str, Dict[str, Any]] = {}

    def submit(self, code: str) -> str:
        jid = "j%d" % (len(self.jobs) + 1)
        self.jobs[jid] = {"state": "dang chay", "out": "", "t0": time.time()}

        def work():
            try:
                out = self.run(code)
                self.jobs[jid].update(state="xong", out=out)
            except Exception as e:
                self.jobs[jid].update(state="hong",
                                      out="%s: %s" % (type(e).__name__, e))
            self.jobs[jid]["dt"] = time.time() - self.jobs[jid]["t0"]
        threading.Thread(target=work, daemon=True).start()
        return jid

    def handle(self, req: dict) -> dict:
        op = req.get("op")
        if op == "status":
            return {"ok": True, "out": "ready=%s  %s" % (self.ready, self.note)}
        if op == "submit":
            return {"ok": True, "out": self.submit(req["code"])}
        if op == "poll":
            j = self.jobs.get(req.get("jid"), {})
            if not j:
                return {"ok": False, "out": "khong co job %r" % req.get("jid")}
            return {"ok": True, "out": "[%s %.0fs]\n%s"
                    % (j["state"], j.get("dt", time.time() - j["t0"]), j["out"])}
        if op == "jobs":
            return {"ok": True, "out": "\n".join(
                "%s  %-10s %.0fs" % (k, v["state"],
                                     v.get("dt", time.time() - v["t0"]))
                for k, v in self.jobs.items()) or "(chua co job)"}
        with self.lock:
            if op == "run":
                return {"ok": True, "out": self.run(req["code"])}
            if op == "cells":
                return {"ok": True, "out": json.dumps(
                    [{"i": i, "id": c["id"],
                      "src": "".join(c.get("source") or [])[:70]}
                     for i, c in enumerate(self.cells())], indent=1)}
            if op == "reconnect":
                return {"ok": True, "out": str(self.connect())}
            if op == "stop":
                return {"ok": True, "out": "stopping"}
        return {"ok": False, "out": "op la? %r" % op}


def serve() -> int:
    s = Server()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(8)
    print("daemon san sang tren %s:%d — mot tien trinh MCP duy nhat" % (HOST, PORT))
    sys.stdout.flush()
    while True:
        conn, _ = srv.accept()
        stop = False
        try:
            req = json.loads(_recv_all(conn) or "{}")
            stop = req.get("op") == "stop"
            try:
                res = s.handle(req)
            except Exception as e:
                res = {"ok": False, "out": "%s: %s" % (type(e).__name__, e)}
            conn.sendall((json.dumps(res) + "\n").encode("utf-8"))
        except Exception as e:
            # Client ngat giua chung (bi timeout cat, Ctrl-C, ...) thi KHONG
            # duoc lam sap daemon — job dang chay se chet theo.  Da mat mot
            # me 90 phut vi thieu dung cho nay.
            print("  (bo qua loi client: %s)" % type(e).__name__, flush=True)
        finally:
            try:
                conn.close()
            except Exception:
                pass
        if stop:
            break
    s.m.close()
    return 0


def client(req: dict) -> int:
    c = socket.create_connection((HOST, PORT), timeout=1800)
    c.sendall((json.dumps(req) + "\n").encode("utf-8"))
    res = json.loads(_recv_all(c) or "{}")
    c.close()
    print(res.get("out", ""))
    return 0 if res.get("ok") else 1


def main() -> int:
    a = sys.argv[1] if len(sys.argv) > 1 else "cells"
    if a == "serve":
        return serve()
    if a == "run":
        return client({"op": "run", "code": sys.argv[2]})
    if a == "runf":
        return client({"op": "run",
                       "code": pathlib.Path(sys.argv[2]).read_text(encoding="utf-8")})
    if a == "submitf":        # viec dai: tra ma job ngay, poll sau
        return client({"op": "submit",
                       "code": pathlib.Path(sys.argv[2]).read_text(encoding="utf-8")})
    if a == "poll":
        return client({"op": "poll", "jid": sys.argv[2]})
    return client({"op": a})


if __name__ == "__main__":
    sys.exit(main())
