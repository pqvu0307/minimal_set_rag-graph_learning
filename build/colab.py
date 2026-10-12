# -*- coding: utf-8 -*-
"""Driver Colab qua ColabMCP — chay code tren notebook dang mo trong Chrome.

Vi sao khong dung cong cu MCP cua Claude Code truc tiep: chung duoc nap luc
KHOI DONG phien.  Server dang ky giua chung thi phai restart.  File nay noi
JSON-RPC thang voi server nen dung duoc ngay.

Dung:
    python build/colab.py cells
    python build/colab.py run  "print(1+1)"
    python build/colab.py runf path/to/file.py     # chay noi dung mot file
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mcp_stdio import MCP  # noqa: E402


class Colab:
    def __init__(self, timeout: float = 900.0):
        self.m = MCP(timeout=timeout)
        self.m.start()
        self.call("open_colab_browser_connection", {})

    def call(self, name: str, args: Dict[str, Any]) -> str:
        r = self.m.rpc("tools/call", {"name": name, "arguments": args})
        if r.get("isError"):
            raise RuntimeError("%s -> %s" % (name, r))
        return "".join(c.get("text", "") for c in r.get("content", []))

    def cells(self) -> List[Dict[str, Any]]:
        return json.loads(self.call("get_cells", {})).get("cells", [])

    def _blank_or_new(self) -> str:
        """Dung lai cell trong cuoi cung neu co, khong thi them cell moi."""
        cs = self.cells()
        if cs and not "".join(cs[-1].get("source") or []).strip():
            return cs[-1]["id"]
        self.call("add_code_cell",
                  {"cellIndex": len(cs), "language": "python", "code": ""})
        return self.cells()[-1]["id"]

    def run(self, code: str) -> str:
        cid = self._blank_or_new()
        self.call("update_cell", {"cellId": cid, "content": code})
        t0 = time.time()
        out = self.call("run_code_cell", {"cellId": cid})
        return "%s\n--- %.1fs ---" % (out, time.time() - t0)

    def close(self):
        self.m.close()


def main() -> int:
    what = sys.argv[1] if len(sys.argv) > 1 else "cells"
    c = Colab()
    try:
        if what == "cells":
            for i, cell in enumerate(c.cells()):
                src = "".join(cell.get("source") or [])
                print("[%d] %s %s" % (i, cell["id"], src.replace("\n", " ")[:70]))
        elif what == "run":
            print(c.run(sys.argv[2]))
        elif what == "runf":
            print(c.run(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8")))
        return 0
    finally:
        c.close()


if __name__ == "__main__":
    sys.exit(main())
