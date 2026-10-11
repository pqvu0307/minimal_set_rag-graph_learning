# -*- coding: utf-8 -*-
"""Tinh lai ngan sach sau khi: sinh chay LOCAL (Qwen3-8B) + judge DeepSeek.

Bang cu ($23.300) gia dinh CA HAI nua chay qua API gia $3/M vao, $12/M ra.
Gio:
    sinh    local  ->  0 tien, ton GIO GPU
    judge   DeepSeek ~$0.27/M vao, $1.10/M ra  ->  re hon ~11 lan
"""
IN, OUT = 0.27, 1.10          # DeepSeek, PHAI XAC THUC gia hien hanh

def usd(tin, tout, n):
    return n * (tin * IN + tout * OUT) / 1e6

# ── mot luot judge: q + y + aliases + cau model sinh + huong dan ──
J_IN, J_OUT = 180, 1
# ── mot luot che cau hoi: 1 LUOT MOI TRUY VAN ─────────────────────
D_IN, D_OUT = 300, 100
# ── mot luot OpenIE moi document ──────────────────────────────────
O_IN, O_OUT = 800, 150

rows = []

# GD 0.5 — tren test index, khong indexing
n_calib, n_dry, n_pgate = 500, 2000, 500
calls = (n_calib + n_dry + n_pgate) * 111
rows.append(("GD 0.5  judge", usd(J_IN, J_OUT, calls), "%.0fk luot" % (calls/1e3)))
rows.append(("GD 0.5  che cau hoi", usd(D_IN, D_OUT, n_calib+n_dry+n_pgate),
             "%d truy van" % (n_calib+n_dry+n_pgate)))

# GD 2 — pha 1
p1 = 18200
rows.append(("GD 2    judge", usd(J_IN, J_OUT, p1*111), "%.1fM luot" % (p1*111/1e6)))
rows.append(("GD 2    che cau hoi", usd(D_IN, D_OUT, p1), "%d truy van" % p1))
rows.append(("GD 2.0  OpenIE index", usd(O_IN, O_OUT, int(p1*10.7)),
             "%.0fk doc" % (p1*10.7/1e3)))

# GD 5 — pha 2
p2_run, p2_in = 110000, 200000
rows.append(("GD 5    judge", usd(J_IN, J_OUT, p2_run*10),
             "%.1fM luot" % (p2_run*10/1e6)))
rows.append(("GD 5    che cau hoi", usd(D_IN, D_OUT, p2_in),
             "%d truy van" % p2_in))
rows.append(("GD 5    cuu judge", usd(J_IN, J_OUT, 3000*111), "3k ca x 111"))
rows.append(("GD 5.0  OpenIE index", usd(O_IN, O_OUT, int(p2_in*10.7)),
             "%.1fM doc" % (p2_in*10.7/1e6)))

print("=" * 72)
print("  TIEN  (sinh chay local -> 0 tien)")
print("=" * 72)
tot = 0.0
for name, u, note in rows:
    tot += u
    print("  %-22s $%9s   %s" % (name, "%.0f" % u if u >= 1 else "%.2f" % u, note))
print("  " + "-" * 56)
print("  %-22s $%9.0f" % ("TONG", tot))
print("  %-22s $%9.0f" % ("  trong do OpenIE", rows[4][1] + rows[8][1]))
print("  %-22s $%9.0f" % ("  khong tinh OpenIE", tot - rows[4][1] - rows[8][1]))
print("\n  bang CU (ca hai nua qua API $3/$12):   $23.300")

# ── GIO GPU cho phan sinh ─────────────────────────────────────────
print("\n" + "=" * 72)
print("  GIO GPU  (rang buoc MOI — thay cho tien)")
print("=" * 72)
for rate in (10, 20, 30):
    g05 = (n_calib+n_dry+n_pgate)*111/rate/3600
    g2 = p1*111/rate/3600
    g5 = (p2_run*10 + 3000*111)/rate/3600
    print("  %2d sinh/giay   GD 0.5 %5.1f h   GD 2 %6.1f h   GD 5 %6.1f h   tong %6.1f h"
          % (rate, g05, g2, g5, g05+g2+g5))
print("\n  OpenIE neu chay LOCAL thay vi API:")
for rate in (5, 10):
    h = p2_in*10.7/rate/3600
    print("  %2d doc/giay    %6.0f h   (doi lai $%.0f tien API)"
          % (rate, h, usd(O_IN, O_OUT, int(p2_in*10.7))))
print("\n  Colab: session co gioi han -> DiskCache la BAT BUOC, khong phai tuy chon.")
