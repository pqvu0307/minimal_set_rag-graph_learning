# -*- coding: utf-8 -*-
"""Dung slide trinh bay, theo phong cach MAGMA.pptx."""
from pptx import Presentation
from pptx.util import Inches as I, Pt
from pptx.dml.color import RGBColor as C
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

NAVY, PALE, INK, WHITE = C(0x0E,0x3F,0x92), C(0xE8,0xF0,0xFC), C(0x33,0x33,0x33), C(0xFF,0xFF,0xFF)
ORANGE, PEACH, GREY   = C(0xE8,0x77,0x2E), C(0xFD,0xEE,0xE3), C(0x88,0x88,0x88)
GREEN                 = C(0x2E,0x9E,0x5B)
F = "Arial"
W, H = 13.333, 7.5
TOTAL = 20

prs = Presentation(); prs.slide_width, prs.slide_height = I(W), I(H)
BLANK = prs.slide_layouts[6]

def _tf(sh, size, color, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE, space=0):
    tf = sh.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = I(0.08); tf.margin_top = tf.margin_bottom = I(0.02)
    for p in tf.paragraphs:
        p.alignment = align
        if space: p.space_after = Pt(space)
        for r in p.runs:
            r.font.name, r.font.size, r.font.color.rgb, r.font.bold = F, Pt(size), color, bold
    return tf

def text(sl, x, y, w, h, s, size=15, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, mono=False, space=3):
    sh = sl.shapes.add_textbox(I(x), I(y), I(w), I(h)); tf = sh.text_frame
    tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = I(0.06); tf.margin_top = tf.margin_bottom = I(0.02)
    import re as _re
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(space)
        for part in _re.split(r"([_^]\{[^}]*\})", line):
            if not part: continue
            base = None
            if part.startswith("_{") and part.endswith("}"):
                part, base = part[2:-1], "-25000"
            elif part.startswith("^{") and part.endswith("}"):
                part, base = part[2:-1], "30000"
            r = p.add_run(); r.text = part
            r.font.name = "Consolas" if mono else F
            r.font.size, r.font.color.rgb, r.font.bold = Pt(size), color, bold
            if base: r.font._rPr.set("baseline", base)
    return sh

RAD = 0.07          # ban kinh goc, inch — giu TUYET DOI, khong theo ti le shape

def rect(sl, x, y, w, h, fill=None, line=None, rounded=False, radius=RAD):
    sh = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                             I(x), I(y), I(w), I(h))
    if rounded:
        # adjustment la ti le cua NUA canh ngan -> quy doi ra ban kinh tuyet doi
        try: sh.adjustments[0] = max(0.0, min(0.5, radius / (min(w, h) / 2.0)))
        except Exception: pass
    if fill is None: sh.fill.background()
    else: sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None: sh.line.fill.background()
    else: sh.line.color.rgb = line; sh.line.width = Pt(1)
    sh.shadow.inherit = False
    return sh

def chrome(sl, title, section, n):
    b = rect(sl, 0, 0, W, 0.85, NAVY)
    _tf(b, 26, WHITE, True, PP_ALIGN.LEFT); b.text_frame.text = title
    for p in b.text_frame.paragraphs:
        for r in p.runs: r.font.name, r.font.size, r.font.color.rgb, r.font.bold = F, Pt(25), WHITE, True
    b.text_frame.margin_left = I(0.32)
    rect(sl, 0, 7.18, W, 0.32, NAVY)
    text(sl, 0.15, 7.20, 7.0, 0.28, section, 10, WHITE, anchor=MSO_ANCHOR.MIDDLE)
    text(sl, 11.3, 7.20, 1.9, 0.28, f"{n}/{TOTAL}", 10, WHITE, align=PP_ALIGN.RIGHT,
         anchor=MSO_ANCHOR.MIDDLE)

def panel(sl, x, y, w, h, header, fill=PALE, hdr=NAVY):
    """nen bo goc cho ca khoi, roi dai tieu de de len -> mep ngoai mem, mach trong phang"""
    rect(sl, x, y, w, h + 0.36, fill, rounded=True)
    rect(sl, x, y, w, 0.36, hdr, rounded=True)
    rect(sl, x, y + 0.20, w, 0.16, hdr)          # lap goc duoi cua dai tieu de
    text(sl, x+0.04, y+0.02, w-0.1, 0.32, header, 15, WHITE, True, anchor=MSO_ANCHOR.MIDDLE)

def band(sl, x, y, w, h, header, body, size=15, mono=False, fill=PALE, hdr=NAVY):
    """strip tieu de + panel noi dung — mau dac trung cua MAGMA"""
    panel(sl, x, y, w, h, header, fill, hdr)
    text(sl, x+0.08, y+0.44, w-0.2, h-0.14, body, size, INK, mono=mono)
    return y + 0.36 + h

def takeaway(sl, s, y=6.03):
    panel(sl, 0.7, y, 11.93, 0.44, "Điểm chốt")
    text(sl, 0.8, y+0.37, 11.7, 0.42, s, 16, INK, anchor=MSO_ANCHOR.MIDDLE)

def source(sl, s):
    text(sl, 0.7, 6.88, 11.0, 0.28, s, 10, NAVY)

def node(sl, x, y, w, h, s, fill=WHITE, fg=INK, size=13, bold=False, edge=True):
    r = rect(sl, x, y, w, h, fill, C(0x9A,0xA3,0xB2) if edge else None, rounded=True)
    text(sl, x+0.04, y, w-0.08, h, s, size, fg, bold, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE, space=1)
    return r

def arrow_down(sl, cx, y, h=0.26):
    a = sl.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, I(cx-0.055), I(y), I(0.11), I(h))
    a.fill.solid(); a.fill.fore_color.rgb = C(0x9A,0xA3,0xB2); a.line.fill.background()
    a.shadow.inherit = False

def arrow_right(sl, x, cy, w=0.3):
    a = sl.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, I(x), I(cy-0.055), I(w), I(0.11))
    a.fill.solid(); a.fill.fore_color.rgb = C(0x9A,0xA3,0xB2); a.line.fill.background()
    a.shadow.inherit = False

S = lambda: prs.slides.add_slide(BLANK)

# ───────────────────────────────── 1 · bìa
sl = S()
rect(sl, 0, 0, W, H, NAVY)
rect(sl, 0, 2.55, W, 0.06, ORANGE)
text(sl, 1.1, 1.45, 11.2, 1.0, "Quy gán nguồn cho RAG hai tầng", 40, WHITE, True)
text(sl, 1.1, 2.85, 11.2, 0.9,
     "Shapley ở tầng sinh  ·  Integrated Gradients trên đồ thị ở tầng truy hồi",
     19, C(0xC8,0xD6,0xF0))
text(sl, 1.1, 4.25, 11.2, 1.4,
     "Câu hỏi:  trong các đoạn được truy hồi, đoạn nào thật sự tạo ra câu trả lời —\n"
     "và đáng lẽ retriever phải đi lối nào trong đồ thị để tới các đoạn đó?",
     16, C(0xC8,0xD6,0xF0), space=6)
text(sl, 1.1, 6.35, 11.2, 0.4, "Tài liệu thiết kế · chưa có kết quả thực nghiệm", 13, ORANGE)

# ───────────────────────────────── 2 · bài toán
sl = S(); chrome(sl, "Bài toán", "Động cơ", 2)
text(sl, 0.7, 1.05, 12.0, 0.4,
     "Mô hình trả lời xong. Ta đưa cho nó tới 60 đoạn truy hồi. Đoạn nào thật sự tạo ra câu trả lời đó?",
     17, INK)
for i, lb in enumerate(["D₁","D₂","D₃","…","D₆₀"]):
    node(sl, 0.7+i*0.78, 1.75, 0.66, 0.42, lb, PALE, NAVY, 13)
arrow_right(sl, 4.72, 1.96)
node(sl, 5.15, 1.70, 1.5, 0.52, "LLM", NAVY, WHITE, 15, True)
arrow_right(sl, 6.78, 1.96)
node(sl, 7.2, 1.70, 2.5, 0.52, "câu trả lời", WHITE, INK, 14)
band(sl, 0.7, 2.85, 5.85, 2.35, "Khó ở chỗ thứ nhất — dư thừa",
     "Ba đoạn cùng nói một ý thì không thể tính công đầy đủ\ncho cả ba. Như vậy là cộng khống.\n\n"
     "Và bỏ riêng từng đoạn thì mất rất ít — nên phép thử\n«bỏ một đoạn» cũng không phát hiện được.", 14)
band(sl, 6.78, 2.85, 5.85, 2.35, "Khó ở chỗ thứ hai — chi phí",
     "Cách công bằng về mặt toán học là giá trị Shapley:\nthử mọi tổ hợp đoạn.\n\n"
     "Với K = 60 đoạn, đó là 2^{60} ≈ 10^{18} lượt chạy\nmô hình. Không khả thi.", 14)
source(sl, "Chi tiết: tài liệu thiết kế §1, §3.1")

# ───────────────────────────────── 3 · hai công cụ
sl = S(); chrome(sl, "Hai công cụ, mỗi cái hỏng một nửa", "Động cơ", 3)
band(sl, 0.7, 1.10, 5.85, 2.60, "Shapley",
     "Mạnh\n"
     "· công bằng theo tiên đề\n"
     "· xử lý đúng dư thừa và bổ sung\n\n"
     "Yếu\n"
     "· chi phí 2^{K} — bất khả thi với K = 60", 15)
band(sl, 6.78, 1.10, 5.85, 2.60, "Integrated Gradients",
     "Mạnh\n"
     "· rẻ: M lượt, không phụ thuộc số đơn vị chấm\n"
     "· chi tiết tới từng token / từng cạnh\n\n"
     "Yếu\n"
     "· không hỏi được phản thực «nếu bỏ đi thì sao»", 15)
text(sl, 0.7, 4.35, 11.9, 1.4,
     "▸  Gom các đoạn giống nhau thành cụm:  số «người chơi» tụt từ K ≤ 60 xuống C = 6,  Shapley chạy được.\n"
     "▸  Shapley chia công trạng giữa các cụm;  IG chia tiếp bên trong cụm và trên đồ thị.",
     16, ORANGE, space=8)
takeaway(sl, "Shapley quyết định bao nhiêu.   IG quyết định ở đâu.")
source(sl, "Chi tiết: §1")

# ───────────────────────────────── 4 · IG là gì  [MỚI]
sl = S(); chrome(sl, "Integrated Gradients — công cụ chính", "Nền tảng", 4)
text(sl, 0.7, 1.00, 12.0, 0.35,
     "Câu hỏi IG trả lời:  một đầu ra vô hướng F chịu ảnh hưởng bao nhiêu từ mỗi đầu vào?", 16, INK)
rect(sl, 0.7, 1.48, 11.93, 0.86, PALE)
text(sl, 0.95, 1.56, 11.4, 0.45,
     "IG_{i}   =   ( x_{i} − b_{i} )   ×   ∫ gradient dọc đường từ mốc b tới điểm thật x", 19, NAVY, True)
text(sl, 0.95, 2.00, 11.4, 0.3,
     "đi xa mốc bao nhiêu                      không chỉ lấy gradient tại đích", 13, GREY)
band(sl, 0.7, 2.58, 5.85, 1.88, "Vì sao phải tích phân — ví dụ",
     "F(a,b) = min(a+b, 1),  mốc (0,0),  thật (0.8, 0.6)\n\n"
     "gradient tại điểm thật:  a+b = 1.4 > 1 → bão hòa\n     ∂F/∂a = ∂F/∂b = 0\n"
     "     «cả hai đều vô dụng» — sai hẳn\n\n"
     "IG dọc đường:   IG_a = 0.57,   IG_b = 0.43", 13)
band(sl, 6.78, 2.58, 5.85, 1.88, "Tính chất dùng xuyên suốt — completeness",
     "Σ IG_{i}  =  F(x) − F(mốc)\n\n"
     "Trong ví dụ:  0.57 + 0.43 = 1.00  ✓\n\n"
     "Các phần cộng lại đúng bằng cái tổng. Nhờ nó\nmọi con số IG đọc được như «chiếm bao nhiêu\n"
     "phần trăm», và mọi phép kiểm đều dựa vào nó.", 13)
rect(sl, 0.7, 4.96, 11.93, 0.36, ORANGE)
text(sl, 0.74, 4.98, 11.8, 0.32, "Hai nghĩa của chữ «đường» — đừng lẫn", 15, WHITE, True, anchor=MSO_ANCHOR.MIDDLE)
rect(sl, 0.7, 5.32, 11.93, 0.72, PEACH)
text(sl, 0.95, 5.40, 11.4, 0.64,
     "đường tích phân   đoạn thẳng trong không gian đầu vào, từ mốc tới điểm thật — thiết bị toán học bên trong IG\n"
     "đường trên đồ thị   chuỗi cạnh  seed → … → D  — thứ tầng B đi tìm", 14, INK, space=3)
source(sl, "Chi tiết: §4 mở đầu")

# ───────────────────────────────── 5 · toàn cảnh
sl = S(); chrome(sl, "Toàn cảnh — hai tầng, hai câu hỏi khác nhau", "Tổng quan", 5)
y = 1.15
node(sl, 4.55, y, 4.2, 0.42, "truy vấn", WHITE, INK, 14); arrow_down(sl, 6.65, y+0.42)
y += 0.70
node(sl, 4.55, y, 4.2, 0.42, "truy hồi rộng  ·  K ≤ 60, đa nguồn", WHITE, INK, 13); arrow_down(sl, 6.65, y+0.42)
y += 0.70
node(sl, 3.1, y, 7.1, 0.62, "Tầng A — Shapley trên LLM sinh\ngom cụm → φ_{c} → cắt cụm → greedy trong cụm",
     PALE, NAVY, 14, True); arrow_down(sl, 6.65, y+0.62)
y += 0.90
node(sl, 4.55, y, 4.2, 0.42, "tập đích  T = { D₂, D₇, D₁₅ }", WHITE, INK, 14); arrow_down(sl, 6.65, y+0.42)
y += 0.70
node(sl, 3.1, y, 7.1, 0.62, "Tầng B — IG trên GFM truy hồi\ngate cạnh → điểm cạnh → quy hoạch động trên DAG",
     PALE, NAVY, 14, True); arrow_down(sl, 6.65, y+0.62)
y += 0.90
node(sl, 4.55, y, 4.2, 0.42, "kiểm chứng bằng ablation  →  nhãn train", WHITE, INK, 13)
band(sl, 0.7, 1.15, 2.2, 1.0, "Tầng A hỏi", "Đoạn nào thật sự\ncần?", 14)
band(sl, 10.43, 1.15, 2.2, 1.0, "Tầng B hỏi", "Đáng lẽ phải đi\nlối nào tới đó?", 14)
text(sl, 0.7, 5.95, 11.9, 0.5,
     "Phần đắt nằm hết ở offline. Lúc chạy thật: truy vấn → GFM đã train → đoạn → LLM → trả lời.", 15, INK)
source(sl, "Chi tiết: §1, sơ đồ luồng")

# ───────────────────────────────── 6 · tầng A, Shapley cụm
sl = S(); chrome(sl, "Tầng A · gom cụm rồi chấm Shapley trên cụm", "Tầng A", 6)
band(sl, 0.7, 1.10, 5.85, 2.05, "Gom cụm làm hai việc cùng lúc",
     "1.  Giảm chi phí:  2^{K} → 2^{C}.\n     K ≤ 60 → C = 6,  tức 2^{6} = 64 lượt.\n     C co theo K:  C = clamp(K/10, 4, 8)\n\n"
     "2.  Gom dư thừa: đoạn nói cùng một ý vào\n     cùng một cụm, nên Shapley chia công ở\n     mức ngữ nghĩa.", 14)
band(sl, 6.78, 1.10, 5.85, 2.05, "Giá trị Shapley của cụm",
     "φ_{c}  =  đóng góp biên trung bình của cụm c\n          trên mọi thứ tự xuất hiện.\n\n"
     "Hai cụm trùng bằng chứng thì chia nhau công.\nCụm vô dụng có φ_{c} = 0.\n"
     "Cụm nhiễu có φ_{c} âm.", 14)
rect(sl, 0.7, 3.62, 11.93, 0.36, NAVY)
text(sl, 0.74, 3.64, 11.8, 0.32, "Tiên đề then chốt — tính đầy đủ", 15, WHITE, True, anchor=MSO_ANCHOR.MIDDLE)
rect(sl, 0.7, 3.98, 11.93, 1.68, PALE)
text(sl, 0.9, 4.16, 11.5, 0.5,
     "Σ_{c}  φ_{c}   =   F(E) − F(E′)        tổng công trạng các cụm  =  toàn bộ ảnh hưởng của khâu truy hồi",
     18, NAVY, True)
text(sl, 0.9, 4.66, 11.5, 0.9,
     "F(E′) là điểm số mô hình đạt được chỉ bằng tri thức tham số — không nhìn đoạn nào. Hiệu số đó là «cái bánh»\n"
     "mà cả quy trình đem chia, và là mốc để mọi đẳng thức phía sau quy về.",
     14, INK, space=5)
takeaway(sl, "Không mất mát, không tính trùng — và nó cho biết tổng ngân sách sẽ chia xuống từng đoạn.")
source(sl, "Chi tiết: §3.2 – §3.4,  Kiểm tra 1")

# ───────────────────────────────── 7 · bẫy cắt tỉa
sl = S(); chrome(sl, "Tầng A · vì sao không cắt tỉa bằng điểm attribution", "Tầng A", 7)
text(sl, 0.7, 1.02, 12.0, 0.4, "Một cụm có ba đoạn gần trùng nhau. Mỗi đoạn một mình đều đủ trả lời.", 16, INK)
band(sl, 0.7, 1.55, 5.85, 2.35, "Chấm điểm rồi đặt ngưỡng  ✗",
     "φ_{1} = 2.52 cho cả cụm\n\n"
     "     đoạn a:  điểm 0.9\n     đoạn b:  điểm 0.8\n     đoạn c:  điểm 0.7\n\n"
     "Ngưỡng «dưới 1.0 thì cắt» → cắt cả ba → mất trắng.", 14)
band(sl, 6.78, 1.55, 5.85, 2.35, "Greedy chọn dần từ rỗng  ✓",
     "Vòng 1   đứng một mình:  a = 1.90\n\n"
     "Vòng 2   phần thêm vào a:\n              b: +0.05      ← bản sao của a\n"
     "              c: +0.40      ← bổ sung cho a\n\n"
     "Giữ { a, c }.   Bỏ b dù b mạnh hơn c khi đứng riêng.", 14)
text(sl, 0.7, 4.42, 11.9, 0.55,
     "Mô hình bão hòa trên nhóm trùng lặp nên nó dìm điểm của từng thành viên. Lợi ích biên thay đổi theo\n"
     "những gì đã chọn — chính khoảng cách giữa hai con số đó là tín hiệu dư thừa.", 14, INK, space=4)
band(sl, 0.7, 5.05, 11.93, 0.80, "Dừng ở đâu — một ngân sách cho cả bài, không phải ngưỡng riêng từng cụm",
     "B = (1 − θ_g)·Δ_full.  Mỗi cụm tỉa xong thì trừ vào B đúng phần ĐIỂM SỐ TOÀN CỤC đã mất — không phải "
     "«95% của riêng cụm».\nCụm dư thừa nhiều thì tỉa mạnh và tiêu nhiều ngân sách; cụm sau tự bị siết lại.", 13)
takeaway(sl, "Phân bổ và cắt tỉa là hai câu hỏi khác nhau: «chia công thế nào» ≠ «bỏ đi thì mất gì».", 6.15)
source(sl, "Chi tiết: §3.6; phụ lục «Những chỗ dễ sai» §2")

# ───────────────────────────────── 8 · đầu ra tầng A
sl = S(); chrome(sl, "Tầng A · đầu ra là tập đích T", "Tầng A", 8)
band(sl, 0.7, 1.10, 5.85, 2.75, "Bốn bước",
     "1.  Gom cụm            ≤ 60 đơn vị → 6 cụm\n\n"
     "2.  Shapley mức cụm    φ_c,  64 lượt LLM\n\n"
     "3.  Cắt cụm            giữ cụm có φ_{c} > τ\n\n"
     "4.  Greedy trong cụm   tuần tự theo φ_{c} giảm dần,\n                                  tiêu chung một ngân sách B", 14)
band(sl, 6.78, 1.10, 5.85, 2.75, "Hai phép kiểm cuối",
     "Bất biến toàn cục\nΔ(T) ≥ θ_g·Δ_full — giờ đúng THEO CẤU TRÚC\nvì B không bao giờ âm. Sai ở đây là bug.\n\n"
     "Tỉa ngược\nbỏ thử từng đoạn trong T; đoạn nào bỏ mà\nvẫn đạt ngưỡng thì loại hẳn.\n\n"
     "Tổng thêm |T| + 1 lượt.", 14)
rect(sl, 0.7, 4.32, 11.93, 0.36, NAVY)
text(sl, 0.74, 4.34, 11.8, 0.32, "Kết quả", 15, WHITE, True, anchor=MSO_ANCHOR.MIDDLE)
rect(sl, 0.7, 4.68, 11.93, 1.42, PALE)
text(sl, 0.95, 4.86, 11.4, 0.5, "T  =  { D₂, D₇, D₁₅ }        và        S_{final}(D_{k}) cho từng đoạn", 18, NAVY, True)
text(sl, 0.95, 5.42, 11.4, 0.5,
     "Σ S_{final}  =  cái bánh − phần dư do cắt cụm.  Phần dư nên báo cáo như một chỉ số chẩn đoán.", 14, INK)
source(sl, "Chi tiết: §3.5 – §3.8,  Kiểm tra 2 và 3")

# ───────────────────────────────── 9 · G-reasoner
sl = S(); chrome(sl, "G-reasoner — kiến trúc, và ba chỗ quan trọng với ta", "Tầng B", 9)
band(sl, 0.7, 1.05, 5.85, 2.25, "QuadGraph — bốn loại node, nối nhau hai chiều",
     "community    cụm chủ đề, có bản tóm tắt\n"
     "document     chunk văn bản\n"
     "entity       thực thể\n"
     "attribute    thuộc tính\n\n"
     "Cạnh liên tầng cho phép entity → document → entity\nkhác: multi-hop ở ngay mức tài liệu.", 13, mono=False)
band(sl, 6.78, 1.05, 5.85, 2.25, "Một lượt chạy",
     "h⁰_{v}  =  MLP( emb(v), emb(q) )     nếu v ∈ V_q\n"
     "        =  emb(v)                      nếu không\n\n"
     "L tầng, L = 3–6:\n"
     "     m_{e}  =  DistMult( h_{v′}, h_{r}, h_{v} )\n"
     "     h_{v}  =  Upd( h_{v},  Σ m_{e} )\n\n"
     "p(v)  =  Predictor_{type(v)}( h^{L}_{v}, emb(v), emb(q) )", 13)
rect(sl, 0.7, 3.74, 11.93, 0.36, ORANGE, rounded=True)
text(sl, 0.74, 3.76, 11.8, 0.32, "Ba tính chất — hai trong số đó là lý do ta phải can thiệp",
     15, WHITE, True, anchor=MSO_ANCHOR.MIDDLE)
rows9 = [("inductive", "mọi đầu vào suy từ văn bản, không tham số gắn ID node",
          "chạy được trên đồ thị mới, không train lại  ✓", GREEN),
         ("không có trọng số cạnh", "DistMult rồi cộng tổng — không attention, không hệ số riêng từng cạnh",
          "IG không có gì để bám  →  chèn gate, §4.2", ORANGE),
         ("truy vấn chỉ vào lúc khởi tạo", "sau đó L tầng lan truyền chạy như nhau bất kể hỏi gì",
          "gate phụ thuộc truy vấn, §4.2 chế độ 2", ORANGE)]
for i, (a, b, c, fg) in enumerate(rows9):
    y = 4.22 + i*0.80
    rect(sl, 0.7, y, 11.93, 0.72, PEACH if fg is ORANGE else PALE, rounded=True)
    text(sl, 0.95, y, 3.3, 0.72, a, 14, fg, True, anchor=MSO_ANCHOR.MIDDLE)
    text(sl, 4.3, y, 4.9, 0.72, b, 13, INK, anchor=MSO_ANCHOR.MIDDLE)
    text(sl, 9.3, y, 3.2, 0.72, c, 13, fg, anchor=MSO_ANCHOR.MIDDLE)
source(sl, "34M tham số · EM 81,1 trên HotpotQA · arXiv:2509.24276 — chi tiết kiến trúc nên đối chiếu bản gốc")

# ───────────────────────────────── 10 · tầng B bối cảnh
sl = S(); chrome(sl, "Tầng B · bối cảnh và chỗ thiếu", "Tầng B", 10)
band(sl, 0.7, 1.10, 5.85, 2.45, "Graph foundation model — và ta dùng loại node nào",
     "G-reasoner — QuadGraph bốn loại node. 34M tham số,\nkhông nhớ thực thể nào nên chạy được trên đồ thị mới.\n\n"
     "   document  · community    VÀO prompt  ← nhãn T\n"
     "   entity    · attribute    không       ← nhãn Π\n\n"
     "Quy tắc: vào prompt ⟺ từng là người chơi của Shapley.\nHai loại sau là chất nền lan truyền, không phải văn bản.", 13)
band(sl, 6.78, 1.10, 5.85, 2.45, "Vì sao phải là đồ thị",
     "Bi-encoder chấm từng đoạn độc lập, nên không\ndiễn đạt được «d chỉ có ích khi đi cùng c».\n\n"
     "    A: X sinh ở thành phố C      v({A}) ≈ 0\n"
     "    B: C thuộc nước D             v({B}) ≈ 0\n"
     "    hỏi «X sinh ở nước nào?»   v({A,B}) cao\n\n"
     "Đồ thị biểu diễn tính kết hợp một cách tự nhiên.", 13)
band(sl, 0.7, 4.02, 11.93, 1.52, "Chỗ thiếu khiến ta phải can thiệp",
     "G-reasoner tính thông điệp bằng DistMult rồi cộng tổng: không attention, không hệ số riêng cho từng cạnh —\n"
     "mỗi cạnh hoặc gửi đủ, hoặc không tồn tại, không có mức trung gian.\n\n"
     "Nhưng IG cần một biến liên tục vô hướng mới lấy đạo hàm được. Không có biến nào thì IG không bám vào đâu.", 15)
source(sl, "G-reasoner: arXiv:2509.24276 · GFM-RAG: arXiv:2502.01113, NeurIPS 2025")

# ───────────────────────────────── 10 · chèn gate
sl = S(); chrome(sl, "Tầng B · chèn gate lên cạnh", "Tầng B", 11)
rect(sl, 0.7, 1.05, 11.93, 0.95, PALE)
text(sl, 0.95, 1.14, 11.4, 0.45, "thông điệp mới   =   g   ×   Msg( · )", 22, NAVY, True)
text(sl, 0.95, 1.60, 11.4, 0.35,
     "g  là một số trong [0,1]            Msg( · )  là vector d chiều", 15, INK)
text(sl, 0.7, 2.14, 11.9, 0.38,
     "Một số duy nhất điều khiển cả vector — nhờ đó mỗi cạnh chỉ có một đại lượng cần quy gán, không phải d đại lượng.",
     15, INK)
band(sl, 0.7, 2.70, 5.85, 1.90, "Chế độ 1 — CÁCH ĐO",
     "g = 1  cho mọi cạnh, mọi tầng.\n\n"
     "Nhân với 1 nên model chạy y hệt bản gốc.\nGate chỉ tồn tại để IG có chỗ bám.\n\n"
     "Dùng để đo ở MỌI mốc — không bao giờ bỏ.", 14)
band(sl, 6.78, 2.70, 5.85, 1.90, "Chế độ 2 — MỘT TÍNH NĂNG CỦA MODEL",
     "ĝ = σ( MLP( h_{v} , h_{r} , h_{v′} , h_{q} ) )\n\n"
     "Cho model định tuyến khác nhau theo từng\ncâu hỏi — bản gốc chỉ cho truy vấn tác động\nlúc khởi tạo. Bật từ mốc 2. Vẫn inductive.", 14)
rect(sl, 0.7, 4.78, 11.93, 0.36, ORANGE)
text(sl, 0.74, 4.80, 11.8, 0.32, "Hai chế độ không phải hai lựa chọn cho cùng một việc", 15, WHITE, True, anchor=MSO_ANCHOR.MIDDLE)
rect(sl, 0.7, 5.14, 11.93, 0.80, PEACH)
text(sl, 0.95, 5.22, 11.4, 0.72,
     "ĝ là thứ MODEL LÀM, không phải thứ ta dùng để ĐO.  Nếu đo bằng ĝ thì cạnh model đang đóng nhận IG ≈ 0 —\n"
     "mà đó đúng là những cạnh «lẽ ra phải mở» cần tìm.  Nên đào nhãn thì luôn đo với g = 1, kể cả từ mốc 2.",
     14, INK, space=3)
source(sl, "Chi tiết: §4.2, §4.4")

# ───────────────────────────────── 11 · gate theo tầng
sl = S(); chrome(sl, "Tầng B · gate phải riêng từng tầng", "Tầng B", 12)
text(sl, 0.7, 1.02, 12.0, 0.35,
     "Một cạnh vật lý được dùng L lần, mỗi tầng một lần — và các lần đó không tương đương.", 16, INK)
for i, (lb, fill, fg) in enumerate([("X", NAVY, WHITE), ("C", WHITE, INK), ("D", WHITE, INK)]):
    node(sl, 3.5+i*2.6, 1.55, 1.3, 0.5, lb, fill, fg, 16, True)
    if i < 2: arrow_right(sl, 4.9+i*2.6, 1.80, 1.15)
text(sl, 4.9, 1.26, 1.2, 0.28, "sinh_ở", 12, GREY, align=PP_ALIGN.CENTER)
text(sl, 7.5, 1.26, 1.2, 0.28, "thuộc", 12, GREY, align=PP_ALIGN.CENTER)
text(sl, 2.35, 1.62, 1.1, 0.38, "seed", 12, ORANGE, align=PP_ALIGN.RIGHT)
band(sl, 0.7, 2.45, 5.85, 1.85, "Tầng 1",
     "X → C  nổ    C nhận tín hiệu truy vấn       hữu ích\n\n"
     "C → D  nổ    C còn trống, D nhận đồ\n             vô nghĩa                          vô dụng", 13, mono=True)
band(sl, 6.78, 2.45, 5.85, 1.85, "Tầng 2",
     "C → D  nổ    C đã có tín hiệu, D nhận\n             được                              hữu ích\n\n"
     "X → C  nổ    C đã có rồi, thừa                 gần vô dụng", 13, mono=True)
text(sl, 0.7, 4.80, 11.9, 0.8,
     "Gate chung cho cả L tầng sẽ trộn hai vai trò đó lại và mất thông tin thứ tự — tức mất đúng cái quan trọng\n"
     "nhất của multi-hop. Nên số gate là  |𝓔| × L,  một gate cho mỗi cặp (cạnh, tầng).", 15, INK, space=6)
takeaway(sl, "Cùng một cạnh: tầng này thiết yếu, tầng kia vô dụng. Thứ tự chính là nội dung của multi-hop.")
source(sl, "Chi tiết: §4.3")

# ───────────────────────────────── 12 · bài toán chuỗi  [MỚI]
sl = S(); chrome(sl, "Tầng B · thứ cần đọc ra là một chuỗi", "Tầng B", 13)
text(sl, 0.7, 1.00, 12.0, 0.35,
     "IG cho mỗi cặp (cạnh, tầng) một con số. Đó là một đống số rời rạc — chưa phải lời giải thích.", 16, INK)
rect(sl, 0.7, 1.48, 11.93, 0.36, NAVY)
text(sl, 0.74, 1.50, 11.8, 0.32, "Ví dụ:  «vợ của đạo diễn phim X là ai?»", 15, WHITE, True, anchor=MSO_ANCHOR.MIDDLE)
rect(sl, 0.7, 1.84, 11.93, 0.92, PALE)
node(sl, 2.2, 2.02, 2.1, 0.54, "Phim-X", WHITE, NAVY, 14, True); arrow_right(sl, 4.45, 2.29, 1.25)
node(sl, 5.9, 2.02, 2.0, 0.54, "Ông-Z", WHITE, INK, 14); arrow_right(sl, 8.05, 2.29, 1.25)
node(sl, 9.5, 2.02, 2.1, 0.54, "D  (bà Y)", WHITE, INK, 14)
text(sl, 4.4, 1.86, 1.4, 0.26, "đạo diễn", 11, GREY, align=PP_ALIGN.CENTER)
text(sl, 8.0, 1.86, 1.4, 0.26, "vợ", 11, GREY, align=PP_ALIGN.CENTER)
text(sl, 0.95, 2.30, 1.2, 0.3, "seed", 11, ORANGE)
band(sl, 0.7, 2.95, 5.85, 2.28, "Vì sao không lấy top-k cạnh điểm cao  ✗",
     "X --0.18--> A     P --0.21--> Q     R --0.19--> D\n"
     "seed                                          đích\n\n"
     "Ba con số to nhất, không mảnh nào nối X tới D.\n\n"
     "Và một TẬP cạnh vẫn không nói THỨ TỰ — mà thứ tự\nchính là thứ multi-hop cần đọc ra.", 13, mono=False)
band(sl, 6.78, 2.95, 5.85, 2.28, "Ràng buộc thứ tự là nhân quả, không phải quy ước",
     "tầng 1   Phim-X → Ông-Z    Ông-Z mới «biết» nó\n                                      liên quan truy vấn\n"
     "tầng 2   Ông-Z → D            tin này mới có nội dung\n\n"
     "Nếu tầng 1 không gửi thì sang tầng 2 Ông-Z vẫn\nrỗng — tin nó gửi cho D chẳng mang gì.\n\n"
     "Cạnh điểm IG cao mà nguồn chưa nhận được gì thì\nkhông truyền được gì.", 13)
takeaway(sl, "Cần một chuỗi liền mạch theo thời gian: cạnh thứ i dùng ở đúng tầng i. Đó chính là một circuit.")
source(sl, "Chi tiết: §4.6")

# ───────────────────────────────── 13 · ba bước
sl = S(); chrome(sl, "Tầng B · ba bước: đo → nối → thử", "Tầng B", 14)
for i, (h, b, fill, hc) in enumerate([
    ("1 · ĐO — IG trên gate",
     "Vặn mọi gate theo một tham số α\ntừ 0 lên 1, cộng dồn gradient.\n\n"
     "Một lượt backward trả về đạo hàm\ntheo mọi gate cùng lúc.\n\n"
     "M ≈ 20 lượt, bất kể bao nhiêu cạnh.", PALE, NAVY),
    ("2 · NỐI — vẽ lại rồi quét",
     "Vẽ mỗi node một lần cho mỗi tầng.\nMọi cạnh đi sang phải đúng một\ncột → DAG, không chu trình.\n\n"
     "Quét một lượt trái sang phải, mỗi\nô giữ lối tốt nhất tới nó.", PALE, NAVY),
    ("3 · THỬ — suff",
     "Bật đúng đường đó, tắt hết còn lại,\nchạy model thật.\n\n"
     "suff = phần cái bánh lấy lại được.\nNhận khi suff ≥ 0.95.\n\n"
     "1 lượt forward mỗi đường.", PEACH, ORANGE)]):
    x = 0.7 + i*4.1
    rect(sl, x, 1.08, 3.8, 0.36, hc)
    text(sl, x+0.04, 1.10, 3.7, 0.32, h, 14, WHITE, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    rect(sl, x, 1.44, 3.8, 2.00, fill)
    text(sl, x+0.12, 1.56, 3.56, 1.88, b, 13, INK, space=4)
    if i < 2: arrow_right(sl, x+3.85, 2.30, 0.2)
rect(sl, 0.7, 3.70, 11.93, 0.36, NAVY)
text(sl, 0.74, 3.72, 11.8, 0.32, "Bảng IG của một chuỗi sạch — chú ý hình chéo", 15, WHITE, True, anchor=MSO_ANCHOR.MIDDLE)
rect(sl, 0.7, 4.06, 11.93, 1.50, PALE)
text(sl, 1.1, 4.16, 11.0, 1.4,
     "cạnh            tầng 1    tầng 2    tầng 3    tầng 4\n"
     "X → C            0.11      0.01      0.00      0.00\n"
     "C → D₇           0.00      0.14      0.01      0.00\n"
     "D₇ → B           0.00      0.00      0.17      0.01\n"
     "B → D₁₅          0.00      0.00      0.00      0.21",
     13, INK, mono=True, space=2)
takeaway(sl, "IG là bộ sàng, không phải bằng chứng — nó nói «vặn mạnh thì tăng», không nói «chừng này là đủ».")
source(sl, "Chi tiết: §4.4, §4.6 – §4.10,  Kiểm tra 4 và 5")

# ───────────────────────────────── 14 · max-sum
sl = S(); chrome(sl, "Tầng B · vì sao cộng chứ không nhân", "Tầng B", 15)
text(sl, 0.7, 1.02, 12.0, 0.6,
     "Phản xạ thông thường: đặt cost = −log(w) rồi tìm đường rẻ nhất, như Dijkstra trên đồ thị xác suất.\n"
     "Ở đây làm vậy là sai loại — IG là phân rã CỘNG tính, −log rồi cộng thì tương đương NHÂN.", 16, INK, space=5)
rect(sl, 0.7, 1.85, 11.93, 0.36, NAVY)
text(sl, 0.74, 1.87, 11.8, 0.32, "Hai đường giá trị hoàn toàn bằng nhau,  c = 0.60", 15, WHITE, True, anchor=MSO_ANCHOR.MIDDLE)
rect(sl, 0.7, 2.21, 11.93, 1.30, PALE)
text(sl, 1.3, 2.32, 10.5, 1.2,
     "                      max-sum                     min(−log)\n"
     "A   2 hop     0.30 + 0.30  =  0.60          2 × 1.20  =    2.41\n"
     "B   6 hop     6 × 0.10     =  0.60          6 × 2.30  =  13.82",
     14, INK, mono=True, space=3)
text(sl, 0.7, 3.62, 11.9, 0.6,
     "max-sum:  hòa — đúng, chúng đáng giá như nhau.          min(−log):  A thắng gấp 5.7 lần — sai.", 15, ORANGE)
band(sl, 0.7, 4.20, 5.85, 1.55, "Phạt hai lần, tách ra bằng công thức",
     "chi phí(h)  =  h·(−log c)  +  h·log h\n"
     "                      do SỐ CẠNH       do c/h pha loãng\n"
     "                      tuyến tính          trên tuyến tính", 13)
band(sl, 6.78, 4.20, 5.85, 1.55, "Vì sao chí mạng ở đây",
     "Đường dài chính là thứ ta muốn tìm — multi-hop\nlà lý do tồn tại của cả tầng B.\n\n"
     "Hàm mục tiêu thiên vị đường ngắn sẽ chọn lối tắt\ngiả 2 hop và bỏ chuỗi suy luận thật 4 hop.", 13)
source(sl, "Chi tiết: §4.9")

# ───────────────────────────────── 15 · nối hai tầng
sl = S(); chrome(sl, "Nối hai tầng — chỉ một thứ đi qua ranh giới", "Nối tầng", 16)
node(sl, 1.1, 1.25, 4.3, 0.95, "Tầng A\nF = log P(câu trả lời)\nbaseline: xóa mọi đoạn", PALE, NAVY, 14, True)
node(sl, 7.9, 1.25, 4.3, 0.95, "Tầng B\nF_{B} = log p(node)\nbaseline: tắt mọi gate", PALE, NAVY, 14, True)
node(sl, 5.65, 1.42, 2.0, 0.6, "T", WHITE, NAVY, 20, True)
arrow_right(sl, 5.45, 1.72, 0.18); arrow_right(sl, 7.68, 1.72, 0.18)
text(sl, 5.3, 2.08, 2.7, 0.3, "tập đích", 13, GREY, align=PP_ALIGN.CENTER)
band(sl, 0.7, 2.75, 5.85, 2.20, "Hợp lệ  ✓",
     "Nối một tập.\n\n"
     "T là đầu ra của tầng A và là đầu vào\n(đích) của tầng B. Một tập hợp đi qua\nranh giới thì không mang theo\nđơn vị đo nào.", 14)
band(sl, 6.78, 2.75, 5.85, 2.20, "Vô nghĩa  ✗",
     "Nhân hai điểm số.\n\n"
     "F và F_B có mục tiêu khác nhau và\nbaseline khác nhau, nên chuỗi tính đầy\nđủ không bắc cầu. Không có bất biến\nnào nối tầng A với tầng B.", 14, fill=PEACH, hdr=ORANGE)
takeaway(sl, "Nối tập, đừng nhân điểm. Trên giao diện thì hiển thị hai cột riêng.")
source(sl, "Chi tiết: §6,  phụ lục §1")

# ───────────────────────────────── 16 · train
sl = S(); chrome(sl, "Dùng hai tầng làm tín hiệu train retriever", "Train", 17)
band(sl, 0.7, 1.08, 3.75, 2.15, "L_{node}   ← T",
     "«node nào đúng»\n\nThay nhãn cũ «đoạn có chứa\nchuỗi đáp án» — nhãn cũ đầy\ndương tính giả.\n\nKhông đổi kiến trúc.", 13)
band(sl, 4.78, 1.08, 3.75, 2.15, "L_{hop}   ← Π",
     "«đi tới đó bằng lối nào»\n\nGiám sát trung gian từng tầng,\nvá chỗ credit assignment bị\nđứt qua L bước.\n\n"
     "Thêm một head nhỏ mỗi tầng,\nvà BẮT BUỘC lấy mẫu âm.", 13)
band(sl, 8.87, 1.08, 3.75, 2.15, "L_{edge}   ← IG cạnh",
     "«cạnh nào nên mở»\n\nDạy trực tiếp cho gate phụ\nthuộc truy vấn.\n\nCần module gate — đổi\nkiến trúc, làm sau cùng.", 13)
text(sl, 0.7, 3.68, 11.9, 0.35,
     "Cả ba loss đọc từ một lượt forward, nên chi phí train không nhân ba. Cả ba cũng cập nhật chung một mạng.", 15, ORANGE)
band(sl, 0.7, 4.14, 5.85, 1.85, "Thứ tự bị ép bởi phụ thuộc dữ liệu",
     "L_node   cần T   có sẵn, không phụ thuộc model\n"
     "L_hop    cần Π   Π đào bằng IG trên model hiện tại\n"
     "L_edge   cần ĝ   cần kiến trúc gate\n\n"
     "Nên: train L_node tới hội tụ → chẩn đoán → mới thêm.", 13, mono=True)
band(sl, 6.78, 4.14, 5.85, 1.85, "Hai chỗ dễ sai",
     "Bộ lọc  chỉ áp L_{hop} / L_{edge} khi model đang xếp sai\nvà đường đã kiểm — điều kiện tự co lại khi model\nkhá lên, không cần lịch giảm λ.\n\n"
     "Miền cộng  L_{hop} cộng hết thì dương/âm ≈ 10 trên 10⁷.\nPhải lấy mẫu âm; âm khó nhất là đúng node, SAI tầng.", 13)
source(sl, "Chi tiết: §6.2 – §6.4")

# ───────────────────────────────── 17 · bảng chẩn đoán  [MỚI]
sl = S(); chrome(sl, "Bảng chẩn đoán — và nó cũng chính là nguồn nhãn", "Train", 18)
text(sl, 0.7, 1.00, 12.0, 0.35,
     "Hai đại lượng, đo hai thứ hoàn toàn độc lập nhau — nên xếp thành hai trục, không phải một danh sách.", 16, INK)
# 2x2
x0, y0, cw, ch = 2.55, 1.95, 4.55, 0.92
text(sl, x0, 1.52, cw, 0.36, "rank ≤ k_top(q)\nmodel lấy được D", 13, NAVY, True, PP_ALIGN.CENTER, space=0)
text(sl, x0+cw+0.1, 1.52, cw, 0.36, "rank > k_top(q)\nmodel bỏ sót D", 13, NAVY, True, PP_ALIGN.CENTER, space=0)
text(sl, 0.7, y0, 1.75, ch, "suff ≥ θ_{s}\ncó lối, đã kiểm", 13, NAVY, True, PP_ALIGN.RIGHT, MSO_ANCHOR.MIDDLE, space=0)
text(sl, 0.7, y0+ch+0.12, 1.75, ch, "suff < θ_{s}\nkhông lối nào đủ", 13, NAVY, True, PP_ALIGN.RIGHT, MSO_ANCHOR.MIDDLE, space=0)
cells = [("ổn — không cần làm gì", PALE, INK),
         ("LỖI MODEL  →  tập S_{k}\nđường tồn tại mà model không đi;  Π, ŵ là nhãn train", PALE, NAVY),
         ("nghi ĐƯỜNG TẮT\nmodel đúng mà không lối nào giải thích nổi", PEACH, ORANGE),
         ("LỖI ĐỒ THỊ\nbổ sung cạnh lúc xây graph", PALE, INK)]
for i, (s, fill, fg) in enumerate(cells):
    cx, cy = x0 + (i % 2)*(cw+0.1), y0 + (i//2)*(ch+0.12)
    rect(sl, cx, cy, cw, ch, fill, C(0x9A,0xA3,0xB2))
    text(sl, cx+0.1, cy, cw-0.2, ch, s, 13, fg, i in (1,2), PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE, space=1)
band(sl, 0.7, 4.12, 5.85, 1.45, "Nó chạy ở MỖI mốc, không phải một lần",
     "MỐC 0   đào T → train L_node tới hội tụ\n"
     "MỐC 1   ▸ bảng chẩn đoán ◂ → ô «lỗi model» = S₁\n"
     "              → train L_node + λ₁·L_hop\n"
     "MỐC 2   ▸ bảng chẩn đoán ◂ → thêm gate, L_edge", 13, mono=True)
band(sl, 6.78, 4.12, 5.85, 1.45, "Một lần chạy, hai cách đọc",
     "báo cáo   phân bố các ô = tiến độ của đợt train\n                  vừa xong\n\n"
     "nguồn nhãn   ô «lỗi model» chính là tập ví dụ\n                       để đào Π và ŵ cho đợt tới", 13)
takeaway(sl, "Chẩn đoán và đào nhãn là cùng một thao tác — chỉ khác cách đọc kết quả.")
source(sl, "Chi tiết: §6.1, §6.5")

# ───────────────────────────────── 18 · chi phí
sl = S(); chrome(sl, "Chi phí", "Chi phí", 19)
rows = [("Shapley chính xác trên 60 đoạn", "2^{60}  ≈  10^{18} lượt LLM", "bất khả thi", PEACH, ORANGE),
        ("Tầng A — gom cụm + greedy + kiểm", "64 + 57 + 7  =  128 lượt LLM", "đắt, nhưng offline", PALE, NAVY),
        ("Tầng B — IG + đường + xác minh", "≈ 150 lượt GFM 34M", "gần như miễn phí", PALE, NAVY)]
for i, (a, b, c, fill, fg) in enumerate(rows):
    y = 1.25 + i*1.10
    rect(sl, 0.7, y, 11.93, 0.92, fill)
    text(sl, 0.95, y, 5.1, 0.92, a, 16, INK, anchor=MSO_ANCHOR.MIDDLE)
    text(sl, 6.1, y, 3.7, 0.92, b, 17, fg, True, anchor=MSO_ANCHOR.MIDDLE)
    text(sl, 9.9, y, 2.6, 0.92, c, 14, GREY, align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
band(sl, 0.7, 4.72, 5.85, 1.42, "Quy tắc chọn IG hay ablation",
     "IG rẻ hơn  ⟺  N > 3M\n\nvới N là số đơn vị cần chấm, và điều kiện\nlà prompt/đồ thị không phình theo N.", 14)
band(sl, 6.78, 4.72, 5.85, 1.42, "Hệ quả thực tế",
     "Hai tầng chênh khoảng 140 lần.\n\nK là TRẦN: K = 40 cho 16 + 57 + 7 = 80 lượt.\n\n"
     "Cache T: nhãn (q → T) không phụ thuộc kiến trúc GFM.", 14)
source(sl, "Chi tiết: §7")

# ───────────────────────────────── 19 · phạm vi
sl = S(); chrome(sl, "Phạm vi đóng góp và những gì chưa chắc", "Kết", 20)
band(sl, 0.7, 1.08, 5.85, 2.75, "Từ literature",
     "Shapley value  ·  Integrated Gradients\n"
     "Mở rộng đa tuyến tính  →  tính chất c/h\n"
     "Owen value  (giá trị có cấu trúc liên minh)\n"
     "PDist / LOOP của Atlas  (JMLR 2023)\n"
     "GFM-RAG, G-reasoner  (NeurIPS 2025 / 2025)\n"
     "Mask cạnh để giải thích GNN", 14)
band(sl, 6.78, 1.08, 5.85, 2.75, "Đề xuất trong tài liệu này",
     "Shapley ở mức cụm để hạ 2^{K} xuống 2^{C}\n"
     "IG gắn vào gate cạnh thay vì embedding\n"
     "Gate riêng từng tầng, và gate phụ thuộc truy vấn\n"
     "max-sum trên DAG trải phẳng (thay min-cost)\n"
     "Greedy xóa cạnh + tỉa ngược cho hội tụ\n"
     "Tập đích Shapley làm nhãn train retriever", 14)
rect(sl, 0.7, 4.20, 11.93, 0.36, ORANGE)
text(sl, 0.74, 4.22, 11.8, 0.32, "Cần nói rõ", 15, WHITE, True, anchor=MSO_ANCHOR.MIDDLE)
rect(sl, 0.7, 4.56, 11.93, 1.70, PEACH)
text(sl, 0.95, 4.70, 11.4, 1.5,
     "·  Chưa có kết quả thực nghiệm. Đây là thiết kế, không phải báo cáo kết quả.\n"
     "·  Kỳ vọng cải thiện nằm ở multi-hop; single-hop gần như không đổi vì nhãn mức node đã đủ.\n"
     "·  Khẳng định «G-reasoner không có trọng số cạnh» dựa trên bản HTML của paper — cần đối chiếu bản gốc.\n"
     "·  L_hop không sửa được đường tắt giả: bộ lọc loại chính những ví dụ model đang xếp đúng.",
     14, INK, space=5)
source(sl, "Chi tiết: §1 «Phạm vi đóng góp», §6.7")

OUT = r"D:\ragproj\quy-gan-rag-hai-tang.pptx"
prs.save(OUT)
print(f"  da luu: {OUT}")
print(f"  {len(prs.slides.__iter__.__self__._sldIdLst)} slide, 16:9")
