# Quy gán nguồn cho RAG hai tầng

Thiết kế và thi hành một pipeline quy gán nguồn hai tầng cho RAG:

```
tầng A   tìm TẬP ĐỦ TỐI TIỂU trong pool ứng viên      (MWC · Deegan–Packel · Banzhaf)
tầng B   tìm ĐƯỜNG trên đồ thị tri thức                (Integrated Gradients trên gate)
```

Chỉ `T` — tập đích của tầng A — đi qua ranh giới hai tầng. Không điểm số nào đi qua.

## Tài liệu

| file | nội dung |
|---|---|
| [`pipeline-shapley-ig-graph.md`](pipeline-shapley-ig-graph.md) | **tài liệu chính**, 53 trang — thiết kế đầy đủ cả hai tầng, và *vì sao* |
| [`tang-a-pipeline.md`](tang-a-pipeline.md) | tầng A chi tiết, 18 trang — từng công thức, và một ví dụ số chạy xuyên suốt |
| [`ke-hoach.md`](ke-hoach.md) | kế hoạch thi hành, 13 trang — thứ tự làm, **cổng quyết định**, ngân sách |
| [`tang-a-shapley.md`](tang-a-shapley.md) | bản Shapley cũ của tầng A — **lưu trữ**, giữ làm đối chứng |
| [`cho-de-sai.md`](cho-de-sai.md) | phụ lục những chỗ dễ sai |

Mỗi `.md` có `.tex` và `.pdf` đi kèm. `.tex` của `tang-a-pipeline` và `ke-hoach` **sinh tự động**
từ `.md` — sửa `.md` rồi chạy lại, đừng sửa `.tex` tay:

```bash
python build/mk_pipeline_tex.py ke-hoach "Kế hoạch thi hành --- tầng A" "thứ tự làm, cổng quyết định, ngân sách"
build/tools/tectonic.exe ke-hoach.tex
```

## Slide

`slides.tex` → `slides.pdf`, **37 slide beamer**, 16:9. Nguồn chia thành `build/sl_1..6.tex`
theo phần; ghép, biên dịch, rồi kiểm — đúng thứ tự này:

```bash
python build/mk_slides.py           # ghép, tự thêm [fragile] cho frame có verbatim
build/tools/tectonic.exe slides.tex
python build/checklayout.py         # ĐO bố cục trên PDF vừa sinh
```

Thứ tự quan trọng: `checklayout` phải chạy **sau** tectonic, không thì nó đo bản
biên dịch cũ.

**Thứ tự slide = thứ tự chạy của pipeline** (`c0`→`c7`, rồi `b1`→`b3`). Không slide nào
được dùng một ký hiệu trước slide khai sinh ra nó — `τ_q` xuất hiện ở `c4`, nên định
nghĩa «đủ» phải nằm sau đó, không phải trước.

Nhãn bước ở góc phải tiêu đề đi qua `\framesubtitle`, và dùng **đúng tên bước của
`pipeline-shapley-ig-graph.md`**. Template `frametitle` là tự viết, nên **phải** tự gọi
`\insertframesubtitle` — thiếu nó thì mọi nhãn âm thầm biến mất.

### Năm lỗi im lặng mà bộ kiểm tồn tại để bắt

Cả năm đều **biên dịch sạch**, không một dòng lỗi:

```
1  verbatim        beamer không xử lý verbatim trong frame thường
                   -> mk_slides.py tự thêm [fragile]
2  tràn lề         beamer chỉ ghi "Overfull vbox" vào log, không báo lỗi
                   -> checklayout đo trên PDF, cả tràn trang LẪN đè lên footer
3  lệch ô bảng     \color ở ĐẦU ô p{} sinh whatsit trước hộp đầu tiên
                   -> \vtop cao 0 -> cả ô tụt xuống ~6,6pt
                   -> chữa ở gốc: \leavevmode trong \hl \no \yes \dd
4  tabular trong   tabular mặc định căn GIỮA, nên cột chứa nó bị kéo lên
   column          -> \begin{tabular}[t]
5  chữ bị nuốt     \dd{tiếng Việt} bên trong align*/$...$ chạy ở font toán;
                   cmmi10 không có glyph có dấu -> "trùng lặp" in ra "trnglp"
                   -> không bao giờ đặt văn bản tiếng Việt trong math mode
```

`quy-gan-rag-hai-tang.pptx` (33 slide) là **bản cũ**, đã được `slides.tex` thay thế. Giữ lại
làm lịch sử. Đừng chạy `build/deck.py` — nó là bản Shapley 20 slide còn cũ hơn nữa.

## Code

[`stage0/`](stage0/) — hạ tầng của `GĐ 0`. Xem [`stage0/README.md`](stage0/README.md).

```bash
cd stage0
pip install pyyaml tiktoken bm25s sentence-transformers transformers torch openai
python test_logic.py       # PHẢI pass trước
python run_stage0.py       # 0.A → 0.J, mỗi bước tự báo PASS / THIEU / HOI
```

Trên Colab: mở `stage0/Stage0.ipynb`.

## Công cụ kiểm

```bash
python build/audit.py       # tham chiếu §x.y · ký hiệu chưa khai báo · md↔tex lệch
python build/checkdau.py    # chữ tiếng Việt thiếu dấu, trong 4 .tex và .pptx
python build/test_dau.py    # kiểm rằng ngoại lệ của checkdau không che mất lỗi thật
python build/checklayout.py # cỡ chữ · tràn · lệch ô · lệch cột (chạy sau tectonic)
```

Cả bốn phải sạch trước khi commit.

## Trạng thái

```
thiết kế     xong, 53 + 18 + 13 trang  ·  slide 37 beamer
GĐ 0 code    xong, test logic pass, chưa chạy trên hiện vật thật
GĐ 0.5       chưa chạy — $18, chạy cả bốn cổng trên test index
ngân sách    $1.120 ước tính (sinh chạy local qwen3-8b), trong đó $889 là
             indexing và nó là QUYẾT ĐỊNH SAU CỔNG, không phải chi phí đầu vào
ràng buộc    ~53 giờ GPU, không phải tiền
```

Bảy quyết định còn treo — xem *"Phải chốt trước khi giai đoạn 0 chạy"* trong `ke-hoach.md`.
