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

Slide: `quy-gan-rag-hai-tang.pptx` (33 slide). **Sửa tay** — đừng chạy `build/deck.py`, nó là
bản Shapley 20 slide cũ và sẽ xóa hết.

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
python build/checkdau.py    # chữ tiếng Việt thiếu dấu, trong .tex và .pptx
python build/test_dau.py    # kiểm rằng ngoại lệ của checkdau không che mất lỗi thật
```

Cả ba phải sạch trước khi commit.

## Trạng thái

```
thiết kế     xong, 53 + 18 + 13 trang
GĐ 0 code    xong, test logic pass, chưa chạy trên hiện vật thật
GĐ 0.5       chưa chạy — $945, chạy cả bốn cổng trên test index
ngân sách    $23.300 ước tính, trong đó $9.820 là indexing và nó là
             QUYẾT ĐỊNH SAU CỔNG, không phải chi phí đầu vào
```

Bảy quyết định còn treo — xem *"Phải chốt trước khi giai đoạn 0 chạy"* trong `ke-hoach.md`.
