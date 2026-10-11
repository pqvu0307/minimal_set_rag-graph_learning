# Giai đoạn 0 — hạ tầng

Code cho `GĐ 0` của [ke-hoach.md](../ke-hoach.md). Tốn công nhiều nhất, tốn tiền ít nhất
(`$0` LLM trừ vài lượt thử ở `0.E`/`0.F`). Không giai đoạn nào sau chạy được nếu chỗ này sai.

## Đã chốt

```
retriever   rmanluo/G-reasoner-34M, checkpoint PRETRAIN
            KHÔNG chạy sft_training — pool phải độc lập với tín hiệu đang đào
corpus      train set HotpotQA + MuSiQue + 2Wiki  ·  tiếng Anh
chunk       một entry documents.json = một document node = một đoạn Wikipedia
            KHÔNG cắt lại — vì T phải là tập NODE, tầng B định tuyến tới node
pool        node type ∈ {document, summary};  entity không vào prompt
cost(S)     overhead + Σ len(k)   — không phải 250·|S|, vì document (~100) và
            summary (~600) là hai lớp độ dài khác nhau
```

## Chạy

```bash
pip install pyyaml tiktoken bm25s sentence-transformers transformers torch openai
python run_stage0.py            # chạy hết
python run_stage0.py 0.A 0.B    # chạy vài bước
```

Trên Colab: mở `Stage0.ipynb`.

## Các bước

| bước | file | trả lời gì |
|---|---|---|
| `0.A` | `s0a_inventory.py` | bốn câu hỏi hiện vật: đồ thị có sẵn? constructor nào? một đồ thị hay một mỗi bộ? đếm |
| `0.B` | `s0b_chunks.py` | **đo** độ dài đoạn và chồng lấp — không quy định |
| `0.C` | `s0c_pool.py` | pool ba nguồn trên **đúng** tập node, rerank theo nguồn, hạn ngạch 20/15/5 |
| `0.E` | `s0e_judge.py` | `v(S)` với logprob token đầu, cache băm nội dung, và **đo prompt caching có ăn** |
| `0.F` | `s0f_decompose.py` | chẻ câu hỏi; MuSiQue dùng phân rã gán sẵn, và dùng nó để **đo LLM có sót hop** |
| `0.G` | `s0g_nli.py` | bảng phủ, `adq`, cửa thoát, nghiệm phủ — 0 lượt LLM |
| `0.J` | `s0j_splits.py` | chia tập và **khóa** lại; chọn split hiệu chuẩn không phải train |

Mỗi bước ghi một JSON vào `reports/`. Không bước nào crash vì thiếu hiện vật — nó báo
`THIEU` và nói phải làm gì.

## Ba chỗ dễ làm sai, và chúng nằm ở đâu

**1 · `cost(S)` sai thì không cổng nào bắt được.** Mọi cổng đều so với chính công thức đó.
`0.B` đo độ dài thật và in lại bảng chi phí; `0.E` so `est` với `real` của API.

```
tài liệu đang ghi:   500  ·  750  ·  3.500  ·  10.500      (v(∅) · v({k}) · v(R) · v(C))
0.B in ra con số thật thay vào
```

**2 · Judge phải trả số liên tục.** Thang `1–5` làm hàng chục tập con hòa nhau, `S` và
`S∖{c}` không phân biệt được, và *"tối tiểu"* mất nghĩa. `0.E` đếm số giá trị `v` phân biệt
trên 6 tập con thử và báo động nếu quá ít.

**3 · Cache phải băm NỘI DUNG, không băm tên.** Chunk lại corpus là tên va nhau và cache trả
sai. `s0_common.cache_key()` băm nội dung đã sắp.

## Cần người quyết định

```
#2  model judge      config.judge.gen_model / judge_model đang là PHAI_DIEN
                     endpoint PHẢI hỗ trợ logprobs + top_logprobs
#3  model NLI        mặc định MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
                     0.G in độ chính xác trên ví dụ có đoạn bẫy — xem rồi quyết
#5  ba trích dẫn     Deegan–Packel 1978 · Banzhaf 1965 · Holler–Packel 1983
0.A-1  đồ thị        phát hành sẵn, hay tự dựng trên 2.972.931 document?
0.A-2  constructor   paper nói có BA — phải chốt MỘT
0.A-3  phạm vi       một đồ thị cho cả ba bộ, hay một mỗi bộ?
```

`0.A-1` là chênh lệch công sức lớn nhất còn lại trong cả kế hoạch.

## Hai điều `0.J` canh

```
RÒ RỈ      loss giám sát của G-reasoner train trên "labeled relevant nodes",
           mà đó chính là supporting_documents — chính chuẩn ngoài của GĐ 1
           ->  GĐ 1 PHẢI chạy trên split checkpoint CHƯA train
           ->  và báo cáo mỗi con số HAI LẦN: pool chỉ-BM25 vs pool đầy đủ
               chênh lệch chính là cỡ của rò rỉ

HELD-OUT   218.200 / 277.839 = 79% bị đào nhãn
           phải khóa phần còn lại TRƯỚC khi GĐ 2 chạy
           không khóa thì tầng B train-on-train và mọi con số phồng giả
```

File khóa `reports/splits.lock.json` **không được ghi lại**. Chạy lại chỉ đọc và kiểm; muốn
ghi lại phải truyền `force=True` có ý thức.

## Một bug `test_logic.py` đã bắt được

`minimal_covers` ban đầu bắt đầu duyệt từ cỡ `J` (mỗi mảnh một đoạn), nên **bỏ sót** đúng ca
một node phủ nhiều mảnh — tức ca một `summary` bao nhiều `document`. Đã sửa thành duyệt từ cỡ
`1`. Đây chính là ca làm phát sinh câu hỏi `§3.8 parsimony` ở trên.
