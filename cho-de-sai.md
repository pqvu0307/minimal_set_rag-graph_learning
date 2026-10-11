# Những chỗ dễ sai

Phụ lục của [Quy gán RAG hai tầng](pipeline-shapley-ig-graph.md).
Mọi tham chiếu dạng `§x.y` (hai cấp) là mục trong tài liệu chính;
`§n` (một cấp) là mục trong chính tài liệu này.

1. [Nối tập, đừng nhân điểm](#1-nối tập, đừng nhân điểm)
2. [Đừng cắt tỉa bằng điểm attribution](#2-đừng cắt tỉa bằng điểm attribution)
3. [Bất đối xứng giữa hai mức cắt](#3-bất đối xứng giữa hai mức cắt)
4. [Vì sao nhãn node chưa đủ](#4-vì sao nhãn node chưa đủ)
5. [Exposure bias](#5-exposure bias)
6. [Không verify được "có dùng thật"](#6-không verify được "có dùng thật")
7. [Ablation bằng vector 0 chỉ là xấp xỉ](#7-ablation bằng vector 0 chỉ là xấp xỉ)
8. [Mẫu số có thể tiến về 0](#8-mẫu số có thể tiến về 0)
9. [Gate học được có thể học thói quen của corpus](#9-gate học được có thể học thói quen của corpus)
10. [Tỷ phần IG không phải suff](#10-tỷ phần ig không phải suff)

---

## 1. Nối tập, đừng nhân điểm

`F` và `F_B` có mục tiêu khác nhau (`log P` của câu trả lời so với `log p` của node) và baseline khác nhau, nên chuỗi completeness không bắc cầu. Nối một *tập* thì hợp lệ; nhân hai *điểm số* thì vô nghĩa. Hiển thị hai cột riêng.

## 2. Đừng cắt tỉa bằng điểm attribution

```
phân bổ    "chia công trạng sao cho công bằng?"    Shapley / IG
cắt tỉa    "bỏ cái này đi thì mất gì?"             phản thực
```

```
Cụm C₃:  φ₃ = 2.40
  đoạn a: IG = 0.9
  đoạn b: IG = 0.8      ← ba đoạn gần trùng nhau
  đoạn c: IG = 0.7
         tổng = 2.40

Ngưỡng "IG < 1.0 thì cắt"  →  cắt cả ba  →  mất trắng 2.40
```

Cùng vấn đề ở mức cạnh, cộng thêm đứt kết nối:

```
[hỏi]--0.9--[A]      [B]--0.8--[đoạn]
               |       |
            cạnh 0.3 bị cắt → hai mảnh rời → giải thích 0%
```

Một cạnh điểm thấp vẫn có thể là cầu nối duy nhất. Phải tìm đường liền mạch, không phải tập cạnh điểm cao.

## 3. Bất đối xứng giữa hai mức cắt

```
Cắt cụm      →  mất tín hiệu thật  →  phải trừ:  Σ S_final = [F(E)−F(E')] − ρ
Cắt đoạn     →  không mất gì       →  không trừ, chỉ dồn lại:  Σ_{R_c} = φ_c
trong cụm
```

Đó là lý do greedy trong cụm chạy sau bước cắt cụm.

## 4. Vì sao nhãn node chưa đủ

```
hỏi:  "X sinh ở nước nào?"
đúng: [X] --sinh_ở--> [thành phố C] --thuộc--> [nước D]

model có thể học:  "truy vấn có chữ X, mà D hay xuất hiện gần X
                    trong dữ liệu train → cứ chấm D cao"

→ đúng trên tập train, sai trên truy vấn mới
→ nhãn node không phát hiện được, vì kết quả cuối vẫn đúng
```

`L_hop` ép model đi đúng lộ trình.

## 5. Exposure bias

Đoạn tốt nhất không lọt pool thì Shapley không bao giờ thấy nó. Chống bằng pool rộng đa nguồn (bm25 + GFM + ngẫu nhiên, `K=100` — không phải top-100 của chính GFM) và vòng lặp ngược (tài liệu chính §6.6).

## 6. Không verify được "có dùng thật"

Cả `φ_c` lẫn IG đều neo vào cùng một cái bánh:

```
F(E') = −0.20,  F(E) = −0.15   →  cái bánh = 0.05 nat  →  mọi đoạn ≈ 0
```

Không phân biệt được "bỏ qua" với "đọc rồi thấy trùng trí nhớ". Cần phép can thiệp thứ ba — đổi nội dung:

```
use_k = log P(trả lời cũ | đoạn gốc) − log P(trả lời cũ | đoạn đã sửa)
```

Chi phí 1 lượt forward mỗi đoạn. Để thành cột riêng cạnh `S_final`: `S_final` cao nhưng `use_k ≈ 0` là cảnh báo mô hình trả lời từ trí nhớ, đoạn chỉ là trang trí.

## 7. Ablation bằng vector 0 chỉ là xấp xỉ

Vector 0 nằm ngoài phân phối embedding tự nhiên nên `v(S)` dính nhiễu off-manifold. Shapley chính xác tuyệt đối *với trò chơi `v` như đã định nghĩa*, nhưng `v` chỉ là proxy của "bỏ đoạn đi". Thay thế: token `[MASK]`, hoặc cắt hẳn token rồi đánh lại chỉ số vị trí.

## 8. Mẫu số có thể tiến về 0

Ở §3.7 của tài liệu chính, nếu các đoạn trong cụm có `ψ` trái dấu thì tỷ phần bùng nổ hoặc lật dấu. Hai cách chữa, cả hai giữ nguyên Kiểm tra 2:

```
1. chia theo độ lớn:  |ψ_k| / Σ_j |ψ_j|
2. khi |Σ_j ψ_j| < ε: chia đều  φ_c / |R_c|
```

## 9. Gate học được có thể học thói quen của corpus

`g_e^(l)` vẫn inductive (không có ID node trong đầu vào MLP), nhưng nhãn IG sinh trên một corpus có thể dạy nó lối định tuyến riêng của corpus đó. Chống bằng cách đào nhãn trên nhiều corpus, hoặc giữ `λ₂` nhỏ. Đây là lý do `L_edge` nằm ở bước cuối trong thứ tự triển khai.

## 10. Tỷ phần IG không phải suff

```
tỷ phần IG   Σ IG trên đường / tổng IG      suy ra từ phân rã cộng tính
suff         bật đúng gate đó, chạy model    đo thật
```

Hai chỗ chúng lệch nhau:

```
1. model phi tuyến   →  68% tỷ phần IG có thể chỉ cho suff = 0.61
                        (IG tuyến tính hóa dọc một đường, model thì không)

2. tính chất c/h     →  nhánh nhiều cạnh có tỷ phần cao hơn nhánh ít cạnh
   (tài liệu chính §4.6, §4.10)        dù hai nhánh thiết yếu ngang nhau
```

```
tỷ phần IG  →  xếp hạng ứng viên     miễn phí
suff        →  quyết định            1 lượt forward
```

Đừng hiển thị tỷ phần IG cho người dùng như thể nó là mức độ quan trọng của nhánh.
