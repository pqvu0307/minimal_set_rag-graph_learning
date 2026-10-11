# Tầng A bản Shapley — bản thay thế đã rời khỏi tài liệu chính

> Bản này **không còn là thiết kế hiện hành**. Tài liệu chính đã chuyển tầng A sang
> *tập đủ tối tiểu* (MWC / Deegan–Packel) — xem §3 của [tài liệu chính](pipeline-shapley-ig-graph.md).
>
> Giữ lại đây vì ba lý do:
>
> 1. **Nó sinh dữ liệu huấn luyện cho surrogate.** Vòng greedy ở §A.6 đánh giá hàng chục
>    tập con mỗi truy vấn; đó chính là dữ liệu vòng 0–1 của §3.4 bản mới.
> 2. **Nó có bộ bất biến chính xác** mà bản MWC không có — `Σ φ_c = F(E) − F(E')`.
>    Dùng được như phép đối chứng trên một mẫu nhỏ.
> 3. **Nó là phương án lui** nếu surrogate không đạt độ chính xác quanh ngưỡng.

Mục được đánh lại thành §A.x để khỏi lẫn với §3 của tài liệu chính.

## A. Tầng A — Tìm tập đích (bản Shapley) — Tìm tập đích

**Ký hiệu mục này giới thiệu**

```
q, D_k, K        truy vấn; đoạn (passage) thứ k; số đoạn ứng viên (20–100)
x, n, n_q, d     prompt [q, D₁..D_K]; số token của x; của q; chiều ẩn
y                câu trả lời mô hình sinh ra
span(D_k)        tập chỉ số token của đoạn D_k
span(C_c)        hợp các span(D_k) của mọi đoạn k trong cụm c
E, E', E_S       ma trận embedding; baseline xóa mọi đoạn; chỉ bật cụm trong S
F(E)             điểm mục tiêu tầng A
C, C_c           số cụm (4–8); cụm thứ c
κ(k), μ_c        ánh xạ đoạn → cụm; tâm cụm
v(S), φ_c        lợi ích liên minh cụm; giá trị Shapley của cụm
τ, C*, C⁻, ρ     ngưỡng cắt; cụm giữ / cụm bỏ; phần dư do cắt
v_c(R), v_c(∅)   lợi ích cụm c, có điều kiện trên các cụm khác; mốc khi R = ∅
Δ(S)             F(S) − F(q), mốc chung cho mọi đại lượng ở §3.6
Δ_full           Δ( q + mọi đoạn của C* ) — cái bánh mà greedy được phép hao
B, κ             ngân sách hao hụt (1−θ_g)·Δ_full; trần mỗi cụm (0.04)
θ_g, R_c         ngưỡng bảo toàn toàn cục (0.95); đoạn sống sót của cụm
full_c           v_c(C_c) − v_c(∅) — đại lượng của bản cũ, nay chỉ còn dùng để
                 giải thích vì sao ngưỡng cục bộ không đủ (§3.6)
ψ_k              Shapley cục bộ trong cụm
S_final(D_k)     điểm cuối cùng của đoạn D_k
T                tập đích = hợp các R_c                   ← đầu ra tầng A
```

### A.1 Thiết lập

```
x  = [ q, D₁, ..., D_K ]                prompt
E  ∈ R^(n×d)                            ma trận embedding
F(E) = log P(y | q, D₁..D_K)            điểm mục tiêu

E'[t] = E[t]   nếu t ≤ n_q              baseline: giữ truy vấn,
      = 0      nếu t > n_q                        xóa mọi đoạn
```

`F(E) − F(E')` là cái bánh: toàn bộ phần mà truy hồi đóng góp.

**Đơn vị của pool.** `K` là một **trần**, không phải con số cố định:

```
K_max      =  60          ≤ 45 chunk tài liệu  +  ≤ 15 tóm tắt community

K thực tế  =  số ứng viên vượt ngưỡng liên quan của từng nguồn, sau khử trùng lặp
```

Chỉ **hai** trong bốn loại node của QuadGraph vào pool: `document` và `community`. Hai loại
còn lại — `entity`, `attribute` — không phải đơn vị văn bản nên không bao giờ vào prompt;
vai trò của chúng ở §6.3.

Đừng độn cho đủ 60. Truy vấn hẹp thì cả BM25 lẫn GFM đều chỉ trả về ít ứng viên đáng xét;
nhồi thêm rác chỉ làm bẩn khâu gom cụm và tốn thêm lượt greedy. Mỗi nguồn có trần riêng
nhưng không có sàn.

Ba ràng buộc về đơn vị, cả ba đều dễ vi phạm lúc triển khai:

```
cỡ chunk     200–300 token, cắt theo ranh giới đoạn văn / mục, không cắt giữa câu
             quá dài -> quy gán thô; quá ngắn -> một sự kiện bị cắt đôi và sinh ra
             chữ ký multi-hop giả (hai nửa đều vô dụng khi đứng riêng)

không chồng lấn   chunk chồng nhau tạo ra dư thừa nhân tạo — đúng thứ §3.6 sinh ra
             để phát hiện. Nếu khâu truy hồi cần chồng lấn thì khử trùng lặp
             trước khi lập pool.

chung một cách chunk   pool và đồ thị phải dùng cùng một cách cắt. Nhánh BM25 và
             nhánh ngẫu nhiên chạy trên index riêng; nếu index đó cắt khác thì
             đoạn lấy về không có node nào trong đồ thị (xem tiền kiểm §6.1).
```

Chunk nhỏ hơn thì một mạch suy luận trải qua nhiều chunk hơn, tức đường trong đồ thị dài
hơn — phải tăng `L` tương ứng. Với bộ chuẩn (MuSiQue, 2Wiki) thì dùng luôn đơn vị cấp đoạn
có sẵn của dataset, đừng cắt lại.

### A.2 Gom cụm

```
κ(k) = argmin_c  ‖ emb(D_k) − μ_c ‖₂
μ_c  = (1/|C_c|) · Σ_{k ∈ C_c} emb(D_k)
```

Kéo chi phí Shapley từ `2^K` xuống `2^C`, và gom đoạn trùng ý vào một chỗ.

**`C` co theo `K`**, giữ cỡ cụm ổn định quanh 10:

```
C  =  clamp( round(K / 10),  4,  8 )

K = 60  →  C = 6,  cụm ~10        K = 40  →  C = 4,  cụm ~10
K = 30  →  C = 4,  cụm ~8  (chạm sàn)
```

Để `C` cố định mà `K` co lại thì cụm còn 2–3 phần tử, và việc gom cụm mất cả hai công dụng:
không giảm được gì đáng kể, mà cũng không còn gom được đoạn trùng ý vào một chỗ.

`K` không vào `2^C`, nhưng nó **vào chi phí greedy** ở §3.6 một cách tuyến tính — cụm to thì
mỗi vòng greedy phải quét nhiều ứng viên hơn. Đó mới là thứ chặn `K` lại, không phải Shapley.

### A.3 Trò chơi hợp tác

```
v(S) = F(E_S),      S ⊆ {1..C}

           ⎧ E[t]   nếu t ≤ n_q
E_S[t]  =  ⎨ E[t]   nếu t ∈ span(C_c), c ∈ S
           ⎩ 0      ngược lại

v(∅) = F(E')          v({1..C}) = F(E)
```

### A.4 Shapley mức cụm

```
                      |S|! · (C−|S|−1)!
φ_c  =   Σ           ───────────────────  · [ v(S∪{c}) − v(S) ]
      S ⊆ {1..C}\{c}         C!
```

```
φ_c lớn dương   cụm đóng góp thật
φ_c ≈ 0         mô hình bỏ qua cụm này
φ_c âm          cụm đẩy ngược câu trả lời
```

```
Kiểm tra 1:   Σ_{c=1..C} φ_c  =  F(E) − F(E')
```

**Vì sao không dùng leave-one-out cho rẻ?** loo chỉ hỏi *một* câu: bỏ cụm này khi mọi cụm
khác còn nguyên. Với bằng chứng dư thừa thì câu đó luôn trả lời "chẳng mất gì":

```
ba cụm nói cùng một ý, mỗi cụm đủ để trả lời đúng:
    loo(c) = v(tất cả) − v(tất cả trừ c) ≈ 0   cho cả ba
    → tổng loo ≈ 0, trong khi cái bánh thật sự là lớn

Shapley lấy trung bình trên mọi thứ tự, nên cụm đến đầu tiên được tính công đầy đủ
    → tổng luôn khớp cái bánh (Kiểm tra 1)
```

loo là marginal contribution ở *đúng vị trí cuối*; Shapley là trung bình trên *mọi vị trí*.
Dư thừa làm hai con số này lệch nhau rất xa.

### A.5 Cắt cụm

```
C* = { c : φ_c > τ }       C⁻ = {1..C} \ C*       ρ = Σ_{c ∈ C⁻} φ_c
```

### A.6 Greedy trong cụm

Hàm giá trị có điều kiện trên các cụm khác:

```
v_c(R) = F( q + R + mọi đoạn của các cụm khác trong C* )
```

Điều kiện hóa là bắt buộc. Nếu đo cụm cô lập thì một cụm chỉ có giá trị khi kết hợp với cụm
khác — đúng ca multi-hop — sẽ có `v_c` gần 0 ở mọi tập con; điều kiện dừng thỏa ngay sau
vòng đầu và greedy giữ một đoạn chọn theo nhiễu.

Vì `v_c(∅) ≠ 0`, mọi đại lượng phải đo theo phần tăng thêm. Bản trước lấy **mốc riêng cho
từng cụm** là `full_c = v_c(C_c) − v_c(∅)`; các con số đo từ những mốc khác nhau thì không
cộng trừ được với nhau, nên §3.6 dưới đây quy tất cả về **một mốc chung `F(q)`**.

#### Phải tỉa tuần tự

Tỉa song song thì hai đoạn ở hai cụm cùng nắm một thông tin sẽ triệt tiêu nhau: mỗi đoạn bị
cắt vì sự có mặt của đoạn kia, mà đoạn kia cũng đang bị cắt, kết quả là mất sạch. Tuần tự
khử được điều đó, với điều kiện **cụm chưa xử lý vào context ở dạng đầy đủ** — nhờ vậy ở mọi
thời điểm thông tin luôn được nắm bởi ít nhất một đoạn, hoặc đã sống sót hoặc chưa bị xét.

```
Δ(S)    =  F(S) − F(q)                  mọi đại lượng quy về cùng mốc F(q)
Δ_full  =  Δ( q + mọi đoạn của C* )
B       =  ( 1 − θ_g ) · Δ_full         ngân sách hao hụt cho cả bài, θ_g = 0.95

sắp C* theo φ_c giảm dần
R ← {}                                  # các cụm đã xử lý, đã tỉa

cho mỗi c theo thứ tự:
    ctx(R') =  q  +  ⋃ R_{c'}           c' đã xử lý, đã tỉa
                  +  ⋃ C_{c''}          c'' chưa xử lý, lấy đầy đủ
                  +  R'

    Δ_trước =  Δ( ctx(C_c) )            = Δ_sau của cụm liền trước
                                          (cụm đầu tiên thì bằng Δ_full)
    R_c ← rỗng
    lặp:
        k* = argmax  [ Δ(ctx(R_c + {k})) − Δ(ctx(R_c)) ]
             k ∈ C_c \ R_c
        R_c ← R_c + {k*}
        dừng khi  Δ_trước − Δ(ctx(R_c))  ≤  min( B, κ·Δ_full )      κ = 0.04

    Δ_sau =  Δ( ctx(R_c) )              ← chính lượt vừa chạy ở điều kiện dừng
    B     ←  B − ( Δ_trước − Δ_sau )    ← trừ phần thật sự mất, đo toàn cục
    R     ←  R + R_c
```

**Vì sao không dùng ngưỡng cục bộ `θ · full_c`.** Bản trước dừng khi cụm giữ được 95% đóng
góp *của riêng nó*. Nhưng `full_c` đo trong ngữ cảnh mà **các cụm chưa xử lý còn đầy đủ**:

```
full_c  =  F( q + R + C_c + C''_sau )  −  F( q + R + C''_sau )
```

Khi cụm `c` và một cụm phía sau cùng nắm một thông tin thì `v_c(∅)` đã có thông tin đó rồi,
nên `full_c` **thấp hơn** đóng góp thật của `c`. Mất 5% của một con số bị đánh giá thấp thì
thực tế mất nhiều hơn 5%, và `Σ full_c ≠ Δ_full`. Qua `|C*|` cụm thì trôi đủ xa để chốt
chặn §3.8 bật, kéo theo chạy lại toàn bộ.

Ngân sách toàn cục chữa tận gốc: phần hao hụt **không đo trong cụm nữa**, mà đo bằng điểm
số toàn cục trước và sau khi tỉa cụm đó. Mọi con số cùng mốc `F(q)` nên cộng trừ được với
nhau, và tổng hao hụt bị chặn bởi `B` theo cấu trúc.

Nó cũng tự thích nghi: cụm nào dư thừa thật thì tỉa mạnh và tiêu nhiều ngân sách, cụm sau
bị siết lại tương ứng. Trần `κ` giữ cho cụm đầu không ăn sạch — với `θ_g = 0.95` và
`|C*| = 3` thì `κ = 0.04` cho một cụm dùng tối đa 80% ngân sách. Không có trần thì cụm đầu
có thể tiêu hết, buộc mọi cụm sau giữ nguyên vẹn, và `|T|` phình quá `k_infer`.

**Không tốn thêm lượt nào.** `Δ_trước` là `Δ_sau` của cụm trước; `Δ_sau` chính là lượt vừa
chạy để kiểm điều kiện dừng. Ngược lại còn bỏ được `v_c(∅)` và `v_c(C_c)` mà bản cũ cần cho
mỗi cụm — chỉ thêm `F(q)` một lần cho cả bài.

Thứ tự theo `φ_c` giảm dần: khi hai cụm cùng nắm một thông tin, cụm được giữ nó là cụm mà
Shapley chấm cao hơn. Đây là greedy có thứ tự thứ ba trong tài liệu, cùng khuôn với §4.11
(thứ tự theo giá trị đường) và §5.3 (thứ tự theo depth) — phụ thuộc thứ tự, không tối ưu
toàn cục, đổi lại là chi phí đa thức.

Đánh đổi: không còn tỉa song song được nữa. Số lượt LLM không đổi nhưng thời gian treo tăng
theo `|C*|`.

#### Ví dụ

Cụm `C₁` có 4 đoạn `{a,b,c,d}`, được xử lý đầu tiên nên `Δ_trước = Δ_full = 3.00`:

```
B            =  0.05 × 3.00  =  0.15          ngân sách cho cả bài
κ·Δ_full     =  0.04 × 3.00  =  0.12          trần của riêng cụm này
Δ(ctx(∅))    =  0.60                          điểm khi bỏ hẳn C₁

vòng 1   thêm vào ∅:   a +1.90   b +1.85   c +1.20   d +0.30    → chọn a
         Δ(ctx) = 2.50      hụt = 3.00 − 2.50 = 0.50  >  0.12     chưa dừng

vòng 2   thêm vào {a}:  b +0.05   c +0.40   d +0.10              → chọn c
         Δ(ctx) = 2.90      hụt = 0.10  ≤  0.12                   dừng

giữ {a, c}    bỏ {b, d}    chi phí 7 lượt
B ← 0.15 − 0.10 = 0.05        ← hai cụm còn lại chỉ còn 1/3 ngân sách
```

`b` mạnh hơn `c` khi đứng một mình nhưng là bản sao của `a`. Chỉ phép thử phản thực mới thấy.

(Giá trị Shapley `φ₁ = 2.52` là đại lượng khác, dùng ở §3.7.)

#### Chẩn đoán miễn phí

Giữ thêm phép đo cô lập, không để tỉa mà để phát hiện:

```
so  v_c^cô-lập(C_c)  với  φ_c

    ≈ nhau     cụm tự đứng được một mình
    ≪ φ_c      cụm chỉ có giá trị khi kết hợp  →  đối tác multi-hop
```

Cả hai số đều đã tính rồi. Chữ ký `v_c ≪ φ_c` chỉ ra cặp cụm nào đang làm multi-hop với nhau.

### A.7 Chia φ_c cho các đoạn sống sót

`|R_c|` chỉ 2–3 nên Shapley chính xác được, và mọi `v_c(·)` đã tính ở §3.6 — không tốn thêm lượt nào:

```
                   |S|! · (|R_c|−|S|−1)!
ψ_k  =   Σ        ───────────────────────  · [ v_c(S∪{k}) − v_c(S) ]
      S ⊆ R_c\{k}         |R_c|!

                        ψ_k
S_final(D_k)  =  φ_c · ──────────────        k ∈ R_c
                       Σ_{j∈R_c} ψ_j
              =  0                           k ∉ R_c
```

Phải chuẩn hóa vì `Σψ_k = v_c(R_c)` (cụm chạy riêng) còn `φ_c` là đóng góp biên giữa các
cụm khác — hai đại lượng khác nhau.

Tiếp ví dụ §3.6, với `R_c = {a, c}` và mọi `v_c(·)` đã có sẵn:

```
ψ_a = ½[v(a) − v(∅)] + ½[v(ac) − v(c)] = ½(1.90) + ½(1.10) = 1.50
ψ_c = ½[v(c) − v(∅)] + ½[v(ac) − v(a)] = ½(1.20) + ½(0.40) = 0.80
                                                        Σψ = 2.30 = v_c(R_c)

S_final(a) = φ₁ · 1.50/2.30 = 2.52 × 0.652 = 1.643
S_final(c) = φ₁ · 0.80/2.30 = 2.52 × 0.348 = 0.877
                                             ─────
                                             2.520 = φ₁   ✓ Kiểm tra 2
```

Thấy rõ `Σψ = 2.30` khác `φ₁ = 2.52`. Nếu bỏ bước chuẩn hóa mà cộng thẳng `ψ` thì
Kiểm tra 2 gãy ngay.

```
Kiểm tra 2:   Σ_{k ∈ R_c} S_final(D_k) = φ_c
Kiểm tra 3:   Σ_k S_final(D_k) = [F(E) − F(E')] − ρ
```

### A.8 Đầu ra

```
T = ⋃_{c ∈ C*} R_c
```

Hai phép kiểm cuối, tổng `|T| + 1` lượt. Gọi `Δ(S) = F(q + S) − F(q)`:

```
chốt chặn toàn cục    Δ(T)  ≥  θ_g · Δ_full
                      ← giờ đúng theo cấu trúc: ngân sách B ở §3.6 không bao
                        giờ xuống dưới 0, nên tổng hao hụt luôn ≤ (1−θ_g)·Δ_full.
                        Nó thành phép kiểm, không còn là cơ chế sửa sai —
                        sai ở đây nghĩa là bug cài đặt, không phải ngưỡng chặt.

tỉa ngược             cho mỗi D ∈ T:
                          nếu  Δ(T \ {D}) ≥ θ · Δ(T):   T ← T \ {D}
                      ← bắt đoạn mà thứ tự xử lý vô tình giữ lại
                        (cùng vai với §4.11 pha 2)
```

---
