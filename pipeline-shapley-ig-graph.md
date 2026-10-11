# Quy gán RAG hai tầng: tập đủ tối tiểu (sinh) + IG trên đồ thị (truy hồi)

Tài liệu thiết kế. Toán viết dạng plain-text trong khối mã để đọc được ở mọi trình xem, không cần renderer LaTeX.

**Mục lục**

1. [Tổng quan](#1-tổng-quan)
2. [Quy ước ký hiệu](#2-quy-ước-ký-hiệu)
3. [Tầng A — Tìm tập đích](#3-tầng-a--tìm-tập-đích)
4. [Tầng B — Tìm đường trên đồ thị](#4-tầng-b--tìm-đường-trên-đồ-thị)
7. [Nhiều đích và quan hệ phụ thuộc](#5-nhiều-đích-và-quan-hệ-phụ-thuộc)
8. [Nối hai tầng và train](#6-nối-hai-tầng-và-train)
9. [Chi phí](#7-chi-phí)
10. [Bất biến phải kiểm](#8-bất-biến-phải-kiểm)
11. [Thứ tự triển khai](#9-thứ-tự-triển-khai)

---

## 1. Tổng quan

```
Tầng A — sinh          "trong các đoạn đã có, đoạn nào thật sự tạo ra câu trả lời?"
                        công cụ: liệt kê tập đủ tối tiểu bằng surrogate
                        đầu ra:  tập đích T

Tầng B — truy hồi      "đáng lẽ retriever phải đi lối nào để tới các đoạn đó?"
                        công cụ: IG trên gate cạnh + quy hoạch động trên DAG
                        đầu ra:  cây đường đi Π
```

```
truy vấn
   │
   ├─ truy hồi rộng (K ≤ 60, đa nguồn)
   │
   ├─ Tầng A ─────→  T = {D₂, D₇, D₁₅}
   │                      │
   │                      └── làm đích cho tầng B
   │
   ├─ Tầng B ─────→  Π = cây đường đi từ V_q tới từng D
   │
   ├─ kiểm chứng bằng ablation
   │
   └─ nhãn train  =  (q, T, Π, depth, w)      ← chi tiết ở §6
                              │
                              ▼
                         train GFM
                              │
                              ▼
            Online: truy vấn → GFM giỏi hơn → đoạn → LLM → trả lời
```

### Vì sao cần hai tầng

| | Câu hỏi | Công cụ đúng | Vì sao |
|---|---|---|---|
| A | "đoạn nào cần?" | tập đủ tối tiểu | dư thừa hiện ra thành nhiều MWC, không cần gom cụm |
| B | "đi lối nào?" | IG | chi phí không phụ thuộc số cạnh cần chấm |

### Vai của graph foundation model

```
LLM     hàng tỷ tham số   biết ngôn ngữ + tri thức    → tầng A
GFM     34M               biết cách đi                 → tầng B
đồ thị  dữ liệu           chứa tri thức cụ thể         → thay trí nhớ tham số
```

GFM không nhớ thực thể nào. Khởi tạo của nó là `h_v⁽⁰⁾ = MLP(emb(v), emb(q))` cho seed và `emb(v)` cho phần còn lại — `emb(v)` là embedding văn bản, không phải tham số gắn với ID. Nhờ vậy nó chạy được trên đồ thị mới, thực thể mới, không cần train lại. Đó là nghĩa của chữ "foundation" ở đây.

Hệ quả: 34M tham số đủ, nên tầng B gần như miễn phí so với tầng A — và IG chạy nổi trên hàng nghìn cạnh.


### Phạm vi đóng góp

Tài liệu này là thiết kế, chưa có kết quả thực nghiệm. Phần dưới tách rõ cái gì lấy từ
literature, cái gì là đề xuất.

```
từ literature                        đề xuất Ở đây
─────────────────────────────────────────────────────────────────────────────
Shapley value [1]                    bản thay thế, đã tách sang file riêng
Deegan–Packel, PGI [4,6]             credit chia đều trong tập đủ tối tiểu (§3.9)

Integrated Gradients [2]             gắn vào gate cạnh của GNN thay vì embedding
                                     đầu vào (§4.4)

mở rộng đa tuyến tính [3]            dùng tính chất c/h làm phép kiểm nội bộ
                                     (Kiểm tra 5, §4.5)

Banzhaf [5]                          bản có dấu: β < 0 bắt đoạn đẩy ngược (§3.7)

PDist / LOOP của Atlas [7,8]         thay target bằng tập đủ tối tiểu;
                                     thêm w(D) cho đích phụ thuộc nhau (§6.2)

GFM-RAG, G-reasoner [9,10]            chèn gate phụ thuộc truy vấn (§4.2 chế độ 2);
                                     L_hop, L_edge (§6.3)

mask cạnh để giải thích GNN [9,10]   gate riêng từng tầng, không phải một gate
                                     chung cho mỗi cạnh (§4.3)

                                     ─── thuần đề xuất ───
                                     max-sum trên DAG trải phẳng, thay min-cost (§4.9)
                                     greedy xóa cạnh + tỉa ngược cho hội tụ (§4.11)
                                     thứ tự theo depth cho đích phụ thuộc (§5.3)
                                     vòng lặp ngược R(T) \ R(V_q) (§6.6)
```

**Vị trí so với Atlas.** Atlas [8] đưa ra ba cách lấy tín hiệu từ LM để train retriever.
Nhìn qua lăng kính Shapley thì chúng là ba *vị trí* khác nhau của cùng một đại lượng:

```
PDist     log p(y | q, d_k)                    marginal khi d_k đến đầu tiên
LOOP      v(tất cả) − v(tất cả trừ d_k)        marginal khi d_k đến cuối cùng
Shapley   trung bình trên mọi vị trí
```

Atlas chọn PDist vì ổn định hơn EMDR²/ADist và rẻ hơn LOOP. Tài liệu này lập luận rằng
với bằng chứng dư thừa thì LOOP tụt về 0 (xem [Những chỗ dễ sai](cho-de-sai.md) §2),
còn với multi-hop thì PDist tụt về 0 (phụ lục §4) — tức cả hai đầu mút đều lệch, mỗi cái
lệch ở một loại truy vấn.

Đề xuất ở đây không đứng ở đầu mút nào: thay vì hỏi *"bỏ đoạn này thì mất gì"* ở một vị trí
cố định, nó hỏi *"tập nào là đủ và tối tiểu"* — và trả lời bằng **nhiều** tập, nên dư thừa
hiện ra thành bội số thay vì bị triệt tiêu.

---

## 2. Quy ước ký hiệu và khái niệm nền

### 2.1 Lý thuyết trò chơi hợp tác — bảy khái niệm, một ví dụ

Tài liệu mượn công cụ từ lý thuyết trò chơi hợp tác. Mục này giải thích đủ để đọc tiếp, không
giả định người đọc biết trước gì cả.

```
người chơi      một đoạn văn trong pool
liên minh S     một tập đoạn.  n đoạn thì có 2^n liên minh
hàm đặc trưng   v(S) = liên minh này làm được gì
                     = chất lượng câu trả lời khi chỉ đưa S cho model
```

Hai dòng cuối là toàn bộ cầu nối giữa RAG và lý thuyết trò chơi: **đoạn văn là người chơi,
chất lượng câu trả lời là tiền thưởng.**

#### Từ trò chơi có giá trị sang trò chơi thắng/thua

`v(S)` là số thực. Đặt một ngưỡng `τ` thì nó thành **trò chơi đơn giản** — chỉ còn thắng hoặc
thua:

```
S thắng  ⟺  v(S) ≥ τ          "chừng này đoạn là đủ để trả lời đúng"
```

Và khái niệm trung tâm của cả tầng A:

```
liên minh thắng tối tiểu  (minimal winning coalition, MWC)
      thắng,  mà bỏ bất kỳ thành viên nào cũng thành thua
```

#### Một ví dụ chạy qua hết

Ba đoạn `a, b, c`. Giả sử:

```
v(∅)  thua      v(a)  thua      v(b)  thua      v(c)  thua
v(ab) thắng     v(ac) thắng     v(bc) thua      v(abc) thắng
```

```
MWC:   { a,b }  và  { a,c }
       {a,b,c} thắng nhưng không tối tiểu — bỏ c vẫn thắng
```

Từ đó đọc ra ba chỉ số, và cả ba đều nói cùng một điều: **`a` là đoạn then chốt.**

```
PGI_i          tỉ lệ MWC có chứa i          (Public Good Index)
   PGI(a) = 2/2 = 1      PGI(b) = PGI(c) = 1/2

Deegan–Packel  chia đều trong mỗi MWC, rồi trung bình trên các MWC
   a:  ½·(½ + ½) = ½          b:  ½·½ = ¼          c:  ¼

swing          thêm i vào S làm S từ thua thành thắng
Banzhaf β_i    tỉ lệ swing trên mọi S không chứa i
   a:  {b}→{ab} ✓   {c}→{ac} ✓   {b,c}→{abc} ✓   ∅→{a} ✗   →  β_a = ¾
   b:  {a}→{ab} ✓   còn lại ✗                               →  β_b = ¼
   c:  đối xứng với b                                        →  β_c = ¼
```

#### Shapley, và vì sao tài liệu này không dùng nó

```
Shapley φ_i    đóng góp biên trung bình của i trên mọi thứ tự xuất hiện
               Σ_i φ_i = v(toàn bộ) − v(∅)      ← tiên đề đầy đủ (completeness)
```

Shapley trả lời *"chia công thế nào"*. Nó công bằng theo tiên đề, nhưng cần duyệt `2^n` liên
minh — với `n = 40` là bất khả thi. Và câu ta thật sự cần không phải là chia, mà là **chọn**:
*tập nào là đủ*. Đó là lý do tài liệu chuyển sang MWC (§3), và giữ bản Shapley ở
[file riêng](tang-a-shapley.md).

#### Hai khái niệm phụ, dùng ở §3.8

```
trội Pareto    A trội B nếu A tốt hơn hoặc bằng B ở mọi tiêu chí,
               và tốt hơn hẳn ở ít nhất một.  B bị trội thì loại thẳng.

đơn điệu       thêm người chơi không bao giờ làm liên minh thắng thành thua.
               Lý thuyết chuẩn hay giả định điều này — ta thì không,
               vì đoạn gây nhiễu làm v giảm.  Xem §3.2.
```

#### Và một khái niệm từ phía quy gán, dùng cho tầng B

```
tiên đề đầy đủ (completeness)
      tổng các phần quy gán  =  đúng bằng hiệu giữa "có tất cả" và "không có gì"

baseline (mốc)
      trạng thái "không có gì" — tầng A là xóa mọi đoạn, tầng B là tắt mọi gate
```

Completeness là thứ làm mọi con số đọc được như *"chiếm bao nhiêu phần trăm"*, và là nền của
mọi phép kiểm trong §8.

### 2.2 Quy ước ký hiệu

Mỗi mục lớn mở đầu bằng một bảng nhỏ chỉ liệt kê ký hiệu mục đó giới thiệu. Ở đây chỉ
nêu quy ước chung và ba chỗ dễ nhầm.

```
chỉ số trên  ^(l)     luôn là tầng GNN            l = 0..L
chỉ số dưới  _k, _i   luôn là chỉ số đoạn         k, i = 1..n
             _e       cạnh đồ thị
dấu mũ       ĝ, ŵ     đại lượng đã học hoặc đã chuẩn hóa
dấu phẩy     E'       baseline — trạng thái "đã tắt"
```

```
T  vs  n      T là tập đích (§3.10);  n là số ứng viên sau rerank (§3.1)
S  vs  S*     S là một tập con bất kỳ;  S* là MWC được chọn (§3.8)
E  vs  𝓔      E = ma trận embedding (§3);  𝓔 = tập cạnh đồ thị (§4)
              hai nghĩa không bao giờ xuất hiện trong cùng một công thức
h  vs  h_v    h trần = số hop của một đường (§4.5)
              h có chỉ số dưới (h_v, h_r, h_q) = embedding (§4.2)
```

---

## 3. Tầng A — Tìm tập đủ tối tiểu

**Bài toán.** Model đã trả lời xong với `K` đoạn truy hồi. Đoạn nào **thật sự** tạo ra câu
trả lời đó?

**Mục tiêu.** Không phải chia công cho từng đoạn, mà tìm các **tập đoạn đủ và tối tiểu**:
bỏ bất kỳ phần tử nào thì không còn đủ.

**Vì sao đây mới là câu hỏi đúng.** Shapley trả lời *"mỗi đoạn đóng góp bao nhiêu"* — bài
toán **chia**, và nó cần `2^K` liên minh. Câu ta thật sự cần là *"tập nào là đủ"* — bài toán
**chọn**. Đổi đối tượng thì đổi luôn cả chi phí lẫn cách đọc kết quả.

```
Shapley   φ_i ∈ R cho mọi đoạn      một con số mỗi đoạn, cộng lại bằng cái bánh
MWC       W = { S₁, ..., S_m }      nhiều tập, mỗi tập tự nó là một lời giải thích
```

Bản Shapley vẫn còn trong [Tầng A bản Shapley](tang-a-shapley.md) — nó sinh dữ liệu huấn
luyện cho §3.5 và là phương án lui.

**Ký hiệu mục này giới thiệu**

```
K, C             số đơn vị trong pool (≤ 60); tập ứng viên { c_1, ..., c_n }
n                |C| sau rerank, 10–30
r_i              điểm rerank của đơn vị i — chỉ dùng để phá hòa ở §3.8
v(S)             log P( câu trả lời | q, S )
τ_q              ngưỡng đủ của truy vấn q (§3.3)
W                tập các MWC tìm được
f_θ(q,S)         surrogate — xấp xỉ 1[ v(S) ≥ τ_q ]
β_i, PGI_i       Banzhaf có dấu; số MWC chứa i
S*               MWC được chọn để báo cáo
C, m             số cụm; số ứng viên được chấm mỗi vòng greedy (§3.4)
mảnh             sub-claim — câu hỏi chẻ thành các khẳng định 1-hop (§3.2)
adq(S)           S có phủ hết mảnh không — CÓ BẰNG CHỨNG, khác với v (§3.2)
bảng phủ         mảnh nào được đoạn nào entail; tính 0 lượt LLM (§3.2)
r                SỐ VÒNG greedy — KẾT QUẢ, không phải núm vặn:
                 r = |S| lúc greedy dừng, tức cỡ tập đủ nó tìm được
spread           max_k v({k}) − v(∅) — singleton có tín hiệu hay không
```

### 3.1 Thiết lập

```
sinh    â_S   =  generate( q, S )                      ~80 token
chấm    v(S)  =  P_judge( "đúng" | q, â_S, y )  ∈ [0,1]   đọc logprob token đầu,
                                                           không decode câu trả lời judge
v(∅)          =  điểm khi không có đoạn nào — chỉ tri thức tham số
```

**Vì sao không dùng log-likelihood của đáp án vàng.** Nó rẻ hơn nhưng đo sai thứ: model trả
lời `"Paris"` trong khi `y = "Paris, France"` là **đúng về ngữ nghĩa** mà `log P(y)` lại
thấp. Và sai lệch đó không triệt tiêu khi so giữa các tập con — ca xấu là đoạn tốt làm model
trả lời dứt khoát (gold dài hơn nên điểm thấp), còn đoạn mơ hồ làm nó vòng vo và vô tình
khớp gold hơn. **Thứ tự lật.**

**Vì sao judge phải trả số liên tục, không phải thang 1–5.** Đây mới là chỗ nguy hiểm, và nó
không liên quan tới tiền:

```
judge trả 1–5   ->  v chỉ có 5 mức  ->  hàng chục tập con hòa nhau
                ->  không phân biệt được S với S \ {c}
                ->  "tối tiểu" mất nghĩa, và greedy không tìm được chỗ cắt τ_q
```

Nên hỏi judge một câu yes/no rồi **đọc xác suất của token đầu**, chứ đừng bắt nó chấm điểm.
Toàn bộ §3.2–3.4 dựa vào `v` liên tục để dò ranh giới.

**Chi phí so với log-likelihood: khoảng +12%, không phải vài chục lần.**

```
                      input                output       quy đổi
log-likelihood        5.500 + gold 30      0            ≈ 5.530
sinh + judge          5.500                ~80  (×4)    ≈ 5.820
                      + lượt judge 400     1            ≈   400
                                                         ───────
                                                         ≈ 6.220
```

Chi phí bị chi phối bởi việc **nạp 5.500 token prompt**, mà cả hai cách đều phải trả. Sinh
thêm 80 token và một lượt judge ngắn chỉ là phần rìa. Nhờ vậy ngân sách theo từng truy vấn ở
§3.4 vẫn giữ được.

Ở pha 1 (§3.11) thì `g_θ` chỉ cần khớp **trong** một truy vấn, vì truy vấn nào cũng đã trả
vòng 0. Ở pha 2 thì không — nó phải tổng quát hóa **qua** truy vấn, và đó là rủi ro lớn nhất
của chế độ quy mô lớn. Cổng `p` ở §3.11 tồn tại để đo đúng chỗ đó trước khi cam kết ngân
sách.

**Phân tầng (tùy chọn).** Nếu muốn tiết kiệm nốt 12% đó thì dùng log-likelihood làm proxy
cho vòng 0–2 và chỉ chạy judge ở vòng 3:

```
hiệu chuẩn trên mẫu 100 tập con:   ρ = Spearman( v_cheap, v_judge )
    ρ ≥ 0.7    giữ phân tầng
    ρ < 0.7    bỏ proxy, chạy judge cho cả vòng 0–2 — chỉ đắt thêm 12%

chỉ số theo dõi:  tỷ lệ S* bị vòng 3 bác.  Cao nghĩa là proxy đang dẫn f_θ đi lạc.
```

Phân tầng ở đây là **tối ưu hóa tùy chọn, không phải điều kiện sống còn** — proxy dở thì bỏ
nó, chứ không phải đi vay rủi ro tổng-quát-hóa-qua-truy-vấn để bù.

Cần có đáp án vàng `y`, nên cách này dành cho khâu **đào nhãn offline**.

**Đơn vị của pool, và hai con số đừng lẫn.**

```
K ≤ 60     pool thô, đa nguồn, sau khử trùng lặp
   ↓ rerank theo từng nguồn
n  = 40    20 BM25  +  15 GFM  +  5 ngẫu nhiên      ← đây mới là thứ vào §3.4
```

Chỉ **một** trong ba loại node vào pool làm ứng viên. `nodes.csv` của gfm-rag dùng
`type ∈ {entity, document, summary}`, với `attribute` là **cột** trên node chứ không phải loại
riêng:

```
document   đơn vị văn bản  ->  VÀO pool
summary    đơn vị văn bản  ->  vào pool khi có (lớp community của QuadGraph)
entity     KHÔNG phải đơn vị văn bản  ->  không bao giờ vào prompt; vai ở §6.3
attribute  là cột, không phải node
```

Paper gọi bốn *lớp* — `Attribute` · `Knowledge graph` · `Document` · `Community` — còn code
hiện ba *loại node* cộng cột thuộc tính. Hai cách mô tả cùng một thứ; ở đây dùng tên của code,
vì đó là thứ ta đọc ra từ `processed/stage1/nodes.csv`.

**Rerank theo từng nguồn, không gộp chung.** Đây không phải chi tiết nhỏ. Pool đa nguồn sinh
ra để bắt thứ GFM đang bỏ sót; nhưng nếu reranker là cross-encoder huấn luyện trên cùng loại
dữ liệu với retriever thì nó **chia sẻ điểm mù** — đoạn mà BM25 lôi về đúng vì GFM không
thấy sẽ bị nó loại ngay. Thế là quay về chỗ xuất phát, và ô «lỗi đồ thị» của bảng chẩn đoán
lại không bao giờ khác rỗng. Giữ hạn ngạch riêng cho mỗi nguồn thì không nguồn nào bị xóa
sổ.

**Đơn vị chunk do công cụ quyết định, không phải ta chọn.** `documents.json` của gfm-rag là
`{ "tên tài liệu": "nội dung" }`, và **mỗi entry là một document node** — framework không cắt
chunk bên trong. Với HotpotQA / MuSiQue / 2Wiki thì đơn vị là **đoạn Wikipedia có sẵn của
dataset**.

```
pool ứng viên      =  các node có type == "document"
index BM25         =  dựng trên ĐÚNG văn bản của các node đó
bộ bốc ngẫu nhiên  =  bốc từ ĐÚNG tập đó
```

Nhờ vậy ràng buộc *"pool và đồ thị phải dùng cùng một cách cắt"* trở thành **đúng theo cấu
trúc**: mọi đoạn trong pool đều là một node, vì nó lấy từ đồ thị ra. Tiền kiểm §6.1 (*"đoạn
lấy về không có node nào trong đồ thị"*) không thể fail nữa.

Hai ràng buộc còn lại vẫn phải kiểm, vì chúng là tính chất của dataset chứ không của ta:

```
độ dài       đoạn Wikipedia dài bao nhiêu?  cost(S) ≈ 500 + 250·|S| dựa vào con số 250,
             mà đó là GIẢ ĐỊNH — phải đo phân bố thật, rồi tính lại mọi bảng chi phí.
             quá dài -> quy gán thô;  quá ngắn -> một sự kiện bị cắt đôi và sinh ra
             chữ ký multi-hop giả (hai nửa đều vô dụng khi đứng riêng)

chồng lấn    các đoạn có chồng nhau không?  nếu có thì khử trùng lặp TRƯỚC khi lập pool,
             vì chunk chồng nhau tạo dư thừa nhân tạo — đúng thứ §3.6 sinh ra để phát hiện
```

Pool lấy từ **nhiều nguồn** — BM25 + GFM + ngẫu nhiên. Lý do ở §6.6: nếu pool chỉ từ GFM thì
nhãn không bao giờ chứa thứ model đang bỏ sót, và ô «lỗi đồ thị» của bảng chẩn đoán không
bao giờ có thể khác rỗng.

Chunk nhỏ hơn thì một mạch suy luận trải qua nhiều chunk hơn, tức đường trong đồ thị dài hơn
— phải tăng `L` tương ứng. Với bộ chuẩn (MuSiQue, 2Wiki) thì dùng luôn đơn vị cấp đoạn có
sẵn của dataset, đừng cắt lại.

### 3.2 Tập đủ tối tiểu — đối tượng cần tìm

```
S đủ         v(S) ≥ τ_q          VÀ     adq(S) = phủ hết
S tối tiểu   đủ,  và không tập con thật sự nào của S còn đủ

W  =  { mọi S đủ tối tiểu }
```

#### «Đủ» có hai thành phần, và `v` chỉ đo một

Các bản trước của tài liệu này định nghĩa `S đủ ⟺ v(S) ≥ τ_q`. Điều đó **sai**, và sai theo
hướng sinh ra nhãn quy gán giả.

```
v(S)      model đọc S rồi trả lời — câu đó có khớp y không?      ->  TRẢ LỜI ĐÚNG
adq(S)    các đoạn trong S có THẬT SỰ nói ra đáp án không?       ->  CÓ BẰNG CHỨNG
```

Hai thứ này lệch nhau, vì model có thể trả lời đúng **nhờ nhớ sẵn**:

```
q = "ai đạo diễn phim thắng Oscar phim hay nhất 1994?"      y = "Robert Zemeckis"

đoạn C = "Forrest Gump là một hiện tượng văn hóa thập niên 1990."
         -> không nói ai đạo diễn, cũng không nói thắng giải nào

model đọc C  ->  thấy "Forrest Gump"  ->  tự nhớ ra Zemeckis  ->  trả lời ĐÚNG
->  v({C}) cao  ->  định nghĩa cũ nhận C vào T
->  rồi đem nhãn đó dạy retriever đi lấy C.   NHÃN SAI.
```

Phép trừ `v(∅)` không cứu được: `q` trần không gợi ra ký ức, còn chữ *"Forrest Gump"* trong
`C` thì gợi ra, nên `v({C}) ≫ v(∅)` thật — phép trừ đang **tính công cho cái mồi**.

Đây không phải ca giả định. [Joren et al., ICLR 2025](https://arxiv.org/abs/2411.06037) đo
được trên HotpotQA rằng model trả lời **đúng** `35–62%` số ca mà ngữ cảnh **không đủ**; và tỷ
lệ trả lời sai tăng từ `10,2%` (không ngữ cảnh) lên `66,1%` (ngữ cảnh không đủ). Họ cũng đo
tỷ lệ ca có ngữ cảnh đủ sau truy hồi: **MuSiQue-Ans `46,2%`, HotpotQA `44,6%`** — tức hơn
nửa số truy vấn trên đúng hai bộ §3.1 nhắm tới **không có** tập đủ trong pool.

#### `adq` đo thế nào — chẻ câu hỏi, rồi đối chiếu

Định nghĩa "đủ ngữ cảnh" của Joren et al. **không** tham chiếu đáp án vàng, nên bài của họ
khó (autorater `93%`, RINSE `~84%`). Tầng A chạy **offline, có `y`**, nên hỏi được câu dễ
hơn: *"các đoạn này có chống đỡ cho `y` không?"* — tức **kéo theo**, không phải suy đoán.

```
bước 1   CHẺ câu hỏi thành mảnh          1 lượt LLM, MỖI TRUY VẤN (không phải mỗi tập con)
         vào  q + y
         ra   mảnh 1  "phim thắng Oscar phim hay nhất 1994 là Forrest Gump"
              mảnh 2  "Forrest Gump do Robert Zemeckis đạo diễn"

bước 2   ĐỐI CHIẾU mỗi đoạn × mỗi mảnh    model NLI nhỏ, local, 0 LƯỢT API
         ra   entail / neutral / contradict
```

Phải chẻ ra, vì **không chẻ thì không chạy được**, hai lý do độc lập:

```
ĐỘ DÀI    premise = cả S có thể tới 10.000 token, model NLI chỉ nhận 512.
          chẻ ra thì mỗi lượt kiểm là 1 đoạn (250 token) đối 1 câu ngắn.

BẮC CẦU   claim ghép "phim thắng 1994 do Zemeckis đạo diễn" thì KHÔNG đoạn nào
          một mình khẳng định được -> loại oan cả {A,B}, tức loại đúng ca multi-hop.
          chẻ ra thì mỗi mảnh có đúng một đoạn khẳng định nó.
```

Điểm thứ hai là phát hiện trung tâm của
[arXiv:2608.00585](https://arxiv.org/pdf/2608.00585): kiểm từng đoạn đối câu hỏi gốc thì thất
bại ở multi-hop, và phân rã sửa được. (Tài liệu này lấy **cơ chế** từ đó; chưa trích số vì
chưa đọc được bản text.)

#### Bảng phủ — thứ mọi mục sau đây dùng tới

Kết quả bước 2 là một bảng, tính xong với **0 lượt LLM**:

```
                              mảnh 1              mảnh 2
đoạn  3                       entail                 —
đoạn  7                       entail                 —
đoạn 12                         —                 entail
đoạn 23                         —                 entail
đoạn 31                         —                 entail
28 đoạn còn lại                 —                    —

adq(S) = phủ hết  ⟺  mọi mảnh có ít nhất một đoạn trong S entail nó
```

Bảng này cho ba thứ, và cả ba đều miễn phí:

```
1  CỬA THOÁT    có mảnh nào không đoạn nào chống đỡ  ->  pool không trả lời được
                ->  bỏ truy vấn trước khi tiêu đồng nào   (§3.11 A0/C0)
2  ỨNG VIÊN     bài toán phủ tập nhỏ, giải CHÍNH XÁC:
                   mảnh 1 ← {3,7}   mảnh 2 ← {12,23,31}
                   nghiệm:  {3,12} {3,23} {3,31} {7,12} {7,23} {7,31}
                ->  dùng làm điểm khởi cho §3.4 và §3.6
3  PRIOR VỀ |T| số mảnh nói |T| NÊN bằng bao nhiêu.  |S*| = 8 mà 2 mảnh -> phình, chắc chắn
```

#### Vì sao vẫn cần **cả hai**, không bỏ cái nào

```
                     adq phủ hết                    adq KHÔNG phủ
v(S) ≥ τ_q      nhãn TỐT — nhận               MỒI THAM SỐ — loại
                                              (lỗ mà định nghĩa cũ không thấy)
v(S) < τ_q      bằng chứng CÓ mà model         pool THIẾU — thoát sớm ở A0/C0
                không nối được
                -> lỗi MODEL SINH, đáng báo cáo
```

Không mệnh đề nào suy ra mệnh đề kia. `adq` giữ **tách khỏi** `v`, không gộp thành một điểm
số — gộp thì mất bảng 2×2 này, và mất luôn khả năng tách "lỗi truy hồi" khỏi "lỗi model sinh".

Ô dưới-trái là ô mới mà tầng A trước đây không tách được: bằng chứng đủ mà model vẫn sai thì
**tầng B không sửa được** — phải đổi model sinh hoặc đổi prompt.

**Không cần `v` đơn điệu.** Lý thuyết simple game thường đòi đơn điệu, mà `v` của ta thì
không — thêm một đoạn gây nhiễu làm `v` **giảm**. Nhưng định nghĩa trên không dùng tới đơn
điệu ở đâu cả. Và nếu có ai muốn cưỡng bức đơn điệu bằng bao đơn điệu `v̄(S) = max_{S'⊆S}
v(S')` thì cũng vô ích:

```
S là thắng tối tiểu của v̄  ⟹  S thắng nhờ một S' ⊆ S
                           ⟹  nếu S' ⊊ S thì S' cũng thắng, trái tính tối tiểu
                           ⟹  S' = S,  tức v(S) ≥ τ_q

MWC(v̄)  =  W        hai bên trùng khít
```

Nên đơn điệu hóa không đổi gì ngoài chi phí. Bỏ qua.

**Dư thừa hiện ra thành bội số MWC.** Ba đoạn trùng ý sinh ra ba MWC khác nhau, mỗi tập chứa
một đoạn. Đây là lý do bản này **không cần gom cụm**: cụm sinh ra để gom đoạn trùng ý và để
hạ `2^K`; ở đây dư thừa tự lộ trong cấu trúc `W`, còn `2^K` thì surrogate lo.

### 3.3 Ngưỡng và mốc tham chiếu

```
τ_q  =  θ · [ v_ref(q) − v(∅) ]  +  v(∅)              θ = 0.95

v_ref(q)  =  max(  v(C),  max v(S) đã quan sát  )
```

Vẫn trừ `v(∅)`, dù `v` giờ nằm trong `[0,1]` chứ không còn âm. Lý do đổi: với log xác suất
thì phép trừ là **bắt buộc** (không trừ thì điều kiện không bao giờ thỏa); với judge thì nó
thành **lựa chọn**, và ta chọn giữ vì nó đo đúng thứ cần — *phần mà truy hồi đóng góp thêm*,
chứ không phải độ đúng tuyệt đối.

```
không trừ   τ = 0.95 × 0.95 = 0.90      một đoạn yếu đạt 0.90 cũng "đủ",
                                         dù model vốn đã đạt 0.60 khi không có đoạn nào
có trừ      τ = 0.60 + 0.95×0.35 = 0.93  đòi lấy lại 95% phần truy hồi thật sự thêm vào
```

**Chốt chặn khi mẫu số nhỏ.** Nếu `v_ref − v(∅) < δ` (δ ≈ 0.05) thì truy hồi gần như không
đóng góp gì — model tự trả lời được. Lúc đó đừng chia cho một số gần 0: báo
`T = ∅` kèm cờ *"truy hồi không đóng góp"*. Đây là một ca có thật và đáng báo cáo, không
phải lỗi.

**Và mốc phải là cực đại quan sát được, không phải `v(C)`.** Khi trong `C` còn đoạn gây
nhiễu thì một tập con có thể ghi điểm **cao hơn** cả `C`; neo vào `v(C)` là neo vào một mốc
bị dìm, và ngưỡng hóa ra dễ vượt một cách giả tạo. `v(C)` tốn một lượt; phần `max` thì miễn
phí vì §3.4 vốn đã đánh giá hàng chục tập con.

### 3.4 Thu thập dữ liệu — bốn vòng

Đây là mục quyết định cả phương án, nên nói rõ **`f_θ` cần học cái gì** trước.

§3.6 dùng `f_θ` để làm đúng một việc: hỏi `f_θ(S) ≥ τ_q ?` rồi tỉa về tối tiểu. Nó không bao
giờ cần giá trị `v(S)` chính xác ở chỗ khác.

```
không phải   hồi quy chính xác v(S) trên toàn không gian
mà là        phân loại đúng dấu của ( v(S) − τ_q ), sát biên
             + thứ tự thô đủ tốt cho beam chọn nhánh
```

Tức mục tiêu là **bộ phân loại ở biên**. Lấy mẫu phải dồn vào biên, và hai kiểu lấy mẫu phổ
biến đều sai hướng:

```
mọi cặp đôi        n = 20 cho 190 lượt/truy vấn, gần như toàn negative dễ
tập con ngẫu nhiên phần lớn rơi xa ngưỡng — không dạy được gì cho bộ phân loại
```

Với `|MWC|` điển hình 3–5 thì cặp đôi hầu hết thiếu, và thiếu **rất xa** ngưỡng.

#### Vòng 0 — khung bắt buộc

```
v(∅)             1      mốc dưới
v(C)             1      mốc trên  ->  τ_q
mọi singleton    n      "đoạn nào tự đủ một mình", và tín hiệu bậc 1 cho f_θ
                ───
                n + 2   ≈ 22 lượt
```

#### Vòng 1 — đi dọc biên bằng greedy

Greedy từ rỗng, mỗi vòng chọn thêm một đoạn, tới khi vượt `τ_q`. Ghi lại **mọi** lượt chấm,
cả cái bị loại:

```
S ∪ {k} được chọn   ->  mẫu ở biên, phía trên
S ∪ {k} bị loại     ->  mẫu ở biên, phía dưới      <- đây mới là thứ quý
```

**Không quét hết `n` ứng viên mỗi vòng.** Quét đầy đủ tốn `r·n − r(r−1)/2` lượt — với
`n = 40, r = 3` là 117, và nó tăng tuyến tính theo `n`. Thay vào đó **gom cụm rồi chỉ thăm
dò đại diện**:

```
gom pool thành C cụm                      C ≤ ⌊m/2⌋,  xem dưới
shortlist mỗi vòng:  m ứng viên           thay vì cả 40
```

**`C` có trần cứng, không phải hằng số cho sẵn.** Mỗi cụm được một **sàn** 1 lượt, nên sàn
tiêu hết `C` trong ngân sách `m`. Nếu `C = m` thì sàn ăn trọn ngân sách, phần bốc theo xác
suất biến mất, và phân bổ thoái hóa thành chia đều — mất đúng cơ chế "cụm nóng được nhiều
hơn" mà mục dưới dựng lên:

```
ràng buộc:   C  ≤  ⌊ m / 2 ⌋              m = 12 -> C ≤ 6;  m = 30 -> C ≤ 15
```

Con số `C = 6` dùng trong các ví dụ dưới đây là **trần ở `m = 12`**, không phải lựa chọn tùy
ý. Trong trần đó thì dữ liệu quyết:

```
pha 1   C = argmax silhouette trên embedding pool,  với  2 ≤ C ≤ ⌊m/2⌋
pha 2   gom theo PHẦN DƯ của g_θ, không theo cosine embedding:
            d(k,k')  =  g_θ(q,{k}) + g_θ(q,{k'}) − g_θ(q,{k,k'})
            d lớn   -> trùng lặp -> CÙNG cụm
            d âm    -> bắc cầu   -> KHÁC cụm
        780 cặp cho 40 chunk, mỗi cặp một lượt forward mô hình nhỏ. 0 lượt LLM.
```

Mục đích gom cụm là nhóm các chunk **thay thế được cho nhau** lại, để sàn-mỗi-cụm bảo đảm
shortlist chạm nhiều chủ đề. Cosine embedding chỉ là proxy cho tính thay thế đó; sau pha 1 ta
có thứ đo trực tiếp nó, nên nên dùng.

#### Greedy đi lên có ràng buộc phủ

Bảng phủ (§3.2) đổi ba chỗ trong greedy, và một trong ba làm nó **rẻ hơn**.

```
HƯỚNG ĐI   trước   chọn đoạn làm v tăng nhiều nhất
           sau     trong các đoạn phủ được một mảnh CHƯA PHỦ,
                   chọn đoạn làm v tăng nhiều nhất
                   -> phủ là ràng buộc CỨNG, v là mục tiêu trong ràng buộc

DỪNG       trước   v(S) ≥ τ_q
           sau     v(S) ≥ τ_q  VÀ  S phủ hết mảnh
                   v ≥ τ_q mà chưa phủ hết  =  ĐÚNG ca mồi tham số  ->  đi tiếp

π_c        trước   max gain ước từ singleton — mù ở multi-hop (xem mục dưới)
           sau     ưu tiên cụm chứa đoạn chống đỡ mảnh CÒN CHƯA PHỦ
```

Chỗ `π_c` là chỗ đáng tiền. Mục dưới chỉ ra rằng ở multi-hop **mọi** singleton `≈ v(∅)` nên
`π_c` là nhiễu, và đó là lý do `m` phải nâng lên 30. Bảng phủ nói đoạn nào mang bằng chứng
**kể cả khi không singleton nào làm `v` nhích**, tức sửa đúng cái mù đó:

```
trước   multi-hop: spread thấp -> m = 30 -> 67.500 token greedy
sau     prior không mù nữa      -> m = 12 -> 27.000 token
                                   tiết kiệm 40.500 token mỗi truy vấn multi-hop
```

Nên mục *«`m` phải thích nghi»* dưới đây vẫn đúng về **lý do** (prior mù thì phải quét rộng),
nhưng điều kiện kích hoạt nó đổi: `m = 30` chỉ cần khi `spread` thấp **và** bảng phủ cũng
không cho tín hiệu — tức khi phân rã thất bại.

**`r` là kết quả, không phải núm vặn.** Greedy thêm một đoạn mỗi vòng tới khi vượt `τ_q`, nên
`r = |S|` lúc dừng — tức **`r` chính là cỡ tập đủ mà greedy tìm được**. Câu 2-hop cho `r = 2`,
3-hop cho `r = 3`. Con số `r = 3` dùng ở phần ngân sách là ước lượng để tính tiền, không phải
cấu hình. Sau khi tỉa ở §3.6 thì `|S*|` có thể còn nhỏ hơn `r`.

Và **vòng greedy thứ nhất miễn phí** — nó chính là chấm singleton mà vòng 0 đã làm. Nên chi
phí là `(r − 1)·m`, không phải `r·m`.

Gom cụm ở đây **không phải để hạ `2^K`** — không còn liệt kê mũ nữa. Nó để shortlist có
**tính đa dạng**: lấy top-12 theo prior có thể vớ phải 12 bản sao của nhau, cả vòng lãng
phí; lấy từ mọi cụm thì bảo đảm mỗi vòng nhìn thấy ứng viên từ *mọi* chủ đề.

**Vòng greedy đầu tiên miễn phí** — nó chính là chấm singleton, mà vòng 0 đã làm. Câu hỏi
phân bổ chỉ phát sinh từ vòng thứ hai, và lúc đó đã có đủ `n` giá trị singleton làm tín hiệu.

#### Chia `m` lượt cho các cụm thế nào

Chia đều `a = m/C` cho mọi cụm là cứng nhắc theo đúng hai chiều: cụm vô dụng vẫn ngốn đủ
phần của nó, còn cụm chứa ba ý quan trọng thì bị bóp. Thay bằng **sàn cứng cộng phần còn lại
bốc theo xác suất**:

```
tiềm năng của cụm c tại vòng t:

    π_c  =  max_{k ∈ c \ S_t}  ĝain( k | S_t )

    ĝain   vòng 2:    v({k}) − v(∅)                   dùng singleton đã có sẵn
           vòng 3+:   f_θ(S_t ∪ {k}) − f_θ(S_t)       khi f_θ đã dùng được

phân bổ:

    a_c  =  1  +  Multinomial( m − C ;  softmax(π_c / T) )_c
            ───               ───────────────────────────
            sàn, mọi cụm      phần dư chia theo xác suất
```

Với `C = 6, m = 12`: sàn tiêu 6, còn 6 lượt bốc theo `π`. Cụm nóng có thể nhận `1+4 = 5`,
cụm chết chỉ nhận `1`. **Tổng vẫn đúng `m`** — chỉ đổi cách chia, không đổi ngân sách.

#### `m` phải thích nghi, vì prior mù đúng ở ca ta quan tâm

Ở vòng 2, `π_c` lấy từ `v({k}) − v(∅)`. Nhưng:

```
single-hop   một đoạn tự nó gần đủ   →  singleton có tín hiệu rõ  →  prior TỐT
multi-hop    MỌI đoạn đứng riêng ≈ v(∅)  →  prior là NHIỄU
```

Mà multi-hop chính là lý do tồn tại của cả thiết kế. Trong ca đó, chọn 12 trên 40 gần như bốc
ngẫu nhiên: đoạn cần nằm trong một cụm 7 phần tử, xác suất lọt vào 2 lượt thăm dò của cụm đó
chỉ `≈ 29%`.

Và `m` nhỏ **chưa chắc rẻ hơn**: shortlist trượt thì greedy cần thêm vòng, tức `r` tăng — chi
phí chuyển từ "vòng rộng" sang "nhiều vòng" chứ không biến mất, mà `S*` thu được lại kém hơn.

Ta đã đo sẵn thứ cần để quyết, ngay ở vòng 0:

```
spread  =  max_k v({k}) − v(∅)

spread ≥ 0.15 · [ v_ref − v(∅) ]    prior dùng được      →  m = 12
spread <  ngưỡng đó                 prior mù, multi-hop  →  m = 30
```

Chỉ truy vấn multi-hop phải trả thêm — đúng nhóm đáng trả:

```
m      lượt    token greedy    tổng mỗi truy vấn
12      24        27.000          124.000
20      40        45.000          142.000
30      60        67.500          164.000
40      80        90.000          187.000   (quét đầy đủ, không gom cụm)
```

Nếu một nửa tập là multi-hop thì trung bình `≈ 143.000` token, tức **+15%**.

#### Ô "khám phá" nên đo ĐỘ PHỦ TRUY VẤN, không phải khoảng cách tới S

Khi singleton phẳng, vẫn còn một tín hiệu rẻ chưa dùng: **phần truy vấn mà `S` hiện tại chưa
phủ**.

```
π_c  =  max_{k ∈ c \ S}   độ phủ(  q trừ đi phần đã phủ bởi S ,  k  )
```

Thuần embedding, không tốn lượt LLM nào, và nó **có tín hiệu đúng lúc singleton không có** —
vì multi-hop nghĩa là mỗi đoạn phủ một mảnh khác nhau của câu hỏi. Khoảng cách tới `S` chỉ nói
*khác `S`*; độ phủ truy vấn nói *bù đúng chỗ `S` còn thiếu*. Hai thứ khác nhau, và cái sau mới
là cái cần.

**Vì sao bốc xác suất chứ không chia tỉ lệ tất định.** Chia tất định thì một cụm bị chấm
thấp ở vòng 2 sẽ *vĩnh viễn* không được thăm lại — khóa chặt sai lầm sớm. Bốc xác suất giữ
cho thăm dò nằm sẵn trong cơ chế: `π` còn mơ hồ thì phân bố tự trải ra, `π` rõ thì nó tự
tập trung. Đây là tinh thần Thompson sampling, mà không phải dựng hẳn một bandit — với
`r ≈ 3` vòng thì bandit đầy đủ chưa kịp học gì.

**Sàn `= 1` là lưới an toàn cho khâu gom cụm, không phải cho công bằng.** Nếu gom cụm nhầm
hai sự kiện khác nhau vào một chỗ mà cụm đó bị đánh giá thấp, không có sàn thì nó bị bỏ hẳn
và mất luôn một đoạn cần thiết — **không có đường cứu**. Có sàn thì qua `r = 3` vòng mỗi cụm
được thăm ít nhất 3 lượt.

**Trong một cụm chọn ai:**

```
a_c = 1      lấy khai thác                 — tiềm năng π cao nhất
a_c ≥ 2      1 khai thác + (a_c − 1) khám phá  — xa S_t nhất theo embedding
```

**Độ phủ tích lũy qua các vòng.** Ứng viên "khám phá" đổi theo `S`, mà `S` đổi mỗi vòng —
nên qua 3 vòng, một cụm 7 phần tử được thăm dò tới 6 phần tử khác nhau chứ không phải mãi
cùng 2. Đây là chỗ dễ đánh giá thấp.

**Nhiệt độ `T`** là núm duy nhất thêm vào, và nó thay cho `a`: `T → 0` dồn hết phần dư vào
cụm nóng nhất (tham, dễ kẹt); `T → ∞` chia đều, tức quay về `a` cố định. Đặt `T` sao cho cụm
đứng đầu nhận khoảng 2–3 lần cụm trung vị.

```
tham số trước:   C, a₀, a, r
tham số sau:     C, a₀, m, r, T        a tự sinh ra từ phân bổ
```

Đường greedy **cắt ngang `τ_q` đúng một lần**, và các ứng viên bị loại ở vòng cuối là những
tập *suýt đủ*. Không có cách lấy mẫu nào dày ở biên hơn thế, và nó rơi ra tự nhiên chứ không
phải dò mò. Đây cũng chính là lý do bản Shapley cũ vẫn có ích: vòng greedy của nó sinh ra
đúng loại dữ liệu này.

#### Vòng 2 — active learning, chỉ nơi còn lưỡng lự

```
train k = 5 surrogate trên bootstrap của vòng 0–1
chọn mẫu thỏa cả hai:
    ensemble bất đồng cao          (chưa chắc)
    f̄_θ(S) nằm trong τ_q ± δ       (sát biên)

~30 lượt mỗi truy vấn, lặp 2–3 vòng
```

#### Vòng 3 — xác nhận bằng model thật

**Bắt buộc, và đây là chỗ bản gốc của phương án thiếu hẳn.** Beam search ở §3.6 chạy trên
`f_θ`, nên `W` là "tối tiểu theo surrogate". Nếu `f_θ` lệch quanh `τ_q` thì sai cả bộ mà
không gì bắt được.

```
với S* và top-k MWC:  chạy v(S) thật, xác nhận v(S) ≥ τ_q
                      và với mỗi c ∈ S*:  xác nhận v(S* \ {c}) < τ_q   (thật sự tối tiểu)

≈ 15 lượt
```

Lệch thì ghi log để retrain. Nguyên tắc giữ nguyên từ §4.10: **không bao giờ để proxy quyết
định nhận hay bỏ.**

#### Ngân sách — đếm bằng token, không phải số lượt

Các lượt gọi **không bằng giá nhau**: prompt chỉ chứa `S`, nên

```
cost(S)  ≈  500 + 250·|S|   token            (chunk 250 token, chỉ dẫn 500)

v(∅)              500
v({c})            750          ← rẻ hơn v(C) hai mươi lần
v(S), |S| = 3   1.250
v(C), |C| = 40 10.500
```

Mà greedy **xây dần từ rỗng** nên mọi lượt của nó đều có `|S|` nhỏ. Lượt đắt duy nhất là
`v(C)`, và nó chạy một lần. Đếm theo số lượt sẽ ước lượng sai hẳn.

```
n = 40,  C = 6,  m = 12,  r = 3        (ca prior tot; multi-hop thì m = 30)

v(∅) + v(C)                              11.000
singleton × 40                           30.000
greedy   vòng 2:  12 × 1.000             12.000
         vòng 3:  12 × 1.250             15.000
active learning   30 × 1.250             37.500
xác nhận          15 × 1.250             18.750
                                        ────────
                                       ≈ 124.000 token / truy vấn
```

Ở mức `$3`/triệu token đầu vào thì **≈ $0,37 mỗi truy vấn**; 1000 truy vấn huấn luyện
≈ `$370`, một lần, cache vĩnh viễn.

Đối chiếu: quét đầy đủ với `n = 60` tốn ~260.000 token. Giữ được 40 ứng viên mà rẻ hơn
**2.1 lần**, và hơn hẳn phương án cắt xuống `n = 20` về độ phủ.

**Vì sao quét hết singleton** (`a₀ = n`) thay vì cũng gom cụm: chúng là lượt rẻ nhất (750
token) và chúng bắt được ca *"một đoạn tự nó đủ"* — loại MWC cỡ 1, gọn nhất và đáng tin
nhất. Hạ xuống 2 mỗi cụm tiết kiệm 21.000 token nhưng có thể bỏ sót đúng ca đó.

#### Ba chi tiết cài đặt đáng tiền

```
cache          khóa = hash( model, template, nội dung chunk đã sắp )
               băm nội dung chứ không phải ID — chunk lại corpus là ID va nhau

thứ tự chuẩn   xếp chunk theo một thứ tự cố định rồi gom lô theo tiền tố chung.
               Prompt caching chỉ tái dùng được tiền tố:
                   S₁ = {c1, c3, c7}
                   S₂ = {c1, c3, c9}   chung tiền tố "c1 c3" -> chỉ encode đuôi
               Vòng 1 sinh đúng dạng đó: mọi ứng viên trong một vòng là S ∪ {k}
               với cùng S, nên gom cả vòng thành một lô thì tiền tố encode một lần.

lưu kèm        τ_q · vòng · chọn-hay-bị-loại · |S|
               có mấy trường này thì sau còn đánh trọng số lại hướng về biên
               mà không phải lấy mẫu thêm
```

**Chia tập theo cả hai cách, và đọc hai con số khác nhau.** Hai cách chia đo hai năng lực
khác nhau, mà pha 1 và pha 2 cần đúng hai năng lực đó:

```
giữ lại TẬP CON trong truy vấn đã thấy    = điều kiện pha 1
                                            mô hình có ṽ_k của truy vấn này; việc nó
                                            "học thuộc" giá trị từng chunk ở đây là MỤC TIÊU,
                                            không phải rò rỉ
giữ lại TRUY VẤN                           = điều kiện pha 2
                                            mô hình chưa từng thấy truy vấn, không có ṽ_k
                                            -> đây mới là con số quyết định §3.11
```

Trước đây mục này chỉ nêu cách chia thứ hai và gọi cách thứ nhất là rò rỉ. Điều đó sai: với
một truy vấn đã trả vòng 0, nhớ giá trị từng chunk của nó chính là thứ beam search cần.

### 3.5 Surrogate

Một hàm duy nhất, dùng cho cả hai pha ở §3.11:

```
g_θ( q,  S,  đặc trưng mỗi chunk )   ->   một số thực = dự đoán v(S)
```

Bất biến hoán vị trên các phần tử của `S`, vì `S` là tập.

**Đầu vào.**

```
q                văn bản truy vấn, embed
mỗi k ∈ S        văn bản chunk, embed
                 + ŝ(q,k)   điểm rerank          LUÔN có, cả hai pha, 0 lượt LLM
                 + ṽ_k      v({k}) − v(∅)        chỉ có khi truy vấn đã trả vòng 0
v(∅)             mốc dưới của thang đo           đo trước mọi lần gọi g_θ
```

**`τ_q` không phải đầu vào, và không được phép là đầu vào.** `g_θ` hồi quy lên `v(S)`; việc
so với ngưỡng xảy ra sau, bên ngoài mô hình. Ràng buộc này đến từ thứ tự chạy ở §3.11: bước
`C2` gọi `g_θ` để quyết định có mua `v(C)` hay không, mà `v(C)` mới là thứ chốt `v_ref` rồi
chốt `τ_q`. Cho `τ_q` vào đầu vào là tạo ra một vòng tròn thật sự. `v(∅)` thì ngược lại — nó
đo được ở bước `C1`, trước `C2`, nên dùng được.

#### Dữ liệu huấn luyện

Không thu riêng. Nó **là log của chính các lượt gọi `v`** ở §3.4 và §3.11 — mỗi lần tính
`v(S)`, ghi lại `(q, S, v(S))`.

```
nguồn                                truy vấn    nhãn/truy vấn      tổng
pha 1, bốn vòng (§3.4)                10.000         111         1.110.000
verification đường rẻ (§3.11)        188.000          10         1.880.000
tỉa, |S*| = 6..10 (§3.11)              8.000          13           104.000
cứu -> pipeline đầy đủ                 7.400         111           821.000
ca trượt bị bỏ, không cứu             13.600           5            68.000
                                                               ───────────
                                                                 3.983.000
```

Một mẫu là `(q, S, v(S))` lưu kèm `τ_q`, vòng sinh ra, chọn-hay-bị-loại, `|S|`. Cỡ `S` chạy
từ 0 tới 40: `|S| = 0` là `v(∅)`, `|S| = 1` là các singleton của vòng 0, `|S| = 12` là
`v(R)`, `|S| = 40` là `v(C)`. Không cỡ nào được coi là đặc biệt — đó là lý do **một** mô hình
chứ không phải một mô hình cho singleton và một mô hình cho phần còn lại.

**Nhánh leo thang lớn nhất, và quý nhất.** Nó chỉ chiếm 10% truy vấn nhưng đóng gần một nửa
số nhãn, vì ba lẽ: log đầy đủ chứ không phải bảy điểm quanh `S*`; toàn ca `g_θ` đoán sai, tức
hard negative ở cấp truy vấn; và có `ṽ_k` thật cho cả 40 chunk vì nó chạy vòng 0.

#### Che `ṽ_k` theo từng chunk, trộn ba khuôn

`ṽ_k` có ở pha 1 mà không có ở pha 2. Train thẳng với nó thì lúc chạy mô hình gặp phân bố
khác hẳn. Nhưng che tất-cả-hoặc-không-gì cũng sai, vì thực tế có **ba** chế độ:

```
pha 1 + leo thang      cả 40 chunk có ṽ_k        vòng 0 đầy đủ
verification đường rẻ  CHỈ |S*| chunk có ṽ_k     3 lượt đơn tử trong 7 lượt kiểm
beam ở bước C5         không chunk nào có        chưa gọi LLM lần nào
```

Trộn ba khuôn theo đúng tỷ lệ chúng xuất hiện. Khuôn thứ hai — biết giá trị của 3 chunk trong
`S`, không biết 37 chunk còn lại — là khuôn mà che tất-cả-hoặc-không-gì không bao giờ dạy
được, mà nó lại là khuôn của mọi lượt verification ở pha 2.

#### Mất mát

```
L  =  L_cls  +  λ · L_rank

L_cls    BCE trên nhãn  1[ v(S) ≥ τ_q ]         ← mục tiêu chính, vì §3.6 chỉ cần dấu
L_rank   hinge theo cặp TRONG cùng một truy vấn  ← giữ thứ tự thô cho beam
```

`τ_q` xuất hiện ở đây, trong mất mát, chứ không ở đầu vào. Nó định hình mô hình chính xác **ở
đâu** — dồn vào biên của từng truy vấn. Để `L_cls` là chính chứ không phải MSE trên `v(S)`:
hồi quy chính xác ở vùng xa ngưỡng là công sức bỏ phí.

`L_rank` phải so theo cặp **trong cùng một truy vấn**. So chéo truy vấn thì nó học thứ tự
giữa các `τ_q` khác nhau, mà `τ_q` khác nhau theo từng truy vấn nên thứ tự đó vô nghĩa.

Nhãn dương chỉ chiếm `~10–15%`: greedy đi từ rỗng nên hầu hết tập con nằm dưới ngưỡng. Lệch
này là thứ ta muốn — chúng là negative **sát biên** — nhưng phải cân trọng số lớp trong
`L_cls`.

#### Chỉ số nghiệm thu, và sàn phải vượt

```
AUC TRONG từng truy vấn, rồi trung bình qua các truy vấn
sàn:   mô hình chỉ nhìn |S|
```

AUC gộp mọi truy vấn vào một rổ là cái bẫy. Tập lớn thì `v` cao — đúng khi xét xuyên truy vấn
— nên một mô hình **chỉ dùng `|S|`** đã có AUC gộp đẹp. Nhưng nó có năng lực phân biệt trong
truy vấn bằng **0**: mọi tập cỡ 3 hòa điểm, beam thành chọn bừa. Không vượt được sàn đó thì
`g_θ` vô dụng dù con số gộp bao nhiêu. Vì lý do tương tự, đừng báo cáo accuracy trần — dồn
mẫu ở biên làm nó không so được với accuracy trên mẫu ngẫu nhiên.

#### Mốc phải vượt trước khi chốt kiến trúc

Phần thực sự phải học là **độ lệch khỏi cộng tính**: hai chunk nói cùng một điều thì không
cộng dồn; hai chunk ghép lại mới trả lời được thì cộng hơn tổng.

```
truy vấn 2 hop "ai đạo diễn phim thắng Oscar phim hay nhất 1994?"

                                       v      cộng tính     lệch
v(∅)                                 0,05
chunk 7  "Forrest Gump thắng 1994"   0,10
chunk 23 "Zemeckis đạo diễn F.Gump"  0,15
S = {7, 23}                          0,95      0,20       +0,75   bắc cầu
S = {7, 8}  (chunk 8 ~ trùng 7)      0,10      0,15       −0,05   trùng lặp
```

`+0,75` và `−0,05` là hai thứ duy nhất mô hình phải đoán, và chúng là **quy luật chung**: hai
đoạn gần trùng thì lệch âm, hai đoạn cùng thực thể mà khác quan hệ thì lệch dương. Đây là chỗ
gộp nhiều truy vấn có nghĩa.

Bài toán đó nhỏ hơn hẳn việc đoán `v(S)` từ đầu, nên **một GBM trên đặc trưng tay phải được
chạy trước làm mốc**:

```
|S| · Σ_{k∈S} ŝ(q,k) · max ŝ · số cụm S chạm · tương đồng cặp lớn nhất trong S
     · tương đồng với q · (Σ ṽ_k khi có)
```

Set Transformer có mặt nạ là kiến trúc đề xuất, và lý lẽ là tự chú ý bắt tương tác theo cặp —
đúng là hiện tượng cần mô hình hóa. Nhưng đó là lý lẽ, không phải bằng chứng: nó chỉ đáng
chọn nếu **vượt mốc GBM trên AUC trong truy vấn**.

#### Rò rỉ ngữ cảnh pool

Nếu mặt nạ chạy trên cả pool 40 chunk thì mô hình thấy được những chunk **không** nằm trong
`S`, và có thể học dựa vào "cả pool trông thế nào" thay vì vào `S`. Lúc chấm trong beam thì
pool cố định nên đặc trưng đó vô hại; nhưng nó làm con số nghiệm thu đẹp giả, và tệ hơn, nó
biến mất đúng lúc pool đổi. Cách chặn: mã hóa `S` cùng các scalar theo từng chunk, không cho
mô hình nhìn phần bù của `S` trong pool.

### 3.6 Liệt kê MWC

Bảng phủ (§3.2) làm không gian tìm kiếm co lại rất mạnh. Lý do:

```
tập KHÔNG phủ hết mảnh  ->  nếu nó vượt τ_q thì chỉ có thể vì mồi tham số
                        ->  mà ta đã quyết LOẠI ca đó (§3.2)
                        ->  tìm trong vùng đó là tìm thứ ta sẽ bỏ
```

Nên cả dàn `2^n` co lại thành **các nghiệm phủ, và mọi tập chứa một nghiệm phủ**. Tìm theo
từng tầng, đi lên:

```
tầng 0   các nghiệm phủ tối tiểu (giải chính xác từ bảng phủ)      |T| = số mảnh
tầng 1   nghiệm phủ + 1 đoạn
tầng 2   nghiệm phủ + 2 đoạn
...
mỗi tầng:  chấm bằng g_θ, lấy top-beam, đo v theo thứ tự đó
```

Hai lợi ích, cả hai đều rơi ra từ cấu trúc:

```
1  MỌI ứng viên đã đủ về bằng chứng — không còn ứng viên mồi tham số
2  thứ tự theo CỠ, nên tập nhỏ luôn thử trước
   -> "tối tiểu trước" là miễn phí theo cấu trúc, không phải một bước tỉa riêng
```

**Tính tối tiểu gần như không tốn lượt nào.** Vì điều kiện nhận có hai mệnh đề (§3.2), một tập
con chỉ **đe dọa** tính tối tiểu nếu nó **cũng phủ hết**:

```
T = {3, 12, 19}      3 phủ mảnh 1 · 12 phủ mảnh 2 · 19 là đoạn thêm

tập con thực sự:           7 tập
trong đó CÒN phủ hết:      1 tập  —  {3,12}
     ({3,19} mất mảnh 2 · {12,19} mất mảnh 1 · mọi đơn tử mất một mảnh)

=>  chỉ cần kiểm {3,12},  mà tầng 0 ĐÃ đo nó rồi.
```

Nên nếu tìm theo tầng, các phép đo cần cho tính tối tiểu **chính là các tầng dưới, đã trả
tiền**. Không có lượt nào thêm ở cuối — đây là chỗ làm gọn hẳn phần ba ngưỡng `|S*|` ở §3.11.

**Beam mò vẫn giữ làm dự bị.** Bảng phủ có thể thiếu cột (NLI bỏ sót bằng chứng diễn đạt
lại), nên nếu mọi tầng đều trượt thì quay về cách cũ, tìm trên toàn dàn:

```
frontier = { ∅ }
cho size = 1, 2, ..., n:
    ứng viên = { S ∪ {i} : S ∈ frontier, i ∉ S }
    chấm bằng g_θ
    đủ = { S : g_θ(S) ≥ τ_q }
    nếu đủ khác rỗng:  thu hoạch, và chạy tiếp thêm 1–2 cỡ nữa
    frontier = top-beam theo g_θ

tỉa về tối tiểu:  với mỗi S đủ, bỏ c ∈ S nếu g_θ(S \ {c}) ≥ τ_q, lặp tới khi không bỏ được
```

Tầng dự bị này chính là nhánh `(2)` của `C7` (§3.11) — không cần cơ chế mới, chỉ là danh sách
ứng viên giờ có hai nguồn.

**Ca `v < τ_q` dù đã phủ hết — đọc nó cho đúng.** Đây là ô dưới-trái của bảng 2×2 (§3.2):

```
bằng chứng CÓ trong T mà model vẫn trả lời sai
->  lỗi của MODEL SINH, không phải của truy hồi
->  thêm đoạn là đang "giúp" model, không phải bổ bằng chứng
```

Nên khi phải đi lên tầng 1, 2 mới qua `τ_q`, **ghi lại số đoạn thêm**. Con số đó lớn và hay
xảy ra thì kết luận không phải *"pool thiếu"* mà là *"model sinh yếu ở bắc cầu"* — và tầng B
không sửa được điều đó.

Hai chỗ phải cẩn thận ở tầng dự bị:

**Đừng dừng ở cỡ đầu tiên đạt đủ.** Dừng ngay thì thiên vị tập nhỏ và có thể bỏ sót một MWC
lớn hơn nhưng vững hơn nhiều. Chạy thêm 1–2 cỡ rồi mới thu hoạch.

**Tỉa phụ thuộc thứ tự.** Vì `v` không đơn điệu, bỏ một phần tử có thể làm `v` **tăng**, nên
phải tỉa nhiều vòng và kết quả phụ thuộc thứ tự bỏ. Chốt một thứ tự và ghi rõ: bỏ theo `r_i`
tăng dần (đoạn rerank xếp thấp bị thử bỏ trước). Đây đúng là căn bệnh mà §4.11 chặng 2 cũng
mắc, và cách chữa giống nhau — chốt thứ tự, đừng giả vờ là không có.

#### Mở rộng trong cụm — trả lại thứ gom cụm đã lấy mất

Lập luận *"không cần gom cụm vì dư thừa hiện ra thành bội số MWC"* (§3.2) **gãy** khi §3.4
chỉ thăm dò đại diện: nếu `c3` và `c7` cùng cụm mà chỉ `c3` được chấm, thì `{c12, c7}` không
bao giờ lộ ra, và tín hiệu `PGI` mất theo.

Chữa bằng một bước chạy **hoàn toàn trên surrogate**, không tốn lượt LLM nào:

```
với mỗi S trong W:
    với mỗi c ∈ S,  với mỗi c' cùng cụm với c,  c' ∉ S:
        nếu f_θ( S − {c} + {c'} ) ≥ τ_q:   thêm vào W

xác nhận 2–3 MWC mới bằng model thật
```

Tức **gom cụm để tìm, mở rộng trong cụm để báo cáo**. Hai việc khác nhau, và chỉ việc đầu
cần tiết kiệm. Sau bước này thì `PGI`, `overlap` và entropy ở §3.7–3.9 quay lại đầy đủ.

### 3.7 Chỉ số cấu trúc

```
β_i    Banzhaf có dấu
       β_i⁺  số lần thêm i biến thiếu -> đủ
       β_i⁻  số lần thêm i biến đủ -> thiếu          ← đơn điệu thì bằng 0
       β_i  =  ( β_i⁺ − β_i⁻ ) / 2^(n−1)             ước lượng bằng lấy mẫu trên f_θ

PGI_i  số MWC trong W có chứa i

overlap(S)  =  Σ_{S' ≠ S}  |S ∩ S'| / |S ∪ S'|
```

`β_i < 0` **chỉ thẳng ra đoạn gây hại** — đúng vai mà `φ_c < 0` đóng trong bản Shapley. Tính
không đơn điệu ở đây là thông tin, không phải khuyết tật.

**Chuẩn hóa `β` và `PGI` riêng rồi mới gộp.** `β ∈ [−1,1]`, còn `PGI` là một số đếm có thể
tới hàng chục; cộng thẳng thì `PGI` nuốt `β` một đến hai bậc độ lớn và `σ_g` thực chất chỉ
còn là `PGI`.

```
σ_g^sum(S)  =  Σ_{i∈S} ( β̃_i + PGĨ_i )
σ_g^min(S)  =  min_{i∈S} ( β̃_i + PGĨ_i )           mắt xích yếu nhất
```

### 3.8 Chọn S*

Ba tiêu chí, chuẩn hóa rồi gộp có trọng số:

```
σ_s(S)  =  −|S|                   gọn
σ_g(S)                            cấu trúc (sum hoặc min)
σ_r(S)  =  Σ_{i∈S} r_i            khớp rerank

S*  =  argmax  w_s·σ̃_s + w_g·σ̃_g + w_r·σ̃_r
```

Hướng: **vững** thì cực đại `σ_g` (thưởng bằng chứng dùng được rộng rãi); **đặc thù** thì
cực tiểu `σ_g` (thưởng bằng chứng riêng biệt). Phải ghi rõ chọn hướng nào — hai hướng cho
hai `S*` khác nhau.

Nên lọc Pareto trước: bỏ các MWC bị trội hoàn toàn trên mọi tiêu chí. Phá hòa cuối cùng bằng
`Σ r_i`, rồi bằng chỉ số từ điển cho tái lập được. **Rerank chỉ quyết định ở bước cuối** —
cấu trúc lo phần thực chất.

### 3.9 Credit và độ tin cậy

```
cứng   credit(c_i)  =  1/|S*| nếu c_i ∈ S*,  0 nếu không

mềm    p(S) = softmax( Φ(S)/T )        credit(c_i) = Σ_{S ∋ i} p(S)/|S|
       T → 0   quay về bản cứng
       T → ∞   quay về Deegan–Packel chuẩn (đều trên mọi MWC)
```

**Entropy của `p(S)` là tín hiệu độ tin cậy.** Thấp nghĩa là có một người thắng rõ ràng; cao
nghĩa là nhiều MWC suýt soát và quy gán đang không chắc. Bản Shapley không có đại lượng
tương đương — nó luôn trả về một `T` duy nhất với vẻ tự tin như nhau.

### 3.10 Đầu ra

```
T  =  S*                          tập đích, đi sang tầng B
W, p(S)                           họ MWC và phân bố — để báo cáo
entropy( p )                      độ tin cậy
β_i                               β < 0 -> ứng viên negative cho V_neg (§6.3)
```

**Chỉ `T` đi qua ranh giới hai tầng; không điểm số nào đi qua** — xem §1 của phụ lục
*Những chỗ dễ sai*. Mọi thứ còn lại là để báo cáo và chẩn đoán.

#### Phép kiểm

```
1  v(T) ≥ τ_q                                    đo bằng model thật, không phải g_θ
2  adq(T) = phủ hết mảnh                         bằng chứng THẬT SỰ nằm trong T (§3.2)
3  v(S') < τ_q  với mọi tập con thực sự S' ⊂ T
   mà S' CÒN phủ hết                             T thật sự tối tiểu (§3.6)
4  mọi i có β_i < 0 đều không nằm trong T         không đưa đoạn gây hại vào đích
5  Σ_i credit(c_i) = 1                            theo cấu trúc
```

**Phép kiểm 2 phải quét hết tập con, không được dùng bỏ-một-phần-tử.** Nếu `v` đơn điệu thì
chỉ cần kiểm các tập `S* \ {c}`: mọi tập nhỏ hơn đều nằm trong một tập đó, nên đơn điệu cho
`v(nhỏ hơn) ≤ v(S* \ {c}) < τ_q` miễn phí. Nhưng `v` **không đơn điệu** (§3.2), nên

```
v(S* \ {c}) < τ_q   KHÔNG kéo theo   v(tập con nhỏ hơn) < τ_q
```

Ca cụ thể, `b` là chunk gây nhiễu:

```
τ_q = 0,76
v({a})     = 0,85  ≥ τ_q      ->  {a} MỘT MÌNH đã đủ
v({a,b})   = 0,60  < τ_q          b làm model lạc
v({a,c})   = 0,71  < τ_q
v({b,c})   = 0,40  < τ_q
v({a,b,c}) = 0,80  ≥ τ_q

bỏ-một-phần-tử:  cả ba cặp trượt  ->  tuyên "S* tối tiểu"       SAI
sự thật:         {a} thắng        ->  S* thừa cả b và c
```

Chi phí quét hết là `2^|S*| − 2` lượt (bỏ chính `S*`, và `∅` đã đo sẵn). Sai chỗ này đắt vì
`T` là tập nguồn vàng của tầng B: `T` thừa thì tầng B học định tuyến tới đoạn không cần, tức
nhãn nhiễu theo hướng **thừa**, đúng hướng tệ nhất.

**Nhưng không phải quét hết dàn dưới `S*` ứng viên — chỉ dưới `T` cuối cùng.** Mục này đòi
`T` *là một* MWC, không đòi nó là MWC duy nhất hay nhỏ nhất. Nên với `S*` lớn, cách đúng là
**tỉa xuống rồi quét hết ở cuối** (§3.11): nếu `T = {a,b,c}` có `v(T) ≥ τ_q` và cả 6 tập con
thực sự đều dưới ngưỡng, thì `T` **là** một tập đủ tối tiểu — 247 tập con khác của một `S*`
cỡ 8 không cần xét, vì ta không khẳng định gì về chúng. Thứ trở nên không đầy đủ là `W`, và
§3.11 đã ghi rằng `W` ở pha 2 vốn không dùng để kết luận cấu trúc.

Bộ này **yếu hơn** bộ bất biến của bản Shapley: `Σ φ_c = F(E) − F(E')` là một đẳng thức
chính xác bắt được mọi lỗi cài đặt ở khâu quy gán, còn phép kiểm 1–3 ở đây chỉ xác nhận
`S*` chứ không xác nhận cả phép tính. Đó là cái giá phải trả khi bỏ completeness, và nên
nói thẳng.

#### Chuẩn ngoài — hiệu chuẩn trên HotpotQA

Mọi phép kiểm ở trên là **nội bộ**: `v(T) ≥ τ_q` do chính `v` đo, `τ_q` do chính `v_ref` đặt,
`adq` do chính model NLI của ta phán. Không có gì bên ngoài nói *"`T` này đúng là nguồn"*.

HotpotQA có thứ đó: nó gán sẵn **2 đoạn chống đỡ** và tối đa **5 câu chống đỡ** cho mỗi câu
hỏi. Đó là nhãn quy gán do người gán — chuẩn ngoài duy nhất sẵn có.

```
lấy ~500 truy vấn HotpotQA  ->  chạy TOÀN BỘ tầng A  ->  so T với 2 đoạn đã gán
giá:  500 × $0,38  =  $190
```

`$190` đổi lấy ba con số không lấy được cách nào khác:

```
1  tỷ lệ ca mồi tham số THẬT     T chứa đoạn không nằm trong nguồn vàng
2  model NLI có đáng tin không    adq có đồng ý với nhãn người không
3  precision/recall của T         tầng A đúng bao nhiêu phần so với chuẩn ngoài
```

Con số 2 quan trọng nhất: đem `adq` đi loại nhãn trên 200.000 truy vấn mà **chưa bao giờ biết
nó sai bao nhiêu** thì chỉ là đổi một lỗi không biết sang một lỗi khác không biết. Hai lỗi cụ
thể phải đo ở đây:

```
phân rã sót một hop     ->  bảng phủ thiếu cột  ->  T được nhận khi còn thiếu bằng chứng
NLI "neutral" giả       ->  thoát sớm oan, hoặc loại oan T tốt
```

Một chỗ phải cẩn thận: HotpotQA gán nguồn cho **bối cảnh 10 đoạn có nhiễu** của riêng nó, còn
pool của ta dựng từ 3 nguồn trên corpus riêng với cách cắt chunk của ta. Nên cần bước **khớp
chunk của ta với đoạn của họ** trước khi so — làm được, nhưng là một bước thật. §3.1 vốn đã
đòi *"pool và đồ thị phải dùng cùng một cách cắt"*; giờ thêm một đối tượng thứ ba phải khớp.

**MuSiQue bù chỗ khác:** nó không gán câu chống đỡ, nhưng gán **phân rã thành câu hỏi 1-hop
kèm đáp án trung gian**. Tức nó cho sẵn bước 1 của `adq` (§3.2), miễn phí — dùng nó để kiểm
chất lượng của LLM chẻ câu hỏi.

Và bù lại cho phần bất biến: nên chạy bản Shapley ở [file riêng](tang-a-shapley.md) trên **một
mẫu nhỏ** làm đối chứng — `S*` có nằm trong tập đoạn mà Shapley chấm cao không. Lệch nhiều là dấu hiệu `f_θ`
hoặc `τ_q` đang sai.

#### Phân tích độ nhạy

```
quét w_s, w_g, w_r mỗi cái ±50%       S* đổi bao nhiêu lần?
quét τ_q quanh θ ∈ [0.90, 0.98]       W có ổn định không?
bootstrap f_θ, train lại               phương sai của credit
```

Nếu `S*` lật lung tung khi nhiễu nhỏ thì **báo cáo cả biên Pareto** thay vì một `S*` duy
nhất. Thà nói "có bốn lời giải thích ngang nhau" còn hơn chọn bừa một cái rồi trình bày như
thể nó chắc chắn.


### 3.11 Hai pha — mô hình ở pha 1, nhãn ở pha 2

Mọi thứ từ §3.1 tới §3.10 là một truy vấn, chạy LLM đầy đủ, `~$0,38`. Ở quy mô thật thì
không trả nổi:

```
chạy §3.4 cho 200.000 truy vấn      200.000 × $0,38   =   $76.000
```

Nên tách làm hai pha có mục đích khác nhau:

```
PHA 1   N₁ = 10.000 truy vấn, LLM đầy đủ
        sản phẩm:  g_θ        — KHÔNG phải nhãn cho tầng B
PHA 2   N₂ = 200.000 truy vấn, đường rẻ
        sản phẩm:  nhãn (q, T) cho tầng B ở quy mô N₂
```

Ở `N₁ = 300` như các bản trước của tài liệu này thì hai pha trùng nhau nên không cần tách; ở
`N₂ = 200.000` thì pha 1 là **đầu tư vào mô hình** còn pha 2 là **sản lượng**.

#### Bước truy hồi — chung cho cả hai pha, 0 lượt LLM

```
3 nguồn -> pool thô K ≤ 60 -> khử trùng lặp
              ↓  rerank THEO TỪNG NGUỒN, hạn ngạch riêng (§3.1)
n = 40  =  20 BM25  +  15 GFM  +  5 ngẫu nhiên
              ↓
gom C cụm,  C ≤ ⌊m/2⌋ (§3.4)
```

#### Pha 1 — 10.000 truy vấn

```
A0  chẻ câu hỏi thành mảnh + bảng phủ NLI (§3.2)      400 token
      thiếu mảnh nào không đoạn nào chống đỡ
      ->  BỎ truy vấn, cờ "pool thiếu".  Đã tiêu 400, không phải 127.500.
A1  vòng 0 — quét CỐ ĐỊNH, không cần τ_q
      v(∅)                                           500 token
      v(R)     R = top-2 mỗi cụm, |R| = 12         3.500 token
      v(C)     cả 40 chunk                        10.500 token
      v({k})   cả 40 chunk đơn                    30.000 token
      ── v_ref = max( v(C), v(R), max_k v({k}) ) ;  τ_q chốt ──
A2  vòng 1 — greedy biên (§3.4)                    27.000 token
A3  vòng 2 — active learning quanh τ_q            ~37.000 token
A4  vòng 3 — xác nhận bằng model thật             ~15.000 token
                                                 ─────────────
                                                  ~127.500 token  ≈ $0,38
```

#### Giữa hai pha

```
B1  gộp log 10.000 truy vấn        ~1.200.000 mẫu, |S| từ 0 tới 40
B2  train g_θ (§3.5), che ṽ_k theo ba khuôn
B3  CỔNG p: giữ 500 truy vấn pha 1, chạy TOÀN BỘ đường pha 2 trên chúng,
      so S* thu được với S* thật đã biết
```

`B3` dùng nhãn đã có nên **không tốn lượt LLM nào**, và nó là cổng quyết định:

```
p ≥ 0,90                 triển khai 200k
p ≈ 0,75                 tiền cứu ăn gần hết phần tiết kiệm -> tăng N₁ trước, chưa tăng N₂
p tụt mạnh theo số hop   g_θ yếu ở ca bắc cầu -> đổ thêm truy vấn multi-hop vào pha 1
p cao mà |S*| phình      phần dư ÂM (trùng lặp) chưa học được
```

#### Pha 2 — 200.000 truy vấn, đường rẻ

```
C0  chẻ câu hỏi thành mảnh + bảng phủ NLI (§3.2)              400 token
      thiếu mảnh  ->  THOÁT, cờ "pool thiếu", bỏ truy vấn
C1  v(∅)                                                     500 token
C2  g_θ chấm 40 chunk đơn,  ṽ_k che                              0 token
C3  v(R)   R = 12 chunk, cách chọn ở dưới                    3.500 token
C4  v(C)   cả 40 chunk                                      10.500 token
      ── v_ref = max( v(R), v(C) ) ;  τ_q chốt ──
      v_ref − v(∅) < 0,05  ->  T = ∅, cờ "truy hồi không đóng góp" (§3.3)
C5  tìm theo tầng từ nghiệm phủ của C0 (§3.6)  ->  ứng viên xếp hạng   0 token
      mọi tầng trượt  ->  beam mò làm dự bị
C6  verification bằng LLM THẬT — đủ, và tối tiểu            ~7.400 token
C7  ba nhánh: xuất nhãn · thử ứng viên kế · cứu
                                                           ─────────────
                                                            ~21.900 token  ≈ $0,066
```

**`C2` phải đứng trước `C5`** vì beam cần điểm từng chunk làm prior mở rộng; nó hợp lệ vì
`g_θ` không nhận `τ_q` (§3.5).

#### `A0`/`C0` không tốn thêm tiền — nó tiết kiệm gần một nửa

Cửa thoát ở `C0` tốn `400` token. Cửa thoát cũ ở `C4` phải trả
`v(∅) + v(R) + v(C) = 14.500` token mới biết. Và theo §3.2, tỷ lệ truy vấn không có tập đủ
trong pool là `~45–55%` trên các bộ multi-hop:

```
nếu 45% thoát ở C0:
   kỳ vọng  =  0,45 × 400  +  0,55 × 22.300   =   12.400 token  ≈ $0,037
   trước đó                                       21.900 token  ≈ $0,066
```

Ở pha 1 còn đậm hơn, vì mỗi truy vấn bỏ được là `127.500` token chứ không phải `21.900`.

**Nhưng truy vấn bị bỏ không phải rác.** Chúng là cờ *"pool thiếu"*, tức nguồn nuôi ô «lỗi đồ
thị» của §6.1 và vòng lặp ngược §6.6 — và là nguồn **duy nhất** làm ô đó khác rỗng ở quy mô
lớn. Ghi lại, đừng xóa.

**Cửa thoát phải đặt THẬN TRỌNG.** NLI hay bỏ sót bằng chứng diễn đạt lại hoặc ngầm, nên chỉ
thoát khi điểm entail của **mọi** đoạn đối mảnh đó đều rất thấp, không phải chỉ khi nhãn là
`neutral`. Và phải lấy mẫu kiểm tay một phần ca bị bỏ — bỏ mù `45%` truy vấn dựa trên một
model chưa hiệu chuẩn là rủi ro lớn nhất của cả cơ chế này (xem chuẩn ngoài ở §3.10).

#### Vì sao mua `v(C)` chứ không bù bằng một hằng số

Bản trước của mục này bỏ `v(C)` ở pha 2 và bù `v(R)` bằng một hằng số hiệu chuẩn ở pha 1. Bỏ
cách đó, vì ba lẽ:

```
tiền      v(C) chỉ là 10.500 trong 106.000 token mà pha 2 tiết kiệm được —
          10%. Phần tiết kiệm thật nằm ở 40 singleton (30.000) và vòng 1–2 (64.000).
nhãn sai  hằng số đặt ở phân vị 90 nghĩa là ~10% truy vấn vẫn bị trần thấp,
          mà trần thấp sinh NHÃN SAI ÂM THẦM — verification không bắt được,
          vì nó so với chính τ_q đã bị dìm đó.
bộ máy    hằng số đó kéo theo một bước đo riêng, một bước hiệu chuẩn, một cổng
          kích hoạt, và một ký hiệu trùng tên với δ của §3.3.
```

Mua `v(C)` cho mọi truy vấn: `+$4.700` trên 200k, tức `$0,029` mỗi nhãn, và **pha 1 với pha 2
dùng cùng một định nghĩa `v_ref`**.

**Nhưng `v(R)` vẫn phải giữ.** §3.3 đã cảnh báo: `v(C)` nhồi 40 chunk nên chính nó bị phân
tán, một tập con có thể ghi điểm cao hơn cả `C`, và neo vào `v(C)` là neo vào **mốc bị dìm**.
Pha 1 không gặp vấn đề vì nó lấy max trên hàng chục tập con đã quan sát; pha 2 không quan sát
gì, nên cần `v(R)` — 12 chunk, ít nhiễu hơn hẳn — làm ứng viên thứ hai cho `max`.

#### Chọn `R`: sàn cụm thắng hạn ngạch nguồn

Hai ràng buộc này xung đột thường xuyên, vì cụm gom theo embedding chứ không theo nguồn nên
một cụm có thể toàn chunk GFM. Thứ tự quyết:

```
6 suất   mỗi cụm 1 chunk, lấy chunk hạng cao nhất trong cụm       SÀN CỤM — bắt buộc
6 suất   còn lại: trước hết bảo đảm mỗi nguồn có ≥ 1 trong R
                  rồi lấp phần dư theo hạng
                                                                  |R| = 12
```

Sàn cụm thắng vì nhiệm vụ của `v(R)` là đo **trần**, mà trần phải chạm *mọi* chủ đề. Và 2
chunk mỗi cụm chứ không 1, vì với 1 thì một **cặp bắc cầu nằm trong cùng một cụm** không bao
giờ hiện ra — đúng ca 2-hop mà cả thiết kế nhắm tới.

#### `C6` — ba ngưỡng theo `|S*|`

Phép kiểm tối tiểu phải quét hết tập con (§3.10), mà chi phí đó nổ theo `2^|S*|`. Lối ra
không phải bỏ phép kiểm, mà là **tỉa xuống trước rồi mới quét**:

```
S ← S* ứng viên
lặp khi |S| > 5:
    xếp c ∈ S theo g_θ(q, S∖{c}) giảm dần                   0 lượt LLM
    thử c đầu bảng:  đo v(S∖{c})
        ≥ τ_q  ->  S ← S∖{c},  lặp
        < τ_q  ->  thử c kế tiếp, tối đa 3 lần
    3 lần đều trượt  ->  KẸT
|S| ≤ 5  ->  quét hết dàn dưới S  ->  T, đã CHỨNG MINH
```

**Tỉa rồi quét vẫn là chứng minh đầy đủ.** §3.10 chỉ đòi `T` *là một* MWC, không đòi nó là
MWC duy nhất hay nhỏ nhất. Nếu `S' = {a,b,c}` có `v(S') ≥ τ_q` và cả 6 tập con thực sự đều
dưới ngưỡng, thì `S'` **là** một tập đủ tối tiểu — 247 tập con khác của một `S*` cỡ 8 không
cần xét, vì ta không khẳng định gì về chúng. Cái không đầy đủ là `W`, mà `W` ở pha 2 vốn đã
không dùng để kết luận cấu trúc.

Chi phí quét hết nổ theo `2^s`, còn tỉa chỉ tăng **tuyến tính** theo `s`:

```
|S*|   quét hết ngay    tỉa rồi quét ở cuối    cứu
 3        $0,020               —              $0,34
 5        $0,107            $0,107            $0,34
 6        $0,239            $0,043            $0,34
 8        $1,14             $0,071            $0,34
10        $5,37             $0,098            $0,34
15        $172              $0,218            $0,34
20        $5.500            $0,326            $0,34
```

Nên ba ngưỡng:

```
|S*| ≤ 5         quét hết ngay                         $0,02 – $0,11
6 ≤ |S*| ≤ 10    tỉa xuống ≤ 5 rồi quét hết            $0,04 – $0,10
|S*| ≥ 11        cứu                                   $0,34
tỉa KẸT ở ≥ 6    cứu                                   $0,34
```

Tỉa còn rẻ hơn cứu tới `s ≈ 20`, nên ranh giới `10/11` **không** đặt theo tiền: ở cỡ đó ta
không còn tin điểm xuất phát đủ để tỉa 15 lượt theo chỉ dẫn của chính mô hình đang sai. Đây là
một núm mà `B3` đo được rồi chỉnh — đo tỷ lệ tỉa-kẹt theo `|S*|`.

**Tỉa phụ thuộc thứ tự bỏ**, vì `v` không đơn điệu — đúng căn bệnh của §3.6, và cách chữa
giống nhau: chốt một thứ tự (ở đây là `g_θ` giảm dần) và ghi rõ, đừng giả vờ là không có.

**Tỉa chỉ đi xuống được.** Nếu `S*` vừa phình vừa **thiếu** một chunk cần thiết thì mọi tập
con đều trượt và tỉa đứng tại chỗ. Kẹt ở `|S| ≥ 6` nghĩa là `g_θ` sai cả hai chiều — đó là
lúc cứu, và vì vậy tỉa **không thay thế** cứu, nó **lọc** xem truy vấn nào thật sự cần cứu.

#### `C7` — ba nhánh

```
(1) v(S*) ≥ τ_q, và quét ra được một tập tối tiểu
        ->  T, xuất sang tầng B

(2) v(S*) < τ_q                        S* KHÔNG đủ — g_θ đoán cao quá
        ->  thử ứng viên kế của beam, tối đa 3 lần
            mỗi lần thử chỉ 1 lượt (hết đủ thì dừng, không kiểm tối tiểu)
        ->  qua  ->  nhánh (1)
            3 lần đều trượt  ->  nhánh (3)

(3) |S*| ≥ 11,  tỉa kẹt ở ≥ 6,  hoặc 3 lần thử đều trượt
        ->  hàng cứu có phân tầng
```

**Ca "đủ mà không tối tiểu" không vào hàng cứu.** Khi quét hết, `C6` đo **trọn vẹn** dàn tập
con dưới `S*` — với `|S*| = 3` là cả 8 tập con, `|S*| = 4` là cả 16 — nên các phần tử tối tiểu
trong đó đọc ra ngay, 0 lượt thêm:

```
v({a,b,c}) = 0,80 ≥ τ_q        S* đủ
v({a,b})   = 0,78 ≥ τ_q        tập con cũng đủ  ->  S* KHÔNG tối tiểu
v({a}) = 0,30   v({b}) = 0,22  cả hai đơn tử dưới ngưỡng
->  T = {a,b}
```

Đây là ca xảy ra thường xuyên, vì lỗi hay gặp nhất của `g_θ` là đề xuất tập **phình**.

**Nhánh (2) rẻ hơn tưởng.** Hết đủ thì dừng ngay, nên một ca trượt chỉ tốn 1 lượt
(`1.250` token) chứ không phải 7. Và một lượt cứu tốn `113.000` token — **mua được 90 lần thử
bằng giá một lần cứu**. Dừng ở 3 vì ứng viên sau cùng một mô hình nên cùng thiên lệch, lợi ích
giảm nhanh. Với tỷ lệ qua có điều kiện `~30%` mỗi lần thử, 3 lần vớt `1 − 0,7³ = 66%` số ca:

```
p hiệu dụng    0,75  ->  0,75 + 0,25 × 0,66  =  0,91
```

Đây **không** phải đi tìm ngưỡng dễ: mỗi ca qua đều là `v(S') ≥ τ_q` **đo thật**, rồi tối
tiểu vẫn bị quét hết. Nhưng phải **ghi lại hạng beam được nhận**, vì nhận ứng viên #3 đẩy phân
bố `v(S*) − τ_q` xuống sát ngưỡng hơn, và tầng B cần biết để đánh trọng số.

#### Cứu là gì

**Cứu = chạy `A1`–`A4` lên truy vấn đó**, tức pipeline đầy đủ với LLM thật, thay vì tin vào
đề xuất của `g_θ`. Nó không bắt đầu từ 0:

```
đã có trong cache     v(∅) 500 · v(R) 3.500 · v(C) 10.500            14.500
                      + v(S*) và dàn tập con đã đo lúc C6
còn phải chạy         40 singleton                                   30.000
                      vòng 1 greedy biên                            27.000
                      vòng 2 active learning                        37.000
                      vòng 3 xác nhận                                6.500
                                                                  ─────────
                      cứu, tính biên                                113.000 token  ≈ $0,34
```

**`τ_q` có thể TĂNG khi cứu, và đó là đúng.** Pha 2 chỉ quan sát `v(∅)`, `v(R)`, `v(C)`; cứu
quan sát thêm 40 singleton và hàng chục tập con greedy, nên

```
pha 2   v_ref = max( v(R), v(C) )
cứu     v_ref = max( v(R), v(C), max_k v({k}), mọi v(S) quan sát trong greedy )
                                 └────────── chỉ xuất hiện khi cứu ──────────┘
```

`v_ref` chỉ có thể lên, nên `τ_q` lên theo: một truy vấn trượt ngưỡng pha 2 có thể gặp ngưỡng
**cao hơn** lúc được cứu. Nghe nghịch lý nhưng đúng — pha 2 đang đo trần thiếu, và cứu sửa
chỗ đó.

**Cứu mà vẫn trượt là ca có giá trị nhất.** Greedy chạy hết, active learning xong, không tập
nào vượt `τ_q`:

```
->  T = ∅  +  cờ "pool không chứa tập đủ"
```

Đây không phải thất bại của `g_θ` mà là bằng chứng **pool thiếu** — đúng thứ thiết kế đa nguồn
sinh ra để phát hiện. Nó nuôi ô «lỗi đồ thị» của bảng chẩn đoán §6.1 và vòng lặp ngược §6.6.
Nếu ô đó luôn rỗng thì hoặc đồ thị hoàn hảo, hoặc ta đang không đo được nó; ca cứu-mà-trượt là
nguồn duy nhất ở pha 2 làm nó khác rỗng.

#### Cứu phải lấy mẫu, và phân tầng theo cái gì

Cứu hết thì `~17.000` ca trượt × `$0,34` = `$5.780` để lấy `17.000` nhãn — `$0,34` mỗi nhãn,
đắt gấp 5 đường rẻ. Nhưng tính theo **mẫu huấn luyện** thì ngược lại:

```
cứu              111 mẫu / $0,34    =   $0,0031 / mẫu
đường rẻ, qua     10 mẫu / $0,066   =   $0,0066 / mẫu
```

Nên cứu không phải để lấy *cái nhãn đó*. Nó có hai việc, và hai việc muốn hai cách lấy mẫu
ngược nhau:

```
ĐO     "bao nhiêu % ca trượt là g_θ sai, bao nhiêu là pool thiếu?"
       -> cần mẫu KHÔNG LỆCH: ngẫu nhiên trong từng tầng, kèm trọng số N_s/n_s
TRAIN  nuôi g_θ đúng chỗ nó yếu
       -> muốn càng nhiều càng tốt, và lệch về tầng khó nhất
```

Cách chia: một hạn mức nhỏ cố định cho mục đích **đo** (ví dụ 50 ca mỗi tầng, đủ ước lượng tỷ
lệ), rồi phần còn lại dồn hết cho tầng khó để **train**.

```
tầng                            ý nghĩa                           ưu tiên cứu
trượt, hỏng ở ứng viên #1       g_θ lệch mạnh                      cao
trượt, hỏng cả 3 ứng viên       g_θ sai hệ thống ở truy vấn này    CAO NHẤT
tỉa kẹt ở |S| ≥ 6               sai cả hai chiều                   CAO NHẤT
|S*| ≥ 11                       phình nặng                         cao
nhiều hop                       ca bắc cầu — đúng thứ ta cần       CAO NHẤT
v(S*) sát ngay dưới τ_q         có thể chỉ là trần đo thiếu        thấp
phiên bản g_θ^(b)               để tách xem lô nào yếu             (chỉ để đo)
```

**Ca bị bỏ vẫn không vô giá trị.** Mỗi ca trượt không được cứu vẫn xuất `v(∅)`, `v(R)`,
`v(C)`, `v(S*)` và dàn tập con đã đo — `4+` nhãn thật — cộng với sự kiện *"`g_θ` đề xuất `S*`
mà LLM bác"*, là hard negative ở cấp truy vấn. Ghi vào log train; chỉ **không** xuất `T`.

#### Dấu xuất xứ trên mỗi nhãn

```
T = S*
+ phiên bản g_θ        g_θ^(1) ... g_θ^(10)      lô nào gán
+ đường đi             qua trực tiếp · giải trong dàn · tỉa · thử lại #k · được cứu
+ trọng số importance  N_s / n_s của tầng lấy mẫu
+ v(S*) − τ_q          khoảng vượt ngưỡng
```

Có bốn trường này thì tầng B lọc được xuống tập tin cậy nhất ("chỉ pha 1 + ca được cứu") khi
cần, hoặc đánh trọng số lại. Không có thì nó train trên một tập chất lượng không đều mà không
có cách nào biết.

#### Chạy theo lô, không một lượt

```
lô b    chạy 20.000 truy vấn bằng g_θ^(b)
        ├─ qua    ->  xuất T sang tầng B
        └─ trượt  ->  hàng cứu
        lấy mẫu hàng cứu theo tầng  ->  chạy pipeline đầy đủ        <- ở ĐÂY
        gộp mọi nhãn mới  ->  train  g_θ^(b+1)
lô b+1  ...
```

Cứu phải xử lý **ở ranh giới lô**, cùng lúc với train lại, vì nhãn cứu là đầu vào giá trị nhất
cho `g_θ^(b+1)`. Dồn hết sang cuối chương trình thì chúng không dạy được gì cho các lô sau và
mất toàn bộ bánh đà.

Nhãn mà một lô sinh ra nằm **đúng chỗ `g_θ` yếu** — chúng là các tập con do chính nó đề xuất
rồi bị LLM bác, tức đúng vùng phân bố mà beam tra tới mà greedy chưa từng đi qua.

```
lô    truy vấn   trượt sau thử lại   tỷ lệ cứu   số ca cứu
 1     20.000         ~9%   1.800        40%        720
 5     20.000         ~7%   1.400        25%        350
10     20.000         ~5%   1.000        10%        100
                                                   ─────
                                                  ~3.400  (+ ca |S*|≥11 và tỉa kẹt)
```

Giảm dần vì `p` lên thì mỗi ca khó bớt giá trị biên. Nhưng **sàn 5% phải giữ**, không bao giờ
xuống 0 — nếu không thì ô «lỗi đồ thị» mất nguồn và ta không còn đo được tỷ lệ trượt thật.

Và lý do chính của việc chia lô **không phải tiền**: mỗi ranh giới lô là một **cửa thoát**.

```
sau lô 1   p = 0,70 như dự  ->  chạy tiếp
           p = 0,45         ->  DỪNG. mới tốn ~$1.900, chưa tốn $19.000
           |S*| phình đều   ->  dừng, sửa phần dư âm trước khi chạy tiếp
```

**Phải lưu phiên bản mô hình kèm mỗi nhãn** — xem mục dấu xuất xứ ở trên.

#### Một chỗ dễ dùng sai trọng số importance

```
ĐO     p, tỷ lệ trượt, mọi đại lượng quần thể   ->  PHẢI dùng trọng số
                                                    (chỉ cứu 10–40%, mẫu lệch)
TRAIN  g_θ                                      ->  ĐỪNG hạ trọng số ca leo thang
                                                    chúng là hard example, muốn chúng nặng
```

Hai mục đích ngược chiều nhau trên cùng một tập dữ liệu. Dùng chung một bộ trọng số cho cả hai
là lỗi âm thầm: mô hình vẫn train, chỉ là nó học nhẹ đi đúng ở chỗ nó yếu.

#### Chất lượng `W` và `β` ở pha 2 thấp hơn pha 1

`W`, `p(S)`, `entropy`, `β_i` ở §3.7 đều đến từ beam trên `g_θ`, và ở pha 2 verification
**chỉ kiểm `T`**, không kiểm cả họ `W`. Nên `β_i < 0 -> V_neg` (§6.3) ở pha 2 là bằng chứng
yếu hơn hẳn. Nếu `V_neg` quan trọng thì lấy nó từ 10.000 truy vấn pha 1, đừng lấy từ pha 2.

Thêm nữa, `T` ở pha 2 là một MWC **trong `S*`**, không phải MWC toàn cục của `C`: có thể còn
tập thắng nhỏ hơn nằm ngoài `S*` mà ta không bao giờ thấy, vì `C5` chỉ beam chứ không liệt kê
hết. Pha 1 cũng vậy nhưng nhẹ hơn, vì nó có greedy thật và active learning quét quanh biên.

#### Sổ chi phí

```
pha 1    18.200 nạp vào, 10.000 chạy đủ      =   $3.850
            8.200 thoát ở A0 × 400 token          (bao gồm trong số trên)
pha 2   200.000 nạp vào, 110.000 chạy đủ     =   $7.500
           90.000 thoát ở C0 × 400 token          (bao gồm trong số trên)
cứu     ~3.000 ca × $0,34                    =   $1.000
hiệu chuẩn chuẩn ngoài, 500 HotpotQA (§3.10) =     $190
                                              ─────────
                                                $12.540      ~104k nhãn ĐÁNG TIN

chạy pipeline đầy đủ cho cả 200k                $76.000
```

So với bản trước (`$19.500` cho `~188k` nhãn), bản này **rẻ hơn và cho ít nhãn hơn** — và đó
là tiến bộ, không phải thoái bộ: `84.000` nhãn chênh lệch kia là các truy vấn mà pool **không
chứa** tập đủ, nên nhãn sinh ra từ chúng hoặc rỗng hoặc là mồi tham số. Trước đây ta trả tiền
để sản xuất chúng rồi đem đầu độc tầng B.

Khoảng **3,9 lần rẻ hơn**. Phần tiết kiệm đến từ việc bỏ vòng 0 ở pha 2 — 40 lượt singleton
(`30.000` token) thay bằng `g_θ`, cộng vòng 1–2 (`64.000`) — **không** từ việc cắt
verification hay bỏ `v(C)`.

```
truy vấn THOÁT ở C0  (~45%)
C0  chẻ + bảng phủ            400 token   100%

truy vấn CHẠY ĐỦ  (~55%)
C0  chẻ + bảng phủ            400 token    2%
C1  v(∅)                      500 token    2%
C3  v(R)                    3.500 token   16%
C4  v(C)                   10.500 token   47%
C6  verification, kỳ vọng   7.400 token   33%
                          ─────────────
                           22.300 token

kỳ vọng mỗi truy vấn nạp vào:  0,45×400 + 0,55×22.300  =  12.400 token  ≈ $0,037
```

Cắt `C6` thì `T` không tối tiểu vẫn được xuất nhãn, và tầng B học định tuyến tới đoạn không
cần — nhãn nhiễu theo hướng **thừa**, đúng hướng tệ nhất (§3.10). Còn `C0` thì ngược lại: nó
là bước **duy nhất** trong cả pha 2 mà bỏ đi sẽ làm chi phí *tăng*.

#### Bảng này chưa được kiểm chứng — nói thẳng

```
p = 0,75              GIẢ ĐỊNH, chưa đo. toàn bộ cột chi phí treo vào nó.
30% mỗi lần thử lại   GIẢ ĐỊNH. nếu thấp hơn thì p hiệu dụng 0,91 không đạt.
~4% có |S*| ≥ 6       minh họa, theo lý thuyết 2–3 hop. chưa đo.
|S*| = 3 là cột chính chưa đo trên dữ liệu thật.
```

Cổng `B3` tồn tại chính vì bốn dòng trên. Chạy nó **trước** khi cam kết `N₂`; nó miễn phí.

---

## 4. Tầng B — Tìm đường trên đồ thị

**Đầu vào của tầng này là tập đích `T` vừa ra ở §3.8.** Mọi thứ từ đây chạy trên GFM 34M,
không đụng tới LLM nữa.

**Bài toán.** `T` nói *đoạn nào* cần; nó không nói *đi lối nào* tới đó. Biết model xếp sai
`D` thì phải dạy lại — nhưng dạy nó cái gì?

**Mục tiêu.** Với mỗi `D ∈ T`, tìm một tập đường từ seed tới `D`: nhỏ, đọc hiểu được, và
kiểm chứng được — bật riêng nó lên thì model vẫn tìm ra `D`.

**Trở ngại.** G-reasoner cộng tổng thông điệp, không có trọng số cạnh nào để quy gán. Và
tín hiệu đi qua `L` tầng nên credit assignment loãng dần.

#### Công cụ: Integrated Gradients

IG trả lời một câu hỏi: *một đầu ra vô hướng `F` chịu ảnh hưởng bao nhiêu từ mỗi đầu vào?*

Lấy gradient tại điểm đang xét là cách hiển nhiên, nhưng nó hỏng khi hàm bão hòa. IG sửa
bằng cách **đi từ một mốc «không có gì» tới điểm thật, cộng dồn gradient dọc đường**:

```
IG_i  =  (x_i − b_i) · ∫₀¹  ∂F/∂x_i |_{x = b + α(x−b)}  dα
          ───────────       ──────────────────────────
          đi xa mốc         gradient dọc đường, không chỉ ở đích
          bao nhiêu
```

Ví dụ nhỏ cho thấy vì sao cần tích phân. Lấy `F(a,b) = min(a+b, 1)`, mốc `(0,0)`, điểm thật
`(0.8, 0.6)`:

```
gradient tại điểm thật:   a+b = 1.4 > 1, F đã bão hòa  →  ∂F/∂a = ∂F/∂b = 0
                          gradient kết luận "cả hai đều vô dụng" — sai hẳn

IG đi dọc đường (0.8α, 0.6α):   α < 0.71  thì chưa bão hòa, gradient = 1
                                α > 0.71  thì bão hòa,      gradient = 0

     IG_a = 0.8 × 0.71 = 0.57
     IG_b = 0.6 × 0.71 = 0.43
     tổng = 1.00 = F(0.8,0.6) − F(0,0)   ✓
```

Dòng cuối là tính chất quan trọng nhất, gọi là **completeness**: các phần cộng lại đúng
bằng cái tổng. Nhờ nó mà mọi con số IG đọc được như "chiếm bao nhiêu phần trăm", và mọi
phép kiểm trong tài liệu này đều dựa vào nó.

> **Hai nghĩa của chữ «đường», đừng lẫn.**
> *Đường tích phân* là đoạn thẳng trong **không gian đầu vào**, đi từ mốc tới điểm thật —
> một thiết bị toán học bên trong công thức IG.
> *Đường trên đồ thị* là chuỗi cạnh `seed → … → D` — thứ ta đang đi tìm.
> Tài liệu nói «đường» mà không nói rõ thì luôn là nghĩa thứ hai.

#### Áp vào bài toán này — nhìn tổng thể

G-reasoner không có trọng số cạnh, nên trước hết phải **tạo ra đầu vào để IG chấm**: gắn một
gate vô hướng lên mỗi thông điệp. Tắt hết gate là mốc, bật như model thật là điểm thật. Rồi
ba bước:

```
1  đo    IG chia cái bánh "nhờ đồ thị mà p(D) cao lên" xuống từng (cạnh, tầng)
         →  một bảng số

2  nối   bảng số chưa phải lời giải thích. Ghép các ô điểm cao thành chuỗi liền
         mạch theo thứ tự tầng — cạnh nào ở tầng 1, cạnh nào ở tầng 2, ...
         →  một đường đi

3  thử   IG là đạo hàm: nó nói "vặn mạnh thì p(D) tăng", không nói "chừng này là đủ".
         Bật riêng đường đó rồi chạy model, xem có còn tìm ra D không.
         →  nhận hoặc bỏ
```

Ba bước đó chính là ba phần của mục này:

```
T  (tầng A: đoạn nào cần)
 |
 |   A. Đo       §4.1-4.5      hỏi gì -> gắn gate -> chạy IG -> tự kiểm bảng số
 v
bảng IG:  mỗi (cạnh, tầng) một con số
 |
 |   B. Nối      §4.6-4.9      chuỗi là gì -> vẽ lại đồ thị -> quét -> vì sao cộng
 v
đường đi tới một đích
 |
 |   C. Thử      §4.10-4.12    nhận hay bỏ -> một đích nhiều seed -> nhiều đích
 v
Π, depth(D), ŵ        nhãn train cho tầng B
```

Viết theo G-reasoner ([arXiv:2509.24276](https://arxiv.org/abs/2509.24276)): QuadGraph bốn tầng node (attribute / entity / document / community), GNN phụ thuộc truy vấn, 34M tham số. Đầu ra của nó là top-`k` node mỗi loại, rồi các node tài liệu được nhét vào prompt cho LLM sinh câu trả lời.

**Ký hiệu mục này giới thiệu**

```
V, 𝓔, R          node; cạnh; quan hệ.      Một cạnh: e = (v', r, v)
L, V_q           số tầng GNN (3–6); tập node seed trích từ truy vấn
v_D              node đồ thị ứng với tài liệu D
h_v^(l), h_r^(l) embedding node v sau l tầng; embedding quan hệ r
h_q              embedding truy vấn
m_e^(l)          thông điệp mà cạnh e mang ở tầng l
g_e^(l), ĝ_e^(l) gate; gate đã học, phụ thuộc truy vấn (chế độ 2)
g*               điểm cuối khi đo: luôn = 1 để đào nhãn; = ĝ chỉ khi làm báo cáo
ε_g, E_act       ngưỡng lọc gate (0.05); tập (cạnh, tầng) còn lại sau lọc
p(v), p^(l)(v)   điểm node ở tầng cuối; điểm node ở tầng l
F_B(g)           điểm mục tiêu tầng B = log p(v_D)
M, α, α_m        số bước nội suy (20–50); tham số nội suy, α_m = m/M
IG_e^(l), ŵ_e^(l)  quy gán cho cạnh; IG chuẩn hóa [0,1] làm nhãn L_edge
h = |π|          số hop (số cạnh) của một đường
V̂, Ê            node / cạnh của đồ thị trải phẳng theo tầng
best[(v,l)]      giá trị QHĐ tốt nhất tới ô (v,l)
back[(v,l)]      con trỏ truy ngược
π, π*_D          một đường; đường tốt nhất tới đích D
Π_D, Π           cây đường của đích D; toàn bộ cây      ← đầu ra tầng B
ε̃, eps           phần cái bánh tối thiểu một đường phải mang (ε̃ = 0.10); eps = ε̃·cake
n_max            trần số đường, suy ra: ⌈θ_s/ε̃⌉ = 10
σ(D)             số seed với tới được D (§4.8)
E_reach          cạnh có thể nằm trên một đường từ seed
suff_max         trần của suff do credit mồ côi (§4.11)
depth(D)         tầng đầu tiên mà D tới được (§5.2)
Ecov             tập cạnh đã nằm trong báo cáo (§5.3)
g_Π              gate bị che theo Π: giữ ĝ trên Π, 0 chỗ khác
suff, nec, θ_s   tỷ lệ cái bánh lấy lại / mất đi; ngưỡng suff (0.95)
```

---

**A. Đo cạnh**

Biến câu hỏi thành một bảng số, và tự kiểm bảng đó.

### 4.1 Điểm mục tiêu

```
p(v)   = Predictor_{type(v)}( h_v^(L), emb(v), emb(q) )

F_B(g) = log p(v_D)        với mỗi D ∈ T, chạy riêng từng đích
```

`F_B` là hàm của vector gate `g`, không phải của embedding. Đó là khác biệt cốt lõi so với tầng A.

### 4.2 Gate trên cạnh

G-reasoner tính thông điệp bằng DistMult rồi cộng tổng: không attention, không hệ số riêng
cho từng cạnh. IG cần một biến liên tục vô hướng để lấy đạo hàm, nên ta chèn một hệ số nhân
vào mỗi thông điệp.

Thông điệp là vector `d` chiều, gate là một vô hướng trong `[0,1]`. Một gate điều khiển cả
vector, nhờ đó mỗi cạnh chỉ có một đại lượng cần quy gán thay vì `d` đại lượng. Gate đánh
chỉ số theo cặp (cạnh, tầng), tổng cộng `|𝓔| × L` gate; lý do tách theo tầng ở §4.3.

```
m_e^(l)  =  g_e^(l) · Msg( h_v^(l-1) , h_r^(l) , h_v'^(l-1) )
 vector     vô hướng       đích        quan hệ     nguồn

h_v^(l)  =  Upd( h_v^(l-1) ,  Σ  m_e^(l) )
                             e vào v
```

Ba dạng của gate:

```
g         gate nói chung; ở chế độ 1 luôn bằng 1
ĝ         gate đã học ở chế độ 2
g(α)      gate vặn theo α khi chạy IG (§4.4):   g(α) = α · ĝ
```

**Chế độ 1 — gate là công cụ phân tích.** Đặt `g_e^(l) = 1` cho mọi cạnh, mọi tầng; model
chạy y hệt bản gốc, gate chỉ tồn tại để IG có chỗ bám. Dùng ở vòng khởi động (§6.5).

**Chế độ 2 — gate là bộ phận của model, phụ thuộc truy vấn.**

```
ĝ_e^(l) = σ( MLP( h_v^(l-1), h_r^(l), h_v'^(l-1), h_q ) )   ∈ (0,1)
```

Ba đầu vào đầu mô tả cạnh; đầu vào thứ tư là embedding truy vấn, cho phép model định tuyến
khác nhau theo từng truy vấn. Bản gốc không có cơ chế này: truy vấn chỉ tác động lúc khởi
tạo, sau đó `L` tầng lan truyền chạy như nhau bất kể hỏi gì.

Gate vẫn inductive — cả bốn đầu vào của MLP suy từ văn bản hoặc quan hệ, không có ID node.

### 4.3 Gate phải riêng từng tầng

Một cạnh vật lý được dùng `L` lần, mỗi tầng một lần — và các lần đó không tương đương:

```
X --sinh_ở--> C --thuộc--> D          seed = {X}

tầng 1   X→C nổ  →  C nhận tín hiệu truy vấn        ✓ hữu ích
         C→D nổ  →  C còn trống, D nhận đồ vô nghĩa ✗ vô dụng

tầng 2   C→D nổ  →  C đã có tín hiệu, D nhận được   ✓ hữu ích
         X→C nổ  →  C đã có rồi, thừa               ✗ gần vô dụng
```

Gate chung sẽ trộn hai vai trò đó lại và mất thông tin thứ tự — tức mất đúng cái quan trọng nhất của multi-hop. Nên số gate là `|E| × L`.

### 4.4 IG trên gate

Gọi `g*` là **điểm cuối** — vặn gate tới đâu khi đo. Đường tích phân vặn mọi gate theo một
tham số chung `g(α) = α · g*`, với `α = 0` là trạng thái không thông điệp nào chảy (mọi node
giữ `h^(0)`). Quy gán cho mỗi cặp (cạnh, tầng):

```
IG_e^(l)  =  ( g*_e^(l) − 0 )  ×  (1/M) Σ_{m=1..M}  ∂F_B(α_m·g*) / ∂g_e^(l)
             └ đầu vào − baseline ┘    └── gradient trung bình dọc đường ──┘
```

Baseline bằng 0 nên thừa số đầu rút gọn thành `g*_e^(l)`. Đây là dạng chuẩn của IG, chỉ khác
biến được móc vào: tầng A móc vào embedding token, tầng B móc vào gate cạnh.

**Đo thì luôn đặt `g* = 1`**, kể cả sau khi model đã có gate học được. Thừa số đầu thành 1 và
rụng hẳn, còn lại đúng gradient trung bình. Lý do ở ngay dưới.

```
Kiểm tra 4:   Σ_{e,l} IG_e^(l) = F_B(g*) − F_B(0)
```

**Cái bánh của tầng B lấy ở đâu ra.** Tổng `Σ IG` **không phải cộng từ bảng IG** — nó là
hiệu của hai lượt chạy, biết được trước cả khi chạy IG:

```
lượt 1   g = 0      p(D) = 0.0183     F_B(0)  = −4.00
lượt 2   g = g*     p(D) = 0.0412     F_B(g*) = −3.19
                                     ─────────────
                             cái bánh = 0.81 nat
```

tức lan truyền trên đồ thị nhân xác suất của `D` lên `e^0.81 = 2.25` lần.

`F_B(0)` không bằng 0: tắt hết gate thì node mất tin từ hàng xóm, nhưng predictor vẫn
nhận được `emb(v)` và `emb(q)` nên vẫn so khớp văn bản được. Vậy:

```
F_B(0)     điểm đạt được chỉ bằng so khớp văn bản, không dùng cấu trúc đồ thị
cái bánh   phần mà cấu trúc đồ thị đóng góp thêm
```

Cùng khuôn với tầng A: ở đó `F(E')` là "chỉ tri thức tham số", ở đây `F_B(0)` là "chỉ văn
bản". IG chỉ làm một việc: chia cái bánh đó xuống từng cặp (cạnh, tầng). Kiểm tra 4 là
phép đối chiếu giữa hai cách tính — hiệu hai lượt chạy, và tổng bảng IG.

#### Vì sao đo thì luôn `g* = 1`

Từ mốc 2 model có gate học được, và sẽ rất cám dỗ khi đặt `g* = ĝ` — đo đúng chỗ model đang
vận hành. Nhưng làm vậy thì hỏng việc, vì thừa số đầu chính là `ĝ`:

```
cạnh     ĝ      gradient trung bình    IG = ĝ × gradient
e1      0.90           0.30                  0.27
e2      0.02           0.85                  0.017
```

`e2` nhạy gấp gần ba lần `e1` — mở nó ra thì `p(D)` tăng mạnh — nhưng vì model đang đóng
nó, IG của nó nhỏ hơn 16 lần và nó rơi khỏi bảng.

Mà công việc đào nhãn chính là **tìm cạnh model lẽ ra nên mở**. Cạnh đó theo định nghĩa có
`ĝ ≈ 0`: nếu model đã mở rồi thì nó đã không xếp sai. Đặt `g* = ĝ` tức là xóa sạch đúng
những thứ cần tìm, và `L_edge` khi đó chỉ dạy gate củng cố chính nó.

```
ĝ  là thứ model làm, không phải thứ ta dùng để đo.
```

Nên ở mọi mốc, mọi lượt đào nhãn: `g* = 1`. Chế độ 2 xuất hiện trong model, không xuất hiện
trong cách đo.

**Một ngoại lệ: báo cáo trung thực.** Khi cần giải thích cho người dùng *"hệ thống đã lấy
`D` qua lối nào"*, báo cáo không được kể tên cạnh model chưa từng dùng — lúc đó mới đặt
`g* = ĝ`. Đây là tính năng phụ, không nằm trong vòng train.

Lượt báo cáo còn được một prior miễn phí để lọc trước:

```
E_act = { (e,l) : ĝ_e^(l) > ε_g }        ε_g ≈ 0.05

chỉ chạy QHĐ trên E_act  →  thường còn vài phần trăm số cạnh
```

Lượt đào nhãn không dùng được prior này — cạnh cần tìm nằm đúng ở `ĝ ≈ 0`, tức ngoài
`E_act` theo định nghĩa. Nó phải quét tất, đúng bằng chi phí chế độ 1 ở §7.

> Chạy `g = 1` trên model đã có gate nghĩa là đánh giá nó ở cấu hình nó không vận hành thật.
> Về ngữ nghĩa thì đúng — đó là mốc *"mọi thông điệp chảy đủ"*, cũng chính là hành vi của
> model trước khi thêm gate ở mốc 2.


### 4.5 Tính chất chia đều — kiểm tra nội bộ miễn phí

Một đường chỉ truyền được tin khi mọi gate trên nó đều mở: đóng một gate ở giữa là cả đường
vô dụng, dù các gate còn lại mở hết. Nên đóng góp của đường vào `F_B` là **tích** của các
gate, không phải tổng:

```
số hạng của đường  =  c · ∏_{i=1..h} ( g_i / g*_i )
```

Chia cho `g*_i` chỉ để chuẩn hóa: tại `g = g*` mọi thừa số bằng 1, số hạng bằng đúng `c`.
Với lượt đào nhãn thì `g* = 1` nên phép chia này biến mất luôn.

IG vặn tất cả gate bằng một tham số chung `g = α·g*`, nên mỗi thừa số thành `α`, và có `h`
thừa số — số hạng thành `c·α^h`. Đường càng dài càng tắt nhanh khi `α` giảm. Đạo hàm theo
`g_i` rồi tích phân:

```
IG_i  =  g*_i · ∫₀¹ ( c·α^(h−1) / g*_i ) dα  =  c · [ α^h / h ]₀¹  =  c / h
```

`g*_i` triệt tiêu, nên kết quả không phụ thuộc gate đó mở to hay nhỏ — và cũng không phụ
thuộc ta chọn điểm cuối nào. Chỉ `c` đổi nghĩa: với `g* = 1` nó là giá trị đường khi mọi
gate mở hết, với `g* = ĝ` là giá trị tại định tuyến thật của model.

```
kiểm tra 5:   đường h hop, tổng giá trị c  ⟹  mỗi hop nhận c/h
```

Tích phân chỉ xác nhận một điều vốn đã bị ép buộc: mọi hop trên đường đều thiết yếu như
nhau — bỏ bất kỳ cái nào thì đường chết — nên chúng đối xứng hoàn toàn; completeness thì ép
tổng phải bằng `c`. Đối xứng cộng với tính đầy đủ đã khóa chặt đáp án.

(Lưu ý `h` trần là số hop; `h` có chỉ số dưới — `h_v`, `h_q` — là embedding.)

#### Lệch nhiều nghĩa là đang nhìn hai đường

Credit chia đều cho các hop *của riêng từng đường*. Nhưng một cạnh nằm trên hai đường thì
nhận credit từ cả hai:

```
          e1             e2
     X --------> C -----------> D        π_A   2 hop,  c_A = 0.30
                 |
                 | e3         e4
                 +---> E ---------> D    π_B   3 hop,  c_B = 0.60

bảng IG     e1  =  0.15 + 0.20  =  0.35      ← nằm trên cả hai đường
            e2  =  0.15
            e3  =  e4  =  0.20
                               tổng  =  0.90  =  c_A + c_B
```

Quy hoạch động trả về `{e1, e3, e4}` giá trị `0.75`, nhưng giá trị thật của `π_B` chỉ là
`0.60`. Thừa `0.15` — đúng bằng `c_A/2`, phần rò sang qua cạnh dùng chung.

Áp Kiểm tra 5: `h = 3`, tổng `0.75`, mỗi hop lẽ ra `0.25`.

```
e1  =  0.35    +40%     cạnh dùng chung, bị thổi lên
e3  =  0.20    −20%
e4  =  0.20    −20%     cạnh riêng, bị kéo xuống so với trung bình
```

Hệ quả thứ hai nối thẳng vào `suff`: bật đúng `{e1,e3,e4}` chỉ lấy lại `c_B = 0.60`, vì
`π_A` cần `e2` mà `e2` đang tắt.

```
tỷ phần IG  =  0.75 / 0.90  =  0.83
suff        =  0.60 / 0.90  =  0.67        chênh 0.16  =  phần rò
```

Không phải lỗi cần sửa mà là cấu trúc cần báo cáo đúng: `suff` hụt → vòng greedy §4.11 chạy
tiếp → `π_A` hiện ra → báo cáo một cây phân nhánh tại `C`, không phải một đường.

#### Đọc độ lệch: hình dạng quan trọng hơn độ lớn

`c/h` giả định đóng góp của đường là tích thuần túy của các gate. Model thật còn số hạng
khác và còn phi tuyến, nên nó là mốc lý tưởng chứ không phải đẳng thức đúng tới sai số máy.
Phân biệt bằng **hình dạng** của độ lệch, không bằng biên độ:

```
0.11  0.14  0.17  0.21     tăng đều, giãn cách đều     →  một đường thật
                                                           (chính là ví dụ §4.8)

0.35  0.20  0.20           một hop tách hẳn khỏi       →  hai đường chồng nhau
                           cụm các hop bằng nhau
```

Hai ca lệch ngang nhau về biên độ (±30–40%) nhưng khác hẳn về hình dạng. Một hop đứng riêng
bên trên một cụm đều nhau là chữ ký của cạnh dùng chung.

---

**B. Ra đường đi — từ bảng số thành cấu trúc**

Bảng số chưa phải lời giải thích. Phần này ghép nó thành chuỗi có thứ tự.

### 4.6 Bài toán: đọc ra chuỗi thông điệp

Mỗi tầng GNN là một vòng truyền tin: mọi cạnh cùng gửi thông điệp một lượt. §4.4 đã chấm
xong — mỗi cặp *(cạnh, tầng)* có một điểm IG.

Việc còn lại là đọc ra **chuỗi**: tầng 1 đi cạnh nào, tầng 2 cạnh nào, cho tới khi chạm `D`.

#### Một ví dụ cụ thể

Truy vấn *"vợ của đạo diễn phim X là ai?"*. Seed trích từ truy vấn là node `Phim-X`. Tầng A
chỉ định đích `D` là đoạn nói về bà `Y`. Trong đồ thị:

```
Phim-X  --đạo diễn-->  Ông-Z  --vợ-->  D
```

Chuỗi cần đọc ra: *tầng 1 đi cạnh «đạo diễn», tầng 2 đi cạnh «vợ»*. Hai hop, đúng thứ tự.
Đó vừa là lời giải thích, vừa là nhãn để dạy model lần sau đi đúng lối này.

#### Vì sao không lấy luôn mấy cạnh điểm cao nhất

Đồ thị thật có hàng nghìn cạnh và bảng IG chấm tất cả ở mọi tầng. Lấy top-k thì thường được
một nhúm cạnh rời rạc:

```
X --0.18--> A        P --0.21--> Q        R --0.19--> D
seed                                                  đích
```

Ba con số to nhất, mà không mảnh nào nối `X` tới `D`. Và kể cả khi chúng có nối được, một
*tập* cạnh vẫn không nói **thứ tự** — mà thứ tự chính là thứ multi-hop cần đọc ra.

#### Ràng buộc thứ tự không phải quy ước, nó là nhân quả

Cạnh thứ `i` của chuỗi phải được dùng ở đúng tầng `i`. Nghe như luật tùy tiện, nhưng:

```
tầng 1   Phim-X gửi tin cho Ông-Z    ->  Ông-Z giờ mới "biết" nó liên quan truy vấn
tầng 2   Ông-Z gửi tin cho D         ->  tin này mới có nội dung

nếu tầng 1 không gửi, thì sang tầng 2 Ông-Z vẫn rỗng,
và tin nó gửi cho D chẳng mang thông tin gì
```

Một cạnh điểm IG cao mà node nguồn chưa nhận được gì thì không truyền được gì. Chuỗi phải
liền mạch theo thời gian.

#### Phát biểu gọn

```
đường π    dãy (e₁,1), (e₂,2), ..., (e_h,h)
           nối đuôi nhau, e₁ bắt đầu ở một seed ∈ V_q, e_h kết thúc ở D

giá trị    val(π)  =  Σ_i IG_{e_i}^(i)

cần tìm    π*  =  đường có val lớn nhất
```

Đây chính là khái niệm **circuit** trong mech-interp — đồ thị con nhỏ nhất, liên thông, đủ
tái tạo hành vi đang xét. Khác ở chỗ nó chạy trên đồ thị tri thức chứ không trên attention
head, và trọng số do IG cấp.

Số đường từ seed tới `D` lớn theo hàm mũ nên không duyệt hết được. Hai mục sau tìm `π*`
trong **một lượt quét**: §4.7 đổi cách vẽ đồ thị, §4.8 quét.

### 4.7 Vẽ lại đồ thị: mỗi tầng một bản sao

Ràng buộc *"cạnh thứ `i` ở tầng `i`"* khó cài trên đồ thị gốc, vì cùng một cạnh dùng được ở
nhiều tầng với điểm khác nhau — mà trên đồ thị gốc không có chỗ nào ghi «tầng mấy».

Cách gỡ: **vẽ mỗi node một lần cho mỗi tầng.** Hàng là node, cột là tầng; ô `(v,l)` là trạng
thái của `v` sau `l` vòng truyền tin.

```
             tầng 0     tầng 1     tầng 2     tầng 3
    X          ●          ○          ○          ○
    C          ○          ○          ○          ○
    D          ○          ○          ○          ★
             ● seed                            ★ đích
```

Giờ mỗi thông điệp thành một mũi tên **đi sang phải đúng một cột**: cạnh `v'→v` dùng ở tầng
`l` nối ô `(v',l−1)` sang ô `(v,l)`, mang điểm `IG_e^(l)`. Thêm mũi tên *đứng yên*
`(v,l−1) → (v,l)` điểm 0, cho node nào không nhận gì ở tầng đó:

```
V̂ = V × {0, 1, ..., L}

Ê = { ((v',l-1),(v,l)) : e=(v',r,v) ∈ 𝓔 }       cạnh thật,      điểm IG_e^(l)
  ∪ { ((v,l-1),(v,l))  : v ∈ V }                cạnh đứng yên,  điểm 0
```

Ba thứ được ngay:

1. **Ràng buộc thứ tự biến mất** — nó nằm trong hình vẽ rồi. Mọi đường trên đồ thị này tự
   động dùng cạnh thứ `i` ở đúng tầng `i`.
2. **Không có chu trình**, vì mũi tên chỉ đi sang phải. Quét từ trái sang phải một lượt là
   xong; không cần Dijkstra, thứ tự đã do kiến trúc `L` tầng áp đặt sẵn.
3. **Đích là ô `(D, L)`**, không phải «`D` ở tầng nào cũng được» — vì `p(D)` tính từ
   `h_D^(L)`.

Điểm 0 của mũi tên đứng yên không phải lựa chọn thiết kế: gate gắn trên thông điệp, mà đứng
yên thì không có thông điệp nên không có IG để tính. Nó tồn tại để đường ngắn hơn `L` hop
vẫn sống tới được cột cuối.

### 4.8 Quét một lượt (quy hoạch động)

Ý tưởng gói trong một câu: **đi từ trái sang phải, tới mỗi ô chỉ giữ lại lối tốt nhất đến ô
đó.**

Đứng ở ô `(v,l)` mà hỏi *"có những cách nào tới đây?"* thì chỉ có hai loại — `v` đứng yên từ
`(v,l−1)`, hoặc một cạnh nào đó vừa đổ vào từ `(v',l−1)`. Cả hai loại đều xuất phát từ **cột
`l−1`**, mà cột đó đã tính xong. Nên chỉ việc so vài con số, giữ cái lớn nhất, và ghi lại nó
đến từ đâu:

```
khởi tạo (cột 0)

    best[(v,0)]  =  0      nếu v ∈ V_q          ← mọi seed, không chỉ một
                 =  −∞     nếu không


đệ quy (cột l = 1..L)

    best[(v,l)]  =  max {
                        best[(v, l-1)]                    ← đứng yên, +0
                        best[(v', l-1)] + IG_e^(l)        ← mọi cạnh e=(v',r,v)
                    }

    back[(v,l)]  =  cách nào thắng       (để truy ngược)
```

Không bao giờ phải quay lại cột đã qua — đó là lý do một lượt quét là đủ, và vì sao chi phí
chỉ là `O((|V| + |𝓔|) × L)`.

`−∞` nghĩa là chưa có tín hiệu nào tới được ô này. Cộng gì vào `−∞` cũng vẫn `−∞` — đó là cơ chế tự động chặn các cạnh nổ quá sớm.

**Nhiều seed chỉ đổi dòng khởi tạo.** QHĐ đa nguồn tự tìm đường tốt nhất xuất phát từ bất kỳ seed nào, và truy ngược cho biết nó từ seed nào.

```
Chi phí:  O( (|V| + |E|) × L )        một lượt quét
```

#### `σ(D)` — mấy seed với tới được đích, miễn phí cùng lượt quét

Thêm vào mỗi ô một bitmask rộng `|V_q|`: seed nào với tới được ô đó.

```
reach[(v,0)]  =  {s}   nếu v = s ∈ V_q,    ∅  nếu không

reach[(v,l)]  =  reach[(v, l-1)]                 <- đứng yên
                 ∪  ⋃ reach[(v', l-1)]           <- mọi cạnh e=(v',r,v)

σ(D)  =  | reach[(D,L)] |
```

Khác `best` ở chỗ `best` lấy **max** — một người thắng — còn `reach` lấy **hợp**: mọi seed
với tới được, không riêng seed nằm trên đường tốt nhất.

Chi phí một từ máy mỗi ô khi `|V_q| ≤ 64`, phép hợp là OR bit. Không thêm lượt quét nào.
Nhất quán với tiền kiểm §4.11: `σ(D) = 0` ⟺ `best[(D,L)] = −∞`.

`σ(D)` **không vào công thức nào**. Nó là đại lượng chẩn đoán, dùng ở §6.1 để tách hai ca
«bằng chứng phân tán» khác hẳn nhau.

#### Chạy tay

```
X --e1--> C --e2--> D₇ --e3--> B --e4--> D₁₅       V_q={X}, L=4
```

Bảng IG cho đích `D₁₅`:

```
cạnh          tầng1   tầng2   tầng3   tầng4
e1  X→C        0.11    0.01    0.00    0.00
e2  C→D₇       0.00    0.14    0.01    0.00
e3  D₇→B       0.00    0.00    0.17    0.01
e4  B→D₁₅      0.00    0.00    0.00    0.21
```

Để ý hình chéo: mỗi cạnh chỉ có giá trị ở đúng một tầng. Đó là chữ ký của một chuỗi sạch, và nó đọc ra luôn thứ tự hop.

```
cột 1   C    đứng yên −∞ | e1: best[(X,0)]+0.11 = 0+0.11 = 0.11   → 0.11 từ (X,0)
        D₇   đứng yên −∞ | e2: best[(C,0)]+0.00 = −∞             → −∞
                                        ↑ C chưa nhận gì ở cột 0

cột 2   C    đứng yên 0.11 ← thắng | e1 tầng2: 0+0.01 = 0.01      → 0.11
        D₇   đứng yên −∞   | e2: best[(C,1)]+0.14 = 0.25          → 0.25 từ (C,1)

cột 3   D₇   đứng yên 0.25 ← thắng | e2 tầng3: 0.11+0.01 = 0.12   → 0.25
        B    đứng yên −∞   | e3: best[(D₇,2)]+0.17 = 0.42         → 0.42 từ (D₇,2)

cột 4   B    đứng yên 0.42 ← thắng | e3 tầng4: 0.25+0.01 = 0.26   → 0.42
        D₁₅  đứng yên −∞   | e4: best[(B,3)]+0.21 = 0.63          → 0.63 từ (B,3)
```

```
          tầng0   tầng1   tầng2   tầng3   tầng4
  X         0       0       0       0       0
  C        −∞     0.11    0.11    0.11    0.11
  D₇       −∞      −∞     0.25    0.25    0.25   ★
  B        −∞      −∞      −∞     0.42    0.42
  D₁₅      −∞      −∞      −∞      −∞     0.63   ★
```

Ranh giới giữa `−∞` và số hữu hạn chạy chéo xuống — đó là mặt sóng thông tin, cho thấy tín hiệu lan tới đâu sau mỗi tầng.

### 4.9 Trích đường — max-sum, không phải min-cost

```
π*_D  =  argmax         Σ        IG_e^(l)      =  truy ngược back[] từ ô (D, L)
         π:(V_q,0)→(D,L)  (e,l)∈π
```

Phản xạ thông thường là đặt `cost = −log(w)` rồi tìm đường rẻ nhất, như Dijkstra trên đồ thị
xác suất. **Ở đây làm vậy là sai loại.**

`−log` rồi cộng tương đương với **nhân** các trọng số — hợp lý khi `w` là xác suất của những
sự kiện độc lập. Nhưng IG là một phân rã **cộng tính**: `Σ IG` bằng đúng cái bánh (§4.4,
Kiểm tra 4). Đem phân rã cộng tính xử lý như nhân tính là lệch ngay từ gốc; thiên vị độ dài
dưới đây chỉ là triệu chứng dễ đo nhất.

#### Triệu chứng: thiên vị đường ngắn, phạt hai lần

Trọng số đã chuẩn hóa nên `w ≤ 1`, kéo theo `−log(w) ≥ 0` — mỗi cạnh thêm vào luôn cộng một
khoản dương. Đồng thời §4.5 nói mỗi cạnh chỉ nhận `c/h`, nên đường dài hơn có trọng số mỗi
cạnh **nhẹ hơn**, tức `−log` của nó **lớn hơn**. Hai hiệu ứng nhân lên nhau.

Hai đường giá trị hoàn toàn bằng nhau, `c = 0.60`:

```
                      max-sum                       min(−log)
A   2 hop    0.30 + 0.30         = 0.60      2 × 1.20  =   2.41
B   6 hop    6 × 0.10            = 0.60      6 × 2.30  =  13.82

max-sum    : hòa — đúng, chúng đáng giá như nhau
min(−log)  : A thắng gấp 5.7 lần — sai
```

Viết thành công thức thì hai khoản phạt hiện ra tách bạch:

```
chi phí(h)  =  h · ( −log(c/h) )  =   h·(−log c)   +   h·log h
                                      └─ phạt 1 ─┘     └─ phạt 2 ─┘
                                      do số cạnh       do c/h pha loãng
                                      tuyến tính       trên tuyến tính
```

Điều này chí mạng ở đây vì **đường dài chính là thứ ta muốn tìm** — multi-hop là lý do tồn
tại của cả tầng B. Một hàm mục tiêu thiên vị đường ngắn sẽ chọn lối tắt giả 2 hop và bỏ
chuỗi suy luận thật 4 hop.

Và max-sum còn được thêm hai thứ: không cần ReLU để bỏ cạnh âm — nó tự né, mà vẫn giữ
được thông tin *"cạnh này đẩy ngược"*; và giá trị đường đọc thẳng ra phần trăm, so trực
tiếp được với ngưỡng `suff`. Max-sum trên DAG vẫn tuyến tính — longest-path chỉ khó trên đồ
thị có chu trình.

Truy ngược ví dụ trên:

```
(D₁₅,4) ← e4 tầng4 ← (B,3) ← e3 tầng3 ← (D₇,2) ← e2 tầng2 ← (C,1) ← e1 tầng1 ← (X,0)

giá trị  0.11 + 0.14 + 0.17 + 0.21 = 0.63      khớp ô ★  ✓
```

---

**C. Kiểm chứng và ca khó**

Đường tìm được phải qua một phép thử thật, rồi mới tới các ca rắc rối.

### 4.10 Kiểm chứng (bắt buộc)

Che gate theo `Π` rồi chạy một lượt. Gọi `g*` là điểm cuối của lần chạy IG đang xét (§4.4)
— `ĝ` khi báo cáo, `1` khi đào nhãn:

```
g_Π[e,l]  =  g*[e,l]   nếu (e,l) ∈ Π
          =  0         ngược lại
```

Cả hai đại lượng chuẩn hóa theo cái bánh `F_B(g*) − F_B(0)` nên nằm trong `[0,1]`:

```
             F_B(g_Π) − F_B(0)
suff(Π)  =  ───────────────────        phần cái bánh mà Π lấy lại được
             F_B(g*)  − F_B(0)

             F_B(g*) − F_B(g_¬Π)
nec(Π)   =  ───────────────────        phần cái bánh mất khi tắt Π
             F_B(g*) − F_B(0)

nhận Π   ⟺   suff(Π) ≥ θ_s             θ_s = 0.95
```

Cả chuỗi phía sau — bảng IG, QHĐ, `suff`, cái bánh, `ε̃` — dùng **cùng một `g*`**. Hai lần
chạy ở §4.4 là hai lượt độc lập; đừng trộn số của lượt này vào lượt kia.

Chuẩn hóa là bắt buộc chứ không phải cho đẹp: `F_B` là log xác suất nên âm, so trực tiếp
`suff ≥ θ_s · F_B(g*)` sẽ đòi `suff` tốt hơn cả khi bật đủ gate — điều kiện không bao giờ
thỏa.

```
nec ≈ 1    Π thiết yếu
nec ≈ 0    có đường song song — phát hiện đáng ghi, không phải lỗi
```

IG là bộ sàng, không phải bằng chứng. Bước này không bỏ được — và nó là vai thứ ba trong
một chuỗi ba vai tách bạch:

```
IG     chấm điểm từng (cạnh, tầng)       →  một bảng số, chưa phải đường
       3M ≈ 60 lượt, bất kể bao nhiêu cạnh      đề xuất

QHĐ    nối các điểm đó thành đường       →  đường đi
       O((|V|+|𝓔|)·L), không gọi model          dựng hình

suff   bật đúng đường đó, chạy model     →  nhận hoặc bỏ
       1 lượt forward mỗi lần                   phán quyết
```

Lý do phải tách: IG nói *"nếu vặn mạnh các cạnh này thì p(D) tăng"* — một phát biểu về **đạo
hàm**. Nó không nói *"chỉ riêng các cạnh này là đủ"* — đó là phát biểu về **giá trị hàm tại
một điểm khác**. Model phi tuyến nên hai điều đó lệch nhau.

> **Đừng nhầm `suff` với tỷ phần IG.** Tỷ phần IG (`Σ IG trên đường / tổng IG`) là con số
> *suy ra* từ phép phân rã cộng tính; `suff` là con số *đo thật* khi bật đúng các gate đó
> rồi chạy model. Chúng khác nhau vì model phi tuyến — một đường chiếm 68% tỷ phần IG có
> thể chỉ đạt `suff = 0.61`. Chỉ `suff` mới quyết định nhận hay không. Xem thêm [Những chỗ dễ sai](cho-de-sai.md) §10.

### 4.11 Hội tụ nhiều seed — greedy xóa cạnh, rồi tỉa ngược

Khi một đích cần nhiều lối đi, một đường không đủ. Không cần Steiner tree — một tiền kiểm và
hai pha, tất cả đều dùng lại QHĐ §4.8.

#### Tiền kiểm: trần của `suff`

Luôn tồn tại tập cho `suff = 1` — bật hết mọi gate. Nhưng ta chỉ được dùng **đường xuất phát
từ `V_q`**, mà cái bánh thì gồm cả đóng góp của những cạnh không đường nào chạm tới:

```
V_q = {X}.  Node Z không với tới được từ X trong L hop, nhưng có cạnh Z -> D ở tầng 3.

g = 0   thông điệp đó không chảy
g = ĝ   nó chảy, làm đổi h_D và kéo theo p(D)

=>  cạnh đó có trong cái bánh  (mẫu số)
=>  nhưng QHĐ không bao giờ đưa nó vào đường nào, vì best[(Z,2)] = -inf   (tử số)
```

`h_Z` không mang thông tin truy vấn, nhưng vẫn là nội dung văn bản và vẫn xê dịch điểm số.
Phần **credit mồ côi** đó chặn `suff` ở một trần nhỏ hơn 1. Trần này đọc miễn phí từ bảng QHĐ:

```
E_reach  =  { (e,l) : best[(v', l-1)] > -inf }      cạnh có thể nằm trên một đường

suff_max  =  tổng IG trên E_reach  /  tổng IG trên mọi (e,l)

nếu suff_max < theta_s  ->  dừng ngay, gom bao nhiêu đường cũng vô ích
```

Kết luận lúc đó **không phải lỗi đồ thị** mà là: lối đúng không xuất phát từ seed. Nguyên nhân
thường gặp là `V_q` trích thiếu thực thể — sửa ở khâu entity linking, không phải khâu xây graph.

#### Thuật toán

```
tiền kiểm   nếu suff_max < theta_s:  dừng, ghi "không giải thích được bằng đường từ seed"

-- chặng 1: gom --------------------------------------------
Π_D <- rỗng
lặp:
    pi   = truy ngược back[] từ (D,L)
    gain = tổng IG của các cạnh mới trong pi
    Π_D <- Π_D + pi

    nếu suff(Π_D) >= theta_s   ->  dừng:  thành công
    nếu gain < eps             ->  dừng:  bằng chứng phân tán   (eps = ε̃·cake)
    nếu |Π_D| = n_max          ->  dừng:  hết ngân sách đọc hiểu

    đặt IG_e^(l) = 0 cho mọi (e,l) thuộc Π_D      # đã trả tiền rồi
    chạy lại QHĐ §4.8

-- chặng 2: tỉa ngược --------------------------------------
cho mỗi pi trong Π_D (theo thứ tự đã thêm):
    nếu suff(Π_D \ {pi}) >= theta_s:
        Π_D <- Π_D \ {pi}                # đường thừa, bỏ
```

#### `n_max` không phải hằng chọn bừa — nó là `⌈θ_s / ε̃⌉`

Tham số gốc là `ε̃`: **một đường phải mang ít nhất bao nhiêu phần cái bánh mới đáng báo cáo**.

```
ε̃  =  eps / [ F_B(g*) − F_B(0) ]          mặc định  ε̃ = 0.10
```

`n_max` suy ra từ đó, không chọn:

```
mỗi đường được nhận có       gain_i ≥ eps
các gain rời nhau            cạnh đã lấy bị đặt IG = 0, nên mỗi cạnh chỉ
                             đóng góp vào gain của đúng một đường
vòng lặp còn chạy nghĩa là   tích lũy < θ_s · cake

   (n−1) · eps  ≤  Σ_{i<n} gain_i  <  θ_s · cake

⟹  n  ≤  ⌈ θ_s / ε̃ ⌉  =  ⌈ 0.95 / 0.10 ⌉  =  10
```

Nên `eps` và `n_max` là **một núm, không phải hai**. Hằng `n_max = 5` tương đương `ε̃ ≈ 0.19`
— tức từ chối mọi đường mang dưới 19% cái bánh, hà khắc hơn nhiều so với vẻ ngoài.

Chặn dùng `Σ gain ≤ cake`, đúng khi mọi cạnh trên đường có IG dương. Nếu đồ thị có cạnh IG
âm thì mẫu số phải là `Σ IG⁺` và chặn nới ra chút; QHĐ max-sum vốn né cạnh âm nên hiếm gặp.

#### Vòng lặp luôn dừng, kể cả khi bỏ hết trần

Mỗi đường được nhận phải mang ít nhất một cạnh mới — cạnh cũ có `IG = 0` nên không sinh
`gain`. Số cạnh với tới được là hữu hạn, nên `n ≤ |E_reach|`. Chặn này vô dụng về mặt số
nhưng bảo đảm thuật toán không chạy vô hạn dù tháo hết mọi ngưỡng.

#### Hai chặn đã thử và loại

```
2|V_q|  (hay |V_q| + k)
    số seed không chặn được phân nhánh — một seed vẫn tách nhiều lối (hình dưới).
    Seed thuộc vế sàn: nếu bằng chứng đến từ k seed thì cần ít nhất k đường.
    Đặt nó làm trần là lỗi của |V_q| + 1 ở bản trước.

mincut (Menger)
    số đường rời cạnh tối đa = mincut, đúng và tính được — nhưng đường của ta
    không rời cạnh:
        X→A→D  và  X→A→E→D   dùng chung thân X→A,  mincut = 1
    mincut đếm lối vào độc lập, không đếm đường cần thiết.
    Ép rời cạnh cũng không cứu được: ca hội tụ ở Ví dụ A dùng chung đuôi F2→D₇
    — ép rời là giết đúng trường hợp thuật toán sinh ra để xử lý.
```

```
        A
      /   \
X -> B --> D          một seed, ba lối, cả ba đều cần
      \   /
        C
```

#### Ba lý do dừng, ba chẩn đoán khác nhau

```
suff đạt θ_s         thành công

gain biên < eps      không lối nào mang đủ, bằng chứng rải mỏng
                     đọc kèm σ(D) của §4.8 để biết phân tán kiểu gì:
                        σ(D) cao  -> nhiều nguồn thật, mỗi nguồn nhỏ — bằng chứng vững
                        σ(D) = 1  -> một nguồn duy nhất và mong manh — đáng ngờ

|Π_D| = n_max        cần hơn n_max lối độc lập mới đủ
                     -> không phải thất bại: đồ thị có nhiều nguồn bằng chứng song song
                        cho D, bản thân nó là thông tin đáng ghi
```

#### Pha 1 — vì sao xóa cạnh

```
cạnh mới    vẫn có giá trị  →  đường kế tiếp được thưởng vì mang thêm cái mới
cạnh đã có  giá trị 0       →  dùng lại miễn phí, không bị tính hai lần
```

`gain` ở trên đo đúng phần mới đó, nên nó tụt tự nhiên khi các đường còn lại chỉ dùng lại
cạnh cũ.

#### Pha 2 — vì sao cần

Greedy luôn lấy đường đơn giá trị cao nhất trước, mà đường đó chưa chắc thuộc tập tối thiểu:

```
ba nhánh A, B, C cùng vào một node
A + B cùng nhau là đủ; C một mình giá trị cao nhất nhưng vô dụng nếu thiếu A,B

vòng 1 chọn C  →  suff thấp
vòng 2 chọn A  →  suff thấp
vòng 3 chọn B  →  suff đạt
→ báo cáo thừa nhánh C
```

Pha 2 phát hiện và loại `C`. Chi phí `|Π_D|` lượt forward, thường 2–3. Cùng tinh thần với
vòng đi biên §3.4: chọn dần từ rỗng rồi dọn lại, đừng tin greedy là xong.

#### Ví dụ A — hội tụ ngay tầng kế tiếp

```
X --đạo_diễn--+
              +--> F2 --included_in--> D₇
Y --đóng------+

IG:  X→F2 tầng1 = 0.22,  Y→F2 tầng1 = 0.19,  F2→D₇ tầng2 = 0.33
     tổng toàn đồ thị = 0.81   (0.07 rải ở cạnh nhiễu)
```

```
vòng 1   best[(F2,1)] = max(0.22 từ X, 0.19 từ Y) = 0.22
         best[(D₇,2)] = 0.22 + 0.33 = 0.55        → π₁ qua X
         suff = 0.61  <  0.95      chưa đủ

vòng 2   xóa (X→F2,1) và (F2→D₇,2)
         best[(F2,1)] = 0.19 từ Y
         best[(D₇,2)] = 0.19 + 0.00 = 0.19        → π₂ qua Y
                               ^ cạnh dùng chung giờ miễn phí

         hợp cạnh: 0.22+0.19+0.33 = 0.74 / 0.81 = 91%
         suff = 0.97    đủ

chặng 2  bỏ π₁ → suff 0.29 ;  bỏ π₂ → suff 0.61    cả hai đều < 0.95
         → giữ cả hai
```

Phép cộng khớp hai cách — đó là hệ quả của việc xóa cạnh:

```
0.55 (π₁ đầy đủ)  +  0.19 (π₂ biên)  =  0.74  =  hợp cạnh
```

Kết quả là một cây hội tụ, tìm ra bằng đúng vòng lặp đường đi:

```
   X --0.22--+
             +--> F2 --0.33--> D₇
   Y --0.19--+
   tầng1          tầng2
```

#### Ví dụ B — hội tụ ở tầng sâu

Độ sâu không đổi gì trong thuật toán: xóa cạnh làm toàn bộ đuôi dùng chung thành miễn phí, bất kể đuôi dài 1 hay 5 cạnh.

```
X --> A --> B --+
                +--> M --> D        hội tụ ở M, tầng 3
Y --> C --> E --+

X→A t1 0.11   A→B t2 0.09   B→M t3 0.12
Y→C t1 0.10   C→E t2 0.08   E→M t3 0.10      M→D t4 0.10
tổng toàn đồ thị = 0.76
```

```
vòng 1   best[(A,1)]=0.11   best[(C,1)]=0.10
         best[(B,2)]=0.20   best[(E,2)]=0.18
         best[(M,3)]= max(0.20+0.12=0.32 qua B,  0.18+0.10=0.28 qua E) = 0.32
         best[(D,4)]= 0.32+0.10 = 0.42           → π₁ = X→A→B→M→D
         suff = 0.58  <  0.95      chưa đủ

vòng 2   xóa cả bốn cạnh của π₁
         best[(B,2)] = 0 + 0 = 0                  nhánh X giờ vô giá trị
         best[(M,3)] = max(0, 0.18+0.10=0.28) = 0.28 qua E
         best[(D,4)] = 0.28 + 0 = 0.28            → π₂ = Y→C→E→M→D
                               ^ M→D miễn phí

         hợp cạnh: 0.70 / 0.76 = 92%
         suff = 0.96    đủ
```

#### Điểm hội tụ đọc được miễn phí

```
π₁ = (X,0) (A,1) (B,2) (M,3) (D,4)
π₂ = (Y,0) (C,1) (E,2) (M,3) (D,4)
                        \___/
              ô chung có l nhỏ nhất = điểm hội tụ
```

So hai danh sách ô, lấy ô chung có `l` nhỏ nhất. Không cần thuật toán riêng.

#### Cảnh báo: tỷ phần IG không đo được tầm quan trọng của nhánh

Theo tính chất `c/h` ở §4.5, một cấu trúc gồm `h` cạnh cùng cần thiết chia đều `c` cho cả `h`. Nên tỷ phần của một nhánh phản ánh số cạnh của nó, không phải tầm quan trọng:

```
nhánh X  4 cạnh  →  0.42  =  55%
nhánh Y  3 cạnh  →  0.28  =  37%       nhưng hai nhánh thiết yếu ngang nhau
```

Cực đoan hơn:

```
X --> A --> B --> C --> M --> D        5 cạnh  →  83% tỷ phần IG
Y --------------------> M               1 cạnh  →  17% tỷ phần IG

bỏ Y đi thì D chết hệt như bỏ X đi
```

Càng sâu, hai nhánh càng dễ lệch độ dài, tỷ phần IG càng dễ gây hiểu nhầm.

```
Sai    so hai nhánh bằng tổng IG        (bị số cạnh chi phối)
Đúng   so bằng mức tụt của suff khi bỏ nhánh đó đi
```

Phép đo đúng tốn 1 lượt forward mỗi nhánh, và chặng 2 đã chạy sẵn rồi:

```
bỏ nhánh X  →  suff tụt về 0.31
bỏ nhánh Y  →  suff tụt về 0.35       →  thiết yếu ngang nhau
trong khi tỷ phần IG nói 55% / 37%
```

> **Thay thế cho chặng 1:** giữ `k` giá trị tốt nhất mỗi ô (k-best), chi phí `O((|V|+|E|)·L·k)`. Nhanh hơn, nhưng vòng lặp xóa cạnh xử lý hội tụ tự nhiên hơn vì cạnh dùng chung tự thành miễn phí.

---
### 4.12 Nhiều đích — chia chung lượt forward

`F_B` khác nhau theo đích nên không gộp được — gộp thành `Σ_D log p(D)` thì rẻ hơn nhưng mất khả năng biết cạnh nào phục vụ đích nào.

Tiết kiệm bằng cách dùng chung forward:

```
tại mỗi α:   1 forward  →  |T| lần backward trên cùng đồ thị đó

M=20, |T|=6:   20 forward + 120 backward
     thay vì   120 forward + 120 backward
```


## 5. Nhiều đích và quan hệ phụ thuộc

Một đích có thể nằm trên đường tới đích khác:

```
X ──sinh_ở──→ C ──inc_in──→ D₇ ──mentions──→ B ──thuộc──→ D₁₅
                             │
                        D₇ vừa là đích, vừa là trạm trung chuyển
```

Xảy ra được vì QuadGraph có cạnh liên tầng hai chiều: entity → document, rồi document → entity khác. Đây chính là multi-hop ở mức tài liệu.

### 5.1 Quan hệ này không bao giờ vòng

```
D₇ tới được ở tầng 2,  D₁₅ ở tầng 4 qua D₇
→ không có cách nào D₁₅ nằm trên đường tới D₇  (chỉ số tầng chỉ tăng)
```

Nên quan hệ phụ thuộc giữa các đích tự nó là một DAG, và tầng-tới-được cho sẵn thứ tự topo.

### 5.2 Hai thứ rơi ra miễn phí từ bảng QHĐ

```
depth(D) = min { l : best[(D,l)] > −∞ }
```

tức cột đầu tiên mà ô hết `−∞`. Trong bảng ở §4.8: `depth(D₇)=2`, `depth(D₁₅)=4`. Không cần lượt quét riêng.

```
D là trạm trung chuyển của D'   ⟺   truy ngược của D' chạm ô (D, l) nào đó
```

Trong ví dụ §4.9, truy ngược `D₁₅` đi xuyên qua ô `(D₇, 2)` — mà `D₇ ∈ T`. Phát hiện quan hệ phụ thuộc không cần thuật toán riêng, nó lộ ra ngay trong lúc dựng đường.

### 5.3 Giải theo thứ tự nông → sâu

```
── bước 1: thứ tự ─────────────────────────────────────────
sắp T theo depth(D) tăng dần        (đọc từ bảng QHĐ, miễn phí)

── bước 2: giải lần lượt ──────────────────────────────────
Ecov <- rỗng                        # cạnh đã nằm trong báo cáo

cho mỗi D theo thứ tự depth:
    IG_D  <- chạy IG với F_B = log p(D)
    IG_D[e,l] <- 0  với mọi (e,l) thuộc Ecov      # chỉ cho mục tiêu tìm kiếm
    chạy QHĐ + greedy §4.11  →  Π_D
    Ecov <- Ecov + cạnh(Π_D)
```

Kết quả trên ví dụ:

```
── D₇  (depth 2) ──────────────────────────────
đường X → C → D₇            0.52 / 0.60 = 87%
Ecov = { (X→C,1), (C→D₇,2) }

── D₁₅ (depth 4) ──────────────────────────────
bảng IG của D₁₅ sau khi xóa:
    X→C    tầng1   0.11 → 0.00   đã có
    C→D₇   tầng2   0.14 → 0.00   đã có
    D₇→B   tầng3   0.17
    B→D₁₅  tầng4   0.21
                   ───── mới = 0.38
```

Báo cáo thành một cây, không phải hai đường rời:

```
X ──0.24──→ C ──0.28──→ D₇ ★          87% của D₇
                         │
                         └──0.17──→ B ──0.21──→ D₁₅ ★    bổ sung 0.38
```

Trung thực hơn hẳn: báo riêng thì hai cạnh đầu bị tính hai lần, người đọc tưởng có hai chuỗi bằng chứng trong khi chỉ có một chuỗi phân nhánh.

### 5.4 Chỗ dễ sai: suff tính trên toàn đường

```
xóa cạnh  →  chỉ để làm mục tiêu tìm kiếm và báo cáo giá trị biên
suff      →  luôn tính trên toàn bộ cạnh của đường, kể cả tiền tố

Sai:   suff(D₁₅) = bật chỉ { D₇→B, B→D₁₅ }
Đúng:  suff(D₁₅) = bật cả  { X→C, C→D₇, D₇→B, B→D₁₅ }
```

Bật thiếu tiền tố thì `D₇` không có tín hiệu, `B` cũng không, `p(D₁₅)` tụt — bạn sẽ kết luận nhầm là đường sai.

### 5.5 Tiền tố có thể khác nhau giữa các bảng IG

Bảng IG của `D₁₅` và của `D₇` là hai bảng khác nhau (hai `F_B` khác nhau):

```
trong bảng của D₁₅:   đường tới D₇ giá trị 0.25
trong bảng của D₇:    đường tới D₇ giá trị 0.52
```

Thường cấu trúc trùng, chỉ khác độ lớn. Nếu tiền tố khác nhau thì đó là thông tin thật, không phải lỗi:

```
cùng tiền tố   →  một chuỗi duy nhất, D₇ thật sự là trạm trung chuyển
khác tiền tố   →  có hai lối tới D₇; lối tốt cho D₇ khác lối tốt để đi tiếp
                  → báo cáo cả hai, đừng gộp
```

Khi kiểm tra quan hệ phụ thuộc, so tập cạnh, đừng chỉ nhìn node.

### 5.6 Đánh đổi

Thuật toán này phụ thuộc thứ tự: giải `D₁₅` trước có thể ra lối trực tiếp khác không qua `D₇`. Sắp theo `depth` là lựa chọn có cơ sở (khớp chiều thông tin chảy) nhưng là greedy, không tối ưu toàn cục. Tối ưu thì thành Steiner tree nhiều đích, NP-khó. Với `|T|` cỡ 5–8 thì greedy theo depth là đủ, và cái bạn được lại là cấu trúc đọc hiểu được.

---

## 6. Nối hai tầng và train

```
nhãn mỗi ví dụ = ( q ,  T ,  { Π_D : D ∈ T } ,  { depth(D) } ,  { w(D) } )
```

**Chỉ tập `T` đi qua ranh giới hai tầng; không điểm số nào đi qua** — xem [Những chỗ dễ sai](cho-de-sai.md) §1.

**Ký hiệu mục này giới thiệu**

```
w(D)             trọng số theo quan hệ phụ thuộc giữa các đích
S_k              tập ví dụ «lỗi model» ở mốc k — cũng chính là tập nhãn
rank_GFM(D)      vị trí của node tài liệu D khi sắp p(v) giảm dần
k_top(q)         ngưỡng hạng, theo từng ví dụ: ⌈β·|T(q)|⌉, β ≈ 1.5
k_infer          ngưỡng top-k lúc chạy thật — dùng để báo cáo, không để lọc
V_neg            node âm tính lấy mẫu (dùng chung cho L_node và L_hop)
V_hop(l)         miền cộng của L_hop ở tầng l (§6.3)
λ₁, λ₂           hệ số của L_hop và L_edge
R(T), R(V_q)     tập node với tới được từ T / từ V_q trong vài hop
```

### 6.1 Bảng chẩn đoán — chạy ở mỗi mốc

Bảng này **không chạy một lần rồi thôi**, và nó không nằm trước hay sau việc train — nó nằm
*giữa*, và lặp lại ở mỗi mốc (§6.5):

```
mốc 0    đào T  -->  train L_node tới hội tụ
                      |   model đã cố hết sức với nhãn rẻ nhất
                      v
mốc 1          +===================+
               |  bảng chẩn đoán   |   <- chạy tầng B ở đây
               +=========+=========+
                 +-------+--------+
      đọc cách 1 |                | đọc cách 2
 "tiến độ tới đâu?"          "lấy nhãn ở đâu?"
  phân bố các ô              ô «lỗi model» = S_1
                                    |
                                    v
                         train L_node + lambda_1 * L_hop
                                    v
mốc 2          +===================+
               |  bảng chẩn đoán   |   <- chạy lại
               +===================+   ...
```

Một lần chạy, hai cách đọc: **báo cáo về đợt train vừa xong**, và **nguồn nhãn cho đợt tới**.

#### Lấy số từ đâu

Bảng không phải một thủ tục riêng: nó là đầu ra của tầng B đọc theo một góc khác. Bốn đại
lượng, cả bốn rơi ra từ một lần chạy tầng B:

```
đại lượng        lấy từ đâu                              chi phí
──────────────────────────────────────────────────────────────────────
rank_GFM(D)      1 lượt forward, đọc p(v) của mọi node   rẻ
best[(D,L)]      bảng QHĐ §4.8 — hữu hạn hay −∞?         miễn phí
suff(Π_D)        kiểm chứng §4.10                        1 lượt mỗi đường
|Π_D|, độ dài    truy ngược                              miễn phí
σ(D)             bitmask reach §4.8                      miễn phí
```

Hai điều kiện chính đo **hai thứ độc lập nhau**, nên xếp thành hai trục chứ không phải một
danh sách:

```
suff(Π_D)     đồ thị có chứa lối đi kiểm chứng được không?
rank_GFM(D)   model có thật sự dùng lối đó không?
```

**Tiền kiểm** — chạy trước, vì chúng loại hẳn ví dụ ra khỏi bảng chính:

```
trên truy vấn q    |T(q)| > k_infer    ->  lệch cấu hình
                                           nới k_infer, hoặc hạ theta ở §3.3

trên từng (q,D)    best[(D,L)] = -inf  ->  lỗi cấu trúc: D không với tới trong L hop
                                           tăng L, hoặc nối lại đồ thị

                   suff_max < theta_s  ->  lối đúng không xuất phát từ seed
                                           sửa entity linking (V_q thiếu),
                                           hoặc chấp nhận giới hạn phương pháp
```

**Bảng chính** — hai trục:

```
                     rank <= k_top(q)         rank > k_top(q)
                     model lấy được D         model bỏ sót D
   ---------------------------------------------------------------------
   suff >= theta_s   ổn                       lỗi model  ->  tập S_k
   có lối, đã kiểm   không cần làm gì         đường tồn tại mà model không đi
                                              Π và ŵ của nó là nhãn train

   suff <  theta_s   nghi đường tắt           lỗi đồ thị
   không lối nào đủ  model đúng mà không      bổ sung cạnh lúc xây graph
                     lối nào giải thích nổi
```

Ô **lỗi model** vừa là một chẩn đoán, vừa là tập nhãn. Chẩn đoán và đào nhãn là cùng một thao
tác, chỉ khác cách đọc kết quả.

Ô **đường tắt** là ca đáng chú ý: model lấy đúng `D` mà không đường liền mạch nào qua được
kiểm chứng — nó đang trả lời bằng tín hiệu lan man chứ không bằng chuỗi suy luận. Đây là một
chỉ số kiểm toán nên báo cáo riêng:

```
tỷ lệ đường tắt  =  |{ rank tốt, suff < theta_s }|  /  |{ rank tốt }|
```

Lấy mẫu 5–10% ví dụ `rank` tốt để đo, vì bộ lọc rẻ-trước-đắt-sau bình thường bỏ qua chúng.

Khi rơi vào ô **lỗi đồ thị**, đọc thêm lý do dừng của greedy (§4.11) để biết sửa ở đâu:

```
gain biên < eps    bằng chứng phân tán — đọc kèm σ(D):
                      σ(D) cao -> nhiều nguồn thật, mỗi nguồn nhỏ — vững
                      σ(D) = 1 -> một nguồn mong manh — đáng ngờ
|Π_D| = n_max      cần hơn n_max lối độc lập — đồ thị dư thừa cao, đáng ghi
```

#### «Xếp sai» nghĩa là gì

`rank_GFM(D)` là vị trí của node tài liệu `D` khi sắp mọi node tài liệu theo `p(v)` giảm dần:

```
rank_GFM(D)  =  | { v : type(v) = document,  p(v) > p(v_D) } |  +  1
```

Đây không phải phát biểu trừu tượng về xác suất. Vượt ngưỡng nghĩa là **lúc chạy thật, LLM
sẽ không nhìn thấy đoạn này** — dù tầng A đã xác nhận nó cần.

#### `k_top` phải thay đổi theo từng ví dụ

```
k_top(q)  =  ⌈ β · |T(q)| ⌉           β ≈ 1.5
```

Để `k_top` là hằng số thì hỏng theo cách khó thấy. Với `k_top = 5` và `|T| = 8`, ít nhất ba
đích **luôn luôn** bị gắn "lỗi model" bất kể model tốt đến đâu:

```
ba ví dụ đó không bao giờ rời khỏi S_k
    →  |S_{k+1}| ≈ |S_k| ngay từ mốc 2
    →  tiêu chí dừng §6.5 báo bão hòa giả
    →  kết luận nhầm "lỗi đồ thị, train thêm vô ích"
```

`k_top = |T|` là đòi hỏi chặt nhất mà vẫn luôn thỏa mãn được — model xếp đúng tập cần vào
đầu danh sách thì không còn gì để dạy. Hệ số `β ≈ 1.5` chỉ nới chút cho thực dụng.

Hai ngưỡng, hai mục đích, đừng trộn:

```
bộ lọc train        rank ≤ k_top(q)    "model đã xếp tốt nhất có thể chưa?"
báo cáo hiệu năng   rank ≤ k_infer     "đoạn này có thật sự được truy hồi không?"
```

#### Thứ tự: rẻ trước, đắt sau

`rank_GFM(D)` rẻ hơn IG cả trăm lần, nên lọc bằng nó **trước khi** chạy tầng B:

```
bước A   rẻ     1 lượt forward mỗi ví dụ  →  rank_GFM(D)
                loại ngay ví dụ model đã xếp đúng
                còn lại  F = { model đang xếp sai }

bước B   đắt    chạy tầng B chỉ trên F
                IG → QHĐ → greedy → suff
                tách F theo bảng chẩn đoán ở trên
```

Chi phí IG chỉ trả cho `|F|` ví dụ thay vì cả tập.

#### Vì sao nó đứng sau `L_node` nhưng trước `L_hop`

Hai đại lượng của bảng đều vô nghĩa trên model chưa train: `rank_GFM(D)` thì mọi đích đều tệ,
`suff` thì IG chỉ là nhiễu. Chạy sớm là ra *100% lỗi model* — không phân biệt được gì.

Mà mục đích của bảng chính là phân biệt **model dở** với **đồ thị thiếu**. Muốn tách được
thì model phải được cho cơ hội công bằng trước đã. Nên nó đứng:

```
sau   L_node     rẻ, chỉ đổi nhãn, không đổi kiến trúc
trước L_hop      thêm head p^(l) mỗi tầng
      L_edge     thêm module gate — đổi kiến trúc
```

Bảng là cái cổng quyết định có nên bước qua vạch đó không. Nếu phần lớn rơi vào **lỗi đồ
thị** hoặc **lỗi cấu trúc** thì dừng lại sửa graph, đừng thêm loss. Với QuadGraph, "không có
đường trong `L` hop" thường nghĩa là thiếu cạnh liên tầng (entity→document), không phải
thiếu cạnh KG.

Khi lý do dừng của greedy vẫn chưa tách bạch, dùng thêm **độ dài đường** làm gợi ý: đường
dài sát `L` thì nghiêng về đồ thị thiếu lối; có đường ngắn mà `suff` vẫn thấp thì nghiêng về
bằng chứng phân tán.

#### Đọc phân bố qua các mốc

```
             mốc 1    mốc 2    mốc 3
ổn            45%      68%      74%
lỗi model     38%      17%      12%     ← S_k, co lại = đang ăn
lỗi đồ thị    14%      13%      13%     ← không co = phải sửa graph
lỗi cấu trúc   3%       2%       1%
```

Hàng thứ hai chính là thứ mà tiêu chí dừng §6.5 đọc.

### 6.2 Trọng số theo quan hệ phụ thuộc

Nhãn các đích không độc lập:

```
model trượt D₇  ⟹  không đời nào với tới D₁₅     (D₁₅ chỉ tới được qua D₇)
```

BCE phẳng giả định độc lập nên sai ở đây. Hai cách chữa, dùng được cả hai:

```
w(D) = 1 + | { D' ∈ T : D nằm trên Π_{D'} } |        trọng số

ví dụ:  w(D₇) = 2,  w(D₁₅) = 1

curriculum: train đích depth nhỏ trước        (depth cũng chỉ có từ mốc 1)
```

**`w(D)` chưa có ở mốc 0.** Công thức trên cần `Π`, mà `Π` chỉ xuất hiện từ mốc 1 (§6.5).
Ở mốc 0 dùng `w(D) = 1` cho mọi đích, tức BCE phẳng. Hai cách xử lý:

```
a)  w(D) = 1 ở mốc 0, bật w(D) thật từ mốc 1        đơn giản, đúng

b)  suy quan hệ phụ thuộc ngay ở tầng A, không cần đồ thị:
        v({D₁₅}) ≈ 0  nhưng  v({D₇,D₁₅}) cao   ⟹  D₁₅ phụ thuộc D₇
    → w(D) độc lập với model, có từ mốc 0, cache cùng T
```

Cách (b) giữ đúng nguyên tắc «cái gì không phụ thuộc GFM thì cache vĩnh viễn», nhưng tốn
thêm vài lượt LLM cho các cặp đích. Cách (a) an toàn và miễn phí.

### 6.3 Ba hàm mất mát

```
L_node = − Σ_{D ∈ T} w(D) · log p(D)  −  Σ_{v ∈ V_neg} log( 1 − p(v) )

L_hop  = Σ_{l=1..L} Σ_{v ∈ V_hop(l)}  BCE( p^(l)(v),  1[ ∃π ∈ Π : (v,l) ∈ π ] )

L_edge = Σ_{(e,l)}  BCE( ĝ_e^(l),  ŵ_e^(l) )          ŵ = IG chuẩn hóa về [0,1]

L = L_node + λ₁ · L_hop + λ₂ · L_edge
```

`L_node` thay nhãn cũ "đoạn có chứa chuỗi đáp án" — nhãn cũ đầy dương tính giả (đoạn nhắc tới đáp án nhưng generator không dùng).

`L_hop` cần thêm predictor nhỏ `p^(l)` ở mỗi tầng. Đây là giám sát trung gian, vá chỗ credit assignment bị đứt qua `L` bước — xem [Những chỗ dễ sai](cho-de-sai.md) §4.

**`L_hop` phải lấy mẫu âm, không cộng trên mọi node.** Cộng hết thì tỷ lệ dương/âm vô vọng:

```
ô dương   node nằm trên Π, đúng tầng     ≤ n_max đường × L hop   →  cỡ 10 ô
ô âm      mọi ô còn lại                  |V| × L                 →  cỡ 10⁷ ô
```

Ở tỷ lệ `1 : 10⁶` thì nghiệm tầm thường *"luôn đoán 0"* đã gần tối ưu và loss gần như không
nhúc nhích. `L_node` đã tránh điều này bằng `V_neg`; `L_hop` gặp đúng vấn đề đó, còn nặng
hơn vì nhân thêm `L`. Nên hạn chế miền cộng:

```
V_hop(l)  =  { v : (v,l)  ∈ Π }              dương — node trên đường, đúng tầng
          ∪  { v : (v,l') ∈ Π,  l' ≠ l }     âm khó — đúng node, sai tầng
          ∪  V_neg                           âm thường — dùng lại mẫu của L_node
```

Nhóm giữa mới là nhóm đáng giá: nó dạy **thời điểm**, thứ mà nhãn chỉ ở tầng cuối không nói
được. Ví dụ ngay dưới có đúng một ô như vậy — `(D₇, tầng1) = 0` — và tài liệu vốn đã chỉ ra
nó là ô đáng chú ý nhất. Lấy mẫu kiểu này giữ nguyên mọi ô mà ví dụ đó nêu.

`L_edge` dạy trực tiếp cho gate phụ thuộc truy vấn ở §4.2 chế độ 2.

**Cả ba loss đọc từ một lượt forward.** Một lần chạy GFM cho ra `p(v)`, `p^(l)(v)` và `ĝ`
cùng lúc, nên chi phí train không nhân ba.

#### Nhãn cho từng loại node

QuadGraph có bốn loại node, mà `T` chỉ nói về đơn vị văn bản. Chia vai:

```
              vào prompt?   nguồn nhãn      loss
document         có         T  (tầng A)     L_node
community        có         T  (tầng A)     L_node
entity           không      Π  (tầng B)     L_hop
attribute        không      Π  (tầng B)     L_hop
```

Quy tắc chốt: **thứ gì vào prompt thì phải từng là người chơi trong trò chơi §3.3.** Nếu
nhét vào prompt một thứ Shapley chưa bao giờ chấm thì `T` không phủ được nó, `L_node` không
giám sát nó, và chuỗi quy gán đứt.

`entity` và `attribute` không bao giờ vào prompt. Chúng là **chất nền để lan truyền** và là
vật liệu cho lời giải thích — đường `Phim-X → Ông-Z → D` chạy xuyên qua chúng — nên được
giám sát gián tiếp qua `Π`, không qua `T`.

Kéo theo: `k_infer` là ngưỡng của **riêng** `document` + `community`. G-reasoner vẫn xuất
top-`k` cho cả bốn loại, nhưng hai loại kia không phải ngân sách truy hồi.

**`p^(l)` dùng `4L` head**, một head cho mỗi cặp (tầng GNN × loại node). Khớp với
`Predictor_{type(v)}` mà G-reasoner vốn đã có, và cần thiết vì tần suất "nằm trên đường" của
`document` khác hẳn của `entity` — một head chung sẽ trộn hai phân bố rất khác nhau. Head
nhỏ nên `4L ≈ 16–24` head không đáng kể.

**`V_neg` lấy trong loại `document`**, và tầng A cho sẵn hai nguồn với chất lượng khác hẳn
nhau:

```
β_i < 0              negative tốt nhất — đoạn đẩy ngược câu trả lời (§3.7)
không nằm trong MWC nào   negative thật — vô dụng ở mọi tập đủ tối tiểu

nằm trong một số MWC nhưng không trong S*
                     không phải negative. Đoạn này có ích, chỉ là lời giải thích
                     khác đã dùng nó. Phạt nó là dạy retriever tránh một đoạn đúng.
```

Đây là chỗ bản MWC đọc sạch hơn bản Shapley: "không nằm trong MWC nào" là một phát biểu dứt
khoát, còn "thuộc cụm bị cắt" thì phụ thuộc ngưỡng cắt cụm.

Lấy mẫu đều trên toàn đồ thị cũng sai: phần lớn sẽ rơi vào `entity`/`attribute` (đông hơn
nhiều), và loss chỉ dạy được "đây không phải node loại document" — phân biệt **loại**, không
phải phân biệt **nội dung**.

#### Ví dụ số — cùng một đồ thị cho cả hai loss

```
   X --e1--> C --e2--> D7          đường đúng,  V_q={X}, L=3
    \                   ↗
     \--e3--> J --e4--/            nhánh lạc

Π_D7 = [ (e1,tầng1), (e2,tầng2) ]   →  ô: (X,0) (C,1) (D7,2) (D7,3)
```

**L_hop** — nhãn đọc thẳng từ Π, `y=1` nếu ô nằm trên đường:

```
nhãn y⁽ˡ⁾(v)        tầng1  tầng2  tầng3        model đang đoán p⁽ˡ⁾(v)
  X                   0      0      0           0.30   0.20   0.15
  C                   1      0      0           0.45   0.30   0.25
  J                   0      0      0           0.40   0.35   0.30
  D7                  0      1      1           0.10   0.35   0.55

ba số hạng nặng nhất:
  (C,  tầng1)  y=1 p=0.45  →  −log 0.45 = 0.799   "nâng C ở tầng 1"
  (D7, tầng2)  y=1 p=0.35  →  −log 0.35 = 1.050   "nâng D7 ở tầng 2"
  (D7, tầng3)  y=1 p=0.55  →  −log 0.55 = 0.598   "giữ D7 tới cuối"
                                        L_hop = 5.236
```

Chú ý ô `(D7, tầng1) = 0`: D7 không được phép sáng ở tầng 1 vì tin chưa kịp tới. Nhãn
chỉ ở tầng cuối không nói được điều này.

**L_edge** — nhãn là IG chuẩn hóa `ŵ = IG / max IG`:

```
ŵ_e⁽ˡ⁾     tầng1  tầng2  tầng3      ĝ hiện tại   tầng1  tầng2  tầng3
e1 X→C      1.00   0.06   0.03                    0.62   0.55   0.50
e2 C→D7     0.00   0.94   0.10                    0.58   0.60   0.52
e3 X→J      0.06   0.00   0.00                    0.65   0.50   0.48
e4 J→D7     0.00   0.03   0.00                    0.55   0.52   0.50

ba số hạng nặng nhất:
  e3 tầng1   ŵ=0.06 ĝ=0.65  →  1.013   "đóng X→J, nó vô dụng"
  e2 tầng1   ŵ=0.00 ĝ=0.58  →  0.868   "đóng C→D7 Ở tầng 1"
  e2 tầng2   ŵ=0.94 ĝ=0.60  →  0.535   "nhưng mở C→D7 ở tầng 2"
                                        L_edge = 8.671
```

Hai dòng `e2` là cả bài học: cùng một cạnh, đóng ở tầng 1 và mở ở tầng 2. Gate chung
cho cả `L` tầng không diễn đạt nổi câu đó — nó chỉ học được một con số trung bình.

> **BCE với nhãn mềm không bao giờ về 0.** Sàn của nó là entropy của chính nhãn,
> `Σ H(ŵ) = 1.276` trong ví dụ này. Thấy `L_edge` dừng ở `1.28` là đã hội tụ, không
> phải kẹt.

```
L_hop    giám sát node theo tầng     "ai sáng lúc nào"      kết quả
L_edge   giám sát cạnh theo tầng     "lối nào mở lúc nào"   cơ chế
```

Không thừa nhau: `L_hop` nói *C phải sáng ở tầng 1* nhưng không nói đi qua cạnh nào để làm
C sáng (có thể nhiều cạnh vào C). Điểm chung là cả hai dạy thời điểm — đúng thứ mà
`L_node`, chỉ nhìn `p(v)` ở tầng cuối, mù hoàn toàn.

### 6.4 Bộ lọc cho giám sát đường đi

Chỉ áp `L_hop` và `L_edge` lên ví dụ thỏa cả ba:

```
D ∈ T                      tầng A nói cần
rank_GFM(D) > k_top(q)     model đang xếp sai    ← nếu đã đúng thì dạy số 0
                           k_top(q) = ⌈1.5·|T(q)|⌉, theo từng ví dụ (§6.1)
suff(Π_D) ≥ θ_s            đường đã kiểm chứng   ← thầy sai thì đừng dạy
```

Lý do điều kiện thứ hai: `T` đến từ LLM nên là thông tin mới thật sự; còn `Π` đến từ chính
GFM nên phần lớn là tự chưng cất. Tập con mang thông tin mới là những ca model đang sai — ở
đó IG chỉ ra các cạnh *lẽ ra* phải dùng để nâng `p(D)` mà model hiện chưa dùng.

Câu trên **chỉ đúng khi đo với `g* = 1`** (§4.4). Nếu đo bằng `ĝ` thì cạnh model chưa dùng
nhận `IG ≈ 0` và không bao giờ vào `Π`; `L_edge` khi đó chỉ dạy gate củng cố đúng cái nó
đang làm.

### 6.5 Lịch đào nhãn và vòng lặp khởi động

**Nhãn không được tính trong lúc train.** Chúng do một job riêng sinh ra, chạy *giữa* các
đợt train, với trọng số GFM đóng băng, ghi ra file. Lúc train chỉ đọc file đó.

```
job đào nhãn     model đóng băng, chạy inference + IG, ghi file nhãn
train            đọc file, cập nhật trọng số — nhãn giữ nguyên suốt nhiều epoch
job đào nhãn     model đóng băng (phiên bản mới), ghi đè file nhãn
train            ...
```

#### Nhãn nào có từ lúc nào

```
nhãn        nguồn                        phụ thuộc model?   có từ
─────────────────────────────────────────────────────────────────────────────
T, neg      tầng A — MWC trên LLM        không              mốc 0, cache vĩnh viễn
Π           tầng B — IG + QHĐ            có                 mốc 1, đào lại từng mốc
ŵ           tầng B — IG chuẩn hóa        có                 mốc 1, đào lại từng mốc
depth(D)    tầng B — bảng QHĐ §4.8       có                 mốc 1
w(D)        suy từ Π                     có                 mốc 1  (mốc 0 dùng w = 1)
```

Bốn hàng dưới **đều cần tầng B**, mà tầng B cần GFM — nên không thứ nào có ở mốc 0. Kéo
theo: curriculum theo `depth` (§6.2) cũng chỉ bật được từ mốc 1.

`Π` và `ŵ` ra từ **cùng một bảng `IG_e^(l)`**, chỉ khác cách đọc:

```
bảng IG ──┬── QHĐ + greedy §4.11   →  Π_D  và  depth(D)
          └── chuẩn hóa về [0,1]   →  ŵ_e^(l)
```

#### Vì sao `Π` và `ŵ` không thể có từ đầu

```
IG_e^(l)  =  ĝ_e^(l) × ∂F_B / ∂g_e^(l)
                        └── đạo hàm của model, phụ thuộc trọng số hiện tại
```

Để biết *"model lẽ ra phải đi lối nào"* thì phải hỏi chính model, mà câu trả lời đổi khi
model đổi. Cách gỡ vòng luẩn quẩn: đóng băng model, hỏi một lần, train một hồi trên câu trả
lời đó, rồi hỏi lại. `T` thì khác hẳn — nó đến từ LLM sinh, không liên quan GFM, nên đào một
lần là xong vĩnh viễn.

#### Vì sao không đào lại mỗi epoch

```
đào một ví dụ     M forward              =  20
                + M·|T| backward         = 120 bwd  ≈ 240 forward-tương-đương
                + kiểm chứng             ≈  13
                                           ─────
                                           ≈ 273 lượt forward-tương-đương

train một ví dụ   1 fwd + 1 bwd          ≈   3 lượt
```

Đào một ví dụ tốn bằng **~91 bước train** trên chính nó, nên đào lại cả tập tương đương 91
epoch tính toán. Cứu cánh là bộ lọc §6.4: chỉ đào cho ví dụ model **đang sai**, và tập đó co
lại theo thời gian.

#### Job đào nhãn làm gì

```
job  mine_stage_B( checkpoint, tập ví dụ ):

    tải trọng số GFM từ checkpoint, đóng băng — không cập nhật gì

    cho mỗi ví dụ q:
        1 forward  →  p(v) mọi node  →  rank_GFM(D) cho từng D ∈ T
        nếu mọi D đều có rank ≤ k_top(q):
            bỏ qua ví dụ này            ← model đã đúng, Π sẽ chỉ là tự chưng cất
        ngược lại, với mỗi D còn lại:
            M forward + M backward trên gate   →  bảng IG_e^(l)
            QHĐ trên bảng đó                   →  best[], back[]
            depth(D) = cột đầu tiên hết −∞
            truy ngược + greedy + suff §4.11   →  Π_D
            IG / max IG                        →  ŵ
        w(D) = 1 + |{ D′ : D nằm trên Π_{D′} }|

    ghi file nhãn
```

#### Ba mốc, và một ví dụ đi qua chúng

```
mốc 0   một lần, vĩnh viễn — chỉ chạy LLM, không đụng GFM
        { q, V_q, T = [D₇], neg = [...], w = {D₇: 1} }
        chưa có Π, chưa có ŵ, chưa có depth
        train  L_node  tới hội tụ

mốc 1   chạy job trên, gate = 1 (chế độ 1)
        { ... cộng thêm:
          rank_lúc_đào = {D₇: 23},        ngoài k_top → ví dụ này được giữ
          depth = {D₇: 2},
          Π = {D₇: [ [(X→F2,t1), (F2→D₇,t2)],
                     [(Y→F2,t1), (F2→D₇,t2)] ]},
          ŵ = {(X→F2,t1): 0.67, (Y→F2,t1): 0.58, (F2→D₇,t2): 1.00, ...},
          suff = {D₇: 0.97},  w = {D₇: 1} }

        Π dùng ngay      →  L_node + λ₁·L_hop
        ŵ nằm trong file nhưng chưa dùng — chưa có ĝ để dạy

mốc 2   thêm module gate (chế độ 2) vào model, lọc lại S₂ ⊂ S₁, chạy lại job
        cách đo không đổi — vẫn g* = 1 như mốc 1 (§4.4)
        Π và ŵ bị ghi đè bằng bản sắc hơn
        ŵ giờ mới dùng được  →  + λ₂·L_edge

mốc 3   tùy chọn, lặp lại mốc 2
```

Ba mốc này chính là lời giải cho bài toán con gà quả trứng ở trên: mốc 1 chạy IG với gate
hằng bằng 1, mốc 2 chạy lại với gate đã học.

Giữa hai mốc thì **nhãn đóng băng** — train nhiều epoch trên cùng một `Π`. Chấp nhận được vì
`Π` mã hóa "lẽ ra phải đi lối nào", thứ không đổi nhanh theo điểm số của model. Chi phí một
lần đào ở mốc 1 là `|S₁| / |D|` phần chi phí đầy đủ; nếu 20% ví dụ sai thì `0.2 × 91 ≈ 18`
epoch tính toán — đáng nếu làm 2–3 lần, không đáng nếu làm hàng chục lần.

#### Tiêu chí dừng

```
dừng khi  |S_{k+1}| ≈ |S_k|        tập lọc ngừng co lại
```

Tiêu chí này **chỉ đúng khi `k_top` đặt theo từng ví dụ** (§6.1). Với `k_top` hằng số, những
ví dụ có `|T|` lớn kẹt lại trong `S_k` vĩnh viễn và làm tiêu chí báo bão hòa giả.

Tập lọc không nhỏ đi nữa nghĩa là `L_hop` và `L_edge` đã hết chỗ sửa: đào thêm chỉ lấy về
nhãn cho những ca model sẽ tiếp tục sai vì lý do khác, thường là thiếu cạnh trong đồ thị —
hàng thứ ba của bảng §6.1.

Thực tế 2–3 mốc là bão hòa. Nếu mốc 3 vẫn co mạnh thì nhiều khả năng đang học nhiễu của
corpus chứ không phải kỹ năng định tuyến — giảm `λ₂` hoặc đa dạng hóa corpus đào nhãn.

### 6.6 Vòng lặp ngược — dùng đích để tìm ứng viên mới

```
bình thường:  seed = V_q  (thực thể trong câu hỏi)
vòng ngược:   seed = T    (đoạn tầng A đã xác nhận là tốt)
```

Phần sắc nhất nằm ở phép trừ tập hợp:

```
R(T)    = node với tới được từ T trong vài hop
R(V_q)  = node với tới được từ V_q

ứng viên đáng giá  =  R(T) \ R(V_q)
```

Tức đoạn mà bằng chứng tốt dẫn tới, nhưng câu hỏi không dẫn tới. Nếu nằm trong cả hai thì retriever lẽ ra đã tìm được rồi, chẳng học được gì mới. Phần hiệu mới là chỗ exposure bias đang trốn.

```
vòng n:    pool = bm25 + GFM + ngẫu nhiên          → tầng A → T_n
vòng n+1:  pool = pool cũ + top20( R(T_n) \ R(V_q) ) → tầng A → T_{n+1}
```

Chi phí phía đồ thị gần bằng 0. Chi phí thật là chạy lại tầng A trên pool đã mở rộng — `f_θ` dùng lại được nhưng vòng 0 và vòng 3 phải chạy lại, nên giới hạn 10–20 node. Hai vòng là bão hòa; vòng 3 vẫn lôi ra đoạn `φ` cao nghĩa là pool ban đầu hẹp nghiêm trọng.

### 6.7 Kỳ vọng thực tế

Single-hop gần như không cải thiện — nhãn node đã nói đủ, không có lộ trình gì để học. Multi-hop là chỗ nên ăn. G-reasoner đã đạt EM 81.1 trên HotpotQA (gần trần), nên đo trên bộ khó hơn (MuSiQue, 2Wiki). Đây là giả thuyết có cơ sở, chưa phải kết quả đã chứng minh.

---

## 7. Chi phí

### 7.1 Một truy vấn, pipeline đầy đủ

```
Tầng A    n + 3                              vòng 0: v(∅), v(C), v(R), mọi singleton (§3.4)
        + (r − 1) · m                        vòng 1: đi dọc biên, shortlist theo cụm
        + n_act                              vòng 2: active learning
        + |S*| + 2^|S*| − 2                  vòng 3: xác nhận đủ VÀ quét hết tập con (§3.10)
Tầng B    M forward + M·|T| backward  +  ( 1 + 2·Σ_D |Π_D| ) forward   lượt GFM
          1 lượt chung cho F_B(g*)  — một forward cho ra p(v) của mọi node
          2 lượt mỗi đường          — suff ở chặng 1, tỉa ngược ở chặng 2
```

Quy ra token, với `n = 40`, `C = 6`, `m = 12`, `r = 3`, `|S*| = 3`:

```
vòng 0    v(∅) 500 + v(R) 3.500 + v(C) 10.500 + 40 singleton 30.000    44.500
vòng 1    (r−1)·m = 24 lượt                                            27.000
vòng 2    ~30 lượt                                                     37.000
vòng 3    1 + 6 lượt                                                    6.500
                                                                     ─────────
                                                                      115.000 token
                                                             ≈ $0,35 ở $3/triệu
```

Tầng B gần như miễn phí bên cạnh: `20 fwd + 120 bwd + 13 fwd` trên một mô hình `34M` tham số.
Hai nửa chênh khoảng **140 lần**.

Hệ quả: **cache tập đích `T` lại.** Nhãn `(q → T)` không phụ thuộc kiến trúc GFM — đào một
lần, dùng lại cho mọi lần train, mọi thay đổi đồ thị, mọi giá trị `L`.

### 7.2 Quy mô thật — hai pha

Con số `$0,35` ở trên nhân với số truy vấn thật thì không trả nổi, nên §3.11 tách hai pha:

```
                           nạp vào    chạy đủ   $/truy vấn       tổng
pha 1, pipeline đầy đủ      18.200     10.000     $0,38        $3.850
pha 2, đường rẻ            200.000    110.000     $0,037       $7.500
cứu, có phân tầng            3.000      3.000     $0,34        $1.000
hiệu chuẩn HotpotQA (§3.10)    500        500     $0,38          $190
                                                              ─────────
                                                               $12.540    ~104k nhãn

chạy pipeline đầy đủ cho cả 200.000                            $76.000
```

Cột *"nạp vào"* khác cột *"chạy đủ"* vì cửa thoát `A0`/`C0` (§3.11) bỏ `~45%` truy vấn với giá
`400` token — chúng là các ca mà bảng phủ nói pool **không chứa** tập đủ. Chúng không bị xóa:
chúng là cờ *"pool thiếu"*, nguồn nuôi ô «lỗi đồ thị» §6.1 và vòng lặp ngược §6.6.

Khoảng **3,9 lần rẻ hơn**. Phần tiết kiệm đến từ việc bỏ vòng 0 ở pha 2 — 40 lượt singleton
(`30.000` token) thay bằng `g_θ`, cộng vòng 1–2 (`64.000`). **Không** từ việc cắt
verification, và **không** từ việc bỏ `v(C)`: `v(C)` chỉ là `10.500` trong `106.000` token
tiết kiệm được, tức 10%.

Chia theo bước ở pha 2:

```
C1  v(∅)                      500 token    2%
C3  v(R)                    3.500 token   16%
C4  v(C)                   10.500 token   48%
C6  verification, kỳ vọng   7.400 token   34%
                          ─────────────
                           21.900 token
```

`v(C)` thành phần nặng nhất, và đó là **đúng giá của một cái trần chính xác**: bù nó bằng một
hằng số hiệu chuẩn thì `~10%` truy vấn bị trần thấp, sinh nhãn sai mà verification không bắt
được (§3.11). Còn cắt `C6` thì `T` không tối tiểu vẫn được xuất, và tầng B học hướng thừa.

### 7.3 Sổ dữ liệu

Chi phí trên mua về hai thứ cùng lúc: nhãn cho tầng B, và dữ liệu train cho `g_θ`.

```
nguồn                                truy vấn    nhãn/truy vấn      tổng
pha 1, bốn vòng                       10.000         111         1.110.000
verification đường rẻ                188.000          10         1.880.000
tỉa, |S*| = 6..10                      8.000          13           104.000
cứu -> pipeline đầy đủ                 7.400         111           821.000
ca trượt bị bỏ, không cứu             13.600           5            68.000
                                                               ───────────
                                                                 3.983.000
```

Pha 2 **sinh ra nhiều dữ liệu train hơn cả pha 1**, và chúng nằm đúng vùng phân bố mà beam
tra tới còn greedy chưa đi qua. Đó là lý do chạy theo lô (§3.11): mỗi lô nạp lại thì `p` tăng
và lô sau rẻ hơn.

Ba ngưỡng `|S*|` ở §3.11 đổi cán cân này: trước đây mọi ca `|S*| ≥ 6` đều bị cứu nên nhánh
leo thang đóng gần nửa số nhãn. Giờ phần lớn chúng được **tỉa** tại chỗ, chỉ cho `~13` mẫu
thay vì `111`. Mất `~780.000` mẫu, nhưng `13` mẫu đó nhắm đúng lỗi *dư-âm-chưa-học* hơn một
lượt greedy chạy lại từ rỗng, và `$2.080` tiết kiệm được mua lại `~6.100` lượt cứu ở tầng
khác. Gần như một đổi một, mà được thêm `~8.000` nhãn.

### 7.4 So với bản Shapley

Bản Shapley ở [file riêng](tang-a-shapley.md) tốn `~128` lượt mỗi truy vấn — xấp xỉ bằng. Giá
như nhau nhưng đổi được cả họ MWC, chỉ số cấu trúc và độ tin cậy thay vì một tập `T` duy
nhất. Cái mất là completeness (§8).

Điều bản Shapley **không** làm được là chế độ hai pha: điểm `φ_c` của nó định nghĩa trên toàn
bộ pool nên không có cách nào xấp xỉ bằng một hàm tập rồi xác nhận lại bằng vài lượt. MWC thì
có, vì "đủ và tối tiểu" là một vị từ kiểm được trên một tập con cụ thể.

### Quy tắc chọn IG hay ablation

```
IG rẻ hơn ablation  ⟺  N > 3M
```

với `N` = số đơn vị cần chấm, điều kiện: prompt hoặc đồ thị không phình theo `N`. Hệ số 3 vì backward ≈ 2× forward.

```
cạnh đồ thị (hàng nghìn, đồ thị cố định)      →  IG
token trong đoạn (hàng trăm, prompt cố định)  →  IG
đoạn trong pool (prompt dài theo số đoạn)     →  ablation / tập đủ tối tiểu
```

Chứng minh dòng cuối — khi prompt phình theo `K`, gọi `c` là chi phí forward trên prompt ngắn:

```
standalone = K · c
IG         = M · 3 · (K · c)          prompt dài ≈ K × ngắn
tỷ lệ      = 3M                       ← K triệt tiêu, không có điểm hòa vốn
```

Cảm giác "IG chấm một lượt được hết K đoạn" là ảo giác, vì mỗi lượt phải nuốt cả K đoạn.

---

## 8. Bất biến phải kiểm

```
tầng A
1.  v(S*)  ≥  τ_q                                     đo bằng model thật
2.  v(S* \ {c})  <  τ_q      mọi c ∈ S*               S* thật sự tối tiểu
3.  { i : β_i < 0 }  ∩  S*  =  ∅                      không đưa đoạn gây hại vào đích
4.  Σ_i credit(c_i)  =  1                             theo cấu trúc

tầng B
7.  Σ_{e,l} IG_e^(l)        =  F_B(g*) − F_B(0)
8.  đường h hop giá trị c   →  mỗi hop ≈ c/h          (kiểm tra nội bộ)
```

1–3 phải chạy bằng **model thật**, không phải `f_θ`; đó là toàn bộ ý nghĩa của chúng. 4 đúng
theo cấu trúc. 5 chỉ đúng tới sai số rời rạc hóa của `M` — lệch quá 5% thì tăng `M`. 6 là
chẩn đoán, lệch nhiều nghĩa là bạn đang nhìn hai đường chứ không phải một.

**Bộ của tầng A giờ yếu hơn bản Shapley.** Đẳng thức `Σ φ_c = F(E) − F(E')` là một ràng buộc
chính xác trên **toàn bộ phép tính**, bắt được mọi lỗi cài đặt ở khâu quy gán. Phép kiểm 1–3
chỉ xác nhận `S*`, không xác nhận cách ta tìm ra nó. Đó là cái giá của việc bỏ completeness,
và nên nói thẳng chứ đừng giấu.

Bù lại bằng đối chứng: chạy [bản Shapley](tang-a-shapley.md) trên một mẫu nhỏ, kiểm xem `S*`
có nằm trong nhóm đoạn mà Shapley chấm cao không. Lệch nhiều là dấu hiệu `f_θ` hoặc `τ_q`
đang sai.

**Không có bất biến nào nối tầng A với tầng B.**

---

## 9. Thứ tự triển khai

```
1  Tầng A đầy đủ, xuất T                         rẻ nhất, dùng được ngay
2  Đổi nhãn train GFM sang T, w(D) = 1          chắc ăn nhất
                                                 (w(D) thật cần Π, có từ mốc 1)
3  Chạy bảng chẩn đoán §6.1                      lỗi ở model hay ở đồ thị?
4  Nếu lỗi model → thêm L_hop                    cách dễ trước
5  Nếu vẫn thiếu → gate query-aware + L_edge     đổi kiến trúc, làm sau cùng
6  Vòng lặp ngược §6.6                           khi nghi exposure bias
```

```
bước 1–2  =  mốc 0  (§6.5)       bước 3–4  =  mốc 1       bước 5  =  mốc 2
```

**Đừng làm bước 5 trước bước 3.** Nếu phần lớn lỗi là thiếu cạnh trong đồ thị, mọi cải tiến loss đều vô nghĩa.

---

## Tham chiếu

**Quy gán**

1. L. S. Shapley. *A value for n-person games.* Contributions to the Theory of Games II, 1953.
2. M. Sundararajan, A. Taly, Q. Yan. *Axiomatic Attribution for Deep Networks.* ICML 2017.
3. G. Owen. *Multilinear extensions of games.* Management Science 18(5), 1972.
4. J. Deegan, E. W. Packel. *A new index of power for simple n-person games.* International Journal of Game Theory 7(2), 1978. — credit chia đều trong mỗi liên minh thắng tối tiểu.
5. J. F. Banzhaf III. *Weighted voting doesn't work: A mathematical analysis.* Rutgers Law Review 19, 1965.
6. M. J. Holler, E. W. Packel. *Power, luck and the right index.* Journal of Economics 43(1), 1983. — Public Good Index.

**Train retriever bằng tín hiệu từ LM**

7. G. Izacard, E. Grave. *Distilling Knowledge from Reader to Retriever for Question Answering.* ICLR 2021.
8. G. Izacard và cộng sự. *Atlas: Few-shot Learning with Retrieval Augmented Language Models.* JMLR 24, 2023. — nguồn của ADist / PDist / LOOP / EMDR².

**Graph foundation model cho RAG**

9. L. Luo, Z. Zhao, R. Haffari, D. Phung, C. Gong, S. Pan. *GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation.* [arXiv:2502.01113](https://arxiv.org/abs/2502.01113), NeurIPS 2025. — 8M tham số, chạy được trên dataset chưa từng thấy mà không cần fine-tune.
10. L. Luo, S. Pan và cộng sự. *G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge.* [arXiv:2509.24276](https://arxiv.org/abs/2509.24276), 2025. — QuadGraph, 34M tham số; kiến trúc mà tài liệu này bám theo.

**Giải thích GNN bằng mask cạnh**

11. R. Ying và cộng sự. *GNNExplainer: Generating Explanations for Graph Neural Networks.* NeurIPS 2019.
12. M. S. Schlichtkrull, N. De Cao, I. Titov. *Interpreting Graph Neural Networks for NLP With Differentiable Edge Masking.* ICLR 2021.
