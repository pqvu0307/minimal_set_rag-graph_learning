# Quy gán nguồn cho RAG: Shapley theo cụm + Integrated Gradients

## 1. Vấn đề

Mô hình trả lời xong. Bạn đưa cho nó 20 đoạn văn bản truy hồi. **Đoạn nào thực sự tạo ra câu trả lời đó?**

Câu hỏi khó ở hai chỗ:

- **Dư thừa.** Nếu 3 đoạn cùng nói một ý, không thể tính công đầy đủ cho cả 3 — như vậy là cộng khống.
- **Chi phí.** Cách công bằng về mặt toán học (Shapley) đòi hỏi thử mọi tổ hợp đoạn: $2^{20} \approx 10^6$ lượt chạy mô hình. Không khả thi.

Hai công cụ quy gán sẵn có, mỗi cái hỏng một nửa:

| | Mạnh ở | Yếu ở |
|---|---|---|
| **Shapley** | Công bằng theo tiên đề, xử lý đúng dư thừa và tương tác | Chi phí $2^K$ |
| **Integrated Gradients (IG)** | Rẻ, chi tiết tới từng token | "Mù" dư thừa: hai đoạn trùng nội dung đều được tính công đầy đủ |

**Cách ghép.** Gom các đoạn giống nhau thành cụm → số "người chơi" tụt từ $K=20$ xuống $C=6$, Shapley chạy được. Shapley chia công trạng giữa các cụm, IG chia tiếp bên trong từng cụm:

> **Shapley quyết định *bao nhiêu*. IG quyết định *ở đâu*.**

Toàn bộ quy trình:

```
20 đoạn  →  6 cụm  →  Shapley: φ_c  →  giữ 3 cụm tốt  →  IG trong 3 cụm  →  điểm cuối cho từng đoạn
         gom cụm              2^6 lượt        cắt tỉa            50 lượt           ghép hai tầng
```

Chi phí: **114 lượt chạy** thay vì một triệu.

Và quan trọng nhất — điểm số cuối cùng của 20 đoạn **cộng lại đúng bằng** tổng ảnh hưởng của khâu truy hồi. Không mất mát, không tính trùng. Đây là thứ khiến quy trình này là *nguyên lý* chứ không phải *heuristic*.

---

## 2. Ký hiệu

| Ký hiệu | Nghĩa |
|---|---|
| $q = (q_1,\dots,q_{T_q})$ | truy vấn của người dùng |
| $D_1,\dots,D_K$ | các đoạn truy hồi ($K$ thường 5–20) |
| $x = [q, D_1, \dots, D_K]$ | prompt đầy đủ, dài $T = T_q + \sum_k T_k$ token |
| $E \in \mathbb{R}^{T \times d}$ | ma trận embedding của prompt; $E_{t,j}$ là phần tử $(t,j)$ |
| $\mathrm{span}(D_k)$ | tập chỉ số token thuộc đoạn $D_k$ (các tập này rời nhau) |
| $F(E) \in \mathbb{R}$ | **điểm mục tiêu** — đại lượng ta muốn giải thích |
| $E'$ | **đường cơ sở**: giữ truy vấn, xóa hết các đoạn |
| $\mathcal{C}_1,\dots,\mathcal{C}_C$ | các cụm đoạn; $\kappa(k) = c$ nghĩa là $D_k \in \mathcal{C}_c$ |
| $\phi_c$ | giá trị Shapley của cụm $c$ |

Hai điểm cần nói rõ ngay:

**Tại sao quy gán trên embedding, không phải token?** Token ID rời rạc, không khả vi — không có gradient để lấy. Embedding liên tục nên IG mới chạy được.

**$F$ là gì?** Bất kỳ đại lượng vô hướng khả vi nào đo "mô hình tin câu trả lời đến đâu":

$$
F(E) =
\begin{cases}
\log P(y \mid q, D_1,\dots,D_K) & \text{log-xác suất — chuẩn cho RAG, bao quát cả câu trả lời} \\[4pt]
z_{y^*}(E) & \text{logit — rẻ hơn, tránh bão hòa softmax} \\[4pt]
z_{y^*}(E) - \max_{k \neq y^*} z_k(E) & \text{biên — đo trực tiếp mức ưu tiên so với đối thủ mạnh nhất}
\end{cases}
$$

Chọn cái nào cũng được, miễn **dùng nhất quán** suốt cả quy trình.

**$E'$ là gì?** Giữ nguyên truy vấn, đưa embedding của mọi đoạn về 0:

$$E'_{t,j} = \begin{cases} E_{t,j} & t \le T_q \\ 0 & t > T_q \end{cases}$$

$F(E')$ chính là điểm số mô hình đạt được *chỉ bằng tri thức có sẵn trong tham số*. Vậy nên:

$$F(E) - F(E') \;=\; \text{toàn bộ phần mà khâu truy hồi đóng góp}$$

Đây là **cái bánh** mà cả quy trình sẽ đem chia. Mọi đẳng thức "tính đầy đủ" phía sau đều quy về con số này.

---

## 3. Bước 1 — Gom cụm

Nhúng từng đoạn bằng một sentence encoder, chạy k-means:

$$\kappa(k) = \arg\min_{c} \lVert \mathrm{emb}(D_k) - \mu_c \rVert_2, \qquad \mu_c = \frac{1}{|\mathcal{C}_c|}\sum_{k \in \mathcal{C}_c} \mathrm{emb}(D_k)$$

Mỗi đoạn vào đúng một cụm; các cụm phủ kín $\{1,\dots,K\}$.

Bước này làm **hai việc** cùng lúc:

1. **Giảm chi phí**: $2^K \to 2^C$. Đây là lý do toàn bộ quy trình chạy được.
2. **Gom dư thừa**: những đoạn nói cùng một điều rơi vào cùng một cụm, nên Shapley sẽ chia công trạng ở *mức ngữ nghĩa* thay vì xử lý từng bản sao như một nguồn độc lập.

**Chọn $C$ bao nhiêu?** $C \in [4,8]$, tức $2^C \in [16,256]$. Nhỏ hơn thì quá thô, lớn hơn thì Shapley đắt lên theo hàm mũ.

---

## 4. Bước 2 — Shapley trên các cụm

Coi mỗi cụm là một **người chơi** trong một trò chơi hợp tác. Lợi ích của một liên minh $S \subseteq \{1,\dots,C\}$:

$$v(S) = F(E_S)$$

trong đó $E_S$ là prompt chỉ gồm truy vấn và các đoạn thuộc cụm trong $S$ (phần còn lại bị xóa về 0). Nói cách khác, $v(S)$ trả lời: *"nếu chỉ cho mô hình xem nhóm cụm này, nó tin câu trả lời đến đâu?"*

Hai đầu mút: $v(\emptyset) = F(E')$ (không đoạn nào) và $v(\{1,\dots,C\}) = F(E)$ (đủ cả).

**Giá trị Shapley** của cụm $c$:

$$\phi_c = \sum_{S \subseteq \{1,\dots,C\} \setminus \{c\}} \frac{|S|!\,(C-|S|-1)!}{C!} \Big[ \underbrace{v(S \cup \{c\}) - v(S)}_{\text{thêm } c \text{ vào } S \text{ lợi bao nhiêu}} \Big]$$

Đọc công thức này theo cách dễ hình dung hơn — xếp các cụm vào một hàng theo thứ tự ngẫu nhiên, rồi hỏi: *trung bình thì $c$ làm điểm số tăng bao nhiêu khi đến lượt nó?*

$$\phi_c = \mathbb{E}_{\pi}\big[\, v(\mathrm{Pre}_c^{\pi} \cup \{c\}) - v(\mathrm{Pre}_c^{\pi}) \,\big], \qquad \mathrm{Pre}_c^{\pi} = \{c' : \pi(c') < \pi(c)\}$$

Hai dạng này tương đương; trọng số giai thừa ở dạng trên chính là xác suất để tập cụm đứng trước $c$ đúng bằng $S$. Dạng hoán vị là dạng dùng cho ước lượng Monte Carlo khi $C$ lớn.

**Kết quả đọc thế nào:**

| $\phi_c$ | Nghĩa |
|---|---|
| lớn, dương | cụm đóng góp thật |
| $\approx 0$ | mô hình bỏ qua cụm này |
| âm | cụm **đẩy ngược** câu trả lời — nhiễu hoặc mâu thuẫn |

**Dư thừa được xử lý ở đây.** Hai cụm chứa cùng một bằng chứng sẽ *chia nhau* công trạng, vì khi cụm kia đã có mặt trong liên minh thì đóng góp biên của cụm này gần như bằng 0. Một cụm vô dụng có đóng góp biên bằng 0 trong mọi liên minh, nên $\phi_c = 0$.

**Tính đầy đủ** — tiên đề then chốt:

$$\boxed{\;\sum_{c=1}^{C} \phi_c = F(E) - F(E')\;}$$

Cả cái bánh được chia hết, không dư không thiếu.

**Chi phí:** $2^C$ lượt forward (mỗi tập con một lượt). $C=6 \Rightarrow$ 64 lượt.

---

## 5. Bước 3 — Cắt tỉa

Giữ lại các cụm đáng kể, bỏ phần còn lại:

$$C^* = \{c : \phi_c > \tau\} \quad\text{(hoặc: top-}m\text{ cụm theo }\phi_c\text{)}, \qquad C^- = \{1,\dots,C\} \setminus C^*$$

Dựng lại prompt rút gọn $x^* = [\,q, \{D_k : \kappa(k) \in C^*\}\,]$ với ma trận embedding $E^*$.

Lý do cắt tỉa thuần túy là kinh tế: IG tốn một lượt forward **+ backward** cho mỗi bước nội suy, nên đừng tiêu tiền đó lên những token mà Shapley đã nói là vô nghĩa. Các đoạn thuộc $C^-$ nhận điểm cuối bằng 0.

---

## 6. Bước 4 — IG bên trong các cụm giữ lại

Shapley đã nói *cụm nào* quan trọng. Giờ cần biết *token nào trong cụm đó* làm nên chuyện.

**Ý tưởng:** vặn một núm từ "các cụm giữ lại đều tắt" sang "bật hoàn toàn", rồi cộng dồn gradient dọc đường đi.

Đường cơ sở ($\alpha = 0$) và đường đi:

$$
\gamma_{t,j}(\alpha) =
\begin{cases}
E^*_{t,j} & t \le T_q \quad \text{(truy vấn — luôn bật)} \\
\alpha\,E^*_{t,j} & t \in \mathrm{span}(C^*) \quad \text{(cụm giữ lại — hiện dần)}
\end{cases}
$$

Quy gán cho từng chiều embedding của từng token:

$$\mathrm{IG}_{t,j} = \underbrace{E^*_{t,j}}_{\text{đầu vào} - \text{cơ sở}} \cdot \underbrace{\frac{1}{M}\sum_{m=1}^{M} \frac{\partial F(\gamma(\alpha_m))}{\partial \gamma_{t,j}(\alpha_m)}}_{\text{gradient trung bình dọc đường}}, \qquad \alpha_m = \frac{m}{M}$$

Gọn lại: **IG = (đầu vào − đường cơ sở) × gradient trung bình**, là xấp xỉ Riemann của tích phân đường.

Token truy vấn có $\mathrm{IG} = 0$ theo đúng thiết kế, vì chúng giống nhau ở cả hai đầu đường đi.

Cộng lên để có điểm token rồi điểm đoạn:

$$S_t = \sum_{j=1}^{d} \mathrm{IG}_{t,j}, \qquad S^{\mathrm{IG}}_{D_k} = \sum_{t \in \mathrm{span}(D_k)} S_t$$

Dấu có ý nghĩa: dương = ủng hộ câu trả lời, âm = đẩy ngược lại.

**Chi phí:** $M$ lượt forward + backward. $M = 20$–$100$ là đủ; $M = 50$ là mức điển hình.

> ⚠️ **Một chỗ rất dễ hiểu nhầm.** IG cũng có đẳng thức đầy đủ của riêng nó, nhưng nó quy về **prompt rút gọn**:
> $$\sum_{t \in \mathrm{span}(C^*)} S_t = F(E^*) - F\big((E^*)'\big) \;\neq\; F(E) - F(E')$$
> Nghĩa là **không được dùng $S^{\mathrm{IG}}$ làm điểm cuối cùng.** Ở bước sau, IG chỉ đóng vai *tỷ lệ phân chia*; còn *ngân sách* luôn do Shapley cấp. Chính sự phân vai này giữ cho tính đầy đủ đầu–cuối không vỡ.

---

## 7. Bước 5 — Ghép hai tầng

Với đoạn $D_k$ thuộc cụm $c = \kappa(k) \in C^*$:

$$\boxed{\;S^{\mathrm{final}}_{D_k} = \underbrace{\phi_c}_{\substack{\text{ngân sách của cụm} \\ \text{(Shapley)}}} \times \underbrace{\frac{S^{\mathrm{IG}}_{D_k}}{\sum_{j \in \mathcal{C}_c} S^{\mathrm{IG}}_{D_j}}}_{\substack{\text{tỷ phần trong cụm} \\ \text{(IG)}}}\;}$$

Đoạn thuộc cụm bị cắt tỉa: $S^{\mathrm{final}}_{D_k} = 0$.

### Ví dụ bằng số

Giả sử $F$ là log-xác suất, và $F(E) - F(E') = 4.0$ nat. Shapley trả về:

| Cụm | $\phi_c$ | Quyết định |
|---|---|---|
| $\mathcal{C}_1$ | 2.40 | giữ |
| $\mathcal{C}_2$ | 1.10 | giữ |
| $\mathcal{C}_3$ | 0.60 | giữ |
| $\mathcal{C}_4$ | 0.00 | cắt |
| $\mathcal{C}_5$ | 0.00 | cắt |
| $\mathcal{C}_6$ | −0.10 | cắt (nhiễu) |
| **Tổng** | **4.00** | = $F(E)-F(E')$ ✓ |

Cụm $\mathcal{C}_1$ có 3 đoạn, IG trả về $S^{\mathrm{IG}} = 3.0,\ 1.0,\ 2.0$ (tổng 6.0). Chia $\phi_1 = 2.40$ theo tỷ lệ đó:

$$D_a: 2.40 \times \tfrac{3.0}{6.0} = 1.20, \qquad D_b: 2.40 \times \tfrac{1.0}{6.0} = 0.40, \qquad D_c: 2.40 \times \tfrac{2.0}{6.0} = 0.80$$

Cộng lại đúng 2.40 — bằng $\phi_1$. Làm tương tự cho $\mathcal{C}_2, \mathcal{C}_3$.

### Tính đầy đủ đầu–cuối

Trong mỗi cụm, tổng tỷ phần bằng 1, nên $\sum_{k \in \mathcal{C}_c} S^{\mathrm{final}}_{D_k} = \phi_c$: ngân sách của cụm được chia trọn vẹn. Cộng tất cả các cụm:

$$\boxed{\;\sum_{k=1}^{K} S^{\mathrm{final}}_{D_k} \;=\; \sum_{c \in C^*} \phi_c \;=\; \big(F(E) - F(E')\big) - \underbrace{\sum_{c \in C^-} \phi_c}_{\text{phần dư do cắt tỉa}}\;}$$

Trong ví dụ trên, phần dư là $0 + 0 - 0.1 = -0.1$ — nhỏ, nên tổng điểm 20 đoạn là 4.10, xấp xỉ trọn vẹn 4.0 của khâu truy hồi.

**Phần dư là chỉ số chẩn đoán, nên báo cáo kèm.** Nó bằng 0 khi cắt tỉa chỉ loại những cụm có $\phi_c \approx 0$ — lúc đó đẳng thức đầy đủ đúng tuyệt đối. Phần dư lớn nghĩa là bạn đã cắt mất tín hiệu thật: hạ $\tau$ hoặc tăng $m$.

---

## 8. Chi phí

$$\text{Tổng} = \underbrace{2^C}_{\text{Shapley}} + \underbrace{M}_{\text{IG}}$$

| Thành phần | $K=20,\ C=6,\ M=50$ |
|---|---|
| Shapley trên cụm | $2^6 = 64$ lượt forward |
| IG | 50 lượt forward + backward |
| **Tổng** | **114 lượt** |
| *Shapley đầy đủ trên 20 đoạn (để so sánh)* | *$2^{20} \approx 1{,}050{,}000$ lượt* |

Chi phí bị chi phối bởi số hạng $2^C$. Giữ $C \le 8$ thì cả quy trình gói gọn trong vài trăm lượt forward — chạy được trong production.

---

## 9. Những chỗ dễ sai

1. **Xóa bằng vector 0 chỉ là xấp xỉ của "bỏ đoạn đi".** Vector 0 nằm ngoài phân phối embedding tự nhiên, nên $v(S)$ dính nhiễu off-manifold. Thay thế: dùng token `[MASK]`, hoặc cắt hẳn token rồi đánh lại chỉ số vị trí.

2. **Phải dùng chung $F$ và chung quy ước xóa ở cả hai giai đoạn.** Lệch một chỗ là chuỗi đẳng thức đầy đủ gãy, và điểm số cuối cùng mất ý nghĩa.

3. **Mẫu số ở Bước 5 có thể tiến về 0 hoặc đổi dấu** khi các đoạn trong cùng cụm có IG trái dấu — tỷ phần sẽ bùng nổ hoặc lật dấu. Hai cách chữa, cả hai đều giữ nguyên $\sum_{k \in \mathcal{C}_c} S^{\mathrm{final}}_{D_k} = \phi_c$:
   - chia theo độ lớn: $\lvert S^{\mathrm{IG}}_{D_k}\rvert \big/ \sum_j \lvert S^{\mathrm{IG}}_{D_j}\rvert$;
   - khi $\lvert \sum_j S^{\mathrm{IG}}_{D_j} \rvert < \varepsilon$: chia đều $\phi_c / |\mathcal{C}_c|$.

4. **$C$ và $\tau$ là hai núm đánh đổi chính.** $C$: chi phí đổi lấy độ mịn ngữ nghĩa. $\tau$: chi phí IG đổi lấy phần dư do cắt tỉa.
