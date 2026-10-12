# -*- coding: utf-8 -*-
"""Test logic thuan — khong can torch / API / hien vat.

Kiem dung may cho ma bug se IM LANG neu sai:
  adq / cov          dinh nghia o §3.2
  minimal_covers     tang 0 cua §3.6 — phai TOI TIEU va dung co
  should_exit        cua thoat phai THAN TRONG
  cost_tokens        overhead + tong len(k), khong phai 250*|S|
  cache_key          bam NOI DUNG, khong bam ten
  canonical          thu tu chuan de tien to chung dung duoc
"""
import sys

from s0_common import cache_key, cost_tokens
from s0g_nli import adq, minimal_covers, should_exit

fails = []


def check(name, got, want):
    ok = got == want
    print("  %s %-46s got=%s want=%s" % ("ok " if ok else "FAIL", name, got, want))
    if not ok:
        fails.append(name)


def table(covered_by, J=None, thr=0.5, maxent=None):
    J = J if J is not None else len(covered_by)
    docs = sorted({d for ks in covered_by for d in ks})
    grid = {d: [1.0 if d in covered_by[j] else 0.0 for j in range(J)] for d in docs}
    return {"claims": ["c%d" % j for j in range(J)], "J": J, "threshold": thr,
            "grid": grid, "covered_by": covered_by,
            "max_entail_per_claim": maxent or
            [max([grid[d][j] for d in docs], default=0.0) for j in range(J)]}


print("\n== adq / cov  (§3.2) ==")
tb = table([["d3", "d7"], ["d12", "d23", "d31"]])
check("adq({d3,d12}) phu het", adq(tb, ["d3", "d12"])[0], True)
check("adq({d3,d7}) trung y van thieu manh 2", adq(tb, ["d3", "d7"])[0], False)
check("cov({d3,d7}) = {0}", sorted(adq(tb, ["d3", "d7"])[1]), [0])
check("adq({d19}) doan bay khong phu gi", adq(tb, ["d19"])[0], False)
check("cov({d19}) rong", sorted(adq(tb, ["d19"])[1]), [])
check("adq({d3,d12,d19}) van phu het (d19 thua)",
      adq(tb, ["d3", "d12", "d19"])[0], True)
check("adq(rong) khong phu", adq(tb, [])[0], False)

print("\n== minimal_covers  (tang 0 cua §3.6) ==")
cv = minimal_covers(tb)
check("so nghiem phu = 2x3", len(cv), 6)
check("moi nghiem co co = J = 2", sorted({len(c) for c in cv}), [2])
check("moi nghiem deu phu het", all(adq(tb, c)[0] for c in cv), True)
check("{d3,d12} nam trong nghiem", any(set(c)=={"d3","d12"} for c in cv), True)

# mot doan phu CA HAI manh -> nghiem co 1, va phai TOI TIEU (khong tra co 2)
tb1 = table([["s1", "d3"], ["s1", "d12"]])
cv1 = minimal_covers(tb1)
check("summary phu ca hai -> co nghiem co 1", min(len(c) for c in cv1), 1)
check("['s1'] la nghiem", any(set(c)=={"s1"} for c in cv1), True)
check("KHONG tra nghiem co 2 khi da co co 1",
      sorted({len(c) for c in cv1}), [1])

# 3 manh, can 3 doan
tb3 = table([["a"], ["b"], ["c"]])
check("3 manh roi rac -> dung 1 nghiem co 3",
      [sorted(c) for c in minimal_covers(tb3)], [["a", "b", "c"]])

# mot manh khong ai phu -> KHONG co nghiem
tb0 = table([["a"], []])
check("manh khong ai phu -> khong nghiem", minimal_covers(tb0), [])

print("\n== should_exit  (cua thoat phai THAN TRONG) ==")
cfg_strict = {"nli": {"exit_max_entail": 0.20}}
check("phu day -> khong thoat", should_exit(tb, cfg_strict)[0], False)
tb_low = table([["a"], ["b"]], maxent=[0.95, 0.08])
check("manh 2 moi doan deu 0.08 -> THOAT", should_exit(tb_low, cfg_strict)[0], True)
check("chi ra dung manh thieu", should_exit(tb_low, cfg_strict)[1], [1])
tb_mid = table([["a"], ["b"]], maxent=[0.95, 0.35])
check("manh 2 co doan 0.35 (duoi nguong 0.5 nhung tren 0.20) -> KHONG thoat",
      should_exit(tb_mid, cfg_strict)[0], False)

print("\n== cost_tokens  (overhead + tong len(k)) ==")
check("cost(rong) = overhead", cost_tokens([], 500), 500)
check("cost 3 doan 100 token", cost_tokens([100, 100, 100], 500), 800)
check("document + summary la hai lop do dai",
      cost_tokens([100, 600], 500), 1200)
check("KHONG phai 250*|S|", cost_tokens([100, 600], 500) == 500 + 250 * 2, False)

print("\n== cache_key  (bam NOI DUNG, khong bam ten) ==")
k_ab = cache_key("m", "t", ["alpha", "beta"])
k_ba = cache_key("m", "t", ["beta", "alpha"])
check("thu tu khong doi khoa", k_ab, k_ba)
check("noi dung khac -> khoa khac", cache_key("m", "t", ["alpha", "gamma"]) != k_ab, True)
check("model khac -> khoa khac", cache_key("m2", "t", ["alpha", "beta"]) != k_ab, True)
check("template khac -> khoa khac", cache_key("m", "t2", ["alpha", "beta"]) != k_ab, True)

print("\n== canonical  (tien to chung) ==")
from s0e_judge import Judge
check("canonical sap xep on dinh", Judge.canonical(["c7", "c1", "c3"]),
      ["c1", "c3", "c7"])
check("hai tap chung tien to -> chung 2 phan tu dau",
      Judge.canonical(["c1", "c3", "c7"])[:2] == Judge.canonical(["c1", "c3", "c9"])[:2],
      True)

print("\n== declarativize  (manh phai la CAU KHANG DINH) ==")
# Vi sao can test nay: model NLI duoc huan luyen tren cau khang dinh.  Ban
# dau from_annotation tra ve "Who directed X? -> Y" va dua thang vao NLI lam
# hypothesis — ngoai phan phoi, bang phu thanh rac, khong mot canh bao nao.
# test_logic cu khong bat duoc vi no test coverage_table bang claim viet tay.
from s0f_decompose import declarativize, from_annotation
_row = {"question_decomposition": [
    {"question": "Which film won Best Picture in 1994?", "answer": "Forrest Gump"},
    {"question": "Who directed #1 ?", "answer": "Robert Zemeckis"}]}
_st: dict = {}
_claims = from_annotation(_row, _st)
check("sinh dung J manh", len(_claims), 2)
check("khong con dau '->' trong manh",
      any("->" in c for c in _claims), False)
check("khong con dau '?' cuoi manh",
      any(c.rstrip().endswith("?") for c in _claims), False)
check("manh nao cung ket thuc bang dau cham",
      all(c.rstrip().endswith(".") for c in _claims), True)
check("giai duoc tham chieu #1 ve dap an con truoc",
      "Forrest Gump" in _claims[1], True)
check("dem duoc mau / khuon", sorted(_st), ["fallback", "pattern"])
check("mau 'who <verb>ed X' ra menh de",
      declarativize("Who directed Forrest Gump?", "Zemeckis")[0],
      "Zemeckis directed Forrest Gump.")
check("mau 'the <rel> of X' ra menh de",
      declarativize("What is the capital of France?", "Paris")[0],
      "Paris is the capital of France.")
check("khuon bao phu van la cau hoan chinh",
      declarativize("How many ducks?", "7")[0].endswith("."), True)

print("\n== suc khoe manh  (mo coi vs rong) ==")
# Hai kieu manh hong, HONG NGUOC NHAU.  Mo coi lam truy van thoat oan; rong
# lam adq thanh luon-dung nen bo loc ngung lam viec — kieu thu hai am tham hon.


class _FakeNLI:
    def entail_probs(self, prem, hyp):
        return [0.9 if ("Gump" in h and "Gump" in p) else 0.05
                for p, h in zip(prem, hyp)]


from s0g_nli import coverage_table as _cov
_cfg = {"nli": {"entail_threshold": 0.5, "vacuous_fraction": 0.5}}
_tb = _cov(_FakeNLI(), ["d1", "d2", "d3"],
                     ["Forrest Gump won.", "Zemeckis directed Forrest Gump.",
                      "Unrelated text."],
                     ["Forrest Gump won Best Picture.", "Nothing matches this."],
                     _cfg)
check("dem duoc so doan phu moi manh", _tb["coverage_count"], [2, 0])
check("bat manh MO COI", _tb["orphan_claims"], [1])
check("bat manh RONG (qua nua pool phu)", _tb["vacuous_claims"], [0])

print("\n== neo manh vao doan ==")
from s0f_decompose import passages_of
check("doc duoc doan kieu MuSiQue",
      len(passages_of({"paragraphs": [
          {"title": "A", "paragraph_text": "x"},
          {"title": "B", "paragraph_text": "y"}]})), 2)
check("doc duoc doan kieu HotpotQA",
      len(passages_of({"context": [["A", ["x", "y"]], ["B", ["z"]]]})), 2)
check("khong co doan -> rong, khong no", passages_of({}), [])

print("\n" + "=" * 62)
if fails:
    print("  %d TEST FAIL: %s" % (len(fails), fails))
    sys.exit(1)
print("  tat ca test logic PASS")
print("=" * 62)
