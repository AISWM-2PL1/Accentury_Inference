#!/usr/bin/env python3
"""manner 범주 필터 — 논문 §4 / 저자 `analysis/data_prep.R` 를 옮긴 것

논문 §4:
  "각 경계마다 앞뒤의 참조·가설 음소를 일반 manner 범주(vowel, stop,
   approximant, nasal, fricative, sibilant/affricate)로 묶고, silence 와
   unknown 범주를 더했다. **manner 범주가 다른 경계는 평가에서 제외했다.**"

저자는 MFA 채점기가 아니라 **R 스크립트에서 사후로** 걸렀다.
`analysis/data_prep.R:253` 이 실제 제외 지점이다.

    filtered_boundary_data = boundary_data %>% subset(
        previous_reference_category == previous_test_category &
        following_reference_category == following_test_category &
        following_test_category != 'unknown' & following_reference_category != 'unknown' &
        previous_test_category  != 'unknown' & previous_reference_category  != 'unknown')

즉 **앞뒤 네 범주가 모두 짝이 맞고, 하나도 unknown 이 아닌 경계만** 남긴다.

음소 목록은 `data_prep.R` 의 `reference_phone_lists` · `test_phone_lists` 를
그대로 옮겼다. 여기서는 Seoul Corpus(참조)와 MFA(가설)만 쓴다.

사용
    python3 src/align_manner_filter.py work/align/paper_full/eval
"""
import argparse
import csv
import json
import statistics as st
from pathlib import Path

# ── 참조 쪽 — Seoul Corpus (data_prep.R: reference_phone_lists$seoul_corpus)
REF = {
    "vowel": "aa ee ii oo uu vv wa we wi wv xi xx ya ye yo yu yv",
    "stop": "k0 kh kk p0 ph pp t0 th tt",
    "approximant": "ll, ll ll",
    "nasal": "mm, mm mm, ng, nn, nn nn",
    "fricative": "hh",
    "sibilant": "s0 ss c0 cc ch",
    "silence": "sil",
}

# ── 가설 쪽 — MFA (data_prep.R: test_phone_lists$mfa)
TEST = {
    "vowel": ("a ɐ ɑ ɒ aː ɑː ɒː æ aj aw ɔj e ə ɚ eː ej ɛ ɝ ɛː, ɥ i, ɥ iː, i ɪ ɨ i̥ ɨ̥ iː ɨː,"
              "j ɐ, j e, j eː, j ɛː, j o, j oː, j u, j uː, j ʌ, j ʌː, ɯ ɯ̥, ɰ i, ɰ iː, ɯː "
              "o oː ow u ʉ ʊ uː ʉː ʌ ʌː, w ɐ, w e, w eː, w ʌ, w ʌː"),
    "stop": ("pʰː b bʲ c c͈ cː cʰ cʷ d d̪ dː dʲ ɡ ɡː ɡʷ ɟ ɟʷ k k̚ k͈ kː k͈ː kʰ kʷ k͈ʷ "
             "p p̚ p͈ pː pʰ pʲ p͈ʲ pʲː pʷ t t̚ t̪ t͈ tː tʰ tʲ tʲː tʷ ʔ t͈ʲ"),
    "approximant": "ɥ j l ɫ ɭ ɫ̩ ɭː ɰ ɹ ɾ ɾʲ w ʎ ʎː",
    "nasal": "m m̩ ɰ̃ mː mʲ mʲː n ɴ ɲ n̩ nː ɴː ɲː ŋ ɾ̃",
    "fricative": "ç ð f fʲ ɣ h ɦ ʝ ɸ ɸː ɸʲ ɸʷ v vʲ x β βʷ θ",
    "sibilant": ("s͈ː, t s, d z, tɕ͈ː ɕ ɕ͈ ɕː s s͈ ʃ sː sʰ sʷ z ʑ ʒ ɕʰ dz dʑ dʑː dʒ "
                 "tɕ tɕ͈ tɕː tɕʰ tɕʷ tɕ͈ʷ ts tʃ tsː"),
    "silence": "sil",
}


def build(table):
    """{범주: '공백/쉼표로 나열'} → {음소: 범주}. 쉼표는 다음소 항목 구분."""
    out = {}
    for cat, s in table.items():
        for chunk in s.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if " " in chunk and chunk.count(" ") <= 2 and len(chunk.split()) <= 3 \
                    and any(len(x) <= 3 for x in chunk.split()):
                # 「j ɐ」 처럼 공백이 든 한 항목일 수 있다. 통째로도 넣고 낱개로도 넣는다.
                out.setdefault(chunk, cat)
            for p in chunk.split():
                out.setdefault(p, cat)
    return out


REF_CAT, TEST_CAT = build(REF), build(TEST)

# 저자 표에 빠진 장음 변형 — cʰ 는 있는데 cʰː 는 없다. 기본형의 범주를 물려준다.
for _t in (REF_CAT, TEST_CAT):
    for _p, _c in list(_t.items()):
        for _suf in ("ː", "ːː"):
            _t.setdefault(_p + _suf, _c)

# 한글 이중모음은 「활음 + 모음」이다.
#   정답(Seoul)  ya · we   → 한 글자
#   MFA(IPA)     j ɐ · w e → 두 음소
# 음절을 세려면 활음을 뒤 모음에 붙여 하나로 봐야 한다.
GLIDES = {"j", "w", "ɥ", "ɰ"}

# 🚨 build() 가 「j ɐ」 같은 다음소 항목을 낱개로도 등록하는 바람에
#    j · w 가 vowel 로 잡혔다. 활음은 approximant 가 맞다.
for _g in GLIDES:
    TEST_CAT[_g] = "approximant"


def syllable_starts(labels, ref_side):
    """음절 시작 위치 목록. 모음 하나 = 음절 하나.

    한글 이중모음은 「활음 + 모음」이라 MFA 는 `j ɐ` 로 둘로 쪼갠다.
    정답(Seoul)은 `ya` 한 글자다. 모음 바로 앞이 활음이면 그 활음을
    음절 시작으로 삼아 **둘을 하나로** 센다.
    """
    cat = REF_CAT if ref_side else TEST_CAT
    starts = []
    for i, l in enumerate(labels):
        if cat.get(l) != "vowel":
            continue
        starts.append(i - 1 if i > 0 and labels[i - 1] in GLIDES else i)
    return starts


# data_prep.R 의 Seoul Corpus 수동 예외 — 표기 차이일 뿐인 쌍
EXCEPT = {("d", "nn"), ("dʲ", "nn")}     # (가설, 참조)


def category(phone, ref_side):
    return (REF_CAT if ref_side else TEST_CAT).get(phone, "unknown")


def keep(row):
    """data_prep.R:253 — 앞뒤 네 범주가 모두 일치하고 unknown 이 없어야 남긴다."""
    for pos in ("previous", "following"):
        rp = row[f"{pos}_reference_phone"]
        tp = row[f"{pos}_test_phone"]
        rc, tc = category(rp, True), category(tp, False)
        if (tp, rp) in EXCEPT:
            tc = rc
        if rc == "unknown" or tc == "unknown" or rc != tc:
            return False
    return True


def summarize(errs):
    return (len(errs), st.mean(errs),
            [sum(1 for x in errs if x <= t) / len(errs) * 100 for t in (10, 25, 50)])


def main():
    ap = argparse.ArgumentParser(description="manner 필터 (논문 §4 · data_prep.R)")
    ap.add_argument("eval_dir", type=Path, help="compare_alignments 출력 폴더")
    ap.add_argument("--json", type=Path, default=None,
                    help="요약을 이 경로에 쓴다 (registry 커밋용)")
    ap.add_argument("--stamp", default="", help="json 에 박을 날짜 (YYYY-MM-DD)")
    a = ap.parse_args()

    f = a.eval_dir / "alignment_reference_evaluation_boundaries.csv"
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    allv = [abs(float(r["boundary_error"])) * 1000 for r in rows]
    kept = [abs(float(r["boundary_error"])) * 1000 for r in rows if keep(r)]

    print(f"{'':16s} {'경계':>10s} {'평균':>7s} {'t≤10':>7s} {'t≤25':>7s} {'t≤50':>7s}")
    for name, e in (("필터 없음", allv), ("manner 필터", kept)):
        n, m, t = summarize(e)
        print(f"{name:16s} {n:10,} {m:7.2f} {t[0]:7.2f} {t[1]:7.2f} {t[2]:7.2f}")
    print(f"{'논문':16s} {'':10s} {14.78:7.2f} {61.11:7.2f} {81.53:7.2f} {95.70:7.2f}")
    print(f"\n제외 {len(allv)-len(kept):,}건 ({(len(allv)-len(kept))/len(allv)*100:.2f}%)")

    if a.json:
        def block(e):
            n, m, t = summarize(e)
            return {"boundaries": n, "mean_ms": round(m, 2),
                    "t10_pct": round(t[0], 2), "t25_pct": round(t[1], 2),
                    "t50_pct": round(t[2], 2)}
        out = {
            "stamp": a.stamp,
            "script": "src/align/align_manner_filter.py",
            "eval_dir": str(a.eval_dir),
            "unit": "phone",
            "reference": "Seoul Corpus TextGrid (사람이 미세조정)",
            "aligner": "MFA Korean 3.0",
            "paper": {"id": "arXiv 2606.18466", "venue": "Interspeech 2026",
                      "mean_ms": 14.78, "t10_pct": 61.11,
                      "t25_pct": 81.53, "t50_pct": 95.70},
            "ours": {"no_filter": block(allv), "manner_filter": block(kept)},
            "excluded": {"n": len(allv) - len(kept),
                         "pct": round((len(allv) - len(kept)) / len(allv) * 100, 2)},
            "note": "manner 필터는 논문 §4 / 저자 data_prep.R:253 을 옮긴 것이다. "
                    "논문과 대조하는 값은 manner_filter 쪽이다. "
                    "이 값은 음소 단위이며 음절 단위(35.69ms)와 섞지 않는다.",
        }
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
        print(f"→ {a.json}")


if __name__ == "__main__":
    main()
