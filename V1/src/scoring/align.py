"""§4 자리 맞추기 — 전사·대본·소리를 한 자리에 (`docs/TRACK1_SCORING_SPEC.md` v2 §4).

규칙 (§4-1 ~ §4-3)
  · 「같은 음절」 = 초성이 같음. 디/데 · 겄/겠 · 사/새 는 같은 음절, 카/하 는 다른 음절.
  · 정렬 비용: 같은 음절 0 · 치환 0.5 · 삽입/삭제 1. 최소 비용 경로.
  · 자리 판정은 위치 — 짝지어진 음절은 글자가 달라도 그 대본 자리에 붙는다 (사람 결정 ②).
  · 문장 게이트: 같은 음절이 대본 음절의 GATE 미만이면 다른 문장 (§4-2).
  · 어절 판정: 어절 음절의 절반 이상이 같은 음절이면 「있음」, 아니면 어절 빈칸 (§4-3).
  · 띄어쓰기는 어디에도 안 들어간다. 어절 경계는 대본에서.

소리 쪽 (`mfa_syllable_spans`)
  MFA words 티어의 각 항목(형태소)은 라벨이 한글이고 이어 붙이면 전사(공백 없음)가 된다
  (`pipeline.features.group_eojeol` 의 전제, 2026-08-28 파일럿 200발화에서 lab.txt == 전사 확인).
  항목 라벨의 음절 수와 `syllables_of` 가 낸 음절 수가 다르면 그 항목은 전부 `spn`
  (실측 61/61) — 그 음절들은 구간 없음(None) 으로 둔다.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.features import syllables_of  # noqa: E402

GATE = 0.60            # §4-2 【미검증 임시값】 P2
WORD_FRAC = 0.5        # §4-3 어절 「있음」 — 같은 음절 절반 이상 P9
MAX_BLANK_WORD = 0.20  # §4-3 어절 빈칸 초과 시 발화 제외 P9
SUB_COST = 0.5

_HANGUL = re.compile(r"[가-힣]")


def syllables(s: str) -> str:
    """한글 음절만 남긴다 (공백·구두점·숫자 제거)."""
    return "".join(_HANGUL.findall(s))


def cho(c: str) -> int:
    return (ord(c) - 0xAC00) // 588


def same(a: str, b: str) -> bool:
    return cho(a) == cho(b)


def script_words(script: str) -> list[str]:
    """대본을 어절로. 구두점은 뗀다."""
    return [w for w in (syllables(x) for x in script.split()) if w]


def align_syllables(S: str, T: str) -> tuple[list[int | None], list[bool]]:
    """대본 음절열 S 와 전사 음절열 T 를 편집 정렬.

    반환 `(pair, exact)` — 대본 자리 t 마다 짝지어진 전사 음절 index(없으면 None),
    그리고 그 짝이 「같은 음절」인가.
    """
    n, m = len(S), len(T)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0:
                dp[i][j] = float(j)
            elif j == 0:
                dp[i][j] = float(i)
            else:
                c = 0.0 if same(S[i - 1], T[j - 1]) else SUB_COST
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c)
    pair: list[int | None] = [None] * n
    exact = [False] * n
    i, j = n, m
    while i > 0 and j > 0:
        c = 0.0 if same(S[i - 1], T[j - 1]) else SUB_COST
        if abs(dp[i][j] - (dp[i - 1][j - 1] + c)) < 1e-9:
            pair[i - 1] = j - 1
            exact[i - 1] = c == 0.0
            i, j = i - 1, j - 1
        elif abs(dp[i][j] - (dp[i - 1][j] + 1)) < 1e-9:
            i -= 1
        else:
            j -= 1
    return pair, exact


@dataclass
class Alignment:
    script_key: str = ""
    words: list[str] = field(default_factory=list)          # 대본 어절
    bounds: list[tuple[int, int]] = field(default_factory=list)  # 어절 j 의 음절 자리 [a, e)
    pair: list[int | None] = field(default_factory=list)    # 자리 t → 전사 음절 index
    exact: list[bool] = field(default_factory=list)
    present: list[bool] = field(default_factory=list)       # 어절 j 「있음」 (§4-3)
    gate_ratio: float = 0.0
    passed: bool = False                                    # 게이트 통과
    blank_word_ratio: float = 1.0
    usable: bool = False                                    # 통과 + 어절 빈칸 ≤ MAX

    @property
    def n_syl(self) -> int:
        return len(self.pair)


def align_transcript(script: str, transcript: str, gate: float = GATE,
                     word_frac: float = WORD_FRAC, max_blank: float = MAX_BLANK_WORD) -> Alignment:
    W = script_words(script)
    S = "".join(W)
    T = syllables(transcript)
    pair, exact = align_syllables(S, T)
    bounds, p = [], 0
    for w in W:
        bounds.append((p, p + len(w)))
        p += len(w)
    present = [sum(exact[a:e]) >= word_frac * (e - a) for a, e in bounds]
    A = Alignment(words=W, bounds=bounds, pair=pair, exact=exact, present=present)
    A.gate_ratio = (sum(exact) / len(S)) if S else 0.0
    A.passed = A.gate_ratio >= gate
    A.blank_word_ratio = 1 - (sum(present) / len(W) if W else 0.0)
    A.usable = A.passed and A.blank_word_ratio <= max_blank
    return A


def mfa_syllable_spans(words: list, phones: list, transcript: str) -> list[tuple[float, float] | None] | None:
    """MFA (words, phones) → 전사 음절마다 [s, e) 구간. `spn` 항목의 음절은 None.

    words 라벨을 이어 붙인 것이 전사 음절열과 다르면 None 을 돌려준다 (정렬 입력 불일치).
    """
    T = syllables(transcript)
    out: list[tuple[float, float] | None] = []
    for s, e, label in words:
        k = len(syllables(label))
        if k == 0:
            continue
        spans = syllables_of(phones, s, e)
        if len(spans) == k:
            out.extend(spans)
        else:                       # spn — 음절 구조 없음
            out.extend([None] * k)
    if len(out) != len(T):
        return None
    return out


def position_spans(A: Alignment, syl_spans: list[tuple[float, float] | None]) -> list[tuple[float, float] | None]:
    """대본 자리 t → 소리 구간. 짝이 없거나(삭제) 구간이 없으면(spn) None."""
    return [syl_spans[j] if (j is not None and j < len(syl_spans)) else None for j in A.pair]


def word_spans(A: Alignment, pos_spans: list) -> list[tuple[float, float] | None]:
    """대본 어절 j → [첫 자리 시작, 마지막 자리 끝). 어절 빈칸이거나 구간이 하나도 없으면 None."""
    out = []
    for (a, e), ok in zip(A.bounds, A.present):
        sp = [x for x in pos_spans[a:e] if x is not None]
        out.append((min(x[0] for x in sp), max(x[1] for x in sp)) if (ok and sp) else None)
    return out


# ─── selftest — §4-3 표와 §4-4 반례 3건 ────────────────────────────────────

def _selftest() -> int:
    import csv
    import json
    import statistics

    repo = Path(__file__).resolve().parents[2]
    rows = list(csv.DictReader(open(repo / "registry/pipeline/pilot_d4_prep_2026-08-23.csv", encoding="utf-8")))
    sent = json.load(open(repo / "registry/pipeline/pilot_d4_sentences_2026-08-23.json", encoding="utf-8"))
    scripts = {c["script_key"]: c["대본"] for v in sent["유형별_후보"].values() for c in v["상위 5"]}

    # §4-3 표 (2026-08-28 실측): 게이트 탈락 · 어절 판정 빈칸 0 / ≤10% / ≤20% / 평균%
    expect = {"1|1": (10, 200, 213, 226, 2.6), "1|38": (0, 155, 189, 198, 2.8), "1|5": (6, 193, 207, 210, 1.0)}
    fails = 0
    for key, exp in expect.items():
        rs = [r for r in rows if r["script_key"] == key]
        drop, blank = 0, []
        for r in rs:
            A = align_transcript(scripts[key], r["transcript"])
            if not A.passed:
                drop += 1
                continue
            blank.append(A.blank_word_ratio)
        got = (drop, sum(b == 0 for b in blank), sum(b <= .1 + 1e-9 for b in blank),
               sum(b <= .2 + 1e-9 for b in blank), round(100 * statistics.mean(blank), 1))
        ok = got == exp
        fails += not ok
        print(f"[{'ok' if ok else 'FAIL'}] {key} 기대 {exp} 실측 {got}")

    # §4-4 반례 3건 (CSV 행 번호 = 헤더 포함)
    def show(line: int) -> Alignment:
        r = rows[line - 2]
        A = align_transcript(scripts[r["script_key"]], r["transcript"])
        A.script_key = r["script_key"]
        return A

    a = show(221)
    ok = a.passed and a.blank_word_ratio == 0
    fails += not ok
    print(f"[{'ok' if ok else 'FAIL'}] 221행 {a.script_key} 통과 {a.passed} 어절 빈칸 {a.blank_word_ratio:.2f} (기대: 통과 · 0)")
    a = show(326)
    blank_words = [w for w, p in zip(a.words, a.present) if not p]
    ok = a.passed and blank_words == ["큰"]
    fails += not ok
    print(f"[{'ok' if ok else 'FAIL'}] 326행 {a.script_key} 통과 {a.passed} 빈칸 어절 {blank_words} (기대: 통과 · ['큰'])")
    a = show(558)
    blank_words = [w for w, p in zip(a.words, a.present) if not p]
    ok = a.passed and blank_words == ["안", "델낀디"]     # 「추우마→추워진」은 추·우 2/3 로 있음
    fails += not ok
    print(f"[{'ok' if ok else 'FAIL'}] 558행 {a.script_key} 통과 {a.passed} ({a.gate_ratio:.0%}) 빈칸 어절 {blank_words} (기대: 통과 · ['안', '델낀디'])")
    print("selftest", "통과" if fails == 0 else f"실패 {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
