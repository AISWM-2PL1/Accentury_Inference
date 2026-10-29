#!/usr/bin/env python3
"""S3 특징 추출기 — 트랙 1 판 A·B 공용 한 벌

명세: `docs/POC_PIPELINE_SPEC.md` §3-3 (특징 목록) · §5-1-1 (판 A·B) · §4-1 (정규화·금지)

**참조 분포를 만들 때와 사용자를 잴 때 같은 코드를 탄다.** S2(`f0_extract.py`)와 같은 이유다.

**판 A·B 는 코드가 한 벌이고 `use_syllable` 스위치 하나로 갈린다** (§5-1-1).
두 벌로 짜면 A·B 성적차가 음절 때문인지 코드 차이 때문인지 못 가른다.

입력 셋
  · WAV 샘플 (mono float) — S2 가 F0·semitone·상대 RMS 로 바꾼다
  · MFA TextGrid — `words`·`phones` 티어. 음절 경계는 `phones` 에서 만든다
    (`align_manner_filter.syllable_starts`, 「모음 하나 = 음절 하나」)
  · 대본 문자열 — **어절 복원에 필요하다. 아래 경고를 읽는다.**

🚨 **MFA `words` 티어는 어절이 아니라 형태소다.** 한국어 토크나이저(`python-mecab-ko`)가
어절을 쪼개기 때문이다 (명세 §5-1-4 가 사전 매칭 쪽으로만 적어 둔 사실의 다른 얼굴).
    원문   초봄이나 안 그라모 농사짓기 전 겨울에 약초를 캐러 가신다꼬예   (어절 9)
    words  초봄 이나 안 그라모 농사 짓 기 전 겨울 에 약초 를 캐 러 가 신다 꼬 예  (18)
그대로 쓰면 블록 1 이 「어절 성조」가 아니라 「형태소 성조」가 된다. 명세 §3-3 의 단위와
다르다. 그래서 **대본의 공백으로 형태소를 되묶어 어절을 복원한다**(`group_eojeol`).
복원에 실패한 발화는 `eojeol_grouped=False` 로 표시하고 형태소 단위로 떨어진다 —
**조용히 넘어가지 않는다.**

여기서 내지 않는 것 셋 — 발화 하나만 봐서는 못 내기 때문이다
  · `dur_resid` (§3-3 블록 3) — 말 속도·음절 수·ip 위치에 대한 **로그 잔차**다.
    회귀는 코퍼스 전체에서 적합해야 하므로 **S4 에서 낸다.** 여기서는 그 회귀의
    입력(`word_dur`·`n_syl`·`ip_pos`·`rate_wps`)만 함께 내보낸다.
  · `syl_dur_resid` (§3-3 블록 1 · D-11) — 같은 이유다. `syl_dur`(절대 초)는 여기서 내고
    그 로그 잔차는 회귀가 필요하므로 **S4 에서 낸다.** 파일럿은 `pilot_run.py` 가 국소
    회귀로 **잠정값**을 낸다 (`dur_resid_local` 과 같은 취급).
  · `warp_angle`·`interval_distortion` — 참조 medoid 와 맞대야 나온다. **S5 다.**
  · 블록 합산 — V2 로 세 블록을 분리 보고한다. 총점을 만들지 않는다.

사용
  uv run --no-project --with praat-parselmouth --with numpy --with praatio \\
      python src/pipeline/features.py --selftest
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prosody.f0_extract import extract as f0_extract, F0Result, TIME_STEP  # noqa: E402

# ─── 잠정값 ────────────────────────────────────────────────────────────────
# 🚨 ip 경계 규칙은 정해진 바 없다 (명세 §3-3 블록 2). v4 §7 이 ip 경계 골드
# 300~500발화를 요구했으나 D-3(사람 검수 안 함)으로 그 자가 없다.
# 아래는 **근거 없는 잠정값**이고 파일럿에서 민감도를 본다. 못 박지 않는다.
IP_PAUSE_SEC = 0.20

SPREAD_DROP_ST = 1.0      # peak 에서 이만큼 아래까지를 「고음 구간」으로 본다 (조작화)
ONSET_HEAD_FRAC = 0.20    # 어절·음절 앞 이 비율을 onset 으로 본다 (조작화)
SPLINE_DEG = 3            # spline_c1..c3 — 상수항은 register 와 겹치므로 뺀다 (§4-1 이중 채점 금지)

MIN_VOICED_FRAMES_WORD = 3   # 이보다 적으면 그 어절의 F0 특징은 결측


# ─── 자료 구조 ─────────────────────────────────────────────────────────────

@dataclass
class SylFeat:
    """판 B 에서만 채워진다 (§3-3 블록 1)."""
    idx: int = -1
    start: float = 0.0
    end: float = 0.0
    tone_level: float = float("nan")     # 음절 p 중앙값
    onset_level: float = float("nan")    # 음절 앞 20% p 중앙값
    syl_delta: float = float("nan")      # 앞 음절 tone_level 과의 차
    syl_dur_prop: float = float("nan")   # 어절 길이에서 이 음절이 차지하는 비율
    syl_dur: float = float("nan")        # 음절 절대 지속시간(초) — 상성(장음) 재료 (D-11)


@dataclass
class WordFeat:
    """블록 1 + 블록 3 의 어절 단위 (§3-3)."""
    idx: int = -1
    label: str = ""
    start: float = 0.0
    end: float = 0.0
    ip: int = -1
    missing: str | None = None           # 결측 사유. 차면 아래 F0 값은 nan

    # 블록 1 — 음 높이·성조
    onset: float = float("nan")
    register: float = float("nan")
    peak_pos: float = float("nan")       # 어절 길이 정규화 위치 0~1
    peak_height: float = float("nan")    # max p − register
    valley: float = float("nan")         # min p − register
    excursion: float = float("nan")      # max p − min p  (§7 안전장치가 명시)
    post_peak_fall: float = float("nan") # peak p − 어절 마지막 p  (§7 안전장치가 명시)
    spread: float = float("nan")
    plateau: float = float("nan")
    spline_c: list[float] = field(default_factory=list)   # c1..c3 (§4-1 채점 표현)
    log_rms_rel: float = float("nan")    # 기록만. 점수로 직결시키지 않는다 (§3-1)

    # 블록 3 — 리듬·길이. dur_resid 의 회귀 입력이다 (잔차는 S4)
    word_dur: float = 0.0
    n_syl: int = 0
    ip_pos: float = float("nan")         # ip 안 상대 위치 0~1
    pause_after: float = 0.0

    # 판 B 전용
    syllables: list[SylFeat] = field(default_factory=list)
    peak_syl_idx: int = -1               # 1-based. penult 는 n_syl-1 로 나온다


@dataclass
class IpFeat:
    """블록 2 (§3-3). ip = 억양구, 한 숨에 말하는 덩어리."""
    idx: int = -1
    start: float = 0.0
    end: float = 0.0
    n_words: int = 0
    declination: float = float("nan")    # ip 안 p 의 시간 회귀 기울기 (st/s)
    p_range: float = float("nan")        # ip 안 p 의 max − min
    downstep: float = float("nan")       # 인접 어절 register 차의 음 성분 합
    upstep: float = float("nan")         # 같은 것의 양 성분 합
    reset: float = float("nan")          # 다음 ip onset − 이 ip 마지막 p. 마지막 ip 는 nan
    r_uw: list[float] = field(default_factory=list)   # M_u − b_g(t_u) — 어절별 잔차
    d_r_uw: list[float] = field(default_factory=list) # ΔR_uw — 인접 어절 잔차 차


@dataclass
class UttFeat:
    utt_id: str = ""
    use_syllable: bool = False
    unscorable: str | None = None
    duration: float = 0.0
    eojeol_grouped: bool = False         # 🚨 False 면 아래 words 는 어절이 아니라 형태소다

    words: list[WordFeat] = field(default_factory=list)
    ips: list[IpFeat] = field(default_factory=list)

    # 발화 단위 (U1~U3) — 블록 3 의 회귀 공변량이기도 하다
    rate_wps: float = float("nan")       # 어절/초. 말 속도는 공변량으로 뺀다 (v4:132)
    pause_ratio: float = float("nan")
    ip_count: int = 0
    pause_pos: list[float] = field(default_factory=list)   # 정규화 시각

    # 발화 마지막 (블록 2)
    final_slope: float = float("nan")
    final_lengthening_raw: float = float("nan")            # 잔차는 S4

    # 재현용
    median_f0: float = 0.0
    pass2_floor: float = 0.0
    pass2_ceiling: float = 0.0
    ip_pause_sec: float = IP_PAUSE_SEC   # 잠정값을 산출물에 박아 둔다


# ─── TextGrid ──────────────────────────────────────────────────────────────

def _clean(s: str) -> bool:
    """`bench_disagree.clean` 과 같은 규칙 — 빈 라벨과 `<...>` 를 뺀다."""
    return bool(s.strip()) and "<" not in s


def read_textgrid(path: str | Path) -> tuple[list, list]:
    """MFA TextGrid → (words, phones). 각 원소는 (start, end, label)."""
    from praatio import textgrid

    tg = textgrid.openTextgrid(str(path), includeEmptyIntervals=False)
    words = [(e.start, e.end, e.label) for e in tg.getTier("words").entries if _clean(e.label)]
    phones = [(e.start, e.end, e.label) for e in tg.getTier("phones").entries if _clean(e.label)]
    return words, phones


def _norm(s: str) -> str:
    """구두점·공백을 떼고 글자만 남긴다. 대본과 MFA 라벨을 같은 자로 만든다."""
    return "".join(ch for ch in s if ch.isalnum())


def group_eojeol(words: list, script: str) -> tuple[list, bool]:
    """형태소 티어 → 어절. 대본의 공백을 자로 쓴다.

    MFA 는 순서를 지키므로 형태소 라벨을 이어 붙이면 대본의 공백 없는 표기가 된다.
    앞에서부터 대본의 어절을 하나씩 채워 나가며 자른다.

    반환 `(어절 목록, 복원 성공 여부)`. 실패하면 입력을 그대로 돌려준다 —
    형태소 단위로 떨어지되 그 사실이 산출물에 남는다.
    """
    # 대본에는 구두점이 있고 MFA 라벨에는 없다 — 양쪽을 같은 자로 깎아 맞춘다.
    # (실측: 「하거든요.」의 마침표 하나로 30발화 중 1건이 복원 실패했다, 2026-08-23)
    targets = [t for t in (_norm(x) for x in script.split()) if t]
    if not targets or not words:
        return words, False

    out, ti, buf, acc = [], 0, [], ""
    for w in words:
        if ti >= len(targets):
            return words, False
        buf.append(w)
        acc += _norm(w[2])
        tgt = targets[ti]
        if acc == tgt:
            out.append((buf[0][0], buf[-1][1], tgt))
            buf, acc, ti = [], "", ti + 1
        elif not tgt.startswith(acc):
            return words, False        # 어긋났다 — 되묶지 않는다
    if buf or ti != len(targets):
        return words, False
    return out, True


def syllables_of(phones: list, w_start: float, w_end: float,
                 ref_side: bool = False) -> list[tuple[float, float]]:
    """어절 구간 안 음절 경계. `align_manner_filter.syllable_starts` 규칙을 쓴다.

    `ref_side` 는 음소 기호 집합을 고른다. MFA 산출물은 False(IPA 계열),
    Seoul Corpus 정답 TextGrid 는 True(`th`·`aa` 계열) 다.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "align"))
    from align_manner_filter import syllable_starts  # noqa: E402

    inside = [p for p in phones if p[0] >= w_start - 1e-6 and p[1] <= w_end + 1e-6]
    if not inside:
        return []
    starts = syllable_starts([p[2] for p in inside], ref_side)
    if not starts:
        return []
    out = []
    for k, i in enumerate(starts):
        j = starts[k + 1] - 1 if k + 1 < len(starts) else len(inside) - 1
        out.append((inside[i][0], inside[j][1]))
    return out


def has_spn(phones: list, w_start: float, w_end: float) -> bool:
    """§5-1-1 간단 마스크 규칙 2 — `spn` 이 걸린 어절의 음절은 결측 처리한다."""
    return any(p[2] == "spn" for p in phones if p[0] >= w_start - 1e-6 and p[1] <= w_end + 1e-6)


# ─── 계산 도우미 ───────────────────────────────────────────────────────────

def _slice(r: F0Result, a: float, b: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[a, b) 구간의 (시각, semitone, 상대 RMS). 유성 프레임만."""
    if r.times.size == 0:
        e = np.array([])
        return e, e, e
    m = (r.times >= a) & (r.times < b) & r.voiced
    return r.times[m], r.semitone[m], r.rms_rel[m]


def _head(t: np.ndarray, v: np.ndarray, a: float, b: float, frac: float) -> float:
    """구간 앞 `frac` 의 중앙값. 프레임이 없으면 첫 프레임으로 떨어진다."""
    if v.size == 0:
        return float("nan")
    cut = a + (b - a) * frac
    m = t < cut
    return float(np.median(v[m])) if m.any() else float(v[0])


def _slope(t: np.ndarray, v: np.ndarray) -> float:
    """시간에 대한 1차 회귀 기울기 (st/s). 점이 둘 미만이면 nan."""
    if v.size < 2:
        return float("nan")
    return float(np.polyfit(t, v, 1)[0])


def _spline_c(t: np.ndarray, v: np.ndarray, a: float, b: float) -> list[float]:
    """정규화 시간 위 3차 다항 계수 c1..c3. 상수항은 register 와 겹치므로 뺀다."""
    if v.size < SPLINE_DEG + 1 or b <= a:
        return [float("nan")] * SPLINE_DEG
    x = (t - a) / (b - a)
    c = np.polyfit(x, v - float(np.median(v)), SPLINE_DEG)   # [c3, c2, c1, c0]
    return [float(c[2]), float(c[1]), float(c[0])]           # c1, c2, c3


# ─── 본체 ──────────────────────────────────────────────────────────────────

def segment_ip(words: list, pause_sec: float = IP_PAUSE_SEC) -> list[int]:
    """어절마다 ip 번호. 앞 어절과의 무음이 임계값 이상이면 새 ip."""
    ids, cur = [], 0
    for i, (s, e, _) in enumerate(words):
        if i > 0 and s - words[i - 1][1] >= pause_sec:
            cur += 1
        ids.append(cur)
    return ids


def word_features(r: F0Result, phones: list, w: tuple, idx: int,
                  use_syllable: bool, ref_side: bool = False) -> WordFeat:
    s, e, label = w
    f = WordFeat(idx=idx, label=label, start=s, end=e, word_dur=e - s)

    t, p, rms = _slice(r, s, e)
    syls = syllables_of(phones, s, e, ref_side)
    f.n_syl = len(syls)

    if rms.size:
        f.log_rms_rel = float(np.median(rms))

    if p.size < MIN_VOICED_FRAMES_WORD:
        f.missing = "유성 프레임 부족"
        return f

    f.register = float(np.median(p))
    f.onset = _head(t, p, s, e, ONSET_HEAD_FRAC)
    k = int(np.argmax(p))
    f.peak_pos = float((t[k] - s) / (e - s)) if e > s else float("nan")
    f.peak_height = float(p[k]) - f.register
    f.valley = float(np.min(p)) - f.register
    f.excursion = float(np.max(p) - np.min(p))
    f.post_peak_fall = float(p[k] - p[-1])

    hi = p >= (p[k] - SPREAD_DROP_ST)
    f.spread = float(hi.sum()) / p.size
    f.plateau = float(np.std(p[hi])) if hi.sum() > 1 else 0.0
    f.spline_c = _spline_c(t, p, s, e)

    if use_syllable:
        if has_spn(phones, s, e):
            f.missing = "spn 어절 — 음절 결측 (§5-1-1 마스크 2)"
            return f
        prev = float("nan")
        for j, (a, b) in enumerate(syls, start=1):
            st_, sp, _ = _slice(r, a, b)
            sf = SylFeat(idx=j, start=a, end=b, syl_dur=b - a,
                         syl_dur_prop=(b - a) / (e - s) if e > s else float("nan"))
            if sp.size:
                sf.tone_level = float(np.median(sp))
                sf.onset_level = _head(st_, sp, a, b, ONSET_HEAD_FRAC)
                sf.syl_delta = sf.tone_level - prev if not math.isnan(prev) else float("nan")
                prev = sf.tone_level
            f.syllables.append(sf)
        levels = [x.tone_level for x in f.syllables]
        if levels and not all(math.isnan(v) for v in levels):
            f.peak_syl_idx = int(np.nanargmax(levels)) + 1
    return f


def ip_features(r: F0Result, words: list[WordFeat], idx: int,
                next_onset: float) -> IpFeat:
    a, b = words[0].start, words[-1].end
    ip = IpFeat(idx=idx, start=a, end=b, n_words=len(words))

    t, p, _ = _slice(r, a, b)
    if p.size >= 2:
        ip.declination = _slope(t, p)
        ip.p_range = float(np.max(p) - np.min(p))
        # R_uw = M_u − b_g(t_u) — ip 안에서 declination 만 분리한 잔차 (§4-1 승계)
        c = np.polyfit(t, p, 1)
        for w in words:
            if math.isnan(w.register):
                ip.r_uw.append(float("nan"))
                continue
            tm = (w.start + w.end) / 2
            ip.r_uw.append(w.register - float(np.polyval(c, tm)))
        ip.d_r_uw = [ip.r_uw[i] - ip.r_uw[i - 1] for i in range(1, len(ip.r_uw))]
        if p.size and not math.isnan(next_onset):
            ip.reset = next_onset - float(p[-1])

    diffs = [words[i].register - words[i - 1].register for i in range(1, len(words))
             if not (math.isnan(words[i].register) or math.isnan(words[i - 1].register))]
    if diffs:
        ip.downstep = float(sum(d for d in diffs if d < 0))
        ip.upstep = float(sum(d for d in diffs if d > 0))
    return ip


def extract_features(x: np.ndarray, sr: int, textgrid_path: str | Path | None = None,
                     *, words_phones: tuple | None = None, script: str | None = None,
                     use_syllable: bool = False, ref_side: bool = False,
                     utt_id: str = "", ip_pause_sec: float = IP_PAUSE_SEC,
                     f0: F0Result | None = None) -> UttFeat:
    """WAV + TextGrid → 세 블록 특징 (§3-3).

    `words_phones` 를 직접 주면 TextGrid 파일 없이도 돈다 (selftest·단위 시험용).
    `script` 를 주면 형태소 티어를 어절로 되묶는다 — **주지 않으면 단위가 형태소다.**
    `f0` 는 같은 발화를 여러 설정으로 돌릴 때 S2 결과를 재사용하는 통로다
    (판 A·B · ip 임계값 훑기). **주지 않으면 여기서 뽑는다 — 계산 결과는 같다.**
    """
    out = UttFeat(utt_id=utt_id, use_syllable=use_syllable, ip_pause_sec=ip_pause_sec)

    r = f0 if f0 is not None else f0_extract(x, sr)
    out.duration = x.size / sr
    out.median_f0, out.pass2_floor, out.pass2_ceiling = r.median_f0, r.pass2_floor, r.pass2_ceiling
    if r.unscorable:
        out.unscorable = r.unscorable
        return out

    if words_phones is None:
        if textgrid_path is None:
            out.unscorable = "정렬 입력 없음"
            return out
        words_phones = read_textgrid(textgrid_path)
    words, phones = words_phones
    if not words:
        out.unscorable = "정렬 어절 0개"
        return out

    # 🚨 형태소 → 어절 복원. 대본이 없으면 형태소 단위로 떨어진다 (모듈 docstring 참고)
    if script:
        words, out.eojeol_grouped = group_eojeol(words, script)

    ip_of = segment_ip(words, ip_pause_sec)

    for i, w in enumerate(words):
        wf = word_features(r, phones, w, i, use_syllable, ref_side)
        wf.ip = ip_of[i]
        wf.pause_after = (words[i + 1][0] - w[1]) if i + 1 < len(words) else 0.0
        out.words.append(wf)

    # ip 안 상대 위치 — dur_resid 회귀 입력 (§3-3 블록 3)
    for g in set(ip_of):
        grp = [w for w in out.words if w.ip == g]
        for k, w in enumerate(grp):
            w.ip_pos = k / (len(grp) - 1) if len(grp) > 1 else 0.0

    groups = []
    for g in sorted(set(ip_of)):
        groups.append([w for w in out.words if w.ip == g])
    for gi, grp in enumerate(groups):
        nxt = groups[gi + 1][0].onset if gi + 1 < len(groups) else float("nan")
        out.ips.append(ip_features(r, grp, gi, nxt))
    out.ip_count = len(groups)

    speech = sum(w.end - w.start for w in out.words)
    pauses = [w.pause_after for w in out.words[:-1] if w.pause_after > 0]
    out.rate_wps = len(out.words) / out.duration if out.duration else float("nan")
    out.pause_ratio = (out.duration - speech) / out.duration if out.duration else float("nan")
    out.pause_pos = [round((w.end) / out.duration, 4) for w in out.words[:-1]
                     if w.pause_after > 0] if out.duration else []

    last = out.words[-1]
    lt, lp, _ = _slice(r, last.start, last.end)
    out.final_slope = _slope(lt, lp)
    out.final_lengthening_raw = last.word_dur      # 잔차는 S4 (회귀가 코퍼스 단위)
    return out


# ─── golden test ───────────────────────────────────────────────────────────

def _synth_word(f0_start: float, f0_peak: float, dur: float, sr: int,
                peak_at: float = 0.4) -> np.ndarray:
    """하모닉 5개 + 지정한 위치에서 정점을 찍는 F0 궤적. 사인 하나는 Praat 이 못 잡는다."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    x = t / dur
    # 삼각 궤적: 0 → peak_at 에서 최고 → 1
    up = np.clip(x / peak_at, 0, 1)
    down = np.clip((1 - x) / (1 - peak_at), 0, 1)
    shape = np.minimum(up, down)
    f0 = f0_start + (f0_peak - f0_start) * shape
    ph = 2 * np.pi * np.cumsum(f0) / sr
    sig = sum(np.sin(k * ph) / k for k in range(1, 6))
    return 0.3 * sig / np.max(np.abs(sig))


def _selftest() -> int:
    sr = 16000
    ok, fail = 0, 0

    def chk(name: str, cond: bool, note: str = "") -> None:
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  PASS  {name}")
        else:
            fail += 1
            print(f"  FAIL  {name}  {note}")

    # 어절 둘 + 그 사이 300ms 무음 → ip 가 둘로 갈려야 한다
    w1 = _synth_word(120, 200, 0.6, sr, peak_at=0.3)
    gap = np.zeros(int(0.30 * sr))
    w2 = _synth_word(110, 150, 0.5, sr, peak_at=0.7)
    x = np.concatenate([w1, gap, w2])
    d1, d2 = 0.6, 0.5
    t2 = d1 + 0.30
    words = [(0.0, d1, "가나"), (t2, t2 + d2, "다라")]
    # phones: 음절 둘씩. 모음 라벨은 align_manner_filter 가 vowel 로 아는 것으로 둔다
    phones = [(0.0, 0.15, "k"), (0.15, 0.30, "ɐ"), (0.30, 0.45, "n"), (0.45, d1, "ɐ"),
              (t2, t2 + 0.12, "t"), (t2 + 0.12, t2 + 0.25, "ɐ"),
              (t2 + 0.25, t2 + 0.37, "ɾ"), (t2 + 0.37, t2 + d2, "ɐ")]

    a = extract_features(x, sr, words_phones=(words, phones), use_syllable=False, utt_id="A")
    b = extract_features(x, sr, words_phones=(words, phones), use_syllable=True, utt_id="B")

    chk("판 A 가 정상 산출", a.unscorable is None, str(a.unscorable))
    chk("어절 2개", len(a.words) == 2, str(len(a.words)))
    chk("ip 2개 (300ms 무음 > 200ms 임계)", a.ip_count == 2, str(a.ip_count))
    chk("판 A 는 음절을 내지 않는다", all(not w.syllables for w in a.words))
    chk("판 B 는 음절을 낸다", all(len(w.syllables) == 2 for w in b.words),
        str([len(w.syllables) for w in b.words]))

    # 판 A·B 는 같은 코드다 — 어절 특징이 비트 단위로 같아야 한다
    same = all(
        (math.isnan(getattr(wa, k)) and math.isnan(getattr(wb, k))) or
        getattr(wa, k) == getattr(wb, k)
        for wa, wb in zip(a.words, b.words)
        for k in ("onset", "register", "peak_pos", "peak_height", "excursion",
                  "post_peak_fall", "spread", "plateau", "log_rms_rel")
    )
    chk("판 A·B 의 어절 특징이 동일 (스위치가 어절층을 안 건드린다)", same)

    w0 = a.words[0]
    chk("peak_pos 가 앞쪽 (합성 정점 0.3)", 0.10 < w0.peak_pos < 0.55, f"{w0.peak_pos:.3f}")
    chk("peak_height > 0", w0.peak_height > 0, f"{w0.peak_height:.3f}")
    chk("excursion ≥ peak_height", w0.excursion >= w0.peak_height - 1e-9)
    chk("spline_c 3개", len(w0.spline_c) == 3)
    chk("pause_after 가 무음과 맞는다", abs(w0.pause_after - 0.30) < 1e-6, f"{w0.pause_after}")

    w1f = a.words[1]
    chk("둘째 어절 peak_pos 가 뒤쪽 (합성 정점 0.7)", w1f.peak_pos > w0.peak_pos,
        f"{w0.peak_pos:.3f} vs {w1f.peak_pos:.3f}")

    chk("ip[0].reset 이 채워진다", not math.isnan(a.ips[0].reset), str(a.ips[0].reset))
    chk("마지막 ip 의 reset 은 nan", math.isnan(a.ips[-1].reset))
    chk("발화 단위 rate_wps > 0", a.rate_wps > 0, f"{a.rate_wps:.3f}")

    # syl_dur — 절대 초. 비율(syl_dur_prop)만으로는 장음을 못 본다 (D-11)
    bw = b.words[0]
    chk("syl_dur 이 채워진다", all(s.syl_dur > 0 for s in bw.syllables),
        str([round(s.syl_dur, 4) for s in bw.syllables]))
    # 🚨 음절 구간은 **첫 모음부터** 시작한다 (`syllable_starts` 규칙 — 「모음 하나 = 음절
    # 하나」). 어절 첫 자음은 어느 음절에도 안 들어간다. 그래서 합 < 어절 길이다.
    # `syl_dur_prop` 도 처음부터 같은 성질이었다 — `syl_dur` 이 그것을 물려받는다.
    chk("syl_dur 합 ≤ 어절 길이 (어절 첫 자음은 빠진다)",
        sum(s.syl_dur for s in bw.syllables) <= (bw.end - bw.start) + 1e-9,
        f"{sum(s.syl_dur for s in bw.syllables):.4f} / {bw.end - bw.start:.4f}")
    chk("어절 첫 자음이 빠져 합 < 어절 길이",
        sum(s.syl_dur for s in bw.syllables) < (bw.end - bw.start) - 1e-9)
    chk("syl_dur / 어절길이 = syl_dur_prop",
        all(abs(s.syl_dur / (bw.end - bw.start) - s.syl_dur_prop) < 1e-9
            for s in bw.syllables))

    # 임계값을 올리면 ip 가 하나로 합쳐져야 한다 (잠정값이 파라미터임을 확인)
    c = extract_features(x, sr, words_phones=(words, phones), ip_pause_sec=0.50)
    chk("ip 임계값이 파라미터로 먹는다 (0.5s → ip 1개)", c.ip_count == 1, str(c.ip_count))

    # spn 어절은 음절이 결측 (§5-1-1 마스크 2)
    ph_spn = [(0.0, d1, "spn")] + phones[4:]
    d = extract_features(x, sr, words_phones=(words, ph_spn), use_syllable=True)
    chk("spn 어절은 음절 결측", d.words[0].missing is not None, str(d.words[0].missing))

    print(f"\n{ok}/{ok + fail} 통과")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="S3 특징 추출기 (명세 §3-3)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--wav", type=Path)
    ap.add_argument("--textgrid", type=Path)
    ap.add_argument("--script", type=Path, help="대본 .txt — 어절 복원에 쓴다")
    ap.add_argument("--syllable", action="store_true", help="판 B")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(_selftest())

    if not (a.wav and a.textgrid):
        ap.error("--wav 와 --textgrid 가 필요하다 (또는 --selftest)")

    import soundfile as sf

    x, sr = sf.read(str(a.wav))
    r = extract_features(np.asarray(x, dtype=float), sr, a.textgrid,
                         script=a.script.read_text(encoding="utf-8").strip() if a.script else None,
                         use_syllable=a.syllable, utt_id=a.wav.stem)
    s = json.dumps(asdict(r), ensure_ascii=False, indent=1)
    if a.json:
        a.json.write_text(s, encoding="utf-8")
    else:
        print(s)
