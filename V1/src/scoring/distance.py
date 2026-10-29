"""§5 거리 — z 통계 · 두 발화 거리 · 화자 단위 이웃 거리 (`TRACK1_SCORING_SPEC.md` v2 §5).

  d_curve   두 곡선이 모두 값 있는 칸에서 |a−b| 평균 (반음). 공유 <MIN_SHARE_D 면 nan
  fit_z     특징 f · 자리 t 마다 μ·s (ddof=1). 유효 n<Z_MIN_N 이거나 s<Z_MIN_S_FRAC·중앙값이면 무효
  zscore    (x−μ)/s, |z|≤Z_CLIP
  d_feat    층(음절·어절·문장)별 RMS 를 먼저, 층 평균 (1:1:1). 공유 <MIN_SHARE_D 면 nan
  knn_speaker  화자마다 가장 가까운 발화 하나 → 앞 K 명 중앙값. 공유 <MIN_SHARE_USE 는 뺌
"""
from __future__ import annotations

import math

import numpy as np

K = 10                  # P1 【미검증 임시값】
MIN_SHARE_D = 0.5       # P7 거리 정의역
MIN_SHARE_USE = 0.7     # P7 점수·청소에서 쓰는 비교
Z_MIN_N = 10            # P8
Z_MIN_S_FRAC = 0.05     # P8
Z_CLIP = 4.0

LAYERS = ("syl", "word", "sent")


def d_curve(a: np.ndarray, b: np.ndarray, min_share: float = MIN_SHARE_D) -> tuple[float, float]:
    """반환 (거리, 공유 비율)."""
    m = ~np.isnan(a) & ~np.isnan(b)
    share = float(m.mean()) if a.size else 0.0
    if share < min_share or not m.any():
        return float("nan"), share
    return float(np.mean(np.abs(a[m] - b[m]))), share


class ZStats:
    """특징별 (자리 수,) μ · s · valid. `fit` 은 참조 fit 발화 행렬(n × T)에서."""

    def __init__(self) -> None:
        self.mu: dict[str, np.ndarray] = {}
        self.s: dict[str, np.ndarray] = {}
        self.valid: dict[str, np.ndarray] = {}
        self.n: dict[str, np.ndarray] = {}

    def fit(self, name: str, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        n = (~np.isnan(X)).sum(axis=0)
        with np.errstate(all="ignore"):
            mu = np.nanmean(X, axis=0)
            s = np.nanstd(X, axis=0, ddof=1)
        med = float(np.nanmedian(s)) if np.isfinite(s).any() else float("nan")
        valid = (n >= Z_MIN_N) & np.isfinite(s) & (s >= Z_MIN_S_FRAC * med if med == med else False)
        self.mu[name], self.s[name], self.valid[name], self.n[name] = mu, s, valid, n

    def z(self, name: str, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            x = x[None]
        with np.errstate(all="ignore"):
            z = (x - self.mu[name]) / self.s[name]
        z = np.where(self.valid[name], z, np.nan)
        return np.clip(z, -Z_CLIP, Z_CLIP)

    def to_json(self) -> dict:
        return {k: {"mu": _lst(self.mu[k]), "s": _lst(self.s[k]),
                    "valid": self.valid[k].tolist(), "n": self.n[k].tolist()} for k in self.mu}

    @classmethod
    def from_json(cls, d: dict) -> "ZStats":
        o = cls()
        for k, v in d.items():
            o.mu[k] = np.array(v["mu"], dtype=float)
            o.s[k] = np.array(v["s"], dtype=float)
            o.valid[k] = np.array(v["valid"], dtype=bool)
            o.n[k] = np.array(v["n"], dtype=int)
        return o


def _lst(a: np.ndarray) -> list:
    return [None if not np.isfinite(v) else float(v) for v in a]


def zvec(zs: ZStats, feats: dict[str, np.ndarray], layer_of: dict[str, str]) -> dict[str, np.ndarray]:
    """특징 dict → 층별로 이어 붙인 z 벡터."""
    out: dict[str, list] = {L: [] for L in LAYERS}
    for name, x in feats.items():
        out[layer_of[name]].append(zs.z(name, x))
    return {L: (np.concatenate(v) if v else np.array([])) for L, v in out.items()}


def d_feat(za: dict[str, np.ndarray], zb: dict[str, np.ndarray],
           min_share: float = MIN_SHARE_D) -> tuple[float, float]:
    """층별 RMS 의 평균. 공유 비율은 전 층 합산."""
    per, tot_m, tot_n = [], 0, 0
    for L in LAYERS:
        a, b = za.get(L, np.array([])), zb.get(L, np.array([]))
        if a.size == 0:
            continue
        m = ~np.isnan(a) & ~np.isnan(b)
        tot_m += int(m.sum())
        tot_n += a.size
        if m.any():
            per.append(float(np.sqrt(np.mean((a[m] - b[m]) ** 2))))
    share = tot_m / tot_n if tot_n else 0.0
    if share < min_share or not per:
        return float("nan"), share
    return float(np.mean(per)), share


MIN_RANK_WORDS = 2      # 어절 안 등수 — 짝지어진 2음절+ 어절 하한
MIN_RANK_SYL = 10       # 문장 안 등수 — 공유 음절 하한
TIE_ST = 0.0            # 어절 안 등수 동률 문턱(st). 0.5 로 실험(2026-08-28)했으나 재현성이 안 좋아지고(두 녹음 차 평균 6.9→6.9) 1|5 눈금이 q50 0 으로 무너져 0 으로 되돌림


def word_bounds(words: list[str]) -> list[tuple[int, int]]:
    out, p = [], 0
    for w in words:
        out.append((p, p + len(w))); p += len(w)
    return out


def d_word_rank(ta: np.ndarray, tb: np.ndarray, bounds: list[tuple[int, int]],
                min_syl: int = 2, min_words: int = MIN_RANK_WORDS, tie: float = TIE_ST) -> tuple[float, float]:
    """③′ 축 1 — 어절 안 음절 높이 순서. 2음절+ 어절 **안에서만** 음절 쌍을 만들고, 그 쌍을 문장 전체에서
    한 바구니에 모아 뒤집힌 비율을 낸다 (v3.2 · 2026-08-29).

    v3.1 까지는 어절마다 비율을 내고 **평균**했다 — 2음절 어절은 쌍이 1개뿐이라 0 아니면 1 이 되고,
    그것이 어절 수분의 1 만큼 통째로 들어가 계단이 컸다 (`1|1` 한 칸 10%). 쌍 합산이면 같은 문장이 37칸이 된다.
    덤으로 긴 어절이 쌍 수만큼 무거워진다 (「모르겄더라고예」 21쌍 : 「버스」 1쌍).

    두 발화 모두 그 어절 음절이 전부 있어야 센다. 어느 쪽이든 두 음절 차가 `tie` 이하인 쌍은 분자·분모에서 뺀다.
    반환 (거리 0~1, 센 어절 비율)."""
    used, n2, disc, tot = 0, 0, 0, 0
    for a, e in bounds:
        if e - a < min_syl:
            continue
        n2 += 1
        x, y = ta[a:e], tb[a:e]
        if np.isnan(x).any() or np.isnan(y).any():
            continue
        used += 1
        n = x.size
        for i in range(n):
            for j in range(i + 1, n):
                if abs(x[i] - x[j]) <= tie or abs(y[i] - y[j]) <= tie:
                    continue                    # 동률 — 분모에서도 뺀다 (2026-08-29 · 코덱스 2차 중대)
                tot += 1
                if (x[i] - x[j]) * (y[i] - y[j]) < 0:
                    disc += 1
    if used < min_words or tot == 0:
        return float("nan"), 0.0
    return disc / tot, used / max(n2, 1)


def expand_bounds(bounds: list[tuple[int, int]], k: int) -> list[tuple[int, int]]:
    """음절당 k 점 벡터용 어절 경계 — 자리 t 가 [t·k, t·k+k) 로 늘어난다 (실험 축 w2 · 2026-08-30)."""
    return [(a * k, e * k) for a, e in bounds]


MIN_SCALE_SYL = 5       # 발화 퍼짐(std)을 낼 최소 음절 수 (실험 축 m)


def tone_scale(t: np.ndarray, min_n: int = MIN_SCALE_SYL) -> float:
    """발화의 음절 높이 퍼짐(std, 반음). 실험 축 m 의 정규화 분모 — 목소리 폭이 큰 사람·작은 사람을 같은 자로 놓는다."""
    v = t[~np.isnan(t)]
    return float(v.std()) if v.size >= min_n and v.std() > 0 else float("nan")


MIN_DELTA_PAIRS = 3


def d_delta_norm(ta: np.ndarray, tb: np.ndarray, bounds: list[tuple[int, int]], sa: float, sb: float,
                 min_pairs: int = MIN_DELTA_PAIRS) -> tuple[float, float]:
    """실험 축 m — 어절 안 인접 음절 변화량을 각 발화의 퍼짐으로 나눈 뒤 |Δa − Δb| 평균 (2026-08-30 · §17-16 후속).
    정규화 없는 변화량은 경상도 TTS 를 표준어보다 더 깎았다(방향 반대, 2026-08-28 §5 「버린 후보」와 같음);
    퍼짐으로 나누면 방향이 맞았다 (`scratchpad` 스윕 2026-08-30 — 정식 기록은 §17). 반환 (거리, 센 쌍 비율)."""
    if sa != sa or sb != sb:
        return float("nan"), 0.0
    diffs, tot = [], 0
    for a, e in bounds:
        for i in range(a, e - 1):
            tot += 1
            da, db = (ta[i + 1] - ta[i]) / sa, (tb[i + 1] - tb[i]) / sb
            if da == da and db == db:
                diffs.append(abs(da - db))
    if len(diffs) < min_pairs:
        return float("nan"), (len(diffs) / tot if tot else 0.0)
    return float(np.mean(diffs)), len(diffs) / tot


FLAT_CUT_LO, FLAT_CUT_HI = 10.0, 90.0   # P12 자리 백분위 컷 — <10 밋밋 · >90 과장 (§17-17 ③)
FLAT_MIN_POS = 5                         # 잰 자리가 이보다 적으면 flat/exag 를 내지 않는다
FLAT_MIN_REF = 20                        # 자리 분포의 참조 값이 이보다 적은 자리는 건너뛴다


def delta_pos(t: np.ndarray, bounds: list[tuple[int, int]]) -> np.ndarray:
    """자리별 음절 변화량 — 자리 i 는 어절 안 인접 쌍 i→i+1, Δ_i = t[i+1] − t[i] (반음). 어절 마지막 음절 자리는 nan.
    길이는 t 와 같다 (P12 · §17-17 ③ · `flat_probe.delta` 와 같은 식)."""
    d = np.full(t.size, np.nan)
    for a, e in bounds:
        for i in range(a, e - 1):
            d[i] = t[i + 1] - t[i]
    return d


def flat_exag(x: np.ndarray, R: np.ndarray, keep: np.ndarray | None = None, cut_lo: float = FLAT_CUT_LO, cut_hi: float = FLAT_CUT_HI,
              min_pos: int = FLAT_MIN_POS, min_ref: int = FLAT_MIN_REF, max_abs: float | None = None,
              mode: str = "count") -> tuple[float, float, np.ndarray]:
    """P12 밋밋(flat)·과장(exag) — 자리마다 Δ 를 경남 참조 전원의 그 자리 분포에 대고 백분위를 낸다 (§17-17 ③).

    방향 = sign(참조 Δ 중앙값) — 경남이 내리는 자리면 −, 올리는 자리면 +. 백분위 p_i = (참조 Δ×방향 < Δ_i×방향) 비율 × 100.
    낮을수록 경남보다 덜 움직였다. flat = p_i < cut_lo 인 자리 비율 · exag = p_i > cut_hi 인 자리 비율.
    이웃 10 명이 아니라 **참조 전원**과 비교하고, 어절로 뭉치지 않고 자리마다 본다 — w 축이 서울말을 못 가른 두 원인을 다 피한다.

    x (T,) 사용자 Δ · R (n, T) 참조 Δ 행렬 · keep (n,) 쓸 행 마스크(자기 화자 제외용, None 이면 전부).
    max_abs: |Δ| 가 이보다 큰 자리는 재지 않는다 (옥타브 오류·creak 방어 후보 — 실험용, None 이면 끔 · §17-18).
    mode: flat 의 집계 방식 (실험 · 2026-08-31 데모 후속). "count" = 현행 (p_i < cut_lo 인 자리 비율 — 자리 하나가 0/1 로 들어가
      잰 자리 12개면 자리 1개 = 0.083 계단) · "deficit" = mean(max(0, cut_lo − p_i) / cut_lo) (부족량 평균, 0~1 연속) ·
      "pctmean" = mean(100 − p_i) / 100 (백분위 평균의 보수, cut_lo 무관). exag 는 mode 와 무관하게 현행 count.
    반환 (flat, exag, 자리별 백분위 (T,) — 안 잰 자리는 nan). 잰 자리 < min_pos 면 flat·exag 는 nan."""
    if keep is not None:
        R = R[keep]
    ps = np.full(x.size, np.nan)
    for j in range(x.size):
        if x[j] != x[j] or (max_abs is not None and abs(x[j]) > max_abs):
            continue
        col = R[:, j]
        col = col[~np.isnan(col)]
        if col.size < min_ref:
            continue
        s = np.sign(np.median(col)) or 1.0
        ps[j] = ((col * s) < (x[j] * s)).mean() * 100
    got = ps[~np.isnan(ps)]
    if got.size < min_pos:
        return float("nan"), float("nan"), ps
    exag = float(np.mean(got > cut_hi))
    if mode == "count":
        return float(np.mean(got < cut_lo)), exag, ps
    if mode == "deficit":
        return float(np.mean(np.clip(cut_lo - got, 0.0, None) / cut_lo)), exag, ps
    if mode == "pctmean":
        return float(np.mean(100.0 - got) / 100.0), exag, ps
    raise ValueError(f"flat mode {mode!r}")


def rank_avg(v: np.ndarray) -> np.ndarray:
    """동점은 평균 등수 (`scipy.stats.rankdata` 'average' 와 같다 — scipy 없이).

    `argsort(argsort())` 는 동점에 입력 순서대로 다른 등수를 준다 (2026-08-29 · 코덱스 2차 중대)."""
    o = np.argsort(v, kind="mergesort")
    s = v[o]
    r = np.empty(v.size, dtype=float)
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and s[j + 1] == s[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def d_sent_rank(ta: np.ndarray, tb: np.ndarray, min_n: int = MIN_RANK_SYL) -> tuple[float, float]:
    """③′ 축 2 — 문장 안 음절 높이 등수. 공유 음절 위에서 스피어만 상관(동점 평균 등수), 거리 = 1 − ρ."""
    m = ~np.isnan(ta) & ~np.isnan(tb)
    share = float(m.mean()) if ta.size else 0.0
    if m.sum() < min_n:
        return float("nan"), share
    rx, ry = rank_avg(ta[m]), rank_avg(tb[m])
    if rx.std() == 0 or ry.std() == 0:          # 전부 동점 — 등수가 서지 않는다
        return float("nan"), share
    return float(1 - np.corrcoef(rx, ry)[0, 1]), share


def n2_effective(t: np.ndarray, bounds: list[tuple[int, int]], min_syl: int = 2) -> int:
    """이 발화에서 실제로 셀 수 있는 2음절+ 어절 수 — w 계단의 실제 크기는 1/이 값이다 (2026-08-29)."""
    return sum(1 for a, e in bounds if e - a >= min_syl and not np.isnan(t[a:e]).any())


MIN_TONE_PAIRS = 3


def d_tone(a: list[int], b: list[int], min_pairs: int = MIN_TONE_PAIRS) -> tuple[float, float]:
    """성조형 불일치율 — 두 발화 모두 정점 음절이 있는 어절 중 `peak_syl_idx` 가 다른 비율.
    반환 (거리 0~1, 공유 비율). 짝이 min_pairs 미만이면 nan."""
    pairs = [(x, y) for x, y in zip(a, b) if x > 0 and y > 0]
    share = len(pairs) / len(a) if a else 0.0
    if len(pairs) < min_pairs:
        return float("nan"), share
    return float(np.mean([x != y for x, y in pairs])), share


def knn_speaker(cands: list[tuple[str, float, float]], k: int = K,
                min_share: float = MIN_SHARE_USE) -> tuple[float, list[tuple[str, float]]]:
    """cands = [(speaker_id, d, share)]. 반환 (이웃 중앙값 거리, 이웃 목록[(화자, d)]).

    화자마다 가장 가까운 발화 하나만 남기고, 앞 k 명의 중앙값. 유효 화자 <k 면 nan.
    """
    best: dict[str, float] = {}
    for spk, d, sh in cands:
        if d != d or sh < min_share:
            continue
        if spk not in best or d < best[spk]:
            best[spk] = d
    ranked = sorted(best.items(), key=lambda kv: kv[1])[:k]
    if len(ranked) < k:
        return float("nan"), ranked
    return float(np.median([d for _, d in ranked])), ranked


def resid_fit(log_dur: np.ndarray, log_rate: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """§3-3 회귀 — log(syl_dur) ~ 1 + log(rate_sps) + 자리 비율. 계수 [b0, b1, b2]."""
    m = np.isfinite(log_dur) & np.isfinite(log_rate) & np.isfinite(pos)
    X = np.column_stack([np.ones(m.sum()), log_rate[m], pos[m]])
    b, *_ = np.linalg.lstsq(X, log_dur[m], rcond=None)
    return b


def resid_apply(b: np.ndarray, syl_dur: np.ndarray, rate_sps: float) -> np.ndarray:
    n = syl_dur.size
    pos = np.arange(n) / max(n - 1, 1)
    with np.errstate(all="ignore"):
        pred = b[0] + b[1] * math.log(rate_sps) + b[2] * pos if rate_sps == rate_sps and rate_sps > 0 else np.full(n, np.nan)
        return np.log(syl_dur) - pred
