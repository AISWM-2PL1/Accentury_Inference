#!/usr/bin/env python3
"""F0·RMS 추출 — 트랙 1 model lock 구현 (A2)

명세: `docs/POC_PIPELINE_SPEC.md` §5-2-1

**참조 분포를 만들 때와 사용자를 잴 때 같은 설정을 타야 한다.** 설정이 다르면
거리의 의미가 깨진다. 그래서 여기 한 곳에만 두고 양쪽이 이 모듈을 부른다.

기존 프로브(`probe_cohort_prosody.py`·`probe_reading_effect.py`)와 세 곳이 다르다.

1. **성별 pitch 분기를 쓰지 않는다.** 추론에서 사용자 성별을 얻을 방법이 없다.
   대신 화자마다 **2단계 추정**으로 범위를 잡는다.
   🚨 기존 코드 주석이 분기 이유를 적어 두었다 — *"성별로 따로 잡지 않으면 여성 고음이
   잘리거나 남성이 옥타브 점프한다"*(`probe_reading_effect.py:34`). 2단계 추정은 그
   회피책이고 **일반론이다. 파일럿에서 「그냥 합치기」와 옥타브 오류율을 비교해 확정한다.**
2. **RMS 를 발화 단위로 상대화한다.** 마이크 거리·녹음 게인은 전 구간에 곱해지므로
   로그에서 상수가 되고 중앙값을 빼면 사라진다. v4 에 없던 규칙이고 이 명세가 정했다.
3. **semitone 정규화를 여기서 한다.** `p(t) = 12·log₂(F0/median_발화)`.
4. **출력 격자를 절대 시각으로 고정한다.** 🚨 Praat 은 `pitch_floor` 에 따라 분석 창 길이가
   달라져 **첫 프레임 위치와 개수가 화자마다 어긋난다**(2026-08-23 golden test 에서 잡았다 —
   180Hz 98프레임 대 360Hz 99프레임). 2단계 추정을 쓰면 화자마다 floor 가 다르므로
   그대로 두면 참조·추론 격자가 맞지 않는다. 그래서 `t_k = (k+0.5)·10ms` 격자에
   **최근접 프레임을 얹는다**(보간하지 않는다 — 유성·무성 경계가 뭉개진다).

사용
  uv run --no-project --with praat-parselmouth --with numpy python src/prosody/f0_extract.py --selftest
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import parselmouth

# ─── model lock ────────────────────────────────────────────────────────────
# 이 값을 바꾸면 참조 분포를 다시 만들어야 한다. registry/reference/f0_model_lock.json 과 같아야 한다.
LOCK_VERSION = "1.0.0"

TIME_STEP = 0.01          # 10ms. MFA 경계와 자를 맞추는 격자다
PASS1_FLOOR = 50.0        # 1차 통과: 남녀 어느 쪽도 자르지 않는 넓은 범위
PASS1_CEILING = 600.0
# 2차 통과 범위 = [q15 × K_LO, q65 × K_HI] (Hirst 2011 방식). 일반론이고 우리 데이터 검증 전이다.
K_LO, K_HI = 0.75, 1.5
CLAMP_FLOOR = 40.0        # 2차 범위가 터무니없이 벌어지지 않게 잠근다
CLAMP_CEILING = 600.0
MIN_RANGE_RATIO = 2.0     # ceiling/floor 가 이보다 좁으면 넓힌다
GRID_ORIGIN = 0.0         # 출력 격자 t_k = k × TIME_STEP. 절대 시각으로 고정한다.
                          # 🚨 0.5 로 두면 Praat 프레임(0.01 배수)과 정확히 5ms 어긋나
                          # 허용 경계에서 잘린다 — 2026-08-23 golden test 에서 잡았다
GRID_TOL = 0.5 + 1e-9     # 최근접 프레임 허용 거리 (TIME_STEP 배수). 경계의 부동소수 여유

# Praat to_pitch_ac 의 나머지 인자. 기본값이지만 **명시해서 잠근다** —
# parselmouth 나 Praat 버전이 바뀌어도 값이 따라 움직이지 않게 한다.
AC_PARAMS = dict(
    max_number_of_candidates=15,
    very_accurate=False,
    silence_threshold=0.03,
    voicing_threshold=0.45,
    octave_cost=0.01,
    octave_jump_cost=0.35,
    voiced_unvoiced_cost=0.14,
)

# 유성 판정: Praat 이 후보를 못 고른 프레임은 frequency == 0 으로 온다.
# 트래커마다 규칙이 다르므로(legacy-handoff-code) 우리 규칙을 못박는다.
VOICED = "parselmouth Pitch.selected_array['frequency'] > 0"

# UNSCORABLE_AUDIO — 명세 §4-1 의 여섯 중 이 모듈이 판정할 수 있는 것들
MIN_VOICED_FRAMES = 30
MIN_VOICED_SECONDS = 0.30
CLIP_RATIO_LIMIT = 0.10
MIN_F0_COVERAGE_WHEN_CLIPPED = 0.30


@dataclass
class F0Result:
    """한 발화의 추출 결과. `unscorable` 이 차면 나머지는 참고값이다."""
    times: np.ndarray = field(default_factory=lambda: np.array([]))
    f0_hz: np.ndarray = field(default_factory=lambda: np.array([]))
    voiced: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    semitone: np.ndarray = field(default_factory=lambda: np.array([]))   # p(t)
    rms_rel: np.ndarray = field(default_factory=lambda: np.array([]))    # log rms − median
    median_f0: float = 0.0
    pass2_floor: float = 0.0
    pass2_ceiling: float = 0.0
    voiced_ratio: float = 0.0
    clip_ratio: float = 0.0
    unscorable: str | None = None
    lock_version: str = LOCK_VERSION


def _pitch(snd: parselmouth.Sound, floor: float, ceiling: float) -> parselmouth.Pitch:
    return snd.to_pitch_ac(
        time_step=TIME_STEP, pitch_floor=floor, pitch_ceiling=ceiling, **AC_PARAMS
    )


def estimate_range(snd: parselmouth.Sound) -> tuple[float, float, np.ndarray]:
    """2단계 추정 1차 — 넓은 범위로 훑어 그 화자의 범위를 잡는다."""
    p1 = _pitch(snd, PASS1_FLOOR, PASS1_CEILING)
    f = p1.selected_array["frequency"]
    v = f[f > 0]
    if v.size == 0:
        return PASS1_FLOOR, PASS1_CEILING, v
    q15, q65 = float(np.percentile(v, 15)), float(np.percentile(v, 65))
    floor = max(CLAMP_FLOOR, q15 * K_LO)
    ceiling = min(CLAMP_CEILING, q65 * K_HI)
    if ceiling / floor < MIN_RANGE_RATIO:      # 너무 좁으면 잡히던 것도 놓친다
        mid = math.sqrt(floor * ceiling)
        floor = max(CLAMP_FLOOR, mid / math.sqrt(MIN_RANGE_RATIO))
        ceiling = min(CLAMP_CEILING, mid * math.sqrt(MIN_RANGE_RATIO))
    return floor, ceiling, v


def fixed_grid(duration: float) -> np.ndarray:
    """절대 시각 격자. pitch_floor 와 무관하게 항상 같다."""
    n = int(math.floor(duration / TIME_STEP))
    return (np.arange(n) + GRID_ORIGIN) * TIME_STEP


def snap(grid: np.ndarray, src_t: np.ndarray, src_v: np.ndarray, fill: float) -> np.ndarray:
    """격자에 최근접 프레임을 얹는다. 너무 멀면 fill 을 넣는다(보간 금지)."""
    out = np.full(grid.shape, fill, dtype=float)
    if src_t.size == 0:
        return out
    idx = np.clip(np.searchsorted(src_t, grid), 0, src_t.size - 1)
    for j, i in enumerate(idx):
        best = i
        if i > 0 and abs(src_t[i - 1] - grid[j]) < abs(src_t[i] - grid[j]):
            best = i - 1
        if abs(src_t[best] - grid[j]) <= GRID_TOL * TIME_STEP:
            out[j] = src_v[best]
    return out


def frame_rms(x: np.ndarray, sr: int, n_frames: int, t0: float) -> np.ndarray:
    """F0 와 같은 격자에서 프레임 RMS. 창 길이는 hop 의 2배(겹침 50%)."""
    win = int(round(TIME_STEP * 2 * sr))
    out = np.zeros(n_frames)
    for i in range(n_frames):
        c = int(round((t0 + i * TIME_STEP) * sr))
        a, b = max(0, c - win // 2), min(x.size, c + win // 2)
        seg = x[a:b]
        out[i] = float(np.sqrt(np.mean(seg * seg))) if seg.size else 0.0
    return out


def extract(x: np.ndarray, sr: int) -> F0Result:
    """WAV 샘플(mono float, −1~1) → F0 궤적 · semitone · 상대 RMS."""
    r = F0Result()
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.size == 0:
        r.unscorable = "decode 실패 — 빈 신호"
        return r

    r.clip_ratio = float(np.mean(np.abs(x) >= 0.999))

    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=sr)
    floor, ceiling, _ = estimate_range(snd)
    r.pass2_floor, r.pass2_ceiling = floor, ceiling

    p2 = _pitch(snd, floor, ceiling)
    raw_f = p2.selected_array["frequency"]
    raw_t = p2.xs()

    # 절대 격자로 옮긴다 — floor 가 달라도 참조·추론이 같은 자를 쓰게 한다
    t = fixed_grid(x.size / sr)
    f = snap(t, raw_t, raw_f, fill=0.0)
    r.times, r.f0_hz = t, f
    r.voiced = f > 0
    n_voiced = int(r.voiced.sum())
    r.voiced_ratio = n_voiced / f.size if f.size else 0.0

    if n_voiced < MIN_VOICED_FRAMES or n_voiced * TIME_STEP < MIN_VOICED_SECONDS:
        r.unscorable = "유성 F0 부족 (<30 프레임 또는 <0.30초)"
        return r
    if r.clip_ratio >= CLIP_RATIO_LIMIT and r.voiced_ratio < MIN_F0_COVERAGE_WHEN_CLIPPED:
        r.unscorable = "clipping ≥10% 이면서 F0 coverage <30%"
        return r

    med = float(np.median(f[r.voiced]))
    r.median_f0 = med
    st = np.full(f.shape, np.nan)
    st[r.voiced] = 12.0 * np.log2(f[r.voiced] / med)
    r.semitone = st

    rms = frame_rms(x, sr, f.size, float(t[0]) if t.size else 0.0)  # 같은 격자
    with np.errstate(divide="ignore"):
        lg = np.log(np.maximum(rms, 1e-12))
    r.rms_rel = lg - float(np.median(lg))     # 마이크 거리·게인이 여기서 사라진다
    return r


# ─── golden test ───────────────────────────────────────────────────────────

def _synth(f0: float, sec: float = 1.0, sr: int = 16000, gain: float = 0.3) -> np.ndarray:
    """하모닉 5개짜리 인공 성대음. 사인파 하나는 Praat 이 잘 못 잡는다."""
    t = np.arange(int(sec * sr)) / sr
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    return gain * x / np.max(np.abs(x))


def selftest() -> int:
    """model lock 이 지켜지는지 확인한다. 값을 바꾸면 여기가 먼저 깨져야 한다."""
    import parselmouth as pm
    fails = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'✅' if ok else '❌'} {name}{'  ' + detail if detail else ''}")
        if not ok:
            fails.append(name)

    print(f"parselmouth {pm.VERSION} · Praat {pm.PRAAT_VERSION} · lock {LOCK_VERSION}")

    # 1. 알려진 F0 를 되찾는가
    for hz in (110.0, 220.0, 330.0):
        r = extract(_synth(hz), 16000)
        got = r.median_f0
        check(f"F0 복원 {hz:.0f}Hz", r.unscorable is None and abs(got - hz) / hz < 0.02,
              f"→ {got:.1f}Hz")

    # 2. 2단계 추정이 남녀 범위를 모두 잡는가 (성별 분기 없이)
    for hz, who in ((95.0, "저음"), (260.0, "고음")):
        r = extract(_synth(hz), 16000)
        inside = r.pass2_floor <= hz <= r.pass2_ceiling
        check(f"2단계 범위가 {who}({hz:.0f}Hz)를 감싼다", inside,
              f"[{r.pass2_floor:.0f}, {r.pass2_ceiling:.0f}]")

    # 3. gain 불변 — 볼륨만 키워도 상대 RMS 는 같아야 한다
    a = extract(_synth(180.0, gain=0.2), 16000)
    b = extract(_synth(180.0, gain=0.8), 16000)
    d = float(np.nanmax(np.abs(a.rms_rel - b.rms_rel)))
    check("gain 불변 (RMS 상대화)", d < 1e-9, f"최대차 {d:.2e}")

    # 4. pitch shift 불변 — 통째로 올려도 semitone 궤적은 같아야 한다
    c = extract(_synth(180.0), 16000)
    e = extract(_synth(360.0), 16000)
    m = c.voiced & e.voiced
    d2 = float(np.nanmax(np.abs(c.semitone[m] - e.semitone[m]))) if m.any() else 9e9
    check("pitch shift 불변 (semitone 정규화)", d2 < 0.05, f"최대차 {d2:.3f} st")

    # 4-b. 격자 일치 — floor 가 달라도 프레임 수·시각이 같아야 한다
    check("격자 일치 (floor 가 달라도)",
          c.times.size == e.times.size and np.allclose(c.times, e.times),
          f"{c.times.size} vs {e.times.size} 프레임")

    # 5. 재현성 — 같은 입력이면 bit 단위로 같아야 한다
    x = _synth(200.0)
    r1, r2 = extract(x, 16000), extract(x, 16000)
    same = (np.array_equal(r1.f0_hz, r2.f0_hz)
            and np.array_equal(np.nan_to_num(r1.semitone), np.nan_to_num(r2.semitone))
            and np.array_equal(r1.rms_rel, r2.rms_rel))
    check("재현성 (byte-identical)", same)

    # 6. 격자 — hop 이 10ms 인가
    step = float(np.median(np.diff(r1.times))) if r1.times.size > 1 else 0.0
    check("hop 10ms", abs(step - TIME_STEP) < 1e-9, f"→ {step*1000:.2f}ms")

    # 7. UNSCORABLE — 무음은 측정 불가로 나와야 한다
    r0 = extract(np.zeros(16000), 16000)
    check("무음 → UNSCORABLE", r0.unscorable is not None, f"→ {r0.unscorable}")

    print(f"\n{'통과' if not fails else '실패: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    print(__doc__)
