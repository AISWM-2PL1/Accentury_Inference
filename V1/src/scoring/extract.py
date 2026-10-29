"""§3 표현 — 발화 하나를 곡선(어절 토막 20점)과 자리 특징으로 (`TRACK1_SCORING_SPEC.md` v2 §3·§4).

참조 빌드(§6)와 서비스 채점(§9)이 **이 함수 하나**(`extract_utterance`)를 부른다.

산출 (dict, JSON 직렬화 가능)
  curve         20·J 개 (어절 j 토막 20점, 빈칸 nan)
  syl           자리 N 개 × {tone_level, onset_level, syl_dur}  (syl_dur_resid 는 참조가 회귀 계수로 낸다)
  word          어절 J 개 × {peak_pos, peak_height, excursion, post_peak_fall, spread, plateau, peak_syl_idx}
  slope         tone_level 의 자리 번호 회귀 기울기 (st/자리)
  rate_sps      말속도 = 있는 음절 수 / 음절 구간 합 (음절/초)
  align         게이트·어절 판정 결과 (§4)

사용
  uv run --no-project --with praat-parselmouth --with numpy --with praatio --with soundfile \\
      python src/scoring/extract.py --prep registry/pipeline/pilot_d4_prep_2026-08-23.csv \\
      --base ~/AccenturyData/work/pilot/d4 --out ~/AccenturyData/work/scoring/pilot_gn.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.features import _head, _slice, read_textgrid, word_features  # noqa: E402
from prosody import f0_extract as F0  # noqa: E402
from scoring.align import (align_transcript, mfa_syllable_spans, position_spans,  # noqa: E402
                           word_spans)

TRAJ_K = 20                  # 어절 토막 점 수 (POC pilot_run.TRAJ_K 승계)
ONSET_HEAD_FRAC = 0.2
MIN_SLOPE_POINTS = 5
MIN_VOICED_FRAMES_SYL = 5    # 음절 높이는 유성 프레임 5개(50ms) 이상일 때만 — 3~4프레임 「치」가 74/144 Hz 로 튀어 문장 등수를 흔들었다 (2026-08-28) 【미검증 임시값】
WORD_FEATS = ("peak_pos", "peak_height", "excursion", "post_peak_fall", "spread", "plateau")
SYL_FEATS = ("tone_level", "onset_level", "syl_dur")
# 실험 축 (2026-08-30 · §17-16 후속): 음절을 시간 기준 앞반·뒤반으로 나눈 반음 중앙값. SYL_FEATS 에 넣지 않는다 —
# 넣으면 f 축(자리 특징 z)의 특징 집합이 바뀐다. 두 반쪽 각각 유성 프레임 MIN_HALF_FRAMES 이상일 때만 값이 있다.
SYL_HALVES = ("half_head", "half_tail")
MIN_HALF_FRAMES = 2

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("ACCENTURY_DATA", Path.home() / "AccenturyData"))


def word_grid(r: F0.F0Result, s: float, e: float, k: int = TRAJ_K) -> list[float]:
    """어절 [s,e) 의 `p` 를 정규화 시간 위 k 점으로. 무성 자리는 nan (보간하지 않는다 — POC 와 같다)."""
    out = [float("nan")] * k
    if e <= s or r.times.size == 0:
        return out
    m = (r.times >= s) & (r.times < e) & r.voiced
    t, v = r.times[m], r.semitone[m]
    if t.size == 0:
        return out
    tol = (e - s) / (2 * k) + F0.TIME_STEP / 2
    for i in range(k):
        c = s + (e - s) * (i + 0.5) / k
        d = np.abs(t - c)
        j = int(np.argmin(d))
        if d[j] <= tol:
            out[i] = float(v[j])
    return out


def _nan(x) -> float:
    return float(x) if x is not None and x == x else float("nan")


def extract_utterance(x: np.ndarray, sr: int, words_phones: tuple, transcript: str, script: str,
                      *, utt_id: str = "", f0: F0.F0Result | None = None) -> dict:
    """WAV + MFA + 전사 + 대본 → §3 표현. 실패 사유는 `unscorable` 에."""
    out: dict = {"file_id": utt_id, "unscorable": None}
    A = align_transcript(script, transcript)
    out["align"] = {"gate_ratio": round(A.gate_ratio, 4), "passed": A.passed,
                    "blank_word_ratio": round(A.blank_word_ratio, 4), "usable": A.usable,
                    "present": A.present, "n_syl": A.n_syl, "n_word": len(A.words)}
    if not A.usable:
        out["unscorable"] = "게이트 탈락" if not A.passed else "어절 빈칸 초과"
        return out

    r = f0 if f0 is not None else F0.extract(x, sr)
    if r.unscorable:
        out["unscorable"] = f"S2 결측 — {r.unscorable}"
        return out
    words, phones = words_phones
    syl_spans = mfa_syllable_spans(words, phones, transcript)
    if syl_spans is None:
        out["unscorable"] = "MFA 라벨 ≠ 전사"
        return out
    pos = position_spans(A, syl_spans)
    wsp = word_spans(A, pos)
    blank = [False] * len(pos)          # 「없음」 어절(§4-3)에 속한 자리 — 음절층도 빈칸 (2026-08-29)
    for (a0, e0), ok_w in zip(A.bounds, A.present):
        if not ok_w:
            for t0 in range(a0, e0):
                blank[t0] = True

    # 음절층 — 자리마다 결측 사유를 남긴다 (§6 저장 계약, 2026-08-29):
    #   "" 있음 · sub 다른 말이 온 어절 · del 짝 없음(삭제) · spn MFA 구간 없음 · vf 유성 프레임 < 5
    syl = {k: [] for k in SYL_FEATS + SYL_HALVES}
    miss: list[str] = []
    for t0, sp in enumerate(pos):
        if blank[t0] or sp is None:
            for k in SYL_FEATS + SYL_HALVES:
                syl[k].append(float("nan"))
            miss.append("sub" if blank[t0] else ("del" if A.pair[t0] is None else "spn"))
            continue
        a, b = sp
        t, p, _ = _slice(r, a, b)
        syl["syl_dur"].append(b - a)
        ok = p.size >= MIN_VOICED_FRAMES_SYL
        syl["tone_level"].append(float(np.median(p)) if ok else float("nan"))
        syl["onset_level"].append(_head(t, p, a, b, ONSET_HEAD_FRAC) if ok else float("nan"))
        mid = (a + b) / 2                      # 실험 축 — 시간 기준 앞반·뒤반 (유성 프레임만, 각 반쪽 MIN_HALF_FRAMES 이상)
        for k, part in (("half_head", p[t < mid]), ("half_tail", p[t >= mid])):
            syl[k].append(float(np.median(part)) if ok and part.size >= MIN_HALF_FRAMES else float("nan"))
        miss.append("" if ok else "vf")

    # 어절층 + 곡선
    word = {k: [] for k in WORD_FEATS}
    peak_syl_idx: list[int] = []
    curve: list[float] = []
    for j, (label, sp) in enumerate(zip(A.words, wsp)):
        if sp is None:
            for k in WORD_FEATS:
                word[k].append(float("nan"))
            peak_syl_idx.append(-1)
            curve.extend([float("nan")] * TRAJ_K)
            continue
        s, e = sp
        wf = word_features(r, phones, (s, e, label), j, use_syllable=True)
        for k in WORD_FEATS:
            word[k].append(_nan(getattr(wf, k)) if wf.missing is None else float("nan"))
        peak_syl_idx.append(wf.peak_syl_idx if wf.missing is None else -1)
        curve.extend(word_grid(r, s, e))

    # 문장 기울기 · 말속도
    tl = np.array(syl["tone_level"])
    idx = np.arange(tl.size)
    m = ~np.isnan(tl)
    slope = float(np.polyfit(idx[m], tl[m], 1)[0]) if m.sum() >= MIN_SLOPE_POINTS else float("nan")
    durs = [d for d in syl["syl_dur"] if d == d]
    rate = (len(durs) / sum(durs)) if durs and sum(durs) > 0 else float("nan")

    out.update({"curve": curve, "syl": syl, "word": word, "peak_syl_idx": peak_syl_idx,
                "slope": slope, "rate_sps": rate, "words": A.words,
                "syl_miss": miss, "curve_blank": round(float(np.mean(np.isnan(curve))), 4),
                "median_f0": round(r.median_f0, 1)})
    return out


def load_scripts(sentences_json: Path) -> dict[str, str]:
    sent = json.load(open(sentences_json, encoding="utf-8"))
    if "문장" in sent:  # s4_build.py select 의 형식 — 문장 목록 [{script_key, 대본, …}]
        return {c["script_key"]: c["대본"] for c in sent["문장"]}
    return {c["script_key"]: c["대본"] for v in sent["유형별_후보"].values() for c in v["상위 5"]}


def run_batch(prep: Path, base: Path, out: Path, scripts: dict[str, str],
              keys: set[str] | None = None, limit: int | None = None) -> dict:
    import soundfile as sf

    rows = [r for r in csv.DictReader(open(prep, encoding="utf-8")) if (keys is None or r["script_key"] in keys)]
    if limit:
        rows = rows[:limit]
    out.parent.mkdir(parents=True, exist_ok=True)
    stat = {"n": 0, "usable": 0, "fail": {}}
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as fo:
        for i, r in enumerate(rows):
            fid, spk = r["file_id"], r["speaker_id"]
            wav, tg = base / "lab" / spk / f"{fid}.wav", base / "aligned" / spk / f"{fid}.TextGrid"
            rec = {"file_id": fid, "script_key": r["script_key"], "speaker_id": spk,
                   "collector_id": r.get("collector_id", ""), "gender": r.get("gender", ""),
                   "city_name": r.get("city_name", ""), "transcript": r["transcript"]}
            if not tg.exists():
                rec["unscorable"] = "TextGrid 없음"
            else:
                x, sr = sf.read(str(wav))
                if x.ndim > 1:
                    x = x.mean(axis=1)
                rec.update(extract_utterance(x, sr, read_textgrid(tg), r["transcript"],
                                             scripts[r["script_key"]], utt_id=fid))
            stat["n"] += 1
            if rec["unscorable"]:
                stat["fail"][rec["unscorable"]] = stat["fail"].get(rec["unscorable"], 0) + 1
            else:
                stat["usable"] += 1
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(rows)} · {time.time() - t0:.0f}s", flush=True)
    stat["sec"] = round(time.time() - t0, 1)
    return stat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--sentences", default=str(REPO / "registry/pipeline/pilot_d4_sentences_2026-08-23.json"))
    ap.add_argument("--base", default=str(DATA / "work/pilot/d4"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--keys", default=None, help="script_key 를 쉼표로 (기본 전부)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    scripts = load_scripts(Path(a.sentences))
    keys = set(a.keys.split(",")) if a.keys else None
    stat = run_batch(Path(a.prep), Path(a.base).expanduser(), Path(a.out).expanduser(), scripts, keys, a.limit)
    print(json.dumps(stat, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
