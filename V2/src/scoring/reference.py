"""§6 참조 분포 만들기 — 문장별 (`TRACK1_SCORING_SPEC.md` v3 §6).

채점 축 (③′ · 2026-08-28 확정 후보):
  w  어절 안 음절 높이 순서 (2음절+ 어절, 켄달 거리 평균)           distance.d_word_rank
  r  문장 안 음절 높이 등수 (공유 음절 스피어만, 1−ρ)                distance.d_sent_rank
피드백 축 (점수에 안 들어감): c 곡선(반음) · f 자리 특징 z — 같은 LOSO 로 눈금을 저장해 둔다 (어느 어절이 달랐나 · 폭 평평/과장).
자리 축 (P12 · 2026-08-31 · §17-17): flat 밋밋 · exag 과장 — 자리별 음절 변화량의 경남 백분위 (`distance.flat_exag`). 이웃이 아니라 참조 전원 분포.
  점수 = min(현행 w·r, flat) (`params.combine` = "min_flat"). exag 는 피드백·기록 전용. 눈금은 교차적합 발화의 자기-제외 flat/exag 의 q50/q90.

문장마다
  1. 게이트·어절 빈칸 통과 (`usable`)
  2. §3-3 회귀 계수 · §5-1 z 통계 적합 (통과 발화 전체)
  3. 자기 화자 제외 이웃 거리 (LOSO) → w·r 상위 CLEAN_FRAC 청소
  4. 남은 발화로 LOSO 재계산 → 눈금 q50 · q95 (축마다)
  4′. flat/exag 눈금은 4 와 같은 교차적합(fold 참조, 자기 화자 없음)의 D 로 낸다 (2026-08-31 코덱스 4차 반영). 발화 LOO 값은 참고로만 저장
  5. 저장 — 본체 `work/scoring/reference_<key>.json`, 요약 `registry/scoring/reference_<key>_<stamp>.json`

사용
  uv run --no-project --with numpy python src/scoring/reference.py \\
      --jsonl ~/AccenturyData/work/scoring/pilot_gn.jsonl --stamp 2026-08-28
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scoring import distance as D  # noqa: E402
from scoring.extract import SYL_FEATS, WORD_FEATS  # noqa: E402

CLEAN_FRAC = 0.05       # P3 【미검증 임시값】
MIN_SPEAKERS = 30       # P6
SCALE_FOLDS = 10        # P11 (2026-08-30): 눈금은 화자 K-fold 교차적합 D 의 분위수로 — 참조 화자 자신의 LOSO 는 새 화자에게 5% 라 적고 20% 를 줬다 (§17-7·§17-10)
SCALE_HI = 90           # P11 위끝 백분위 — 90 이면 v3.2 까지의 점수와 거의 같다 (현행 눈금의 정체가 교차적합 q90 이었다, §17-10). 95 는 경남 보호 쪽
SCALE_SEED = 20260830
SCORE_AXES = ("w", "r")
# 실험 축 (2026-08-30 · §17-16 후속): w2 어절 안 앞반|뒤반 2점 순서 · r2 문장 안 2점 등수 · m 퍼짐 정규화 변화량. 눈금은 같이 저장하되
# 점수에는 `params.weights` 가 그 축을 들 때만 들어간다 — 청소(P3)는 여전히 SCORE_AXES 로만 한다.
EXP_AXES = ("w2", "r2", "m")
ALL_AXES = ("w", "r", "c", "f") + EXP_AXES          # 이웃 거리 축 — loso()·knn 이 도는 축
# 자리 축 (P12 · 2026-08-31): 이웃 거리가 아니라 참조 전원의 자리별 Δ 분포 대비 백분위 비율. loso() 를 돌지 않고 build() 에서 따로 눈금을 낸다.
POS_AXES = ("flat", "exag")
SCALE_AXES = ALL_AXES + POS_AXES                    # 눈금(scale)이 저장되는 축 전부
COMBINE = "min_flat"    # P12 합산 규칙 — 점수 = min(현행 w·r, flat). 사람 결정 2026-08-31: 서울 1명 근거로 잠정 적용해 전달 (추가 녹음 확인은 전달 뒤)
REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("ACCENTURY_DATA", Path.home() / "AccenturyData"))
LAYER_OF = {**{f: "syl" for f in SYL_FEATS if f != "syl_dur"}, "syl_dur_resid": "syl",
            **{f: "word" for f in WORD_FEATS}}
FEAT_NAMES = [f for f in SYL_FEATS if f != "syl_dur"] + ["syl_dur_resid"] + list(WORD_FEATS)


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "?"


def feats_of(u: dict, coef: np.ndarray) -> dict[str, np.ndarray]:
    f: dict[str, np.ndarray] = {}
    for k in SYL_FEATS:
        if k != "syl_dur":
            f[k] = np.array(u["syl"][k], dtype=float)
    f["syl_dur_resid"] = D.resid_apply(coef, np.array(u["syl"]["syl_dur"], dtype=float), u["rate_sps"])
    for k in WORD_FEATS:
        f[k] = np.array(u["word"][k], dtype=float)
    return f


def fit_resid(us: list[dict]) -> np.ndarray:
    ld, lr, ps = [], [], []
    for u in us:
        d = np.array(u["syl"]["syl_dur"], dtype=float)
        n = d.size
        ld.extend(np.log(d))
        lr.extend([math.log(u["rate_sps"]) if u["rate_sps"] == u["rate_sps"] and u["rate_sps"] > 0 else np.nan] * n)
        ps.extend(np.arange(n) / max(n - 1, 1))
    return D.resid_fit(np.array(ld), np.array(lr), np.array(ps))


def fit_z(us: list[dict], coef: np.ndarray) -> D.ZStats:
    zs = D.ZStats()
    F = [feats_of(u, coef) for u in us]
    for name in FEAT_NAMES:
        zs.fit(name, np.stack([f[name] for f in F]))
    return zs


def tones(u: dict) -> np.ndarray:
    return np.array(u["syl"]["tone_level"], dtype=float)


def tones2(u: dict) -> np.ndarray | None:
    """실험 축 w2 — 자리마다 (앞반, 뒤반) 두 점을 이어 붙인 벡터. 옛 추출물(칸 없음)이면 None."""
    s = u["syl"]
    if "half_head" not in s or "half_tail" not in s:
        return None
    return np.column_stack([np.array(s["half_head"], dtype=float), np.array(s["half_tail"], dtype=float)]).ravel()


def loso(us: list[dict], bounds, curves: np.ndarray, Z: list[dict], k: int = D.K) -> list[dict]:
    """발화마다 자기 화자를 뺀 이웃 거리 — 축 전부(ALL_AXES). 반환 [{D_w, D_r, D_c, D_f, D_w2, D_m, nb_w, nb_r}]."""
    T = [tones(u) for u in us]
    T2 = [tones2(u) for u in us]
    S = [D.tone_scale(t) for t in T]
    bounds2 = D.expand_bounds(bounds, 2)
    out = []
    for i, u in enumerate(us):
        c = {a: [] for a in ALL_AXES}
        for j, v in enumerate(us):
            if v["speaker_id"] == u["speaker_id"]:
                continue
            dw, sw = D.d_word_rank(T[i], T[j], bounds); c["w"].append((v["speaker_id"], dw, 1.0))
            dr, sr = D.d_sent_rank(T[i], T[j]); c["r"].append((v["speaker_id"], dr, 1.0))
            dc, sc = D.d_curve(curves[i], curves[j]); c["c"].append((v["speaker_id"], dc, sc))
            df, sf = D.d_feat(Z[i], Z[j]); c["f"].append((v["speaker_id"], df, sf))
            dw2 = D.d_word_rank(T2[i], T2[j], bounds2)[0] if (T2[i] is not None and T2[j] is not None) else float("nan")
            c["w2"].append((v["speaker_id"], dw2, 1.0))
            dr2 = D.d_sent_rank(T2[i], T2[j], min_n=2 * D.MIN_RANK_SYL)[0] if (T2[i] is not None and T2[j] is not None) else float("nan")
            c["r2"].append((v["speaker_id"], dr2, 1.0))
            dm, sm = D.d_delta_norm(T[i], T[j], bounds, S[i], S[j]); c["m"].append((v["speaker_id"], dm, 1.0))
        rec = {"file_id": u["file_id"], "speaker_id": u["speaker_id"]}
        for a in ALL_AXES:
            Da, nb = D.knn_speaker(c[a], k, min_share=0.0 if a in SCORE_AXES else D.MIN_SHARE_USE)
            rec[f"D_{a}"] = Da
            if a in SCORE_AXES:
                rec[f"nb_{a}"] = [s for s, _ in nb]
        out.append(rec)
    return out


def q(vals: list[float], p: float) -> float:
    v = [x for x in vals if x is not None and x == x]
    return float(np.percentile(v, p)) if v else float("nan")


def _core(us: list[dict], key: str, k: int, clean_frac: float):
    """§6 2~4 단계 — 회귀·z → LOSO 1차 → 청소 → LOSO 2차. 눈금은 안 정한다."""
    words = us[0]["words"]
    bounds = D.word_bounds(words)
    coef = fit_resid(us)
    zs = fit_z(us, coef)
    curves = np.array([u["curve"] for u in us], dtype=float)
    Z = [D.zvec(zs, feats_of(u, coef), LAYER_OF) for u in us]
    first = loso(us, bounds, curves, Z, k)

    def cut(vals):
        v = sorted(x for x in vals if x == x)
        return v[int(math.floor(len(v) * (1 - clean_frac)))] if v else float("inf")
    th = {a: cut([r[f"D_{a}"] for r in first]) for a in SCORE_AXES}
    removed = [r["file_id"] for r in first if any((r[f"D_{a}"] != r[f"D_{a}"]) or (r[f"D_{a}"] >= th[a]) for a in SCORE_AXES)]
    keep = [i for i, r in enumerate(first) if r["file_id"] not in set(removed)]
    us2 = [us[i] for i in keep]
    coef2 = fit_resid(us2)
    zs2 = fit_z(us2, coef2)
    curves2 = curves[keep]
    Z2 = [D.zvec(zs2, feats_of(u, coef2), LAYER_OF) for u in us2]
    second = loso(us2, bounds, curves2, Z2, k)
    return dict(us=us, words=words, bounds=bounds, first=first, th=th, removed=removed, us2=us2, coef2=coef2, zs2=zs2,
                curves2=curves2, Z2=Z2, second=second)


def cross_scale(us: list[dict], key: str, k: int, clean_frac: float, folds: int = SCALE_FOLDS, seed: int = SCALE_SEED) -> tuple[list[dict], dict]:
    """P11 — 화자 K-fold 교차적합 D. 9/10 화자로 §6 참조(청소 포함)를 만들고 1/10 화자를 그 참조에 대고 잰다 (자기 화자는 참조에 없다).
    모은 D 가 「처음 보는 화자」의 거리 분포다 (§17-10). 반환 (발화별 D 목록, 축별 fold 수)."""
    import random
    from scoring.score import Reference, score_extracted
    spk = sorted({u["speaker_id"] for u in us})
    random.Random(seed).shuffle(spk)
    parts = [set(spk[i::folds]) for i in range(folds)]
    rows = []
    for f in parts:
        tr = [u for u in us if u["speaker_id"] not in f]
        te = [u for u in us if u["speaker_id"] in f]
        if not tr or not te:
            continue
        c = _core(tr, key, k, clean_frac)
        body = _body(c, key, "cv", k, clean_frac, scale=None)
        ref = Reference(body)
        for u in te:
            r = score_extracted(u, ref)
            # flat/exag 도 fold 참조(자기 화자 없음)로 잰다 — w·r 과 같은 「처음 보는 화자」 분포 (코덱스 4차 치명 2 · 2026-08-31)
            rows.append({"file_id": u["file_id"], "speaker_id": u["speaker_id"], **{f"D_{a}": r[f"D_{a}"] for a in SCALE_AXES}})
    return rows, {"folds": folds, "seed": seed}


def pos_scale(us: list[dict], c: dict, cv: list[dict]) -> None:
    """P12 4′ **참고값** — 교차적합 발화마다 청소 뒤 최종 참조(`c["us2"]`) 전원에 대고 낸 flat/exag 를 `D_flat_loo`·`D_exag_loo` 로 적는다.
    자기 **발화** 하나만 뺀 값(LOO · `flat_probe.py` 와 같은 정의)이라 같은 화자의 다른 발화가 참조에 남는다 — v3.4 첫 빌드(`2026-08-31b`)는
    이 값으로 눈금을 냈다. 코덱스 4차(치명 2)가 「w·r 과 같은 교차적합이 아니다」라 지적해, 눈금은 `cross_scale` 의 fold 참조 값(`D_flat`)으로
    바꾸고 이 값은 둘의 차이를 보는 참고로만 남긴다 (2026-08-31)."""
    bounds = c["bounds"]
    R = np.array([D.delta_pos(tones(u), bounds) for u in c["us2"]])
    ref_row = {u["file_id"]: i for i, u in enumerate(c["us2"])}
    by_id = {u["file_id"]: u for u in us}
    for r in cv:
        u = by_id[r["file_id"]]
        keep = None
        if r["file_id"] in ref_row:
            keep = np.ones(len(c["us2"]), dtype=bool)
            keep[ref_row[r["file_id"]]] = False
        f, e, _ = D.flat_exag(D.delta_pos(tones(u), bounds), R, keep)
        r["D_flat_loo"], r["D_exag_loo"] = f, e


def _body(c: dict, key: str, stamp: str, k: int, clean_frac: float, scale: dict | None) -> dict:
    us2, coef2, zs2 = c["us2"], c["coef2"], c["zs2"]
    if scale is None:   # 교차적합 안쪽에서 쓰는 임시 참조 — 눈금은 채점에 안 쓰인다 (D 만 본다)
        scale = {a: {"q50": 0.0, "q95": 1.0, "n": 0} for a in SCALE_AXES}
    E = np.array([u["word"]["excursion"] for u in us2], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        exc_range = {"q05": np.nanpercentile(E, 5, axis=0).tolist(), "q95": np.nanpercentile(E, 95, axis=0).tolist()}
    from scoring.score import MIN_SCALE_WIDTH   # 지연 임포트 — score → reference 순환 회피
    params = {"min_scale_width": MIN_SCALE_WIDTH, "scale_source": f"cross-fit {SCALE_FOLDS}-fold q50/q{SCALE_HI} (P11)", "scale_hi": SCALE_HI,
              "k": k, "clean_frac": clean_frac, "score_axes": list(SCORE_AXES), "weights": {"w": 0.5, "r": 0.5},
              "combine": COMBINE, "pos_axes": list(POS_AXES), "flat_cut": [D.FLAT_CUT_LO, D.FLAT_CUT_HI],   # P12
              "flat_min_pos": D.FLAT_MIN_POS, "flat_min_ref": D.FLAT_MIN_REF,
              "exp_axes": list(EXP_AXES), "min_half_frames": 2, "min_scale_syl": D.MIN_SCALE_SYL, "min_delta_pairs": D.MIN_DELTA_PAIRS,
              "drop_mult": [1.0, 3.0], "min_rank_words": D.MIN_RANK_WORDS, "min_rank_syl": D.MIN_RANK_SYL, "tie_st": D.TIE_ST,
              "min_share_d": D.MIN_SHARE_D, "min_share_use": D.MIN_SHARE_USE,
              "z_min_n": D.Z_MIN_N, "z_min_s_frac": D.Z_MIN_S_FRAC, "z_clip": D.Z_CLIP,
              "gate": 0.6, "word_frac": 0.5, "max_blank_word": 0.2, "traj_k": 20}
    return {"script_key": key, "stamp": stamp, "code": git_hash(), "params": params,
            "words": c["words"], "resid_coef": coef2.tolist(), "z": zs2.to_json(), "scale": scale, "exc_range": exc_range,
            "utterances": [{"file_id": u["file_id"], "speaker_id": u["speaker_id"],
                            "collector_id": u.get("collector_id", ""), "curve": u["curve"],
                            "syl": u["syl"], "word": u["word"], "rate_sps": u["rate_sps"],
                            "peak_syl_idx": u["peak_syl_idx"], "present": u["align"]["present"],
                            "syl_miss": u.get("syl_miss")} for u in us2]}


def build(us: list[dict], key: str, stamp: str, k: int = D.K, clean_frac: float = CLEAN_FRAC) -> tuple[dict, dict]:
    us = [u for u in us if not u.get("unscorable") and u["align"]["usable"]]
    c = _core(us, key, k, clean_frac)
    cv, cv_meta = cross_scale(us, key, k, clean_frac)
    pos_scale(us, c, cv)
    # 눈금 — 교차적합 D 의 q50 · q{SCALE_HI}. 키 이름 "q95" 는 서비스 코드 호환을 위해 유지한다 (뜻은 params.scale_hi)
    scale = {a: {"q50": q([r[f"D_{a}"] for r in cv], 50), "q95": q([r[f"D_{a}"] for r in cv], SCALE_HI),
                 "n": sum(r[f"D_{a}"] == r[f"D_{a}"] for r in cv)} for a in SCALE_AXES}
    loso_scale = {a: {"q50": q([r[f"D_{a}"] for r in c["second"]], 50), "q95": q([r[f"D_{a}"] for r in c["second"]], 95)} for a in ALL_AXES}
    body = _body(c, key, stamp, k, clean_frac, scale)
    us2 = c["us2"]; speakers = {u["speaker_id"] for u in us2}
    summary = {"무엇": f"트랙 1 채점 참조 분포 — 문장 {key} (TRACK1_SCORING_SPEC v3.4 §6, 축 ③′ + P12 flat · 눈금 P11 교차적합)", "stamp": stamp,
               "code": body["code"], "params": body["params"],
               "입력": {"usable 발화": len(us), "화자": len({u['speaker_id'] for u in us})},
               "청소": {**{f"임계 D_{a}": c["th"][a] for a in SCORE_AXES}, "뺀 발화": c["removed"]},
               "참조": {"발화": len(us2), "화자": len(speakers), "서비스 가능(화자≥30)": len(speakers) >= MIN_SPEAKERS},
               "눈금": scale, "눈금 출처": {**cv_meta, "발화": len(cv)},
               "눈금(참고 · v3.2 LOSO 2차 q50/q95)": loso_scale,
               "눈금(참고 · flat/exag 발화 LOO · 2026-08-31b 방식)": {a: {"q50": q([r.get(f"D_{a}_loo") for r in cv], 50), "q95": q([r.get(f"D_{a}_loo") for r in cv], SCALE_HI)} for a in POS_AXES},
               "LOSO 1차": c["first"], "LOSO 2차": c["second"], "교차적합 D(눈금)": cv,
               "본체": f"work/scoring/reference_{key.replace('|', '_')}.json"}
    return body, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--keys", default=None)
    ap.add_argument("--k", type=int, default=D.K)
    ap.add_argument("--clean", type=float, default=CLEAN_FRAC)
    ap.add_argument("--out-dir", default=None, help="본체를 쓸 폴더 (기본 work/scoring — 실험은 다른 폴더로 써서 서비스 참조를 덮지 않는다)")
    a = ap.parse_args()
    out_dir = Path(a.out_dir).expanduser() if a.out_dir else DATA / "work/scoring"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(Path(a.jsonl).expanduser(), encoding="utf-8")]
    keys = a.keys.split(",") if a.keys else sorted({r["script_key"] for r in rows})
    (DATA / "work/scoring").mkdir(parents=True, exist_ok=True)
    (REPO / "registry/scoring").mkdir(parents=True, exist_ok=True)
    for key in keys:
        us = [r for r in rows if r["script_key"] == key]
        body, summary = build(us, key, a.stamp, a.k, a.clean)
        kk = key.replace("|", "_")
        (out_dir / f"reference_{kk}.json").write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
        summary["본체"] = str(out_dir / f"reference_{kk}.json")
        (REPO / f"registry/scoring/reference_{kk}_{a.stamp}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        s = summary
        print(f"{key}: usable {s['입력']['usable 발화']} → 참조 {s['참조']['발화']}발화 {s['참조']['화자']}화자 · 청소 {len(s['청소']['뺀 발화'])} · "
              + " · ".join(f"{a} q50 {s['눈금'][a]['q50']:.3f} q95 {s['눈금'][a]['q95']:.3f}" for a in SCALE_AXES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
