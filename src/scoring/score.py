"""§7~§9 채점 · 피드백 · 세션 합산 · 왕복 검사 (`TRACK1_SCORING_SPEC.md` v3, 축 ③′).

  점수 축   w 어절 안 음절 높이 순서 · r 문장 안 음절 높이 등수 — 각 0.5 (P4 · 사람 결정 2026-08-28 유지, 잠정)
  자리 축   flat 밋밋 · exag 과장 — 자리별 음절 변화량의 경남 백분위 비율 (P12 · §17-17 · `distance.flat_exag`)
  합산      `params.combine` — "min_flat": 점수 = min(0.5w+0.5r, flat) (P12 · 2026-08-31 잠정 적용) · "wr" 또는 없음(옛 참조): 0.5w+0.5r
  눈금 (§7) D ≤ q50 → 100 · q50~q95 → 100→90 · q95~q95+1w → 90→50 · ~q95+3w → 50→0 (w = q95−q50)
  피드백    이웃과 순서가 가장 자주 다른 어절 · 폭이 경남 q05~q95 밖인 어절(평평/과장) · 밋밋/과장 자리(P12) · 곡선이 가장 벌어진 어절 · 이웃 1위 음성
  세션      문장 여러 개 → 어절 수 가중 평균

왕복 검사 (§9): 참조 발화를 서비스 경로로 채점(자기 화자 제외)하면 §6-4 LOSO 2차와 같아야 한다.

사용
  ... --roundtrip 30 --summary registry/scoring/reference_1_5_<stamp>.json
  ... --jsonl A.jsonl --out B.json          # extract 산출물 일괄 채점
  ... --wav X.wav --textgrid X.TextGrid --transcript "..." --script "..."
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scoring import distance as D  # noqa: E402
from scoring.reference import LAYER_OF, feats_of, tones, tones2  # noqa: E402

WEIGHTS = {"w": 0.5, "r": 0.5}       # P4 기본값 — **참조 본체 `params.weights` 가 있으면 그것이 우선** (2026-08-30, 설정값/본체 분리 · KAN-159)
DROP_MULT_50, DROP_MULT_0 = 1.0, 3.0  # P5 기본값 — 참조 `params.drop_mult` 가 우선. 이 두 상수는 참조에 값이 없는 옛 파일을 위한 폴백이다
MIN_SCALE_WIDTH = 1e-6   # q95 − q50 이 이보다 좁으면 눈금이 없는 것으로 본다 — 그 축은 점수를 내지 않는다 (2026-08-29 · 코덱스 2차 중대)
W_STEP_MIN_SCORE = None  # v3.1 의 「계단이 크면 w 를 버린다」 규칙 — v3.2 에서 폐기 (w 를 쌍 합산으로 바꿔 계단이 촘촘해졌다)
FB_WORDS = 3
DATA = Path(os.environ.get("ACCENTURY_DATA", Path.home() / "AccenturyData"))


class Reference:
    def __init__(self, body: dict) -> None:
        self.body = body
        self.key = body["script_key"]
        self.words = body["words"]
        self.bounds = D.word_bounds(self.words)
        self.coef = np.array(body["resid_coef"], dtype=float)
        self.zs = D.ZStats.from_json(body["z"])
        self.scale = body["scale"]
        self.exc_lo = np.array(body["exc_range"]["q05"], dtype=float)
        self.exc_hi = np.array(body["exc_range"]["q95"], dtype=float)
        self.k = body["params"]["k"]
        prm = body.get("params", {})
        self.weights = dict(prm.get("weights") or WEIGHTS)                 # P4 — 참조가 정본
        dm = prm.get("drop_mult") or [DROP_MULT_50, DROP_MULT_0]           # P5
        self.drop = (float(dm[0]), float(dm[1]))
        self.score_version = f"{body.get('stamp', '?')}+{body.get('code', '?')}"   # KAN-137 scoreVersion 후보 — 참조 stamp + 코드 해시
        self.us = body["utterances"]
        self.T = [tones(u) for u in self.us]
        self.curves = np.array([u["curve"] for u in self.us], dtype=float)
        self.Z = [D.zvec(self.zs, feats_of(u, self.coef), LAYER_OF) for u in self.us]
        # 실험 축 (2026-08-30): w2 앞반|뒤반 2점 순서 · r2 문장 안 2점 등수 · m 퍼짐 정규화 변화량. 옛 참조(칸·눈금 없음)면 nan 으로 흐른다
        self.bounds2 = D.expand_bounds(self.bounds, 2)
        self.T2 = [tones2(u) for u in self.us]
        self.S = [D.tone_scale(t) for t in self.T]
        # 자리 축 (P12 · 2026-08-31): 참조 전원의 자리별 Δ 행렬. 합산 규칙·컷은 참조 params 가 정본 — 옛 참조(칸 없음)면 "wr" 로 현행 점수 그대로
        self.R = np.array([D.delta_pos(t, self.bounds) for t in self.T])
        self.speakers = np.array([u["speaker_id"] for u in self.us])
        self.combine = prm.get("combine", "wr")
        fc = prm.get("flat_cut") or [D.FLAT_CUT_LO, D.FLAT_CUT_HI]
        self.flat_kw = dict(cut_lo=float(fc[0]), cut_hi=float(fc[1]), min_pos=int(prm.get("flat_min_pos", D.FLAT_MIN_POS)),
                            min_ref=int(prm.get("flat_min_ref", D.FLAT_MIN_REF)),
                            max_abs=prm.get("flat_max_abs"), mode=prm.get("flat_mode", "count"))   # 실험 변형 (2026-08-31 데모 후속) — 참조에 없으면 현행

    @classmethod
    def load(cls, path: Path) -> "Reference":
        return cls(json.load(open(path, encoding="utf-8")))


def scale_score(d: float, q50: float, q95: float, m50: float = DROP_MULT_50, m0: float = DROP_MULT_0) -> float:
    """§7. `q95` 는 키 이름일 뿐 v3.3 부터 값은 교차적합 q90 이다 (`params.scale_hi`). m50·m0 = P5 하강 배수."""
    if d != d or q50 != q50 or q95 != q95 or (q95 - q50) < MIN_SCALE_WIDTH:
        return float("nan")          # 눈금 붕괴(q95 = q50)를 1e-9 로 때우면 계단 함수가 된다
    w = q95 - q50
    if d <= q50:
        return 100.0
    if d <= q95:
        return 100.0 - 10.0 * (d - q50) / w
    if d <= q95 + m50 * w:
        return 90.0 - 40.0 * (d - q95) / (m50 * w)
    if d <= q95 + m0 * w:
        return 50.0 - 50.0 * (d - q95 - m50 * w) / ((m0 - m50) * w)
    return 0.0


def score_extracted(u: dict, ref: Reference, exclude_speaker: str | None = None) -> dict:
    """extract_utterance 산출 u 를 참조에 대고 채점. exclude_speaker 는 왕복 검사용."""
    if u.get("unscorable"):
        return {"file_id": u.get("file_id"), "unscorable": u["unscorable"]}
    tu = tones(u)
    tu2 = tones2(u); su = D.tone_scale(tu)
    cu = np.array(u["curve"], dtype=float)
    zu = D.zvec(ref.zs, feats_of(u, ref.coef), LAYER_OF)
    c = {a: [] for a in ("w", "r", "c", "f", "w2", "r2", "m")}
    for v, tv, cv, zv, tv2, sv in zip(ref.us, ref.T, ref.curves, ref.Z, ref.T2, ref.S):
        if exclude_speaker and v["speaker_id"] == exclude_speaker:
            continue
        dw, _ = D.d_word_rank(tu, tv, ref.bounds); c["w"].append((v["speaker_id"], dw, 1.0))
        dr, _ = D.d_sent_rank(tu, tv); c["r"].append((v["speaker_id"], dr, 1.0))
        dc, sc = D.d_curve(cu, cv); c["c"].append((v["speaker_id"], dc, sc))
        df, sf = D.d_feat(zu, zv); c["f"].append((v["speaker_id"], df, sf))
        dw2 = D.d_word_rank(tu2, tv2, ref.bounds2)[0] if (tu2 is not None and tv2 is not None) else float("nan")
        c["w2"].append((v["speaker_id"], dw2, 1.0))
        dr2 = D.d_sent_rank(tu2, tv2, min_n=2 * D.MIN_RANK_SYL)[0] if (tu2 is not None and tv2 is not None) else float("nan")
        c["r2"].append((v["speaker_id"], dr2, 1.0))
        dm, _ = D.d_delta_norm(tu, tv, ref.bounds, su, sv); c["m"].append((v["speaker_id"], dm, 1.0))
    Dv, nb = {}, {}
    for a in c:
        Dv[a], nbs = D.knn_speaker(c[a], ref.k, min_share=0.0 if a in ref.weights else D.MIN_SHARE_USE)
        nb[a] = [s for s, _ in nbs]
    # 자리 축 (P12): 이웃이 아니라 참조 전원 대비. exclude_speaker 는 왕복·LOSO 용
    keep = None if not exclude_speaker else (ref.speakers != exclude_speaker)
    Dv["flat"], Dv["exag"], pct = D.flat_exag(D.delta_pos(tu, ref.bounds), ref.R, keep, **ref.flat_kw)
    W = ref.weights
    nan_scale = {"q50": float("nan"), "q95": float("nan")}     # 옛 참조에 없는 축 → 점수 nan
    sc = {a: scale_score(Dv[a], ref.scale.get(a, nan_scale)["q50"], ref.scale.get(a, nan_scale)["q95"], *ref.drop) for a in Dv}
    # v3.2 (2026-08-29): w 를 쌍 합산으로 바꿔 계단이 촘촘해졌으므로 v3.1 의 「계단이 크면 w 를 버린다」 규칙은 뺐다.
    # n2_eff 는 기록만 한다 (계단 크기를 사후에 볼 수 있게).
    n2e = D.n2_effective(tu, ref.bounds)
    ok = [a for a in W if sc[a] == sc[a]]
    if len(ok) == len(W):
        base, partial = sum(W[a] * sc[a] for a in W), None
    elif ok:                                   # 한 축을 못 재면(결측 많은 발화) 남은 축만으로 점수 + 표시 (2026-08-28)
        base, partial = float(np.mean([sc[a] for a in ok])), "+".join(a for a in W if a not in ok) + " 없음"
    else:
        base, partial = float("nan"), "w+r 없음"
    # P12 합산 (2026-08-31): min(현행, flat). flat 을 못 재면(자리 <min_pos) 현행 그대로 — 그 사실은 `partial` 에도 남긴다 (§7 「일부 축만
    # 측정」 표시 · 코덱스 4차 중대 1). 옛 참조(combine="wr")는 flat 을 점수·피드백 어디에도 안 쓴다 — D_flat 은 기록만.
    capped = None
    total = base
    if ref.combine == "min_flat":
        if sc["flat"] == sc["flat"] and base == base:
            if sc["flat"] < base:
                total, capped = sc["flat"], "flat"
        elif sc["flat"] != sc["flat"] and base == base:
            capped = "flat 없음"
            partial = (partial + "+flat 없음") if partial else "flat 없음"
    return {"file_id": u.get("file_id"), "script_key": ref.key, "n_words": len(ref.words), "partial": partial,
            "n2_eff": n2e, "score_version": ref.score_version,
            "params": {"weights": W, "drop_mult": list(ref.drop), "combine": ref.combine},
            **{f"D_{a}": Dv[a] for a in Dv}, **{f"score_{a}": sc[a] for a in Dv}, "score_wr": base, "score": total, "capped": capped,
            "nb_w": nb["w"], "nb_r": nb["r"],
            "feedback": feedback(u, tu, cu, ref, nb["w"], nb["c"], pct if ref.combine == "min_flat" else None)}


def feedback(u: dict, tu: np.ndarray, cu: np.ndarray, ref: Reference, nb_w: list[str], nb_c: list[str], pct: np.ndarray | None = None) -> dict:
    """§8 — 순서가 이웃과 자주 다른 어절 · 폭 평평/과장 · 밋밋/과장 자리(P12) · 곡선이 가장 벌어진 어절 · 이웃 1위."""
    fb: dict = {"order": [], "excursion": [], "flat_pos": [], "exag_pos": [], "curve_word": None, "nearest": None, "rate_sps": u.get("rate_sps")}
    # P12 자리 피드백 (2026-08-31): 자리 i = 어절 안 음절 i→i+1. 백분위 < cut_lo 밋밋(경남보다 덜 움직임) · > cut_hi 과장.
    # 자리는 밀릴 수 있다 (§17-6) — 어절 단위로 읽는다
    if pct is not None:
        for j, (a, e) in enumerate(ref.bounds):
            syl = ref.words[j]
            for i in range(a, e - 1):
                p = pct[i]
                if p != p:
                    continue
                row = {"word": ref.words[j], "idx": j, "pair": f"{syl[i - a]}→{syl[i - a + 1]}", "pct": round(float(p), 1)}
                if p < ref.flat_kw["cut_lo"]:
                    fb["flat_pos"].append(row)
                elif p > ref.flat_kw["cut_hi"]:
                    fb["exag_pos"].append(row)
    idx = [i for i, v in enumerate(ref.us) if v["speaker_id"] in set(nb_w)]
    rows = []
    for j, (a, e) in enumerate(ref.bounds):
        if e - a < 2 or np.isnan(tu[a:e]).any():
            continue
        mine = tuple(np.argsort(np.argsort(-tu[a:e])).tolist())
        ref_o = [tuple(np.argsort(np.argsort(-ref.T[i][a:e])).tolist()) for i in idx if not np.isnan(ref.T[i][a:e]).any()]
        if ref_o:
            rows.append((sum(o != mine for o in ref_o) / len(ref_o), j))
    rows.sort(reverse=True)
    fb["order"] = [{"word": ref.words[j], "idx": j, "mismatch": round(r, 2)} for r, j in rows[:FB_WORDS] if r >= 0.5]
    # 방향 피드백 (2026-08-30): 순서가 다른 어절마다 「경남 이웃은 어느 음절을 가장 높이나」와 음절별 반음 차.
    # 반음은 발화 중앙값 기준(§3)이라 「자기 평균보다 높게/낮게」로 읽는다. 자리는 밀릴 수 있다 (§17-6) — 어절 안 상대 순서만 믿는다.
    fb["order_detail"] = []
    for o in fb["order"]:
        j = o["idx"]; a, e = ref.bounds[j]; syl = list(ref.words[j])
        mine_t = tu[a:e]
        ref_T = np.array([ref.T[i][a:e] for i in idx if not np.isnan(ref.T[i][a:e]).any()])
        if ref_T.size == 0:
            continue
        from collections import Counter
        modal = Counter(tuple(np.argsort(np.argsort(-t)).tolist()) for t in ref_T).most_common(1)[0][0]
        mine_rank = tuple(np.argsort(np.argsort(-mine_t)).tolist())
        def order_str(rank):   # rank[i] = i번째 음절의 등수(0 = 가장 높음)
            return ">".join(syl[i] for i in sorted(range(len(rank)), key=lambda i: rank[i]))
        diff = (mine_t - np.median(ref_T, axis=0)).round(1)
        top_ref = int(np.argmin(modal)); top_me = int(np.argmin(mine_rank))
        same = mine_rank == modal
        o["same_as_modal"] = same           # 사용자 순서 = 경남 최빈 순서 — 「다르다」가 아니라 경남끼리 갈리는 어절 (2026-08-30)
        fb["order_detail"].append({
            "word": ref.words[j], "mine": order_str(mine_rank), "gyeongnam": order_str(modal), "same_as_modal": same,
            "raise": syl[top_ref] if top_ref != top_me else None,           # 경남은 이 음절을 가장 높인다
            "lower": syl[top_me] if top_ref != top_me else None,            # 사용자는 이 음절을 가장 높였다
            "syl_diff_st": {s: float(d) for s, d in zip(syl, diff)},        # 음절별 (사용자 − 이웃 중앙) 반음
            "ref_agree": round(sum(1 for t in ref_T if tuple(np.argsort(np.argsort(-t)).tolist()) == modal) / len(ref_T), 2)})
    exc = np.array(u["word"]["excursion"], dtype=float)
    for j, w in enumerate(ref.words):
        if exc[j] == exc[j] and exc[j] < ref.exc_lo[j]:
            fb["excursion"].append({"word": w, "idx": j, "kind": "평평", "st": round(float(exc[j]), 1)})
        elif exc[j] == exc[j] and exc[j] > ref.exc_hi[j]:
            fb["excursion"].append({"word": w, "idx": j, "kind": "과장", "st": round(float(exc[j]), 1)})
    if nb_c:
        ci = [i for i, v in enumerate(ref.us) if v["speaker_id"] in set(nb_c)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            medc = np.nanmedian(ref.curves[ci], axis=0)
            k = 20
            gaps = [float(np.nanmean(np.abs(cu[j * k:(j + 1) * k] - medc[j * k:(j + 1) * k]))) for j in range(len(ref.words))]
        if any(g == g for g in gaps):
            jb = int(np.nanargmax(gaps))
            fb["curve_word"] = {"word": ref.words[jb], "idx": jb, "gap_st": round(gaps[jb], 2)}
    if nb_w:
        fb["nearest"] = next((v["file_id"] for v in ref.us if v["speaker_id"] == nb_w[0]), None)
    return fb


def score_session(results: list[dict]) -> dict:
    """문장 여러 개 → 어절 수 가중 평균 (축별·최종). 채점 불가 문장은 뺀다."""
    ok = [r for r in results if r.get("score") == r.get("score")]
    if not ok:
        return {"n_sentences": 0, "score": float("nan")}
    wts = np.array([r["n_words"] for r in ok], dtype=float)
    out = {"n_sentences": len(ok), "sentences": [r["script_key"] for r in ok]}
    for a in list(WEIGHTS) + ["c", "f", "flat", "exag", "wr"]:
        v = np.array([r.get(f"score_{a}", float("nan")) for r in ok], dtype=float)
        m = ~np.isnan(v)
        out[f"score_{a}"] = float(np.average(v[m], weights=wts[m])) if m.any() else float("nan")
    out["score"] = float(np.average([r["score"] for r in ok], weights=wts))
    # 합산 규칙·부분 측정·참조 버전을 잃지 않는다 (코덱스 4차 중대 5 · 2026-08-31): 문장마다 다른 규칙·버전이 섞이면 여기서 보인다
    out["combine"] = sorted({(r.get("params") or {}).get("combine", "wr") for r in ok})
    out["score_version"] = sorted({r.get("score_version", "?") for r in ok})
    out["capped_flat"] = sum(1 for r in ok if r.get("capped") == "flat")
    out["partial"] = [r["script_key"] + ":" + r["partial"] for r in ok if r.get("partial")]
    out["mixed"] = len(out["combine"]) > 1 or len(out["score_version"]) > 1
    return out


def roundtrip(ref: Reference, summary_path: Path, n: int) -> dict:
    summ = json.load(open(summary_path, encoding="utf-8"))
    stored = {r["file_id"]: r for r in summ["LOSO 2차"]}
    diffs = {a: [] for a in ("w", "r", "c", "f")}
    for u in ref.us[:n]:
        r = score_extracted(u, ref, exclude_speaker=u["speaker_id"])
        s = stored[u["file_id"]]
        for a in diffs:
            x, y = r[f"D_{a}"], s[f"D_{a}"]
            diffs[a].append(0.0 if (x != x and y != y) else abs(x - y))
    return {"n": n, **{f"max|ΔD_{a}|": max(v) for a, v in diffs.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--roundtrip", type=int, default=0)
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--wav"); ap.add_argument("--textgrid"); ap.add_argument("--transcript"); ap.add_argument("--script")
    a = ap.parse_args()
    ref = Reference.load(Path(a.ref).expanduser())
    if a.roundtrip:
        print(json.dumps(roundtrip(ref, Path(a.summary), a.roundtrip), ensure_ascii=False))
        return 0
    if a.jsonl:
        rows = [json.loads(l) for l in open(Path(a.jsonl).expanduser(), encoding="utf-8")]
        res = [score_extracted(u, ref) for u in rows if u["script_key"] == ref.key]
        if a.out:
            Path(a.out).expanduser().write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        ok = [r for r in res if "score" in r and r["score"] == r["score"]]
        print(json.dumps({"n": len(res), "scored": len(ok),
                          **{f"D_{x} 중앙": (float(np.median([r[f"D_{x}"] for r in ok])) if ok else None) for x in "wr"},
                          "score 중앙": float(np.median([r["score"] for r in ok])) if ok else None}, ensure_ascii=False))
        return 0
    if a.wav:
        import soundfile as sf
        from pipeline.features import read_textgrid
        from scoring.extract import extract_utterance
        x, sr = sf.read(a.wav)
        if x.ndim > 1:
            x = x.mean(axis=1)
        u = extract_utterance(x, sr, read_textgrid(a.textgrid), a.transcript, a.script, utt_id=Path(a.wav).stem)
        print(json.dumps(score_extracted(u, ref), ensure_ascii=False, indent=1))
        return 0
    ap.error("--roundtrip · --jsonl · --wav 중 하나")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
