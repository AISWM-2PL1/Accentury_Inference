"""KAN-57 서버 벤치 — 이미지 안에서 wav 목록을 한 설정으로 채점하고 단계별 시간·점수·RSS 를 적는다.

설정은 환경변수로 받는다 (컨테이너 실행 시 -e):
  ACCENTURY_MFA_MODE       align | align_one
  ACCENTURY_WHISPER_DTYPE  float32 | bfloat16 | int8

    python src/scoring/bench_server.py --manifest /in/manifest.csv --wav-dir /in --out /out/bench.json [--n N]

manifest.csv: file_id,script_key[,transcript]  — transcript 가 비면 Whisper 를 돌린다 (서비스 경로).
동시 처리량은 이 스크립트가 아니라 컨테이너를 N 개 동시에 띄워 잰다 (서비스도 프로세스 단위다).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def rss_gb() -> dict:
    """ru_maxrss 단위 — Linux KB · Darwin B (코덱스 2026-09-08 #13). 자식(MFA)까지 따로 본다."""
    k = 1024 ** 3 if sys.platform == "darwin" else 1024 ** 2
    return {"self": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / k, 2),
            "children": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / k, 2)}


def pct(xs, q):
    """선형 보간 백분위 (numpy 기본과 같음) — 코덱스 2026-09-08 #6."""
    xs = sorted(xs)
    if not xs:
        return None
    pos = q * (len(xs) - 1); lo = int(pos); hi = min(lo + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--wav-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--no-stt", action="store_true", help="transcript 열을 그대로 쓴다 (Whisper 생략)")
    a = ap.parse_args()

    cfg = {"mfa_mode": os.environ.get("ACCENTURY_MFA_MODE", "align"),
           "whisper_dtype": os.environ.get("ACCENTURY_WHISPER_DTYPE", "float32(default)")}
    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
    if a.n:
        rows = rows[: a.n]
    print(f"설정 {cfg} · 대상 {len(rows)}건", flush=True)

    from scoring.serve import Track1Scorer
    t = time.time()
    sc = Track1Scorer(stt=not a.no_stt, whisper_device="cpu")
    cold = round(time.time() - t, 2)
    print(f"콜드 스타트 {cold}s · RSS {rss_gb()}", flush=True)

    res = []
    for i, r in enumerate(rows, 1):
        wav = a.wav_dir / f"{r['file_id']}.wav"
        tr = (r.get("transcript") or None) if a.no_stt else None
        t = time.time()
        try:
            env = sc.score(str(wav), r["script_key"], transcript=tr, with_feedback=False)
            res.append({"id": r["file_id"], "key": r["script_key"], "status": env.get("status"),
                        "quality": env.get("quality", {}).get("code"), "score": env.get("intonationScore"),
                        "stageMs": env.get("detail", {}).get("stageMs", {}), "totalMs": int((time.time() - t) * 1000),
                        "transcript": env.get("detail", {}).get("transcript")})
        except Exception as e:  # noqa: BLE001
            res.append({"id": r["file_id"], "key": r["script_key"], "status": "EXC", "error": f"{type(e).__name__}: {e}"[:200],
                        "totalMs": int((time.time() - t) * 1000)})
        if i % 10 == 0:
            print(f"  {i}/{len(rows)} · 최근 {res[-1].get('totalMs')}ms · RSS {rss_gb()}", flush=True)

    ok = [r for r in res if r["status"] == "OK"]
    tot = [r["totalMs"] for r in res if "totalMs" in r]
    tot_ok = [r["totalMs"] for r in ok]          # 실패 건(게이트·길이·예외)을 섞지 않은 합계 — 코덱스 2026-09-08 #6
    stages = {}
    for k in ("transcribe", "gate", "align", "f0", "scoring"):
        v = [r["stageMs"].get(k) for r in ok if r.get("stageMs", {}).get(k) is not None]
        if v:
            stages[k] = {"p50": pct(v, .5), "p95": pct(v, .95), "max": max(v)}
    out = {"무엇": "KAN-57 서버 벤치 — 단계별 시간·점수·RSS", "일시": time.strftime("%Y-%m-%d %H:%M"),
           "생성_코드": "src/scoring/bench_server.py", "설정": cfg, "code": os.environ.get("ACCENTURY_CODE_HASH"),
           "대상": len(rows), "OK": len(ok), "실패": len(rows) - len(ok),
           "콜드 스타트 s": cold, "RSS 최대 GB": rss_gb(),
           "합계 ms": {"p50": pct(tot, .5), "p95": pct(tot, .95), "max": max(tot) if tot else None},
           "합계 ms(성공만)": {"p50": pct(tot_ok, .5), "p95": pct(tot_ok, .95), "max": max(tot_ok) if tot_ok else None},
           "단계 ms": stages, "행": res}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "행"}, ensure_ascii=False, indent=1))
    exc = sum(1 for r in res if r["status"] == "EXC")
    return 1 if (exc or not ok) else 0            # 예외·전량 실패는 실패로 — 코덱스 2026-09-08 #8


if __name__ == "__main__":
    sys.exit(main())
