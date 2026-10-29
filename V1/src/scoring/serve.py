"""트랙 1 채점 — 백엔드 전달본 진입점 (KAN-159 · 2026-08-30).

WAV 하나 + 문장 ID → 점수 JSON 하나. 서비스 경로 §9 를 함수 하나로 묶는다:
  WAV → (Whisper) → §4 게이트 → MFA → 추출 → 참조 거리 → §7 점수 + §8 피드백 → **팀 AI 서버 계약 형식**

계약은 팀 리포 `ai/app/engine.py` (KAN-135) 의 `AnalysisOutcome` 이다 — 성공: `status OK` · `intonationScore` 정수 0~100 ·
`confidence` · `quality.code` · `segments`; 실패: `status FAILED` · `quality.code` ∈ {AUDIO_TOO_QUIET, AUDIO_TOO_LONG,
AUDIO_FORMAT_UNSUPPORTED, ANALYSIS_MISREAD} · `retryable` bool. `modelVersion` 은 엔진이 보고, `scoreVersion`·`processingMs` 는 라우트가 채운다.
**축·계산은 `score.py` 그대로이고 여기서는 이름만 바꿔 담는다.**

사용
  .venv/bin/python src/scoring/serve.py --wav X.wav --script-key "1|5"            # Whisper 포함
  .venv/bin/python src/scoring/serve.py --wav X.wav --script-key "1|5" --no-stt --transcript "…"
  .venv/bin/python src/scoring/serve.py --wav X.wav --script-key "1|5" --bench 5   # 단계별 시간 (KAN-57 입력)

팀 AI 서버에 붙이는 법 (`ai/app/engine.py` 옆에 어댑터 하나):
  from scoring.serve import Track1Scorer
  scorer = Track1Scorer(ref_dir=..., sentences=...)          # 기동 시 1회
  class Track1Engine:                                          # AnalysisEngine 프로토콜 — 구조만 맞추면 된다
      model_version = scorer.model_version
      async def analyze(self, request):
          env = await asyncio.to_thread(scorer.score, request.audio_path, request.meta.get("scriptKey") or request.meta.get("itemId"))
          return AnalysisOutcome.ok(env["intonationScore"], env["confidence"], env["quality"]["code"], env["segments"]) if env["status"] == "OK" \\
              else AnalysisOutcome.failure(env["quality"]["code"], env["retryable"])
  ⚠️ `to_thread` 는 취소가 안 닿는다 (엔진 계약 2) — 서브프로세스 실행으로 바꾸는 것은 KAN-22 몫.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import select
import shutil
import signal
import statistics as st
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scoring.align import align_transcript  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("ACCENTURY_DATA", Path.home() / "AccenturyData"))
SR = 16000
MAX_SECONDS = 10.0                       # KAN-22 입력 상한
MFA_BIN = os.environ.get("ACCENTURY_MFA_BIN", "/opt/homebrew/Caskroom/miniforge/base/envs/mfa311/bin")
MFA_DICT = Path(os.environ.get("ACCENTURY_MFA_DICT", DATA / "work/align/bench/dict_aug_track1.txt"))
MFA_ACOUSTIC = "korean_mfa"
DEFAULT_REF_DIR = DATA / "work/s4/ref"
DEFAULT_SENTENCES = REPO / "registry/scoring/handoff_sentences_2026-09-01.json"   # v3.4 참조 31b + 09-01 문장 목록 157 (사람 선별·대본 수정 · handoff_pack.py --from-list)
QUALITY = {"gate": "ANALYSIS_MISREAD", "blank": "ANALYSIS_MISREAD", "mfa": "ANALYSIS_MISREAD",
           "f0": "AUDIO_TOO_QUIET", "long": "AUDIO_TOO_LONG", "format": "AUDIO_FORMAT_UNSUPPORTED"}


def git_hash() -> str:
    """코드 해시 — 컨테이너에는 .git 이 없으므로 빌드 때 넣은 ACCENTURY_CODE_HASH 를 먼저 본다."""
    env = os.environ.get("ACCENTURY_CODE_HASH")
    if env:
        return env
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "?"


def _mfa_env() -> dict:
    return {**os.environ, "PATH": f"{MFA_BIN}:{os.environ.get('PATH', '')}"}


def _align_corpus(lab: Path, aligned: Path) -> tuple[Path | None, str]:
    """발화 하나짜리 코퍼스로 `mfa align` — 옛 경로. 약 13~15 s (실측 2026-09-04: 14.78·13.38·14.46)."""
    r = subprocess.run([f"{MFA_BIN}/mfa", "align", "--clean", "--overwrite", "--quiet",
                        str(lab), str(MFA_DICT), MFA_ACOUSTIC, str(aligned)],
                       env=_mfa_env(), capture_output=True, text=True)
    tg = list(aligned.rglob("*.TextGrid"))
    return (tg[0] if r.returncode == 0 and tg else None), ((r.stderr or r.stdout) or "")[-400:]


def _align_one(lab: Path, aligned: Path) -> tuple[Path | None, str]:
    """`mfa align_one` + 라벨 복원 (`mfa_one.py`) — 약 2.3~2.6 s. 산출물은 옛 경로와 동일하다 (§KAN-57)."""
    aligned.mkdir(parents=True, exist_ok=True)
    out = aligned / "u.TextGrid"
    r = subprocess.run([f"{MFA_BIN}/python", str(Path(__file__).with_name("mfa_one.py")),
                        str(lab / "u.wav"), str(lab / "u.txt"), str(MFA_DICT), MFA_ACOUSTIC, str(out)],
                       env=_mfa_env(), capture_output=True, text=True)
    return (out if r.returncode == 0 and out.exists() else None), ((r.stderr or r.stdout) or "")[-400:]


class _ResidentMFA:
    """`mfa_one.py --serve` 를 한 번 띄워 두고 발화마다 한 줄 JSON 으로 정렬을 맡긴다 (KAN-296).

    요청마다 MFA 파이썬을 새로 띄우던 `_align_one` 과 같은 함수를 같은 순서로 부르고, 파이썬 기동과 MFA import,
    토크나이저 생성만 아낀다. 정렬은 상주 프로세스가 요청마다 fork 한 자식에서 한다. 한 프로세스에서 연달아
    정렬하면 둘째 건부터 결과가 달라지기 때문이다 (`mfa_one.py` 상주 모드 설명). 상주 메모리는 약 0.6 GB 다
    (KAN-296 사전 측정, 요청마다 뜨던 프로세스와 같은 크기).

    실패하면 그 요청은 호출한 쪽이 한 번짜리 `_align_one` 으로 다시 간다. 그래서 상주 프로세스가 죽거나
    멈춰도 채점은 이어진다. 다음 요청에서 새로 띄운다.

    - 응답이 `ACCENTURY_MFA_RESIDENT_TIMEOUT` 초(기본 30) 안에 없으면 죽이고 실패로 돌린다.
    - `ACCENTURY_MFA_RESIDENT_RECYCLE` 건(기본 200)마다 새로 띄운다. 15 건에서 건당 약 2 MB 씩 늘다 둔화했는데
      멈추는지는 확인하지 못했다 (KAN-296 사전 측정). 다시 띄우는 비용은 오디오 예열(`mfa_one._prewarm_audio`)과 발음 사전 FST
      (`mfa_one._prebuild_lexicon`)를 더한 뒤 staging 에서 요청 하나가 약 5.9 초 길어질 만큼이었다 (2026-10-28, 같은 음성 비교).
      그래서 마지막 건을 답한 직전 프로세스를 바로 닫고 새 프로세스는 뒤에서 띄운다(`_prepare_next`). 다음 요청은 정렬
      직전에 남은 기동 시간만 기다리는데, 그 앞의 Whisper 전사(약 6 초)가 대개 그보다 길다. 두 프로세스가 함께 떠 있는
      순간은 없다.
    - 워커와 같은 프로세스 그룹에 남긴다. 어댑터가 취소 때 그룹째 죽이면 함께 거둬진다.
    """

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.jobs = 0
        self._next: tuple[threading.Thread, dict] | None = None
        self.timeout = float(os.environ.get("ACCENTURY_MFA_RESIDENT_TIMEOUT", "30"))
        self.recycle = int(os.environ.get("ACCENTURY_MFA_RESIDENT_RECYCLE", "200"))

    def start(self) -> None:
        self._take_next()
        self.close()
        self.proc = self._spawn()
        self.jobs = 0

    def _spawn(self) -> subprocess.Popen:
        """새 상주 프로세스를 띄우고 기동(예열 포함)이 끝날 때까지 기다린다. self 는 건드리지 않는다."""
        p = subprocess.Popen([f"{MFA_BIN}/python", str(Path(__file__).with_name("mfa_one.py")), "--serve"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             env=_mfa_env(), text=True, encoding="utf-8", bufsize=1)
        try:
            res = self._ask({"op": "ping", "dic": str(MFA_DICT), "acoustic": MFA_ACOUSTIC}, max(self.timeout, 60.0), p)
            if not res.get("ok"):
                raise RuntimeError(f"상주 MFA 기동 실패: {res.get('error', '')}")
        except BaseException:
            _close_proc(p)
            raise
        return p

    def _prepare_next(self) -> None:
        """다음 상주 프로세스를 뒤에서 띄운다. 결과는 `_take_next` 가 거둔다."""
        box: dict = {}

        def run() -> None:
            try:
                box["proc"] = self._spawn()
            except Exception as e:  # noqa: BLE001 - 실패하면 다음 요청이 직접 띄운다
                box["error"] = e

        t = threading.Thread(target=run, name="mfa-resident-recycle", daemon=True)
        t.start()
        self._next = (t, box)

    def _take_next(self) -> None:
        """뒤에서 띄우던 프로세스가 있으면 끝날 때까지 기다려 지금 프로세스로 삼는다. 실패했으면 버린다."""
        if self._next is None:
            return
        t, box = self._next
        self._next = None
        t.join()
        p = box.get("proc")
        if p is not None:
            self.close()
            self.proc, self.jobs = p, 0

    def _ask(self, msg: dict, timeout: float, proc: subprocess.Popen | None = None) -> dict:
        p = proc or self.proc
        assert p and p.stdin and p.stdout
        p.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        p.stdin.flush()
        ready, _, _ = select.select([p.stdout], [], [], timeout)
        if not ready:
            raise TimeoutError(f"상주 MFA 가 {timeout:.0f} 초 안에 답하지 않았다")
        line = p.stdout.readline()
        if not line:
            raise EOFError("상주 MFA 가 응답 없이 끝났다")
        return json.loads(line)

    def align(self, lab: Path, aligned: Path) -> tuple[Path | None, str, bool]:
        """(TextGrid 또는 None, 실패 사유, 상주 프로세스가 답했는가).

        답했는데 실패했으면 한 번짜리 `align_one` 도 같은 입력에서 같은 이유로 실패하므로 호출한 쪽은
        그것을 건너뛰고 `align` 으로 간다 (현행 `align_one` 실패와 같은 길). 답하지 못했으면 한 번짜리로 간다.
        """
        aligned.mkdir(parents=True, exist_ok=True)
        out = aligned / "u.TextGrid"
        try:
            self._take_next()
            if self.proc is None or self.proc.poll() is not None or self.jobs >= self.recycle:
                self.start()
            res = self._ask({"op": "align", "wav": str(lab / "u.wav"), "txt": str(lab / "u.txt"),
                             "dic": str(MFA_DICT), "acoustic": MFA_ACOUSTIC, "out": str(out)}, self.timeout)
            self.jobs += 1
        except Exception as e:  # noqa: BLE001 - 어떤 실패든 이 프로세스를 버리고 호출한 쪽이 한 번짜리로 간다
            self.close()
            return None, f"{type(e).__name__}: {e}"[-400:], False
        if self.jobs >= self.recycle:        # 이번 건의 TextGrid 는 이미 나왔다. 다음 요청을 기다리게 하지 않도록 지금 갈아 둔다
            self.close()
            self._prepare_next()
        return (out if res.get("ok") and out.exists() else None), res.get("error", ""), True

    def close(self) -> None:
        """기다리지 않고 바로 죽인다. 정렬 중이던 자손까지 먼저 죽인다.

        `mfa_one.py` 의 PR_SET_PDEATHSIG 는 직접 fork 한 자식에만 걸리고, 그 자식이 띄운 Kaldi 프로세스에는 이어지지
        않는다. 그것들이 남으면 호출한 쪽의 한 번짜리 대체 경로와 동시에 돌고, 어댑터가 작업 폴더를 지운 뒤에도 파일을
        건드린다 (Codex 리뷰 P2). 그래서 자손을 먼저 멈춰 더 퍼지지 못하게 한 뒤 모두 죽이고 나서 상주 프로세스를 죽인다.
        """
        p, self.proc = self.proc, None
        if p is not None:
            _close_proc(p)

    def shutdown(self) -> None:
        """뒤에서 띄우던 프로세스까지 거둬 닫는다 (종료 때)."""
        self._take_next()
        self.close()


def _close_proc(p: subprocess.Popen) -> None:
    _kill_descendants(p.pid)
    p.kill()
    p.wait()
    for f in (p.stdin, p.stdout):
        try:
            f and f.close()
        except OSError:
            pass


def _descendants(root: int) -> set[int]:
    """`/proc` 의 부모 관계로 root 의 자손 pid 를 모은다. `/proc` 가 없으면(개발 맥) 빈 집합이다."""
    children: dict[int, list[int]] = {}
    for d in os.listdir("/proc") if os.path.isdir("/proc") else []:
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat", encoding="utf-8", errors="replace") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])   # comm 에 공백이나 괄호가 있어도 마지막 ")" 뒤가 필드다
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(d))
    found, stack = set(), [root]
    while stack:
        for c in children.get(stack.pop(), []):
            if c not in found:
                found.add(c)
                stack.append(c)
    return found


def _kill_descendants(root: int) -> None:
    """root 의 자손을 SIGSTOP 으로 얼린 뒤(더 이상 fork 하지 못하게) 다시 모아 SIGKILL 한다."""
    stopped: set[int] = set()
    for _ in range(5):                       # 얼리는 사이에 새로 뜬 자손까지 잡는다
        new = _descendants(root) - stopped
        if not new:
            break
        for pid in new:
            try:
                os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError:
                pass
        stopped |= new
    for pid in stopped:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


_RESIDENT = _ResidentMFA()
atexit.register(_RESIDENT.shutdown)
MFA_MODES = ("resident", "align_one", "align")


def _mfa_mode() -> str:
    mode = os.environ.get("ACCENTURY_MFA_MODE", "resident")     # 기본 resident (KAN-296). 되돌리기: ACCENTURY_MFA_MODE=align_one
    if mode not in MFA_MODES:        # 오타나 대소문자가 새 경로를 켜지 않게 (코덱스 2026-09-08 #3)
        raise ValueError(f"ACCENTURY_MFA_MODE 는 {', '.join(MFA_MODES)} 중 하나여야 한다. 받은 값 {mode!r}")
    return mode


def mfa_align_single(wav: Path, transcript: str, workdir: Path) -> Path | None:
    """발화 하나를 MFA 로 정렬한다. TextGrid 경로 또는 None.

    **KAN-296 이후 기본은 `resident`** 다. `align_one` 과 같은 함수를 상주 프로세스(`_ResidentMFA`)에서 부른다.
    상주 프로세스가 답하지 못하면 그 요청은 한 번짜리 `align_one` 으로, 정렬이 실패하면 `align` 으로 간다.
    되돌리기는 `ACCENTURY_MFA_MODE=align_one`(KAN-296 이전 동작). 아래는 KAN-57 때의 설명이다.

    **기본은 `align_one`** (KAN-57 채택, 2026-09-08 — 회귀 811 건에서 지역별 분포가 `align` 과 같았다,
    `docs/KAN57_RESULTS_2026-09-07.md` §4). `align_one` 이 실패하면 `align` 으로 한 번 더 간다(그 요청만 17 s).
    되돌리기는 `ACCENTURY_MFA_MODE=align`. 그 외 값은 기동 실패.

    ## 두 경로 대조 실측 (개발 맥 · x86 실측 아님)

    | 무엇 | 값 | 출처 (registry/scoring/) |
    | --- | --- | --- |
    | 지연 | 13.5 s → 2.6 s (5.2 배) | `mfa_one_score_2026-09-04.json` |
    | 정렬 라벨 | 30/30 동일 (칸 수도 동일) | `mfa_one_delta_2026-09-07.json` |
    | 정렬 경계 | 칸(interval) 기준 words 6/293(2.0%)·phones 16/901(1.8%) 이동, 최대 **60 ms** · 23/30 발화는 경계까지 동일 | 〃 (09-08 칸 단위로 재생성) |
    | 최종 점수 (정수) | 70 건 중 67 동일 · 3 건 +1·+1·+5 | `mfa_one_bias_2026-09-07.json` |
    | 최종 점수 (실수 축) | 10 건 이동 · **9 올라감 : 1 내려감** · 부호검정 p=0.0215 | 〃 |
    | 순 이동 | +2.60 점 → 발화당 +0.037 점 | 〃 |

    위 「9 올라감 : 1 내려감, p=0.0215」는 중간 축 `scoreWR` 의 검정이다. 최종 점수는 `min(scoreWR, scoreFlat)`
    이라 최종 정수의 검정은 3:0, p=0.25 로 편향을 말할 수 없고, 회귀 811 건(`kan57_c7i_2026-09-07/reg_align*`)에서는
    39↑ 65↓ 로 방향이 반대다 — **「위로 민다」는 성립하지 않는다** (코덱스 2026-09-08 #2). 대조군(`align` 두 번)은
    70/70 소수점까지 동일해 서비스 경로는 결정적이다. 811 건 집단 분포(중앙·사분위·<90 비율)는 두 경로가 같다.

    2026-09-04 판 주석에서 셋을 고쳤다. 「경계 10 ms(1 프레임) 차」는 과소평가였고(실측 최대 60 ms),
    「참조 제작과 서비스가 이미 최대 230 ms 어긋난다」는 근거 파일을 찾지 못해 뺐으며, 「등급 변화
    없음」도 뺐다 — 점수를 등급으로 바꾸는 선은 이 repo 에 없다(`intonationScore` 정수 하나뿐).
    """
    lab, aligned = workdir / "lab", workdir / "aligned"
    lab.mkdir(parents=True, exist_ok=True)
    shutil.copy(wav, lab / "u.wav"); (lab / "u.txt").write_text(transcript, encoding="utf-8")

    mode = _mfa_mode()
    errs = []
    one_shot = mode == "align_one"
    if mode == "resident":           # KAN-296. 상주 프로세스가 답하지 못했을 때만 한 번짜리 align_one 으로 이어간다
        tg, err, answered = _RESIDENT.align(lab, aligned / "resident")
        if tg:
            return tg
        errs.append(f"resident: {err}")
        one_shot = not answered
    if one_shot:
        tg, err = _align_one(lab, aligned / "one")
        if tg:
            return tg
        errs.append(f"align_one: {err}")
    tg, err = _align_corpus(lab, aligned / "corpus")
    if tg:
        return tg
    errs.append(f"align: {err}")
    mfa_align_single.last_error = " | ".join(errs)[-400:]   # 실패 사유를 detail 에 싣는다
    return None


class Track1Scorer:
    """참조와 (선택) Whisper 를 한 번 올려 두고 발화마다 `score()` 를 부른다."""

    def __init__(self, ref_dir: Path = DEFAULT_REF_DIR, sentences: Path = DEFAULT_SENTENCES, stt: bool = True, whisper_device: str = "auto") -> None:
        from scoring.score import Reference
        self.ref_dir = Path(ref_dir)
        man = json.load(open(sentences, encoding="utf-8"))
        self.sentences = {s["script_key"]: s for s in man["문장"]}
        self.reference_version = man.get("참조 stamp", "?")
        self.expected_combine = man.get("combine", "wr")          # 목록이 정한 합산 규칙 — 참조 params.combine 과 같아야 한다 (ref() 에서 검증)
        self._Reference = Reference
        self._refs: dict[str, Reference] = {}
        self.code = git_hash()
        self.model_version = f"track1-v3.4-{self.reference_version}+{self.code}"   # v3.4 = P12 flat 축 min 합산 (2026-08-31)
        self.stt = None
        if stt:
            from stt import track2_asr
            dev = track2_asr.pick_device(whisper_device)
            self.stt, self.stt_repo, self.stt_dtype, _ = track2_asr.build(dev)
            self.model_version += f"+{Path(self.stt_repo).name}+{self.stt_dtype}"   # dtype 도 버전에 — 코덱스 2026-09-08 #4
        if _mfa_mode() == "resident":
            # 기동 때 띄워 import 비용을 워밍업에 넣는다 (KAN-296). 실패해도 첫 요청에서 다시 띄우므로 기동은 막지 않는다
            try:
                _RESIDENT.start()
            except Exception:  # noqa: BLE001
                _RESIDENT.close()

    def ref(self, key: str):
        if key not in self._refs:
            r = self._Reference.load(self.ref_dir / f"reference_{key.replace('|', '_')}.json")
            # 참조 파일 ↔ 문장 목록 계약 검증 (코덱스 4차 인계 위험 ① · 2026-08-31): 옛 참조를 꽂으면 점수 규칙이 조용히 바뀌므로
            # stamp 와 combine 이 문장 목록(handoff_sentences)의 값과 다르면 채점을 거부한다. 실험용 우회: ACCENTURY_REF_CHECK=0
            if os.environ.get("ACCENTURY_REF_CHECK", "1") != "0":
                got = (r.body.get("stamp"), r.combine)
                want = (self.reference_version, self.expected_combine)
                if got != want:
                    raise RuntimeError(f"참조 {key} 가 문장 목록과 다르다 — 참조 (stamp, combine)={got} vs 목록 {want}. "
                                       f"참조 폴더({self.ref_dir})와 --sentences 를 같은 인계본으로 맞춰라")
            self._refs[key] = r
        return self._refs[key]

    # ── 한 발화 ──────────────────────────────────────────────────────────────
    def score(self, wav: str | Path, script_key: str, transcript: str | None = None, with_feedback: bool = True) -> dict:
        """계약 봉투의 재료를 돌려준다. 예외를 던지는 것은 입력 자체가 계약 밖일 때(모르는 문장 ID)뿐이다."""
        import soundfile as sf
        from pipeline.features import read_textgrid
        from scoring.extract import extract_utterance
        from scoring.score import score_extracted
        t_all = time.time(); stage: dict[str, float] = {}
        if script_key not in self.sentences:
            raise KeyError(f"서비스 문장이 아니다: {script_key!r}")
        script = self.sentences[script_key]["대본"]
        try:
            x, sr = sf.read(str(wav), dtype="float32")
        except Exception:  # noqa: BLE001
            return self._fail("format", False, stage, t_all, detail="WAV 를 읽을 수 없다")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, SR); x = resample_poly(x, SR // g, sr // g).astype(np.float32); sr = SR
        if x.size / SR > MAX_SECONDS:
            return self._fail("long", False, stage, t_all, detail=f"{x.size / SR:.1f}s > {MAX_SECONDS}s")
        # STT
        t = time.time()
        if transcript is None:
            if self.stt is None:
                transcript = script
            else:
                transcript = self.stt(x)
        stage["transcribe"] = time.time() - t
        # 게이트 (S0 · §4-2 · §4-3)
        t = time.time()
        A = align_transcript(script, transcript)
        stage["gate"] = time.time() - t
        gate = {"transcript": transcript, "gateRatio": round(A.gate_ratio, 3), "blankWordRatio": round(A.blank_word_ratio, 3)}
        if not A.passed:
            return self._fail("gate", False, stage, t_all, detail="대본과 다른 내용", **gate)
        if not A.usable:
            return self._fail("blank", False, stage, t_all, detail="빠진 어절이 많다", **gate)
        # MFA
        t = time.time()
        with tempfile.TemporaryDirectory(prefix="track1-") as td:
            tg = mfa_align_single(Path(wav) if sr == SR and not isinstance(wav, np.ndarray) else self._tmp_wav(x, td), transcript, Path(td))
            stage["align"] = time.time() - t
            if tg is None:
                return self._fail("mfa", False, stage, t_all, detail="정렬 실패: " + getattr(mfa_align_single, "last_error", ""), **gate)
            words_phones = read_textgrid(tg)
        # 추출 · 채점
        t = time.time()
        u = extract_utterance(x, SR, words_phones, transcript, script, utt_id=Path(str(wav)).stem)
        stage["f0"] = time.time() - t
        if u.get("unscorable"):
            code = "f0" if str(u["unscorable"]).startswith("S2") else "mfa"
            return self._fail(code, code == "f0", stage, t_all, detail=u["unscorable"], **gate)
        t = time.time()
        res = score_extracted(u, self.ref(script_key))
        stage["scoring"] = time.time() - t
        if res["score"] != res["score"]:
            return self._fail("blank", False, stage, t_all, detail="두 축 모두 결측", **gate)
        # 신뢰도 【휴리스틱 · 미검증】: 측정된 어절 비율 × (한 축 결측이면 0.5)
        conf = round(float((1 - A.blank_word_ratio) * (0.5 if res.get("partial") else 1.0)), 3)
        fb = res["feedback"]
        segments = []
        if with_feedback:
            for d in fb.get("order_detail", []):
                if d.get("same_as_modal"):
                    continue
                segments.append({"kind": "order", "word": d["word"], "mine": d["mine"], "gyeongnam": d["gyeongnam"],
                                 "raise": d["raise"] if d["ref_agree"] >= 0.6 else None, "lower": d["lower"] if d["ref_agree"] >= 0.6 else None,
                                 "refAgree": d["ref_agree"]})
            for e in fb["excursion"]:
                segments.append({"kind": "excursion", "word": e["word"], "idx": e["idx"], "type": e["kind"], "st": e["st"]})
            # P12 (2026-08-31): 밋밋/과장 자리 — 자리 = 어절 안 음절 쌍. flat 은 점수를 깎는 축, exag 는 피드백 전용
            for kind, key in (("flat", "flat_pos"), ("exag", "exag_pos")):
                for e in fb.get(key, []):
                    segments.append({"kind": kind, "word": e["word"], "idx": e["idx"], "pair": e["pair"], "pct": e["pct"]})
        return {
            "status": "OK", "intonationScore": int(round(res["score"])), "confidence": conf,
            "quality": {"code": "OK"}, "segments": segments,
            "modelVersion": self.model_version, "referenceVersion": f"{res.get('score_version')}+{res['params']['combine']}",   # stamp+코드+합산 규칙
            "processingMs": int((time.time() - t_all) * 1000),
            "detail": {"scriptKey": script_key, **gate, "scoreW": res["score_w"], "scoreR": res["score_r"], "scoreWR": res.get("score_wr"),
                       "scoreFlat": res.get("score_flat"), "scoreExag": res.get("score_exag"), "capped": res.get("capped"), "partial": res.get("partial"),
                       "D_w": res["D_w"], "D_r": res["D_r"], "D_flat": res.get("D_flat"), "D_exag": res.get("D_exag"),
                       "params": res.get("params"), "nearest": fb.get("nearest"),
                       "stageMs": {k: int(v * 1000) for k, v in stage.items()}},
        }

    @staticmethod
    def _tmp_wav(x: np.ndarray, td: str) -> Path:
        import soundfile as sf
        p = Path(td) / "in.wav"; sf.write(str(p), x, SR, subtype="PCM_16"); return p

    def _fail(self, kind: str, retryable: bool, stage: dict, t_all: float, detail: str = "", **extra) -> dict:
        return {"status": "FAILED", "quality": {"code": QUALITY[kind]}, "retryable": retryable, "modelVersion": self.model_version,
                "processingMs": int((time.time() - t_all) * 1000),
                "detail": {"reason": kind, "message": detail, **extra, "stageMs": {k: int(v * 1000) for k, v in stage.items()}}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True); ap.add_argument("--script-key", required=True)
    ap.add_argument("--transcript", default=None); ap.add_argument("--no-stt", action="store_true")
    ap.add_argument("--ref-dir", default=str(DEFAULT_REF_DIR)); ap.add_argument("--sentences", default=str(DEFAULT_SENTENCES))
    ap.add_argument("--bench", type=int, default=0, help="같은 입력을 N 회 돌려 단계별 시간 p50/p95 를 낸다")
    ap.add_argument("--no-feedback", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    sc = Track1Scorer(Path(a.ref_dir), Path(a.sentences), stt=not a.no_stt)
    load_s = time.time() - t0
    env = sc.score(a.wav, a.script_key, a.transcript, with_feedback=not a.no_feedback)
    print(json.dumps(env, ensure_ascii=False, indent=1))
    if a.bench:
        runs = [sc.score(a.wav, a.script_key, a.transcript, with_feedback=False) for _ in range(a.bench)]
        ms = [r["processingMs"] for r in runs]
        stages = {k: [r["detail"]["stageMs"].get(k, 0) for r in runs] for k in ("transcribe", "gate", "align", "f0", "scoring")}
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024)
        print(json.dumps({"bench": {"n": a.bench, "load_s": round(load_s, 1), "total_ms_p50": st.median(ms), "total_ms_max": max(ms),
                                    **{f"{k}_ms_p50": st.median(v) for k, v in stages.items()}, "max_rss_MB": int(rss),
                                    "scores": [r.get("intonationScore") for r in runs]}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
