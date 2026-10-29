"""발화 하나를 `mfa align_one` 으로 정렬하고 라벨을 현행(`mfa align`) 형태로 되돌린다 — KAN-57.

**이 파일은 MFA conda env 의 파이썬으로 돈다** (`$ACCENTURY_MFA_BIN/python`). 서비스 본체(`serve.py`)의
파이썬에는 `montreal_forced_aligner` 가 없다 (도커에서 MFA 는 `/opt/conda/envs/mfa` 에 따로 있다).

## 왜 필요한가 (실측 2026-09-04, 개발 맥)

`serve.py` 는 발화 하나를 코퍼스 폴더에 넣고 `mfa align` 을 부른다 — **13.4~14.8 초**. 그 대부분이
코퍼스 데이터베이스를 세우는 준비 작업이고 오디오 길이와 무관한 고정비다 (1.48 초 음성도 14 초).
`mfa align_one` 은 그 준비가 없어 **2.29~2.62 초**, 6 배 빠르다.

## 대가와 그 처리

`align_one` 은 원본 텍스트를 들고 있지 않고 결과를 **사전 단어표**에서 되짚어 적는다. 그래서
사전에 없는 형태소가 `<unk>` 로 뭉개진다. 서비스 문장 157 개 기준 형태소 2,941 개 중 147 개(5.0%),
**문장으로는 90/157(57.3%)** 가 하나 이상 걸린다 (미등록 어휘 120 종, 대부분 경남 방언 —
어예·온나·능교·정구지). `features._clean` 이 `<` 든 라벨을 버리므로 그 칸이 사라지고
`group_eojeol` 의 어절 복원이 실패한다.

되돌리는 방법은 추측이 아니다. **MFA 가 쓰는 것과 같은 토크나이저**를 우리가 돌리면 형태소 목록이
나오고, 그것이 `align_one` 의 words 티어(`<eps>` 제외)와 **칸 수·순서가 1:1** 이다 —
`<unk>` 아닌 칸의 라벨도 전부 일치한다 (12/12 실측, `restore_check.py`). 그래서 티어 라벨을
토크나이저 출력으로 그대로 덮어쓰면 원문이 정확히 복원된다.

복원이 안 되는 입력 둘 (코덱스 2026-09-08 #12): 토크나이저가 영문을 소문자로 바꾸므로 대문자 원문은 복원되지 않고,
원문에 문자열 `<eps>` 가 있으면 묵음과 구분이 안 돼 칸 수 불일치로 폴백한다. 현재 서비스 157 문장과 등록 전사에는
ASCII·`<` 가 없어 영향이 없다.

묵음 표기도 맞춘다: `align_one` 은 `<eps>`(words)·`sil`(phones) 로 적고 `mfa align` 은 빈 라벨로
남긴다. 빈 문자열로 바꿔 `features.read_textgrid`(`includeEmptyIntervals=False`) 가 양쪽에서
같은 것을 읽게 한다.

사용:
    $ACCENTURY_MFA_BIN/python mfa_one.py WAV TXT DICT ACOUSTIC OUT.TextGrid
종료 코드 0 = 성공. 실패 사유는 stderr 로 나간다 (`serve.py` 가 `detail` 에 싣는다).

## 상주 모드 (KAN-296)

    $ACCENTURY_MFA_BIN/python mfa_one.py --serve

한 번 떠서 표준입력으로 한 줄 JSON 요청을 받고 표준출력으로 한 줄 JSON 응답을 낸다. 발화마다 하는 일은
위 한 번짜리와 **같은 함수, 같은 순서**(`_align_one` → `restore_labels`)다. 아끼는 것은 파이썬 기동과
MFA import, 토크나이저 생성, librosa import 와 numba 컴파일(`_prewarm_audio`), 발음 사전 FST 생성
(`_prebuild_lexicon`, ping 에 `dic` 과 `acoustic` 이 있을 때)이다.

**정렬은 요청마다 fork 한 자식에서 한다.** 한 프로세스에서 `align_one` 을 연달아 부르면 둘째 건부터 결과가
달라진다 (2026-10-26 실측: 15 건 중 11 건, 경계 10 ms 이동이 대부분이고 1 건은 묵음 칸이 하나 더 생겼다).
MFA 안에 앞 건의 상태가 남는 것이다. 새로 띄운 프로세스의 첫 건은 한 번짜리와 바이트까지 같았으므로,
import 만 끝낸 부모를 매번 fork 해서 자식이 늘 "첫 건"이 되게 한다. 부모는 정렬을 한 번도 하지 않는다.

    요청  {"op": "ping", "dic"?, "acoustic"?}                       → {"ok": true}
          {"op": "align", "wav", "txt", "dic", "acoustic", "out"}   → {"ok": true} | {"ok": false, "error": "..."}

MFA 와 Kaldi 가 표준출력에 찍는 로그가 응답 줄에 섞이지 않도록, 응답은 기동 때 복제해 둔 fd 로만 쓰고
fd 1 은 stderr 로 덮는다.
"""
from __future__ import annotations

import ctypes
import functools
import json
import os
import signal
import sys
import traceback
from pathlib import Path

SIL_WORD, SIL_PHONE, OOV_WORD = "<eps>", "sil", "<unk>"
PR_SET_PDEATHSIG = 1                         # linux/prctl.h


#: 부모가 미리 만든 발음 사전 FST. 키는 (사전 경로, 크기, 수정 시각, 음향 모델), 값은 (생성 인자, LexiconCompiler)
_LEXICON: dict[tuple, tuple[dict, object]] = {}


def _lexicon_key(dic: Path, acoustic: str) -> tuple:
    st = dic.stat()
    return str(dic), st.st_size, st.st_mtime_ns, acoustic


def _lexicon_kwargs(acoustic: str) -> dict:
    """`align_one` 이 LexiconCompiler 에 넘기는 인자를 같은 출처(음향 모델의 parameters)에서 만든다.

    MFA 의 `align_one_function` 과 같은 식이다. 어긋나면 자식이 미리 만든 것을 쓰지 않고 직접 만든다
    (`_use_prebuilt_lexicon`) - 느려질 뿐 결과는 그대로다.
    """
    from montreal_forced_aligner.command_line.utils import validate_model_arg

    p = validate_model_arg(acoustic, "acoustic").parameters
    return dict(
        disambiguation=False,
        silence_probability=p["silence_probability"],
        initial_silence_probability=p["initial_silence_probability"],
        final_silence_correction=p["final_silence_correction"],
        final_non_silence_correction=p["final_non_silence_correction"],
        silence_phone=p["optional_silence_phone"],
        oov_phone=p["oov_phone"],
        position_dependent_phones=p["position_dependent_phones"],
        phones=p["non_silence_phones"],
        ignore_case=True,
    )


def _prebuild_lexicon(dic: Path, acoustic: str) -> None:
    """발음 사전 FST 를 부모에서 한 번 만든다 (KAN-296 3차).

    `align_one` 은 `--clean` 이면 요청마다 사전 파일 전체(2만 2천 줄)를 읽어 FST 를 새로 만든다. 2026-10-27
    프로파일에서 자식 한 건의 약 1 초였다. 사전은 고정 파일이고 우리는 G2P 를 쓰지 않아 정렬 중에 바뀌지 않으므로,
    부모가 만들어 두면 fork 한 자식이 물려받는다. 메모리는 부모에 약 30 MB 더 든다. FST 를 파일로 남기는 MFA 자체
    캐시(`--no_clean`)는 어댑터가 요청마다 작업 폴더를 비워(KAN-27) 쓸 수 없다.
    실패해도 자식이 직접 만들므로 느려질 뿐이라, 사유만 stderr 로 남기고 넘어간다.
    """
    try:
        key = _lexicon_key(dic, acoustic)
        if key in _LEXICON:
            return
        from kalpy.fstext.lexicon import LexiconCompiler

        kwargs = _lexicon_kwargs(acoustic)
        compiler = LexiconCompiler(**kwargs)
        compiler.load_pronunciations(dic)
        compiler.create_fsts()
        compiler.clear()
        _LEXICON[key] = (kwargs, compiler)
    except Exception:                        # noqa: BLE001 - 사전 예열 실패는 속도 문제일 뿐이다
        print(f"mfa_one: 사전 예열 실패, 자식이 직접 한다\n{traceback.format_exc()[-400:]}", file=sys.stderr)


def _use_prebuilt_lexicon(dic: Path, acoustic: str) -> None:
    """이 프로세스의 `align_one` 이 부모가 만든 FST 를 쓰게 한다. fork 한 자식에서만 부른다.

    `align_one_function` 이 LexiconCompiler 를 만들 때 생성 인자가 부모 것과 같으면 미리 만든 객체를 주고, 그 객체의
    읽기와 생성(`load_pronunciations`, `create_fsts`, `clear`)은 건너뛴다. 인자가 다르면 원래 클래스로 만든다.
    """
    hit = _LEXICON.get(_lexicon_key(dic, acoustic)) if dic.exists() else None
    if hit is None:
        return
    import montreal_forced_aligner.command_line.align_one as cli

    kwargs, compiler = hit
    real = cli.LexiconCompiler

    def factory(**kw):
        if kw != kwargs:
            print("mfa_one: 사전 생성 인자가 달라 미리 만든 FST 를 쓰지 않는다", file=sys.stderr)
            return real(**kw)
        for name in ("load_pronunciations", "create_fsts", "clear"):
            setattr(compiler, name, lambda *a, **k: None)
        return compiler

    cli.LexiconCompiler = factory


def _align_one(wav: Path, txt: Path, dic: Path, acoustic: str, out: Path) -> None:
    """`mfa align_one` 을 같은 프로세스 안에서 부른다 — 토크나이저와 import 비용을 나눠 쓴다."""
    from montreal_forced_aligner.command_line.mfa import mfa_cli

    if _LEXICON:
        _use_prebuilt_lexicon(dic, acoustic)
    mfa_cli.main(
        args=["align_one", "--quiet", "--clean", str(wav), str(txt), str(dic), acoustic, str(out)],
        standalone_mode=False,
    )


@functools.lru_cache(maxsize=1)
def _tokenizer():
    """토크나이저는 입력과 무관하게 같으므로 상주 모드에서는 한 번만 만든다 (KAN-296)."""
    from montreal_forced_aligner.data import Language
    from montreal_forced_aligner.tokenization.spacy import generate_language_tokenizer

    return generate_language_tokenizer(Language.korean)


def _tokens(text: str) -> list[str]:
    """MFA 가 정렬 전에 쓰는 것과 같은 토크나이저의 형태소 목록."""
    return _tokenizer()(text).normalized_text.split()


def restore_labels(path: Path, transcript: str) -> None:
    """words 티어의 `<unk>` 를 원문 형태소로 되돌리고 묵음 표기를 현행 형태(빈 라벨)로 맞춘다.

    칸 수가 토크나이저 출력과 다르면 `ValueError` — 조용히 어긋난 라벨을 쓰지 않는다.
    """
    from praatio import textgrid

    tg = textgrid.openTextgrid(str(path), includeEmptyIntervals=True)
    words = tg.getTier("words")
    spoken = [i for i, e in enumerate(words.entries) if e.label not in (SIL_WORD, "")]
    toks = _tokens(transcript)
    if len(spoken) != len(toks):
        raise ValueError(f"칸 수 불일치: TextGrid {len(spoken)} vs 토크나이저 {len(toks)}")

    def relabel(tier, sil_label: str, new: dict[int, str] | None = None):
        out = []
        for i, e in enumerate(tier.entries):
            label = "" if e.label in (sil_label, "") else (new or {}).get(i, e.label)
            out.append((e.start, e.end, label))
        return tier.new(entries=out)

    tg.replaceTier("words", relabel(words, SIL_WORD, dict(zip(spoken, toks))))
    tg.replaceTier("phones", relabel(tg.getTier("phones"), SIL_PHONE))
    tg.save(str(path), format="long_textgrid", includeBlankSpaces=True, reportingMode="silence")


@functools.lru_cache(maxsize=1)
def _prewarm_audio() -> None:
    """librosa import 와 numba 컴파일을 부모에서 한 번 치른다 (KAN-296 2차).

    `align_one` 은 음성을 `librosa.load` 로 읽는데, librosa 는 처음 쓰일 때 import 되고 그 안의 numba 함수가
    컴파일된다. 부모가 이것을 하지 않으면 fork 한 자식이 요청마다 다시 해서, 상주 뒤에 남은 정렬 시간의 약 절반이
    여기에 든다 (2026-10-26 프로파일, 맥 에뮬레이션). 부모가 가짜 음성(무음 1 초, 44.1 kHz라 리샘플까지 탄다)으로
    한 번 읽어 두면 자식이 컴파일된 상태를 물려받는다. 정렬 결과는 그대로다 (TextGrid 15/15 바이트 동일).
    실패해도 정렬은 자식이 직접 하므로 느려질 뿐이라, 사유만 stderr 로 남기고 넘어간다.
    """
    import tempfile

    try:
        import librosa
        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory(prefix="mfa-warm-") as td:
            wav = os.path.join(td, "warm.wav")
            sf.write(wav, np.zeros(44100, dtype="float32"), 44100)
            librosa.load(wav, sr=16000, mono=False)
    except Exception:                        # noqa: BLE001 - 예열 실패는 속도 문제일 뿐이다
        print(f"mfa_one: 오디오 예열 실패, 자식이 직접 한다\n{traceback.format_exc()[-400:]}", file=sys.stderr)


def _handle(msg: dict) -> dict:
    """요청 하나를 처리한다. 어떤 실패도 응답으로 돌려주고 프로세스는 살려 둔다."""
    try:
        if msg.get("op") == "ping":
            from montreal_forced_aligner.command_line.mfa import mfa_cli  # noqa: F401 - import 비용을 기동 때 치른다
            _tokenizer()
            _prewarm_audio()
            if msg.get("dic") and msg.get("acoustic"):
                _prebuild_lexicon(Path(msg["dic"]), msg["acoustic"])
            return {"ok": True}
        if msg.get("op") != "align":
            return {"ok": False, "error": f"모르는 op: {msg.get('op')!r}"}
        out = Path(msg["out"])
        out.unlink(missing_ok=True)          # 앞 요청의 산출물을 성공으로 읽지 않게
        _align_one(Path(msg["wav"]), Path(msg["txt"]), Path(msg["dic"]), msg["acoustic"], out)
        if not out.exists():
            return {"ok": False, "error": "align_one 이 TextGrid 를 내지 않았다"}
        restore_labels(out, Path(msg["txt"]).read_text(encoding="utf-8").strip())
        return {"ok": True}
    except (Exception, SystemExit):          # click 은 standalone_mode=False 여도 SystemExit 를 던질 수 있다
        return {"ok": False, "error": traceback.format_exc()[-400:]}


def _handle_forked(msg: dict) -> dict:
    """정렬 요청을 fork 한 자식에서 처리하고 결과를 파이프로 받는다 (부모에 정렬 상태를 남기지 않는다)."""
    r, w = os.pipe()
    parent = os.getpid()
    pid = os.fork()
    if pid == 0:                             # 자식: 처리하고 결과 한 줄을 쓰고 바로 끝난다
        os.close(r)
        try:
            # 부모(상주 프로세스)가 시간 초과로 죽으면 자식도 바로 죽는다. 남으면 serve.py 의 대체 경로와 동시에
            # 정렬하고, 어댑터가 작업 폴더를 지운 뒤에도 파일을 건드린다 (Codex 리뷰 P2). prctl 은 리눅스에만 있어
            # 개발 맥에서는 건너뛴다 - 없으면 자식이 결과 없이 끝나 요청마다 느린 `mfa align` 으로 간다 (Codex 리뷰 P2)
            if sys.platform.startswith("linux"):
                ctypes.CDLL(None, use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
            if os.getppid() != parent:       # prctl 전에 부모가 이미 죽었다
                os._exit(1)
            data = json.dumps(_handle(msg), ensure_ascii=False).encode("utf-8")
            os.write(w, data)
        finally:
            os._exit(0)                      # atexit 이나 부모가 연 파일의 flush 를 자식이 돌리지 않게
    os.close(w)
    chunks = []
    while chunk := os.read(r, 65536):
        chunks.append(chunk)
    os.close(r)
    _, status = os.waitpid(pid, 0)
    if not chunks:
        return {"ok": False, "error": f"정렬 자식이 결과 없이 끝났다 (wait status {status})"}
    return json.loads(b"".join(chunks))


def serve() -> int:
    """상주 모드 (KAN-296). 표준입력이 닫히면 끝난다."""
    reply = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            res = {"ok": False, "error": "JSON 이 아니다"}
        else:
            res = _handle_forked(msg) if msg.get("op") == "align" else _handle(msg)
        reply.write(json.dumps(res, ensure_ascii=False) + "\n")
    return 0


def main(argv: list[str]) -> int:
    if argv[1:] == ["--serve"]:
        return serve()
    if len(argv) != 6:
        print(__doc__, file=sys.stderr)
        return 2
    wav, txt, dic, acoustic, out = Path(argv[1]), Path(argv[2]), Path(argv[3]), argv[4], Path(argv[5])
    _align_one(wav, txt, dic, acoustic, out)
    if not out.exists():
        print("align_one 이 TextGrid 를 내지 않았다", file=sys.stderr)
        return 1
    restore_labels(out, txt.read_text(encoding="utf-8").strip())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
