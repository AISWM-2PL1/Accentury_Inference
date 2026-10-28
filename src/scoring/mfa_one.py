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
"""
from __future__ import annotations

import sys
from pathlib import Path

SIL_WORD, SIL_PHONE, OOV_WORD = "<eps>", "sil", "<unk>"


def _align_one(wav: Path, txt: Path, dic: Path, acoustic: str, out: Path) -> None:
    """`mfa align_one` 을 같은 프로세스 안에서 부른다 — 토크나이저와 import 비용을 나눠 쓴다."""
    from montreal_forced_aligner.command_line.mfa import mfa_cli

    mfa_cli.main(
        args=["align_one", "--quiet", "--clean", str(wav), str(txt), str(dic), acoustic, str(out)],
        standalone_mode=False,
    )


def _tokens(text: str) -> list[str]:
    """MFA 가 정렬 전에 쓰는 것과 같은 토크나이저의 형태소 목록."""
    from montreal_forced_aligner.data import Language
    from montreal_forced_aligner.tokenization.spacy import generate_language_tokenizer

    return generate_language_tokenizer(Language.korean)(text).normalized_text.split()


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


def main(argv: list[str]) -> int:
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
