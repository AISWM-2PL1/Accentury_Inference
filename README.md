# Accentury_Inference

Accentury 사투리 억양 채점 모델의 추론 코드입니다. 발화 음성(WAV)과 문장 키를 받아 억양 점수와 구간별 피드백을 계산합니다.
서비스 API는 [Accentury_Server](https://github.com/AISWM-2PL1/Accentury_Server)의 `ai/`가 맡고, 그 어댑터가 `src/scoring/serve.py`의 `Track1Scorer`를 부릅니다.

## 처리 흐름

1. Whisper로 발화를 전사합니다.
2. 음량, 길이, 전사 결과를 보고 채점할 수 있는 발화인지 확인합니다.
3. MFA로 음성과 문장을 맞춰 단어와 음소 구간을 얻습니다.
4. 구간마다 F0 같은 억양 특징을 뽑습니다.
5. 문장별 참조 분포와의 거리로 점수와 피드백을 냅니다.

## 구성

| 경로 | 내용 |
| --- | --- |
| `src/scoring/serve.py` | 진입점. 위 흐름을 함수 하나로 묶어 결과 JSON을 냅니다. |
| `src/scoring/mfa_one.py` | MFA `align_one`으로 발화 하나를 정렬합니다. 상주 모드에서는 한 번 떠서 요청마다 정렬을 맡습니다. |
| `src/scoring/align.py`, `extract.py` | 정렬 결과 읽기와 특징 추출. |
| `src/scoring/reference.py`, `distance.py`, `score.py` | 참조 분포, 거리 계산, 점수와 피드백. |
| `src/scoring/bench_server.py` | 단계별 소요 시간 측정. |
| `src/stt/track2_asr.py` | Whisper 전사. |
| `src/prosody/`, `src/pipeline/`, `src/align/` | F0 추출, 특징 파이프라인, 정렬 필터. |
| `dict_aug_track1.txt` | MFA 발음 사전. |
| `registry/scoring/` | 채점 대상 문장 목록. |

## 실행

```bash
pip install -r requirements.txt

# Whisper 전사부터
python src/scoring/serve.py --wav X.wav --script-key "1|5"

# 전사를 직접 줄 때
python src/scoring/serve.py --wav X.wav --script-key "1|5" --no-stt --transcript "..."

# 단계별 소요 시간
python src/scoring/serve.py --wav X.wav --script-key "1|5" --bench 5
```

실행에는 Whisper 가중치, 문장별 참조 분포(`ACCENTURY_DATA`), MFA 환경(`ACCENTURY_MFA_BIN`, `ACCENTURY_MFA_DICT`)이 필요합니다.
