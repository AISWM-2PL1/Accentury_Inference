# V1 · MFA 최적화 원본

**유현님의 KAN-296 MFA 상주·예열 최적화가 적용된 원본 버전**입니다. 이 V1은 경량화 전 기준이며 `f77011c` 구버전이 아닙니다.

| 항목 | 내용 |
|---|---|
| 코드 | [src/](src/) — 원본 `a6f4096`과 바이트 동일 |
| 모델·정렬 | Whisper(PyTorch/Transformers) + MFA `resident` |
| 원본 이미지 이름 | `e6c813c-kan296-0d72c03` |
| 보관용 이미지 이름 | `base-e6c813c-kan296-0d72c03` |
| 크기 | 압축 레이어 4.19GB / 압축 전 파일 6.94GB |
| 버전·이미지 고정 정보 | [version.json](version.json) |

## 이미지 받기

AWS 인증과 `accentury/inference-releases` 저장소의 읽기 권한이 필요합니다.

```bash
aws ecr get-login-password --region ap-northeast-2 | \
  docker login --username AWS --password-stdin 325771561913.dkr.ecr.ap-northeast-2.amazonaws.com

docker pull --platform linux/amd64 \
  325771561913.dkr.ecr.ap-northeast-2.amazonaws.com/accentury/inference-releases@sha256:f360ac06f2c7374fc18b5af8c1ce860ba0c668b5d01aab92bec3ff8605884935
```

이미지에는 기존 API·모델·MFA 환경·참조 데이터가 들어 있습니다. API 실행 설정은 [Accentury_Server](https://github.com/AISWM-2PL1/Accentury_Server)의 해당 서비스 설정을 따릅니다. 원본 제작 Dockerfile을 새로 추정해 넣지 않았으며, V1 이미지는 위 고정 digest로 받습니다.

## 소스 직접 실행

```bash
cd V1  # 저장소 루트에서 실행할 때
pip install -r requirements.txt
python src/scoring/serve.py --wav X.wav --script-key "1|5"
```

소스 실행에는 별도로 Whisper 가중치, 참조 분포(`ACCENTURY_DATA`), MFA 환경(`ACCENTURY_MFA_BIN`, `ACCENTURY_MFA_DICT`)이 필요합니다. `requirements.txt` 설치만으로 완성 이미지가 재현되는 것은 아닙니다.

발화는 Whisper 전사 → 입력 검사 → MFA 정렬 → 억양 특징 추출 → 참조 분포와 비교하는 순서로 채점합니다. 진입점은 `src/scoring/serve.py`의 `Track1Scorer`이며, 사전은 `dict_aug_track1.txt`, 문장 목록은 `registry/scoring/`입니다.

[V2 경량화본 보기](../V2/README.md) · [버전 선택으로 돌아가기](../README.md)
