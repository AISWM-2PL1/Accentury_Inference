# V2 · 용량 경량화본

**V1의 추론 코드·모델을 유지하면서 이미지 용량을 줄인 버전**입니다. 코드·이미지 주소·빌드와 검사를 이 폴더에서 함께 볼 수 있습니다.

| 항목 | 내용 |
|---|---|
| 코드 | [src/](src/) — V1과 바이트 동일 |
| 경량화 빌드 | [docker/slim/](docker/slim/) |
| 검사 코드 | [validation/](validation/README.md) |
| 변경 내용 | [CHANGES.md](CHANGES.md) |
| 이미지 버전 | `v2026.10.29-kan296-0d72c03-slim.1` |
| 크기 | 압축 레이어 3.82GB / 압축 전 파일 6.02GB |
| 버전·이미지 고정 정보 | [version.json](version.json) |

## 이미지 받기

AWS 인증과 `accentury/inference-releases` 저장소의 읽기 권한이 필요합니다.

```bash
aws ecr get-login-password --region ap-northeast-2 | \
  docker login --username AWS --password-stdin 325771561913.dkr.ecr.ap-northeast-2.amazonaws.com

docker pull --platform linux/amd64 \
  325771561913.dkr.ecr.ap-northeast-2.amazonaws.com/accentury/inference-releases@sha256:eda0fd1e07c4f04bd6db3726a414313c02ec06e61813ed2a529c2c9b19c278c5
```

API 실행 설정은 [Accentury_Server](https://github.com/AISWM-2PL1/Accentury_Server)의 해당 서비스 설정을 사용합니다. 이번 폴더 정리로 이미지 내용이나 운영 서버 설정을 바꾸지 않았습니다.

## 소스와 이미지 빌드

```bash
cd V2  # 저장소 루트에서 실행할 때

# 소스 실행: 가중치·참조 데이터·MFA 환경은 별도 준비
pip install -r requirements.txt
python src/scoring/serve.py --wav X.wav --script-key "1|5"

# V1 이미지를 기준으로 용량을 줄이는 빌드
docker build --platform linux/amd64 -f docker/slim/Dockerfile \
  -t accentury-inference:v2-local docker/slim
```

`pip install`은 소스 개발용이며 V2의 경량 환경을 만드는 과정은 Docker 빌드가 담당합니다. Dockerfile은 고정된 V1 이미지를 정리하며 이 폴더의 수정된 `src/`를 이미지에 복사하지 않습니다. 소스나 런타임을 바꿀 때는 새 이미지와 새 버전 검사가 필요합니다.

위 빌드의 파일 시각·압축·provenance에 따라 결과 digest가 달라질 수 있습니다. 이미 검사한 이미지를 쓰려면 위 고정 digest를 pull합니다.

## 검사 기록

2026-10-29 이미지 제작 때 실제 음성 6건을 포함한 18사례의 처리시간 외 결과가 같았고 API·MFA 상주·예열·교체·복구를 확인했습니다. **운영 x86 속도·RAM·동시성은 미검증이며 운영 배포는 하지 않았습니다.** 폴더 정리는 새로운 추론 시험을 뜻하지 않습니다.

[기존 릴리스와 첨부 검사 기록](https://github.com/AISWM-2PL1/Accentury_Inference/releases/tag/v2026.10.29-kan296-0d72c03-slim.1) · [고정 검사 JSON](https://github.com/AISWM-2PL1/Accentury_Inference/blob/v2026.10.29-kan296-0d72c03-slim.1/releases/v2026.10.29-kan296-0d72c03-slim.1/validation.json)

[V1 원본 보기](../V1/README.md) · [버전 선택으로 돌아가기](../README.md)
