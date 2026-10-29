# KAN-296 최적화 이미지 경량화 — v2026.10.29-kan296-0d72c03-slim.1

## 왜 했나

유현님의 MFA 상주·예열 최적화를 유지하면서 이미지 저장·전송 용량을 줄였다. 학습·양자화·Whisper 런타임 포팅은 하지 않았다. **게시된 후보 이미지이며 운영 배포본임을 뜻하지 않는다.**

## 기준과 결과

| 구분 | 고정 기준 |
|---|---|
| 원본 tag | `e6c813c-kan296-0d72c03` |
| 원본 digest | `sha256:f360ac06f2c7374fc18b5af8c1ce860ba0c668b5d01aab92bec3ff8605884935` |
| 모델 소스 | `AISWM-2PL1/Accentury_Inference@a6f409668c95ee10e301090c53265e30227ca85a` |
| 결과 tag | `v2026.10.29-kan296-0d72c03-slim.1` |
| 결과 digest | `sha256:eda0fd1e07c4f04bd6db3726a414313c02ec06e61813ed2a529c2c9b19c278c5` |
| 실행 경로 | `linux/amd64`, PyTorch/Transformers 4.57.6, MFA `resident` |

이미지 `/app/src`의 Python 13개와 위 소스 커밋의 파일이 모두 SHA-256 동일하다([source-files.json](source-files.json)). `f77011c` 기반 경량화 결과는 이 릴리스의 근거에 포함하지 않는다. 이미지에는 원본의 서비스 API·모델·참조 데이터도 포함돼 있어 비공개 ECR을 사용한다.

정식 보관 위치는 `accentury/inference-releases`다. 원본과 결과의 digest를 그대로 유지해 미러링했으며, 태그는 `IMMUTABLE`이고 자동 삭제 정책이 없다. 기존 `accentury/ai`는 최근 50개만 보존하므로 정식 릴리스 링크로 사용하지 않는다.

## 뭘 줄였나

| 묶음 | 원본 → 경량본 (십진 GB) | 작업 |
|---|---|---|
| Whisper 가중치 | 3.092 → 3.092 | 그대로 보존 |
| MFA 환경·설치 캐시·모델 | 2.186 → 1.749 | 설치 캐시·문서·테스트·헤더·정적 아카이브·pip/micromamba 제거. MFA 실행 환경과 Python 실행 캐시 보존 |
| 추론 Python 라이브러리 | 1.274 → 0.801 | Transformers 부분 벤더링, pip·테스트·헤더·정적 아카이브·불필요 pycache 제거 |
| 참조 데이터·서비스 코드 | 0.154 → 0.154 | 그대로 보존 |
| 기타 | 0.240 → 0.222 | 문서·개발용 파일 정리 |
| **일반 파일 내용 합계** | **6.945 → 6.017** | hardlink 중복 제외 |
| **압축 레이어 합계** | **4.187 → 3.823** | 저장·전송 용량 |

근거: [validation.json](validation.json)의 `sizes`, `folder_groups`. 압축 해제 tar 레이어는 헤더·패딩을 포함해 7.105 → 6.050GB다. 위 파일 내용 합계·디스크 할당량·RAM과 구별한다. Conda 캐시는 환경과 hardlink를 공유하므로 폴더별 겉보기 크기를 단순히 더하지 않는다.

Transformers는 공통 모듈과 `auto`·`encoder_decoder`·`whisper` 모델 코드를 원본 바이트 그대로 `/app/vendor/transformers`에 둔다. Python 3.12에만 `.pth`를 등록하고 MFA Python 3.11 경로는 바꾸지 않았다. 모델·참조·서비스 파일 218개와 MFA 실행 pycache 5,259개의 SHA-256이 동일하다. 라이선스·실행 의존성을 보존했으며 `ensurepip` 복구용 wheel 약 5.2MB는 남아 있다.

## 이미지 받기

기존 AWS/ECR 읽기 권한이 필요하다. tag로 버전을 구분하고 실제 다운로드에는 digest를 고정한다.

```bash
aws ecr get-login-password --region ap-northeast-2 | \
  docker login --username AWS --password-stdin 325771561913.dkr.ecr.ap-northeast-2.amazonaws.com

docker pull --platform linux/amd64 \
  325771561913.dkr.ecr.ap-northeast-2.amazonaws.com/accentury/inference-releases@sha256:eda0fd1e07c4f04bd6db3726a414313c02ec06e61813ed2a529c2c9b19c278c5
```

이 이미지는 원본과 같은 `uvicorn app.main:app`를 실행한다. 실행 시 API 인증·서비스 설정은 기존 `Accentury_Server/ai/` 운영 설정을 따른다. 이 릴리스는 서버 배포 설정을 변경하지 않는다. 별도 전달용 Docker tar의 파일명·체크섬은 [manifest.json](manifest.json)의 `archive`에 있다.

## 재빌드와 바뀐 파일

```bash
git checkout v2026.10.29-kan296-0d72c03-slim.1
python3 tools/verify_release.py
docker build --platform linux/amd64 \
  -f docker/slim/Dockerfile \
  -t accentury-inference:rebuild-kan296-slim-1 docker/slim
```

- `docker/slim/Dockerfile`은 위 원본 **digest**에서 정리한 파일시스템을 새 레이어로 복사한다. 하위 레이어에 삭제 파일이 남아 용량이 유지되는 일을 피한다.
- 시험 당시 정의는 `docker/slim/Dockerfile.tested`에 보존했다. 게시용 Dockerfile은 `FROM`의 저장소 주소만 장기 보관 미러로 바꿨고 digest와 나머지 내용은 동일하다. 게시 이미지는 재빌드하지 않고 이미 검증한 바이트 그대로 보관했다.
- `docker/slim/code/prune_image.py`, `vendor_transformers.py`는 이번에 시험한 빌드 코드 그대로다. Linux/root/모델 코드해시/빌드 전용 marker를 확인하고 실행한다. 호스트에서 실행하지 않는다.
- `src/`는 바꾸지 않았다. Dockerfile은 저장소 소스를 다시 복사하지 않고 고정 원본 이미지의 소스를 유지한다. 따라서 재빌드 명령은 현재 main의 임의 소스 변경을 이미지에 반영하는 명령이 아니다.
- `releases/`는 버전별 근거, `validation/`은 검사 코드, `tools/verify_release.py`와 CI는 현재 코드·빌드 입력·기록의 불일치 검출을 담당한다.

동일 입력으로 재빌드해도 파일 시각·압축·빌드 provenance 때문에 이미지 digest가 달라질 수 있다. **검증한 바이트를 그대로 쓰려면 위 게시 digest를 pull**한다. 새 빌드는 별도 검사와 새 버전이 필요하다.

## 검증 방법과 한계

2026-10-29 Mac의 linux/amd64 에뮬레이션에서 원본·경량본을 비교했다. 실제 음성 6건을 포함한 18사례에서 시간 필드를 제외한 결과가 같았다. API 기동·health·인증·음성 1건·임시파일 정리·종료, MFA 상주 재사용·예열·비동기 교체·복구·대체 경로를 확인했다. Claude Fable 검토의 P2 지적을 수정한 후 최종 검토에서 새 P1/P2 결함이 없었다. 상세 범위·원시 결과 SHA는 [validation.json](validation.json), 실행 코드는 [validation/](../../validation/README.md)에 있다.

**운영 x86 추론시간·메모리·동시성은 미검증**이다. API 시험 timeout만 300초로 늘렸다. 호출 경계에서 강제로 발생시킨 timeout 검사이며 실제 지연·자손 프로세스 정리 전체를 검증한 것은 아니다. 자식 프로세스의 FST 캐시 소비도 직접 계측하지 않았다. Python 3.12 pycache 제거에 따른 첫 기동시간 동일성은 보장하지 않는다.

## 다음 할 일

운영 반영 여부는 실제 대상 서버에서 지연·메모리·동시성을 비교한 뒤 결정한다. 이미지 게시와 운영 배포를 구분하고, 다른 Transformers 모델이나 새 런타임을 적용할 때는 [새 버전 절차](../README.md)를 따른다.
