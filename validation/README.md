# 경량화 검사

이 코드는 2026-10-29 원본 KAN-296 이미지와 경량 이미지를 비교한 검사 코드다. 이번 게시 과정에서는 검증한 이미지에 tag를 추가했으며 모델 시험을 새로 실행한 것으로 기록하지 않는다.

| 파일 | 확인 범위 |
|---|---|
| `run_suite.py`, `compare_results.py` | 실제 음성 6건, 오류 입력, fallback·상주 교체·복구 등 18사례. `processingMs`·`detail.stageMs`는 동등성 비교에서 제외 |
| `api_smoke.py`, `check_http_evidence.py` | 실제 Uvicorn/worker 기동, health·인증·음성 1건·임시파일·종료 |
| `mfa_optimizations_probe.py` | MFA Python에서 예열·사전 캐시 생성과 음성 예열 캐시 재사용 |
| `resident_background_probe.py` | 비동기로 준비한 상주 프로세스 객체를 다음 요청이 그대로 소비하는지 |
| `asset_integrity.py` | 모델·참조·서비스 파일 SHA, MFA Python 실행 캐시·경로·패키지 관리자 |
| `folder_sizes.py`, `measure_image.py` | hardlink를 중복 제거한 파일 바이트와 압축/해제 tar 레이어 바이트 |

## 입력과 실행

표본 음성은 공개 Git에 포함하지 않는다. 팀이 보관한 실제 음성 6건과 오류 입력을 별도 입력 폴더로 준비한다. `manifest.json`은 다음 순서의 항목 10개를 가진다. 나머지 제어 사례 8개는 스크립트가 첫 실제 음성을 재사용해 만든다.

| `id` | `kind` | `expect` | 입력 |
|---|---|---|---|
| `real_01`~`real_06` | `real` | `OK` | 실제 표본 6건 |
| `invalid_format` | `negative` | `not_OK` | 형식이 잘못된 입력 |
| `too_long` | `negative` | `not_OK` | 허용 길이를 넘는 음성 |
| `wrong_transcript` | `negative` | `not_OK` | 제시문과 맞지 않는 `transcript`를 지정 |
| `unknown_script` | `negative` | `KeyError` | 존재하지 않는 `script_key` |

모든 항목에 `id`, `kind`, `expect`, `wav`(입력 폴더 안 파일명), `script_key`가 필요하고, 실제 음성에는 `reference_transcript`가 필요하다. `transcript`는 강제 전사 입력용 선택 필드다. 원시 입력을 보관한 팀 환경에서 동일 manifest·음성을 사용하고 결과의 `input_hashes`를 대조한다. 새 표본을 구성하면 기존 18사례 결과의 재현으로 기록하지 않는다.

다음 명령의 `IMAGE_DIGEST`, `INPUT_DIR`, `OUT_DIR`는 각각 원본 또는 후보의 고정 이미지 주소, 입력 절대경로, 쓰기 가능한 별도 결과 절대경로로 지정한다. 원본·후보 결과 폴더는 분리한다.

```bash
docker run --rm --platform linux/amd64 --network none --cpus 4 --memory 7g \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  --mount "type=bind,src=$INPUT_DIR,dst=/inputs,readonly" \
  --mount "type=bind,src=$OUT_DIR,dst=/out" \
  --mount "type=bind,src=$PWD/validation,dst=/checks,readonly" \
  --entrypoint python "$IMAGE_DIGEST" -B /checks/run_suite.py --out /out/result.json
```

API 검사는 같은 mount·환경에서 `/checks/api_smoke.py --out /out/api.json`을 실행한다. MFA 예열 probe는 이미지의 `/opt/conda/envs/mfa/bin/python`과 해당 bin을 앞에 둔 PATH를 사용해야 한다. 위 명령 하나가 모든 검사 실행을 의미하지 않는다. 보관된 비교 결과와 한계는 해당 릴리스 `validation.json`을 따른다.

맥의 amd64 에뮬레이션 검사는 운영 x86 속도·메모리·동시성 보장이 아니다. API 제한을 시험에서 300초로 늘렸으므로 운영 timeout 준수 근거로도 쓰지 않는다.
