# 버전 관리

- 사용자는 루트 README에서 `V1/` 또는 `V2/`를 선택한다. 각 폴더가 해당 버전의 소스·이미지·안내 묶음이다.
- V1은 KAN-296 MFA 최적화 원본이고 V2는 그 이미지의 용량 경량화본이다. 폴더 이름은 보기 쉬운 구분이며 기존 ECR tag·digest를 바꾸지 않는다.
- 기존 `v2026.10.29-kan296-0d72c03-slim.1` Git tag·GitHub Release와 첨부파일은 그대로 보존한다. 옛 루트 `releases/`·`tools/` 등은 해당 태그에서 볼 수 있다. 태그를 덮어쓰거나 옛 체크섬을 현재 폴더 구조에 맞춰 고치지 않는다.
- 각 `version.json`은 이미지 주소·digest·소스 커밋·과거 검사 기록을 연결한다. 원본·경량 이미지는 `accentury/inference-releases`의 고정 주소를 사용한다.
- CI는 두 소스 묶음을 원본 커밋과, V2 빌드·검사 코드를 기존 게시 커밋과 대조한다. 과거 릴리스 체크섬은 **그 당시 Git 스냅샷**에서 검증하므로 README나 폴더 이동 때문에 과거 기록을 덮어쓸 필요가 없다.

```bash
python3 .github/scripts/verify_versions.py
# 기존 AWS 읽기 권한으로 실제 이미지·보관 정책까지 확인
python3 .github/scripts/verify_versions.py --remote
```

새 추론 코드·런타임·모델·경량화 규칙을 적용할 때는 새 버전 폴더와 새 이미지 digest·검사 기록을 추가한다. V1/V2 파일을 바꾸고 예전 이미지가 그 변경을 포함한다고 안내하지 않는다. ECR의 태그 불변·자동 삭제 정책 부재는 원격 검사 시점의 상태이며, 운영 배포 여부는 별도로 기록한다.
