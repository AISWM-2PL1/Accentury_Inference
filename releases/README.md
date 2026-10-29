# 이미지 릴리스 관리

현재 게시본은 [current.json](current.json)이 가리킨다. 이는 배포된 운영 버전의 표시가 아니다.

## 버전과 고정 기준

- 버전 형식은 `vYYYY.MM.DD-kan번호-모델코드해시-slim.회차`다. 같은 날 수정해도 회차를 올린다.
- Git tag와 ECR tag는 같은 버전을 쓴다. 릴리스 보관용 ECR `accentury/inference-releases`는 `IMMUTABLE`이고 자동 삭제 정책이 없다. 원본 이미지도 같은 저장소에 고정 digest로 보존한다. 기존 Git tag·릴리스 폴더도 덮어쓰지 않는다.
- 서비스 빌드용 `accentury/ai`에는 최근 50개만 유지하는 정책이 있어 장기 보존 정본으로 쓰지 않는다. 그 저장소의 기존 정책·배포는 바꾸지 않는다.
- `manifest.json`은 원본 이미지 digest, 게시 이미지 digest, 모델 소스 커밋, 빌드·검사 코드 SHA-256, 아카이브 SHA-256을 연결한다. 실제 실행·비교에는 tag 대신 digest를 쓴다.
- 코드·빌드 입력과 현재 manifest가 달라지면 `tools/verify_release.py` 및 CI가 실패한다. 소스 변경만으로 과거 이미지가 최신이라고 표시되는 것을 막는다.
- CI는 저장소 내부 정합성 검사다. 원격 보관 상태는 AWS 읽기 권한으로 `python3 tools/verify_release.py --remote`를 실행해 두 이미지 digest·태그 불변·자동 삭제 정책 부재를 재조회한다. 게시 당시 상태를 향후 삭제 불가능 보장으로 읽지 않는다.

## 다음 릴리스

1. 새 원본 이미지와 소스 커밋을 고정하고 이미지 `/app/src`와 저장소 파일 해시를 대조한다.
2. 별도 버전으로 경량화한다. Dockerfile의 digest·가드·Python 버전·환경을 새 원본에 맞추고, 과거 버전을 단순 재태깅하지 않는다.
3. 원본·후보의 점수·전사·오류 처리·API·MFA 상주/예열/교체를 비교하고 원시 결과를 보존한다. 속도는 같은 실제 대상 장비에서 별도로 확인한다.
4. 검토한 이미지에 새 버전 tag를 붙여 ECR에 push하고 원격 digest를 재조회한다. 공개 Git에는 음성·가중치·참조 데이터·인증정보를 넣지 않는다.
5. 새 `releases/<version>/`에 manifest·검사 요약·README·체크섬을 추가하고 `current.json`과 루트 README를 함께 갱신한다. 원래 측정일과 게시일을 구분한다.
6. `python3 tools/verify_release.py`를 통과시킨 뒤 커밋·새 Git tag·GitHub Release를 게시한다. 삭제·force push로 과거 발행본을 교체하지 않는다.

GitHub Release는 고정 이미지 주소와 검사 기록을 찾는 입구다. 이미지 게시와 운영 배포는 별도 작업이다.
