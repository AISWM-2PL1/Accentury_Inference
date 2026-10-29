"""트랙 1 채점 — 이웃 거리 채점기 (`docs/TRACK1_SCORING_SPEC.md` v2).

모듈
  align     §4  전사 ↔ 대본 자리 맞추기 · 문장 게이트 · 어절 판정 · MFA 음절 구간 붙이기
  extract   §3  발화 하나 → 곡선(어절 토막 20점) + 자리 특징
  distance  §5  z 통계 · 두 발화 거리 · 화자 단위 이웃 거리
  reference §6  문장별 참조 분포 빌드 (게이트 → 특징 → 회귀·z → 청소 → 눈금)
  score     §7~9 채점 · 피드백 · 왕복 검사
  tts_probe §11 Typecast 합성 (검증 전용)

참조 빌드와 서비스 채점이 **같은 함수**를 부른다 (§9) — `extract.extract_utterance` ·
`distance.*` · `score.score_extracted`.
"""
