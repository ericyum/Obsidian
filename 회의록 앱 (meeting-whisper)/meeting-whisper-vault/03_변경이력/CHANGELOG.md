# 핵심 변경 이력

세부 작업 로그 대신 현재 구조를 이해하는 데 필요한 마일스톤만 기록한다. 전체 diff와 삭제된 문서는 Git 이력을 사용한다.

## 2026-07-20 — Vault 재구성

- 신규 참여자 중심으로 시작하기, 운영, 기술 설계, 변경 이력, 최소 과거 이력 구조로 정리
- 중복 HTML 백업, 오래된 API 계약 사본, 초기 계획·진행 로그 제거
- AI Core summary template와 direct 검토 호출의 모델 결정 방식 및 `CAP_LLM_MODEL` 역할 명시
- 기준 소스 `0182231`

## 2026-07-16 — 장시간 전사·요약 안정화

- commit `0182231`
- Meeting 표시 길이를 `expectedDurationSec`로 Worker에 전달
- 720초 초과 강제 chunk, STT 2회 재시도와 소분할 복구
- Worker와 CAP의 transcript 저장 전 검증
- 계층형 요약, 녹음 원본 백업 안내, Outlook 내보내기 반영

## 2026-07-10 — 카테고리와 목록 UX

- commits `3a3b3af`, `94e0e65`
- 회의 카테고리, 사용자 정의 카테고리와 카테고리별 요약
- 제목·참여자·요약·전사 검색, 상태·카테고리 필터
- 상세 카테고리 즉시 저장과 polling 재렌더 방어

## 2026-07-08 — 세션 관리

- commit `b52b344`
- Approuter 세션 180분
- 만료 사전 경고, 수동 연장, 녹음 keep-alive와 업로드 전 세션 확인

## 2026-07-02 — CF Worker + SAP AI Core STT 전환

- commits `0709f2d`, `332bedc`
- Kyma CPU Whisper 대신 CF STT Worker가 AI Core `gemini-2.5-flash` 호출
- Worker concurrency 2, CAP dispatch limit 2
- 전사 저장과 background summary 분리
- AI Core usage log 기록

## 2026-06-26 — 공유 권한과 편집 UX

- commit `395763e`
- 생성자·참여자·공유자 권한 분리
- 생성자 전용 삭제, 참여자 편집, 공유자 읽기 전용
- 요약·검토 background task UX

## 2026-06-23~24 — CAP/HANA/Object Store와 Kyma 검증

- CAP, HANA, XSUAA, Object Store, dispatch queue와 Worker callback 골격 검증
- Kyma CPU Whisper의 성능·메모리·운영 지속성 한계 확인
- Worker 인증과 오류 표시 복구
- 결과는 [Kyma 검증과 전환 요약](../90_과거이력/Kyma_검증과_전환_요약.md)에 보존

## 2026-06-07 — 로컬 프로토타입

- FastAPI, SQLite, WhisperX 기반 로컬 회의록 PoC 시작
- 녹음, 파일 업로드, 전사, 요약, 목록과 편집 UX 검증
- 현재 운영 경로는 CAP/CF 구조이며 로컬 FastAPI는 `legacy/fastapi/`에만 남아 있다.
