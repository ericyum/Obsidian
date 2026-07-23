# transcription-architecture-migration.md 완전 분석

> 소스: `docs/transcription-architecture-migration.md`  
> 역할: Kyma CPU Whisper에서 CF Worker + AI Core STT로 전환한 배경과 운영 기준을 기록한다.

## 1. 이 파일이 있는 이유

Kyma CPU Whisper에서 CF Worker + AI Core STT로 전환한 배경과 운영 기준을 기록한다.

이 파일은 **9. docs와 루트 설정** 영역에 속한다. 독립적으로 보기보다 아래의 호출자·의존 대상·데이터 계약과 함께 읽어야 한다.

## 2. 파일 개요

| 항목 | 값 |
|---|---|
| 경로 | `docs/transcription-architecture-migration.md` |
| 형식 | `markdown` |
| 전체 줄 수 | 143 |
| 주석 줄 수 | 10 |
| 주요 심볼 수 | 0 |
| 환경 변수 참조 수 | 0 |

## 6. 경로·엔드포인트 단서

- `/transcribe`

## 7. 코드 흐름 상세

이 파일은 선언형 설정 또는 짧은 보조 파일이다. 위에서 아래로 읽되, 키의 순서보다 각 키가 어느 런타임에서 소비되는지를 확인한다.

## 8. 변경 시 영향 범위
+
- 공개 계약, 상태명, 환경 변수, URL을 바꾸면 호출 측과 테스트를 함께 수정해야 한다.
- 저장 데이터의 형식이 바뀌면 기존 HANA 행과 진행 중인 작업의 호환성을 확인한다.
- 인증·소유권 검사는 편의상 우회하면 안 된다. Approuter 인증과 CAP 역할 검사는 서로 다른 계층이다.

## 9. 관련 문서

- [[00. 전체 구조와 책임 지도]]
- [[01. 녹음부터 회의록까지 End-to-End]]
- [[02. 데이터 모델과 상태 전이]]
- [[03. 인증·권한·신뢰 경계]]
