# ai-core-summary.test.js 완전 분석

> 소스: `tests/ai-core-summary.test.js`  
> 역할: 해당 기능의 정상·오류·경계 조건을 고정하는 자동화 테스트다.

## 1. 이 파일이 있는 이유

해당 기능의 정상·오류·경계 조건을 고정하는 자동화 테스트다.

이 파일은 **7. tests — 검증 체계** 영역에 속한다. 독립적으로 보기보다 아래의 호출자·의존 대상·데이터 계약과 함께 읽어야 한다.

## 2. 파일 개요

| 항목 | 값 |
|---|---|
| 경로 | `tests/ai-core-summary.test.js` |
| 형식 | `javascript` |
| 전체 줄 수 | 290 |
| 주석 줄 수 | 0 |
| 주요 심볼 수 | 4 |
| 환경 변수 참조 수 | 13 |

## 3. 의존성

- `node:assert/strict`
- `node:test`
- `../cap/srv/lib/ai-core`

## 4. 주요 선언과 책임

| 선언 | 읽을 때 확인할 점 |
|---|---|
| `preserveEnvironment` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `useConfiguredOpenAiProvider` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `transcript` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `openAiResponse` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |

## 5. 환경 변수

- `CAP_LLM_PROVIDER`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `CAP_LLM_BASE_URL`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `CAP_LLM_API_KEY`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `CAP_LLM_MODEL`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `AICORE_DEPLOYMENT_ID`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `AICORE_CLIENT_ID`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `AICORE_CLIENT_SECRET`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `AICORE_AUTH_URL`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `AICORE_API_URL`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `AICORE_RESOURCE_GROUP`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `CAP_LLM_SUMMARY_ATTEMPTS`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `CAP_LLM_SUMMARY_CHUNK_CHARS`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `CAP_LLM_SUMMARY_REDUCE_CHARS`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.

## 7. 코드 흐름 상세

### 7.1 `preserveEnvironment`

- 위치: 34~44행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.2 `useConfiguredOpenAiProvider`

- 위치: 45~52행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.3 `transcript`

- 위치: 53~59행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.4 `openAiResponse`

- 위치: 60~290행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

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
