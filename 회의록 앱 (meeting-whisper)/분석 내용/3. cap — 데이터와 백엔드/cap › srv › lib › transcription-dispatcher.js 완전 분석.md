# transcription-dispatcher.js 완전 분석

> 소스: `cap/srv/lib/transcription-dispatcher.js`  
> 역할: DB queue를 polling하고 동시성·재시도·backoff·lock을 제어한다.

## 1. 이 파일이 있는 이유

DB queue를 polling하고 동시성·재시도·backoff·lock을 제어한다.

이 파일은 **3. cap — 데이터와 백엔드** 영역에 속한다. 독립적으로 보기보다 아래의 호출자·의존 대상·데이터 계약과 함께 읽어야 한다.

## 2. 파일 개요

| 항목 | 값 |
|---|---|
| 경로 | `cap/srv/lib/transcription-dispatcher.js` |
| 형식 | `javascript` |
| 전체 줄 수 | 248 |
| 주석 줄 수 | 0 |
| 주요 심볼 수 | 15 |
| 환경 변수 참조 수 | 7 |

## 3. 의존성

- `./status`

## 4. 주요 선언과 책임

| 선언 | 읽을 때 확인할 점 |
|---|---|
| `getMaxAttempts` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `getPollMs` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `getDispatchLimit` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `getBackoffMs` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `createTranscriptionDispatch` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `startTranscriptionDispatcher` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `processTranscriptionDispatchQueue` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `dispatchOne` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `isDue` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `nowIso` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `positiveInt` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `positiveNumberOrNull` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `capInternalBaseUrlForWorker` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `clean` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `parseJsonEnv` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |

## 5. 환경 변수

- `TRANSCRIPTION_DISPATCH_MAX_ATTEMPTS`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `TRANSCRIPTION_DISPATCH_POLL_MS`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `TRANSCRIPTION_DISPATCH_LIMIT`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `TRANSCRIPTION_DISPATCH_BACKOFF_MS`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `TRANSCRIPTION_WORKER_URL`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `TRANSCRIPTION_DISPATCHER_ENABLED`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `CAP_INTERNAL_BASE_URL`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.

## 6. 경로·엔드포인트 단서

- `/internal`

## 7. 코드 흐름 상세

### 7.1 `getMaxAttempts`

- 위치: 19~22행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.2 `getPollMs`

- 위치: 23~26행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.3 `getDispatchLimit`

- 위치: 27~30행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.4 `getBackoffMs`

- 위치: 31~35행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.5 `createTranscriptionDispatch`

- 위치: 36~72행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.6 `startTranscriptionDispatcher`

- 위치: 73~84행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.7 `processTranscriptionDispatchQueue`

- 위치: 85~108행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.8 `dispatchOne`

- 위치: 109~199행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.9 `isDue`

- 위치: 200~204행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.10 `nowIso`

- 위치: 205~208행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.11 `positiveInt`

- 위치: 209~213행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.12 `positiveNumberOrNull`

- 위치: 214~218행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.13 `capInternalBaseUrlForWorker`

- 위치: 219~227행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.14 `clean`

- 위치: 228~231행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.15 `parseJsonEnv`

- 위치: 232~248행
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
