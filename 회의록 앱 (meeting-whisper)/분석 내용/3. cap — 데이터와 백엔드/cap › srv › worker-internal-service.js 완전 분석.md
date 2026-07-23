# worker-internal-service.js 완전 분석

> 소스: `cap/srv/worker-internal-service.js`  
> 역할: Worker callback을 검증하고 상태 전이·전사 저장·비동기 요약을 수행한다.

## 1. 이 파일이 있는 이유

Worker callback을 검증하고 상태 전이·전사 저장·비동기 요약을 수행한다.

이 파일은 **3. cap — 데이터와 백엔드** 영역에 속한다. 독립적으로 보기보다 아래의 호출자·의존 대상·데이터 계약과 함께 읽어야 한다.

## 2. 파일 개요

| 항목 | 값 |
|---|---|
| 경로 | `cap/srv/worker-internal-service.js` |
| 형식 | `javascript` |
| 전체 줄 수 | 336 |
| 주석 줄 수 | 0 |
| 주요 심볼 수 | 6 |
| 환경 변수 참조 수 | 0 |

## 3. 의존성

- `@sap/cds`
- `./lib/ai-core`
- `./lib/ai-usage-log`
- `./lib/audio`
- `./lib/audio-store`
- `./lib/meeting-note`
- `./lib/status`
- `./lib/transcription-dispatcher`

## 4. 주요 선언과 책임

| 선언 | 읽을 때 확인할 점 |
|---|---|
| `action: claimTranscription` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `action: heartbeatTranscription` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `action: getAudio` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `action: saveTranscript` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `action: logAiUsage` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `action: failTranscription` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |

## 7. 코드 흐름 상세

### 7.1 `claimTranscription`

- 위치: 183~216행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.2 `heartbeatTranscription`

- 위치: 217~224행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.3 `getAudio`

- 위치: 225~248행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.4 `saveTranscript`

- 위치: 249~284행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.5 `logAiUsage`

- 위치: 285~317행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.6 `failTranscription`

- 위치: 318~336행
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
