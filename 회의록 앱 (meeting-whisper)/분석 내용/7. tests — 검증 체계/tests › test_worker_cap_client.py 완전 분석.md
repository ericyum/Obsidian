# test_worker_cap_client.py 완전 분석

> 소스: `tests/test_worker_cap_client.py`  
> 역할: 해당 기능의 정상·오류·경계 조건을 고정하는 자동화 테스트다.

## 1. 이 파일이 있는 이유

해당 기능의 정상·오류·경계 조건을 고정하는 자동화 테스트다.

이 파일은 **7. tests — 검증 체계** 영역에 속한다. 독립적으로 보기보다 아래의 호출자·의존 대상·데이터 계약과 함께 읽어야 한다.

## 2. 파일 개요

| 항목 | 값 |
|---|---|
| 경로 | `tests/test_worker_cap_client.py` |
| 형식 | `python` |
| 전체 줄 수 | 95 |
| 주석 줄 수 | 0 |
| 주요 심볼 수 | 5 |
| 환경 변수 참조 수 | 0 |

## 3. 의존성

- `asyncio`
- `workers.transcription.app`
- `workers.transcription.app.cap_client`
- `workers.transcription.app.config`

## 4. 주요 선언과 책임

| 선언 | 읽을 때 확인할 점 |
|---|---|
| `FakeResponse` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `FakeAsyncClient` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_cap_client_uses_xsuaa_client_credentials` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_cap_client_keeps_static_bearer_fallback` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_cap_client_posts_ai_usage_log` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |

## 6. 경로·엔드포인트 단서

- `/claimTranscription`
- `/logAiUsage`

## 7. 코드 흐름 상세

### 7.1 `FakeResponse`

- 위치: 8~17행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.2 `FakeAsyncClient`

- 위치: 18~35행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.3 `test_cap_client_uses_xsuaa_client_credentials`

- 위치: 36~58행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.4 `test_cap_client_keeps_static_bearer_fallback`

- 위치: 59~73행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.5 `test_cap_client_posts_ai_usage_log`

- 위치: 74~95행
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
