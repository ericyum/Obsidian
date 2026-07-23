# main.py 완전 분석

> 소스: `legacy/fastapi/app/main.py`  
> 역할: 현재 CAP 중심 경로 이전의 FastAPI 단일 애플리케이션 구현을 보존한 코드다.

## 1. 이 파일이 있는 이유

현재 CAP 중심 경로 이전의 FastAPI 단일 애플리케이션 구현을 보존한 코드다.

이 파일은 **8. legacy — 이전 FastAPI 구조** 영역에 속한다. 독립적으로 보기보다 아래의 호출자·의존 대상·데이터 계약과 함께 읽어야 한다.

## 2. 파일 개요

| 항목 | 값 |
|---|---|
| 경로 | `legacy/fastapi/app/main.py` |
| 형식 | `python` |
| 전체 줄 수 | 314 |
| 주석 줄 수 | 1 |
| 주요 심볼 수 | 5 |
| 환경 변수 참조 수 | 0 |

## 3. 의존성

- `__future__`
- `json`
- `re`
- `time`
- `datetime`
- `pathlib`
- `typing`
- `urllib.parse`
- `fastapi`
- `fastapi.responses`
- `fastapi.staticfiles`
- `app.jobs`
- `app.models`
- `app.pipeline.orchestrator`
- `app.formatters`
- `app`

## 4. 주요 선언과 책임

| 선언 | 읽을 때 확인할 점 |
|---|---|
| `_safe_filename` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `create_app` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `_append_error` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `_now_iso` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `_validate_speaker_bounds` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |

## 6. 경로·엔드포인트 단서

- `/api/jobs`
- `/api/meetings`
- `/api/jobs/{job_id}`
- `/api/meetings/{job_id}`
- `/api/meetings/{job_id}/title`
- `/api/meetings/{job_id}/category`
- `/api/jobs/{job_id}/speakers`
- `/api/jobs/{job_id}/speakers/{speaker}`
- `/api/jobs/{job_id}/transcript/speaker`
- `/api/jobs/{job_id}/transcript`
- `/api/jobs/{job_id}/note`
- `/api/jobs/{job_id}/summarize`
- `/api/jobs/{job_id}/transcript-review`
- `/api/jobs/{job_id}/download`

## 7. 코드 흐름 상세

### 7.1 `_safe_filename`

- 위치: 22~28행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.2 `create_app`

- 위치: 29~291행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.3 `_append_error`

- 위치: 292~295행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.4 `_now_iso`

- 위치: 296~299행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.5 `_validate_speaker_bounds`

- 위치: 300~314행
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
