# test_editing_api.py 완전 분석

> 소스: `tests/test_editing_api.py`  
> 역할: 해당 기능의 정상·오류·경계 조건을 고정하는 자동화 테스트다.

## 1. 이 파일이 있는 이유

해당 기능의 정상·오류·경계 조건을 고정하는 자동화 테스트다.

이 파일은 **7. tests — 검증 체계** 영역에 속한다. 독립적으로 보기보다 아래의 호출자·의존 대상·데이터 계약과 함께 읽어야 한다.

## 2. 파일 개요

| 항목 | 값 |
|---|---|
| 경로 | `tests/test_editing_api.py` |
| 형식 | `python` |
| 전체 줄 수 | 245 |
| 주석 줄 수 | 0 |
| 주요 심볼 수 | 12 |
| 환경 변수 참조 수 | 0 |

## 3. 의존성

- `io`
- `fastapi.testclient`
- `app`
- `app.main`
- `app.models`
- `app.pipeline.orchestrator`

## 4. 주요 선언과 책임

| 선언 | 읽을 때 확인할 점 |
|---|---|
| `_deps` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `_create_job` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_patch_transcript_clears_note_and_persists` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_patch_note_persists_manual_edit` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_patch_note_clears_transcript_review_items` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_patch_note_can_persist_remaining_review_items` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_run_transcript_review_preserves_existing_note` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_run_transcript_review_marks_done_when_no_items` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_run_transcript_review_returns_error_on_failure` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_patch_segment_speaker_preserves_note` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_delete_speaker_clears_assigned_segments` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_rerun_summary_uses_edited_transcript` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |

## 6. 경로·엔드포인트 단서

- `/api/jobs`
- `/api/jobs/{jid}/transcript`
- `/api/jobs/{jid}`
- `/api/jobs/{jid}/note`
- `/api/jobs/{jid}/transcript-review`
- `/api/jobs/{jid}/transcript/speaker`
- `/api/jobs/{jid}/speakers`
- `/api/jobs/{jid}/speakers/SPEAKER_02`
- `/api/jobs/{jid}/summarize`

## 7. 코드 흐름 상세

### 7.1 `_deps`

- 위치: 11~23행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.2 `_create_job`

- 위치: 24~31행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.3 `test_patch_transcript_clears_note_and_persists`

- 위치: 32~53행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.4 `test_patch_note_persists_manual_edit`

- 위치: 54~71행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.5 `test_patch_note_clears_transcript_review_items`

- 위치: 72~100행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.6 `test_patch_note_can_persist_remaining_review_items`

- 위치: 101~131행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.7 `test_run_transcript_review_preserves_existing_note`

- 위치: 132~157행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.8 `test_run_transcript_review_marks_done_when_no_items`

- 위치: 158~172행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.9 `test_run_transcript_review_returns_error_on_failure`

- 위치: 173~193행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.10 `test_patch_segment_speaker_preserves_note`

- 위치: 194~210행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.11 `test_delete_speaker_clears_assigned_segments`

- 위치: 211~228행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.12 `test_rerun_summary_uses_edited_transcript`

- 위치: 229~245행
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
