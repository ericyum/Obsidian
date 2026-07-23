# test_api.py 완전 분석

> 소스: `tests/test_api.py`  
> 역할: 해당 기능의 정상·오류·경계 조건을 고정하는 자동화 테스트다.

## 1. 이 파일이 있는 이유

해당 기능의 정상·오류·경계 조건을 고정하는 자동화 테스트다.

이 파일은 **7. tests — 검증 체계** 영역에 속한다. 독립적으로 보기보다 아래의 호출자·의존 대상·데이터 계약과 함께 읽어야 한다.

## 2. 파일 개요

| 항목 | 값 |
|---|---|
| 경로 | `tests/test_api.py` |
| 형식 | `python` |
| 전체 줄 수 | 199 |
| 주석 줄 수 | 3 |
| 주요 심볼 수 | 16 |
| 환경 변수 참조 수 | 0 |

## 3. 의존성

- `io`
- `fastapi.testclient`
- `app.models`
- `app.pipeline.orchestrator`
- `app.main`

## 4. 주요 선언과 책임

| 선언 | 읽을 때 확인할 점 |
|---|---|
| `_fake_deps` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_upload_and_status` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_status_404` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_download_markdown` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_upload_with_title_participants` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_existing_meeting_category_can_be_updated` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_patch_speakers` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_upload_records_duration` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_upload_records_speaker_bounds` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_upload_rejects_invalid_speaker_bounds` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_download_rejects_non_markdown_formats` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_list_meetings` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_get_job_falls_back_to_sqlite` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_update_meeting_title` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_delete_meeting_removes_row_and_audio` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_patch_speakers_persists_to_sqlite` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |

## 6. 경로·엔드포인트 단서

- `/api/jobs`
- `/api/jobs/{job_id}`
- `/api/jobs/nope`
- `/api/jobs/{job_id}/download?format=md`
- `/api/jobs/{job_id}/download`
- `/api/meetings/{job_id}/category`
- `/api/jobs/{job_id}/speakers`
- `/api/jobs/nope/speakers`
- `/api/jobs/{job_id}/download?format=txt`
- `/api/jobs/{job_id}/download?format=srt`
- `/api/jobs/{job_id}/download?format=json`
- `/api/meetings`
- `/api/jobs/{jid}`
- `/api/meetings/{jid}/title`
- `/api/meetings/{jid}`
- `/api/jobs/{jid}/speakers`

## 7. 코드 흐름 상세

### 7.1 `_fake_deps`

- 위치: 8~16행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.2 `test_upload_and_status`

- 위치: 17~30행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.3 `test_status_404`

- 위치: 31~36행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.4 `test_download_markdown`

- 위치: 37~50행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.5 `test_upload_with_title_participants`

- 위치: 51~66행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.6 `test_existing_meeting_category_can_be_updated`

- 위치: 67~82행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.7 `test_patch_speakers`

- 위치: 83~95행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.8 `test_upload_records_duration`

- 위치: 96~105행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.9 `test_upload_records_speaker_bounds`

- 위치: 106~116행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.10 `test_upload_rejects_invalid_speaker_bounds`

- 위치: 117~125행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.11 `test_download_rejects_non_markdown_formats`

- 위치: 126~139행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.12 `test_list_meetings`

- 위치: 140~151행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.13 `test_get_job_falls_back_to_sqlite`

- 위치: 152~165행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.14 `test_update_meeting_title`

- 위치: 166~176행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.15 `test_delete_meeting_removes_row_and_audio`

- 위치: 177~188행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.16 `test_patch_speakers_persists_to_sqlite`

- 위치: 189~199행
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
