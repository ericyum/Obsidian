# test_worker_aicore_transcriber.py 완전 분석

> 소스: `tests/test_worker_aicore_transcriber.py`  
> 역할: 해당 기능의 정상·오류·경계 조건을 고정하는 자동화 테스트다.

## 1. 이 파일이 있는 이유

해당 기능의 정상·오류·경계 조건을 고정하는 자동화 테스트다.

이 파일은 **7. tests — 검증 체계** 영역에 속한다. 독립적으로 보기보다 아래의 호출자·의존 대상·데이터 계약과 함께 읽어야 한다.

## 2. 파일 개요

| 항목 | 값 |
|---|---|
| 경로 | `tests/test_worker_aicore_transcriber.py` |
| 형식 | `python` |
| 전체 줄 수 | 535 |
| 주석 줄 수 | 1 |
| 주요 심볼 수 | 13 |
| 환경 변수 참조 수 | 0 |

## 3. 의존성

- `json`
- `workers.transcription.app`
- `workers.transcription.app.config`
- `workers.transcription.app.transcriber`

## 4. 주요 선언과 책임

| 선언 | 읽을 때 확인할 점 |
|---|---|
| `FakeResponse` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_aicore_engine_uses_aicore_transcriber` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_content_to_transcript_accepts_json_segments` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_run_aicore_transcription_posts_file_input` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_run_aicore_transcription_chunks_large_audio` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_expected_duration_forces_chunking_when_media_duration_is_unknown` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_content_to_transcript_rejects_malformed_json_and_backwards_timestamps` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_content_to_transcript_repairs_compact_mmss_timestamps` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_content_to_transcript_does_not_guess_ambiguous_timestamp_format` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_single_chunk_retries_invalid_model_json` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_invalid_chunk_falls_back_to_smaller_chunks` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_invalid_smaller_chunk_is_split_recursively` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_aicore_error_message_keeps_payload_small` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |

## 7. 코드 흐름 상세

### 7.1 `FakeResponse`

- 위치: 8~18행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.2 `test_aicore_engine_uses_aicore_transcriber`

- 위치: 19~55행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.3 `test_content_to_transcript_accepts_json_segments`

- 위치: 56~79행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.4 `test_run_aicore_transcription_posts_file_input`

- 위치: 80~172행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.5 `test_run_aicore_transcription_chunks_large_audio`

- 위치: 173~258행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.6 `test_expected_duration_forces_chunking_when_media_duration_is_unknown`

- 위치: 259~288행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.7 `test_content_to_transcript_rejects_malformed_json_and_backwards_timestamps`

- 위치: 289~318행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.8 `test_content_to_transcript_repairs_compact_mmss_timestamps`

- 위치: 319~340행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.9 `test_content_to_transcript_does_not_guess_ambiguous_timestamp_format`

- 위치: 341~358행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.10 `test_single_chunk_retries_invalid_model_json`

- 위치: 359~410행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.11 `test_invalid_chunk_falls_back_to_smaller_chunks`

- 위치: 411~450행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.12 `test_invalid_smaller_chunk_is_split_recursively`

- 위치: 451~504행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.13 `test_aicore_error_message_keeps_payload_small`

- 위치: 505~535행
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
