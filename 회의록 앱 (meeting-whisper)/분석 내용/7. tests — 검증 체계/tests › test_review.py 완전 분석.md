# test_review.py 완전 분석

> 소스: `tests/test_review.py`  
> 역할: 해당 기능의 정상·오류·경계 조건을 고정하는 자동화 테스트다.

## 1. 이 파일이 있는 이유

해당 기능의 정상·오류·경계 조건을 고정하는 자동화 테스트다.

이 파일은 **7. tests — 검증 체계** 영역에 속한다. 독립적으로 보기보다 아래의 호출자·의존 대상·데이터 계약과 함께 읽어야 한다.

## 2. 파일 개요

| 항목 | 값 |
|---|---|
| 경로 | `tests/test_review.py` |
| 형식 | `python` |
| 전체 줄 수 | 313 |
| 주석 줄 수 | 0 |
| 주요 심볼 수 | 12 |
| 환경 변수 참조 수 | 0 |

## 3. 의존성

- `json`
- `pytest`
- `app.models`
- `app.pipeline.review`

## 4. 주요 선언과 책임

| 선언 | 읽을 때 확인할 점 |
|---|---|
| `_FakeResp` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `_FakeClient` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_review_transcript_parses_json_items` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_review_transcript_drops_out_of_range_items` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_review_prompt_includes_json_contract_for_template_mode` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_transcript_to_indexed_text_caps_long_transcripts_to_suspect_segments` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_review_transcript_reports_non_json_preview` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_review_transcript_filters_sentence_polishing_candidates` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_review_transcript_filters_known_clear_terms` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_review_transcript_adds_heuristic_ambiguous_terms` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_review_transcript_with_note_only_returns_terms_in_summary` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_review_transcript_marks_missing_evidence_segment` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |

## 7. 코드 흐름 상세

### 7.1 `_FakeResp`

- 위치: 9~13행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.2 `_FakeClient`

- 위치: 14~30행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.3 `test_review_transcript_parses_json_items`

- 위치: 31~66행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.4 `test_review_transcript_drops_out_of_range_items`

- 위치: 67~83행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.5 `test_review_prompt_includes_json_contract_for_template_mode`

- 위치: 84~95행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.6 `test_transcript_to_indexed_text_caps_long_transcripts_to_suspect_segments`

- 위치: 96~112행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.7 `test_review_transcript_reports_non_json_preview`

- 위치: 113~120행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.8 `test_review_transcript_filters_sentence_polishing_candidates`

- 위치: 121~163행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.9 `test_review_transcript_filters_known_clear_terms`

- 위치: 164~214행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.10 `test_review_transcript_adds_heuristic_ambiguous_terms`

- 위치: 215~249행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.11 `test_review_transcript_with_note_only_returns_terms_in_summary`

- 위치: 250~293행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.12 `test_review_transcript_marks_missing_evidence_segment`

- 위치: 294~313행
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
