# test_aicore_client.py 완전 분석

> 소스: `tests/test_aicore_client.py`  
> 역할: 해당 기능의 정상·오류·경계 조건을 고정하는 자동화 테스트다.

## 1. 이 파일이 있는 이유

해당 기능의 정상·오류·경계 조건을 고정하는 자동화 테스트다.

이 파일은 **7. tests — 검증 체계** 영역에 속한다. 독립적으로 보기보다 아래의 호출자·의존 대상·데이터 계약과 함께 읽어야 한다.

## 2. 파일 개요

| 항목 | 값 |
|---|---|
| 경로 | `tests/test_aicore_client.py` |
| 형식 | `python` |
| 전체 줄 수 | 216 |
| 주석 줄 수 | 0 |
| 주요 심볼 수 | 8 |
| 환경 변수 참조 수 | 0 |

## 3. 의존성

- `json`
- `app.pipeline.aicore`

## 4. 주요 선언과 책임

| 선언 | 읽을 때 확인할 점 |
|---|---|
| `_FakeService` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `_FakeUnusedParameterError` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `_RejectUnusedService` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_aicore_client_maps_chat_completion_to_orchestration_placeholders` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_aicore_client_can_call_direct_config_without_template_ref` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_template_mode_retries_without_unused_placeholders` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_template_mode_can_limit_placeholder_values` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `test_template_mode_can_use_combined_prompt_for_single_variable_templates` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |

## 7. 코드 흐름 상세

### 7.1 `_FakeService`

- 위치: 6~28행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.2 `_FakeUnusedParameterError`

- 위치: 29~32행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.3 `_RejectUnusedService`

- 위치: 33~53행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.4 `test_aicore_client_maps_chat_completion_to_orchestration_placeholders`

- 위치: 54~91행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.5 `test_aicore_client_can_call_direct_config_without_template_ref`

- 위치: 92~112행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.6 `test_template_mode_retries_without_unused_placeholders`

- 위치: 113~149행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.7 `test_template_mode_can_limit_placeholder_values`

- 위치: 150~182행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.8 `test_template_mode_can_use_combined_prompt_for_single_variable_templates`

- 위치: 183~216행
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
