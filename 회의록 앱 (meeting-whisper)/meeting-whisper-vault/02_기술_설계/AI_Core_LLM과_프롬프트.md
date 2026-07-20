# AI Core LLM과 프롬프트

최종 갱신: 2026-07-20

기준 소스:

- `cap/srv/lib/ai-core.js`
- `cap/srv/lib/meeting-note.js`
- `mta.yaml`

## 역할

CAP은 SAP AI Core Orchestration을 사용해 다음 두 작업을 수행한다.

1. transcript를 구조화된 회의 요약으로 변환
2. 저장된 요약에서 사람이 확인할 검토 후보 생성

STT는 이 문서의 LLM 경로가 아니라 CF Worker의 `gemini-2.5-flash` 경로다.

## 현재 호출 모드

| 작업 | 모드 | 모델 결정 기준 |
| --- | --- | --- |
| 요약 | registered template | `meeting-whisper-summary-ko-v1 / orchestration / 0.0.1`에 등록된 모델 |
| 검토 | direct orchestration | `CAP_LLM_MODEL` (`anthropic--claude-4.7-opus`) |
| STT | Worker AI Core STT | `AICORE_STT_MODEL` (`gemini-2.5-flash`) |

현재 주요 환경값:

```text
CAP_LLM_PROVIDER=aicore
CAP_AICORE_ORCHESTRATION_MODE=direct
CAP_AICORE_SUMMARY_ORCHESTRATION_MODE=template
CAP_AICORE_SUMMARY_CONFIG_NAME=meeting-whisper-summary-ko-v1
CAP_AICORE_SUMMARY_CONFIG_SCENARIO=orchestration
CAP_AICORE_SUMMARY_CONFIG_VERSION=0.0.1
CAP_AICORE_SUMMARY_TEMPLATE_PLACEHOLDERS=user_prompt,json_schema
CAP_LLM_MODEL=anthropic--claude-4.7-opus
```

위 값은 레포의 배포 기본값이며 `mta.yaml`을 기준으로 확인한다. 이미 배포된 앱은 Cloud Foundry 환경변수로 덮어쓴 값이 있을 수 있으므로 런타임 환경도 함께 확인한다.

### `template`과 `direct`의 모델 결정 차이

현재 요약과 검토는 같은 SAP AI Core Orchestration endpoint를 사용하지만 요청 본문과 모델 결정 위치가 다르다.

#### 요약: template가 모델을 결정

- `summarizeMeeting`과 부분 요약·결합 호출은 `orchestrationPurpose: summary`를 전달한다.
- `CAP_AICORE_SUMMARY_ORCHESTRATION_MODE=template`이므로 CAP은 동적 model config 대신 `config_ref`로 등록 template를 호출한다.
- 현재 placeholder 허용 목록은 `user_prompt,json_schema`뿐이어서 `CAP_LLM_MODEL` 값은 요약 요청의 placeholder로도 전달되지 않는다.
- 따라서 요약의 실제 모델은 AI Core에 등록된 `meeting-whisper-summary-ko-v1 / orchestration / 0.0.1` 설정이 결정한다. `CAP_LLM_MODEL`을 바꿔도 이 template의 모델은 바뀌지 않는다.

#### 검토: direct 요청이 모델을 결정

- `reviewMeeting`은 별도 purpose별 template 설정 없이 전역 `CAP_AICORE_ORCHESTRATION_MODE=direct`를 따른다.
- CAP은 prompt와 schema로 Orchestration 요청을 동적으로 구성한다.
- 요청의 model config에는 `CAP_LLM_MODEL`을 `name`으로, `latest`를 `version`으로 넣는다.
- 현재 레포 기본값 기준으로 검토 호출 모델은 `anthropic--claude-4.7-opus`다.

```yaml
model:
  name: anthropic--claude-4.7-opus
  version: latest
```

코드의 direct 모델 선택 우선순위는 `CAP_LLM_MODEL` → `AICORE_MODEL` → `gpt-4o-mini`다. AI usage 응답에 모델명이 없을 때도 이 값이 로그의 기본 모델명으로 사용된다.

## 요약 입력

요약 prompt에는 다음 정보가 들어간다.

- 회의 카테고리
- 참여자
- 시간순 transcript
- 원하는 JSON schema
- 회의록 작성 규칙

모델에 요구하는 핵심 원칙:

- transcript에 없는 사실을 만들지 않는다.
- 결정 사항과 액션 아이템은 실제 근거가 있을 때만 작성한다.
- 사람·회사·제품명과 숫자를 임의로 교정하지 않는다.
- 불확실한 표현은 보수적으로 유지한다.
- 여러 chunk를 처리해도 시간순 흐름을 보존한다.

상세 prompt 문구의 최종 기준은 `cap/srv/lib/ai-core.js`다. Vault에 prompt 전체 사본을 중복 보관하지 않는다.

## 결과 schema

요약은 다음 구조로 정규화한다.

```json
{
  "summary": "string",
  "decisions": ["string"],
  "action_items": [
    {
      "owner": "string",
      "task": "string",
      "due": "string"
    }
  ],
  "topics": [
    {
      "title": "string",
      "points": ["string"]
    }
  ]
}
```

CAP은 결과를 `normalizeNote()`로 정규화하고 JSON과 Markdown을 함께 저장한다.

## 카테고리별 guidance

`CATEGORY_SUMMARY_GUIDANCE`가 카테고리별 강조점을 추가한다.

예를 들어 `내부 세미나 참석`은 다음을 우선한다.

- 발표 주제와 전체 흐름
- 핵심 개념, 기술, 서비스와 사례
- 명시된 수치와 로드맵
- 발표자의 제안과 시사점
- 실제 합의가 없으면 결정·액션을 생성하지 않음

카테고리를 추가하거나 이름을 바꾸면 UI 목록과 `CATEGORY_SUMMARY_GUIDANCE`를 함께 확인한다.

## 긴 transcript 요약

```text
전체 transcript
  -> 12,000자 ordered chunk
  -> chunk별 partial meeting note
  -> 16,000자 단위 partial group
  -> 필요 시 반복 reduce
  -> 최종 meeting note 1개
```

| 환경변수 | 기본값 |
| --- | ---: |
| `CAP_LLM_SUMMARY_CHUNK_CHARS` | `12000` |
| `CAP_LLM_SUMMARY_REDUCE_CHARS` | `16000` |
| `CAP_LLM_SUMMARY_ATTEMPTS` | `2` |

partial summary는 내부 처리에만 사용하고 별도 회의록으로 저장하지 않는다.

## 검토 후보

검토는 현재 transcript 전체가 아니라 저장된 summary note를 대상으로 한다.

주요 후보:

- 이름
- 회사·제품·서비스명
- 기술 용어와 약어
- 숫자와 날짜
- 문맥상 어색한 표현

결과는 `ReviewItems`에 저장하고 사용자가 요약에 적용하거나 무시할 수 있다.

## 실패와 재시도

- 요약·reduce 호출은 기본 2회 시도한다.
- transcript 검증이 실패하면 요약을 시작하지 않는다.
- summary만 실패하면 transcript는 유지되므로 사용자가 요약을 다시 실행할 수 있다.
- 동시 요약/검토 요청은 중복 실행되지 않도록 막는다.

## AI usage

AI Core 호출은 가능한 경우 `AiCoreUsageLogs`에 다음 정보를 남긴다.

- model/config/request ID
- prompt/completion/total token
- latency
- success/error 상태
- response preview와 오류 메시지

운영 내보내기 방법은 [세션과 운영도구](../01_운영/세션과_운영도구.md)를 따른다.

## 변경 체크리스트

- 호출 모드를 `template`과 `direct` 사이에서 바꿀 때 모델 결정 위치도 확인했는가
- direct 모델 변경은 `CAP_LLM_MODEL`, summary 모델 변경은 AI Core 등록 template에서 각각 수행했는가
- template placeholder와 CAP 설정이 일치하는가
- JSON schema 변경이 UI formatter와 저장 모델에 반영됐는가
- 카테고리 추가가 UI와 guidance 양쪽에 반영됐는가
- 긴 transcript chunk/reduce 테스트가 통과하는가
- error/usage log에 secret이 포함되지 않는가
