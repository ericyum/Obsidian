# ai-core.js 완전 분석

> 소스: `cap/srv/lib/ai-core.js`  
> 역할: 전사를 chunk/reduce 방식으로 요약하고 검토 항목을 생성하며 AI Core 응답 JSON을 검증·복구한다.

## 1. 이 파일이 있는 이유

전사를 chunk/reduce 방식으로 요약하고 검토 항목을 생성하며 AI Core 응답 JSON을 검증·복구한다.

이 파일은 **3. cap — 데이터와 백엔드** 영역에 속한다. 독립적으로 보기보다 아래의 호출자·의존 대상·데이터 계약과 함께 읽어야 한다.

## 2. 파일 개요

| 항목 | 값 |
|---|---|
| 경로 | `cap/srv/lib/ai-core.js` |
| 형식 | `javascript` |
| 전체 줄 수 | 1068 |
| 주석 줄 수 | 0 |
| 주요 심볼 수 | 55 |
| 환경 변수 참조 수 | 27 |

## 3. 의존성

- `./meeting-note`
- `./ai-usage-log`

## 4. 주요 선언과 책임

| 선언 | 읽을 때 확인할 점 |
|---|---|
| `summarizeMeeting` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `summarizeText` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `summaryMessages` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `categorySummaryGuidance` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `reducePartialSummaries` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `summarizePartialGroup` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `groupPartialSummaries` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `reviewMeeting` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `normalizeReviewItems` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `fallbackReviewItems` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `noteToSearchText` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `looksSuspicious` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `hasLlmConfig` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `missingAicoreConfig` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `assertAiCoreConfig` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `chatJson` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `chatJsonRaw` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `parseAndValidateChatJsonContent` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `unwrapSchemaPayload` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `repairChatJsonContent` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `parseChatJsonContent` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `retryMessages` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `repairMessages` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `validateJsonSchema` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `matchesJsonSchemaType` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `jsonType` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `summaryAttempts` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `summaryChunkChars` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `summaryReduceChars` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `allowSummaryFallback` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `llmProvider` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `chatJsonViaOpenAiCompatible` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `chatJsonViaAicore` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `getAicoreToken` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `aicoreCompletionBody` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `aicoreCompletionMode` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `aicoreUsageSource` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `isAnthropicModel` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `aicoreDirectMessages` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `aicoreConfigRef` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `aicorePlaceholderValues` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `orchestrationPurposeEnv` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `orchestrationPurposeKey` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `aicoreTokenUrl` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `aicoreCompletionUrl` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `extractAicoreContent` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `parseJsonText` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `stripJsonFence` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `truncateForError` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `positiveIntEnv` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `truthyEnv` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `clean` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `aicoreCredentials` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `vcapAicoreCredentials` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `parseJsonEnv` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |

## 5. 환경 변수

- `CAP_LLM_BASE_URL`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `OPENAI_BASE_URL`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `CAP_LLM_API_KEY`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `OPENAI_API_KEY`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `CAP_LLM_PROVIDER`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `LLM_PROVIDER`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `CAP_LLM_CHAT_PATH`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `CAP_LLM_MODEL`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `AICORE_MODEL`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `OPENAI_MODEL`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `CAP_AICORE_ORCHESTRATION_MODE`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `ORCHESTRATION_CONFIG_ID`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `ORCHESTRATION_CONFIG_NAME`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `ORCHESTRATION_CONFIG_SCENARIO`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `ORCHESTRATION_CONFIG_VERSION`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `CAP_AICORE_TEMPLATE_PLACEHOLDERS`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `AICORE_TEMPLATE_PLACEHOLDERS`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `CAP_AICORE_COMPLETION_URL`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `AICORE_COMPLETION_URL`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `AICORE_DEPLOYMENT_ID`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `AICORE_CLIENT_ID`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `AICORE_CLIENT_SECRET`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `AICORE_AUTH_URL`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `AICORE_API_URL`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `AICORE_BASE_URL`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `AICORE_RESOURCE_GROUP`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `AICORE_VCAP_NAME`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.

## 6. 경로·엔드포인트 단서

- `/v1/chat/completions`

## 7. 코드 흐름 상세

### 7.1 `summarizeMeeting`

- 위치: 114~151행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.2 `summarizeText`

- 위치: 152~180행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.3 `summaryMessages`

- 위치: 181~210행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.4 `categorySummaryGuidance`

- 위치: 211~217행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.5 `reducePartialSummaries`

- 위치: 218~237행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.6 `summarizePartialGroup`

- 위치: 238~289행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.7 `groupPartialSummaries`

- 위치: 290~313행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.8 `reviewMeeting`

- 위치: 314~350행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.9 `normalizeReviewItems`

- 위치: 351~372행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.10 `fallbackReviewItems`

- 위치: 373~395행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.11 `noteToSearchText`

- 위치: 396~405행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.12 `looksSuspicious`

- 위치: 406~412행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.13 `hasLlmConfig`

- 위치: 413~422행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.14 `missingAicoreConfig`

- 위치: 423~434행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.15 `assertAiCoreConfig`

- 위치: 435~442행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.16 `chatJson`

- 위치: 443~464행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.17 `chatJsonRaw`

- 위치: 465~470행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.18 `parseAndValidateChatJsonContent`

- 위치: 471~483행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.19 `unwrapSchemaPayload`

- 위치: 484~497행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.20 `repairChatJsonContent`

- 위치: 498~523행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.21 `parseChatJsonContent`

- 위치: 524~535행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.22 `retryMessages`

- 위치: 536~547행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.23 `repairMessages`

- 위치: 548~568행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.24 `validateJsonSchema`

- 위치: 569~614행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.25 `matchesJsonSchemaType`

- 위치: 615~623행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.26 `jsonType`

- 위치: 624~629행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.27 `summaryAttempts`

- 위치: 630~633행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.28 `summaryChunkChars`

- 위치: 634~637행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.29 `summaryReduceChars`

- 위치: 638~641행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.30 `allowSummaryFallback`

- 위치: 642~645행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.31 `llmProvider`

- 위치: 646~653행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.32 `chatJsonViaOpenAiCompatible`

- 위치: 654~694행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.33 `chatJsonViaAicore`

- 위치: 695~773행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.34 `getAicoreToken`

- 위치: 774~802행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.35 `aicoreCompletionBody`

- 위치: 803~849행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.36 `aicoreCompletionMode`

- 위치: 850~854행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.37 `aicoreUsageSource`

- 위치: 855~859행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.38 `isAnthropicModel`

- 위치: 860~863행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.39 `aicoreDirectMessages`

- 위치: 864~876행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.40 `aicoreConfigRef`

- 위치: 877~908행
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
