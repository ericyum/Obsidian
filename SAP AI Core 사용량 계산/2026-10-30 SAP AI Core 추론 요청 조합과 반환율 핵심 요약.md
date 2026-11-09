# SAP AI Core 공개 추론 텍스트 — 설정별 반환율과 권장 설정

기준: 2026-11-01 조사 결과. **모든 입력에서 추론 텍스트를 반드시 반환하도록 보장하는 파라미터 설정은 확인되지 않았다.** 아래 설정은 반환 가능성을 높이는 후보이며, 보장 수단은 아니다.

## 1. 호출 방법과 파라미터

| 호출 방법 | 대상 | 추론 관련 파라미터 | 응답 방식 |
| --- | --- | --- | --- |
| Native `/responses` | GPT Luna·Terra·Sol | `reasoning.effort`, `reasoning.summary`; 보조 `reasoning.context`, `reasoning.mode` | `stream:false` JSON / `true` SSE |
| Orchestration `/v2/completion` — 공통 | GPT·Claude | `reasoning_effort` | `config.stream.enabled` 생략 JSON / `true` SSE |
| 같은 Orchestration — Claude 고유 설정 | Claude | `thinking.type`, `thinking.budget_tokens`, `thinking.display`, `output_config.effort` | 위와 동일 |

Orchestration의 위 모델 파라미터는 `config.modules.prompt_templating.model.params` 안에 넣는다. Claude 공통 effort 방식과 고유 thinking 방식은 별도로 비교했으며, 명시한 충돌 대조 외에는 함께 보내지 않았다.

**표 읽기:** 값은 **JSON / SSE** 순서이며, 두 값이 같으면 하나만 썼다. 보통 **각 설정·응답 방식별 10회**다. 확률은 모집단 추정치가 아니라 **공개 추론을 반환한 정상 완료 수 ÷ 정상 완료 수**인 관측률이다. 예: 90%=9/10. 분모가 다른 셀은 횟수를 썼다. 거절은 요청 실패이고, 미측정은 확률을 산정하지 않았다는 뜻이다. 내부 추론 토큰·암호문·최종 답변의 설명은 공개 추론으로 세지 않았다.

같은 산술 질문을 사용했다. Native 출력 한도4096, Orchestration32000으로 고정했다. 아래는 추론 관련 파라미터의 **측정한 조합**이며, 모든 API 파라미터·입력·출력 한도의 무한한 조합을 전수 측정한 것은 아니다.

## 2. GPT Native: effort × summary

### Luna 5.6

| effort | summary 생략 | auto | concise | detailed |
| --- | ---: | ---: | ---: | ---: |
| 생략 | 0% | 90% / 100% | 100% | 100% |
| none | 0% | 0% | 0% | 0% |
| minimal | 거절 | 거절 | 거절 | 거절 |
| low | 0% | 100% | 100% | 100% |
| medium | 0% | 100% | 100% | 100% |
| high | 0% | 100% | 100% | 100% |
| xhigh | 0% | 100% | 100% | 100% |
| max | 0% | 100% | 100% | 100% |

### Terra 5.6

| effort  | summary 생략 | auto |    concise |   detailed |
| ------- | ---------: | ---: | ---------: | ---------: |
| 생략      |         0% | 100% |       100% |       100% |
| none    |         0% |   0% |         0% |         0% |
| minimal |         거절 |   거절 |         거절 |         거절 |
| low     |         0% | 100% |       100% |       100% |
| medium  |         0% | 100% |       100% | 90% / 100% |
| high    |         0% | 100% | 90% / 100% |       100% |
| xhigh   |         0% | 100% |       100% | 90% / 100% |
| max     |         0% | 100% |       100% |       100% |

### Sol 5.6 (과거)

| effort | summary 생략 | auto | concise | detailed |
| --- | ---: | ---: | ---: | ---: |
| 생략 | 0% | 0% | 0% | 0% |
| none | 0% | 0% | 0% | 0% |
| minimal | 거절 | 거절 | 거절 | 거절 |
| low | 0% | 10% / 20% | 10% / 30% | 0% / 10% |
| medium | 0% | 0% | 0% | 0% |
| high | 0% | 10% / 0% | 10% / 0% | 10% / 0% |
| xhigh | 0% | 0% / 20% | 30% / 10% | 10% / 0% |
| max | 0% | 100% | 100% | 100% |

### Sol 6.1

| effort | summary 생략 | auto | concise | detailed |
| --- | ---: | ---: | ---: | ---: |
| 생략 | 0% | 100% | 100% | 100% / 70% |
| none | 거절 | 거절 | 거절 | 거절 |
| minimal | 거절 | 거절 | 거절 | 거절 |
| low | 0% / 0% (0/9) | 60% / 70% | 80% / 60% | 70% / 50% |
| medium | 0% | 80% | 100% | 100% / 90% |
| high | 0% | 100% | 100% | 100% |
| xhigh | 0% | 100% | 100% | 100% |
| max | 0% | 100% | 100% | 100% |

Sol6.1의 low+summary생략/SSE는 10회 중 인프라 오류1건을 제외한 정상9회에서 공개0회다. summary 생략 시 내부 추론 토큰이 있어도 공개 텍스트가 없을 수 있다.

## 3. Orchestration: 공통 reasoning_effort

| reasoning_effort | Luna5.6 | Terra5.6 | Sol5.6(과거) | Haiku4.5 | Sonnet4.6 | Opus4.8 |
| ---------------- | ------: | -------: | ---------: | -------: | --------: | ------: |
| 생략               |      0% |       0% |         0% |       0% |        0% |      0% |
| none             |      0% |       0% |         0% |       0% |        0% |      0% |
| minimal          |      거절 |       거절 |         거절 |     100% |      100% |    100% |
| low              |      0% |       0% |         0% |     100% |      100% |    100% |
| medium           |      0% |       0% |         0% |     100% |      100% |    100% |
| high             |      0% |       0% |         0% |     100% |      100% |    100% |
| xhigh            |      거절 |       거절 |         거절 |     100% |        거절 |    100% |
| max              |      거절 |       거절 |         거절 |     100% |      100% |    100% |

**Sol6.1은 현재 Orchestration 배포에서 모델명 미지원으로 거절**됐다. Native 결과를 이 경로에 그대로 적용할 수 없다.

## 4. Claude 고유 thinking 설정

`display`의 생략은 필드를 보내지 않은 것이고, `omitted`는 문자열 값을 전송한 것이다.

| thinking.type | budget_tokens | display | 함께 보낸 공통 effort | Haiku4.5 | Sonnet4.6 | Opus4.8 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| disabled | — | — | 생략 | 0% | 0% | 0% |
| enabled | 1024 | 생략 | 생략 | 100% | 100% | 거절 |
| enabled | 1024 | summarized | 생략 | 100% | 100% | 거절 |
| enabled | 1024 | omitted | 생략 | 0% | 0% | 거절 |
| adaptive | — | 생략 | 생략 | 거절 | 100% | 100% |
| adaptive | — | summarized | 생략 | 거절 | 100% | 100% |
| adaptive | — | omitted | 생략 | 거절 | 100% | 100% |
| adaptive | — | summarized | none | 거절 | 100% | 100% |
| disabled | — | — | high | 0% | 0% | 0% |

adaptive+display:omitted에서도 공개 텍스트가 관측됐다. 이것만으로 display 적용 여부나 상충 설정의 우선순위가 확인된 것은 아니다.

### budget_tokens 변경

`thinking:{type:"enabled",display:"summarized"}`, `max_tokens:32000` 고정.

| budget_tokens | Haiku4.5 | Sonnet4.6 |
| --- | ---: | ---: |
| 0 | 거절 | 거절 |
| 1 | 거절 | 거절 |
| 1023 | 거절 | 거절 |
| 1024 | 100% | 100% |
| 1025 | 100% | 100% |
| 4096 | 100% | 100% |
| 31999 | 100% | 100% |
| 32000 | 거절 | 거절 |
| 32001 | 거절 | 거절 |

Opus의 enabled는 위 기본 조합에서 거절되어 이 예산 조합은 측정하지 않았다. 예산을 크게 해도 항상 공개된다는 뜻은 아니다.

### adaptive의 output_config.effort 변경

`thinking:{type:"adaptive",display:"summarized"}` 고정. 공통 `reasoning_effort`와는 다른 필드다.

| output_config.effort | Sonnet4.6 | Opus4.8 |
| --- | ---: | ---: |
| 생략 | 100% | 100% |
| low | 100% | 100% |
| medium | 100% | 100% |
| high | 100% | 100% |
| xhigh | 미측정 | 100% |
| max | 100% | 100% |
| invalid (진단용 잘못된 값) | 100% | 100% |

잘못된 값 `invalid`도 수락·공개 반환된 관측이 있다. 필드가 의도대로 적용됐다고 단정할 수 없어 권장하지 않는다. Haiku의 adaptive는 거절됐다.

## 5. GPT Native 보조 설정: context × mode

`effort:high`, `summary:auto` 고정. 아래12조합은 모델별로 각각 측정했다. 단일 산술 입력 기준이므로 대화 이력에 따른 개선 효과를 뜻하지 않는다.

| reasoning.context | reasoning.mode | Luna5.6 | Terra5.6 | Sol6.1 |
| --- | --- | ---: | ---: | ---: |
| 생략 | 생략 | 100% | 100% | 100% |
| 생략 | standard | 100% | 100% | 100% |
| 생략 | pro | 100% | 100% (9/9) / 100% | 100% |
| auto | 생략 | 100% | 100% | 100% |
| auto | standard | 100% | 100% | 100% |
| auto | pro | 100% | 100% | 100% |
| current_turn | 생략 | 100% | 100% | 100% |
| current_turn | standard | 100% | 100% | 100% |
| current_turn | pro | 100% | 100% | 100% |
| all_turns | 생략 | 100% | 100% | 100% |
| all_turns | standard | 100% | 100% | 100% |
| all_turns | pro | 100% | 100% (9/9) / 100% | 100% |

Terra의 JSON 두 조건은 인프라 오류 각1건을 제외해9/9다. 요청 수락만으로 context/mode의 실제 적용을 입증하지 못하며, 항상 공개하게 만드는 옵션도 아니다. Sol5.6의 이 보강 조합은 미측정이다.

## 6. 결론: 어떤 설정을 쓰면 좋은가

| 모델·호출 방법 | 반환율을 높이기 위한 우선 후보 | 고정 산술 입력 관측 |
| --- | --- | --- |
| Luna5.6·Terra5.6·Sol6.1 / Native | `reasoning:{effort:"high",summary:"auto"}` | 기본10/10 및 별도 확인10/10, JSON·SSE 각각 |
| 과거 Sol5.6 / Native | `reasoning:{effort:"max",summary:"auto"}` | 기본10/10, JSON·SSE 각각 |
| Haiku4.5 / Orchestration | `reasoning_effort:"high"` 또는 `thinking:{type:"enabled",budget_tokens:1024,display:"summarized"}` | 기본10/10 및 별도 확인10/10, 각각 |
| Sonnet4.6 / Orchestration | `reasoning_effort:"high"` 또는 위 enabled 설정. adaptive+summarized도 지원 관측 | high·enabled 기본/별도 확인 각각10/10; adaptive 기본10/10 |
| Opus4.8 / Orchestration | `reasoning_effort:"high"` 또는 `thinking:{type:"adaptive",display:"summarized"}` | 기본10/10 및 별도 확인10/10, 각각 |

**반드시 반환하는 설정은 없다고 전제해야 한다.** Sol6.1은 같은 `max+auto`/SSE에서도 영어 인사7/10, 한국어 인사5/10이었다. 추론 요청 옵션과 실제 공개 여부는 다르며, 입력에 따라 달라진다. high를 max로 올리는 것만으로 해결되지 않는다.

도구 사용 시 마지막 HTTP 응답만 보면 앞 단계에서 받은 추론을 놓친다. Haiku·Sonnet의 한 도구 조건은 마지막 응답0/10이지만 같은 사용자 턴 누적은10/10이었다. **ZEN AI에는 위 후보 설정 + 같은 사용자 턴의 공개 추론 누적 + 미반환 상태 표시**를 적용하는 것이 타당하다.

요청별 로그·호출ID·실행 경과·비용 집계는 이 문서에서 제외했다. 공개 여부와 직접 관련된 설정표에 집중했으며, 입력·도구·파일·샘플링 설정까지 모두 교차한 보편 확률은 산정하지 않았다.

공식 문서: [OpenAI reasoning summaries](https://developers.openai.com/api/docs/guides/reasoning#reasoning-summaries), [Claude thinking·effort](https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost#effort-levels).

## 8. 2026-11-03 추가 조사 — ZEN AI 추론 정책과 운영 설정

범위: 로컬 소스 b5025030a75806c545740400051350e02c08cf0f의 읽기 조사, 기존 API 관측표, 사용자가 제공한 운영 화면. 새 모델 호출·제품 수정·운영 설정 변경·운영 요청 캡처는 하지 않았다. 향후 개선 참고용이며 구현 확정안이 아니다.

### 설정의 의미와 요청값

| 항목 | 의미 |
| --- | --- |
| 기본 추론 강도 없음 | null 저장. 개인 설정도 없으면 강도 파라미터 생략. 추론 중지 또는 none 전송이 아니다. |
| 최대 추론 강도 추가 제한 없음 | null 저장. 개인 선택에 별도 정책 상한을 두지 않는다. 최대값 자체는 제공자에게 보내지 않는다. |
| 개인별 추론 설정 | 지원되는 값이고 최대값 이내라면 기본값보다 우선한다. 기존 개인값이 상한 초과/미지원이면 기본값으로 복귀한다. |
| 기본값 없음 + 최대값 높음 | high를 자동 선택하지 않는다. 강도를 생략하면 제공자 내부 기본 강도를 상한으로 통제하지 못한다. |

| 화면 라벨 | 내부값 | GPT Native /responses | Claude Orchestration |
| --- | --- | --- | --- |
| 없음 | null | reasoning.effort 생략 | reasoning_effort 생략 |
| 최소 | minimal | reasoning.effort=minimal | reasoning_effort=minimal |
| 낮음 | low | reasoning.effort=low | reasoning_effort=low |
| 보통 | medium | reasoning.effort=medium | reasoning_effort=medium |
| 높음 | high | reasoning.effort=high | reasoning_effort=high |
| 매우 높음 | xhigh | reasoning.effort=xhigh | reasoning_effort=xhigh |

위 표는 문자열 매핑이며 전체 모델의 지원표가 아니다. UI는 모델별 reasoningEfforts 목록과 최대값으로 필터링한다. 순번 변환이 아니므로 선택지 개수 차이로 값이 밀리지 않는다. 모델 변경 시 미지원 기존 강도는 null로 초기화하고 저장 시 기본 강도의 지원 여부를 검증한다. API의 지원 여부는 위 기존 관측표와 별도 대조해야 한다. 특히 기존 GPT 조사에서 minimal은 거절됐다.

### 공개 추론 요약은 강도와 별도

GPT의 SAP Native Responses 경로에는 프록시가 reasoning.summary=auto를 기본 추가한다. effort가 없으면 reasoning={summary:auto}가 된다. auto는 공개 가능한 추론 요약의 수준을 제공자가 선택하도록 요청하며 반환 보장이 아니다. concise/detailed도 기존 조사에서 비교했다. summary를 생략하면 요약을 명시적으로 요청하지 않는 것이고, off/false라는 반대값을 사용하는 구조는 아니다. 현재 프록시는 누락된 summary를 auto로 채우므로 앱의 강도 없음만으로 요약 요청을 끌 수 없다.

Claude에는 동일한 reasoning.summary가 없지만 Orchestration 공통 reasoning_effort 및 고유 thinking.type/budget_tokens/display, output_config.effort가 있다. 현재 ZEN AI 경로는 공통 reasoning_effort를 사용하며 별도 thinking 설정을 추가하지 않는다. summary 생략은 내부 추론 중지를 뜻하지 않는다.

### 운영에서 선택지를 정의하는 곳

| 위치 | 역할 |
| --- | --- |
| 운영 /srv/zen-ai/.env.prod의 MODELS | 모델별 reasoningEfforts·표시명·컨텍스트 등 앱 카탈로그 메타데이터 |
| Docker Compose env_file | 위 파일을 앱/워커 등의 런타임 환경변수로 주입. ZEN_AI_ENV_FILE 지정 시 다른 파일 사용 가능 |
| LiteLLM YAML | 제공 모델과 SAP 호출 경로·배포 라우팅 |
| 에이전트 DB | 선택한 기본/최대 강도, 개인 설정 및 실행 snapshot |

서버는 process.env.MODELS의 메타데이터와 제공자 모델 목록을 결합해 화면 선택지를 만든다. xhigh의 한글 라벨이 코드에 존재해도 모델의 reasoningEfforts에 없으면 표시되지 않는다. 사용자 제공 운영 화면은 없음·최소·낮음·보통·높음의 5개였고 매우 높음은 없었다. 화면만으로 선택 모델이나 실제 운영 MODELS 값은 확정할 수 없다.

운영 선택지 변경은 실제 활성 env 파일의 MODELS를 수정하고 앱 등 영향받는 컨테이너를 재생성하여 반영하는 배포 작업이다. 로컬 .env 변경만으로 운영에 반영되지 않는다. 이번에는 실행하지 않았다.

정정/보존: 조사 당시 로컬 MODELS에 Sol5.6만 있고 Sol6.1 항목이 없는 불일치를 발견했으나 사용자가 이후 직접 수정했다고 알렸다. 현재 미해결 장애로 취급하지 않고 운영 누락으로 확대하지 않는다. 이후 로컬 값을 재검사하지 않았다.

### 후속 개선 시 확인할 사항

- 운영의 모델별 reasoningEfforts, 화면 선택지, 실제 프록시/SAP 요청값을 같은 배포 SHA에서 대조한다. 특히 운영에 보이는 최소가 GPT에도 노출되는지와 해당 모델의 API 수용 여부를 확인한다.
- 없음의 의미(강도 미지정)와 최대값의 의미(사용자 선택 상한)를 UI에서 구분할지 검토한다. 공개 요약 요청과 추론 강도를 별도 설정으로 제공할지는 새 의도로 결정한다.
- 기존/신규 일반·생성 하네스·Builtin, 개인 override, 모델 전환, 허용 모델 snapshot과 재개 run의 적용을 지정 테스트 세션에 이관해 확인한다. 모델 카탈로그 변경이 기존 저장 snapshot까지 자동 갱신한다고 가정하지 않는다.
- 최소/xhigh/max 등 API 지원값과 현재 앱이 표현하는 값의 차이를 모델별로 관리한다. 설정만 추가해서 지원된다고 간주하지 않는다.
- 구현은 사용자 의도 확정 후 intent/harness 경로, 테스트는 헌법의 지정 테스트 담당 분리를 따른다. 이 문서는 구현·테스트 재개 지시가 아니다.

근거 코드: src/components/workspace/agent-policy-fields.tsx(라벨/필터), src/lib/ui/agent-policy-form.ts(모델/최대값 변경), src/lib/agents/effective-config.ts(개인값/상한/지원값), src/lib/agents/policy.ts(저장 검증), src/lib/ai/models.ts(MODELS 카탈로그), src/lib/agents/worker-runtime.ts(미지정 생략), src/lib/builtin-agents/runtime/model-connection.ts, docker/litellm-sap/config.local.yaml, docker/litellm-sap/apply_patch.py(summary 자동 추가), docker/litellm-sap/test_chat_bridge.py(기존 매핑 검사), docker-compose.prod.yml, .github/workflows/deploy.yml, docs/runbooks/litellm-deploy.md. 기준 경로 C:/Users/염정운/project/zen-ai. 기존 테스트 코드를 읽었으며 이번에 실행한 결과는 아니다.

## 9. 2026-11-09 보강 — Luna의 최소(minimal) 실패 근거

**결론: Luna의 `minimal` 거절은 실제 SAP API 응답으로 확인했다. 이번 공개 공식 문서 조사에서는 해당 Luna 버전의 `minimal` 미지원을 직접 명시한 문구를 찾지 못했다.** 따라서 ‘공식 문서에 미지원이라고 적혀 있다’와 ‘해당 배포에서 직접 요청해 거절을 확인했다’를 구분한다.

### 공식 SAP 자료에서 확인한 범위

- [SAP 공식 SDK 모델 목록](https://github.com/SAP/ai-sdk-js/blob/94b36e52e20693998b6f2e8cca1b7229efc01585/scripts/sap-models.json#L923-L932)에 `gpt-5.6-luna`, 버전 `2026-07-09`가 명시돼 있다. 모델 식별·가용성 자료이며 추론 강도 지원표는 아니다.
- [SAP Help — Reasoning](https://help.sap.com/docs/sap-ai-core/generative-ai/reasoning)은 “Not all models support the full range of configuration options.”라고 설명한다. 오케스트레이션은 가능한 경우 미지원 값을 가까운 값으로 매핑한다. 본문은 공식 검색 추출문으로 확인했다.
- [SAP Cloud SDK for AI — Reasoning Content](https://sap.github.io/ai-sdk/docs/js/orchestration/chat-completion#reasoning-content)도 공통 `reasoning_effort` 값과 모델별 매핑을 설명한다. **Orchestration의 공통 선택지에 `minimal`이 있다는 사실은 Native `/responses`의 Luna가 `reasoning.effort=minimal`을 그대로 지원한다는 뜻이 아니다.** 이번 보강에서 Luna의 Orchestration 매핑을 새로 시험한 것은 아니다.
- [SAP Help — Responses API](https://help.sap.com/docs/sap-ai-core/generative-ai/responses-api)는 지원 모델을 [SAP Note 3437766](https://me.sap.com/notes/3437766)으로 안내한다. 인증이 필요한 Note 본문은 읽지 못했으므로 그 안의 Luna별 제한 유무는 미확인이다. 조사한 공개 자료에서 문구를 찾지 못했다는 뜻이며, 공식 자료 어디에도 없다는 단정은 아니다.

### 2026-11-06 실제 API 확인 결과

기존 시험 원기록을 2026-11-09에 재확인했다. 대상은 SAP AI Core의 기존 Luna Foundation Model 배포, 응답이 식별한 모델은 `gpt-5.6-luna-2026-07-09`다. LiteLLM의 Chat 변환 경로를 거치지 않는 Native `/responses`에 동일한 합성 산술 질문을 보냈다. 도구 없음, `stream=true`, `store=false`, 출력 한도 4096, `reasoning.summary=auto` 조건이다.

| 전달한 추론 강도 | 실제 응답 | 해석 |
| --- | --- | --- |
| `reasoning.effort=minimal` | HTTP 400 · `invalid_request_error` · `code=unsupported_value` · `param=reasoning.effort` | 해당 모델·버전에서 요청 파라미터를 거절 |
| `reasoning.effort=low` | HTTP 200 · `response.completed` · 최종 답 `35524` · refusal 없음 | 같은 질문이 지원 강도에서는 정상 완료 |

400 응답은 `'minimal' is not supported with the 'gpt-5.6-luna-2026-07-09' model`이라고 안내했다. 응답에 나열된 지원값은 `none`, `low`, `medium`, `high`, `xhigh`, `max`다. **이 비교에서 직접 호출한 값은 `minimal`과 `low`뿐**이며, 나머지 값은 오류 응답이 안내한 목록이다.

별도의 기존 앱→LiteLLM 경유 시험에서도 `minimal`은 도구 유무와 관계없이 HTTP 400 `UnsupportedParamsError`였고, `low`는 도구 없는 응답과 도구 호출 후 최종 응답이 성공했다. Native 직접 호출도 실패했으므로 프록시의 사전 검증만 우회하면 해결된다고 볼 수 없다. 이번 오류의 직접 원인은 **질문 내용에 대한 안전 거부가 아니라 추론 강도 파라미터 불일치**다.

근거 원기록: [Native 비교 결과](<C:/Users/염정운/project/zen-ai/.context/luna-minimum-test2-20261106/native-result-20261106T100804808Z.json>), [기존 시험 요약](<C:/Users/염정운/project/zen-ai/.context/luna-minimum-test2-20261106/findings.md>). 원기록은 해당 PC의 로컬 보존 자료이며 다른 PC에 자동 동기화되지는 않는다. 위 표에는 판단에 필요한 응답 정보를 남겼다.

이 결론의 범위는 **2026-11-06의 해당 배포·모델 버전·Native 경로**다. 다른 버전·경로·미래 지원까지 일반화하지 않으며, 공개 추론 요약의 모든 요청 반환을 보장하지도 않는다. 이번 보강에서는 새 모델 호출이나 제품·운영 설정 변경을 하지 않았다. 사용자의 지시에 따라 수정 없이 조사 결과만 기록한다.
