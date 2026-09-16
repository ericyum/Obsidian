# zen-ai에서 Codex-LB API 사용 매뉴얼

이 문서는 `zen-ai`를 배포할 때가 아니라, 로컬에서 Codex-LB Gateway를 통해 회의록 Agent의 실제 LLM 왕복을 확인할 때 사용하는 절차다. Codex-LB는 배포 대상 Provider가 아니며, 운영 환경에서는 회사가 지정한 SAP AI Core/OpenAI-compatible Provider 설정을 사용한다.

## 1. 전제

- Codex-LB가 로컬에서 실행 중이어야 한다.
- 기본 Gateway 주소는 `http://127.0.0.1:2455/v1`이다. 실제 주소가 다르면 Codex-LB 설정을 우선 확인한다.
- `/v1/models`로 노출 모델과 인증을 먼저 확인한다.
- API 키는 파일·소스·문서·Git 추적 파일에 복사하지 않는다. PowerShell 현재 프로세스의 환경 변수로만 주입한다.

```powershell
$key = (Get-Content '키 파일의 안전한 로컬 경로' | Select-Object -Last 1).Trim()
$env:OPENAI_BASE_URL = 'http://127.0.0.1:2455/v1'
$env:OPENAI_API_KEY = $key
$env:OPENAI_SUPPORTS_STRUCTURED_OUTPUTS = 'true'
Invoke-RestMethod "$env:OPENAI_BASE_URL/models" -Headers @{ Authorization = "Bearer $key" } |
  Select-Object -ExpandProperty data | Select-Object -ExpandProperty id
```

출력된 모델 ID만 회의록 Builtin의 테스트용 release/config에서 선택한다. 이 저장소의 회의록 manifest에는 `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`가 허용되어 있다. 허용 목록에 없는 모델을 임의로 운영 설정에 추가하지 않는다.

## 2. zen-ai 실행

저장소 루트에서 DB와 migration이 준비된 뒤 실행한다.

```powershell
npm run check
npm run dev
```

별도 PowerShell에서 `http://localhost:3002`를 열고 회의록 Agent를 호출한다. 확인할 흐름은 `channel/API → agent_runs → Builtin Worker → meeting-agent-v1 → channel-adapter → Engine → Meeting Provider → Tool/HITL`이다. Provider 단독 호출만 성공해도 실제 DB·권한·HITL 통합이 성공한 것은 아니다.

## 3. 구조화 출력 설정의 의미

회의록 Provider는 `decide`, `searchResponse`, `response` 세 템플릿에 AI SDK `Output.object()`를 사용한다. `OPENAI_SUPPORTS_STRUCTURED_OUTPUTS=true`는 Gateway가 JSON Schema 출력을 지원한다고 명시하는 선택 스위치다. Codex-LB가 `oneOf`를 거부할 수 있어 decision 전송 Schema는 단일 object로 보내고, 반환 직후 기존 Zod discriminated union으로 의미와 필수 필드를 다시 검증한다.

일반 Custom Agent와 SAP Agent의 공통 Provider는 이 스위치가 없을 때 기존 동작(`false`)을 유지한다. 따라서 이 문서의 환경 변수만으로 다른 Agent의 출력 계약이 바뀌지는 않는다.

## 4. 모델별 최소 검증

각 허용 모델에서 다음을 최소 한 번씩 확인하고 결과를 기록한다.

1. `decide`: 검색·질문·녹화·직접 답변 중 올바른 action과 검색 query를 생성하는지.
2. `searchResponse`: 선택된 회의록과 근거 범위를 벗어나지 않고 `followUp`을 Boolean으로 생성하는지.
3. `answerQuestion`: 합성 회의 문서의 사실(예: 날짜)을 정확히 답하고 원문 밖의 내용을 만들지 않는지.
4. `directResponse`: Tool 없이 처리할 요청에 한국어 객체 응답을 생성하는지.
5. 60초 제한, 빈 응답, 잘못된 JSON, 429/5xx를 별도로 기록한다. 단일 성공은 품질 인증이 아니다.

## 5. 종료 및 원상복구

테스트가 끝나면 실행 중인 dev 서버를 중지하고 현재 PowerShell의 임시 값을 제거한다.

```powershell
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
Remove-Item Env:OPENAI_BASE_URL -ErrorAction SilentlyContinue
Remove-Item Env:OPENAI_SUPPORTS_STRUCTURED_OUTPUTS -ErrorAction SilentlyContinue
```

Codex-LB 테스트를 위해 임시로 변경한 제품 코드는 기준 브랜치 상태로 복구한다. 현재 작업 트리의 다른 사용자 변경을 포함해 전체를 되돌리지 말고, 변경 의도가 Codex-LB 전용이었던 파일만 확인 후 복구한다.

```powershell
git diff -- src/lib/ai/provider.ts src/lib/meeting-agent/provider.ts
git restore --source=HEAD -- src/lib/ai/provider.ts src/lib/meeting-agent/provider.ts
git status --short
```

`git restore` 전에 해당 파일에 다른 작업자의 변경이 섞이지 않았는지 반드시 `git diff`로 확인한다. 테스트 결과 문서에는 모델·시나리오·종료 사유만 기록하고 API 키는 기록하지 않는다.

## 6. 운영 배포와의 구분

Codex-LB 주소와 키는 운영 배포에 넣지 않는다. 운영에서는 배포 환경이 제공하는 `OPENAI_BASE_URL`, 인증 방식, 모델 정책을 사용하고, 회의록 manifest의 허용 모델·release/config 정책을 배포 DB에 반영한다. Codex-LB에서 통과한 결과는 로컬 Gateway 호환성 증거일 뿐 운영 Provider의 가용성·비용·품질을 보증하지 않는다.
