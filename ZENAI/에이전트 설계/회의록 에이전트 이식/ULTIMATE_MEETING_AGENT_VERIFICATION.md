# 회의록 Agent 이식 최종 검수 실행 지침

이 문서는 다른 컴퓨터의 Codex가 `zen-ai` 최신 코드와 Obsidian 설계 문서를 읽은 뒤 검수·수정·테스트를 수행하기 위한 단일 지침이다.

## 1. 작업 범위

- 대상 저장소: 최신 push 상태의 `zen-ai`
- 참고 문서: `ZENAI/에이전트 설계/회의록 에이전트 이식/`의 1~10번 문서 전체
- 우선순위: 최신 코드의 실제 동작 > 최신 설계 문서 > 과거 대화/역사 기록
- 목표: 설계와 코드의 정합성을 검증하고, 필요한 코드·테스트·문서만 수정
- 금지: 사용자 승인 없는 commit, push, merge, PR 생성

## 2. 확정된 구조

`@MEETING`은 Builtin Definition에 등록되고 `executorId: meeting-agent-v1`로 Executor Registry의 회의록 Executor를 호출한다. 공통 Worker/Builtin Runtime은 claim·lease·cancel·terminal 처리를 담당하며, 회의록 Executor 내부의 Engine이 회의 업무 흐름을 조정한다.

회의록 실행은 다음 경계를 따른다.

```text
Builtin Worker → meeting-agent-v1 Executor → Engine
  → Tool 호출(Search/Question/Recording)
  → Provider/LLM 호출(필요한 경우)
  → 최종 결과 또는 HITL 대기
```

## 3. 세 저장소의 책임

### Agent State Store

- 일반적으로 다음 Run에서도 참고할 누적 Scratchpad
- HITL 재개에 필요한 최소 참조 정보(`tool`, `pendingToken`, `resumePoint` 등)
- 회의 Tool만 사용하는 상세 변수나 Snapshot은 저장하지 않음

### Tool State Repository

- Tool 전용 상태(현재는 `authorizedMeetingIds` 등)
- HITL 상세 Snapshot, 후보 목록, 검증에 필요한 토큰·메타데이터
- Tool이 직접 정합성과 권한을 검사

### Workflow Checkpoint Store

- 사용자의 한 요청/업무 Workflow Turn이 완료되거나 HITL 대기로 경계가 닫힌 사실
- 상태(`done`, `hitl_waiting`, `failed`, `cancelled`)와 경계 메타데이터
- Scratchpad나 HITL 상세 Snapshot의 대체 저장소가 아님

기존 `agent_runs.builtin_checkpoint`가 남아 있다면 즉시 삭제하지 말고 호환성 동기화 용도로만 유지한다. 새 저장소가 논리적 정본인지 코드와 테스트로 확인한다.

## 4. HITL 확정 흐름

1. Engine이 Search Tool을 호출한다.
2. Search Tool이 후보를 조회하고 후보 수를 자체 판단한다.
3. 후보가 하나면 즉시 선택하고, 여러 개면 Tool State Repository에 상세 Snapshot을 저장한다.
4. Tool은 Engine에 최소 HITL 참조 정보만 반환한다. 별도의 Engine `select()` 호출은 만들지 않는다.
5. Engine은 Scratchpad와 최소 HITL 참조를 Agent State Store에 반영하고 Workflow Checkpoint를 `hitl_waiting`으로 기록한 뒤 Run을 종료한다.
6. 사용자의 다음 입력은 새 Run이다. Channel Adapter가 pending HITL 여부를 확인한다.
7. Engine은 입력과 token을 Tool State Repository로 전달한다.
8. Repository가 입력·권한·Snapshot을 검증한다.
9. 유효하면 상세 HITL을 소비/삭제하고 선택 결과와 Tool 상태(예: meeting IDs)를 갱신한 뒤 논리적으로 중단된 지점 다음부터 재개한다.
10. 무효하면 pending 상세 상태를 유지하고 재입력 안내를 반환하며 Run을 다시 종료한다.

HITL 상세 검증과 Snapshot 처리는 Tool 경계의 책임이다. Engine은 순서 제어와 Run 경계 저장만 담당한다.

## 5. Tool·Message·Trace 원칙

- Tool 결과는 공통적으로 `toolCallId`, `toolName`, `status`, 필요 시 `output`, `hitl`, `errorCode`를 사용한다.
- `output`은 LLM에 실제로 필요한 경우에만 Context에 포함한다.
- Tool 중간 호출/결과는 `message.posted`에 최종 대화와 섞어 저장하지 않는다. 최종 사용자/Assistant 메시지만 terminal 경로에서 저장한다.
- Trace는 기존 Realtime Hub의 `agent-delta`를 사용한다. 별도 Trace Sink를 새로 만들지 않는다.
- 감사 기록은 기존 Builtin operation receipt/evidence 경로를 사용한다.
- API 키, 회의록 원문, 민감한 Snapshot은 Agent State·Checkpoint·Trace·로그에 저장하지 않는다.

## 6. 반드시 수행할 검수

### 정적·빌드 검수

```text
npm ci
npm run check
npm run build
npm run lint
npm run knip
git diff --check
```

프로젝트에 없는 명령은 무리하게 추가하지 말고 실제 `package.json` 스크립트를 확인한다.

### DB 검수

1. 테스트 전용 PostgreSQL을 별도로 실행한다.
2. `.env`의 URL·사용자·비밀번호·포트가 그 인스턴스와 일치하는지 확인한다.
3. 빈 DB에 최신 migration을 처음부터 적용한다.
4. Agent State, Tool State, Workflow Checkpoint 테이블·인덱스·RLS·동기화 트리거를 확인한다.
5. 개발 DB나 기존 데이터에 의존하지 않는지 확인한다.

### 기능·통합 검수

- 일반 Search 성공/실패
- 단일 후보 자동 선택
- 복수 후보 HITL 생성 및 상세 Snapshot 저장
- 유효한 HITL 입력의 소비·상태 갱신·재개
- 무효 HITL 입력의 pending 유지·재입력 안내
- 권한 없는 회의록 접근 차단 및 재검증
- 다음 일반 질문에서 Scratchpad가 정상적으로 참고되는지
- Recording Tool의 성공 이벤트와 즉시 종료
- 최종 `message.posted`, Realtime delta, operation evidence

전체 Vitest와 회의 Agent 관련 targeted test를 모두 실행한다. 인증 오류, `ECONNREFUSED`, 포트 불일치, 환경 변수 누락으로 실패하면 테스트 실패로 숨기지 말고 환경 문제로 명시한 뒤 환경을 고쳐 재실행한다. 통과 기준은 의도된 skip을 제외한 실패 0건이다.

## 7. 코드 수정 원칙

- 기존 공통 Worker/Builtin/Realtime/감사 구조를 우선 재사용한다.
- 회의록 전용 상세 상태는 Tool State Repository 밖으로 확장하지 않는다.
- Engine에 Tool 내부 판단을 중복 구현하지 않는다.
- `select()` 같은 별도 HITL Tool을 재도입하지 않는다.
- 수정 전 관련 테스트를 확인하고, 수정 후 회귀 테스트를 추가한다.
- 구현과 설계 문서가 다르면 실제 코드 근거를 기록하고 문서의 9번(검수 결과) 또는 10번(후속 작업)에 반영한다.

## 8. 최종 보고 형식

검수 종료 시 다음을 보고한다.

1. 실행한 명령과 각 결과
2. 통과한 기능 시나리오
3. 실패·차단 항목과 정확한 원인
4. 수정한 파일과 변경 이유
5. 설계 문서와 코드의 남은 차이
6. 추가 작업이 필요한 경우의 구체적 다음 단계

모든 검수가 끝나도 commit·push·merge·PR은 수행하지 않는다. 사용자가 별도로 요청할 때만 진행한다.
