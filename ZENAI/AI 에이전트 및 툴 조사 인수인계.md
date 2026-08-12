# AI 에이전트 및 툴 조사 인수인계

> 작성일: 2026-08-12  
> 목적: ZEN AI 회의록 프로젝트의 제품 배경과 AI 에이전트·툴 조사 내용을 다른 컴퓨터 또는 새 대화에서 그대로 이어가기 위한 통합 문서

---

## 0. 이 문서를 다음 AI에게 전달할 때 사용할 요청

아래처럼 요청하면 된다.

```text
첨부한 「AI 에이전트 및 툴 조사 인수인계.md」를 먼저 읽어라.
이 문서는 ZEN AI 회의록 프로젝트와 지금까지 진행한 에이전트·툴 조사 내용이다.
이미 정리된 내용을 처음부터 반복하지 말고, 문서의 '다음 조사 순서'부터 이어서 진행하라.
주요 주장은 최신 논문과 공식 문서를 근거로 검증하고 출처를 링크로 남겨라.
```

---

# 1. 사용자의 상황과 조사 목적

사용자는 2026-08-13까지 AI 에이전트에 관해 학습하고 발표해야 한다. 단순한 개념 소개가 아니라 실제로 진행 중인 ZEN AI 회의록 프로젝트의 에이전트 설계로 연결되는 실용적인 발표가 필요하다.

초기 조사 범위는 다음 세 부분이다.

1. 에이전트란 무엇인가
2. 에이전트 설계 시 권한과 보안을 어떻게 설계하는가
3. 실제 ZEN AI 회의록 프로젝트에 어떻게 적용할 것인가

사용자가 처음부터 중요하다고 본 최근 주제는 다음과 같다.

- Loop: 목표를 받은 에이전트의 반복 실행과 종료·승인 판단
- Compact: 긴 세션 압축, 토큰 절감, 장기 상태 관리
- Tool: 너무 많은 툴에서 발생하는 선택 성능 저하와 적절한 툴 검색
- Sub-agent: 상위 고성능 모델과 하위 저비용 모델의 역할 분담

이후 발표에서 특히 깊게 조사하려는 주제는 **Tool**이다.

핵심 질문:

- 툴이란 무엇인가
- 툴과 툴킷은 무엇이 다른가
- 에이전트는 언제 툴을 써야 한다고 판단하는가
- 많은 툴 중 어떤 툴을 선택하는가
- 툴을 어떻게 잘 만들고 붙이는가
- 특히 툴의 description을 어떻게 작성해야 하는가

---

# 2. 확인한 ZEN AI 자료

다음 파일들을 읽고 제품 배경을 파악했다.

- `ZENAI/개선 내역/UIUX/ZEN-AI-회의록-UIUX-개선안.md`
- `ZENAI/개선 내역/UIUX/회의록 UIUX 구상안 (v2 — 전사 기반).md`
- `ZENAI/개선 내역/UIUX/ZEN-AI-회의록-채널연결-챗봇안.html`
- `ZENAI/회의록/2026-08-11 회의록 및 에이전트 설계 회의.md`

## 2-1. ZEN AI 제품 이해

ZEN AI는 Discord/Slack과 유사한 채널 기반 협업 워크스페이스다. 회의록 기능은 실제 회의 오디오를 원본으로 다음 과정을 제공한다.

```text
회의 녹음 또는 파일 업로드
→ 비공개 오디오 저장
→ STT 및 화자 분리
→ LLM 기반 회의록 구조화
→ AI 초안 저장
→ 사람 검토·확정
→ 관련 채널에 공유
→ 이후 회의록 기반 질의응답
```

회의록은 단순 채팅 요약 기능이 아니다.

데이터의 신뢰 관계:

```text
오디오 → 전사 세그먼트 → 회의록 문장
 원본        1차 파생물       2차 파생물
```

핵심 제품 원칙:

- 오디오와 전사가 원본이고 회의록은 파생물이다.
- 결정 사항과 액션 아이템에는 근거가 되는 전사 세그먼트 ID를 연결한다.
- AI 결과는 항상 초안이며 사람이 확정한다.
- 확정본은 재처리 결과로 덮어쓰지 않고 새 버전 초안을 만든다.
- 긴 전사·구조화 작업은 백그라운드에서 실행한다.
- 단계마다 결과와 오류를 저장하고 중단 후 재개할 수 있어야 한다.
- 관련 채널 멤버십과 커뮤니티 테넌시를 권한의 기본으로 사용한다.

## 2-2. UI/UX 요약

- `/minutes`에 목록과 미리보기 제공
- 브라우저 녹음 또는 오디오 파일 업로드
- 상태: 녹음 중, 업로드 중, 전사 중, 초안 생성 중, 초안, 확정, 실패
- 상세 화면 왼쪽은 회의록 문서, 오른쪽은 전사 원문 패널
- 회의록 문단과 전사 세그먼트의 근거 연결
- STT의 Speaker 1/2/3을 워크스페이스 멤버와 사람이 매핑
- 액션 아이템은 담당자·마감일·체크 상태 포함
- 확정 후 관련 채널에 회의록 카드 게시

## 2-3. 서버 파이프라인 초안

```text
업로드/녹음
→ 1. ingest
→ 2. transcribe
→ 3. structure
→ 4. notify
→ 사람 확정
→ 5. publish
```

단계별 역할:

| 단계 | 역할 | 대표 산출물 |
|---|---|---|
| ingest | 오디오 저장·길이·포맷 검증 | 오디오 URL, 길이 |
| transcribe | STT·화자 분리·타임스탬프 | transcriptSegments |
| structure | LLM 회의록 초안 생성 | meetingMinutes draft |
| notify | 요청자에게 완료 알림 | inbox event |
| publish | 사람 확정 후 채널 게시 | channel event |

구조화 출력 초안:

```ts
{
  title: string,
  summary: string[],
  agenda: { title: string; notes: string }[],
  decisions: {
    text: string;
    evidenceSegmentIds: string[];
  }[],
  actionItems: {
    text: string;
    assigneeHint: string | null;
    dueHint: string | null;
    evidenceSegmentIds: string[];
  }[];
}
```

## 2-4. 발견한 제품 정책 충돌

문서 간에 다음 불일치가 있다.

- UI/UX 개선안: 사용자가 회의 시작 시 전사·요약 에이전트를 선택
- 2026-08-11 회의 결정: 최종 사용자는 에이전트를 직접 조합하거나 수정하지 않고 시스템이 최적 구성을 제공

권장 해결:

- 관리자가 모델·프롬프트·툴·권한을 포함한 에이전트 프리셋을 구성한다.
- 일반 사용자는 필요하다면 `일반 회의`, `고객 회의`, `기술 회의` 같은 검증된 프리셋만 선택한다.
- 모델명, 시스템 프롬프트, MCP 서버와 개별 툴 조합은 일반 사용자에게 직접 맡기지 않는다.

---

# 3. 에이전트에 대한 정의와 현재 합의

## 3-1. 기본 정의

> AI 에이전트는 주어진 목표를 달성하기 위해 현재 상태를 관찰하고, 다음 행동을 선택하며, 툴이나 외부 환경과 상호작용하고, 그 결과를 바탕으로 행동을 반복·수정하는 시스템이다.

발표용 간단한 정의:

> AI 에이전트는 목표를 달성하기 위해 상황을 관찰하고, 다음 행동과 툴을 선택하며, 결과를 바탕으로 계획을 수정하는 AI 시스템이다.

기술적 구성:

```text
Agent =
  Model
  + Goal/Instructions
  + State/Memory
  + Tools
  + Control loop
  + Permissions
  + Validators
  + Human approval
  + Observability/Evaluation
```

핵심 키워드:

```text
목표 → 판단 → 행동 → 관찰/피드백 → 재판단 또는 종료
```

## 3-2. LLM 호출, 워크플로, 에이전트의 차이

### 단일 LLM 호출

```text
전사 원문 → LLM → 회의록 JSON
```

모델이 입력을 받아 출력을 한 번 생성한다.

### 워크플로

```text
ingest → transcribe → structure → notify → publish
```

개발자가 실행 순서를 미리 정한다.

### 에이전트

```text
목표 확인
→ 현재 상태 관찰
→ 다음 행동 또는 툴 선택
→ 실행 결과 관찰
→ 계획 수정
→ 완료 여부 판단
```

차이는 AI 사용 여부가 아니다.

> 다음 실행 경로가 코드로 고정되어 있는가, 아니면 모델이 현재 상황을 보고 선택하는가가 핵심 차이다.

## 3-3. 회의 질의응답 기능은 에이전트인가

다음 요청을 예로 들었다.

```text
“지난번 회의에서 어떤 중점 사안들이 있었으며 어떻게 하겠다고 결정됐는지 알려줘.”
```

가능한 실행:

```text
사용자 요청과 신원 확인
→ “지난번 회의” 후보 검색
→ 검색 결과 관찰
→ 접근 가능한 회의인지 확인
→ 확정 회의록 조회
→ 필요하면 전사 근거 검색
→ 정보가 충분한지 판단
→ 근거와 함께 답변
```

다음 판단을 모델이 상황에 따라 수행한다면 에이전트라고 부를 수 있다.

- “지난번 회의”가 모호하면 사용자에게 다시 물을지 판단
- 관련 채널이나 참석자 조건으로 추가 검색할지 판단
- 확정 회의록만으로 충분한지 판단
- 전사 원문을 추가 검색할지 판단
- 근거가 부족하면 검색 조건을 바꿀지 판단
- 정보가 충분하면 툴 호출을 멈추고 답변할지 판단

호출 순서가 항상 고정되어 있다면 LLM 기반 워크플로에 더 가깝다.

## 3-4. 상태와 관찰

둘은 다르다.

- 관찰: 방금 툴이나 외부 환경에서 돌아온 결과
- 상태: 지금까지 알게 된 정보와 현재 진행 위치를 누적한 값

관찰 예:

```json
{
  "meetings": [
    {
      "meetingId": "meeting-123",
      "title": "ZEN AI 에이전트 설계 회의"
    }
  ]
}
```

상태 예:

```json
{
  "goal": "지난 회의의 주요 안건과 결정 사항 답변",
  "selectedMeetingId": "meeting-123",
  "access": "allowed",
  "minutesStatus": "confirmed",
  "evidenceCollected": true,
  "remainingToolCalls": 4
}
```

상태와 모델 컨텍스트도 구분해야 한다.

- 시스템 상태에는 인증 정보, 실행 횟수, 내부 정책 등도 포함될 수 있다.
- 모델 컨텍스트에는 현재 판단에 필요한 최소 정보만 넣는다.
- 인증 토큰, DB 연결 정보, 내부 권한 규칙 전체는 모델에 노출하지 않는다.

## 3-5. ZEN AI 회의록 생성의 성격

현재 제안된 생성 파이프라인 전체는 자유로운 에이전트보다 durable workflow에 가깝다.

| 단계 | 권장 구현 | 이유 |
|---|---|---|
| ingest | 일반 코드 | 검증 규칙이 결정론적 |
| transcribe | STT 어댑터 | 정해진 제공자 호출 |
| structure | 제한된 LLM 구조화 | 의미 판단이 필요 |
| notify | 일반 코드 | 이벤트 규칙이 고정 |
| publish | 코드 + 사람 승인 | 확정 후 명시적 게시 |

현재 핵심 설계 결론:

> 회의록 생성은 신뢰성이 중요한 워크플로로 구현하고, 생성된 회의 지식을 검색·활용하는 단계에서 제한된 에이전트 기능을 도입한다.

---

# 4. 권한 설계에서 이미 합의한 핵심

## 4-1. 권한 검사 전용 툴에만 의존하면 안 된다

위험한 구조:

```text
search_all_meetings
→ 모델이 회의 메타데이터를 확인
→ check_meeting_access
```

권한 검사 전에 이미 접근하면 안 되는 제목이나 데이터가 모델 컨텍스트에 들어갈 수 있다.

권장 구조:

```text
search_accessible_meetings
get_accessible_meeting_minutes
search_accessible_transcript_segments
```

모든 데이터 툴은 내부에서 현재 사용자, 커뮤니티, 회의 대상에 대한 권한을 강제해야 한다.

> 모델은 권한 검사를 요청할 수 있지만, 실제 인가는 모델의 올바른 툴 호출 여부와 무관하게 서버에서 강제해야 한다.

## 4-2. 기본 보안 원칙

- 모델에게 고유한 광범위 권한을 주지 않는다.
- 현재 사용자의 권한을 제한된 capability로 위임한다.
- RLS와 서버 인가를 데이터 조회 단계에서 적용한다.
- 읽기와 쓰기 툴을 분리한다.
- 구조화 모델에 게시·삭제·관리자 변경 툴을 주지 않는다.
- 외부 게시, Jira 생성, 삭제 등은 사람의 승인 또는 명시적 정책을 거친다.
- 모든 쓰기 툴은 실행 시 권한·현재 상태·입력 스키마를 다시 확인한다.
- 툴 결과에 포함된 외부 텍스트는 명령이 아니라 데이터로 취급한다.

---

# 5. 툴의 정의

## 5-1. 기본 정의

> 툴은 에이전트가 모델 내부의 지식과 텍스트 생성 능력만으로 수행할 수 없는 작업을 하기 위해 호출하는, 명확한 입출력 계약을 가진 외부 기능이다.

좀 더 정확한 정의:

> 툴은 이름·설명·입출력 스키마·실행 로직·권한 정책을 가진 인터페이스로, 언어 모델의 판단을 실제 시스템의 데이터와 행동에 연결한다.

구성:

```text
Tool =
  Name
  + Description
  + Input schema
  + Output contract
  + Execution code
  + Authorization policy
  + Side-effect semantics
  + Error semantics
  + Observability
```

## 5-2. 모델이 툴을 직접 실행하는가

일반적인 function calling 구조에서는 모델이 직접 DB나 API를 실행하지 않는다.

```text
LLM이 툴 이름과 인자를 구조화해 요청
→ 런타임이 요청 검증
→ 애플리케이션이 실제 함수/API 실행
→ 결과를 LLM 컨텍스트에 전달
→ LLM이 다음 행동 또는 답변 결정
```

## 5-3. 툴의 역할

### 읽기·관찰 툴

```text
search_accessible_meetings
get_confirmed_minutes
search_transcript_segments
```

### 계산·검증·변환 툴

```text
validate_minutes_schema
merge_transcript_segments
resolve_due_date
```

### 외부 상태 변경 툴

```text
save_minutes_draft
publish_minutes_to_channel
create_jira_issue
```

변경 툴에는 최소 권한, 승인, 멱등성, 감사 로그가 특히 중요하다.

## 5-4. API, RAG, MCP와의 차이

- API: 소프트웨어 시스템 간 기술적 인터페이스
- Tool: 모델이 목적과 사용 조건을 이해할 수 있게 포장한 의미적 인터페이스
- RAG: 관련 문서를 검색해 컨텍스트에 넣는 패턴
- MCP: 툴과 리소스를 표준화해 외부에 제공하는 프로토콜

RAG 검색기는 하나의 툴로 제공될 수 있다. 모든 툴이 RAG인 것은 아니다.

MCP 서버는 여러 툴을 제공할 수 있다. 내부 함수로 툴을 구현해도 되므로 툴과 MCP는 같은 개념이 아니다.

---

# 6. 툴과 툴킷

## 6-1. 기본 차이

> 툴은 에이전트가 호출하는 개별 기능이고, 툴킷은 같은 업무 목적과 권한 범위를 가진 관련 툴들을 묶어 관리하는 논리적 단위다.

예:

```text
MeetingEvidenceToolkit
├─ get_confirmed_minutes
├─ search_transcript_segments
└─ get_evidence_segments
```

`get_confirmed_minutes`는 툴이고, 세 툴의 묶음은 툴킷이다.

## 6-2. 용어 주의

`Toolkit`은 완전히 표준화된 용어가 아니다.

| 개념 | 사용되는 명칭 |
|---|---|
| 개별 호출 기능 | Tool, Function, Action |
| 관련 기능 묶음 | Toolkit, Toolset, Plugin |
| 외부 제공 서비스 | MCP server, Connector |
| 재사용 업무 능력 | Skill |

발표에서는 다음과 같이 범위를 선언하는 것이 좋다.

> 이 발표에서 툴킷은 하나의 업무 영역을 지원하기 위해 관련 툴과 공통 정책을 묶은 논리적 단위를 의미한다.

## 6-3. 툴킷은 단순 배열 이상일 수 있다

```text
Toolkit =
  관련 툴 목록
  + 적용 업무 영역
  + 공통 인증·인가
  + 공통 데이터 타입
  + 공통 오류 규칙
  + 위험도·승인 정책
  + 사용 가능한 에이전트 역할
  + 버전 정보
```

모델이 일반적으로 직접 호출하는 것은 툴킷이 아니라 개별 툴이다. 툴킷은 런타임이 관련 툴만 로드·제공하기 위한 관리·라우팅 단위다.

## 6-4. 툴이 적을 때

다음처럼 툴이 세 개이고 설명과 스키마가 명확히 구분된다면 동적 툴킷 라우팅은 필요하지 않다.

```text
search_accessible_meetings
get_confirmed_minutes
search_transcript_segments
```

세 툴 모두 LLM에 제공하고 모델이 직접 선택하게 할 수 있다.

코드 정리용 `MeetingToolkit`은 있어도 되지만, 선택 성능을 위한 별도 툴킷 선택 단계는 불필요하다.

## 6-5. 툴과 기능 영역이 많을 때

```text
전체 툴 카탈로그
├─ MeetingDiscoveryToolkit
├─ MeetingEvidenceToolkit
├─ ActionItemToolkit
└─ MeetingPublishToolkit
```

이 경우 2단계 선택을 사용할 수 있다.

```text
1단계: 사용자 요청과 관련된 기능/툴킷 선택
2단계: 선택된 툴킷 안에서 LLM이 개별 툴 선택
```

더 정확한 구현 표현:

> 툴 라우터 또는 검색기가 사용자 요청·권한·현재 상태를 기준으로 관련 툴킷이나 툴 후보를 선택하고, 런타임이 선택된 개별 툴 정의만 LLM에 제공한다.

## 6-6. 툴킷 도입 기준

`툴이 N개 이상이면 툴킷` 같은 고정 기준은 없다.

다음을 기준으로 판단한다.

- 전체 툴 정의가 컨텍스트를 과도하게 차지하는가
- 의미가 비슷한 툴 때문에 잘못된 선택이 발생하는가
- 기능별 권한과 위험도를 분리해야 하는가
- 요청마다 대부분의 툴이 불필요한가
- 실제 평가에서 선택 정확도가 떨어지는가

개수보다 **의미적 충돌, 권한, 비용, 평가 결과**가 중요하다.

---

# 7. 툴을 사용하겠다는 의사결정 과정

## 7-1. 하나의 문제가 아니라 네 단계다

```text
1. 툴이 필요한가?
2. 어떤 기능 영역의 툴이 필요한가?
3. 구체적으로 어떤 툴이 필요한가?
4. 어떤 인자로 호출해야 하는가?
```

연구 용어:

- Tool-use awareness: 툴을 사용해야 하는가
- Tool retrieval: 대규모 툴 카탈로그에서 후보를 검색
- Tool selection: 후보 중 실제 사용할 툴 선택
- Argument generation/filling: 호출 인자 생성
- Tool orchestration: 여러 단계에서 툴 순서·의존성 관리

## 7-2. 툴 사용이 필요한 대표 조건

- 최신 정보가 필요하다.
- 사용자·워크스페이스별 비공개 데이터가 필요하다.
- 외부 시스템의 실제 상태를 조회해야 한다.
- 정확한 계산이나 결정론적 검증이 필요하다.
- 외부 상태를 실제로 변경해야 한다.
- 모델의 기억으로 답하면 안 된다.
- 출처나 근거를 제시해야 한다.

ZEN AI에서는 `내 회의`, `지난 회의`, `회의 결정`, `액션 아이템`처럼 실제 서비스 데이터가 필요한 요청은 모델 기억으로 답하지 않고 반드시 권한이 적용된 툴을 사용하게 해야 한다.

## 7-3. 초기 방식: 모든 툴 정의를 LLM에 제공

```text
사용자 질문
+ 모든 툴의 name
+ description
+ input schema
→ LLM이 직접 선택
```

이 방식은 폐기된 구식 방식이 아니다. 툴이 적고 명확할 때는 지금도 가장 단순하고 유용하다.

장점:

- 구현이 단순하다.
- 별도 검색기가 필요 없다.
- 검색 단계에서 관련 툴을 놓치지 않는다.
- 디버깅과 평가가 쉽다.

툴이 많을 때의 문제:

- 입력 토큰 증가
- 유사한 툴 간 혼동
- 관련 없는 툴 설명 반복
- 권한·위험도별 노출 통제 어려움

---

# 8. 임베딩 기반 Tool Retrieval

## 8-1. 사용자가 들은 방식

사용자가 들은 방식은 대체로 맞으며 보통 다음 이름으로 부른다.

- Tool retrieval
- Dynamic tool selection
- Tool search
- Contextual function selection

하지만 다음 설명은 잘못됐다.

> 벡터가 자연어보다 AI가 인지하기 쉽다.

정확한 설명:

> 임베딩은 LLM이 이해하는 표현을 대신하는 것이 아니라, 별도의 검색기가 많은 툴 중 의미적으로 관련된 후보를 빠르고 저렴하게 좁히는 수학적 표현이다. 최종 LLM은 여전히 툴의 자연어 설명과 JSON 스키마를 본다.

## 8-2. 오프라인 인덱싱

각 툴에 검색용 문서를 만든다.

```json
{
  "toolId": "search_accessible_meetings",
  "text": "사용자가 열람 가능한 회의를 제목, 날짜, 참석자, 채널 조건으로 검색한다. 지난 회의나 특정 주제의 회의를 찾을 때 사용한다.",
  "domain": "meeting",
  "operation": "search",
  "risk": "read",
  "requiredScope": "meeting:read"
}
```

검색용 문서에 포함할 수 있는 정보:

- 툴 이름
- description
- 사용해야 하는 경우
- 사용하지 말아야 하는 경우
- 입력과 출력 의미
- 업무 영역
- 읽기·쓰기 구분
- 대표 사용자 표현

툴 문서를 embedding 모델로 변환하여 vector index에 저장한다.

## 8-3. 실행 시

```text
사용자 질문
→ 동일 embedding 모델로 벡터화
→ 툴 문서 벡터와 유사도 비교
→ Top-K 툴 ID 선택
→ 원래 툴의 자연어 설명·스키마 로드
→ LLM에 후보 툴 제공
→ LLM이 최종 툴과 인자 선택
```

LLM에 벡터 숫자를 직접 넣는 것이 아니다.

## 8-4. 임베딩만으로 부족한 이유

의미적으로 비슷하지만 업무적으로 다른 툴이 존재한다.

```text
get_draft_minutes
get_confirmed_minutes
regenerate_minutes
```

또는:

```text
preview_channel_card
publish_minutes_to_channel
```

단순 의미 유사도는 다음을 충분히 구분하지 못할 수 있다.

- 읽기와 쓰기
- 초안과 확정본
- 미리보기와 실제 게시
- 조회와 생성
- 단일 툴과 복수 툴 조합
- 현재 확보된 입력으로 호출 가능한지 여부

따라서 vector similarity는 최종 권한·행동 판단이 아니라 **후보 생성 단계**로 사용해야 한다.

---

# 9. 임베딩 단독보다 발전된 최신 방법

## 9-1. 정책·권한·상태 필터

검색보다 먼저 또는 검색과 함께 다음을 코드로 제한한다.

```text
사용자 권한
+ 에이전트 역할
+ 현재 작업 단계
+ 리소스 상태
+ 읽기/쓰기 위험도
+ 승인 여부
```

권한 없는 툴은 검색 점수가 높아도 LLM에 제공하지 않는다. 실제 툴 실행 시에도 서버에서 다시 검사한다.

## 9-2. 툴킷 또는 계층 라우팅

```text
사용자 요청
→ 관련 도메인/툴킷 선택
→ 해당 영역의 툴만 검색
→ 소수 후보를 LLM에 제공
```

예:

```text
회의 검색
회의 근거 조회
액션 아이템
채널 게시
관리
```

## 9-3. Hybrid retrieval

```text
Dense embedding
+ BM25/keyword matching
+ metadata filtering
```

임베딩은 표현이 다른 유사 의미를 찾는 데 유리하고, BM25는 `확정`, `초안`, `게시` 같은 정확한 구분어를 보존하는 데 유리하다.

## 9-4. Query rewriting / hypothetical tool description

사용자 표현과 기술적인 툴 문서 사이의 차이를 줄인다.

```text
“지난번에 얘기한 방향이 뭐였지?”
→ “접근 가능한 최근 회의를 검색하고 확정 회의록의 결정 사항을 조회한다.”
```

ToolDreamer는 LLM이 현재 질문에 필요할 것 같은 가상의 툴 설명을 먼저 만들고 이를 실제 툴 문서와 검색하는 방식을 제안했다.

## 9-5. Retrieve-then-rerank

```text
전체 툴 1,000개
→ 빠른 hybrid retrieval로 30개 후보
→ cross-encoder 또는 작은 LLM으로 재순위화
→ 최종 3~8개를 메인 LLM에 제공
```

재순위화 시 평가할 것:

- 사용자 목표와 정확히 맞는가
- 현재 필요한 입력을 확보할 수 있는가
- 출력이 다음 단계에 필요한 형태인가
- 읽기·쓰기 의도가 맞는가
- 현재 리소스 상태에서 실행 가능한가

## 9-6. 중복 툴 정리와 병합

```text
find_meeting
search_meeting
query_meetings
lookup_meeting
```

기능이 겹치는 툴은 설명을 수정하는 것만으로 충분하지 않다. 카탈로그 단계에서 제거·병합해야 한다.

ToolScope는 중복 툴 병합과 현재 문맥에 따른 필터링을 결합한다.

## 9-7. 복합 요청 분해와 multi-step retrieval

```text
“지난 회의 결정 사항을 찾아서 담당자별 액션 아이템으로 정리하고 채널에 공유해줘.”
```

이를 하나의 질문 벡터로만 검색하면 일부 기능을 놓칠 수 있다.

```text
하위 목표 1: 관련 회의 찾기
하위 목표 2: 결정·액션 아이템 조회
하위 목표 3: 채널에 공유
```

ToolQP는 복합 요청을 하위 작업으로 나누고 각 하위 작업에 대해 반복적으로 툴 검색 쿼리를 생성한다.

## 9-8. 현재 상태와 툴 의존성을 반영한 동적 검색

최초 질문만으로 툴을 한 번 선택하는 대신 실행 결과에 따라 다시 선택한다.

```text
get_confirmed_minutes 결과: 확정본 없음
→ 현재 상태 갱신
→ 초안 조회 또는 사용자 질문과 관련된 툴을 다시 검색
```

Dynamic Tool Dependency Retrieval은 초기 질문뿐 아니라 현재 실행 계획과 툴 간 의존성을 반영한다.

## 9-9. 과거 성공 실행 궤적과 그래프

```text
search_accessible_meetings
→ get_confirmed_minutes
→ 근거 부족 시 search_transcript_segments
```

운영 이력이 쌓이면 성공한 툴 전이와 선후관계를 학습할 수 있다.

- AutoTool: 성공 궤적의 툴 전이 그래프로 반복적인 선택 비용 절감
- SkillGraph: 의미 유사도 외에 툴 간 데이터·순서 의존성을 학습

신규 ZEN AI에는 아직 성공 궤적이 없으므로 초기 도입 대상은 아니다.

## 9-10. 학습된 Tool-use Policy

SFT, preference optimization, reinforcement learning 등을 이용해 다음을 함께 학습할 수 있다.

- 툴 필요성
- 툴 검색
- 툴 선택
- 인자 생성
- 실행 성공

ToolOmni 등은 정적 embedding retrieval보다 실행 결과까지 반영하는 open-world tool learning을 제안한다. 그러나 데이터와 학습 비용이 크므로 ZEN AI 초기 제품에는 과하다.

## 9-11. 툴 필요성 판단을 위한 내부 표현 연구

- MeCo: 모델 내부의 메타인지 신호를 이용해 툴 호출 필요성 판단
- When2Tool: hidden state에는 툴 필요성 신호가 있지만 자연어 판단과 실제 행동이 이를 잘 따르지 못할 수 있음을 보고

이 방식은 모델 hidden state에 접근할 수 있는 환경에 적합하다. 일반적인 상용 API 모델을 사용하는 제품에서는 직접 적용하기 어렵다.

ZEN AI에서는 규칙과 모델 분류를 결합하는 것이 현실적이다.

---

# 10. ZEN AI에 대한 단계별 권장안

## 10-1. 초기 버전

초기 툴이 다음 세 개라면 embedding 검색이나 별도 툴 라우터를 만들지 않는다.

```text
search_accessible_meetings
get_confirmed_minutes
search_transcript_segments
```

세 툴 모두 LLM에 직접 제공한다.

필수 정책:

```text
사용자가 실제 회의 내용, 결정 사항, 참석자 또는 액션 아이템을 질문하면
모델의 기억으로 답하지 않는다.
반드시 접근 권한이 내장된 회의 툴을 사용한다.
```

## 10-2. 기능이 늘어난 단계

툴을 다음 도메인으로 논리적으로 분류한다.

```text
MeetingDiscoveryToolkit
MeetingEvidenceToolkit
ActionItemToolkit
MeetingPublishToolkit
MeetingAdminToolkit
```

읽기와 쓰기 툴킷을 분리하고, 게시·관리 툴은 기본 질의응답 에이전트에 제공하지 않는다.

## 10-3. 실제 선택 문제가 관찰될 때

```text
사용자 요청
→ 툴 필요성 판단
→ 권한·정책 필터
→ 툴킷 라우팅
→ Hybrid retrieval
→ Reranking
→ 3~8개 후보 툴의 자연어 설명·스키마 제공
→ 메인 LLM의 최종 선택
→ 실행 전 서버 검증
→ 툴 실행
→ 결과 관찰
→ 충분하면 답변, 부족하면 현재 상태로 재검색
```

## 10-4. 도입 판단은 평가 결과로 한다

다음을 측정해야 한다.

- Tool necessity accuracy: 툴이 필요할 때/불필요할 때 판단 정확도
- Tool retrieval recall@K: 필요한 툴이 후보 K개 안에 들어왔는가
- Tool selection accuracy: 후보 중 올바른 툴을 골랐는가
- Argument accuracy: 입력 인자가 정확한가
- Execution success rate
- 불필요한 툴 호출 수
- 권한 없는 툴 노출·호출 건수
- 쓰기 툴의 잘못된 실행 건수
- 최종 답변의 근거 정확성
- 지연 시간·토큰·비용

---

# 11. 지금까지 바로잡은 주요 오해

## 오해 1

```text
AI를 사용한 여러 단계 처리 = 에이전트
```

수정:

```text
실행 순서가 고정되면 AI 기반 workflow일 수 있다.
현재 상태를 보고 모델이 다음 행동을 선택해야 agentic하다고 볼 수 있다.
```

## 오해 2

```text
권한 검사 툴을 모델이 호출하면 보안이 확보된다.
```

수정:

```text
모든 데이터·행동 툴이 서버에서 자체적으로 인가를 강제해야 한다.
권한 검사는 모델의 선택에 의존할 수 없다.
```

## 오해 3

```text
툴킷은 모델이 호출하는 큰 툴이다.
```

수정:

```text
대부분 툴킷은 관련 툴과 공통 정책을 묶는 관리·라우팅 단위다.
모델은 보통 그 안의 개별 툴을 호출한다.
```

## 오해 4

```text
툴이 3개 이상이면 툴킷이 필요하다.
```

수정:

```text
고정 개수 기준은 없다.
의미 충돌, 컨텍스트 비용, 권한·위험도, 실제 평가 결과가 기준이다.
```

## 오해 5

```text
벡터가 자연어보다 LLM이 이해하기 쉽다.
```

수정:

```text
벡터는 별도 검색기가 후보를 좁히는 표현이다.
최종 LLM은 자연어 설명과 JSON 스키마를 본다.
```

## 오해 6

```text
임베딩 검색이 가장 최신이며 최종 툴 선택도 맡길 수 있다.
```

수정:

```text
임베딩은 여전히 중요한 후보 검색 방법이지만 단독으로는 부족하다.
최신 방향은 정책 필터, 계층 라우팅, hybrid retrieval, reranking,
요청 분해, 동적 상태, 실행 궤적을 함께 사용하는 것이다.
```

---

# 12. 앞서 개괄적으로 조사했으나 아직 깊게 진행하지 않은 주제

## 12-1. Loop

좋은 loop에는 다음이 필요하다는 수준까지 합의했다.

- 명시적 성공 조건
- 최대 단계·툴 호출 수
- 시간·토큰·비용 예산
- 동일 행동 반복 감지
- 실패 유형별 재시도 정책
- 사람 승인이 필요한 경계
- 체크포인트와 중단 후 재개
- 취소 전파

ZEN AI 생성 파이프라인에는 무한 자율 loop 대신 단계별 최대 3회 재시도가 적절하다.

## 12-2. Compact와 Memory

세션 압축은 지능을 자동 복원하는 것이 아니다.

장점:

- 컨텍스트 비용 절감
- 오래된 잡음 제거
- 컨텍스트 한도 방지

위험:

- 결정 근거 누락
- 예외 조건·실패 원인 소실
- 화자·담당자 혼동
- 요약 오류의 누적

ZEN AI에서는 다음을 분리해야 한다.

- 원본층: 오디오·전사 세그먼트
- 확정 사실층: 확정 회의록·사람 화자 매핑
- 실행 상태층: 단계·재시도·오류·사용량
- 작업 컨텍스트: 현재 단계에 필요한 정보
- 압축 요약: 탐색과 토큰 절감용

원본과 확정 상태는 compaction 대상으로 덮어쓰지 않는다.

## 12-3. Sub-agent

상위에는 큰 모델, 하위에는 작은 모델을 둔다는 것은 가능한 비용 최적화 방식이지만 보편 법칙은 아니다.

작은 모델에 적합한 작업:

- 분류
- 스키마 변환
- 후보 검색
- 중복 제거
- 단순 검증

큰 모델에 적합한 작업:

- 모호한 목표 해석
- 여러 제약을 포함한 계획
- 충돌 해결
- 최종 통합 판단

에이전트 수 증가가 성능 증가를 보장하지 않는다. ZEN AI v1에는 multi-agent가 필요하지 않다는 방향이다.

## 12-4. 추가로 중요하다고 확인한 주제

- Evals
- Durable execution
- Provenance/evidence
- Observability
- Prompt injection과 데이터·명령 분리
- 최소 권한과 사람 승인
- 비용·지연·신뢰성을 포함한 운영 평가

---

# 13. 주요 연구·공식 자료

## 에이전트와 반복

- ReAct: https://arxiv.org/abs/2210.03629
- Reflexion: https://arxiv.org/abs/2303.11366

## 툴 사용의 기초

- Toolformer: https://arxiv.org/abs/2302.04761
- Gorilla: https://arxiv.org/abs/2305.15334
- MetaTool: https://openreview.net/pdf?id=R0c2qtalgG

## 메모리와 압축

- MemGPT: https://arxiv.org/abs/2310.08560
- LLMLingua: https://arxiv.org/abs/2310.05736
- LLMLingua-2: https://arxiv.org/abs/2403.12968

## 툴 검색과 선택

- ToolRerank: https://aclanthology.org/2024.lrec-main.1413/
- Retrieval Models Aren’t Tool-Savvy / ToolRet: https://aclanthology.org/2025.findings-acl.1258/
- Tool Preferences in Agentic LLMs are Unreliable: https://aclanthology.org/2025.emnlp-main.1060/
- ToolDreamer: https://aclanthology.org/2026.eacl-long.254/
- Dynamic Tool Dependency Retrieval: https://aclanthology.org/2026.findings-acl.1680/
- ToolOmni: https://aclanthology.org/2026.acl-long.1736/
- ToolQP: https://aclanthology.org/2026.findings-acl.2090/
- ToolScope: https://aclanthology.org/2026.acl-long.1573/
- AutoTool: https://ojs.aaai.org/index.php/AAAI/article/view/40389
- SkillGraph: https://arxiv.org/abs/2604.19793

## 툴 필요성 판단

- Adaptive Tool Use with Meta-Cognition Trigger / MeCo: https://aclanthology.org/2025.acl-long.655/
- When2Tool: https://arxiv.org/abs/2605.09252
- SMART: https://aclanthology.org/2025.findings-acl.239/

## 보안

- AgentDojo: https://arxiv.org/abs/2406.13352
- Instruction Hierarchy: https://arxiv.org/abs/2404.13208

## 에이전트 평가

- Evaluation and Benchmarking of LLM Agents: https://arxiv.org/abs/2507.21504
- Survey on Evaluation of LLM-based Agents: https://arxiv.org/abs/2503.16416
- ReflecTool-Bench: https://aclanthology.org/2026.findings-acl.86/

## 공식 구현 참고

- OpenAI 모델·툴 설계 가이드: https://developers.openai.com/api/docs/guides/latest-model
- Semantic Kernel Plugins: https://learn.microsoft.com/en-us/semantic-kernel/concepts/plugins/
- Semantic Kernel Contextual Function Selection: https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-contextual-function-selection

주의:

- 2026년 최신 자료 중 일부는 새 논문 또는 preprint이므로 방향성 자료로 보고 ZEN AI 자체 평가로 검증해야 한다.
- 논문에서 보고한 개선 수치는 해당 데이터셋·모델·설정에 한정되므로 제품 성능으로 그대로 일반화하지 않는다.

---

# 14. 발표의 현재 중심 메시지

현재까지의 내용을 가장 잘 요약하는 문장:

> 좋은 에이전트는 가장 오래 자율적으로 움직이는 시스템이 아니라, 필요한 순간에만 올바른 툴을 선택하고, 허용된 범위 안에서 행동하며, 결과를 검증하고 멈출 줄 아는 시스템이다.

ZEN AI에 대한 핵심 설계 문장:

> ZEN AI 회의록 생성은 결정론적이고 재개 가능한 워크플로를 뼈대로 삼고, 회의 검색·근거 탐색·후속 업무처럼 상황 판단이 필요한 구간에만 제한된 에이전트성을 도입한다.

툴에 대한 핵심 문장:

> Tool retrieval은 최종 선택이 아니라 후보 생성 과정이다. 권한과 위험도는 코드가 먼저 제한하고, 검색기는 관련 후보를 좁히며, LLM은 소수 후보의 자연어 설명과 스키마를 보고 최종 행동을 선택한다.

---

# 15. 다음 조사 순서

현재 대화는 툴 정의 → 툴·툴킷 → 툴 사용 여부와 retrieval까지 진행했다. 다음에는 한 번에 넓게 가지 말고 아래 순서대로 하나씩 조사하는 것이 좋다.

## 바로 다음 주제

### 좋은 툴을 어떻게 설계하는가

세부 질문:

1. 툴의 적절한 granularity는 무엇인가
2. 하나의 범용 툴과 여러 개의 좁은 툴 중 무엇이 좋은가
3. 읽기와 쓰기를 어떻게 분리하는가
4. 입력·출력 JSON Schema는 어떻게 설계하는가
5. 오류 형식과 재시도 가능성을 어떻게 표현하는가
6. 멱등성과 side effect는 어떻게 보장하는가
7. 모델이 호출하기 좋은 API와 사람이 쓰기 좋은 API는 어떻게 다른가

## 그다음 주제

### 툴 description 작성법

세부 질문:

1. name과 description이 각각 선택에 미치는 영향
2. 언제 사용하고 언제 사용하지 않는지를 어떻게 쓰는가
3. 입력 필드 설명을 어느 정도 자세히 쓰는가
4. 출력 형식과 오류 동작을 description에 포함할 것인가
5. 예시를 포함하면 언제 도움이 되고 언제 방해가 되는가
6. 유사 툴 간 description의 배타성을 어떻게 만드는가
7. description을 자동 최적화·평가하는 최근 연구

## 이후 주제

- Tool eval 설계
- Tool security와 prompt injection
- Tool execution loop, retry, recovery
- Context compaction과 장기 memory
- Sub-agent와 model routing
- ZEN AI 회의 에이전트의 실제 툴 목록·스키마 초안 작성
- 발표 자료의 목차와 슬라이드 구성

---

# 16. 다음 대화에서 유지해야 할 진행 방식

사용자는 한 번에 전체 내용을 받기보다 주제 하나를 차근차근 깊게 이해하는 방식을 원한다.

따라서 다음 원칙을 유지한다.

- 한 응답에서 하나의 핵심 주제만 다룬다.
- 정의 → 작동 방식 → 예시 → ZEN AI 적용 → 오해 교정 순서로 설명한다.
- 최신 연구는 기존 방식과 무엇이 달라졌는지 중심으로 설명한다.
- 연구 방법을 바로 제품에 적용하라고 단정하지 않고 도입 비용과 조건을 함께 설명한다.
- ZEN AI의 초기 툴 수가 적다는 사실을 고려해 과도한 인프라를 추천하지 않는다.
- 보안과 권한은 모델 판단이 아니라 서버 정책에서 강제한다.
- 사용자의 질문이나 의문을 먼저 해결한 뒤 다음 주제로 넘어간다.

