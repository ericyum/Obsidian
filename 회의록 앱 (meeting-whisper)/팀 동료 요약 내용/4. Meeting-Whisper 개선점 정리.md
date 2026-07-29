## 1. 로그아웃 관련 문제

### 문제 상황

- 화면 우상단의 계정 - 로그아웃 클릭 시 '세션 확인이 필요합니다' modal 창이 뜨는데
    - '세션 연장' 버튼을 클릭하면 GET currentUser 401이 뜨면서 '세션 확인에 실패했습니다~'로 메시지가 바뀜.
    - '다시 로그인' 버튼을 클릭하면 GET currentUser, GET Meetings... 가 401이 뜨면서 새로고침만 되고, 원래 modal 창 그대로인 상태임.
- 즉, UI상으로는 modal 창에서 벗어날 방법이 없음.
- 이 상태에서 새로고침(F5)을 하면 로그인 페이지로 보내짐.
- 시간 초과로 인한 세션 만료 시 '다시 로그인' 버튼으로 바로 로그인을 할 수 있음.

### 원인 분석

#### 왜 로그아웃 때만 갇히는가 — 핵심 차이

**타임아웃 만료와 로그아웃은 세션 쿠키 상태가 정반대입니다.**

||시간 초과 만료|로그아웃|
|---|---|---|
|모달을 띄운 주체|**클라이언트 타이머** (`session.js:98` `checkSessionAge`)|서버 401 응답|
|approuter 세션 쿠키|**아직 살아 있음**|**삭제됨**|
|재진입 시 API 호출|성공 → `touchSession()` → `hideExpired()` → 모달 닫힘 ✓|401 → 모달 재등장 ✗|

타임아웃 모달은 `lastTouchAt`(localStorage) 기준으로 **브라우저가 혼자 판단**해서 띄웁니다. 이때 서버 세션은 멀쩡하므로, `다시 로그인`으로 페이지를 다시 띄우기만 하면 API가 성공하고 `session.js:116`의 `hideExpired()`가 모달을 닫아줍니다. **버튼이 제대로 동작해서가 아니라, 세션이 살아 있어서 우연히 해결되는 것**입니다.

로그아웃은 쿠키가 실제로 삭제되므로 이 우연이 성립하지 않고, 버튼의 결함이 그대로 드러납니다.

#### 원인 ① — `다시 로그인` 버튼이 실제로 서버에 묻지 않음

`session.js:209-211`:

```
expiredModal.querySelector("[data-session-login]")?.addEventListener("click", () => { 
 window.location.assign(window.location.href);   // ← 문제 지점
});
```

`location.assign(현재 URL)`은 **일반 내비게이션**이라 브라우저의 기본 HTTP 캐시 규칙을 따릅니다. 반면 F5는 **reload**라서 메인 문서를 강제로 재검증(revalidate)합니다. 이 차이가 사용자가 관찰한 현상을 그대로 설명합니다:

```
[다시 로그인] location.assign(same URL) 
   → 캐시된 index.html 재사용 (approuter에 도달 안 함)  
   → SPA 부팅 → GET currentUser 401, GET Meetings 401  
   → 모달 재등장 ... 무한 루프
   
[F5] reload  
   → 메인 문서 재검증 → approuter 도달  
   → 세션 없음 → 302 로그인 페이지 ✓
```

캐시가 개입할 수 있는 근거는 `xs-app.json`에 있습니다. `/user-api` 라우트에만 `cacheControl: "no-cache, no-store, must-revalidate"`가 붙어 있고(`:13`), **정적 파일을 서빙하는 catch-all 라우트(`:33-38`)에는 캐시 지시자가 없습니다.** 따라서 `index.html`이 휴리스틱 캐싱 대상이 됩니다.

- 실제 확인 : `다시 로그인` 클릭 시 DevTools Network 탭에서 `index.html` 요청이 **(from disk cache)**로 표시

#### 원인 ② — `세션 연장` 실패 시 메시지가 잘못 덮어써짐

이건 코드에서 100% 확정되는 버그입니다. `session.js:63-65`:

```
.catch((error) => {
  showExpired({ source: reason, message: error?.message || "" });  // ← status 없음
  throw error;
})
```

그런데 메시지 분기는 `status`를 봅니다 (`session.js:152`):

```
message.textContent = detail?.status === 401
  ? "로그인 세션이 만료되었습니다. ..."        // 정확한 안내
  : "세션 확인에 실패했습니다. ...";          // 항상 이쪽으로 빠짐
```

`catch`가 `status`를 전달하지 않으므로 **진짜 401이어도 항상 "세션 확인에 실패했습니다"가 됩니다.**

더 정확히는 이렇게 두 번 덮어써집니다:

1. `getCurrentUser()`가 401을 받고 `api.js:167`에서 `markSessionExpired("current-user", 401)` → **status 401이 실린 정확한 메시지**가 먼저 표시됨
2. 곧이어 같은 에러가 `refreshSession`의 catch로 올라가 `showExpired`를 **status 없이 다시 호출** → 일반 메시지로 덮어씀

즉 api.js는 이미 올바른 이벤트를 쐈는데, session.js의 중복 호출이 정보를 훼손하는 구조입니다.

#### 원인 ③ — 모달에 탈출구가 없음

`ensureExpiredModal()`이 만드는 모달에는 `세션 연장`과 `다시 로그인` 두 버튼뿐입니다. 닫기 버튼도, ESC 핸들러도, 배경 클릭 닫기도 없습니다. 두 버튼이 모두 실패하면 **설계상 빠져나올 방법이 없습니다.**

여기에 더해, 로그아웃은 사용자가 **의도한 행동**인데 앱은 이를 "세션 오류"로 취급합니다. `app.js:143-145`가 `/logout`으로 이동하는 동안 진행 중이던 API 요청이 401을 받으면 그대로 `markSessionExpired`가 발동해 모달이 뜹니다. 로그아웃 중임을 알리는 플래그가 없습니다.

### 해결 방법

우선순위 순으로 정리합니다. ①②만 고쳐도 갇히는 현상은 사라집니다.

#### 1순위 — `다시 로그인`이 반드시 서버에 도달하게 (session.js:210)

가장 작은 수정은 `location.reload()`로 바꾸는 것입니다. F5와 동일한 재검증 동작이 됩니다. 다만 프록시·중간 캐시까지 확실히 우회하려면 **캐시 버스팅 쿼리를 붙여 approuter 진입점으로 명시 이동**하는 편이 견고합니다:

```
window.location.assign(`/?_=${Date.now()}`)
```

이러면 캐시 미스가 확정되어 approuter가 반드시 응답하고, 세션이 없으면 로그인으로 리다이렉트됩니다. 부수 효과로 `?meeting=<id>` 같은 기존 쿼리가 초기화되는데, 재로그인 시나리오에서는 오히려 안전한 동작입니다.

#### 2순위 — index.html 캐시 금지 (xs-app.json)

근본 차단책입니다. 정적 라우트에 `cacheControl`을 지정해 HTML이 캐시되지 않게 하면, 어떤 방식으로 재진입하든 approuter를 거치게 됩니다.

이 프로젝트는 이미 JS 모듈을 `?v=20260708-session` 같은 **버전 쿼리로 캐시 무효화**하고 있으므로(`app.js:19`, `:22`), **index.html만 `no-store`로 두고 나머지 정적 자산은 캐시를 유지**하는 조합이 성능 손해 없이 잘 맞습니다.

#### 3순위 — 메시지 버그 수정 (session.js:64)

`refreshSession`의 catch에서 `showExpired`를 다시 호출하지 않는 것이 가장 깔끔합니다. api.js가 이미 정확한 `status`로 `EXPIRED_EVENT`를 발생시키므로 중복 호출이 불필요합니다. 굳이 유지한다면 `getCurrentUser`가 던지는 Error에 `status`를 실어 보내야 합니다 — 현재는 `"current user unavailable: 401"` 문자열뿐이라 파싱 없이는 status를 알 수 없습니다.

#### 4순위 — 로그아웃을 세션 오류로 오인하지 않기

로그아웃 클릭 시 모듈 수준 플래그(예: `signingOut = true`)를 세우고 `showExpired`에서 이를 확인해 조기 반환하면, 로그아웃 도중 뜨는 유령 모달이 사라집니다. `/logout` 내비게이션이 시작된 뒤의 401은 정상 흐름이지 오류가 아닙니다.

#### 5순위 — 모달 탈출구 추가

닫기 버튼 / ESC / 배경 클릭 중 하나는 열어두는 편이 좋습니다. 위 수정이 모두 반영돼도, 네트워크 단절 같은 예기치 못한 상황에서 사용자가 UI에 갇히지 않으려면 최후의 탈출구가 필요합니다.

---

## 2. 요청 폭발 관련 문제

### 문제 상황

`api.js:559` `listMeetings()`가 회의 **1건당 4개의 HTTP 요청**을 발생시킵니다:

```
GET /api/Meetings                    ← 전체 목록 (페이지네이션 없음)
  회의마다 반복:
    POST getMeetingStatus            ← 상태
    GET  Transcripts?$filter=...     ← 전사문 전체
    GET  Summaries?$filter=...       ← 요약 전체
    GET  ReviewItems?$filter=...     ← 검토 후보
```

`capJobFromMeeting()`(`api.js:362-369`)이 목록 항목마다 호출되기 때문입니다. 총 요청 수는 **1 + 4N**:

|회의 수|요청 수|비고|
|---|---|---|
|20건|81|현재는 견딜 만함|
|100건|401|체감 저하 시작|
|300건|1,201|사실상 사용 불가|

더 큰 문제는 **목록 한 번 그리려고 모든 회의의 전사문 전문을 브라우저로 내려받는다**는 점입니다. 1시간 회의 전사문이 수십 KB라면 100건에서 수 MB가 됩니다. 검색이 전사 내용까지 되는 이유가 바로 이것 — `renderMeetingListView()`가 **클라이언트에서 필터링**하기 때문입니다.

같은 패턴이 상세 화면 폴링에도 있습니다. `app.js:2078`이 2초마다 `tick`을 돌리고, `getJob` → `capGetJob` → `capJobFromMeeting`이 또 4요청을 만듭니다. **회의 상세를 열어두면 2초마다 4개 요청**이 나갑니다.

#### 현재 비용 해부

목록 1회 로드 시 회의 N건에 대해:

|호출|횟수|실제 목록에서 쓰이는가|
|---|---|---|
|`GET /api/Meetings`|1|✅|
|`POST getMeetingStatus`|N|✅ 상태·큐 정보|
|`GET Transcripts`|N|⚠️ **검색 텍스트 생성에만**|
|`GET Summaries`|N|⚠️ **검색 텍스트 생성에만**|
|`GET ReviewItems`|N|❌ **전혀 안 씀**|

**총 `1 + 4N` 요청.** 여기에 서버 측 DB 쿼리가 겹칩니다. `meetingStatusFor`(`meeting-service.js:384`)는 회의당 2~4개 쿼리를 돌리는데, 그중 `transcriptionQueueInfo`(`:430`)가 특히 나쁩니다:

```
const rows = await SELECT.from(TranscriptionDispatches).where({status: {in:[...]}})  // 전체 스캔
const meetings = await SELECT.from(Meetings).where({ID: {in: meetingIds}})           // 전체 스캔
```

**큐 정보는 전역 값인데 회의마다 똑같이 다시 계산합니다.** N=100이면 동일한 전체 스캔을 200번 반복합니다.

핵심 진단 3가지:

1. `ReviewItems` N회 조회는 **순수 낭비** (목록 렌더러 어디에도 안 쓰임)
2. `Transcripts`/`Summaries` N회 조회는 **클라이언트 검색 하나 때문**
3. 큐 계산은 **1회면 될 것을 N회** 반복

### 개선 계획

#### Phase 1 — 서버에 목록 전용 action 신설 (핵심)

**목표**: `1 + 4N` 요청 → **1 요청**

#### 1-1. CDS 계약 추가 — `meeting-service.cds`

```
type MeetingListItem {
  ID, title, category, participants, ownerName,
  status, percent, durationSec, createdAt,
  dispatchStatus, queuePosition, queueAheadCount, queueSize,  
  canEdit, canDelete
}
type MeetingListResult {
  items      : array of MeetingListItem;
  total      : Integer;          // 페이지네이션용 전체 건수
  categories : array of String;  // 필터 옵션 (아래 주의사항 참조)
  people     : array of String;  // 참여자 자동완성
}
action getMeetingList(
  search: String, stage: String, category: String,
  top: Integer, skip: Integer
) returns MeetingListResult;
```

필드 목록은 `renderMeetingList`와 `queueStatusText`, `renderMeetingListView`가 실제로 참조하는 것만 추린 결과입니다. `transcript`·`note`·`transcript_review_items`는 목록에서 제외됩니다.

#### 1-2. 큐 계산을 1회로 — 가장 큰 서버 이득

`transcriptionQueueInfo(meeting, dispatch)`를 **인덱스 빌더**로 리팩터링:

```
buildQueueIndex()  →  Map<meetingId, {position, aheadCount, size}>
```

전체 스캔 2회로 전 회의의 큐 위치를 한 번에 계산합니다. 기존 `getMeetingStatus`도 이 인덱스를 쓰도록 바꾸면 상세 화면 폴링도 함께 가벼워집니다.

#### 1-3. dispatch 배치 조회

회의별 `SELECT.one.from(TranscriptionDispatches)` N회 → `WHERE meeting_ID IN (...)` 1회 후 메모리에서 매핑.

#### 1-4. 서버 사이드 검색

2단계로 나눠 각각 O(1) 쿼리:

```
① Meetings 테이블 검색: title, category, participants, sharedWith, ownerName  → LIKE
② 내용 검색: Transcripts.content / Summaries.content LIKE → meeting_ID 집합 추출
③ ① OR ② 를 meetingAccessWhere(req) 와 AND 결합
```

**권한 필터는 반드시 기존 `meetingAccessWhere()`를 재사용**해야 합니다. 새 경로에서 권한 로직을 다시 짜면 우회 취약점이 생깁니다. 이 계획에서 가장 주의할 지점입니다.

> **성능 주의**: `LIKE '%키워드%'`는 HANA에서 전체 스캔입니다. 수백 건까지는 충분하지만, 수천 건이 되면 HANA Full-Text Index를 걸거나 `accessTokens`처럼 비정규화 `searchText` 컬럼을 두는 방식으로 확장해야 합니다. Phase 1에서는 스키마 변경 없이 가고, 실측 후 판단하는 것을 권합니다.

**예상 효과**: 100건 기준 401요청 → 1요청, DB 쿼리 ~600개 → ~5개.

#### Phase 2 — 프론트 전환

#### 2-1. `api.js` `listMeetings()` 교체

새 action 하나만 호출하도록 변경. `capJobFromMeeting`은 **상세 화면 전용**으로 남깁니다(삭제하면 안 됨).

#### 2-2. 클라이언트 필터링 제거

`renderMeetingListView()`의 `meetingsCache.filter(...)`와 `buildMeetingSearchText()`를 걷어내고, 검색어·필터 변경 시 서버 재조회로 전환합니다.

- **debounce 300ms** 필수 (타이핑마다 요청 방지)
- 조회 중 스켈레톤 유지 — 이미 `loadList()`에 있는 패턴 재사용
- 기존 URL 동기화(`syncListFiltersToUrl`)는 **그대로 유지** — 상세에서 돌아왔을 때 조건 복원은 지켜야 할 UX

#### 2-3. 반드시 함께 고쳐야 할 4가지 (놓치기 쉬움)

목록 캐시에 의존하던 기능들이 페이지네이션으로 깨집니다:

|기능|현재 구현|조치|
|---|---|---|
|카테고리 필터 옵션|`knownCategories()`가 `meetingsCache`에서 수집|서버가 `categories` 반환|
|참여자 자동완성|`knownPeopleValues()`가 동일|서버가 `people` 반환|
|전체 건수 표시|`updateListCount()`가 `meetingsCache.length`|서버 `total` 사용|
|목록 캐시 재사용|`if (!meetingsCache.length)` 스켈레톤 분기|페이지 단위 캐시로 의미 재정의|

이 4개를 빠뜨리면 "카테고리 필터에 항목이 안 뜬다" 같은 버그로 돌아옵니다. Phase 1의 `MeetingListResult`에 `categories`/`people`/`total`을 넣어둔 이유입니다.

#### Phase 3 — 상세 화면 폴링 최적화

Phase 1·2와 독립적으로 진행 가능하며, 체감 효과가 큽니다.

현재 `app.js:2078`이 **2초마다** `tick` → `getJob` → `getMeetingStatus` + 3회 조회 = **4요청/2초 = 120요청/분**. 회의 상세를 열어두기만 해도 계속 나갑니다.

**개선**

1. 폴링은 `getMeetingStatus` **1회만** 호출
2. `transcript`/`summary`는 **상태 전이 시점에만** 1회 로드 (`transcribing → summarizing`, `summarizing → done`)
3. `done`/`failed` 도달 시 폴링 중지 — `stopPolling()`이 이미 있으니 호출 조건만 보강
4. 백오프 도입: 처음 30초는 2초, 이후 5초, 2분 후 10초

**예상 효과**: 120요청/분 → 30요청/분, 완료 후 0.

### 검증 계획

|검증 항목|방법|
|---|---|
|권한 회귀|생성자/참여자/공유자/무관한 사용자 4역할로 목록 노출 범위 확인 — **최우선**|
|검색 동등성|전환 전후 동일 키워드로 결과 집합 비교 (제목·참여자·요약·전사 각각)|
|큐 표시|전사 대기 2건 이상 만들고 `queuePosition`/`aheadCount` 일치 확인|
|성능|시딩 스크립트로 200건 생성 후 DevTools 요청 수·전송량·소요시간 before/after|
|필터 UX|카테고리 옵션, 참여자 자동완성, 건수 표시, URL 복원|

앞서 분석에서 확인한 내용입니다:

- JS 테스트 3개(`tests/ai-core-summary.test.js`, `transcript-validation`, `transcription-dispatcher`)가 **어떤 npm 스크립트에도 물려 있지 않습니다**
- `npm test`는 pytest만 돌리는데, 26개 중 21개가 폐기된 `legacy/fastapi` 대상입니다

CAP의 요약 chunking·transcript 검증·큐 로직을 지키는 테스트가 **있는데 안 돌고 있습니다.** `node --test tests/`를 스크립트로 추가하고 pytest를 현재 코드로 좁히는 것만으로 회귀 안전망이 생깁니다. 투입 대비 효과가 가장 큽니다.