# OTT Module 3·6 — 프론트엔드 작업 가이드 (백엔드 기준)

> **작성일**: 2026-06-18
> **백엔드 담당**: 염정운 (기능3 구독자 추세 분석 + 기능6 자유 게시판)
> **대상**: 이 백엔드로 프론트엔드를 구현할 동료
> **목적**: 어떤 화면을 만들고, 어떤 기능을 넣고, 어떤 API를 호출할지 정리

---

## 0. 백엔드 연결 기본

| 항목 | 값 |
|---|---|
| 프로토콜 | **OData V2** (`sap.ui.model.odata.v2.ODataModel`) — 기존 cap-app 전역 V2 사용 |
| Base 경로 | `/srv-api/odata/v2/ott/trend/TrendAnalysis` · `/srv-api/odata/v2/ott/trend/AnomalyAlert` (기능3)<br>`/srv-api/odata/v2/ott/board/FreeBoard` (기능6) |
| 인증 | 기능3 = **`Admin` 롤 필수** (서비스 전체. READ 조회도 Admin)<br>기능6 = `authenticated-user` (회원 누구나) |
| 개발 테스트 계정 (비번 빈칸) | `admin1`(Admin 롤) / `U001`~`U003`(게시판 테스트용) |
| 핵심 원칙 | **액션에 user_id를 보내지 않음** — 로그인 사용자 기준 자동 처리<br>CAP V2에서 액션은 `oModel.callFunction("/actionName", { method: "POST", urlParameters: {...} })` |

> 로컬에서 백엔드만 단독 기동: `cd cap-node/ott && npm run watch` → `http://localhost:8083`. 테스트: `node test/run-all-tests.mjs`

### ⚠️ V2 어댑터 의존성 (필수)

본 명세서의 모든 API는 **OData V2** 경로(`/srv-api/odata/v2/...`)를 기준으로 한다. CAP 백엔드는 기본 V4 프로토콜만 제공하므로, V2 접근을 위해 `@cap-js-community/odata-v2-adapter` 패키지가 **반드시** 설치되어 있어야 한다.

| 항목 | 값 |
|---|---|
| 패키지명 | `@cap-js-community/odata-v2-adapter` |
| 설치 위치 | `cap-node/ott/package.json` → `dependencies` |
| 버전 | `^1.15.1` (`core`와 동일) |
| 설치 확인 | `GET /odata/v2/ott/...` → 200 OK (미설치 시 404) |
| 배포 시 | `dependencies`에 포함되어 있으므로 자동 설치됨 |

> **참고**: `cap-node/core`는 이미 설치되어 있으나, `cap-node/ott`는 별도 프로젝트이므로 개별 추가해야 한다. 배포(MTA build) 시 `package.json` 기준으로 설치되므로 운영 환경에서도 정상 작동한다.

---

## 1. 구독자 추세 분석 대시보드 (기능3) — Admin 전용

### TR-01. 대시보드 메인

- **목적**: 선택한 월의 그룹별 구독·시청 추세를 KPI 카드 + 차트 + 테이블로 시각화
- **데이터 소스**: `ottTrendService` (`/srv-api/odata/v2/ott/trend/TrendAnalysis`)

#### 조회 API

| 용도 | 호출 |
|---|---|
| 그룹 목록 | `GET /SubscriberGroupSet` |
| 선택 월의 그룹별 추세 | `GET /GroupTrendSet?$filter=analysis_month eq '2026-03'&$expand=group` |
| 장르별 소비 통계 | `GET /ContentGroupStatSet?$filter=analysis_month eq '2026-03'&$orderby=watch_count desc` |
| 장르 통계 + 그룹 필터 | `GET /ContentGroupStatSet?$filter=analysis_month eq '2026-03' and group_id eq 'SG_AGE_30S'` |

#### 실행 API

| 동작 | 호출 | 응답 |
|---|---|---|
| **분석 실행** | `POST /calculateTrends` `{ "analysisMonth": "2026-03" }` | `{ "trend_count": 10, "anomaly_count": 1, "anomalies": [...] }` |

- **멱등성**: 동일 월 재호출해도 결과 같음 (기존 데이터 삭제 후 재생성)
- **응답 필드명 주의**: `trend_count`, `anomaly_count` (snake_case)

#### 화면 구성

- **상단 바**: 분석월 선택(DatePicker `yyyy-MM`) + "분석 실행" 버튼 + 🔔 알림 배지(미읽음 개수)
- **KPI 카드 4개**: 총 구독자(`sum total_members`) · 활성 구독자(`sum active_count`) · 평균 이탈률 · 평균 시청시간 — `GroupTrendSet`에서 집계
- **그룹별 추세 테이블**: 그룹명(`group/group_name`) · 인원 · 활성 · 이탈률(%) · 평균시청(초). 이탈률 50%↑ 빨간색 강조
- **장르별 소비 통계**: 그룹 필터(Select) + 장르·시청횟수·시청시간 테이블. `ContentGroupStatSet` 사용

#### ⚠️ 주의

- `SubscriberGroupSet`의 `group_id`는 `SG_AGE_30S` 형식 (prefix `SG_` 포함)
- `GroupTrendSet.group` expand 시 `group_name`은 `group/group_name` 경로
- **전체 서비스 Admin 롤 필요** — 일반 회원이 접근 시 403. UI에서도 권한 분기 처리

---

### TR-02. 이상 알림 목록 (팝업)

- **목적**: 추세 분석 중 발생한 이상 알림을 확인하고 읽음 처리
- **데이터 소스**: `ottAnomalyService` (`/srv-api/odata/v2/ott/trend/AnomalyAlert`)

#### 조회 API

| 용도 | 호출 |
|---|---|
| 미읽음 알림 개수 (배지) | `GET /TrendAnomalySet?$filter=is_read eq false&$count=true` |
| 알림 목록 | `GET /TrendAnomalySet?$filter=is_read eq false&$orderby=anomaly_month desc,severity desc` |

#### 실행 API

| 동작 | 호출 |
|---|---|
| 단일 읽음 처리 | `POST /markAsRead` `{ "anomalyId": "uuid" }` |
| 전체 읽음 처리 | 위 호출을 미읽음 알림 개수만큼 반복 |

#### 에러 처리

| 상황 | 응답 |
|---|---|
| `anomalyId` 누락 | 400 |
| 존재하지 않는 ID | 404 "이상 알림을 찾을 수 없습니다" |

#### 화면 구성

- **Dialog**에 알림 리스트 표시
- 알림 카드: severity 아이콘(🔴🟡🔵) + anomaly_type + anomaly_month + alert_text 한글 메시지 + "읽음 처리" 버튼
- 상단 툴바: "모두 읽음" 버튼

#### Severity → UI 매핑

| severity | 색상 | 아이콘 |
|---|---|---|
| `CRITICAL` | Error (빨강) | `sap-icon://alert` |
| `WARN` | Warning (주황) | `sap-icon://warning` |
| `INFO` | Information (파랑) | `sap-icon://hint` |

---

### TR-03. 대시보드 컨트롤러 핵심 로직

```js
// ── 분석 실행 ──
onPressAnalyze: function() {
    var oModel = this.getOwnerComponent().getModel(); // ottTrendService
    var sMonth = this.getView().getModel("viewModel").getProperty("/analysisMonth");

    oModel.callFunction("/calculateTrends", {
        method: "POST",
        urlParameters: { "analysisMonth": sMonth }
    }).then(function(oResult) {
        // oResult = { trend_count, anomaly_count, anomalies }
        oViewModel.setProperty("/trendCount", oResult.trend_count);
        oViewModel.setProperty("/anomalyCount", oResult.anomaly_count);
        this._loadDashboard();    // GroupTrend + KPI 갱신
        this._loadAnomalyBadge(); // 알림 배지 갱신
    }.bind(this)).catch(function(oErr) {
        MessageBox.error("분석 오류: " + oErr.message);
    });
},

// ── 읽음 처리 ──
onPressMarkAsRead: function(oEvent) {
    var sAnomalyId = oEvent.getSource().getBindingContext("viewModel").getProperty("anomaly_id");
    var oAnomalyModel = this.getOwnerComponent().getModel("anomalyModel");
    oAnomalyModel.callFunction("/markAsRead", {
        method: "POST",
        urlParameters: { "anomalyId": sAnomalyId }
    }).then(function() {
        this._loadAnomalyList();
        this._loadAnomalyBadge();
    }.bind(this));
},

// ── 이탈률 → 색상 (formatter) ──
churnStateFormatter: function(fChurn) {
    if (fChurn >= 50) return "Error";
    if (fChurn >= 20) return "Warning";
    return "Success";
}
```

---

## 2. 자유 게시판 (기능6) — 회원 누구나

### FB-01. 게시글 목록

- **목적**: 삭제되지 않은 게시글을 최신순으로 조회 + 검색
- **데이터 소스**: `ottBoardService` (`/srv-api/odata/v2/ott/board/FreeBoard`)

#### 조회 API

| 용도 | 호출 |
|---|---|
| 게시글 목록 (검색 포함) | `GET /PostSet?$filter=deleted_flag eq false and (contains(title,'검색어') or contains(content,'검색어'))&$orderby=createdAt desc&$expand=comments&$count=true` |

#### 화면 구성

- **상단 검색창** + "글쓰기" 버튼
- **게시글 리스트**: 제목 · 작성자(`author_user_id`) · 작성일시(`createdAt`) · 조회수(`view_count`) · 댓글 수
- 검색어는 제목+내용 OR 조건
- 목록 조회 시에는 `view_count` 증가하지 않음 (단건 조회만 증가)

---

### FB-02. 게시글 상세 + 댓글

#### 조회 API

| 용도 | 호출 |
|---|---|
| 게시글 단건 (댓글 포함) | `GET /PostSet('{post_id}')?$expand=comments` |

- ⚠️ 단건 조회 시 서버에서 **`view_count` 자동 +1** → 조회수 즉시 갱신

#### 실행 API

| 동작 | 메서드 | 호출 | 본인 확인 |
|---|---|---|---|
| 게시글 수정 | `PATCH` | `/PostSet('{post_id}')` `{ "title": "...", "content": "..." }` | ✅ |
| 게시글 삭제 | `DELETE` | `/PostSet('{post_id}')` | ✅ |
| 댓글 작성 | `POST` | `/CommentSet` `{ "post_id": "...", "content": "..." }` | — |
| 댓글 수정 | `PATCH` | `/CommentSet('{comment_id}')` `{ "content": "..." }` | ✅ |
| 댓글 삭제 | `DELETE` | `/CommentSet('{comment_id}')` | ✅ |

#### ⚠️ 핵심 규칙

| 규칙 | 설명 |
|---|---|
| **`author_user_id` 전송 금지** | CREATE 시 백엔드 `before` 핸들러가 자동 주입. 클라이언트는 `title`, `content`만 전송 |
| **게시글 DELETE = 소프트 삭제** | 실제 삭제 안 됨. `deleted_flag = true`로 설정. 목록에서 `deleted_flag eq false` 필터 필수 |
| **댓글 DELETE = 물리 삭제** | 실제로 삭제됨 |
| **본인 확인** | 수정·삭제 시 서버 검증 → 타인이 시도하면 **403** 반환 |

#### 에러 처리 (서버가 한글 메시지 반환 → 토스트로 표시)

| 상황 | 응답 |
|---|---|
| 미인증 | 401 "인증된 사용자만 게시글을 작성할 수 있습니다" |
| 타인 수정/삭제 | 403 "본인이 작성한 게시글만 수정/삭제할 수 있습니다" |
| 존재하지 않는 글 | 404 "게시글을 찾을 수 없습니다" |
| 타인 댓글 수정/삭제 | 403 "본인이 작성한 댓글만 수정/삭제할 수 있습니다" |
| 존재하지 않는 댓글 | 404 "댓글을 찾을 수 없습니다" |

#### 화면 구성

- **상단**: ← 목록 버튼 + [수정] [삭제] 버튼 (본인 글일 때만 활성화)
- **게시글 본문**: 제목 · 내용 · 작성자 · 작성일시 · 조회수
- **수정 모드**: 제목 Input + 내용 TextArea + [저장] [취소]
- **댓글 영역**: 댓글 리스트(작성자·내용·일시) + 각 댓글 [삭제] 버튼 (본인 댓글만)
- **댓글 작성**: TextArea + "댓글 등록" 버튼

#### 수정/삭제 버튼 visible 표현식

```xml
<!-- 수정 버튼: 본인 글 + 수정모드 아닐 때 -->
<Button icon="sap-icon://edit" press="onPressEdit"
    visible="{= ${viewModel>/currentPost/author_user_id} === ${User>/userId} && !${viewModel>/editMode}}"/>

<!-- 삭제 버튼: 본인 글일 때 -->
<Button icon="sap-icon://delete" press="onPressDelete"
    visible="{= ${viewModel>/currentPost/author_user_id} === ${User>/userId}}"/>

<!-- 댓글 삭제 버튼: 본인 댓글일 때 -->
<Button icon="sap-icon://delete" press="onPressDeleteComment"
    visible="{= ${viewModel>author_user_id} === ${User>/userId}}"/>
```

---

### FB-03. 게시글 작성 (Dialog)

#### 실행 API

| 동작 | 호출 |
|---|---|
| 게시글 등록 | `POST /PostSet` `{ "title": "제목", "content": "내용" }` |

- 응답: 201 Created → `post_id` 포함된 객체 반환
- 등록 후 목록으로 이동 또는 목록 재조회

---

## 3. 화면 레이아웃 (와이어프레임)

### TR-01. 대시보드 메인

```
┌──────────────────────────────────────────────────────────────┐
│ 📊 구독자 추세 분석 대시보드        🔔 (1)   [2026-03 ▼] [분석실행]│
├──────────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ 총 구독자 │ │ 활성 구독 │ │ 평균이탈률│ │ 평균시청  │       │
│  │   47명   │ │   38명   │ │  12.5%   │ │ 8,230초  │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│                                                              │
│  ┌──────────────────────┐ ┌──────────────────────┐           │
│  │  그룹별 이탈률 (막대)  │ │  그룹별 시청시간 (막대)│           │
│  └──────────────────────┘ └──────────────────────┘           │
│                                                              │
│  📋 그룹별 월간 추세                                          │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ 그룹명 │ 인원 │ 활성 │ 이탈률 │ 시청시간               │    │
│  │ 20대  │  4  │  3  │  0%   │ 7,500                  │    │
│  │ 40대  │  2  │  1  │ 50% 🔴│ 12,000                 │    │
│  │ ...   │ ... │ ... │ ...   │ ...                    │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  📊 장르별 소비 통계    [그룹 필터▼]                          │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ 그룹  │ 장르   │ 시청수 │ 시청시간                     │    │
│  │ 30대 │ 액션   │  12   │ 8,520                      │    │
│  │ 30대 │ 스릴러 │   8   │ 3,300                      │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

- KPI 4종 = `GroupTrendSet`에서 `total_members`·`active_count` 합계 / `churn_rate`·`avg_watch_seconds` 평균
- 🔔 배지 = `TrendAnomalySet?$filter=is_read eq false&$count=true` 개수

### TR-02. 이상 알림 팝업

```
┌──────────────────────────────────────────┐
│  이상 탐지 알림                 [모두 읽음]│
├──────────────────────────────────────────┤
│  ┌────────────────────────────────────┐  │
│  │ 🔴 CRITICAL | CHURN_SPIKE | 04월  │  │
│  │ SG_AGE_40S 그룹의 이탈률이         │  │
│  │ 전월 대비 50%p 급증했습니다.       │  │
│  │                         [읽음처리]  │  │
│  ├────────────────────────────────────┤  │
│  │ 🟡 WARN | WATCH_DROP | 04월       │  │
│  │ SG_PLAN_PREMIUM 그룹의 시청시간이  │  │
│  │ 전월 대비 35% 감소했습니다.        │  │
│  │                         [읽음처리]  │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
```

- `anomaly_type` + `severity` → 아이콘·색상, `alert_text` 그대로 표시, `anomaly_month` 표시

### FB-01. 게시글 목록

```
┌────────────────────────────────────────────┐
│  자유 게시판                     [글쓰기]    │
├────────────────────────────────────────────┤
│  🔍 [                    ]                  │
├────────────────────────────────────────────┤
│  ┌──────────────────────────────────────┐  │
│  │ 📌 U001의 첫 번째 게시글    👁️ 15    │  │
│  │    Kim Minjun | 2026-06-17 10:30     │  │
│  │    댓글 3개                           │  │
│  ├──────────────────────────────────────┤  │
│  │ 📌 U002의 게시글            👁️ 8     │  │
│  │    Lee Seoyeon | 2026-06-17 09:15    │  │
│  │    댓글 0개                           │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
```

- 각 행 = `title`·`author_user_id`·`createdAt`·`view_count` + `comments.results.length`
- 터치/클릭 → FB-02 상세 이동 (`post_id` 라우팅)

### FB-02. 게시글 상세 + 댓글

```
┌────────────────────────────────────────────┐
│  ← 목록                [✏️ 수정] [🗑️ 삭제]   │
├────────────────────────────────────────────┤
│  U001의 첫 번째 게시글                      │
│                                              │
│  안녕하세요! 자유 게시판 테스트입니다.       │
│  이 글은 U001이 작성했습니다.                │
│                                              │
│  Kim Minjun | 2026-06-17 10:30 | 👁️ 16     │
│                                              │
│  ── 댓글 2개 ────────────────────────────    │
│  ┌──────────────────────────────────────┐   │
│  │ U002: 좋은 글이네요!                  │   │
│  │ 2026-06-17 11:00          [🗑️ 삭제]  │   │
│  ├──────────────────────────────────────┤   │
│  │ U003: 저도 동의합니다.               │   │
│  │ 2026-06-17 11:30          [🗑️ 삭제]  │   │
│  └──────────────────────────────────────┘   │
│                                              │
│  ── 댓글 작성 ──────────────────────────    │
│  ┌──────────────────────────────────────┐   │
│  │ [                                  ] │   │
│  │                       [댓글 등록]     │   │
│  └──────────────────────────────────────┘   │
└────────────────────────────────────────────┘
```

- 수정/삭제 버튼 = `author_user_id === User>/userId` 일 때만 표시
- 단건 조회 = `view_count` 자동 증가 → 재진입 시 숫자 증가 확인 가능
- 수정 모드: 제목 Input + 내용 TextArea + [저장] [취소]

---

## 4. 프론트가 알아야 할 도메인 규칙

| 규칙 | 의미 |
|---|---|
| **기능3 = Admin 전용** | 서비스 전체에 `@(requires: 'Admin')` 적용. SubscriberGroupSet, GroupTrendSet 등 READ 조차 Admin 롤 필요. 일반 회원 접근 시 403 |
| **calculateTrends 멱등성** | 같은 월 여러 번 실행해도 결과 동일 (기존 데이터 삭제 후 재생성). "분석 실행" 버튼 중복 클릭 무방 |
| **응답 필드명 snake_case** | `calculateTrends` 반환값은 `trend_count`, `anomaly_count` (camelCase 아님) |
| **게시글 author_user_id 전송 금지** | CREATE 시 `title`, `content`만 전송. 백엔드가 `req.user.id`로 자동 주입 |
| **게시글 DELETE = 소프트 삭제** | `DELETE /PostSet('uuid')` 호출해도 실제 삭제 안 됨. `deleted_flag = true` 처리. 목록 조회 시 `$filter=deleted_flag eq false` 필수 |
| **댓글 DELETE = 물리 삭제** | `DELETE /CommentSet('uuid')` → 실제 DB에서 삭제됨 |
| **조회수 증가 조건** | 단건 조회(GET by key 또는 `$filter=post_id eq ...`로 1건) 시에만 `view_count` 자동 +1. 목록 조회는 증가 안 함 |
| **본인 확인 = 서버 검증** | 수정·삭제 시 서버에서 `author_user_id` 비교. 타인이면 403 → 클라이언트는 에러 메시지를 토스트로 표시 |

---

## 5. 개발 테스트 시나리오 (mocked 계정)

```
[기능3 — Admin 전용]
admin1 로그인 → 대시보드 진입 → 2026-03 분석 실행 → GroupTrend 10건 확인
             → 2026-04 분석 실행 → 이상 알림 1건 이상 발생 확인
             → 알림 팝업 → 읽음 처리 → 배지 0으로 갱신
             → 분석월 변경 → 테이블/통계 갱신
U001 로 대시보드 접근 → 403 (Admin 롤 없음)

[기능6 — 회원 누구나]
U001 로그인 → 게시글 3개 작성 (U001, U002, U003 각각)
           → 목록에서 3개 확인
           → U001 글 상세 진입 → 조회수 증가 확인 (재진입 시 +1)
           → 본인 글 수정 → 성공
           → U002 글 수정 시도 → 403 "본인이 작성한 게시글만..."
           → U003 글 삭제 → 목록에서 사라짐 (소프트 삭제)
           → 댓글 작성·수정·삭제 → 본인 확인 동작 확인
U004 로그인 → 글쓰기 → 401 (미인증)
```

---

## 6. 화면 ↔ API 한눈 요약

| 화면 | 조회(GET) | 실행(POST/PATCH) |
|---|---|---|
| TR-01 대시보드 | `/TrendAnalysis/SubscriberGroupSet`<br>`/TrendAnalysis/GroupTrendSet?$expand=group`<br>`/TrendAnalysis/ContentGroupStatSet` | `POST /TrendAnalysis/calculateTrends` |
| TR-02 알림 팝업 | `/AnomalyAlert/TrendAnomalySet?$filter=is_read eq false` | `POST /AnomalyAlert/markAsRead` |
| FB-01 게시글 목록 | `/FreeBoard/PostSet?$filter=deleted_flag eq false&$expand=comments` | — |
| FB-02 게시글 상세 | `/FreeBoard/PostSet('{id}')?$expand=comments` | `PATCH /FreeBoard/PostSet('{id}')`<br>`DELETE /FreeBoard/PostSet('{id}')`<br>`POST /FreeBoard/CommentSet`<br>`PATCH /FreeBoard/CommentSet('{id}')`<br>`DELETE /FreeBoard/CommentSet('{id}')` |
| FB-03 게시글 작성 | — | `POST /FreeBoard/PostSet` |

> 참고 문서: 백엔드 테스트 결과는 `작성 내용/설계 및 개발/기능3_6_통합_테스트_결과보고서.md`, `.http` 테스트 파일은 `cap-node/ott/test/Module36Test.http`

---

## 7. manifest.json 데이터소스 설정

```json
"dataSources": {
    "ottTrendService": {
        "uri": "/srv-api/odata/v2/ott/trend/TrendAnalysis",
        "type": "OData",
        "settings": { "odataVersion": "2.0" }
    },
    "ottAnomalyService": {
        "uri": "/srv-api/odata/v2/ott/trend/AnomalyAlert",
        "type": "OData",
        "settings": { "odataVersion": "2.0" }
    },
    "ottBoardService": {
        "uri": "/srv-api/odata/v2/ott/board/FreeBoard",
        "type": "OData",
        "settings": { "odataVersion": "2.0" }
    }
}
```

> `ottTrendService` + `ottAnomalyService`는 Admin 롤 필요 → 라우팅 가드 또는 접근 제어 로직 추가 필수

---

## 8. 개발 순서

### 기능3 (TrendAnalysis)

1. `sysmgt/webapp/trendAnalysis/` 디렉토리 생성
2. `package.json`, `ui5.yaml`, `index.html` 작성 (기존 모듈 복사)
3. `manifest.json` — 데이터소스 2개 등록 (Admin 권한 주석)
4. `Component.js` 작성
5. `controller/Dashboard.controller.js` — 대시보드 로직
6. `view/Dashboard.view.xml` — KPI+테이블+차트
7. `view/AnomalyList.view.xml` — 알림 Dialog
8. 테스트: `admin1` 로그인 후 분석 실행 → 알림 확인

### 기능6 (FreeBoard)

1. `sysmgt/webapp/freeBoard/` 디렉토리 생성
2. 공통 파일 작성
3. `manifest.json` — 라우팅 포함 (`PostList` + `PostDetail`)
4. `Component.js` (Router 초기화)
5. `controller/PostList.controller.js` — 목록 + 검색
6. `controller/PostDetail.controller.js` — 상세 + 수정 + 삭제 + 댓글
7. `view/PostList.view.xml`
8. `view/PostDetail.view.xml`
9. `view/PostCreate.view.xml` (Dialog 프래그먼트)
10. 테스트: U001~U003으로 CRUD + 권한 검증
