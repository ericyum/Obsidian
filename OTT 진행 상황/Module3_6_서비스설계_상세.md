# Module 3 & 6 — 서비스 설계 상세 (2026-06-12)

**작성자**: 염정운 (Backend)
**대상**: `cap-node/ott/srv/`
**근거**: 통합 회의 (2026-06-11) — OTT-통합-Entity설계.md 쟁점#6

---

## 목차

1. [전체 구조](#1-전체-구조)
2. [Module 3 — TrendAnalysis 서비스](#2-module-3--trendanalysis-서비스)
3. [Module 3 — AnomalyAlert 서비스](#3-module-3--anomalyalert-서비스)
4. [Module 6 — FreeBoard 서비스](#4-module-6--freeboard-서비스)
5. [서비스-엔티티 매핑 근거](#5-서비스-엔티티-매핑-근거)
6. [OData API 명세](#6-odata-api-명세)
7. [핸들러 구현 계획](#7-핸들러-구현-계획)
8. [파일 생성 체크리스트](#8-파일-생성-체크리스트)

---

## 1. 전체 구조

```
cap-node/ott/
├── srv/
│   ├── index.cds                          ← 서비스 통합 (3개 import)
│   ├── cds/
│   │   ├── trend/
│   │   │   ├── TrendAnalysis-service.cds   ← Module 3: 분석 + 액션
│   │   │   └── AnomalyAlert-service.cds    ← Module 3: 알림 전용
│   │   └── board/
│   │       └── FreeBoard-service.cds       ← Module 6: 게시글 + 댓글 통합
│   └── src/
│       ├── config/
│       │   └── .gitkeep
│       ├── feature/
│       │   ├── trend/
│       │   │   ├── TrendAnalysis.handler.ts
│       │   │   └── AnomalyAlert.handler.ts
│       │   └── board/
│       │       ├── Post.handler.ts
│       │       └── Comment.handler.ts
│       └── utils/
│           ├── services/
│           └── types/
│               └── typings.d.ts
```

### 서비스 요약

| 서비스 | 경로 | 포트 | 전용 엔티티 | 액션 |
|--------|------|:--:|------------|------|
| `TrendAnalysis` | `/odata/v4/ott/trend/TrendAnalysis` | 8083 | `SubscriberGroup`, `GroupTrend`, `ContentGroupStat` | `calculateTrends` |
| `AnomalyAlert` | `/odata/v4/ott/trend/AnomalyAlert` | 8083 | `TrendAnomaly` | `markAsRead` |
| `FreeBoard` | `/odata/v4/ott/board/FreeBoard` | 8083 | `Post`, `Comment` | — |

> **공통**: `@(requires: 'authenticated-user')`, `namespace com.cap.ott`

---

## 2. Module 3 — TrendAnalysis 서비스

### 2.1 개요

구독자 그룹별 월간 추세 데이터를 조회하고, `calculateTrends` 액션으로 분석을 실행하는 서비스.

### 2.2 노출 엔티티

| EntitySet | 기반 엔티티 | 용도 |
|-----------|------------|------|
| `SubscriberGroupSet` | `SubscriberGroup` | 그룹 마스터 조회 (연령대/성별/플랜 선택) |
| `GroupTrendSet` | `GroupTrend` | 그룹-월 스냅샷 조회 (시계열 그래프 데이터) |
| `ContentGroupStatSet` | `ContentGroupStat` | 그룹-장르 소비 통계 조회 |

### 2.3 액션: `calculateTrends`

```
POST /odata/v4/ott/trend/TrendAnalysis/calculateTrends
Content-Type: application/json

{
  "analysisMonth": "2026-03"
}
```

**내부 처리 흐름** (7단계):

```
┌─ 입력: analysisMonth ("YYYY-MM")
│
├─ 1. SubscriberGroup 목록 조회 (use_flag = true)
│
├─ 2. 각 그룹별 Users 필터링
│     group_type='AGE'    → Users.age_group 매칭
│     group_type='GENDER' → Users.gender 매칭
│     group_type='PLAN'   → Users.subscription_plan 매칭
│
├─ 3. ViewingHistory + Contents JOIN
│     → 그룹 내 사용자의 시청 데이터 집계
│
├─ 4. GroupTrend INSERT
│     - analysis_month = 입력값
│     - total_members, active_count, churn_rate, avg_watch_seconds 계산
│     - 기존 동일 월 데이터 DELETE 후 재생성 (멱등)
│
├─ 5. ContentGroupStat INSERT
│     - Contents.genre 기준 장르별 집계
│     - watch_count, total_seconds
│
├─ 6. 전월 GroupTrend 조회 → 변화율 계산 → TrendAnomaly INSERT
│     ┌─────────────────┬───────────┬─────────────┐
│     │ CHURN_SPIKE     │ +10%p↑    │ +20%p↑      │
│     │ WATCH_DROP      │ -30%↓     │ -50%↓       │
│     │ WATCH_SURGE     │ +40%↑     │ (INFO)      │
│     └─────────────────┴───────────┴─────────────┘
│
└─ 7. 결과 반환
      {
        "trend_count": 14,
        "anomaly_count": 2,
        "anomalies": [...]
      }
```

### 2.4 주요 OData 호출 예시

```http
# 그룹 목록
GET /odata/v4/ott/trend/TrendAnalysis/SubscriberGroupSet

# 특정 그룹의 시계열 추세 (6개월)
GET /odata/v4/ott/trend/TrendAnalysis/GroupTrendSet
  ?$filter=group_group_id eq 'AGE_20S'
   and analysis_month ge '2026-01'
   and analysis_month le '2026-06'
  &$orderby=analysis_month

# 그룹 간 비교: 이탈률
GET /odata/v4/ott/trend/TrendAnalysis/GroupTrendSet
  ?$filter=analysis_month eq '2026-03'
  &$select=group_group_id,churn_rate,active_count
  &$expand=group($select=group_name)

# 20대가 가장 많이 본 장르 TOP 5
GET /odata/v4/ott/trend/TrendAnalysis/ContentGroupStatSet
  ?$filter=group_group_id eq 'AGE_20S' and analysis_month eq '2026-03'
  &$orderby=watch_count desc
  &$top=5
```

---

## 3. Module 3 — AnomalyAlert 서비스

### 3.1 개요

`calculateTrends` 실행 중 자동 생성된 이상 알림을 조회하고 읽음 처리하는 서비스. **TrendAnalysis와 책임 분리**: 분석 실행(무거운 작업)과 알림 확인(가벼운 작업)을 다른 서비스로 구분.

### 3.2 노출 엔티티

| EntitySet | 기반 엔티티 | 용도 |
|-----------|------------|------|
| `TrendAnomalySet` | `TrendAnomaly` | 이상 알림 목록, 상세, 읽음 처리 |

### 3.3 액션: `markAsRead`

```
POST /odata/v4/ott/trend/AnomalyAlert/markAsRead
Content-Type: application/json

{
  "anomalyId": "3f8a7b2c-..."
}
```

→ 해당 `TrendAnomaly.is_read = true` 로 갱신, 갱신된 레코드 반환.

### 3.4 주요 OData 호출 예시

```http
# 읽지 않은 이상 알림 (심각도순)
GET /odata/v4/ott/trend/AnomalyAlert/TrendAnomalySet
  ?$filter=is_read eq false
  &$orderby=severity desc
  &$expand=group($select=group_name)

# 읽음 처리
POST /odata/v4/ott/trend/AnomalyAlert/markAsRead
{"anomalyId": "3f8a7b2c-..."}
```

### 3.5 TrendAnomaly.group 접근

`TrendAnomaly`에는 `group : Association to SubscriberGroup`이 있으므로, `$expand=group`으로 그룹명 조회 가능. 별도로 `SubscriberGroupSet`을 이 서비스에 노출하지 않은 이유:
- 알림 화면에서 새 그룹을 생성할 일은 없음
- 그룹명은 expand로 충분

---

## 4. Module 6 — FreeBoard 서비스

### 4.1 개요

게시글과 댓글을 **단일 서비스**로 통합 제공. Post-Comment는 `Composition` 관계이므로 CAP의 OData 네비게이션을 자연스럽게 활용.

### 4.2 단일 서비스 통합 근거

| 관점 | 설명 |
|------|------|
| **CAP Composition** | Comment는 Post의 자식. 부모 삭제 → 자식 cascade. 하나의 aggregate |
| **OData 네비게이션** | `PostSet({id})/comments` 로 자연스러운 접근 |
| **단순한 도메인** | 카테고리/태그/좋아요/대댓글/첨부파일 모두 의도적 제외. 과도한 분리 불필요 |
| **핸들러 분리** | 코드는 `Post.handler.ts`, `Comment.handler.ts`로 내부 분리 가능 |

### 4.3 노출 엔티티

| EntitySet | 기반 엔티티 | 용도 |
|-----------|------------|------|
| `PostSet` | `Post` | 게시글 CRUD, 목록, 상세, 소프트 삭제 |
| `CommentSet` | `Comment` | 댓글 CRUD (Post 하위) |

### 4.4 CRUD 규칙

| 작업 | Post | Comment |
|------|:--:|:--:|
| **CREATE** | `author` = `$user` 자동 주입 | `author` = `$user`, `post_id` 필수 |
| **READ** | `deleted_flag=false` 기본 필터, 조회 시 `view_count += 1` | — |
| **UPDATE** | 작성자 본인만 | 작성자 본인만 |
| **DELETE** | 소프트 삭제 (`deleted_flag=true`), 작성자 본인만 | 물리 삭제, 작성자 본인만 |

### 4.5 Post.deleted_flag 상세 설계

```
┌─ GET /PostSet
│    └─ @readonly: where deleted_flag = false (기본 필터)
│
├─ GET /PostSet({id})
│    ├─ deleted_flag=false → 정상 조회, view_count += 1
│    └─ deleted_flag=true  → 404 (또는 본인만 조회 가능)
│
├─ DELETE /PostSet({id})
│    └─ 실제 삭제 X → PATCH { deleted_flag: true }
│       댓글은 보존 (soft delete 이므로 cascade 발생 안 함)
│
└─ Permanently deleted record → DB에서 제거되지 않음
```

### 4.6 주요 OData 호출 예시

```http
# 게시글 목록 (최신순, 20개)
GET /odata/v4/ott/board/FreeBoard/PostSet
  ?$filter=deleted_flag eq false
  &$orderby=createdAt desc
  &$top=20

# 게시글 상세 + 댓글 (조회수 자동 증가)
GET /odata/v4/ott/board/FreeBoard/PostSet({{post_id}})
  ?$expand=comments($orderby=createdAt asc)

# 게시글 작성
POST /odata/v4/ott/board/FreeBoard/PostSet
{"title": "제목", "content": "내용"}

# 댓글 작성 (게시글 내비게이션)
POST /odata/v4/ott/board/FreeBoard/PostSet({{post_id}})/comments
{"content": "댓글 내용"}

# 게시글 수정 (본인만)
PATCH /odata/v4/ott/board/FreeBoard/PostSet({{post_id}})
{"title": "수정된 제목"}

# 게시글 소프트 삭제 (본인만)
DELETE /odata/v4/ott/board/FreeBoard/PostSet({{post_id}})
```

---

## 5. 서비스-엔티티 매핑 근거

### 5.1 Module 3: 2개 서비스로 분리한 이유

| 엔티티 | TrendAnalysis | AnomalyAlert | 근거 |
|--------|:--:|:--:|------|
| `SubscriberGroup` | ✅ | ❌ | 그룹 정의는 분석 설정 화면에서 관리. 알림에는 expand로 충분 |
| `GroupTrend` | ✅ | ❌ | 핵심 분석 데이터. `calculateTrends` 액션의 결과물 |
| `ContentGroupStat` | ✅ | ❌ | 분석 보조 데이터. 같은 액션에서 생성 |
| `TrendAnomaly` | ❌ | ✅ | 알림 전용. 읽음/미읽음 관리가 주 용도 |

**핵심**: `calculateTrends`는 무거운 배치 작업(집계 쿼리 다수), `markAsRead`는 가벼운 단일 업데이트. 프론트에서도 "분석 대시보드" vs "알림 센터"는 다른 화면일 가능성이 높음.

### 5.2 Module 6: 1개 서비스로 통합한 이유

| 기준 | 설명 |
|------|------|
| **CAP Composition** | `Post` ──< `Composition` ── `Comment`. 부모-자식 관계를 한 서비스에서 관리하는 게 자연스러움 |
| **RESTful 설계** | `/PostSet({id})/comments` — URL 구조가 직관적 |
| **프론트 화면** | 게시글 목록·상세·작성 화면에서 댓글이 항상 함께 표시됨 |
| **트랜잭션 경계** | 게시글 삭제 시 댓글 cascade — 같은 aggregate에서 처리 |

---

## 6. OData API 명세

### 6.1 공통

- **Base URL**: `https://<host>:8083/odata/v4/ott`
- **인증**: Bearer Token (XSUAA)
- **형식**: JSON

### 6.2 TrendAnalysis

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| `GET` | `/trend/TrendAnalysis/SubscriberGroupSet` | 그룹 목록 |
| `GET` | `/trend/TrendAnalysis/GroupTrendSet` | 추세 스냅샷 조회 |
| `GET` | `/trend/TrendAnalysis/ContentGroupStatSet` | 장르 소비 통계 |
| `POST` | `/trend/TrendAnalysis/calculateTrends` | 분석 실행 |

### 6.3 AnomalyAlert

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| `GET` | `/trend/AnomalyAlert/TrendAnomalySet` | 알림 목록 |
| `GET` | `/trend/AnomalyAlert/TrendAnomalySet({id})` | 알림 상세 |
| `POST` | `/trend/AnomalyAlert/markAsRead` | 읽음 처리 |

### 6.4 FreeBoard

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| `GET` | `/board/FreeBoard/PostSet` | 게시글 목록 (기본 `deleted_flag=false`) |
| `GET` | `/board/FreeBoard/PostSet({id})` | 게시글 상세 (조회수 +1) |
| `POST` | `/board/FreeBoard/PostSet` | 게시글 작성 |
| `PATCH` | `/board/FreeBoard/PostSet({id})` | 게시글 수정 (작성자 본인) |
| `DELETE` | `/board/FreeBoard/PostSet({id})` | 소프트 삭제 (작성자 본인) |
| `GET` | `/board/FreeBoard/CommentSet` | 댓글 목록 (`$filter=post_post_id eq {id}`) |
| `POST` | `/board/FreeBoard/PostSet({id})/comments` | 댓글 작성 (네비게이션) |
| `PATCH` | `/board/FreeBoard/CommentSet({id})` | 댓글 수정 (작성자 본인) |
| `DELETE` | `/board/FreeBoard/CommentSet({id})` | 댓글 삭제 (작성자 본인) |

---

## 7. 핸들러 구현 계획

### 7.1 파일 구성

```
srv/src/feature/
├── trend/
│   ├── TrendAnalysis.handler.ts   ← SubscriberGroupSet, GroupTrendSet,
│   │                                  ContentGroupStatSet + calculateTrends
│   └── AnomalyAlert.handler.ts    ← TrendAnomalySet + markAsRead
│
└── board/
    ├── Post.handler.ts            ← PostSet (CRUD, view_count, soft delete)
    └── Comment.handler.ts         ← CommentSet (CRUD)
```

### 7.2 TrendAnalysis.handler.ts — 주요 구현

```typescript
import cds from '@sap/cds';

export default class TrendAnalysisHandler extends cds.ApplicationService {
  async init() {
    const { SubscriberGroup, GroupTrend, ContentGroupStat, TrendAnomaly,
            Users, ViewingHistory, Contents } = this.entities;

    // ===== calculateTrends 액션 =====
    this.on('calculateTrends', async (req) => {
      const { analysisMonth } = req.data;
      
      // 1. 활성 그룹 목록 조회
      const groups = await SELECT.from(SubscriberGroup);
      
      let trendCount = 0, anomalies = [];
      
      // 2. 기존 데이터 삭제 (멱등)
      await DELETE.from(GroupTrend).where({ analysis_month: analysisMonth });
      await DELETE.from(ContentGroupStat).where({ analysis_month: analysisMonth });
      
      // 3. 그룹별 집계
      for (const g of groups) {
        // Users 필터링 → ViewingHistory 집계 → GroupTrend 생성
        // ... 구현 상세
      }
      
      // 4. 이상 탐지 (전월 비교)
      // ... 구현 상세
      
      return { trend_count: trendCount, anomaly_count: anomalies.length, anomalies };
    });

    await super.init();
  }
}
```

### 7.3 Post.handler.ts — 주요 구현

```typescript
export default class PostHandler extends cds.ApplicationService {
  async init() {
    const { Post } = this.entities;

    // CREATE: 작성자 자동 주입
    this.before('CREATE', Post, (req) => {
      req.data.author_user_id = req.user.id;
    });

    // READ: 조회수 +1 (after READ)
    this.after('READ', Post, async (results, req) => {
      // 단건 조회일 때만 view_count 증가
    });

    // UPDATE/DELETE: 작성자 본인 확인
    this.before(['UPDATE', 'DELETE'], Post, async (req) => {
      // 본인 확인 로직
    });

    // DELETE → 소프트 삭제 (deleted_flag = true)
    this.on('DELETE', Post, async (req) => {
      // 물리 삭제 대신 PATCH
    });

    await super.init();
  }
}
```

### 7.4 핸들러 공통 패턴 (core 참조)

| 패턴 | 코드 | 용도 |
|------|------|------|
| 서비스 클래스 | `export default class Xxx extends cds.ApplicationService` | CAP 서비스 핸들러 |
| 생명주기 훅 | `this.before/on/after('EVENT', Entity, fn)` | CRUD 전후 처리 |
| 트랜잭션 | `cds.transaction(req)` | 복수 테이블 쓰기 |
| 타입 import | `import type { Post } from '#cds-models/com/cap/ott'` | 타입 안전성 |
| 사용자 ID | `req.user.id` | XSUAA 토큰에서 추출 |

---

## 8. 파일 생성 체크리스트

### 이미 생성됨 ✅

```
ott/srv/
├── ✅ index.cds
├── ✅ cds/trend/TrendAnalysis-service.cds
├── ✅ cds/trend/AnomalyAlert-service.cds
└── ✅ cds/board/FreeBoard-service.cds
```

### 아직 미생성 (다음 단계) 🔲

```
ott/srv/src/
├── 🔲 config/.gitkeep
├── 🔲 feature/
│   ├── 🔲 trend/
│   │   ├── 🔲 TrendAnalysis.handler.ts
│   │   └── 🔲 AnomalyAlert.handler.ts
│   └── 🔲 board/
│       ├── 🔲 Post.handler.ts
│       └── 🔲 Comment.handler.ts
└── 🔲 utils/
    ├── 🔲 services/
    └── 🔲 types/
        └── 🔲 typings.d.ts
```

### db 설정 (별도 진행) 🔲

```
ott/db/
├── 🔲 index.cds
├── 🔲 .hdiignore
├── 🔲 undeploy.json
└── 🔲 src/.hdiconfig
```
