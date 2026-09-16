# OTT Module 3·6 — 서비스 아키텍처 및 커스텀 로직 상세

> **Namespace**: `com.cap.ott`  
> **작성일**: 2026-06-15  
> **담당**: 염정운 (Module 3 구독자 추세 분석 + Module 6 자유 게시판)  
> **검증**: `cds build` 0 errors / `cds watch` 서버 구동 & API 테스트 완료

---

## 1. 전체 폴더 구조 (아키텍처)

```
cap-node/ott/
│
├── index.cds                          ← 루트 진입점 (db + srv 통합)
├── package.json                       ← npm scripts (watch, build)
├── .cdsrc.json                        ← 개발환경 SQLite·mock auth 설정
│
├── db/                                ← ── DB 레이어 ──
│   ├── index.cds                      ← 4개 모델 파일 통합 import
│   ├── cds/
│   │   ├── Core-model.cds             ← 공통 마스터 + 콘텐츠 중심 (9 entities)
│   │   ├── Trend-model.cds            ← 구독자 추세 분석 (4 entities)
│   │   ├── MypageAdmin-model.cds      ← 마이페이지·관리자·정산 (4 entities)
│   │   └── Freeboard-model.cds        ← 자유 게시판 (2 entities)
│   ├── data/                          ← CSV 시드 데이터
│   │   ├── Users.csv                  ← 10명 (age_group, gender, 구독정보 포함)
│   │   ├── Contents.csv               ← 10개 콘텐츠 (partner FK 포함)
│   │   ├── SubscriptionPlans.csv      ← BASIC/STANDARD/PREMIUM
│   │   ├── Partners.csv               ← 3개 CP (rsRate 분배율)
│   │   └── ViewingHistory.csv         ← 20건 시청 이력
│   └── src/.hdiconfig
│
├── srv/                               ← ── 서비스 레이어 ──
│   ├── index.cds                      ← 3개 서비스 CDS 통합 import
│   ├── cds/
│   │   ├── trend/
│   │   │   ├── TrendAnalysis-service.cds   ← 구독자 추세 분석 서비스
│   │   │   └── AnomalyAlert-service.cds    ← 이상 탐지 알림 서비스
│   │   └── board/
│   │       └── FreeBoard-service.cds       ← 자유 게시판 서비스
│   └── src/feature/
│       ├── trend/
│       │   ├── TrendAnalysis.handler.ts    ← calculateTrends 액션 로직
│       │   └── AnomalyAlert.handler.ts     ← markAsRead 액션 로직
│       └── board/
│           └── FreeBoard.handler.ts        ← Post + Comment CRUD 통합
│
└── gen/                               ← 빌드 출력 (cds build)
```

---

## 2. Module 연결 흐름

```
[index.cds]
    using from './db'  ──→  [db/index.cds]
    │                         ├── using from './cds/Core-model.cds'      (9 entities)
    │                         ├── using from './cds/Trend-model.cds'     (4 entities)
    │                         ├── using from './cds/MypageAdmin-model.cds' (4 entities)
    │                         └── using from './cds/Freeboard-model.cds' (2 entities)
    │
    using from './srv' ──→  [srv/index.cds]
                              ├── using from './cds/trend/TrendAnalysis-service'
                              ├── using from './cds/trend/AnomalyAlert-service'
                              └── using from './cds/board/FreeBoard-service'
```

---

## 3. DB 모델 — 18개 엔티티 (4개 파일)

### 3.1 Core-model.cds (9 entities) — 공통 마스터 + 콘텐츠 중심

| # | Entity | Key | 설명 |
|:--:|--------|-----|------|
| 1 | `Users` | user_id(10) | 회원 마스터, 8개 필드 통합 (구독 + 분석) |
| 2 | `Contents` | content_id(10) | 콘텐츠 카탈로그, partner FK, N:M 장르·태그 |
| 3 | `Genre` | genre_id(10) | 장르 마스터 (N:M 연결) |
| 4 | `Tag` | tag_id(10) | 태그 마스터 (N:M 연결) |
| 5 | `ContentGenre` | content+genre | 콘텐츠-장르 N:M (managed 미적용) |
| 6 | `ContentTag` | content+tag | 콘텐츠-태그 N:M (managed 미적용) |
| 7 | `Review` | review_id(10) | 평점·리뷰 (user → Association to Users) |
| 8 | `ContentCast` | cast_id(10) | 출연진 정보 (선택) |
| 9 | `ViewingHistory` | history_id(10) | 시청 기록 (managed 미적용) |

### 3.2 Trend-model.cds (4 entities) — 구독자 추세 분석

| # | Entity | Key | 설명 |
|:--:|--------|-----|------|
| 10 | `SubscriberGroup` | group_id(30) | 분석 그룹 기준 (AGE/GENDER/PLAN) |
| 11 | `GroupTrend` | trend_id(UUID) | 그룹-월 추세 스냅샷 (calculateTrends 생성) |
| 12 | `TrendAnomaly` | anomaly_id(UUID) | 이상 탐지 알림 (임계값 초과 시 자동생성) |
| 13 | `ContentGroupStat` | stat_id(UUID) | 그룹-장르 소비 통계 |

### 3.3 MypageAdmin-model.cds (4 entities) — 기능4·5

| # | Entity | Key | 설명 |
|:--:|--------|-----|------|
| 14 | `SubscriptionPlans` | code(20) | 플랜 카탈로그 (BASIC/STANDARD/PREMIUM) |
| 15 | `Partners` | partner_id(10) | 콘텐츠 제공자 (rsRate 분배율) |
| 16 | `SettlementRuns` | month(7) | 정산 헤더 (월당 1행, 멱등) |
| 17 | `Settlements` | ID(UUID) | 파트너별 정산 디테일 (Composition to SettlementRuns) |

### 3.4 Freeboard-model.cds (2 entities) — 자유 게시판

| # | Entity | Key | 설명 |
|:--:|--------|-----|------|
| 18 | `Post` | post_id(UUID) | 게시글 (소프트 삭제, author → Association to Users) |
| — | `Comment` | comment_id(UUID) | 댓글 (Post와 Composition, 물리 삭제) |

---

## 4. 서비스 — 3개 (OData v4)

### 4.1 FreeBoard Service (Module 6)

- **경로**: `/odata/v4/board/FreeBoard`
- **핸들러**: `srv/src/feature/board/FreeBoard.handler.ts`
- **인증**: `@(requires: 'authenticated-user')`
- **limit**: 100 rows

| EntitySet | Projection | 설명 |
|-----------|-----------|------|
| `PostSet` | `com.cap.ott.Post` | 게시글 CRUD |
| `CommentSet` | `com.cap.ott.Comment` | 댓글 CRUD (Composition of Post) |

### 4.2 TrendAnalysis Service (Module 3)

- **경로**: `/odata/v4/trend/TrendAnalysis`
- **핸들러**: `srv/src/feature/trend/TrendAnalysis.handler.ts`
- **인증**: `@(requires: 'authenticated-user')`
- **limit**: 99999 rows

| EntitySet | Projection | 설명 |
|-----------|-----------|------|
| `SubscriberGroupSet` | `com.cap.ott.SubscriberGroup` | 분석 그룹 조회 |
| `GroupTrendSet` | `com.cap.ott.GroupTrend` | 월간 추세 조회 |
| `ContentGroupStatSet` | `com.cap.ott.ContentGroupStat` | 장르별 소비 통계 |

| Action | Input | Returns | 설명 |
|--------|-------|---------|------|
| `calculateTrends` | `analysisMonth: String(7)` | `{ trend_count, anomaly_count, anomalies[] }` | 그룹별 월간 집계 + 이상 탐지 |

### 4.3 AnomalyAlert Service (Module 3)

- **경로**: `/odata/v4/trend/AnomalyAlert`
- **핸들러**: `srv/src/feature/trend/AnomalyAlert.handler.ts`
- **인증**: `@(requires: 'authenticated-user')`
- **limit**: 99999 rows

| EntitySet | Projection | 설명 |
|-----------|-----------|------|
| `TrendAnomalySet` | `com.cap.ott.TrendAnomaly` | 이상 알림 조회 |

| Action | Input | Returns | 설명 |
|--------|-------|---------|------|
| `markAsRead` | `anomalyId: UUID` | TrendAnomaly 필드 | 알림 읽음 처리 (`is_read=true`) |

---

## 5. 커스텀 로직 — 3개 TypeScript Handler

### 5.1 FreeBoard.handler.ts (146줄)

Post와 Comment CRUD를 하나의 클래스로 통합.

```
Post — CREATE  (before)  → req.user.id 주입 → author_user_id 자동 설정
Post — READ    (after)   → 단건 조회 시 view_count += 1
Post — UPDATE  (before)  → verifyPostAuthor() → 본인만 수정 (403)
Post — DELETE  (before)  → verifyPostAuthor() → 본인만 삭제 (403)
Post — DELETE  (on)      → 물리 삭제 대신 deleted_flag = true (소프트 삭제)

Comment — CREATE (before) → req.user.id 주입 → author_user_id 자동 설정
Comment — UPDATE (before) → verifyCommentAuthor() → 본인만 수정 (403)
Comment — DELETE (before) → verifyCommentAuthor() → 본인만 삭제 (403)
Comment — DELETE (default)→ 물리 삭제 (CASCADE)
```

**주요 예외 처리**:
| 코드 | 조건 |
|:---:|------|
| `401` | 미인증 사용자 (req.user.id 없음) |
| `403` | 타인의 게시글/댓글 수정·삭제 시도 |
| `404` | 존재하지 않는 게시글/댓글 |

---

### 5.2 TrendAnalysis.handler.ts (243줄)

`calculateTrends` 액션 — 7단계 알고리즘.

```
Step 1. SubscriberGroup 목록 조회
Step 2. analysisMonth의 전월 계산 (getPreviousMonth)
Step 3. 해당 월 기존 GroupTrend, ContentGroupStat DELETE (멱등성)
Step 4. 그룹별 반복 ─┐
  4-1. group_type에 따라 Users 필터링 (AGE→age_group, GENDER→gender, PLAN→subscription_plan)
  4-2. 활성 구독자 수, 이탈률(churn_rate) 계산
  4-3. ViewingHistory에서 시청시간 집계 (SUM, COUNT DISTINCT)
  4-4. GroupTrend INSERT
  4-5. Contents+ContentGenre JOIN → 장르별 ContentGroupStat INSERT
  4-6. 전월 GroupTrend와 비교 → detectAnomalies
      ├── CHURN_SPIKE: churn_rate +10%p↑(WARN) / +20%p↑(CRITICAL)
      ├── WATCH_DROP:  avg_watch -30%↓(WARN) / -50%↓(CRITICAL)
      └── WATCH_SURGE: avg_watch +40%↑(INFO)
Step 5. 결과 반환 { trend_count, anomaly_count, anomalies[] }
```

**이상 탐지 임계값**:

| 유형 | WARN | CRITICAL |
|------|------|----------|
| CHURN_SPIKE | +10%p↑ | +20%p↑ |
| WATCH_DROP | -30%↓ | -50%↓ |
| WATCH_SURGE | +40%↑ (INFO만) | — |

---

### 5.3 AnomalyAlert.handler.ts (55줄)

`markAsRead` 액션 — 단순 상태 변경.

```
1. anomalyId 파라미터 검증 (400)
2. TrendAnomaly 존재 여부 확인 (404)
3. is_read = true 로 UPDATE
4. 갱신된 레코드 SELECT → 반환
```

**주요 예외 처리**:
| 코드 | 조건 |
|:---:|------|
| `400` | anomalyId 미입력 |
| `404` | 존재하지 않는 알림 ID |

---

## 6. OData API 명세

### 6.1 자유 게시판 (FreeBoard)

```
# 게시글 목록 (소프트 삭제 제외)
GET /odata/v4/board/FreeBoard/PostSet?$filter=deleted_flag eq false&$orderby=createdAt desc

# 게시글 상세 (+ 댓글 확장)
GET /odata/v4/board/FreeBoard/PostSet({post_id})?$expand=comments

# 게시글 작성 (author_user_id 자동주입)
POST /odata/v4/board/FreeBoard/PostSet
{"title":"...","content":"..."}

# 게시글 수정 (본인만)
PATCH /odata/v4/board/FreeBoard/PostSet({post_id})
{"title":"..."}

# 게시글 삭제 (소프트 삭제, 본인만)
DELETE /odata/v4/board/FreeBoard/PostSet({post_id})

# 댓글 작성
POST /odata/v4/board/FreeBoard/CommentSet
{"post_id":"...","content":"..."}
```

### 6.2 구독자 추세 분석 (TrendAnalysis)

```
# 분석 그룹 목록
GET /odata/v4/trend/TrendAnalysis/SubscriberGroupSet

# 그룹별 월간 추세
GET /odata/v4/trend/TrendAnalysis/GroupTrendSet?$filter=analysis_month eq '2026-03'&$expand=group

# 장르별 소비 통계
GET /odata/v4/trend/TrendAnalysis/ContentGroupStatSet?$filter=group/group_id eq 'AGE_20S'

# 추세 계산 실행
POST /odata/v4/trend/TrendAnalysis/calculateTrends
{"analysisMonth":"2026-03"}

# 이상 알림 목록 (미확인만)
GET /odata/v4/trend/AnomalyAlert/TrendAnomalySet?$filter=is_read eq false&$orderby=severity desc

# 알림 읽음 처리
POST /odata/v4/trend/AnomalyAlert/markAsRead
{"anomalyId":"..."}
```

---

## 7. 설계 결정 사항

| # | 결정 | 근거 |
|:--:|------|------|
| 1 | 서비스 CDS에서 `namespace` 제거, fully qualified 사용 | `namespace com.cap.ott;`가 DB 모델과 격리되어 엔티티 미발견 버그 발생 |
| 2 | `returns` 액션 반환 타입을 inline struct로 정의 | fully qualified entity 참조(`com.cap.ott.TrendAnomaly`)가 `returns` 위치에서 파싱되지 않음 |
| 3 | Post-Comment 단일 서비스(FreeBoard)로 통합 | Composition 관계 → 하나의 aggregate, OData 네비게이션 자연스러움 |
| 4 | TrendAnalysis ↔ AnomalyAlert 서비스 분리 | 분석 대시보드 vs 알림 센터 — 책임 분리 (마이크로서비스) |
| 5 | 18개 엔티티 → 4개 모델 파일로 통합 | 모듈 단위 그룹핑, `using from` 간소화 |
| 6 | ContentGenre, ContentTag, ViewingHistory `managed` 미적용 | 단순 연결·로그성 대량 테이블에 감사 필드 불필요 |
| 7 | Post 소프트 삭제 (`deleted_flag`) | 사용자 복구 요청 대비, 데이터 보존 |
| 8 | Comment 물리 삭제 | Post 삭제 시 Composition CASCADE로 자연 소멸 |
| 9 | `calculateTrends` 멱등성 | 동일 월 재실행 시 DELETE 후 재생성 |

---

## 8. 개발 환경

| 항목 | 설정 |
|------|------|
| DB (production) | SAP HANA |
| DB (development) | SQLite in-memory (`:memory:`) |
| Auth (development) | Mocked — user1, user2 (authenticated-user) |
| Port | 8083 |
| 실행 | `npm run watch` |
| 빌드 | `npm run build` |

---

## 9. 테스트 결과 (2026-06-15)

| 테스트 | 결과 |
|--------|:--:|
| `cds build` | ✅ 0 errors |
| `cds watch` 서버 기동 | ✅ SQLite 구동 |
| Post CREATE (작성자 자동주입) | ✅ `author_user_id: "user1"` |
| Comment CREATE | ✅ `author_user_id: "user2"` |
| 타인 글 수정 → 403 | ✅ "본인이 작성한 게시글만" |
| 본인 글 수정 (view_count+1) | ✅ `view_count: 1` |
| 타인 글 삭제 → 403 | ✅ "본인이 작성한 게시글만" |
| 본인 소프트 삭제 | ✅ `deleted_flag: true` |
| OData 메타데이터 | ✅ 3개 서비스 노출 |
