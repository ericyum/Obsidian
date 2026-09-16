# 03 - 백엔드: cap-node/ott (OTT 비즈니스 서비스)

> **네임스페이스:** `com.cap.ott`  
> **패키지명:** `@com/cap-ott`  
> **포트:** 8083 (로컬 개발)  
> **MTA ID:** `com-cap-ott`

---

## 1. 이 모듈이 하는 일

OTT(Over-The-Top) 스트리밍 플랫폼의 비즈니스 로직을 담당하는 **도메인 서비스**다. 넷플릭스나 웨이브 같은 스트리밍 서비스를 운영한다고 상상하면 된다:

- 📺 **콘텐츠 관리**: 영화·시리즈·애니메이션 등 콘텐츠 카탈로그 (장르·태그·출연진·리뷰)
- 👤 **사용자/구독 관리**: 사용자 마스터 + 구독 플랜(BASIC/STANDARD/PREMIUM) + 구독 상태
- 📊 **추세 분석** (TrendAnalysis): 구독자 그룹별(연령·성별·플랜) 월간 추세 집계 + 이상 탐지
- 🚨 **이상 알림** (AnomalyAlert): 이탈 급증·시청 급감 시 자동 알림 생성
- 💰 **정산 관리**: CP(콘텐츠 제공자)별 시청 시간 기반 수익 배분
- 📝 **자유 게시판** (FreeBoard): 사용자 게시글·댓글 CRUD

---

## 2. DB 모델 (db/cds/) - 4개 파일, 20개 엔티티

### Core-model.cds - 핵심 비즈니스 엔티티 (9개)

#### Users - 사용자 마스터
```
Entity: Users (com.cap.ott.Users)
key user_id            : String(10)     // U0001, U0002, ...
    user_name          : String(100)
    subscription_plan  : String(20)     // BASIC / STANDARD / PREMIUM
    subscription_status: String(20)     // active / cancelled / suspended
    subscription_start_date : Date
    subscription_end_date   : Date
    age_group          : String(10)     // 10s / 20s / 30s / 40s / 50s
    gender             : String(1)      // M / F
    plan               : Association → SubscriptionPlans
```

**초기 데이터 (Users.csv):** 10명의 사용자
| ID | 이름 | 플랜 | 상태 | 연령대 | 성별 |
|---|---|---|---|---|---|
| U0001 | Kim Minjun | PREMIUM | active | 30s | M |
| U0002 | Lee Seoyeon | BASIC | active | 20s | F |
| U0003 | Park Jiho | PREMIUM | cancelled | 40s | M |
| U0007 | Yoon Junseo | STANDARD | suspended | 50s | M |

#### Contents - 콘텐츠 카탈로그
```
Entity: Contents
key content_id  : String(10)    // C0001~C0010
    title       : String(200)
    content_type: String(20)    // MOVIE / SERIES / FEATURE / ANIMATION
    avg_rating  : Decimal(2,1)  // 0.0 ~ 5.0
    rating_count: Integer
    partner     : Association → Partners (정산용 CP 연결)
```

**초기 데이터 (Contents.csv):** 10개 콘텐츠
| ID | 제목 | 장르 | 유형 | 평점 | CP |
|---|---|---|---|---|---|
| C0001 | The Last Signal | Sci-Fi | MOVIE | 4.3 | P0001 |
| C0002 | Moonlight Garden | Historical/Romance | SERIES | 4.7 | P0002 |
| C0005 | Neon Seoul | Sci-Fi/Noir | SERIES | 4.5 | P0001 |

#### Genre / Tag - 장르·태그 마스터
```
Genre: genre_id, name
Tag: tag_id, name

N:M 매핑:
  ContentGenre: content ↔ genre
  ContentTag: content ↔ tag
```

**중요:** 장르와 태그는 N:M 관계다. 하나의 콘텐츠는 여러 장르(예: Sci-Fi + Noir)를 가질 수 있고, 하나의 장르는 여러 콘텐츠에 속할 수 있다.

#### Review - 리뷰·평점
```
Entity: Review
key review_id  : String(10)
    content    : Association → Contents
    user       : Association → Users
    rating     : Decimal(2,1)   // 0.0 ~ 5.0
    review_text: String(1000)
    createdAt  : DateTime
    modifiedAt : DateTime
```

#### ContentCast - 출연진
```
Entity: ContentCast
    name           : 배우/감독 이름
    role           : 주연, 조연, 감독, 성우
    character_name : 극중 이름
```

#### ViewingHistory - 시청 기록
```
Entity: ViewingHistory
key history_id           : String(10)     // H00001~H00020
    user                 : Association → Users
    content              : Association → Contents
    watch_datetime       : DateTime
    watch_duration_seconds: Integer       // 정산 배분 기준
    completion_percentage : Decimal(5,2)  // 0.00 ~ 100.00
    device_type          : String(20)     // SmartTV / Smartphone / Tablet / PC
```

**초기 데이터 (ViewingHistory.csv):** 20개 시청 기록 (2026년 3~4월)
```
H00001: U0001(Kim Minjun) → C0005(Neon Seoul), 3300초, SmartTV
H00002: U0001(Kim Minjun) → C0001(The Last Signal), 8520초, Smartphone
...
```

**중요:** 이 테이블은 `managed`를 적용하지 않았다. 로그성 대량 데이터이기 때문에 관리 필드(createdAt 등)가 불필요하다는 설계 판단이다.

---

### MypageAdmin-model.cds - 구독·정산 (4개)

#### SubscriptionPlans - 구독 플랜 카탈로그
```
Entity: SubscriptionPlans
key code      : BASIC / STANDARD / PREMIUM
    name      : 베이직 / 스탠다드 / 프리미엄
    price     : 월 구독료(원) → 9900, 13900, 17900
    maxStreams: 동시 시청 수 → 1, 2, 4
```

#### Partners - 콘텐츠 제공자(CP)
```
Entity: Partners
key partner_id : P0001, P0002, P0003
    name       : Studio Alpha, Studio Beta, Studio Gamma
    rsRate     : 분배율 → 0.70, 0.60, 0.65 (70%, 60%, 65%)
```

#### SettlementRuns + Settlements - 정산 헤더·디테일
```
SettlementRuns (월 정산 헤더):
    month             : '2026-03'
    totalRevenue      : 월 매출 스냅샷
    totalWatchSeconds : 월 전체 시청(초)

Settlements (파트너별 정산 디테일):
    run                 : → SettlementRuns
    partner             : → Partners
    partnerWatchSeconds : 해당 CP 시청(초)
    rsRate              : 분배율 스냅샷 (원본이 바뀌어도 유지)
    sharedRevenue       : 배분 매출
    settlementAmount    : 최종 지급액
```

**정산 계산 공식:**
```
totalRevenue = Σ(active 사용자 × 플랜 가격)
partnerShare = totalRevenue × (partnerWatchSeconds / totalWatchSeconds)
sharedRevenue = partnerShare × rsRate
settlementAmount = sharedRevenue (최종 지급액)
```

**멱등성 설계:** `calculateSettlement` 액션 실행 시, 해당 월의 기존 SettlementRuns를 먼저 삭제한다. 재실행해도 항상 같은 결과가 나온다.

---

### Trend-model.cds - 추세 분석 (4개)

#### SubscriberGroup - 구독자 그룹 정의
```
Entity: SubscriberGroup
    group_type : AGE / GENDER / PLAN
    group_name : 연령 20대 / 여성 / 프리미엄
```

#### GroupTrend - 그룹별 월간 추세
```
Entity: GroupTrend
key trend_id         : UUID
    group_id         : → SubscriberGroup
    analysis_month   : '2026-03'
    total_members    : 전체 회원 수
    active_count     : 활성 구독자 수
    churn_rate       : 이탈률(%)
    avg_watch_seconds: 1인당 평균 시청시간(초)
```

#### TrendAnomaly - 이상 변화 탐지 알림
```
Entity: TrendAnomaly
    anomaly_type : CHURN_SPIKE / WATCH_DROP / WATCH_SURGE
    severity     : INFO / WARN / CRITICAL
    change_rate  : 변화율(%)
    alert_text   : 한글 알림 메시지
    is_read      : 읽음 여부
```

**이상 탐지 임계값:**
- `CHURN_SPIKE`: 이탈률 증가 ≥ 10%p → WARN, ≥ 20%p → CRITICAL
- `WATCH_DROP`: 시청시간 감소 ≥ 30% → WARN, ≥ 50% → CRITICAL
- `WATCH_SURGE`: 시청시간 증가 ≥ 40% → INFO

#### ContentGroupStat - 그룹별 콘텐츠 소비 통계
```
Entity: ContentGroupStat
    group_id, analysis_month, genre, watch_count, total_seconds
```

---

### Freeboard-model.cds - 자유 게시판 (2개)

#### Post - 게시글
```
Entity: Post
key post_id     : UUID (자동 생성)
    title       : String(200)
    content     : String(5000)
    author      : Association → Users
    view_count  : Integer (조회 시 자동 증가)
    deleted_flag: Boolean (소프트 삭제)
    comments    : Composition of Comment
```

#### Comment - 댓글
```
Entity: Comment
key comment_id : UUID
    post_id    : UUID (부모 게시글)
    content    : String(2000)
    author     : Association → Users
    post       : Association → Post (cascade 삭제)
```

---

## 3. 서비스 및 핸들러

### TrendAnalysis Service + Handler

```
@(path: 'trend/TrendAnalysis')
service TrendAnalysis {
    entity SubscriberGroupSet;
    entity GroupTrendSet;
    entity ContentGroupStatSet;

    action calculateTrends(analysisMonth: String(7))
        returns { trend_count, anomaly_count, anomalies[] };
}
```

**`calculateTrends` 액션 처리 흐름:**

```
1. analysisMonth 파라미터 검증 (YYYY-MM 형식)
2. SubscriberGroup 목록 조회
3. 전월 계산 (getPreviousMonth)
4. 기존 동월 데이터 삭제 (멱등성)
5. 각 그룹별 반복:
   5-1. 그룹 타입에 따라 사용자 필터링
        AGE_20S → age_group = '20s'
        GENDER_F → gender = 'F'
        PLAN_BASIC → subscription_plan = 'BASIC'
   5-2. 활성 구독자 수, 이탈률 계산
   5-3. ViewingHistory에서 시청 시간 집계
   5-4. GroupTrend INSERT
   5-5. ContentGenre → Genre 조인하여 장르별 소비 통계 (ContentGroupStat INSERT)
   5-6. 전월 대비 이상 탐지 (detectAnomalies)
6. 결과 반환
```

**`detectAnomalies` 함수**는 전월과 현재 월의 값을 비교하여 3가지 이상 유형을 탐지한다. `TrendAnomaly` 테이블에 자동 INSERT 된다.

### AnomalyAlert Service + Handler

```
service AnomalyAlert {
    entity TrendAnomalySet;
    action markAsRead(anomalyId: UUID) returns TrendAnomaly;
}
```

**`markAsRead`** 액션: 알림의 `is_read`를 `true`로 업데이트하고 갱신된 레코드를 반환한다.

### FreeBoard Service + Handler

```
service FreeBoard {
    entity PostSet;
    entity CommentSet;
}
```

**핸들러 비즈니스 로직:**

| 작업 | 핸들러 | 설명 |
|---|---|---|
| CREATE Post | `before` | 작성자 자동 주입 (`req.user.id`) |
| READ Post | `after` | 단건 조회 시 view_count +1 증가 |
| UPDATE Post | `before` | 작성자 본인 확인 (타인 수정 불가) |
| DELETE Post | `before` + `on` | 작성자 확인 → 소프트 삭제 (deleted_flag = true) |
| CREATE Comment | `before` | 작성자 자동 주입 |
| UPDATE Comment | `before` | 작성자 본인 확인 |
| DELETE Comment | `before` | 작성자 확인 → 물리 삭제 |

**작성자 확인 로직 (`verifyPostAuthor`):**
```typescript
const [post] = await db.run(
    SELECT.from('com.cap.ott.Post').where({ post_id: recordId })
);
if (post.author_user_id !== userId) {
    req.error(403, '본인이 작성한 게시글만 수정/삭제할 수 있습니다.');
}
```

---

## 4. 초기 데이터 (db/data/ CSV 파일들)

### Contents.csv (10개 콘텐츠)
| ID | 제목 | 장르 | 유형 | 평점 | 리뷰수 | CP |
|---|---|---|---|---|---|---|
| C0001 | The Last Signal | Sci-Fi | MOVIE | 4.3 | 1250 | P0001 |
| C0002 | Moonlight Garden | Historical/Romance | SERIES | 4.7 | 3200 | P0002 |
| C0003 | Street Fighters: Underground | Action | SERIES | 3.9 | 890 | P0003 |
| C0004 | Docu: Secrets of the Deep | Documentary | FEATURE | 4.8 | 560 | P0001 |
| C0005 | Neon Seoul | Sci-Fi/Noir | SERIES | 4.5 | 2100 | P0001 |
| C0006 | Comedy Kitchen S4 | Variety | SERIES | 4.2 | 4500 | P0002 |
| C0007 | Guardians Reborn: New Era | Animation | ANIMATION | 4.1 | 780 | P0003 |
| C0008 | The Investor | Drama/True Story | MOVIE | 4.4 | 1600 | P0001 |
| C0009 | Love Reset | Romance/Comedy | MOVIE | 4.0 | 980 | P0002 |
| C0010 | War Zone: Eastern Front | War/History | SERIES | 4.6 | 2700 | P0003 |

### Partners.csv (3개 CP)
| ID | 이름 | 분배율 |
|---|---|---|
| P0001 | Studio Alpha | 70% |
| P0002 | Studio Beta | 60% |
| P0003 | Studio Gamma | 65% |

### SubscriptionPlans.csv (3개 플랜)
| 코드 | 이름 | 가격 | 동시시청 |
|---|---|---|---|
| BASIC | 베이직 | 9,900원 | 1 |
| STANDARD | 스탠다드 | 13,900원 | 2 |
| PREMIUM | 프리미엄 | 17,900원 | 4 |

### ViewingHistory.csv (20개 시청 기록)
2026년 3~4월의 시청 데이터. 사용자 10명이 다양한 디바이스(SmartTV, Smartphone, Tablet, PC)로 콘텐츠를 시청한 기록이다.

**재미있는 통계:**
- U0001(Kim Minjun, 30s, M)은 C0005(Neon Seoul)과 C0001(The Last Signal)을 시청
- C0001(The Last Signal)을 세 명(U0001, U0003, U0009)이 시청 → 가장 인기 있는 콘텐츠
- SmartTV가 가장 많이 사용된 디바이스 (8회)

---

## 5. 개발 이력 (History 주석)

CDS 파일들의 History 주석에 따르면:

- **2026-06-11**: Module 3(추세 분석), Module 6(게시판) 최초 작성
- **2026-06-12**: 통합 회의에서 모델 통합. 주요 쟁점:
  - 쟁점#2: 장르·태그 N:M 정규화
  - 쟁점#3: 구독 필드 + age_group/gender 필드 통합
  - 쟁점#4: user → Association to Users로 통일
  - 쟁점#6: 마이크로서비스 분리 vs 통합 결정
- **2026-06-15**: 구독 플랜 테이블 참조(plan) 추가, 정산 SettlementRuns/Settlements 분리, ContentGenre/ContentTag managed 미적용, ViewingHistory managed 미적용

이 프로젝트는 매우 최근(2026년 6월)에 활발히 개발된 신규 프로젝트임을 알 수 있다.
