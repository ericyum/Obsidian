# OTT 통합 Entity 설계 (기능 1,2 + 3 + 4,5 + 6)

> 작성일: 2026-06-11
> 대상: CAP 온보딩 OTT 프로젝트 전체
> 범위: 기능1·2(안효서) + 기능3(염정운) + 기능4·5(나영일) + 기능6(염정운)

---

## 1. 통합 개요

4개 기능의 엔티티를 하나로 병합했습니다. 충돌나는 부분은 **통일안을 제시**하고, 원래 문서와 다른 점은 하단에 **회의 필요 쟁점**으로 정리해뒀습니다.

| 항목 | 통일안 | 비고 |
|---|---|---|
| **Namespace** | `com.cap.core` | 기존 템플릿/배포 기준. `com.cap.ott`로 변경 가능성은 쟁점 #1 |
| **DB 타입** | SQLite(로컬) / HANA(BTP) | |
| **장르·태그** | N:M 정규화 (`Genre`+`ContentGenre`, `Tag`+`ContentTag`) | 기능1/2 방식采. 기능4/5 원본은 String 평탄화 |
| **엔티티 총수** | 18개 | 공통 9개 + 기능3 4개 + 기능4/5 3개 + 기능6 2개 |

---

## 2. 통합 Entity 목록

### 공통 — 전 기능 사용

| 엔티티 | 주요 역할 | 비고 |
|---|---|---|
| `Users` | 회원 마스터 · 로그인 세션 · 리뷰 작성자 · 구독 상태 · 분석 그룹 필드 | 필드가 많아짐. 쟁점 #3 |
| `Contents` | 콘텐츠 카탈로그 · 장르·태그·리뷰·시청기록 연결 | `partner` FK 추가(정산용) |
| `Genre` | 장르 마스터 | N:M 방식 |
| `Tag` | 태그 마스터 | N:M 방식 |
| `ContentGenre` | 콘텐츠—장르 N:M 연결 | |
| `ContentTag` | 콘텐츠—태그 N:M 연결 | |
| `Review` | 상세 페이지 평점 + 리뷰 | |
| `ContentCast` | 출연진 정보 | 선택. 없어도 무방 |
| `ViewingHistory` | 시청 기록 · 정산 배분 기준 | 팀 공용 |

### 기능 3 전용 (구독자 추세 분석)

| 엔티티 | 주요 역할 | 비고 |
|---|---|---|
| `SubscriberGroup` | 분석 기준 그룹 마스터 (연령대·성별·플랜) | |
| `GroupTrend` | 그룹-월 단위 구독·시청 지표 스냅샷 | `calculateTrends` 액션으로 생성 |
| `TrendAnomaly` | 이상 탐지 알림 | 임계값 초과 시 자동 생성 |
| `ContentGroupStat` | 그룹-장르 소비 통계 | `calculateTrends`에서 함께 생성 |

### 기능 4,5 전용 (마이페이지·관리자·정산)

| 엔티티 | 주요 역할 | 비고 |
|---|---|---|
| `SubscriptionPlans` | 구독 플랜 카탈로그 · 가격 · 동시시청 수 | |
| `Partners` | 콘텐츠 제공자(CP) · 분배율(rsRate) | |
| `Settlements` | 월별·파트너별 정산 결과 스냅샷 | 실행 시 생성. 시드 없음 |

### 기능 6 전용 (자유 게시판)

| 엔티티 | 주요 역할 | 비고 |
|---|---|---|
| `Post` | 게시글 | 소프트 삭제(`deleted_flag`) |
| `Comment` | 댓글 | `Post`와 Composition |

---

## 3. 통합 CDS 정의

```cds
namespace com.cap.core; // 쟁점 #1

// ============================================
// 공통 마스터
// ============================================

entity Users {
    key user_id               : String(10);
        user_name             : String(100);
        // --- 기능4/5: 구독 필드 ---
        subscription_plan     : String(20);   // BASIC / STANDARD / PREMIUM
        subscription_status   : String(20);   // active / cancelled / suspended
        subscription_start_date : Date;
        subscription_end_date : Date;
        // --- 기능3: 분석용 필드 ---
        age_group             : String(10);   // 10s / 20s / 30s / 40s / 50s
        gender                : String(1);    // M / F
}

entity Genre {
    key genre_id : String(10);
        name     : String(50);
    contents : Association to many ContentGenre on contents.genre = $self;
}

entity Tag {
    key tag_id : String(10);
        name   : String(50);
    contents : Association to many ContentTag on contents.tag = $self;
}

// ============================================
// 콘텐츠 중심
// ============================================

entity Contents {
    key content_id   : String(10);
        title        : String(200);
        content_type : String(20);   // MOVIE / SERIES / FEATURE / ANIMATION
        avg_rating   : Decimal(2,1); // 0.0 ~ 5.0
        rating_count : Integer;

        // 기능4/5 확장: 정산용 CP 연결 (nullable)
        partner : Association to Partners;

        // 기능1/2: N:M 장르·태그
        genres  : Composition of many ContentGenre on genres.content = $self;
        tags    : Composition of many ContentTag   on tags.content   = $self;
        reviews : Composition of many Review       on reviews.content = $self;
        casts   : Composition of many ContentCast  on casts.content   = $self;

        // 기능4/5: 시청 기록 연결
        histories : Association to many ViewingHistory on histories.content = $self;
}

entity ContentGenre {
    key content : Association to Contents;
    key genre   : Association to Genre;
}

entity ContentTag {
    key content : Association to Contents;
    key tag     : Association to Tag;
}

entity Review {
    key review_id   : String(10);
        content     : Association to Contents;
        user        : Association to Users;   // 기능1/2 원본은 user_id:String. 통일안은 Association
        rating      : Decimal(2,1);
        review_text : String(1000);
        createdAt   : DateTime;
        modifiedAt  : DateTime;
}

entity ContentCast {
    key cast_id        : String(10);
        content        : Association to Contents;
        name           : String(100);
        role           : String(50);   // 주연, 조연, 감독, 성우
        character_name : String(100);
}

// ============================================
// 기능3: 구독자 추세 분석
// ============================================

entity SubscriberGroup {
    key group_id   : String(30);   // AGE_20S, GENDER_F, PLAN_PREMIUM
        group_type : String(30);   // AGE / GENDER / PLAN
        group_name : String(100);  // 20대 / 여성 / 프리미엄
}

entity GroupTrend {
    key ID                : UUID;
        group             : Association to SubscriberGroup;
        analysis_month    : String(7);   // 'YYYY-MM'
        total_members     : Integer;
        active_count      : Integer;
        churn_rate        : Decimal(5,2);
        avg_watch_seconds : Integer64;
}

entity TrendAnomaly {
    key ID             : UUID;
        group          : Association to SubscriberGroup;
        anomaly_month  : String(7);   // 'YYYY-MM'
        anomaly_type   : String(30);  // CHURN_SPIKE / WATCH_DROP / WATCH_SURGE
        severity       : String(10);  // INFO / WARN / CRITICAL
        change_rate    : Decimal(8,2);
        alert_text     : String(500);
        is_read        : Boolean default false;
}

entity ContentGroupStat {
    key ID             : UUID;
        group          : Association to SubscriberGroup;
        analysis_month : String(7);
        genre          : String(100);
        watch_count    : Integer;
        total_seconds  : Integer64;
}

// ============================================
// 기능4/5: 시청·플랜·정산
// ============================================

entity ViewingHistory {
    key history_id           : String(10);
        user                   : Association to Users;
        content                : Association to Contents;
        watch_datetime         : DateTime;
        watch_duration_seconds : Integer;
        completion_percentage  : Decimal(5,2);
        device_type            : String(20);  // SmartTV / Smartphone / Tablet / PC
}

entity SubscriptionPlans {
    key code       : String(20);   // BASIC / STANDARD / PREMIUM
        name       : String(100);  // 베이직 / 스탠다드 / 프리미엄
        price      : Integer;      // 월 구독료(원)
        maxStreams : Integer;      // 동시 시청 수
}

entity Partners {
    key partner_id : String(10);
        name         : String(100);
        rsRate       : Decimal(5,4);  // 0.0000 ~ 1.0000 (예: 0.70)
        contents     : Association to many Contents on contents.partner = $self;
}

entity Settlements {
    key ID                  : UUID;
        month               : String(7);   // 'YYYY-MM'
        partner             : Association to Partners;
        totalRevenue        : Integer;     // 월 매출 스냅샷
        totalWatchSeconds   : Integer64;
        partnerWatchSeconds : Integer64;
        rsRate              : Decimal(5,4);  // 분배율 스냅샷
        sharedRevenue       : Integer;     // 배분된 매출
        settlementAmount    : Integer;     // 최종 지급액
}

// ============================================
// 기능6: 자유 게시판
// ============================================

entity Post {
    key post_id      : UUID;
        title        : String(200)  not null;
        content      : String(5000) not null;
        author_id    : String(255)  not null;  // FK → Users.user_id (String FK)
        view_count   : Integer      default 0;
        deleted_flag : Boolean      default false;
        comments     : Composition of many Comment on comments.post = $self;
}

entity Comment {
    key comment_id : UUID;
        post       : Association to Post;
        content    : String(2000) not null;
        author_id  : String(255)  not null;  // FK → Users.user_id (String FK)
}
```

---

## 4. 테이블 간 관계도

```text
Users (1) ──────< (N) Review              // 작성자
       │
       │ (1) ────< (N) ViewingHistory (N) >──── (1) Contents ──── (N) ContentGenre (N) ──── (1) Genre
       │                              │                           │
       │                              │                           └── (N) ContentTag   (N) ──── (1) Tag
       │                              │                           │
       │                              │                           └── (N) ContentCast
       │                              │                           │
       │                              └── (1) Partners (1) ────────┘
       │                                  │
       │                                  └── (1) < (N) Settlements
       │
       ├── (N:1) SubscriptionPlans       // subscription_plan = code
       │
       ├── (N:1) < (N) GroupTrend
       │               │
       │               ├── (이상 감지 시) > TrendAnomaly
       │               └── > ContentGroupStat
       │
       ├── (1) ──── (N) Post (1) ──< Composition ── (N) Comment
       │
       └── (N) ──── SubscriberGroup (분석 기준 그룹)
```

---

## 5. 화면별 핵심 OData 호출

### 기능 1,2 (메인·상세)

```http
# 장르 필터링
GET /odata/v4/ContentService/Contents
  ?$expand=genres($expand=genre)
  &$filter=genres/any(g: g/genre/genre_id eq 'G0001')

# 태그 필터링
GET /odata/v4/ContentService/Contents
  ?$expand=tags($expand=tag)
  &$filter=tags/any(t: t/tag/tag_id eq 'T0001')

# 검색
GET /odata/v4/ContentService/Contents
  ?$filter=contains(title,'검색어')

# 상세 (장르 + 태그 + 리뷰 + 출연진)
GET /odata/v4/ContentService/Contents('C0001')
  ?$expand=genres($expand=genre),tags($expand=tag),reviews($expand=user),casts
```

### 기능 3 (구독자 추세 분석)

```http
# 추세 계산 실행
POST /odata/v4/ott/TrendAnalysis/calculateTrends

# 그룹별 추세 조회
GET /odata/v4/ott/TrendAnalysis/GroupTrendSet
  ?$filter=analysis_month eq '2026-03'
  &$expand=group

# 이상 탐지 알림 조회
GET /odata/v4/ott/TrendAnalysis/TrendAnomalySet
  ?$filter=is_read eq false
  &$orderby=severity desc

# 그룹-장르 소비 통계
GET /odata/v4/ott/TrendAnalysis/ContentGroupStatSet
  ?$filter=group/group_id eq 'AGE_20S' and analysis_month eq '2026-03'
```

### 기능 4 (마이페이지)

```http
# 내 멤버십 조회
GET /odata/v4/ott/Mypage/UserSet('U0001')?$expand=plan

# 플랜 목록
GET /odata/v4/ott/Mypage/PlanSet

# 플랜 변경
POST /odata/v4/ott/Mypage/changePlan
{"user_id":"U0001","plan":"PREMIUM"}

# 해지 / 재개
POST /odata/v4/ott/Mypage/cancel
POST /odata/v4/ott/Mypage/resume
```

### 기능 5 (관리자·정산)

```http
# 정산 실행
POST /odata/v4/ott/Admin/runSettlement
{"month":"2026-03"}

# 정산 결과 조회
GET /odata/v4/ott/Admin/SettlementSet
  ?$filter=month eq '2026-03'
  &$expand=partner($select=name)
```

### 기능 6 (자유 게시판)

```http
# 게시글 목록 (소프트 삭제 제외, 최신순)
GET /odata/v4/board/FreeBoard/PostSet
  ?$filter=deleted_flag eq false
  &$orderby=createdAt desc

# 게시글 상세 (조회수 포함)
GET /odata/v4/board/FreeBoard/PostSet({post_id})?$expand=comments

# 게시글 작성
POST /odata/v4/board/FreeBoard/PostSet
{"title":"...","content":"..."}

# 댓글 작성
POST /odata/v4/board/Comment/CommentSet
{"post_id":"...","content":"..."}
```

---

## 6. CSV 시드 데이터 파일 목록

```
db/data/com.cap.core.Genre.csv
db/data/com.cap.core.Tag.csv
db/data/com.cap.core.Contents.csv
db/data/com.cap.core.ContentGenre.csv
db/data/com.cap.core.ContentTag.csv
db/data/com.cap.core.Review.csv
db/data/com.cap.core.ContentCast.csv
db/data/com.cap.core.Users.csv
db/data/com.cap.core.ViewingHistory.csv
db/data/com.cap.core.SubscriptionPlans.csv
db/data/com.cap.core.Partners.csv
db/data/com.cap.core.SubscriberGroup.csv
# GroupTrend, TrendAnomaly, ContentGroupStat — 시드 없음 (액션으로 생성)
# Post, Comment — 시드 없음 (사용자 생성 데이터)
# Settlements — 시드 없음 (액션으로 생성)
```

---

# ⚠️ 회의 필요 쟁점

아래 항목은 본 문서에서 **통일안을 제시**했으나, 원래 각 기능 문서와 달랐거나 아직 확인이 필요한 부분입니다. 팀 회의에서 확정해야 합니다.

### 쟁점 #1 — Namespace
- **기능1/2 문서**: `com.cap.core` (기존 템플릿 유지)
- **기능3/4/5/6 문서**: `com.cap.ott`
- **통일안**: `com.cap.core` (배포 식별자 `com-cap-ott-core`와 이미 맞춰둔 상태)
- **질문**: 그대로 `com.cap.core`를 쓸 것인가, `com.cap.ott`로 변경할 것인가?
- 결론: ott로 통일

### 쟁점 #2 — Contents의 장르·태그 표현 방식
- **기능1/2 문서**: `Genre`/`Tag` 마스터 + `ContentGenre`/`ContentTag` N:M 연결 테이블
- **기능4/5 원본**: `Contents.genre` = `"액션/드라마"` (String, `/` 구분), `Contents.tags` = `"SF|19금"` (String, `|` 구분)
- **통일안**: N:M 정규화 방식采
- **질문**: 기능4/5의 CSV 파일들도 N:M 방식으로 재구성할 것인가? 아니면 `genre`/`tags` String 필드를 병행 유지할 것인가?
- 결론: N:M 연결로 통일

### 쟁점 #3 — Users 필드 통합 범위
- **기능1/2 원본**: `user_id`, `user_name`만 존재
- **기능3 원본**: `age_group`, `gender` 추가 필요
- **기능4/5 원본**: `subscription_plan`, `subscription_status`, `subscription_start_date`, `subscription_end_date` 추가
- **통일안**: 전부 포함 (현재 8개 필드)
- **질문**: Users가 점점 비대해지는데, 메인/상세 화면에서 구독 상태나 연령대를 표시할 필요가 있는가? 없다면 슬림화 고려.
- 결론 8개 필드를 전부 포함

### 쟁점 #4 — Review.user / Post.author_id / Comment.author_id 타입 통일
- **Review**: `user : Association to Users` (통일안)
- **Post/Comment**: `author_id : String(255)` (기능6 원본, 단순 FK 문자열)
- **통일안**: Post/Comment도 `Association to Users`로 변경하거나, 전부 `String FK`로 통일
- **질문**: 작성자 필드 방식을 하나로 맞출 것인가? Association vs String FK?
- 결론: Association to Users로 통일

### 쟁점 #5 — 기존 템플릿 엔티티 처리
- 현재 `db/cds/`에 `CodeHeader`, `CodeItem`, `Favorite`, `Menu`, `Role`, `RoleGroup` 등 OTT와 무관한 엔티티가 남아있음.
- **질문**: 완전 삭제할 것인가? 아니면 주석 처리만 해둘 것인가?
- 결론: 아무것도 하지 않고 놔둠

### 쟁점 #6 — Service 분리 기준
- **기능1/2**: `ContentService` (단일)
- **기능3**: `TrendAnalysisService`
- **기능4/5**: `MypageService`, `AdminService`
- **기능6**: `FreeBoardService`, `CommentService`
- **질문**: 총 6개로 나눌 것인가? 아니면 `OttService` 하나로 통합할 것인가? 또는 업무별로 2~3개로 묶을 것인가?
- 결론: 마이크로 서비스 아키텍처 구현을 위해 나눠진 상태로 놔둠.
