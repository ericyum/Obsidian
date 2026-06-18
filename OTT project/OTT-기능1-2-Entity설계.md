# 기능 1 & 2 — 백엔드 CDS 설계

> 담당: 안효서  
> 메인: 장르/태그/검색 + 세션  
> 상세: 컨텐츠 정보 + 평점 + 리뷰

---

## 1. 설계 목표

메인 페이지에서 **장르/태그 필터**와 **검색**이 가능해야 하고, 상세 페이지에서 **리뷰/평점**을 남길 수 있어야 함.

Contents의 장르와 태그는 **여러 개** 붙을 수 있으므로 별도 마스터와 N:M 연결 테이블로 정규화함.

---

## 2. CDS Entity 목록

### Users

로그인 세션 + 리뷰 작성자.

```cds
entity Users {
    key user_id               : String(10);
        user_name             : String(100);
        subscription_plan     : String(20);   // BASIC / STANDARD / PREMIUM
        subscription_status   : String(20);   // active / cancelled / suspended
        subscription_start_date : Date;
        subscription_end_date : Date;
}
```

---

### Genre

장르 마스터. 예: 액션, 드라마, 코미디.

```cds
entity Genre {
    key genre_id : String(10);
        name     : String(50);

    // 역방향: 이 장르에 속한 컨텐츠 목록
    contents : Association to many ContentGenre on contents.genre = $self;
}
```

---

### Tag

태그 마스터. 예: SF, 19금, 명작.

```cds
entity Tag {
    key tag_id : String(10);
        name   : String(50);

    // 역방향: 이 태그가 붙은 컨텐츠 목록
    contents : Association to many ContentTag on contents.tag = $self;
}
```

---

### Contents

컨텐츠 메인. 장르/태그/리뷰/출연진을 연결함.

```cds
entity Contents {
    key content_id         : String(10);
        title              : String(200);
        content_type       : String(20);   // MOVIE / SERIES / FEATURE / ANIMATION
        avg_rating         : Decimal(2,1); // 0.0 ~ 5.0
        rating_count       : Integer;
        partner_partner_id : String(10);   // 정산용 CP 참조

        // 장르 여러 개 (N:M)
        genres : Composition of many ContentGenre on genres.content = $self;

        // 태그 여러 개 (N:M)
        tags   : Composition of many ContentTag   on tags.content   = $self;

        // 리뷰
        reviews : Composition of many Review on reviews.content = $self;

        // 출연진 (선택)
        casts   : Composition of many ContentCast on casts.content = $self;
}
```

---

### ContentGenre (N:M)

컨텐츠 — 장르 연결.

```cds
entity ContentGenre {
    key content : Association to Contents;
    key genre   : Association to Genre;
}
```

> 메인 페이지에서 `genre = 'G0001'` 필터 시 사용.

---

### ContentTag (N:M)

컨텐츠 — 태그 연결.

```cds
entity ContentTag {
    key content : Association to Contents;
    key tag     : Association to Tag;
}
```

> 메인 페이지에서 태그 필터 및 검색 시 사용.

---

### Review

상세 페이지의 평점 + 리뷰.

```cds
entity Review {
    key review_id   : String(10);
        content      : Association to Contents; // 어느 컨텐츠
        user_id      : String(10);               // 작성자 (Users.user_id)
        rating       : Decimal(2,1);             // 0.0 ~ 5.0
        review_text  : String(1000);
        createdAt    : DateTime;
        modifiedAt   : DateTime;
}
```

> 리뷰 추가/삭제/수정 시 `Contents.avg_rating`와 `rating_count`를 **CAP After Handler**로 재계산.

---

### ContentCast (선택)

상세 페이지의 출연진 정보. 핵심 기능과 무관하므로 없어도 무방.

```cds
entity ContentCast {
    key cast_id        : String(10);
        content        : Association to Contents;
        name           : String(100);
        role           : String(50);   // 주연, 조연, 감독, 성우
        character_name : String(100);  // 극중 이름
}
```

---

## 3. 연결 요약

```text
Users (1) ──> Review (N)                    // 작성자

Contents (1) <── (N) ContentGenre (N) ──> (1) Genre  // 장르 여러 개
Contents (1) <── (N) ContentTag   (N) ──> (1) Tag    // 태그 여러 개
Contents (1) <── (N) Review                        // 리뷰 여러 개
Contents (1) <── (N) ContentCast                   // 출연진 여러 명
```

---

## 4. 핵심 OData 호출

```http
# 장르로 필터링
GET /odata/v4/ContentService/Contents
  ?$expand=genres($expand=genre)
  &$filter=genres/any(g: g/genre/genre_id eq 'G0001')

# 태그로 필터링
GET /odata/v4/ContentService/Contents
  ?$expand=tags($expand=tag)
  &$filter=tags/any(t: t/tag/tag_id eq 'T0001')

# 검색 (title LIKE)
GET /odata/v4/ContentService/Contents
  ?$filter=contains(title,'검색어')

# 상세 조회 (장르 + 태그 + 리뷰 expand)
GET /odata/v4/ContentService/Contents('C0001')
  ?$expand=genres($expand=genre),tags($expand=tag),reviews
```

---

## 5. CSV 시드 데이터 예시

### Genre.csv

```csv
genre_id;name
G0001;액션
G0002;드라마
G0003;코미디
G0004;스릴러
G0005;SF
```

### Tag.csv

```csv
tag_id;name
T0001;SF
T0002;19금
T0003;명작
T0004;감동
```

### ContentGenre.csv

```csv
content_content_id;genre_genre_id
C0001;G0001
C0001;G0005
C0002;G0002
C0002;G0004
```

> 컨텐츠 C0001 = 액션 + SF  
> 컨텐츠 C0002 = 드라마 + 스릴러

---

## 6. 구현할 파일 목록

| 파일                                           | 내용                                   |
| -------------------------------------------- | ------------------------------------ |
| `db/cds/Genre-model.cds`                     | Genre, Tag, ContentGenre, ContentTag |
| `db/cds/Content-model.cds`                   | Contents, Review, ContentCast        |
| `db/cds/User-model.cds`                      | Users (기존 활용 또는 수정)                  |
| `db/index.cds`                               | 모든 파일 import                         |
| `db/data/*.csv`                              | 시드 데이터                               |
| `srv/cds/Content-service.cds`                | ContentService 정의                    |
| `srv/src/feature/content/Content.handler.ts` | 핸들러 구현                               |
