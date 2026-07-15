# SlideTile (`sap.m.SlideTile`)

> 여러 타일을 슬라이드 애니메이션으로 순환하는 컨테이너. Fiori Launchpad/홈 화면에서 사용

🔗 [SDK 문서](https://sdk.openui5.org/#/entity/sap.m.SlideTile)

---

## 기본 구조
```xml
<SlideTile slideInterval="3000" transitionTime="500">
    <tiles>
        <GenericTile header="Sales Orders" subheader="150" />
        <GenericTile header="New Leads" subheader="23" />
        <GenericTile header="Open Tasks" subheader="7" />
    </tiles>
</SlideTile>
```

## 주요 속성
| 속성 | 타입 | 설명 |
|------|------|------|
| `slideInterval` | int | 슬라이드 전환 간격 (ms). `0` → 자동 전환 없음 |
| `transitionTime` | int | 슬라이드 애니메이션 시간 (ms) |
| `tiles` | aggregation | 내부 타일들 (`GenericTile` 등) |

## Aggregation
| 이름 | 설명 |
|------|------|
| `tiles` | 슬라이딩될 타일 목록. 보통 `GenericTile`을 자식으로 사용 |

## 동작 방식
```
┌────────────────────┐
│  [Tile A]           │ ← 현재 표시
├────────────────────┤
│  [Tile B]           │ ← 대기 중
├────────────────────┤
│  [Tile C]           │ ← 대기 중
└────────────────────┘
    ↑ slideInterval마다 아래로 슬라이드
```

- `slideInterval`마다 자동으로 다음 타일로 전환
- 마지막 타일 이후 첫 타일로 순환
- `slideInterval="0"`이면 수동 전환만 가능
- 사용자가 터치/클릭으로 직접 전환 가능

## 이벤트
| 이벤트 | 설명 |
|------|------|
| `press` | 타일 영역 클릭 시 |
| `slideEnd` | 슬라이드 전환 완료 시 |

## 💡 OTT 프로젝트 활용 아이디어
- **홈 화면 상단**: 추천 콘텐츠를 SlideTile로 배너처럼 표시
  ```
  [인기 영화 A] → [신규 시리즈 B] → [장르 추천 C] → 순환
  ```
- **통계 요약**: 오늘의 시청자 수, 신규 가입자, 인기 장르 등 KPI 슬라이드
- `slideInterval="5000"`으로 자동 전환되는 프로모션 배너

## 사용 예시 (OTT 홈)
```xml
<SlideTile slideInterval="4000" transitionTime="600">
    <tiles>
        <GenericTile
            header="오늘의 추천"
            subheader="더 라스트 시그널"
            icon="sap-icon://movie"
            press="onGoToDetail" />
        <GenericTile
            header="인기 시리즈"
            subheader="다크 시티"
            icon="sap-icon://tv"
            press="onGoToDetail" />
        <GenericTile
            header="신규 콘텐츠"
            subheader="12편 추가됨"
            icon="sap-icon://add-product"
            press="onGoToNew" />
    </tiles>
</SlideTile>
```

---

## SlideTile vs Carousel
| | SlideTile | Carousel (`sap.m.Carousel`) |
|------|:--:|:--:|
| 용도 | 작은 타일 카드 | 큰 이미지/페이지 |
| 자식 | GenericTile 등 | 이미지, HTML |
| 크기 | 타일 크기 (작음) | 전체 페이지 크기 |
| Fiori 표준 | ✅ 홈 화면 | 드물게 사용 |
