# GenericTile (`sap.m.GenericTile`)

> 다목적 타일 컴포넌트. 숫자, 텍스트, 아이콘, 차트 등을 표시. Fiori Launchpad/Overview Page의 핵심 구성 요소

🔗 [SDK 문서](https://sdk.openui5.org/#/entity/sap.m.GenericTile)
🔗 [LaunchTile 샘플](https://sdk.openui5.org/#/entity/sap.m.GenericTile/sample/sap.m.sample.GenericTileAsLaunchTile)

---

## 기본 구조
```xml
<GenericTile
    header="Sales Orders"
    subheader="Overdue"
    icon="sap-icon://sales-order"
    press="onTilePress">
    <TileContent unit="EUR" footer="Current Quarter">
        <NumericContent value="150" scale="K" />
    </TileContent>
</GenericTile>
```

## 주요 속성
| 속성 | 설명 |
|------|------|
| `header` | 타일 상단 제목 (볼드) |
| `subheader` | 부제목 (작은 글씨) |
| `icon` | 왼쪽 상단 아이콘 |
| `press` | 클릭 핸들러 (LaunchTile 용도) |
| `mode` | `ContentMode` / `HeaderMode` / `InLineMode` |
| `size` | `Auto` / `S` / `M` / `L` |
| `frameType` | 프레임 스타일 |
| `backgroundImage` | 배경 이미지 URL |

## 콘텐츠 영역 (Aggregation)
| Aggregation | 컴포넌트 | 설명 |
|------|------|------|
| `tileContent` | `TileContent` | 메인 콘텐츠 (숫자, 텍스트) |
| `icon` | `sap.ui.core.Icon` | 아이콘 |

---

## TileContent + NumericContent
```xml
<GenericTile header="Revenue">
    <TileContent unit="USD" footer="This Month">
        <NumericContent
            value="1.2"
            scale="M"
            valueColor="Good"
            indicator="Up"
            withMargin="false" />
    </TileContent>
</GenericTile>
```

### TileContent 속성
| 속성 | 설명 |
|------|------|
| `unit` | 단위 표시 (예: "EUR", "views") |
| `footer` | 하단 텍스트 |
| `content` | 내부 콘텐츠 (`NumericContent` 등) |

### NumericContent 속성
| 속성 | 설명 |
|------|------|
| `value` | 숫자 값 |
| `scale` | 축약 단위 (`K`, `M`, `B`) |
| `valueColor` | `Good`(초록) / `Error`(빨강) / `Critical`(주황) / `Neutral`(회색) |
| `indicator` | `Up`(▲) / `Down`(▼) 트렌드 화살표 |
| `withMargin` | 텍스트 여백 |

---

## GenericTileAsLaunchTile (샘플)

LaunchTile = 클릭 가능한 타일. `press` 이벤트로 화면 이동:
```xml
<GenericTile
    header="영화 상세"
    subheader="더 라스트 시그널"
    icon="sap-icon://detail-view"
    press="onGoToDetail" />
```

### LaunchTile vs ContentTile
| | LaunchTile | ContentTile |
|------|:--:|:--:|
| 용도 | 화면 이동 | 정보 표시 |
| `press` 이벤트 | ✅ | 선택적 |
| 콘텐츠 | 간단 (header+icon) | 풍부 (NumericContent 등) |
| 호버 효과 | 클릭 가능 표시 | 일반 표시 |

---

## 💡 OTT 프로젝트 활용 아이디어

### 1. 홈 화면 메뉴 타일
```
┌──────────┐ ┌──────────┐ ┌──────────┐
│ 🎬        │ │ 📺        │ │ 👤        │
│ 콘텐츠    │ │ 시리즈    │ │ 마이페이지 │
│ 120편     │ │ 35편     │ │ 구독중    │
└──────────┘ └──────────┘ └──────────┘
```

### 2. 통계 대시보드
```xml
<!-- 오늘의 시청 통계 -->
<GenericTile header="오늘의 시청">
    <TileContent unit="views" footer="실시간">
        <NumericContent value="12.4" scale="K" valueColor="Good" indicator="Up" />
    </TileContent>
</GenericTile>

<!-- 신규 가입자 -->
<GenericTile header="신규 가입">
    <TileContent unit="users" footer="이번 달">
        <NumericContent value="856" valueColor="Good" indicator="Up" />
    </TileContent>
</GenericTile>
```

### 3. 추천 콘텐츠 (SlideTile과 조합)
```xml
<SlideTile slideInterval="4000">
    <tiles>
        <GenericTile
            header="오늘의 추천"
            subheader="더 라스트 시그널"
            icon="sap-icon://movie"
            press="onGoToDetail" />
        <GenericTile
            header="인기 상승"
            subheader="다크 시티 시즌2"
            icon="sap-icon://trend-up"
            press="onGoToDetail" />
    </tiles>
</SlideTile>
```

---

## GenericTile vs f:Card vs CSS Card

| | GenericTile | f:Card | CSS Card |
|------|:--:|:--:|:--:|
| Fiori 표준 | ✅ | ✅ | ❌ (커스텀) |
| 숫자 표시 | ✅ NumericContent | 텍스트만 | 자유롭게 |
| KPI 지표 | ✅ scale, indicator | ❌ | ❌ |
| UI5 1.136 지원 | ✅ | ⚠️ 제한적 | ✅ (항상 가능) |
| 복잡한 레이아웃 | 제한적 | 제한적 | 무제한 |
| 난이도 | 중간 | 중간 | 높음 (CSS 직접) |

> **우리 프로젝트**: UI5 1.136 CDN 제약으로 `f:Card` default aggregation 문제가 있어 **CSS Card**를 사용했음. 그러나 GenericTile은 1.136에서도 잘 작동하며, 숫자/KPI 표시에 특히 강력함.
