# ZEN-OTT — 세션 요약 (2026-07-15): MyPage & PlanChange UI 전면 개선

**범위**: MyMembership 헤더/카드/버튼 재구성 → Info Popover 도입 → PlanChange ObjectPageLayout 전환 → 플랜 카드 스타일링
**최종 상태**: MyMembership + PlanChange UI 완전 개선, MyMembership 카드 박스·Info Popover·둥근 버튼, PlanChange ObjectPageLayout·라벨 분리 완료

---

## Part 1 — MyMembership 헤더 개선

### 변경

| 요소 | 전 | 후 |
|------|------|------|
| 사용자명 타이틀 | 기본 H2 | 1.9rem (`membershipHeader .sapMTitle`) |
| Active 배지 텍스트 | 기본 | 1.2rem + `top: -0.2rem` (살짝 위로) |
| 뒤로가기 버튼 | 없음 | nav-back 아이콘 + "뒤로가기", 2.5rem 높이, 1.1rem 폰트 |
| 뒤로가기 버튼 위치 | - | `top: 0.4rem` (살짝 아래로) |
| 섹션 제목 "멤버십 정보" | 기본 | 2.2rem |
| 섹션 제목-카드 간격 | 0.5rem | 1.2rem |

### 구조
```xml
<uxap:headerTitle>
    <uxap:ObjectPageDynamicHeaderTitle>
        <uxap:heading>
            <HBox class="membershipHeader">
                <Title text="사용자명" />
                <ObjectStatus text="Active" class="membershipHeaderStatus" />
            </HBox>
        </uxap:heading>
        <uxap:actions>
            <Button icon="sap-icon://nav-back" press="onNavBack" type="Transparent" />
        </uxap:actions>
    </uxap:ObjectPageDynamicHeaderTitle>
</uxap:headerTitle>
```

---

## Part 2 — 멤버십 정보 카드 박스

### 변경 전
```
SimpleForm (ResponsiveGridLayout 2열)
├── Plan: 프리미엄
├── Monthly Fee: 15,000원
├── Simultaneous Streams: 2명
└── Period: 2026-01-01 ~ 2026-12-31
```

### 변경 후
```
┌───────────────────────────────────────┐
│  흰색 둥근 카드 박스 (membershipInfoCard) │
│                                       │
│  Plan                Monthly Fee      │
│    프리미엄            15,000원        │
│                                       │
│  Simultaneous Streams ⓘ  Period ⓘ    │
│    2명                 2026-01-01~    │
│                       2026-12-31      │
│                                       │
│         [Change Plan] [Cancel]        │
└───────────────────────────────────────┘
```

### 카드 CSS
- `.membershipInfoCard`: `background: #fff`, `border-radius: 0.75rem`, `box-shadow`, `padding: 1rem`
- 폰트: 라벨·값 1.15rem (≈1.3배)
- 값 들여쓰기: `padding-left: 0.75rem`

### 버튼 통합
기존에는 카드 내용 서브섹션 + 버튼 서브섹션 **2개**로 분리되어 있었으나,
버튼 서브섹션의 `sapUxAPSubSectionSeeMoreContainer`가 불필요한 구분선을 생성하고
CSS로 스타일링이 불가능한 문제가 있어 **하나의 서브섹션으로 통합**.
버튼은 카드 박스 내 `HBox justifyContent="End"`로 우측 정렬.

### 액션 버튼 스타일
- `.membershipActionBtn`: 높이 2.75rem, `border-radius: 1.5rem` (알약형)
- 폰트: 1.15rem, 최소폭: 6rem, 좌우 패딩: 1.5rem

---

## Part 3 — Info Popover (ⓘ)

### 위치
"Simultaneous Streams"와 "Period" 라벨 우측에 `message-information` 아이콘 버튼.

### 아이콘
- 초기: `sap-icon://hint` (물음표) → 최종: `sap-icon://message-information` (원 안의 i)
- 크기: 1.5rem × 1.5rem, 파란색(#0854a0)

### Popover 구현

**시행착오**: 초기에는 Text를 그대로 Popover content로 넣었으나 패딩이 전혀 적용되지 않음.
원인은 `sap.m.Popover`의 content 영역이 자체적으로 패딩을 제공하지 않기 때문.
SAP 샘플과 비교한 결과, `sap.ui.layout.VerticalLayout` + `sapUiContentPadding` 클래스가 필요함을 발견.

**최종 코드**:
```javascript
var oLayout = new VerticalLayout({ width: "100%" });
oLayout.addStyleClass("sapUiContentPadding");
oLayout.addContent(new Text({ text: sBody }));
```

**Popover 구조**:
```
┌─────────────────────────┐
│  동시시청 안내            │ ← title
├─────────────────────────┤
│                         │
│  동시시청이란 하나의     │ ← VerticalLayout
│  계정으로 여러 기기에서   │   + sapUiContentPadding
│  ...                    │
│                         │
└─────────────────────────┘
```
- Close 버튼 없음 (바깥 클릭으로 닫힘)
- `contentWidth: "22rem"`

### i18n
- `info.maxStreamsTitle` / `info.maxStreamsBody` (한글)
- `info.periodTitle` / `info.periodBody` (한글)
- 동일 키 영문 버전 (`i18n_en.properties`)
- 브라우저 언어에 따라 자동 전환 (manifest.json 설정 그대로)

---

## Part 4 — PlanChange 구조 변경

### Page → ObjectPageLayout 전환

**변경 전**: `sap.m.Page` 사용. 헤더 바가 얇고 CSS 적용이 전혀 되지 않음.
**변경 후**: `ObjectPageLayout` + `ObjectPageDynamicHeaderTitle` 사용 → MyMembership과 동일한 헤더 스타일 자동 적용.

```xml
<uxap:ObjectPageLayout>
    <uxap:headerTitle>
        <uxap:ObjectPageDynamicHeaderTitle>
            <uxap:heading>
                <HBox class="membershipHeader">
                    <Title text="플랜 선택" />
                </HBox>
            </uxap:heading>
            <uxap:actions>
                <Button icon="sap-icon://nav-back" press="onNavBack" />
            </uxap:actions>
        </uxap:ObjectPageDynamicHeaderTitle>
    </uxap:headerTitle>
    <uxap:sections>
        <uxap:ObjectPageSection showTitle="false">
            <!-- 내용 -->
        </uxap:ObjectPageSection>
    </uxap:sections>
</uxap:ObjectPageLayout>
```

### 숨겨진 섹션 헤더 공간 제거

`showTitle="false"`인 섹션 헤더가 높이를 차지해 컨텐츠와의 간격이 과도하게 벌어지는 문제.
CSS로 완전히 제거:
```css
.sapUxAPObjectPageSectionHeaderHidden {
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}
.sapUxAPObjectPageSectionNoTitle .sapUxAPObjectPageSectionContainer {
    padding-top: 0 !important;
}
```

---

## Part 5 — 플랜 카드 스타일링

### 변경 전
```
┌──────────────────┐
│  베이직           │ ← H3 타이틀
│  9900KRW/month   │ ← 가격
│  동시시청 1명     │ ← 라벨+값 통합
│  [선택]           │
└──────────────────┘
```

### 변경 후
```
┌─────────────────────┐
│  베이직              │ ← 1.4rem 타이틀
│  Monthly Fee         │ ← 회색 라벨 (planCardLabel)
│  9900 KRW/month      │ ← 1.15rem 값
│  Simultaneous Streams│ ← 회색 라벨
│  4 users             │ ← 값
│  [  Select  ]        │ ← 둥근 버튼 2.5rem
└─────────────────────┘
```

### 카드 CSS
- `.planCard`: `background: #fff`, `border-radius: 0.75rem`, `box-shadow`, `padding: 1.5rem 2rem`, `min-width: 14rem`
- `.planCardLabel`: `color: #6a6d70`, `font-size: 0.9rem` (회색 라벨)
- `.planCard .sapMTitle`: 1.4rem, `.planCard .sapMText`: 1.15rem

### 라벨 분리

**변경 전**: `formatMaxStreams` → "동시시청 4명" (라벨+값 통합)

**변경 후**:
- 뷰에서 라벨과 값 분리
- `{i18n>label.monthlyFee}` / `{i18n>label.maxStreams}` → 회색 라벨
- `formatPrice` / `formatMaxStreamsValue` → 값만 반환

### 버튼 스타일
- 카드 내 Select 버튼: 높이 2.5rem, `border-radius: 1.25rem`, 폰트 1.1rem
- 하단 제출 버튼 (`.planSubmitBtn`): 높이 2.75rem, `border-radius: 1.5rem`, 폰트 1.2rem, 최소폭 8rem

---

## 📝 수정 파일 목록

| # | 파일 | 변경 |
|:--:|------|------|
| 1 | `MyMembership.view.xml` | 헤더 액션(뒤로가기), 카드 박스, Info 버튼, 버튼 통합 |
| 2 | `MyMembership.controller.js` | onNavBack, onInfoMaxStreams, onInfoPeriod, _showInfoPopover |
| 3 | `PlanChange.view.xml` | Page→ObjectPageLayout, 카드 라벨 분리, planCard/planSubmitBtn 클래스 |
| 4 | `PlanChange.controller.js` | formatMaxStreamsValue 추가, 데드코드(formatMaxStreams) 제거 |
| 5 | `style.css` | 카드 박스, 폰트 1.3배, 버튼 둥글게, Info 버튼, PlanChange 섹션/카드/버튼 |
| 6 | `i18n.properties` | Info Popover 제목/본문, tooltip 키 추가, button.close 제거 |
| 7 | `i18n_en.properties` | 영문 Info Popover 키 추가, button.close 제거 |
| 8 | `manifest.json` | (원복) |
| 9 | `index.html` | (원복 — 불필요 스타일 제거) |

---

## 🧠 절대 잊지 말 것 (오늘의 핵심 교훈)

| # | 내용 |
|---|------|
| 1 | **Popover 패딩**: `VerticalLayout` + `addStyleClass("sapUiContentPadding")` 사용. `Text`만 넣으면 패딩 없음 |
| 2 | **`sap.ui.layout.VerticalLayout` ≠ `sap.m.VBox`**: 전자는 `sapUiVlt` + `sapUiContentPadding` 렌더링, 후자는 `sapMFlexBox` |
| 3 | **CSS 선택자 공백 vs 붙임**: `.foo .bar`(자손) ≠ `.foo.bar`(동일 요소) |
| 4 | **`sapUxAPSubSectionSeeMoreContainer` CSS 불가**: SAPUI5 내부 요소, 스타일 override 불가 → 구조 변경으로 우회 |
| 5 | **`sap.m.Page` CSS 불가**: ObjectPageLayout과 달리 기본 CSS override가 어려움 → ObjectPageLayout으로 전환 |
| 6 | **ObjectPageLayout 섹션 헤더 숨김**: `showTitle="false"`만으로는 공간이 남음 → CSS로 height/min-height 0 추가 |
| 7 | **라벨/값 분리**: 카드에서 회색 라벨 + 값 분리 시 포매터도 분리 필요 |
| 8 | **`addStyleClass`는 메서드**: 생성자 속성이 아님. `new Control()` 이후 `.addStyleClass()` 호출 |
| 9 | **버튼을 서브섹션 안에 통합**: SeeMoreContainer 문제 회피 + 카드 일관성 향상 |

---

> **세션 일시**: 2026-07-15
> **이전 기록**: `컴팩트_2026-07-14.md`, `2026-07-14 detailPage UI 전면 개선.md`
