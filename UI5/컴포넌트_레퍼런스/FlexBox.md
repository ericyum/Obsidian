# HBox / VBox / FlexBox (`sap.m`)

> Flexbox 기반 레이아웃 컨테이너

---

## FlexBox
가장 기본적인 flex 컨테이너. `HBox`와 `VBox`는 FlexBox의 방향 고정 버전.

### FlexBox 속성
| 속성 | 값 | 설명 |
|------|-----|------|
| `direction` | `Row` / `Column` | 배치 방향 |
| `justifyContent` | `Start` / `Center` / `End` / `SpaceBetween` | 주축 정렬 |
| `alignItems` | `Start` / `Center` / `End` / `Baseline` / `Stretch` | 교차축 정렬 |
| `wrap` | `NoWrap` / `Wrap` | 줄바꿈 여부. enum! `"true"` ❌ |

### HBox / VBox
```xml
<HBox alignItems="Center">
    <Label text="플랜" />
    <Button icon="sap-icon://hint" />
</HBox>

<VBox class="membershipFormVBox">
    <Label text="라벨" />
    <Text text="값" />
</VBox>
```

### FlexBox 두 가지 용법

#### 1. Aggregation Binding (반복)
```xml
<FlexBox items="{viewModel>/Detail/genres/results}">
    <ObjectStatus text="{viewModel>genre/name}" />
</FlexBox>
```
`items`에 배열 바인딩 → 배열 길이만큼 자식 복제 (for문처럼). 장르/태그 칩 렌더링에 사용.

#### 2. 순수 레이아웃 컨테이너
```xml
<FlexBox justifyContent="Center" wrap="Wrap">
    <!-- 자식 요소들 수동 배치 -->
</FlexBox>
```
`items` 없으면 자식들을 정렬하는 컨테이너. 플랜 카드 그리드에 사용.

---

## ⚠️ Flex margin 주의
SAPUI5 flex item에 margin을 주려면 **직접 flex child인 요소**에 걸어야 함:
```xml
<!-- ❌ span에는 margin이 안 먹힘 -->
<Text class="sapUiSmallMarginBegin" text="값" />

<!-- ✅ HBox가 직접 flex child → margin 먹힘 -->
<HBox class="sapUiSmallMarginBegin">
    <Text text="값" />
</HBox>
```

---

## 사용된 곳
- **detailPage**: 장르/태그 칩(FlexBox items), 리뷰 폼 버튼(FlexBox 레이아웃)
- **myPage**: 헤더(HBox), 카드 내 필드(VBox), Info 버튼(HBox)
- **PlanChange**: 플랜 카드 그리드(FlexBox wrap)
