# ObjectPageLayout 및 관련 컨트롤 원리 조사

**작성일**: 2026-07-01
**범위**: 피드백 #1, #2, #3, #6, #7 에 사용된 SAP UI5 컨트롤의 원리 및 속성 정리

---

## 1. `sap.uxap.ObjectPageLayout` — Anchor Bar (상단 배너)

### 개요
`ObjectPageLayout`은 SAP Fiori의 표준 상세 페이지 패턴을 구현하는 컨트롤. 헤더 + 섹션 + anchor bar로 구성된다.

### 구조
```
┌─────────────────────────────────┐
│  ObjectPageDynamicHeaderTitle    │  ← 헤더 (타이틀, 상태, 액션 버튼)
├─────────────────────────────────┤
│  [기본정보] [출연진] [리뷰] ...   │  ← Anchor Bar (섹션 타이틀 버튼)
├─────────────────────────────────┤
│  ■ 기본정보 섹션                  │
│    ...                          │  ← 스크롤 영역
│  ■ 출연진 섹션                    │
│    ...                          │
│  ■ 리뷰 섹션                      │
│    ...                          │
└─────────────────────────────────┘
```

### Anchor Bar 동작 원리

**1. 자동 생성**
- `ObjectPageLayout`에 `ObjectPageSection`을 추가하면, 각 Section의 `title` 속성값이 자동으로 anchor bar 버튼이 된다
- 개발자가 별도 버튼을 만들지 않아도 됨

**2. 클릭 시 이동**
- anchor bar 버튼 클릭 → 해당 `ObjectPageSection`의 DOM 위치로 **자동 스크롤**
- 내부적으로 `scrollIntoView` 또는 `scrollTop` 조작 사용
- 추가 JS 코드 불필요

**3. 고정 (Sticky)**
- 페이지를 아래로 스크롤하면 anchor bar는 **상단에 고정**됨 (CSS `position: sticky`)
- 헤더가 접히면서(`snappedContent` 모드) anchor bar가 viewport 상단에 붙음

### 주요 속성

| 속성 | 설명 |
|------|------|
| `upperCaseAnchorBar` | anchor bar 텍스트 대문자 변환 여부 (기본 `true`) |
| `showAnchorBar` | anchor bar 표시 여부 (기본 `true`) |
| `showAnchorBarPopover` | anchor bar 버튼이 많을 때 팝오버로 축소 여부 |
| `toggleHeaderOnTitleClick` | 헤더 타이틀 클릭 시 확장/축소 토글 |
| `showFooter` | 하단 푸터 표시 여부 |

### ObjectPageDynamicHeaderTitle

헤더 영역을 구성하는 컨트롤. 3가지 콘텐츠 영역을 가짐:

| 영역 | 용도 |
|------|------|
| `heading` | 타이틀 텍스트 |
| `expandedContent` | 헤더 확장 시 표시 (별점, 상태 등) |
| `snappedContent` | 헤더 접힘 시 표시 (축약 정보) |
| `actions` | 우측 액션 버튼들 |

---

## 2. `sap.m.ObjectStatus` — state 속성과 색상

### 개요
아이콘 + 텍스트로 상태를 표시하는 컨트롤. 주로 "활성", "오류", "정보" 등의 상태 표시에 사용.

### `state` 속성별 색상

| state 값 | 텍스트 색상 | 용도 |
|----------|:----------:|------|
| `"None"` | 기본 (검정) | 중립적 정보 |
| `"Information"` | **파란색** | 안내/정보 |
| `"Success"` | 초록색 | 성공/완료 |
| `"Warning"` | 주황색 | 경고 |
| `"Error"` | 빨간색 | 오류 |

### 사용 시 주의
- `state="Information"`은 **파란색**으로 렌더링되어 링크처럼 보일 수 있음
- 단순 정보 표시(태그, 장르 등)에는 `state="None"`이 적합
- 상태 표시가 아닌 정보성 레이블에는 `ObjectStatus` 대신 `Text` + `core:Icon` 조합 고려

### 예시
```xml
<!-- 링크처럼 보이는 문제 -->
<ObjectStatus text="액션" state="Information" icon="sap-icon://tag" />

<!-- 해결: state="None" -->
<ObjectStatus text="액션" state="None" icon="sap-icon://tag" />
```

---

## 3. `sap.ui.table.Table` — visibleRowCount

### 개요
`Table`(sap.ui.table)은 TreeTable 및 AnalyticalTable의 기반이 되는 행/열 테이블 컨트롤.

### `visibleRowCount` 속성

| 속성 | 타입 | 기본값 | 설명 |
|------|------|:------:|------|
| `visibleRowCount` | int | 10 | 한 번에 보여줄 행 개수 |
| `minAutoRowCount` | int | 5 | `visibleRowCount="auto"`일 때 최소 |
| `rowActionCount` | int | 1 | 행 액션 아이콘 개수 |

### 동작 방식
- `visibleRowCount="10"` → 테이블에 10개 행 표시, 초과 시 **세로 스크롤바** 생성
- 미설정 시 **모든 행이 한 번에 렌더링** → 데이터가 많으면 페이지가 과도하게 길어짐
- 행 높이 × visibleRowCount = 테이블 높이 (CSS로 고정)

### 예시: 고정 행 수
```xml
<!-- Before: 행 제한 없음 → 모든 출연진 표시 -->
<t:Table rows="{viewModel>/Detail/casts/results}">
    ...
</t:Table>

<!-- After: 최대 10행 + 스크롤 -->
<t:Table rows="{viewModel>/Detail/casts/results}" visibleRowCount="10">
    ...
</t:Table>
```

### 데이터 개수만큼 동적으로 행 수 조절하기

`visibleRowCount`는 **expression binding이 가능**하므로, 데이터 길이에 따라 자동으로 행 수를 맞출 수 있다.

**방법 A: 데이터 개수만큼** (expression binding)
```xml
<t:Table
    rows="{viewModel>/Detail/casts/results}"
    visibleRowCount="{= ${viewModel>/Detail/casts/results}.length }">
```
> 데이터 3개 → 3행, 15개 → 15행. 단, 50개면 50행이 되어 피드백 #3 문제는 그대로.

**방법 B: 데이터 개수만큼 + 최대값 제한** ← 추천
```xml
<t:Table
    rows="{viewModel>/Detail/casts/results}"
    visibleRowCount="{= Math.min(${viewModel>/Detail/casts/results}.length, 15) }">
```
> 데이터 3개 → 3행, 15개 → 15행, 50개 → 15행(+스크롤). 데이터가 적으면 군더더기 없고, 많으면 제한.

**방법 C: 컨트롤러에서 setVisibleRowCount()** (JS 동적 제어)
```javascript
// _loadDetail() 성공 후
var aCasts = oData.casts && oData.casts.results;
var nCount = aCasts ? aCasts.length : 0;
this.byId("castTable").setVisibleRowCount(Math.min(nCount, 15));
```
> 완전한 제어가 필요할 때. JS 로직으로 조건 분기 등 복잡한 처리 가능.

| 방법 | 데이터 3개 | 데이터 20개 | 장점 | 단점 |
|------|:---------:|:---------:|------|------|
| A: `.length`만 | 3행 | 20행(너무 김) | 간단함 | 많으면 너무 길어짐 |
| **B: `Math.min(.length, 15)`** | 3행 | 15행(+스크롤) | **적당량+제한** | — |
| C: `setVisibleRowCount()` | 3행 | 15행 | 완전 제어 | JS 코드 추가 필요 |

---

## 4. `sap.ui.layout.form.SimpleForm` — ResponsiveGridLayout 정렬

### 개요
`SimpleForm`은 폼 레이아웃을 간편하게 구성하는 컨트롤. `layout` 속성으로 내부 배치 방식을 결정.

### ResponsiveGridLayout
- CSS Grid 기반 반응형 레이아웃
- `columnsL`, `columnsM`, `columnsS` → 화면 크기별 열 개수
- `labelSpanL`, `labelSpanM` → 라벨이 차지하는 그리드 span
- 각 `form:content` 항목은 `GridData`로 개별 span 지정 가능

### Label과 Text의 수직 정렬 문제 원인

```
VBox
├── Label (margin: 0.25rem 0 상하)   ← 상하 여백 있음
│     텍스트 (bold, 더 두꺼움)
└── Text                             ← Label과 다른 margin
      텍스트
```

- `Label`과 `Text`는 각각 고유한 CSS margin/padding을 가짐
- `Label design="Bold"`는 폰트 두께가 달라서 동일 line-height에서도 시각적 높이가 다름
- VBox 내에서는 문제없지만, **다른 VBox와 수평 비교 시** 미묘한 높이 차이 발생

### 해결: CSS 보정
```css
.membershipFormVBox > .sapMLabel {
    margin-bottom: 0.125rem;  /* 하단 margin 최소화 */
}
.membershipFormVBox > .sapMText {
    line-height: 1.4;         /* 일관된 행 높이 */
}
```

---

## 5. `sap.m.MessageBox` vs `sap.m.Dialog` — 팝업

### MessageBox.confirm()
```javascript
MessageBox.confirm("메시지", {
    title: "제목",
    onClose: function(sResult) { /* OK/Cancel */ }
});
```

| 특징 | 설명 |
|------|------|
| 아이콘 | **고정**: `sap-icon://question-mark` (?) |
| 텍스트 | 짧은 메시지에 적합, 자동 줄바꿈 제한적 |
| 너비 | 약 20rem(320px) 고정 |
| 버튼 | OK / Cancel 두 개 |
| 장점 | 간단한 확인 다이얼로그에 빠르게 사용 가능 |
| 단점 | 커스터마이징 여지가 거의 없음, 아이콘 변경 불가 |

### MessageBox의 다른 타입

| 메서드 | 아이콘 | 기본 버튼 |
|--------|:------:|------|
| `.confirm()` | ? | OK + Cancel |
| `.information()` | ℹ️ | OK |
| `.warning()` | ⚠️ | OK |
| `.error()` | ❌ | OK |
| `.show()` | 없음 | 커스텀 |

> **한계**: `.confirm()`의 아이콘은 변경 불가. 해지·재개·구독에 각기 다른 아이콘을 쓰려면 Dialog 필요.

### Dialog (커스텀)
```javascript
new Dialog({
    title: "제목",
    type: "Message",
    icon: "sap-icon://warning2",   // ← 원하는 아이콘 직접 지정
    contentWidth: "auto",          // ← 텍스트 길이에 자동 맞춤
    content: [new Text({ text: "메시지" })],
    beginButton: new Button({ /* 확인 */ }),
    endButton: new Button({ /* 취소 */ })
});
```

| 특징 | 설명 |
|------|------|
| 아이콘 | **자유롭게** 지정 가능 |
| 텍스트 | Text 컨트롤로 자동 줄바꿈 자연스럽게 처리 |
| 너비 | `contentWidth: "auto"`로 텍스트에 맞춤 → 짤림 방지 |
| 버튼 | beginButton + endButton 완전 커스텀 가능 |

### 아이콘 추천

| 상황 | 아이콘 | 의미 |
|------|--------|------|
| 해지 확인 (파괴적 액션) | `sap-icon://warning2` | ⚠ 세모+느낌표 |
| 재개 확인 | `sap-icon://refresh` | 🔄 갱신 |
| 구독 시작 | `sap-icon://add-coursebook` | 📘 신규 등록 |
| 삭제 확인 | `sap-icon://delete` | 🗑️ |

---

## 요약

| 피드백 # | 사용 컨트롤 | 핵심 속성/원리 |
|:--------:|------------|---------------|
| 1 | `ObjectPageLayout` | Section title이 자동으로 anchor bar 생성, 클릭 시 스크롤 |
| 2 | `ObjectStatus` | `state="Information"` → 파란색, `"None"` → 기본색 |
| 3 | `Table` | `visibleRowCount` 미설정 → 모든 행 렌더링 |
| 6 | `SimpleForm` | Label/Text 간 margin 차이로 높낮이 어긋남 |
| 7 | `MessageBox` `Dialog` | confirm()은 아이콘 '?' 고정. Dialog로 커스텀해야 아이콘 변경 가능 |

---

---

## 6. `manifest.json` — `routing.config.transition` (화면 전환 애니메이션)

### 개요
MyMembership에서 PlanChange로 이동할 때 화면이 **스르륵 옆으로 밀리는** 애니메이션. 이건 `manifest.json`의 `routing.config.transition` 한 줄이 결정한다.

### 설정 위치
```json
"routing": {
    "config": {
        "routerClass": "sap.m.routing.Router",
        "controlId": "app",
        "controlAggregation": "pages",
        "transition": "slide",           ← ⭐ 이 한 줄!
        ...
    }
}
```

### 동작 원리
```
① Router가 URL 해시 변경 감지 (#/MyMembership → #/PlanChange)
    │
② config.controlId="app" → sap.m.App 컨트롤 찾음
    │
③ App은 내부적으로 NavContainer를 상속받음
    │
④ NavContainer.to(page, transitionName) 호출
    │  └─ transitionName = config.transition = "slide"
    │
⑤ "slide" 애니메이션으로 페이지 전환 실행
```

**핵심**: `sap.m.App`은 `sap.m.NavContainer`의 자식이다. `NavContainer`는 `to()` 메서드로 페이지를 전환할 때 `transition` 값을 보고 어떤 애니메이션을 쓸지 결정한다. 이 transition 값이 바로 manifest의 `routing.config.transition`에서 온다.

### 가능한 transition 값

| 값 | 애니메이션 | 느낌 |
|------|-----------|------|
| `"slide"` | 페이지가 **좌우로 밀림** (기본값) | 네이티브 앱 같은 페이지 이동 |
| `"fade"` | 페이지가 **서서히 나타남** | 부드러운 전환 |
| `"flip"` | 페이지가 **뒤집힘** | 화려한 전환 |
| `"show"` | 애니메이션 **없음** (즉시 표시) | 깜빡임 없는 전환 |

### View나 Controller에는 없나?

**없다.** View XML이나 Controller JS에는 transition 관련 설정이 전혀 없다. 오직 `manifest.json`의 `routing.config.transition` 한 줄만이 이 동작을 제어한다.

### 왜 Controller에서 `.navTo()`만 호출했는데 애니메이션이 생기나?

```javascript
// MyMembership.controller.js
onGoPlanChange: function () {
    this.getOwnerComponent().getRouter().navTo("PlanChange");
    //                              │
    //  Router.navTo() ──→ Target 생성 ──→ App.to(view, "slide")
    //                                                      ─────
    //                                        routing.config.transition 값!
}
```

`navTo("PlanChange")` 호출 한 줄이 내부적으로:
1. `"PlanChange"` 타겟을 찾아 `PlanChange.view.xml` 로드
2. `sap.m.App`의 `pages` aggregation에 추가
3. `config.transition` 값을 전환 애니메이션으로 지정해서 `to()` 호출

개발자는 transition을 신경 쓸 필요 없이, manifest에 값만 적어두면 Router가 알아서 처리한다.

### 변경하고 싶다면
```json
// 부드러운 페이드 효과를 원하면:
"transition": "fade"

// 애니메이션 없이 즉시 전환:
"transition": "show"
```

---

## 요약

| 피드백 # | 사용 컨트롤 | 핵심 속성/원리 |
|:--------:|------------|---------------|
| 1 | `ObjectPageLayout` | Section title이 자동으로 anchor bar 생성, 클릭 시 스크롤 |
| 2 | `ObjectStatus` | `state="Information"` → 파란색, `"None"` → 기본색 |
| 3 | `Table` | `visibleRowCount` 미설정 → 모든 행 렌더링 |
| 6 | `SimpleForm` | Label/Text 간 margin 차이로 높낮이 어긋남 |
| 7 | `MessageBox` `Dialog` | confirm()은 아이콘 '?' 고정. Dialog로 커스텀해야 아이콘 변경 가능 |
| — | `routing.config` | `transition: "slide"` → NavContainer의 페이지 전환 애니메이션 |

---

> **관련 문서**: [[2026-07-01 프론트엔드_피드백_수정_계획서]]
