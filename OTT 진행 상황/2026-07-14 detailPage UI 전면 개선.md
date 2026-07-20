# ZEN-OTT — 세션 요약 (2026-07-14): detailPage UI 전면 개선

**범위**: detailPage 카드 재구성 → FeedListItem/FeedInput 도입 → 헤더 확대 → 출연진 표 스타일링 → 전체 간격 조정
**최종 상태**: detailPage UI 완전 개선, 기본 정보 4그룹, 리뷰 FeedListItem+FeedInput, 앵커바/헤더 크기 조정 완료

---

## Part 1 — 기본 정보 카드 재구성

### 변경 전 (3그룹)
```
┌─────────────────────┬─────────────────────┐
│  컨텐츠 식별         │  제공 정보           │
│  Content Code: C001  │  Provider: Studio   │
│  Type: MOVIE         │  Registered Date: — │
├─────────────────────┴─────────────────────┤
│  분류                                     │
│  장르/태그                                 │
└───────────────────────────────────────────┘
```

### 변경 후 (4그룹)
```
┌─────────────────────┬─────────────────────┐
│  기본 정보           │  분류                │
│  Content Code: C001  │  장르: [액션] [스릴러] │
│  Type: MOVIE         │  태그: [#인기] [#미국] │
├─────────────────────┼─────────────────────┤
│  Reviews             │  Partner            │
│  Average Rating: ★★★ │  Provider Code: P001 │
│  Review Count: 2개   │  Provider: Studio   │
└─────────────────────┴─────────────────────┘
```

### 카드 CSS Grid 구조
- `detailCardGrid`: `display: grid; grid-template-columns: repeat(auto-fill, minmax(max(220px, 50%-1rem), 1fr))` → 2열 자동 배치
- 각 그룹: `detailCardGroup` > Title(H5) > VBox(`sapUiSmallMargin detailCardGroupContent`) > row들
- 모든 그룹 타이틀: 1.15rem bold, margin-bottom 0.25rem
- 모든 값 텍스트: 1.0rem
- 전체 카드: `detailInfoCard` — border-radius 0.75rem, 흰 배경, 그림자

### `:` 뒤 들여쓰기 문제 및 해결 과정

**문제**: Content Code, Type, Provider Code, Provider, Review Count 등의 값이 `:` 바로 옆에 붙어있음. 반면 장르/태그는 chip이 자연스럽게 들여쓰기 된 것처럼 보임.

**원인 분석**:
- 장르/태그 행: `<FlexBox class="detailChipBox">`가 직접 flex child → margin 먹힘
- 일반 행: `<Text class="detailCardValue">`가 span → 내부 wrapper에 감싸짐 → span margin 무시

**시도한 방법들**:
| # | 방법 | 결과 |
|:-:|------|:--:|
| 1 | `.detailCardValue { margin-left: 0.5rem !important }` | ❌ span 레벨 |
| 2 | `.detailCardRow > .sapMFlexItem:not(:first-child) { margin-left }` | ❌ SAPUI5 flex item 클래스에 막힘 |
| 3 | `gap: 0.5rem` on `.detailCardRow` | ❌ 안 먹힘 |
| 4 | `sapUiTinyMarginBegin` 클래스 직접 | ❌ 여전히 span에 걸림 |
| 5 | `<HBox class="sapUiSmallMarginBegin">`로 값 감싸기 | ✅ HBox가 직접 flex child! |

**최종 해결**: 모든 값 Text를 `<HBox class="sapUiSmallMarginBegin">`(1rem)으로 감쌈. 별점은 `.detailCardRatingBox`에 `margin-left: 1rem !important`, 장르/태그는 `.detailChipBox`에 `margin-left: 1rem !important`.

**줄맞춤**: 라벨 `min-width: 7rem`으로 통일 → 모든 `:` 뒤 값이 동일한 수평 위치에서 시작.

### 주요 교훈
- SAPUI5 flex item에 margin 주려면 **직접 flex child인 요소**에 걸어야 함
- `sapUiSmallMarginTop` vs `sapUiSmallMargin`: 전자는 top만, 후자는 전방향
- H5/H6 태그 차이 → 내용 시작 위치 어긋남 → H5로 통일

---

## Part 2 — 리뷰 목록: CustomListItem → FeedListItem

### 📂 참고 샘플
`C:\Users\염정운\project\작성 내용\sap.m.sample.FeedListItem\`

**샘플 파일 구조**:
```
webapp/
├── List.view.xml       ← FeedListItem 사용법 (핵심!)
├── List.controller.js   ← onPress, onActionPressed, removeItem
├── feed.json            ← 모의 데이터 (EntryCollection[].Actions[])
├── manifest.json
└── Component.js
```

**샘플 JSON 데이터 (`feed.json`)**:
```json
{
  "EntryCollection": [{
    "Author": "Alexandrina Victoria",
    "AuthorPicUrl": "test-resources/sap/m/images/dronning_victoria.jpg",
    "Type": "Request",
    "Date": "March 03 2013",
    "Actions": [
      { "Text": "Delete", "Icon": "sap-icon://delete", "Key": "delete" },
      { "Text": "Edit",   "Icon": "sap-icon://edit",   "Key": "edit" }
    ],
    "Text": "Lorem ipsum..."
  }]
}
```

**샘플 XML (핵심)**:
```xml
<List items="{/EntryCollection}">
    <FeedListItem
        sender="{Author}"           ← 볼드체 이름 링크
        icon="{AuthorPicUrl}"       ← 아바타 이미지
        info="{Type}"               ← 푸터 왼쪽 (회색)
        timestamp="{Date}"          ← 푸터 오른쪽 (회색)
        text="{Text}"               ← 본문
        convertLinksToAnchorTags="All"   ← ⚠️ boolean ❌, enum "All" ✅
        actions="{path: 'Actions', templateShareable: false}">  ← ⚠️ templateShareable 필수!
        <FeedListItemAction text="{Text}" icon="{Icon}" key="{Key}" press=".onActionPressed" />
    </FeedListItem>
</List>
```

**핵심 개념**:
- `sender`: 작성자 이름 → 볼드체 링크. 링크 제거하려면 CSS `pointer-events: none`
- `icon`: 아바타. 실제 이미지 URL 없으면 `sap-icon://person-placeholder`로 기본 아이콘
- `info`: 부가 정보 (푸터 왼쪽). 우리는 평점(예: "3.0 / 5") 표시
- `timestamp`: 날짜/시간 (푸터 오른쪽)
- `text`: 본문. HTML 태그 포함 가능, `convertLinksToAnchorTags="All"`로 URL 자동 링크
- `actions`: `{path: '...', templateShareable: false}` ← **반드시 templateShareable:false** (각 항목별 독립 템플릿)
- `FeedListItemAction`: `text`, `icon`, `key`, `press` 속성. **sap.m.Button이 아님!**
- `...` 버튼: actions 배열이 비어있지 않으면 자동 표시. 빈 배열이면 숨김 → 본인 리뷰만 메뉴 표시

### actions 데이터 흐름 (우리 코드)
```
_checkMyReview()
  → 각 리뷰 순회하며 user_user_id === 내 userId 확인
  → 본인 리뷰: oViewModel.setProperty("/Detail/reviews/results/" + i + "/_actions", [
        { Text: "수정", Icon: "sap-icon://edit", Key: "edit" },
        { Text: "삭제", Icon: "sap-icon://delete", Key: "delete" }
    ])
  → 타인 리뷰: oViewModel.setProperty(".../_actions", [])
  → FeedListItem의 actions="{path: 'viewModel>_actions', templateShareable: false}"
  → FeedListItemAction이 배열 요소마다 생성됨
  → 빈 배열 = ... 버튼 숨김, 요소 있음 = ... 버튼 표시
```

### 시도했으나 실패한 방법들
| # | 시도 | 결과 | 원인 |
|:-:|------|:--:|------|
| 1 | `<actions><Button>` | ❌ | FeedListItemAction만 허용 |
| 2 | `<toolbar><OverflowToolbar><Button>` | ❌ | sap.m.toolbar.js 404 (1.136 미지원) |
| 3 | `{path: 'viewModel>', formatter: '.formatActions'}` | ❌ | formatter 반환값 바인딩 불일치 |
| 4 | 직접 viewModel에 `_actions` 배열 세팅 | ✅ | 최종 해결책 |

### 기타
- 하이퍼링크 제거: `.detailReviewList .sapMFeedListItemTextName a { pointer-events: none; color: inherit; }`
- 볼드체: `.detailReviewList .sapMFeedListItemTextName { font-weight: bold; }`
- 리뷰 목록 박스: `sapMListBGSolid` + `detailReviewList` (border-radius 0.75rem, 흰 배경)

---

## Part 3 — 리뷰 입력: SimpleForm → FeedInput

### 📂 참고 샘플
`C:\Users\염정운\project\작성 내용\sap.m.sample.FeedInput\`

**샘플 파일 구조**:
```
webapp/
├── V.view.xml           ← FeedInput 다양한 용례 (핵심!)
├── C.controller.js       ← onPost, onActionButtonPress
├── index.html
├── manifest.json
└── Component.js
```

**샘플 XML (주요 용례)**:
```xml
<!-- 기본 -->
<FeedInput post=".onPost" showIcon="false" />

<!-- 아바타 포함 -->
<FeedInput post=".onPost" showIcon="true" icon="test-resources/sap/m/images/george_washington.jpg" />

<!-- 행 수 고정 -->
<FeedInput post=".onPost" rows="5" />

<!-- 글자수 제한 + 카운터 -->
<FeedInput post=".onPost" maxLength="20" showExceededText="true" />

<!-- 자동 확장 -->
<FeedInput post=".onPost" growing="true" />

<!-- 커스텀 액션 버튼 -->
<FeedInput post=".onPost" showIcon="false">
    <actions>
        <Button icon="sap-icon://action" press=".onActionButtonPress" />
    </actions>
</FeedInput>
```

**핵심 개념**:
- `post` 이벤트: Enter 키 또는 전송(✈) 버튼 클릭 시 발생
- `oEvent.getParameter("value")` → 입력된 텍스트 값
- `showIcon="true"` → 왼쪽 아바타 표시
- `growing="true"` → 텍스트 양에 따라 입력창 높이 자동 확장
- `maxLength` + `showExceededText` → 글자수 제한 + 카운터 표시
- `rows` → 초기 표시 행 수
- `value` → 초기값 바인딩 (수정 모드에서 기존 텍스트 prefill에 사용)
- `placeholder` → 입력창 placeholder 텍스트

### 우리 코드에 적용

**별점 처리**: FeedInput은 텍스트 입력만 담당. RatingIndicator를 FeedInput 위에 별도 HBox로 배치.
별점 행 `padding-left: 3rem` → FeedInput의 아바타 폭만큼 들여쓰기하여 시각적 정렬.

**컨트롤러 `onFeedPost`**:
```javascript
onFeedPost: function (oEvent) {
    var reviewText = oEvent.getParameter("value");      // ★ FeedInput 값
    var rating = oViewModel.getProperty("/NewReview/rating"); // ★ 별도 RatingIndicator
    // ... 기존 onSubmitReview와 동일한 백엔드 로직 (callFunction → _loadDetail)
}
```

**주의**: FeedInput은 자체 전송 버튼이 있어 기존 Submit 버튼 제거. 수정 모드에서만 취소 버튼 별도 표시.

**흰색 박스 스타일**: 리뷰 목록과 동일하게 `detailReviewWriteBox` (border-radius 0.75rem, 흰 배경, 그림자).
상단 `detailReviewWriteHeader` (흰색 배경, 하단 경계선, H4 타이틀 볼드).

---

## Part 4 — 헤더 영역 확대 (1.5배)

### 변경
| 요소 | 전 | 후 | 방식 |
|------|:--:|:--:|------|
| 제목 (The Last Signal) | 기본 | 1.9rem | `.headerTitle .sapMTitle` |
| MOVIE 텍스트 | 기본 | 1.2rem | `.headerExpanded .sapMObjStatusText` |
| 별점 아이콘 | iconSize 1rem | iconSize **1.3rem** | XML `iconSize` 직접 변경 |
| 3 → 3점 | "3" | "3점" | expression binding `{= ... + '점'}` |
| MOVIE 아이콘 | 기본 | 아래로 0.1rem | `.headerExpanded .sapMObjStatusIcon .sapUiIcon` |

### ⚠️ transform:scale 사용 금지
CSS `transform: scale(1.3)` 사용 시 별 아이콘이 시각적으로만 커지고 실제 공간은 불변 → 뒤의 "3점"과 겹침.
→ XML에서 `iconSize="1.3rem"`으로 실제 크기를 직접 변경.

### 앵커바
- 텍스트: `.sapMITBText { font-size: 1.2rem }`
- 패딩: `.sapMITBHead { padding: 0.25rem }` (0.5rem에서 축소)

---

## Part 5 — 출연진 표 스타일링

### 변경
| 요소 | 전 | 후 |
|------|------|------|
| 컬럼 헤더 배경 | 회색 | 흰색 (#fff) |
| 컬럼 하단 구분선 | 회색 | 검은색 (1px solid #000) |
| 마지막 열 우측 구분선 | 회색 | 검은색 (1px solid #000) |
| 컬럼 정렬 | 비활성 | sortProperty 활성화 |
| 데이터 셀 텍스트 | 기본 | 1.1rem |
| 헤더 폰트 | 기본 | 1.2rem bold |

### ExportCast 버튼 구조 변경
**문제**: 버튼이 `<t:extension>` → `<OverflowToolbar>` 안에 있어 margin-bottom이 안 먹힘.

**해결**: ExportInfo와 동일하게 별도 HBox로 분리.
```xml
<!-- 변경 전 -->
<t:Table>
    <t:extension><OverflowToolbar><Button>Export Cast</Button></OverflowToolbar></t:extension>
</t:Table>

<!-- 변경 후 -->
<VBox>
    <HBox class="sapUiSmallMarginBottom" justifyContent="End">
        <Button>Export Cast</Button>
    </HBox>
    <t:Table>...</t:Table>
</VBox>
```
ExportInfo/ExportCast 버튼 1.3배 확대 (`.detailExportBtn`).

---

## Part 6 — 섹션 타이틀 및 간격 조정

### SubSection 타이틀 숨김
- `.sapUxAPObjectPageSectionHeaderToolbar { display: none !important; }`
- ⚠️ SubSection 전체가 아닌 **toolbar만** 숨겨야 함 (전체 숨기면 내용도 사라짐)
- 커스텀 헤더(`detailReviewWriteHeader`)로 대체

### 리뷰 섹션 구조 (중요!)
**현재 구조 (SubSection 3개)**:
```
uxap:ObjectPageSection title="리뷰"    ← 이 타이틀이 유지되어야 함!
  ├─ SubSection 1: MessageStrip        ← visible when HasReviewed
  ├─ SubSection 2: 리뷰 목록 (title="리뷰 목록")
  └─ SubSection 3: 리뷰 작성 (title="리뷰 작성", visible when !HasReviewed)
```

**왜 3개인가?** SAPUI5 ObjectPageLayout 규칙:
- SubSection 2개 이상 visible → 섹션 타이틀("리뷰") 유지 + 앵커바 드롭다운
- SubSection 1개만 visible → 섹션 타이틀이 SubSection 타이틀로 대체됨 ("리뷰 목록"으로 바뀜)

**리뷰 작성 후**: SubSection 1(MessageStrip) + 2(리뷰 목록) = 2개 visible → "리뷰" 유지 ✅
**리뷰 작성 전**: SubSection 2(리뷰 목록) + 3(리뷰 작성) = 2개 visible → "리뷰" 유지 ✅

**단점**: SubSection 1에 `title`이 없어 앵커바 드롭다운에 빈 `<li>` 하나 생성됨. SAPUI5 한계로 수용.

### 시도했으나 실패한 구조
| 구조 | SubSection 수 | 리뷰 작성 후 visible | 결과 |
|------|:--:|:--:|------|
| MessageStrip 포함 1개 + 리뷰 작성 1개 | 2 | 1개 | "리뷰 목록"으로 대체 ❌ |
| MessageStrip + 리스트 + 폼 모두 1개 | 1 | 1개 | "리뷰 목록"으로 대체 ❌ |
| 별도 3개 | 3 | 2개 | "리뷰" 유지 ✅ |

### 간격 조정
- 섹션 헤더 `padding-top`: 1.8rem → 1.2rem
- 제목 ↔ 확장 콘텐츠: `margin-bottom: 0.5rem`
- 앵커바 padding: 0.5rem → 0.25rem
- Type/Genre 행: `detailCardRowSpaced` (margin-bottom 0.65rem) → Provider/Tags보다 넓게

---

## Part 7 — 기타 수정 사항

### 리뷰 수 괄호 제거
`formatReviewCount`에서 `"(" + ... + ")"` → `...` 로 변경. "리뷰 2개"로 표시.

### `label.contentId` i18n 키 복원
카드 재구성 과정에서 삭제되었던 키를 다시 추가. 한글: "컨텐츠 코드", 영문: "Content Code".

### i18n 신규 키
| 키 | 한글 | 영문 |
|------|------|------|
| card.reviews | 리뷰 | Reviews |
| card.partner | 파트너 | Partner |
| label.partnerCode | 제공사 코드 | Provider Code |
| label.avgRating | 평균 별점 | Average Rating |
| label.reviewCount | 리뷰 수 | Review Count |

---

## 📝 수정 파일 목록

| # | 파일 | 변경 |
|:--:|------|------|
| 1 | `detailPage/view/DetailMain.view.xml` | 카드 4그룹, FeedListItem+FeedInput, 헤더 확대, ExportCast 분리, HBox 래퍼 |
| 2 | `detailPage/controller/DetailMain.controller.js` | formatActions, onFeedPost, formatReviewCount, _actions 세팅 |
| 3 | `detailPage/css/style.css` | 카드/리뷰 박스, 헤더/앵커바, 표 스타일, 간격, min-width 7rem |
| 4 | `detailPage/i18n/i18n.properties` | 신규 키 5개, 불필요 키 제거, contentId 복원 |
| 5 | `detailPage/i18n/i18n_en.properties` | 영문 동기화 |

---

## 🧠 절대 잊지 말 것 (오늘의 핵심 교훈)

| # | 내용 |
|---|------|
| 1 | **Flex margin**: 직접 flex child(HBox 등)에 걸 것. span 내부는 무시됨 |
| 2 | **FeedListItemAction**: `sap.m.Button` ❌, `templateShareable: false` 필수 |
| 3 | **FeedListItem convertLinksToAnchorTags**: boolean ❌, enum `"All"` ✅ |
| 4 | **FeedInput post**: `oEvent.getParameter("value")`로 값 획득 |
| 5 | **transform:scale vs iconSize**: 전자는 공간 불변 → 겹침, 후자는 실제 크기 변경 |
| 6 | **ObjectPageSection Rule**: SubSection 2개 미만 visible → 섹션 타이틀이 SubSection 타이틀로 대체 |
| 7 | **H5/H6**: 브라우저 기본 margin-top 차이 → 동일 태그로 통일 |
| 8 | **sapUiSmallMargin vs sapUiSmallMarginTop**: 전자는 전방향, 후자는 top만 |
| 9 | **min-width 라벨 정렬**: 모든 라벨 동일 폭(7rem) → `:` 이후 값 위치 통일 |
| 10 | **SubSection 타이틀 숨기기**: toolbar(`.sapUxAPObjectPageSubSectionHeaderToolbar`)만 숨기기 |

---

> **세션 일시**: 2026-07-14
> **커밋**: `feat: detailPage 카드 재구성, FeedListItem/FeedInput 적용, 전체 UI 개선`
> **이전 기록**: `컴팩트_2026-07-13.md`
> **샘플 파일**: `C:\Users\염정운\project\작성 내용\sap.m.sample.FeedListItem\`, `C:\Users\염정운\project\작성 내용\sap.m.sample.FeedInput\`
