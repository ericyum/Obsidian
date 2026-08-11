# CSS 핵심 팁 (SAPUI5 스타일링)

> 프로젝트에서 발견한 CSS 이슈와 해결법

---

## 1. 선택자 공백 vs 붙임
```css
/* ❌ .foo 안의 자식 .bar를 찾음 */
.sapMBarChild .sapMBtn { }

/* ✅ .foo이면서 .bar인 요소 */
.sapMBarChild.sapMBtn { }
```

## 2. Flex margin
flex item에 margin을 주려면 **직접 flex child**에 걸어야 함:
```css
/* ❌ span은 내부 wrapper에 막힘 */
.detailCardValue { margin-left: 1rem; }

/* ✅ HBox가 직접 flex child */
<HBox class="sapUiSmallMarginBegin"><Text ... /></HBox>
```

## 3. transform:scale vs iconSize
```css
/* ❌ 시각적 확대만, 공간 불변 → 겹침 */
.rating { transform: scale(1.5); }

/* ✅ 실제 크기 변경 */
<RatingIndicator iconSize="1.3rem" />
```

## 4. 버튼 둥글게 (sapMBtn + sapMBtnInner 둘 다)
```css
.myBtn.sapMBtn { border-radius: 1.5rem !important; }
.myBtn.sapMBtn .sapMBtnInner { border-radius: 1.5rem !important; }
```

## 5. ObjectPageLayout 관련
```css
/* 숨겨진 섹션 헤더 공간 제거 */
.sapUxAPObjectPageSectionHeaderHidden {
    height: 0 !important; min-height: 0 !important;
}
/* SubSection 타이틀만 숨기기 (전체 ❌) */
.sapUxAPObjectPageSubSectionHeaderToolbar {
    display: none !important;
}
/* SubSection overflow 해제 (버튼 잘림 방지) */
.sapUxAPBlockContainer { overflow: visible !important; }
```

## 6. sapMTextMaxWidth 제한 해제
```css
.sapMTextMaxWidth {
    max-width: none !important;
    width: 100% !important;
}
```

## 7. ResponsiveGridLayout → Flex 강제 전환
```css
.sapUiFormResGridCont {
    display: flex !important;
    flex-wrap: wrap !important;
}
.sapUiFormResGridCont > .sapUiRespGridSpanL6 {
    flex: 0 0 50% !important;
}
```

## 8. `sapUiContentPadding` — Popover/레이아웃 패딩
SAP 공식 패딩 클래스. VerticalLayout과 함께 사용:
```javascript
var oLayout = new VerticalLayout({ width: "100%" });
oLayout.addStyleClass("sapUiContentPadding");
```

## 9. sapUxAPSubSectionSeeMoreContainer — CSS 불가
SAPUI5 내부 컨테이너. `display: none`도, `border-top`도 안 먹힘.
→ **구조 변경으로 우회** (버튼을 이전 서브섹션에 통합)

## 10. `addStyleClass`는 메서드
```javascript
// ❌ 생성자 속성 아님
new Control({ addStyleClass: "..." })

// ✅ 생성 후 호출
var ctrl = new Control({ ... });
ctrl.addStyleClass("...");
```
