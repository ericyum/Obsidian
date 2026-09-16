# SimpleForm & ResponsiveGridLayout (`sap.ui.layout.form`)

> 폼 전용 레이아웃. 라벨 자동 정렬, 반응형 그리드, 편집 모드 제공

---

## 기본 구조
```xml
<form:SimpleForm
    layout="ResponsiveGridLayout"
    editable="false"
    labelSpanL="4" labelSpanM="4"
    emptySpanL="0" emptySpanM="0"
    columnsL="2" columnsM="2">
    <form:content>
        <VBox>
            <Label text="플랜" design="Bold" />
            <Text text="프리미엄" />
            <layoutData>
                <layout:GridData span="L6 M6 S12" />
            </layoutData>
        </VBox>
    </form:content>
</form:SimpleForm>
```

## 주요 속성
| 속성 | 설명 |
|------|------|
| `layout` | 레이아웃 방식. 보통 `ResponsiveGridLayout` |
| `editable` | 편집 모드 여부 |
| `labelSpanL/M` | Large/Medium 화면에서 라벨이 차지하는 컬럼 수 |
| `columnsL/M` | Large/Medium 화면에서 전체 컬럼 수 |
| `emptySpanL/M` | 빈 컬럼 수 |

## VBox와의 비교
SimpleForm은 **폼 전용**으로, VBox보다:
- 라벨 자동 정렬
- 반응형 열 배치 (GridData span)
- editable 모드 지원
- `form:content` aggregation 슬롯 사용

## ResponsiveGridLayout 동작 방식

SAPUI5 1.136에서는 **float 기반**으로 렌더링되어 `display: block` + `float: left` 사용.
→ **Flexbox로 강제 전환**이 필요한 경우:
```css
.sapUiFormResGridCont {
    display: flex !important;
    flex-wrap: wrap !important;
    align-items: flex-start !important;
}
.sapUiFormResGridCont > .sapUiRespGridSpanL6 {
    flex: 0 0 50% !important;
    width: 50% !important;
}
```

## GridData span
```xml
<layoutData>
    <layout:GridData span="L6 M6 S12" />
</layoutData>
```
- `L`: Large 화면 (12컬럼 기준)
- `M`: Medium 화면
- `S`: Small/Phone 화면

## 사용된 곳
- **myPage**: MyMembership 멤버십 정보 폼
