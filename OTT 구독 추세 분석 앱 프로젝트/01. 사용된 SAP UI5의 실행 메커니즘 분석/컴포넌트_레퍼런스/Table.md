# Table (`sap.ui.table.Table`)

> 데이터 테이블. 정렬·열고정·다중 선택 지원

---

## 기본 구조
```xml
<t:Table
    id="castTable"
    class="detailCastTable"
    rows="{viewModel>/Detail/casts/results}"
    visibleRowCount="{= Math.min(${viewModel>/Detail/casts/results}.length, 15) }"
    enableSelectAll="false"
    selectionMode="None"
    enableColumnFreeze="true"
    width="auto">
    <t:columns>
        <t:Column hAlign="Left" width="12rem" sortProperty="name">
            <Label text="Name" />
            <t:template>
                <Text wrapping="false" text="{viewModel>name}" />
            </t:template>
        </t:Column>
    </t:columns>
</t:Table>
```

## 주요 속성
| 속성 | 설명 |
|------|------|
| `rows` | 데이터 바인딩 |
| `visibleRowCount` | 표시 행 수. `Math.min(.length, 15)`로 동적 조정 가능 |
| `selectionMode` | `None` / `Single` / `Multi` |
| `enableColumnFreeze` | 열 고정 가능 여부 |
| `enableSelectAll` | 전체 선택 체크박스 |

## Column
| 속성 | 설명 |
|------|------|
| `hAlign` | 가로 정렬 |
| `width` | 컬럼 폭 |
| `sortProperty` | 정렬 기준 필드 (설정 시 정렬 활성화) |

## t:extension — 툴바
```xml
<t:Table>
    <t:extension>
        <OverflowToolbar>
            <ToolbarSpacer />
            <Button text="Export" />
        </OverflowToolbar>
    </t:extension>
</t:Table>
```
⚠️ `t:extension` 내부의 버튼은 margin이 잘 안 먹힘 → 별도 HBox로 분리 권장.

## 스타일링
```css
.detailCastTable .sapUiTableContentCell .sapMText {
    font-size: 1.1rem !important;
}
.detailCastTable .sapUiTableHeaderCell {
    background: #fff !important;
    border-bottom: 1px solid #000 !important;
}
```

## 사용된 곳
- **detailPage**: 출연진 표
