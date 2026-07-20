# CustomData (`sap.ui.core.CustomData`)

> 컨트롤에 임의의 key-value 데이터를 첨부

---

## 기본 사용법
```xml
<Button text="Select" press="onSelectPlan">
    <customData>
        <core:CustomData key="planCode" value="{viewModel>code}" />
    </customData>
</Button>
```

## 값 읽기
```javascript
onSelectPlan: function (oEvent) {
    var sPlanCode = oEvent.getSource().getCustomData()[0].getValue();
    // 또는
    var sPlanCode = oEvent.getSource().data("planCode");
}
```

## 주요 속성
| 속성 | 설명 |
|------|------|
| `key` | 데이터 키 |
| `value` | 데이터 값 (바인딩 가능) |
| `writeToDom` | DOM에 기록 여부 |

## 용도
- 목록에서 클릭된 항목 식별 (plan code, content ID 등)
- 컨트롤에 부가 메타데이터 첨부

## 사용된 곳
- **PlanChange**: 플랜 카드의 planCode 식별
