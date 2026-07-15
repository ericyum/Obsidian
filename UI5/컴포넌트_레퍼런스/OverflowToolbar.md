# OverflowToolbar & ToolbarSpacer (`sap.m`)

> 오버플로우 대응 툴바. 공간 부족 시 `...` 메뉴로 버튼 접힘

---

## 기본 구조
```xml
<OverflowToolbar>
    <ToolbarSpacer />
    <Button text="Change Plan" type="Emphasized" />
    <Button text="Cancel" type="Default" />
</OverflowToolbar>
```

## ToolbarSpacer
- 남은 공간을 모두 차지하는 스페이서
- 버튼을 우측 정렬할 때 앞에 배치

## 주의사항
- 툴바 내부 버튼에는 고정 높이가 적용됨 → 키우려면 `.sapMOTB`에 `height: auto`, `min-height` 설정 필요
- ObjectPageLayout actions 영역에서도 OverflowToolbar가 자동 사용됨
- `sapMTBHiddenElement`(숨겨진 오버플로우 클론 버튼)이 있으므로 CSS 선택자 주의

## 사용된 곳
- **myPage** (초기): 버튼 툴바 → 카드 박스 내 HBox로 대체됨
- **PlanChange**: 제출 버튼 영역 (현재 유지)
- **detailPage**: 헤더 actions (자동)
