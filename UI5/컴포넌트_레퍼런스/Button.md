# Button (`sap.m.Button`)

> 액션을 실행하는 클릭 가능한 버튼

---

## 기본 사용법
```xml
<Button
    text="확인"
    type="Emphasized"
    icon="sap-icon://accept"
    press="onConfirm"
    visible="{= ${viewModel>/status} === 'active' }" />

<Button
    icon="sap-icon://hint"
    type="Transparent"
    press="onInfo"
    tooltip="자세히 보기" />
```

## 주요 속성
| 속성 | 값 | 설명 |
|------|-----|------|
| `text` | string | 버튼 텍스트 |
| `type` | `Default` / `Emphasized` / `Transparent` / `Accept` / `Reject` | 버튼 스타일 |
| `icon` | `sap-icon://...` | 아이콘 (텍스트 왼쪽) |
| `press` | function | 클릭 핸들러 |
| `enabled` | boolean | 비활성화 |
| `visible` | boolean/expression | 가시성 |

## 둥근 버튼 스타일링
```css
.myBtn.sapMBtn {
    height: 2.75rem !important;
    border-radius: 1.5rem !important;   /* 알약형 */
}
.myBtn.sapMBtn .sapMBtnInner {
    height: 2.75rem !important;
    padding: 0 1.5rem !important;
    border-radius: 1.5rem !important;   /* inner도 동일하게 */
}
.myBtn.sapMBtn .sapMBtnContent {
    font-size: 1.15rem !important;
    line-height: 2.75rem !important;    /* 높이와 일치시켜 수직 중앙 */
}
```

**⚠️ 중요**: `sapMBtn`과 `sapMBtnInner` **둘 다** `border-radius`를 줘야 함. inner가 실제 배경을 가지기 때문.

## CSS 선택자 주의
```css
/* ❌ sapMBarChild 안의 자식 sapMBtn을 찾음 → 같은 요소면 매칭 안 됨 */
.sapMBarChild .sapMBtn { }

/* ✅ sapMBarChild이면서 sapMBtn인 요소 → 정확히 매칭 */
.sapMBarChild.sapMBtn { }
```

## 아이콘만 있는 버튼
```xml
<Button
    icon="sap-icon://message-information"
    type="Transparent"
    press="onInfo"
    class="membershipInfoBtn"
    tooltip="자세히 보기" />
```

## 사용된 곳
- **detailPage**: 뒤로가기, 엑셀 내보내기, 리뷰 제출, FeedListItemAction
- **myPage**: 뒤로가기, Info 버튼, 플랜 변경/해지/재개/시작
- **PlanChange**: Select, Change Plan
