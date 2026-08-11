# VerticalLayout (`sap.ui.layout.VerticalLayout`)

> 수직 배치 레이아웃. `sapUiVlt` + `sapUiContentPadding` 렌더링

---

## VBox와의 차이

| | `sap.ui.layout.VerticalLayout` | `sap.m.VBox` |
|------|------|------|
| 렌더링 | `<div class="sapUiVlt sapuiVlt">` | `<div class="sapMFlexBox sapMVBox">` |
| `sapUiContentPadding` | ✅ 자동 지원 | ❌ 직접 추가 필요 |
| SAP 공식 샘플 | ✅ 사용 | 잘 사용 안 함 |
| API | `addContent()` | `addItem()` |

## 주요 용도: Popover 패딩

SAP 공식 샘플에서 Popover/Dialog 내부 콘텐츠 패딩에 사용:
```javascript
var oLayout = new VerticalLayout({ width: "100%" });
oLayout.addStyleClass("sapUiContentPadding");
oLayout.addContent(new Text({ text: sBody }));
```

`addStyleClass`는 **생성자 속성이 아니라 메서드**이므로 별도 호출 필요:
```javascript
// ❌ 생성자에서 안 됨
new VerticalLayout({ addStyleClass: "sapUiContentPadding" })

// ✅ 생성 후 호출
var oLayout = new VerticalLayout({ width: "100%" });
oLayout.addStyleClass("sapUiContentPadding");
```

---

## 사용된 곳
- **myPage**: Info Popover 내부 콘텐츠 패딩
