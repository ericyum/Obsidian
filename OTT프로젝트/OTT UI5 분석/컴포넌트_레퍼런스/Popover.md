# Popover (`sap.m.Popover`)

> 특정 컨트롤 옆에 나타나는 말풍선 형태의 팝업

---

## 기본 사용법
```javascript
var oPopover = new Popover({
    title: "제목",
    content: [new Text({ text: "내용" })],
    contentWidth: "22rem"
});
oPopover.openBy(oEvent.getSource());
```

## 주요 속성
| 속성 | 설명 |
|------|------|
| `title` | 팝오버 상단 제목 |
| `content` | 컨텐츠 (컨트롤 배열) |
| `footer` | 하단 버튼 영역 |
| `contentWidth` | 컨텐츠 너비. 예: `"22rem"` |
| `customClassName` | 커스텀 CSS 클래스 |

## 동적 컨텐츠 변경 (재사용)
```javascript
if (!this._oInfoPopover) {
    this._oInfoPopover = new Popover({ ... });
} else {
    // 기존 인스턴스 재사용하여 컨텐츠만 변경
    this._oInfoPopover.setTitle(sNewTitle);
    this._oInfoPopover.removeAllContent();
    this._oInfoPopover.addContent(new Text({ text: sNewBody }));
}
this._oInfoPopover.openBy(oEvent.getSource());
```

---

## ⚠️ 패딩 문제 (중요!)

**문제**: `Text`만 content에 넣으면 패딩이 전혀 적용되지 않음.

**원인**: `sap.m.Popover`의 content 영역은 자체적으로 패딩을 제공하지 않음.

**해결**: `sap.ui.layout.VerticalLayout` + `sapUiContentPadding` 사용:
```javascript
var oLayout = new VerticalLayout({ width: "100%" });
oLayout.addStyleClass("sapUiContentPadding");
oLayout.addContent(new Text({ text: sBody }));

var oPopover = new Popover({
    title: sTitle,
    content: [oLayout],
    contentWidth: "22rem"
});
```

`addStyleClass`는 **메서드**이므로 생성자 속성으로 전달 불가.

---

## Footer (Close 버튼)
```javascript
footer: new Button({
    text: "닫기",
    press: function () {
        this._oInfoPopover.close();
    }.bind(this)
})
```
- 보통 불필요 — 바깥 클릭으로 자동 닫힘
- 필요한 경우에만 추가

---

## 사용된 곳
- **myPage**: Info Popover (동시시청, 기간 설명)
