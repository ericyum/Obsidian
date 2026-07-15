# MessageBox & MessageToast (`sap.m`)

> 사용자 알림/확인을 위한 다이얼로그

---

## MessageBox.warning — 확인 다이얼로그
```javascript
MessageBox.warning(sMessage, {
    title: sTitle,
    actions: [MessageBox.Action.OK, MessageBox.Action.CANCEL],
    emphasizedAction: MessageBox.Action.OK,
    contentWidth: "30rem",
    onClose: function (sResult) {
        if (sResult === MessageBox.Action.OK) {
            // 확인 로직
        }
    }
});
```

### 주요 옵션
| 옵션 | 설명 |
|------|------|
| `title` | 다이얼로그 제목 (짧게) |
| `actions` | 버튼 배열 |
| `emphasizedAction` | 강조 버튼 (파란색) |
| `contentWidth` | 너비 |
| `onClose` | 닫힐 때 콜백. `sResult`로 어떤 버튼 눌렀는지 판단 |

### vs 커스텀 Dialog
| | MessageBox | Dialog |
|------|:--:|:--:|
| 코드량 | 적음 (~10줄) | 많음 (~20줄) |
| Fiori 표준 | ✅ | 커스텀 |
| 아이콘 | 경고 삼각형 자동 | 수동 설정 |
| 버튼 연타 가드 | 자체 처리 | `_bActionCalled` 필요 |

### MessageBox 텍스트 스타일링
```css
.sapMMessageBox .sapMMsgBoxText {
    white-space: normal !important;
    word-break: keep-all !important;
}
.sapMMessageBox {
    min-width: 24rem !important;
    max-width: 90vw !important;
}
```

---

## MessageToast
```javascript
MessageToast.show("처리되었습니다.");
```
- 화면 하단에 잠시 나타나는 토스트 메시지
- 간단한 성공/실패 알림에 사용

---

## 사용된 곳
- **myPage**: 구독 해지/재개 확인 (MessageBox.warning)
- **detailPage**: 리뷰 삭제 확인 (MessageBox.warning)
- **전체**: 에러/성공 알림 (MessageToast)
