# MessageStrip & ObjectIdentifier (`sap.m`)

> MessageStrip: 인라인 알림 바. ObjectIdentifier: 객체 식별 텍스트

---

## MessageStrip
```xml
<MessageStrip
    text="이미 리뷰를 작성하셨습니다"
    type="Information"
    showIcon="true"
    visible="{= !!${viewModel>/HasReviewed} }" />
```

| 속성 | 값 | 설명 |
|------|-----|------|
| `type` | `Information` / `Warning` / `Error` / `Success` | 색상 |
| `showIcon` | boolean | 왼쪽 아이콘 표시 |
| `text` | string | 메시지 텍스트 |

### ObjectPageSection 내 배치
MessageStrip을 ObjectPageSubSection으로 분리해야 섹션 타이틀이 유지됨:
```xml
<uxap:ObjectPageSection title="리뷰">
    <uxap:subSections>
        <!-- SubSection 1: MessageStrip -->
        <uxap:ObjectPageSubSection visible="{= !!${viewModel>/HasReviewed} }">
            <MessageStrip ... />
        </uxap:ObjectPageSubSection>
        <!-- SubSection 2: 리뷰 목록 -->
        <uxap:ObjectPageSubSection>...</uxap:ObjectPageSubSection>
        <!-- SubSection 3: 리뷰 작성 -->
        <uxap:ObjectPageSubSection>...</uxap:ObjectPageSubSection>
    </uxap:subSections>
</uxap:ObjectPageSection>
```
> SubSection 2개 이상이어야 섹션 타이틀이 유지됨 (SAPUI5 한계)

---

## ObjectIdentifier
```xml
<ObjectIdentifier
    title="C001"
    text="Content Code" />
```
- `title`: 굵은 텍스트
- `text`: 작은 회색 설명

## 사용된 곳
- **detailPage**: 리뷰 안내 MessageStrip
