# FeedListItem & FeedListItemAction (`sap.m`)

> 소셜 피드 스타일의 목록 아이템. 아바타 + 작성자 + 본문 + 액션 메뉴

---

## 기본 구조
```xml
<List items="{viewModel>/Detail/reviews/results}">
    <FeedListItem
        sender="{viewModel>user/user_name}"
        text="{viewModel>review_text}"
        timestamp="{ path: 'viewModel>createdAt', type: 'sap.ui.model.odata.type.DateTime', formatOptions: { pattern: 'yyyy-MM-dd' } }"
        icon="sap-icon://person-placeholder"
        info="{= ${viewModel>rating} + ' / 5'}"
        showIcon="true"
        convertLinksToAnchorTags="All"
        actions="{path: 'viewModel>_actions', templateShareable: false}">
        <FeedListItemAction text="{viewModel>Text}" icon="{viewModel>Icon}" key="{viewModel>Key}" press=".onFeedAction" />
    </FeedListItem>
</List>
```

## 주요 속성
| 속성 | 설명 |
|------|------|
| `sender` | 작성자 이름 (볼드체 링크) |
| `text` | 본문 텍스트 |
| `timestamp` | 날짜/시간 (푸터 오른쪽) |
| `icon` | 아바타 이미지. 없으면 `sap-icon://person-placeholder` |
| `info` | 부가 정보 (푸터 왼쪽). 별점 등 |
| `showIcon` | 아바타 표시 여부 |
| `convertLinksToAnchorTags` | URL 자동 링크. **boolean ❌, enum `"All"` ✅** |
| `actions` | 액션 배열 바인딩 |

---

## FeedListItemAction
```xml
<FeedListItemAction
    text="{viewModel>Text}"
    icon="{viewModel>Icon}"
    key="{viewModel>Key}"
    press=".onFeedAction" />
```

**⚠️ 주의**:
- `sap.m.Button`이 아님! 반드시 `FeedListItemAction` 사용
- `templateShareable: false` 필수 — 각 항목별 독립 템플릿

## 액션 메뉴 (`...` 버튼)
- `actions` 배열이 비어있으면 `...` 버튼 숨김
- 요소가 있으면 `...` 버튼 표시 → 클릭 시 액션 메뉴 팝업

### actions 데이터 세팅 (본인 리뷰만)
```javascript
// 본인 리뷰
oViewModel.setProperty("/Detail/reviews/results/" + i + "/_actions", [
    { Text: "Edit", Icon: "sap-icon://edit", Key: "edit" },
    { Text: "Delete", Icon: "sap-icon://delete", Key: "delete" }
]);
// 타인 리뷰
oViewModel.setProperty("/Detail/reviews/results/" + i + "/_actions", []);
```

## 하이퍼링크 제거
FeedListItem의 sender는 자동으로 링크가 생성됨:
```css
.detailReviewList .sapMFeedListItemTextName a {
    text-decoration: none !important;
    color: inherit !important;
    pointer-events: none !important;
}
```

## 사용된 곳
- **detailPage**: 리뷰 목록
