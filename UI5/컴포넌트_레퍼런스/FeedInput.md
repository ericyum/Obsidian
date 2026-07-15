# FeedInput (`sap.m.FeedInput`)

> 소셜 피드 스타일의 텍스트 입력. 아바타 + 자동확장 + 전송 버튼

---

## 기본 사용법
```xml
<FeedInput
    post=".onFeedPost"
    showIcon="true"
    icon="sap-icon://person-placeholder"
    maxLength="1000"
    showExceededText="true"
    growing="true"
    rows="3"
    value="{viewModel>/NewReview/review_text}"
    placeholder="리뷰를 입력하세요" />
```

## 주요 속성
| 속성 | 설명 |
|------|------|
| `post` | 전송 핸들러 (Enter 또는 ✈ 버튼) |
| `showIcon` | 왼쪽 아바타 표시 |
| `icon` | 아바타 아이콘 URL |
| `maxLength` | 최대 글자수 |
| `showExceededText` | 글자수 초과 시 카운터 표시 |
| `growing` | 텍스트 양에 따라 높이 자동 확장 |
| `rows` | 초기 표시 행 수 |
| `value` | 초기값 (수정 모드 prefill) |
| `placeholder` | placeholder 텍스트 |

## 값 획득
```javascript
onFeedPost: function (oEvent) {
    var sValue = oEvent.getParameter("value");  // ★ 입력된 텍스트
    // ...
}
```

## SimpleForm 대체 시 주의
- FeedInput은 텍스트 입력만 담당 → **별점(RatingIndicator)은 별도 배치**
- 별점 행에 `padding-left: 3rem` → FeedInput 아바타 폭만큼 들여쓰기

```xml
<VBox>
    <HBox alignItems="Center">
        <Label text="별점" />
        <RatingIndicator value="{viewModel>/NewReview/rating}" maxValue="5" iconSize="1.5rem" />
    </HBox>
    <FeedInput post=".onFeedPost" showIcon="true" ... />
</VBox>
```

## 사용된 곳
- **detailPage**: 리뷰 작성/수정 폼
