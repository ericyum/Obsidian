# RatingIndicator (`sap.m.RatingIndicator`)

> 별점 표시/입력 컨트롤

---

## 기본 사용법
```xml
<!-- 표시 전용 -->
<RatingIndicator
    value="{viewModel>/Detail/avg_rating}"
    maxValue="5"
    enabled="false"
    iconSize="1.3rem" />

<!-- 입력 가능 -->
<RatingIndicator
    value="{viewModel>/NewReview/rating}"
    maxValue="5"
    iconSize="1.5rem" />
```

## 주요 속성
| 속성 | 설명 |
|------|------|
| `value` | 현재 별점 값 |
| `maxValue` | 최대 별점 (보통 5) |
| `enabled` | `false` → 읽기 전용 (표시만) |
| `iconSize` | 별 아이콘 크기. 예: `"1.3rem"` |

## ⚠️ transform:scale 사용 금지
```css
/* ❌ 시각적으로만 커지고 공간 불변 → 뒤 요소와 겹침 */
.rating { transform: scale(1.5); }

/* ✅ 실제 크기 변경 */
<RatingIndicator iconSize="1.3rem" />
```

## FeedInput과 함께 사용
별점은 FeedInput 위에 별도 HBox로 배치:
```xml
<HBox alignItems="Center">
    <Label text="별점" />
    <RatingIndicator value="..." maxValue="5" iconSize="1.5rem" />
</HBox>
<FeedInput post=".onFeedPost" ... />
```

## 사용된 곳
- **detailPage**: 헤더 별점(읽기 전용), 카드 평균 별점(읽기 전용), 리뷰 작성 별점(입력)
