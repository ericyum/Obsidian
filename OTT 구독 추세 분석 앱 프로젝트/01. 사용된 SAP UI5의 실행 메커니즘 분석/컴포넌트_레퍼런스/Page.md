# Page (`sap.m.Page`)

> 기본 페이지 레이아웃. 상단 바 + 컨텐츠 영역 제공

---

## 기본 구조
```xml
<Page
    showNavButton="true"
    navButtonPress="onNavBack"
    title="페이지 제목">
    <content>
        <!-- 내용 -->
    </content>
</Page>
```

## 주요 속성
| 속성 | 설명 |
|------|------|
| `title` | 상단 바 타이틀 |
| `showNavButton` | 뒤로가기 버튼 표시 |
| `navButtonPress` | 뒤로가기 버튼 클릭 핸들러 |
| `showHeader` | 헤더 표시 여부 |
| `showFooter` | 푸터 표시 여부 |

## ObjectPageLayout과의 차이
| | `sap.m.Page` | `sap.uxap.ObjectPageLayout` |
|------|:--:|:--:|
| 헤더 | 얇은 바 | DynamicHeader (snapped/expanded) |
| 앵커바 | 없음 | 있음 |
| 섹션 | 직접 구성 | Section/SubSection 체계 |
| 스타일링 | 제한적 (CSS override 어려움) | 유연함 |
| 용도 | 단순 페이지 | Fiori 상세 페이지 |

## ⚠️ 주의
- **CSS override가 어려움**: SAPUI5 1.136에서 `.sapMPage` 관련 CSS 선택자가 잘 먹히지 않음
- **ObjectPageLayout으로 전환 권장**: 더 나은 헤더와 섹션 구조가 필요하면 ObjectPageLayout 사용

## 사용된 곳
- **PlanChange** (초기): Page → ObjectPageLayout으로 전환됨
