# ObjectPageLayout & ObjectPageDynamicHeaderTitle (`sap.uxap`)

> Fiori 마스터-디테일 패턴의 표준 상세 페이지 레이아웃

---

## ObjectPageLayout

### 기본 구조
```xml
<uxap:ObjectPageLayout
    upperCaseAnchorBar="false"
    showFooter="false"
    toggleHeaderOnTitleClick="true">

    <uxap:headerTitle> ... </uxap:headerTitle>
    <uxap:sections>
        <uxap:ObjectPageSection>
            <uxap:subSections>
                <uxap:ObjectPageSubSection> ... </uxap:ObjectPageSubSection>
            </uxap:subSections>
        </uxap:ObjectPageSection>
    </uxap:sections>
</uxap:ObjectPageLayout>
```

### 주요 속성
| 속성 | 설명 |
|------|------|
| `upperCaseAnchorBar` | 앵커바 텍스트 대문자 변환 여부. 보통 `false` |
| `showFooter` | 하단 푸터 표시 여부. 보통 `false` |
| `toggleHeaderOnTitleClick` | 타이틀 클릭 시 헤더 확장/축소 |

### Aggregation (슬롯)
| 슬롯 | 태그 | 설명 |
|------|------|------|
| `headerTitle` | `<uxap:headerTitle>` | 상단 헤더 영역 |
| `headerContent` | `<uxap:headerContent>` | 헤더 아래 확장 콘텐츠 |
| `sections` | `<uxap:sections>` | 본문 섹션들 |
| `footer` | `<uxap:footer>` | 하단 푸터 |

### 주의사항
- **`DynamicPage`와 중첩 금지**: ObjectPageLayout 자체가 스크롤 컨테이너를 가지므로 `DynamicPage` 안에 넣으면 spacer div 무한 확장 버그 발생
- **단독으로 루트에 사용**: `ObjectPageLayout`을 뷰의 최상단에 배치

---

## ObjectPageDynamicHeaderTitle

### 4개 영역
| 영역 | XML 태그 | 표시 시점 |
|------|----------|:--------:|
| **heading** | `<uxap:heading>` | **항상** (헤더 메인 타이틀) |
| **expandedContent** | `<uxap:expandedContent>` | 스크롤 최상단에서만 |
| **snappedContent** | `<uxap:snappedContent>` | 스크롤 내리면 축약 표시 |
| **actions** | `<uxap:actions>` | 항상 (우측 액션 버튼) |

### 예시
```xml
<uxap:ObjectPageDynamicHeaderTitle>
    <uxap:heading>
        <HBox alignItems="Center" class="membershipHeader">
            <Title text="Kim Minjun" level="H2" />
            <ObjectStatus text="Active" state="Success" />
        </HBox>
    </uxap:heading>
    <uxap:expandedContent>
        <ObjectStatus text="MOVIE" icon="sap-icon://video" />
    </uxap:expandedContent>
    <uxap:snappedContent>
        <ObjectStatus text="MOVIE" />
    </uxap:snappedContent>
    <uxap:actions>
        <Button icon="sap-icon://nav-back" text="Back" press="onNavBack" type="Transparent" />
    </uxap:actions>
</uxap:ObjectPageDynamicHeaderTitle>
```

---

## ObjectPageSection & ObjectPageSubSection

### ObjectPageSection
| 속성 | 설명 |
|------|------|
| `title` | 섹션 제목 (앵커바에 표시됨) |
| `titleUppercase` | 제목 대문자 변환. 보통 `false` |
| `showTitle` | 제목 표시 여부 |

### ObjectPageSubSection
- 섹션 내부의 하위 영역
- **SubSection 2개 미만일 때**: 섹션 타이틀이 SubSection 타이틀로 대체됨 (SAPUI5 한계)
- SubSection 타이틀만 숨기려면 `.sapUxAPObjectPageSubSectionHeaderToolbar { display: none }`

### 불필요한 `sapUxAPSubSectionSeeMoreContainer`
- SubSection 내용이 길면 "자세히 보기" 버튼과 구분선이 자동 생성됨
- **CSS로 override 불가** — 구조 변경으로 우회
- 해결: 별도 서브섹션 대신 **하나의 서브섹션에 통합**하거나 버튼을 이전 서브섹션 안으로 이동

### 숨겨진 섹션 헤더 공간 제거
```css
.sapUxAPObjectPageSectionHeaderHidden {
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
}
```

---

## 사용된 곳
- **detailPage**: `DetailMain.view.xml` — ObjectPageLayout + DynamicHeader
- **myPage**: `MyMembership.view.xml` — ObjectPageLayout + DynamicHeader
- **PlanChange**: `PlanChange.view.xml` — ObjectPageLayout + DynamicHeader (Page에서 전환)
