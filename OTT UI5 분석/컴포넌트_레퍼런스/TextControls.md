# Title / Label / Text / ObjectStatus (`sap.m`)

> 텍스트 표시 컨트롤 모음

---

## Title
```xml
<Title text="페이지 제목" level="H2" />
<Title text="{viewModel>/Detail/title}" />
```
| 속성 | 설명 |
|------|------|
| `level` | `H1`~`H6`. 시맨틱 제목 레벨 |
| `text` | 제목 텍스트 |
| `textAlign` | `Begin` / `Center` / `End` |

---

## Label
```xml
<Label text="플랜" design="Bold" />
```
| 속성 | 설명 |
|------|------|
| `design` | `Standard` / `Bold` |
| `text` | 라벨 텍스트 |

---

## Text
```xml
<Text text="{viewModel>/value}" />
<Text text="{path: '...', formatter: '.formatPrice'}" />
```
- 기본 텍스트 표시
- `sapMTextMaxWidth` 클래스로 최대 폭 제한 → `max-width: none`으로 override 가능
- `textAlign` 속성으로 정렬 가능

---

## ObjectStatus
```xml
<ObjectStatus
    text="Active"
    state="Success"
    icon="sap-icon://accept" />
```
| 속성 | 값 | 설명 |
|------|-----|------|
| `state` | `Success` / `Warning` / `Error` / `Information` / `None` | 색상 |
| `icon` | `sap-icon://...` | 왼쪽 아이콘 |
| `text` | string | 표시 텍스트 |

상태 배지, 장르 태그 등에 사용.

---

## i18n 바인딩 경로 차이
```xml
text="{viewModel>/Detail/title}"   ← JSONModel: `/`로 객체 탐색
text="{i18n>section.basicInfo}"    ← ResourceModel: `.`은 키의 일부 (flat key-value)
```
- `viewModel>`: `.`과 `/` 모두 객체 계층 탐색
- `i18n>`: `.`은 네이밍 컨벤션일 뿐. `section.basicInfo` 전체가 하나의 키

---

## 사용된 곳
- **모든 페이지**: Title(제목), Label(라벨), Text(값), ObjectStatus(상태/장르 태그)
