# V2 OData 핵심 버그와 해결: `useBatch` + `router.getView()`

> **출처**: manifest.json 이해 중 발생한 막간 질문
> **관련 파일**: `detailPage/webapp/manifest.json`, `DetailMain.controller.js`, `App.view.xml`

---

## 1️⃣ `useBatch: false` — `callFunction()` 정상 작동

### 배치 모드(`useBatch: true`)란?

OData V2에는 여러 요청을 **하나의 HTTP POST로 묶어서** 보내는 `$batch` 기능이 있다.

```
useBatch: true (배치 ON)
─────────────────────────
뷰에서 3개 요청이 필요함:
 ① GET  /Contents('C001')         ← 컨텐츠 정보
 ② GET  /Contents('C001')/Cast    ← 출연진 목록
 ③ POST /submitReview              ← 리뷰 등록 (callFunction)

브라우저는 이렇게 보냄:
  POST /$batch                     ← 3개를 하나로 포장!
  │
  ├── GET Contents('C001')
  ├── GET Contents('C001')/Cast
  └── POST submitReview            ← ⚠️ 문제 발생!
```

### 왜 깨지는가?

`callFunction()`으로 보내는 액션 요청(리뷰 등록, 삭제 등)은 OData 표준에 맞는 특별한 헤더와 바디 구조가 필요하다. 그런데 **배치로 묶으면서 이 구조가 깨지거나, V2 어댑터가 배치 내 함수 호출을 제대로 처리하지 못하는** 버그가 있었다.

```
useBatch: false (배치 OFF)
─────────────────────────
브라우저는 3개를 각각 따로 보냄:
 ① GET  /detail/Contents('C001')           ← 개별 요청
 ② GET  /detail/Contents('C001')/Cast      ← 개별 요청
 ③ POST /detail/submitReview               ← 개별 요청, 깔끔하게 도착!
```

**개별 요청이면 함수 호출이 포장 없이 깔끔하게 전달돼서 문제가 없다.**

### 📊 비유

```
배치 ON  → 택배 하나에 유리컵, 책, 냉동식품 다 넣음 → 냉동식품 녹음
배치 OFF → 각각 따로 배송 → 각자 상태 그대로 도착
```

### manifest.json 설정 위치

```json
"detail": {
    "dataSource": "detailService",
    "preload": true,
    "settings": {
        "useBatch": false,           // ← 바로 이 설정!
        "defaultUpdateMethod": "MERGE"
    }
}
```

---

## 2️⃣ `router.getView("DetailMain")` 버그

### 배경: `App.view.xml` 셸 생성 이유

`rootView`와 `routing target`이 둘 다 `id="DetailMain"` 같은 상황이면 충돌이 난다. 그래서 별도의 셸 뷰 `App.view.xml`을 만들고, 거기엔 `controllerName`을 뺐다.

```xml
<!-- App.view.xml -->
<mvc:View xmlns:mvc="sap.ui.core.mvc" xmlns="sap.m"
    <!-- controllerName 일부러 뺌! --> >
    <App id="app" />
</mvc:View>
```

### 버그 발생

컨트롤러에서 이렇게 뷰를 가져오려고 했다:

```javascript
// ❌ 버그 코드
var oView = this.getOwnerComponent().getRouter().getView("DetailMain") 
         || this.getView();
```

`router.getView("DetailMain")`은 라우터가 내부적으로 관리하는 **뷰 캐시**에서 `viewId: "DetailMain"`으로 찾는 방식이다. 그런데:

```
App.view.xml에 controllerName이 없음
  → UI5 ViewFactory가 이 뷰를 처리할 때 충돌
  → 라우터의 뷰 캐시가 꼬임
  → router.getView("DetailMain")이 null 반환
```

### 해결: 그냥 `this.getView()`만 쓰면 된다

```javascript
// ✅ 수정 코드
var oView = this.getView();
```

`this.getView()`는 **컨트롤러와 쌍으로 연결된 뷰**를 바로 반환한다. 라우터 캐시를 거칠 필요가 전혀 없다.

`DetailMain.view.xml`에 `controllerName="detailPage.controller.DetailMain"`이 있기 때문에, 이 컨트롤러의 `this.getView()`는 항상 `DetailMain` 뷰를 가리킨다.

### 📊 비유

```
router.getView("DetailMain")
  → "주소록에서 'DetailMain' 찾아줘" → 주소록 꼬임 → 못 찾음

this.getView()
  → "지금 내 옆에 같이 서 있는 사람이 누구야?" → 바로 알 수 있음
```

---

## 🎯 요약

| 개념 | 요점 |
|------|------|
| `useBatch: false` | 액션 호출을 배치로 포장하지 않고 개별 전송 → `callFunction()`이 안 깨짐 |
| `router.getView()` 버그 | 셸 뷰의 `controllerName` 누락 → ViewFactory 충돌 → 라우터 캐시 꼬임 |
| 해결책 | `this.getView()`는 `controllerName`으로 이미 쌍이 맺어져 있어서 항상 안전 |

---

## 💡 교훈

1. **V2 OData에서 `callFunction()`을 쓸 땐 `useBatch: false`가 기본값이어야 한다.** 배치 모드는 조회 최적화용이지, 액션 호출과 궁합이 나쁘다.
2. **컨트롤러 안에서는 `this.getView()`로 충분하다.** `router.getView()`는 특수한 상황(다른 타겟의 뷰 접근)에서만 필요하고, 일반적인 자기 뷰 접근은 `this.getView()`가 가장 안전하다.
3. **`App.view.xml` 같은 셸 뷰에서 `controllerName`을 빼면 ViewFactory가 불안정해질 수 있다.** 가능하면 셸에도 controllerName을 명시하는 것이 좋다.

---

> **작성일**: 2026-06-29
> **관련 문서**: `manifest.json_완전_이해.md`
