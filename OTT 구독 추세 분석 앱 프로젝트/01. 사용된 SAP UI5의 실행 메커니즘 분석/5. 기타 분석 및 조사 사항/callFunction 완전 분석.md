# callFunction 완전 분석 — 기원, 구조, 동작 원리

**작성일**: 2026-07-16
**프로젝트**: ZEN-OTT

---

## 1. `callFunction`이란?

SAPUI5에서 OData **Function Import**(Action)를 호출하는 메서드. 백엔드에 정의된 액션(`submitReview`, `subscribe`, `cancel` 등)을 프론트엔드에서 실행할 때 사용한다.

```javascript
oModel.callFunction("/submitReview", {
    method: "POST",
    urlParameters: { content_id: "C001", rating: 4.0, review_text: "Great!" },
    success: function (data, response) { ... },
    error: function (err) { ... }
});
```

---

## 2. `callFunction`은 어디에 정의되어 있는가?

### 답: `sap.ui.model.odata.v2.ODataModel` 프로토타입에 직통 정의

다른 곳에서 상속받거나 import 해오는 게 아니라, 이 클래스 안에 **처음부터 내장**된 메서드다.

```javascript
// SAPUI5 소스: sap/ui/model/odata/v2/ODataModel.js (line 5679)

ODataModel.prototype.callFunction = function (sFunctionName, mParameters) {
    var fnSuccess = mParameters.success;
    var fnError = mParameters.error;
    var mUrlParams = Object.assign({}, mParameters.urlParameters);
    var sMethod = mParameters.method || "GET";
    
    // 1. 함수명 검증 (절대경로 "/..." 필수)
    if (!sFunctionName.startsWith("/")) {
        Log.fatal("callFunction: sFunctionName has to be absolute, ...");
        return undefined;
    }
    
    // 2. $metadata에서 Function Import 정의 조회
    oFunctionMetadata = this.oMetadata._getFunctionImportMetadata(sFunctionName, sMethod);
    
    // 3. urlParameters를 OData 타입에 맞게 포맷팅 → HTTP 요청
    // 4. 성공 → fnSuccess 호출
    // 5. 실패 → fnError 호출
};
```

---

## 3. 전체 상속 사슬

```
sap.ui.base.Object                    ← 모든 것의 조상
  └─ sap.ui.base.EventProvider        ← 이벤트 기능 (attach/detach/fire)
      └─ sap.ui.model.Model           ← 바인딩 (getProperty, bindContext, bindList...)
          └─ sap.ui.model.odata.ODataModel   ← CRUD (read, create, update, remove, refresh)
              └─ sap.ui.model.odata.v2.ODataModel  ← ⭐ callFunction 여기!
```

| 계층 | 제공 메서드 |
|------|------------|
| `Model` | `getProperty()`, `setProperty()`, `getObject()`, `bindContext()`, `bindList()` |
| `ODataModel` | `read()`, `create()`, `update()`, `remove()`, `refresh()`, `createEntry()` |
| `v2.ODataModel` | **`callFunction()`**, `refreshMetadata()`, 배치 처리, `$metadata` 파싱 |

---

## 4. 모델 인스턴스는 어떻게 생성되는가?

### 우리 프로젝트: `manifest.json` → UI5가 자동 생성

```json
// detailPage/webapp/manifest.json
"dataSources": {
    "detailService": {
        "uri": "/srv-api/odata/v2/detail",
        "settings": { "odataVersion": "2.0" }  ← 결정적!
    }
},
"models": {
    "detail": {
        "dataSource": "detailService",
        "preload": true,
        "settings": { "useBatch": false }
    }
}
```

### 내부 동작 흐름

```
index.html
  │
  ▼ ComponentSupport → new detailPage.Component()
  │
  ▼ Component.init() → super.init()
  │
  ▼ manifest.json 파싱
  │
  ▼ models.detail 처리:
  │   "odataVersion": "2.0" → sap.ui.model.odata.v2.ODataModel 선택
  │
  ▼ new sap.ui.model.odata.v2.ODataModel(
  │     "/srv-api/odata/v2/detail/",
  │     { useBatch: false }
  │ )
  │
  ▼ this.setModel(oModel, "detail")  // Component에 등록
```

**우리는 `new ODataModel(...)`을 한 번도 직접 호출한 적이 없다.** manifest.json이 모든 걸 자동화한다.

---

## 5. 컨트롤러에서 모델에 접근하는 방법

```javascript
// 방법 1: View에 등록된 모델
var oModel = this.getView().getModel("detail");

// 방법 2: Component에서 가져오기 (View가 상속받지 못한 경우)
var oModel = this.getOwnerComponent().getModel("detail");

// 이제 바로 callFunction 호출 가능
oModel.callFunction("/submitReview", { ... });
```

둘 다 **동일한 `v2.ODataModel` 인스턴스**를 반환한다.

---

## 6. `callFunction` 내부 동작 — 단계별 분석

### 소스코드 기반 실제 흐름

```
oModel.callFunction("/submitReview", {
    method: "POST",
    urlParameters: { content_id: "C001", rating: 4.0, review_text: "good" },
    success: fnSuccess,
    error: fnError
});

  │
  ▼ ① 함수명 검증
  │   "/"로 시작하지 않으면 Log.fatal → return undefined
  │
  ▼ ② $metadata에서 Function Import 조회
  │   this.oMetadata._getFunctionImportMetadata("/submitReview", "POST")
  │   → 파라미터 정의 가져옴 (content_id: String, rating: Decimal, review_text: String)
  │   → 없으면 Log.error → reject
  │
  ▼ ③ 파라미터 포맷팅
  │   urlParameters 값을 OData 타입에 맞게 변환
  │   JSON body 생성
  │
  ▼ ④ HTTP 요청
  │   POST /srv-api/odata/v2/detail/submitReview
  │   params: { content_id: "C001", rating: 4.0, review_text: "good" }
  │
  │   → approuter_ott (port 5000) → ott backend (port 8083)
  │   → V2 Adapter → V4 Action 핸들러 (onSubmitReview)
  │
  ▼ ⑤ 응답 처리
  │   성공 → fnSuccess(data, response)
  │   실패 → fnError(error)
```

---

## 7. ❗ 가장 중요한 특징: Promise가 아니다

`callFunction()`의 반환 타입:

```javascript
@returns {{contextCreated: function(): Promise, abort: function(): void}|undefined}
```

### 올바른 사용법 (콜백)

```javascript
oModel.callFunction("/submitReview", {
    urlParameters: { content_id: "C001", rating: 4.0, review_text: "good" },
    success: function (data, response) {
        // 성공 처리
    },
    error: function (err) {
        // 실패 처리
    }
});
```

### 틀린 사용법 (Promise — 이렇게 하면 터짐)

```javascript
// ❌ V2에서는 절대 이렇게 하면 안 됨
oModel.callFunction("/submitReview", { content_id: "C001", ... })
    .then(data => { ... })   // TypeError: Cannot read property 'then' of undefined
    .catch(err => { ... });
```

### 반환값을 써야 한다면?

```javascript
var oHandle = oModel.callFunction("/submitReview", { ... });
// oHandle.contextCreated() → Promise<Context>
// oHandle.abort()         → 요청 중단
```

---

## 8. 우리 프로젝트에서의 사용 패턴

### 기본 패턴 (DetailMain.controller.js)

```javascript
onFeedPost: function (oEvent) {
    var oModel = this.getView().getModel("detail");
    var that = this;
    var editingReviewId = oViewModel.getProperty("/NewReview/editing_review_id");

    var doSubmit = function () {
        oModel.callFunction("/submitReview", {
            method: "POST",
            urlParameters: {
                content_id: contentId,
                rating: rating,
                review_text: reviewText
            },
            success: function () {
                if (that._bDestroyed) return;
                that._loadDetail(contentId);  // 서버 재조회
                MessageToast.show("완료!");
            },
            error: function () {
                MessageToast.show("실패!");
            }
        });
    };

    // 수정 모드: 삭제 → 등록 순차 실행
    if (editingReviewId) {
        oModel.callFunction("/deleteReview", {
            method: "POST",
            urlParameters: { review_id: editingReviewId },
            success: function () {
                if (!that._bDestroyed) doSubmit();
            },
            error: function () { that._bSubmitting = false; }
        });
    } else {
        doSubmit();
    }
}
```

### 현재 사용자 확인 패턴 (공통)

```javascript
var that = this;

oModel.callFunction("/subscribe", {
    method: "POST",
    urlParameters: { plan: sPlan },
    success: function () {
        if (that._bDestroyed) return;     // ← 컴포넌트 파괴됐으면 중단
        MessageToast.show("완료!");
    },
    error: function () {
        if (that._bDestroyed) return;     // ← 항상 확인
        MessageToast.show("실패!");
    }
});
```

---

## 9. 전체 파라미터 목록

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `sFunctionName` | `string` | `/`로 시작하는 절대경로. 예: `"/submitReview"` |
| `mParameters.method` | `string` | HTTP 메서드. 기본값 `"GET"` |
| `mParameters.urlParameters` | `object` | Function Import 파라미터. 키=파라미터명, 값=실제값 |
| `mParameters.success` | `function` | 성공 콜백. 인자: `(data, response)` |
| `mParameters.error` | `function` | 실패 콜백. 인자: `(error)` |
| `mParameters.groupId` | `string` | 배치 요청 그룹 ID |
| `mParameters.changeSetId` | `string` | 변경 집합 ID |
| `mParameters.headers` | `object` | 추가 HTTP 헤더 |
| `mParameters.eTag` | `string` | ETag 값 |
| `mParameters.expand` | `string` | 확장할 navigation property (POST 배치 모드에서만) |
| `mParameters.refreshAfterChange` | `boolean` | 작업 후 바인딩 갱신 여부 |

---

## 10. V2 vs V4 `callFunction` 비교

| | V2 | V4 |
|---|---|---|
| **클래스** | `sap.ui.model.odata.v2.ODataModel` | `sap.ui.model.odata.v4.ODataModel` |
| **비동기 처리** | `success`/`error` 콜백 | `.then()`/`.catch()` Promise |
| **파라미터** | `urlParameters: { key: val }` | 두 번째 인자에 직접: `{ key: val }` |
| **반환값** | `{ contextCreated, abort }` 객체 | `Promise` |
| **manifest** | `odataVersion: "2.0"` | `odataVersion: "4.0"` |

---

> **관련 문서**: `Odata 통신 방식 정리.md`
> **소스코드 출처**: `sap/ui/model/odata/v2/ODataModel.js` (SAPUI5 1.148.0, line 5679)
> **관련 파일**: `cap-app/sysmgt_ott/webapp/detailPage/webapp/controller/DetailMain.controller.js`, `cap-app/sysmgt_ott/webapp/myPage/webapp/controller/PlanChange.controller.js`, `cap-app/sysmgt_ott/webapp/myPage/webapp/controller/MyMembership.controller.js`
