# OData 통신 방식 정리 — V2 vs V4

**작성일**: 2026-07-16
**프로젝트**: ZEN-OTT

---

## 1. 개요

ZEN-OTT 프로젝트는 **백엔드는 V4, 프론트엔드는 V2**로 OData 통신을 구성했다.
이 문서는 V2와 V4 각각의 통신 방식과, 이 프로젝트에서 이 조합을 선택한 이유를 정리한다.

---

## 2. 아키텍처 구조

```
┌──────────────────────────────────────────────────────────┐
│  프론트엔드 (브라우저)                                     │
│  ┌────────────────────────────────────────────────────┐  │
│  │  sap.ui.model.odata.v2.ODataModel                  │  │
│  │  - model.read("/Contents('C001')")                 │  │
│  │  - model.callFunction("/submitReview", {           │  │
│  │      success: fn, error: fn                        │  │
│  │    })                                              │  │
│  │  - useBatch: false                                 │  │
│  └────────────────────────────────────────────────────┘  │
│                         │ HTTP                           │
│                         ▼                                │
│  approuter_ott (port 5000) — 라우팅 + 인증                │
│                         │                                │
│                         ▼                                │
│  CAP Backend (port 8083)                                 │
│  ┌────────────────────────────────────────────────────┐  │
│  │  @cap-js-community/odata-v2-adapter                │  │
│  │  V2 요청을 V4로 변환 (자동)                          │  │
│  └──────────────┬─────────────────────────────────────┘  │
│                 ▼                                         │
│  ┌────────────────────────────────────────────────────┐  │
│  │  CAP Native V4 Services                            │  │
│  │  - CDS 정의: @path: '/odata/v4/detail'              │  │
│  │  - 핸들러: async/await, INSERT, SELECT, DELETE      │  │
│  │  - Action: async onSubmitReview(req) { ... }       │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## 3. 백엔드: CAP Native V4

### 3.1 CDS 서비스 정의

```cds
// cap-node/ott/srv/cds/DetailService.cds

@path: '/odata/v4/detail'
@impl: 'srv/src/config/DetailService.handler'
service DetailService @(requires: 'authenticated-user') {

    entity Contents     as projection on db.Contents;
    entity Review       as projection on db.Review;

    action submitReview(
        content_id  : String(10),
        rating      : Decimal(2,1),
        review_text : String(1000)
    ) returns Review;

    action deleteReview(review_id: String(10)) returns Boolean;
}
```

```cds
// cap-node/ott/srv/cds/Membership-service.cds

@(path: 'ott/Mypage')
service MembershipService {
    @(restrict: [{ grant: 'READ', where: 'user_id = $user' }])
    entity UserSet  as projection on Users;

    @readonly entity PlanSet  as projection on SubscriptionPlans;

    action subscribe(plan : String) returns String;
    action changePlan(plan : String) returns String;
    action cancel()                  returns String;
    action resume()                  returns String;
}
```

### 3.2 핸들러 (TypeScript)

```typescript
// cap-node/ott/srv/src/config/DetailService.handler.ts

export default class DetailService extends cds.ApplicationService {
    async init(): Promise<void> {
        this.on('submitReview', this.onSubmitReview);
        this.on('deleteReview', this.onDeleteReview);
        await super.init();
    }

    async onSubmitReview(req: Request) {
        const { content_id, rating, review_text } = req.data;
        const userId = req.user?.id;

        // CAP CDS Query API (V4 네이티브)
        await INSERT.into(Review).entries({
            review_id: generateReviewId(),
            content: { content_id },
            user: { user_id: userId },
            rating,
            review_text
        });

        // 평균 평점 재계산 (raw SQL)
        await cds.run(`
            UPDATE ... SET avg_rating = (SELECT AVG(...))
            WHERE content_content_id = ?
        `, [content_id, content_id, content_id]);

        return SELECT.one.from(Review).where({ review_id });
    }
}
```

### 3.3 백엔드 V4 특징

| 특징 | 설명 |
|------|------|
| CDS 정의 | `@path: '/odata/v4/...'` 또는 기본 V4 경로 |
| 핸들러 | `cds.ApplicationService`, `async/await` |
| 데이터 접근 | CAP CDS Query API: `INSERT.into()`, `SELECT.one.from()`, `DELETE.from()` |
| Action | `this.on('actionName', handler)` |
| 인증 | `@(requires: 'authenticated-user')`, `@restrict` |
| 보안 | `req.user.id`로 사용자 식별, 파라미터 검증 |

---

## 4. V2 Adapter: V4 → V2 변환

### 4.1 설정

```json
// cap-node/ott/package.json
{
  "dependencies": {
    "@cap-js-community/odata-v2-adapter": "^1.15.1"
  }
}
```

**의존성만 추가하면 CAP이 자동으로 모든 서비스에 V2 엔드포인트를 추가 서빙한다.**

### 4.2 V4/V2 엔드포인트 매핑

| 서비스 | V4 (CAP 네이티브) | V2 (Adapter) |
|--------|------------------|-------------|
| DetailService | `/odata/v4/detail/` | `/odata/v2/detail/` |
| MembershipService | `/odata/v4/ott/Mypage/` | `/odata/v2/ott/Mypage/` |
| MainService | `/odata/v4/main/` | `/odata/v2/main/` |

### 4.3 Adapter가 자동으로 처리하는 것

- `$metadata` 문서 변환 (V4 XML → V2 XML)
- Action 호출을 V2 `callFunction` → V4 Action으로 라우팅
- 응답 데이터 포맷 변환 (V4 JSON ↔ V2 JSON)
- 쿼리 파라미터 변환 (`$expand`, `$filter`, `$select` 등)

---

## 5. 프론트엔드: V2 ODataModel

### 5.1 manifest.json 설정

```json
// detailPage/webapp/manifest.json
"dataSources": {
    "detailService": {
        "uri": "/srv-api/odata/v2/detail",
        "type": "OData",
        "settings": { "odataVersion": "2.0" }
    }
},
"models": {
    "detail": {
        "dataSource": "detailService",
        "preload": true,
        "settings": {
            "useBatch": false,
            "defaultUpdateMethod": "MERGE"
        }
    }
}
```

```json
// myPage/webapp/manifest.json
"dataSources": {
    "mypageService": {
        "uri": "/srv-api/odata/v2/ott/Mypage",
        "settings": { "odataVersion": "2.0" }
    }
}
```

### 5.2 데이터 조회 (Read) — `model.read()` + `readOdata()` 래퍼

```javascript
// library_ott/BaseController.js — Promise 래퍼
readOdata: function ({ model, path, param = {} }) {
    return new Promise((resolve, reject) => {
        model.read(path, $.extend({}, param, {
            success: function (...args) { resolve(...args); },
            error: function (...args) { reject(...args); }
        }));
    });
}
```

```javascript
// DetailMain.controller.js — 실제 사용
this.readOdata({
    model: oModel,
    path: "/Contents('C001')",
    param: {
        urlParameters: {
            "$expand": "partner,genres($expand=genre),tags($expand=tag),casts,reviews($expand=user)"
        }
    }
}).then(function (oData) {
    oViewModel.setProperty("/Detail", oData);
});
```

### 5.3 Action 호출 — `callFunction()` (콜백 기반) ⚠️

```javascript
// DetailMain.controller.js — 리뷰 등록
oModel.callFunction("/submitReview", {
    method: "POST",
    urlParameters: {
        content_id: contentId,
        rating: rating,
        review_text: reviewText
    },
    success: function () {
        // ⚠️ Promise가 아니라 success 콜백으로 처리
        that._loadDetail(contentId);
    },
    error: function () {
        MessageToast.show("실패");
    }
});
```

```javascript
// PlanChange.controller.js — 구독 변경
oModel.callFunction("/subscribe", {
    method: "POST",
    urlParameters: { plan: sPlan },
    success: function () {
        MessageToast.show("완료!");
    },
    error: function () {
        MessageToast.show("실패!");
    }
});
```

**⚠️ V2 `callFunction()`은 Promise를 반환하지 않는다. `.then()`을 쓰면 깨진다.**

---

## 6. V2 vs V4 프론트엔드 비교

| | V2 ODataModel | V4 ODataModel |
|---|---|---|
| **모델 클래스** | `sap.ui.model.odata.v2.ODataModel` | `sap.ui.model.odata.v4.ODataModel` |
| **manifest** | `odataVersion: "2.0"` | `odataVersion: "4.0"` |
| **데이터 조회** | `model.read(path, { success, error })` | `model.requestObject(path)` 또는 바인딩 |
| **Promise 지원** | ❌ `model.read()`도 콜백 기반 | ✅ 모든 요청이 Promise |
| **Action 호출** | `callFunction(name, { success, error })` | `callFunction(name, params).then()` |
| **파라미터 전달** | `urlParameters: { ... }` | 두 번째 인자에 직접: `{ plan: "x" }` |
| **배치 처리** | `useBatch: true/false` | 번들 기반 자동 배치 |
| **$metadata 로드** | `preload: true` 시점에 요청 | 지연 로드 (필요할 때만) |

### V4 ODataModel 예시 (참고 — 이 프로젝트에서는 미사용)

```javascript
// V4 Read — Promise 기반
oModel.requestObject("/Contents('C001')")
    .then(oData => { ... });

// V4 Action — Promise 기반
oModel.callFunction("/submitReview", {
    content_id: "C001",
    rating: 4.5,
    review_text: "Great!"
}).then(result => { ... });
```

---

## 7. 이 프로젝트에서 V4 백엔드 + V2 프론트엔드를 선택한 이유

### 진짜 이유: 템플릿 기반 개발 (`_ott` 패턴)

```
codeManagement 템플릿 (기존)
    │
    │  이미 V2 ODataModel 기반으로 작성되어 있음
    │  - manifest: odataVersion: "2.0"
    │  - library: v2.ODataModel 사용
    │  - controller: callFunction({ success, error })
    │
    ▼ 복사 (_ott 패턴)
    │
    ├── detailPage (기능2)
    ├── myPage (기능4)
    ├── portal_ott
    └── library_ott
```

**선택 과정은 이랬다:**

1. `codeManagement`가 템플릿으로 선정됨
2. 템플릿이 **이미 V2**로 모든 걸 구현해 놓은 상태
3. `_ott` 패턴 = 템플릿을 복사해서 독립적으로 사용 (원본 수정 금지)
4. 복사했으니 자연스럽게 V2
5. 단지 `package.json`에 V2 adapter가 빠져있어서 추가한 것

**V4 vs V2를 저울질하는 "설계상의 선택"은 단 한 번도 없었다.**

### 왜 템플릿이 V2였는가? (추정)

- 템플릿(codeManagement)은 SAP CAP의 표준 예제 기반
- SAPUI5 커뮤니티에서 V2 ODataModel이 오랜 기간 표준으로 사용되어 왔음
- V4 ODataModel은 상대적으로 최근(UI5 1.8x 이후)에 성숙

### 왜 굳이 V2 adapter를 붙였는가?

CAP 백엔드는 기본적으로 **V4만 제공**한다. 그런데 프론트엔드가 V2 ODataModel을 쓰려면 반드시 V2 엔드포인트가 필요하다. 그래서 V2 adapter를 추가했다.

---

## 8. 전체 요약

| 계층 | 프로토콜 | 설명 |
|------|:--------:|------|
| **백엔드 정의** | V4 | CDS로 작성된 서비스, `async/await` 핸들러 |
| **V2 Adapter** | V4→V2 | `@cap-js-community/odata-v2-adapter`가 자동 변환 |
| **프론트엔드 통신** | V2 | `sap.ui.model.odata.v2.ODataModel`, 콜백 기반 |
| **선택 이유** | — | 템플릿(codeManagement)이 V2라서 그대로 복사 (`_ott` 패턴) |

---

## 9. 핵심 포인트

1. **V2 `callFunction()`은 Promise가 아니다** — `success`/`error` 콜백 필수
2. **V2 adapter는 의존성 추가만 하면 자동 작동** — 별도 설정 불필요
3. **V4/V2 선택은 템플릿이 결정했다** — 비교·분석·저울질 없었음
4. **백엔드는 네이티브 V4** — CDS 정의, 핸들러 모두 V4 기반
5. **V2 adapter가 V4↔V2 변환을 투명하게 처리** — 프론트엔드는 자신이 V4 백엔드와 통신하는지 모름

---

> **관련 기록**: `압축_통합_기록_2026-06-22_to_2026-06-26.md`
> **관련 파일**: `cap-node/ott/package.json`, `cap-node/ott/srv/cds/DetailService.cds`, `cap-node/ott/srv/cds/Membership-service.cds`, `cap-app/sysmgt_ott/webapp/detailPage/webapp/manifest.json`, `cap-app/sysmgt_ott/webapp/myPage/webapp/manifest.json`, `cap-app/library_ott/src/controller/BaseController.js`
