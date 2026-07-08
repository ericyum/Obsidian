# 2026-07-07 detailPage·myPage portal 연동 — 전체 과정 및 문제 해결 상세

> detailPage(기능2)와 myPage(기능4)를 portal_ott 런치패드에 통합. MainPage 콘텐츠 행 클릭 → detailPage 진입, 내부 네비게이션, 재진입까지 모든 이슈 해결. 오늘 수정한 8개 파일의 변경 사항과 각각의 원인·해결 과정을 기록.

---

## 0. 작업 전 상태 (배경)

| 항목 | 상태 |
|------|:--:|
| portal_ott 런치패드 | ✅ 작동 중 (OTT_MAIN, OTT_BOARD, OTT_SETTLEMENT 타일 정상) |
| detailPage (기능2) | ✅ 단독 실행 가능 (`/detailPage/webapp/index.html`) |
| myPage (기능4) | ✅ 단독 실행 가능 (`/myPage/webapp/index.html`) |
| Menu.csv OTT_DETAIL 행 | ❌ `menu_app_path` 누락, `menu_route_path` 불일치 |
| Menu.csv OTT_MYPAGE 행 | ❌ `menu_app_path` 누락 |
| detailPage manifest.json | ❌ `"async": true` 누락 (portal 중첩 컴포넌트 필수) |
| MainPage → detailPage 연결 | ❌ 없음 (콘텐츠 목록에서 상세 화면으로 이동 불가) |

---

## 1. 전체 변경 파일 목록

| # | 파일 | 변경 유형 |
|---|------|----------|
| 1 | `core_ott/...Menu.csv` | 시드 데이터 수정 (메뉴 경로 정합) |
| 2 | `detailPage/webapp/manifest.json` | `"async": true` 추가 |
| 3 | `mainPage/webapp/view/Main.view.xml` | `ColumnListItem` Navigation + press 이벤트 |
| 4 | `mainPage/webapp/controller/Main.controller.js` | `onContentPress` 핸들러 추가 |
| 5 | `detailPage/webapp/controller/DetailMain.controller.js` | onInit: 해시 파싱 + `window.hashchange` 리스너 |
| 6 | `myPage/webapp/controller/MyMembership.controller.js` | onInit: `_loadMembership()` 수동 호출 + OData fallback |
| 7 | `myPage/webapp/controller/PlanChange.controller.js` | OData 모델 fallback 패턴 |
| 8 | (detailPage `_loadDetail` 등도 OData fallback 적용) | — |

---

## 2. Menu.csv — 메뉴 경로 정합

### 문제

기존 OTT_DETAIL, OTT_MYPAGE 행의 `menu_app_path`가 비어 있거나 잘못되어 portal에서 타일을 눌러도 Component.js를 찾지 못했다.

### 원인 분석 (approuter 경로 정합)

approuter `dev/xs-app.json`의 sysmgt_ott 라우트:

```json
{ "source": "^(.*)/sysmgt_ott/(.*)$", "target": "$2", "localDir": "../../sysmgt_ott/webapp" }
```

- approuter는 `localDir`(`sysmgt_ott/webapp`)에서 파일을 서빙한다
- `menu_app_path`는 `localDir` 내부의 상대 경로여야 함
- `sysmgt_ott/webapp/` 밑에 `detailPage/webapp/`, `myPage/webapp/`, `mainPage/webapp/`이 있으므로

→ `menu_app_path`는 `/sysmgt_ott/detailPage/webapp`, `/sysmgt_ott/myPage/webapp`가 맞다

> 참고: board/settlement는 `board/webapp/` 구조라 `menu_app_path = /board`만 줘도 됨. sysmgt_ott는 `webapp/` 아래에 **한 단계 더** `{기능}/webapp/`이 있어서 전체 경로를 다 써야 한다. (→ 자세한 원리는 `2026-07-07 OTT-approuter-메뉴경로정합-디버깅.md` 참고)

### `menu_route_path` — detailPage 전용

portal_ott는 `menu_route_path`를 라우트 패턴으로 사용한다:

```javascript
// portal_ott Component.js (내부 동작)
var sPattern = menu_app_id + "/" + menu_route_path;
// 예: "detailPage/Detail/:content_id:"
oRouter.addRoute({
    pattern: sPattern,
    name: menu_code,  // "OTT_DETAIL"
    target: { name: menu_code, prefix: menu_app_id }  // prefix: "detailPage"
});
```

detailPage는 content_id 파라미터를 받아야 하므로:

```
menu_route_path = Detail/:content_id:
```

여기서 `:content_id:`는 **crossroads.js 문법**이다. portal_ott가 `crossroads.addRoute()`로 직접 라우트를 생성하기 때문에, `{?content_id}` 같은 manifest 문법이 아니라 crossroads의 `:param:`(선택적 파라미터)를 써야 한다.

> `:변수명:` — crossroads에서 선택적 파라미터. `Detail/C001`도 매칭되고 `Detail/`도 매칭됨. 타일에서 직접 눌렀을 때 content_id가 없어도 크래시 방지.

myPage는 파라미터가 없으므로 `menu_route_path`는 공백.

### 최종 menu_app_path 및 menu_route_path

```
OTT_DETAIL  ; detailPage ; sysmgt_ott/webapp ; /sysmgt_ott/detailPage/webapp ; Detail/:content_id:
              app_id       repo_path             app_path                         route_path

OTT_MYPAGE  ; myPage     ; sysmgt_ott/webapp ; /sysmgt_ott/myPage/webapp    ; (공백)
```

---

## 3. detailPage manifest.json — `"async": true`

### 문제

portal이 detailPage 컴포넌트를 지연 로드(`lazy: true`)로 생성할 때, `"async": true`가 없으면 SAPUI5 라우터가 **동기 초기화**를 시도하다가 다음과 같은 오류 발생:

```
Uncaught Error: Router has not been initialized yet.
or
Assertion failed: Router is already initialized
```

### 원인

portal에서 `componentUsage` + `lazy: true`로 동적 컴포넌트를 로드할 때, SAPUI5는 컴포넌트의 라우터를 비동기로 초기화해야 부모 라우터와 충돌하지 않는다. `"async": true`는 이 비동기 초기화를 명시적으로 활성화한다.

### 추가 코드

`detailPage/webapp/manifest.json` → `routing.config`:

```json
"routing": {
    "config": {
        "routerClass": "sap.m.routing.Router",
        "viewType": "XML",
        "viewPath": "detailPage.view",
        "targetClass": "sap.m.routing.Targets",
        "controlId": "app",
        "controlAggregation": "pages",
        "transition": "slide",
        "async": true,                    // ★ 추가
        "bypassed": { "target": "DetailMain" }
    },
    ...
}
```

---

## 4. MainPage → detailPage 네비게이션 연결

### 4-1. Main.view.xml — ColumnListItem Navigation화

**파일**: `mainPage/webapp/view/Main.view.xml`

```xml
<!-- 기존: 일반 행 -->
<ColumnListItem>
    <cells>...</cells>
</ColumnListItem>

<!-- 변경: 네비게이션 가능한 행 + press 이벤트 -->
<ColumnListItem type="Navigation" press=".onContentPress">
    <cells>...</cells>
</ColumnListItem>
```

- `type="Navigation"`: 행에 `>` 화살표 아이콘 + hover 효과 추가 (Fiori 표준)
- `press=".onContentPress"`: 행 클릭 시 컨트롤러의 `onContentPress` 호출

### 4-2. Main.controller.js — onContentPress 핸들러

**파일**: `mainPage/webapp/controller/Main.controller.js`

```javascript
/** 컨텐츠 목록 행 클릭 → detailPage로 이동 */
onContentPress: function (oEvent) {
    var contentId = oEvent.getSource().getBindingContext().getProperty("content_id");
    if (contentId) {
        window.location.hash = "detailPage/Detail/" + contentId;
    }
},
```

#### 시도했으나 실패한 방식들

| 시도 | 코드 | 결과 |
|------|------|:--:|
| 1차 | `HashChanger.setHash("detailPage/Detail/" + contentId)` | ✅ 최초 진입 성공, ❌ 재진입 시 이전 데이터 잔류 |
| 2차 | `portalRouter.navTo("OTT_DETAIL", {content_id})` | ❌ `getParentComponent is not a function` (mainPage는 portal의 rootView로 로드 — parent 없음) |
| 3차 | `window.location.hash = "detailPage/Detail/" + contentId` | ✅ 최초 진입 + 재진입 모두 성공 |

#### `window.location.hash`를 선택한 이유

- portal 라우터가 `detailPage/Detail/C001` 패턴을 인식하고 OTT_DETAIL 타겟을 표시함
- SAPUI5 HashChanger가 `window.location.hash` 변경을 감지하여 라우팅 이벤트를 발생시킴
- `setHash`와 거의 동일하게 동작하지만, 재진입 시에도 안정적

---

## 5. detailPage — 중첩 컴포넌트 onInit 문제 & 재진입 해결

**파일**: `detailPage/webapp/controller/DetailMain.controller.js`

### 문제 1: 최초 진입 시 `_onRouteMatched` 미호출

#### 원인

portal이 detailPage 컴포넌트를 `bypassed` 타겟으로 로드할 때:
1. portal 라우터가 `detailPage/Detail/C001` 매칭
2. OTT_DETAIL 타겟 표시 → detailPage 컴포넌트 로드
3. detailPage 라우터 초기화 → `bypassed: { target: "DetailMain" }` → view 렌더링
4. **하지만 `DetailMain` 라우트의 `patternMatched` 이벤트가 발생하지 않음**

이유: portal은 `bypassed` 메커니즘으로 view를 직접 렌더링한다. sub-path(`Detail/C001`)를 detailPage 라우터가 재파싱하지 않고, bypassed target view만 보여준 것.

#### 1차 해결 (onInit에서 수동 호출)

```javascript
onInit: function () {
    BaseController.prototype.onInit.call(this);
    this.i18n = this.getResourceBundle();
    this.getOwnerComponent().getRouter()
        .getRoute("DetailMain")
        .attachPatternMatched(this._onRouteMatched, this);

    // ★ portal 중첩 컴포넌트 진입 시: 해시에서 content_id 직접 추출
    var sHash = sap.ui.core.routing.HashChanger.getInstance().getHash();
    var aMatch = sHash.match(/detailPage\/Detail\/([^\/]+)/);
    if (aMatch) {
        this._onRouteMatched({
            getParameter: function (sName) {
                return sName === "arguments" ? { content_id: aMatch[1] } : undefined;
            }
        });
    }
},
```

- 정규식 `/detailPage\/Detail\/([^\/]+)/`로 해시에서 content_id 추출
- `_onRouteMatched`가 `oEvent.getParameter("arguments").content_id`를 기대하므로 fake event 객체 주입
- ✅ 최초 진입 성공: C001, C002, C003 모두 데이터 정상 로딩

### 문제 2: 재진입 시 이전 데이터 잔류

#### 증상

```
1. MainPage → C001 클릭 → detailPage 로드, C001 데이터 정상 표시 ✓
2. "OTT Main" 타일 → MainPage 복귀 ✓
3. MainPage → C002 클릭 → detailPage는 뜨지만 C001 데이터가 그대로 남아 있음 ❌
```

Console: `Blocked aria-hidden on an element because its descendant retained focus...`

#### 원인

portal이 detailPage 컴포넌트를 **재사용**한다:
- 첫 진입: 새 Component 인스턴스 생성 → `onInit` 실행 → 수동 `_onRouteMatched` → C001 데이터 로드
- MainPage 복귀: detailPage 컴포넌트 숨김 (`aria-hidden="true"`, `sapMNavItemHidden`)
- 재진입: **같은 인스턴스를 다시 보여줌** → `onInit`은 안 돈다 → `attachPatternMatched`도 재발화 안 됨 → 이전 데이터 그대로

#### 시도했으나 실패한 방식들

| 시도 | 방법 | 실패 원인 |
|------|------|-----------|
| 2차 | `HashChanger.attachHashChanged` | onInit 시점 등록 시 첫 로드 간섭, 레이아웃 자체가 안 뜸 |
| 3차 | `view.addEventDelegate({ onAfterRendering })` | 재진입 시 view가 재렌더링되지 않아 이벤트 미발생 (컴포넌트는 살아있고 container만 visible 토글) |
| 4차 | `portal.navTo("OTT_DETAIL", {content_id})` | `getParentComponent()` 없음 (mainPage = rootView) |

#### 최종 해결: `window.addEventListener("hashchange")`

```javascript
onInit: function () {
    BaseController.prototype.onInit.call(this);
    this.i18n = this.getResourceBundle();
    this.getOwnerComponent().getRouter()
        .getRoute("DetailMain")
        .attachPatternMatched(this._onRouteMatched, this);

    var that = this;

    // ★ 재진입 대응: window hashchange로 해시 변경 감지
    window.addEventListener("hashchange", function () {
        if (that._bDestroyed) return;
        var sHash = window.location.hash.replace(/^#/, "");
        var aMatch = sHash.match(/detailPage\/Detail\/([^\/]+)/);
        if (aMatch && aMatch[1] !== that._lastContentId) {
            that._lastContentId = aMatch[1];
            that._onRouteMatched({
                getParameter: function (sName) {
                    return sName === "arguments" ? { content_id: aMatch[1] } : undefined;
                }
            });
        }
    });

    // ★ 최초 진입 시 해시에서 content_id 직접 추출
    var sHash = sap.ui.core.routing.HashChanger.getInstance().getHash();
    var aMatch = sHash.match(/detailPage\/Detail\/([^\/]+)/);
    if (aMatch) {
        this._lastContentId = aMatch[1];
        this._onRouteMatched({
            getParameter: function (sName) {
                return sName === "arguments" ? { content_id: aMatch[1] } : undefined;
            }
        });
    }
},
```

#### 핵심 포인트

| 요소 | 역할 |
|------|------|
| `window.addEventListener("hashchange")` | view 생명주기와 무관하게 **모든** URL 해시 변경 감지 |
| `that._bDestroyed` 가드 | 컴포넌트 파괴 후 불필요한 처리 방지 |
| `that._lastContentId` | 중복 호출 방지 (같은 content_id로 재진입 시 skip) |
| `location.hash.replace(/^#/, "")` | `#detailPage/Detail/C001` → `detailPage/Detail/C001`로 정규화 |
| 최초 수동 호출 | `onInit` 시점에는 `hashchange` 리스너 등록 이전이므로 직접 한 번 호출 |

#### `aria-hidden` 경고 설명

```
Blocked aria-hidden on an element because its descendant retained focus.
Ancestor with aria-hidden: <div class="sapUiComponentContainer sapMNavItem sapMNavItemHidden">
```

- 원인: portal이 detailPage → mainPage 전환 시 detailPage 컴포넌트 컨테이너를 `aria-hidden="true"`로 숨김
- 이때 포커스가 아직 숨겨진 컨테이너 내부 요소에 남아있으면 브라우저 접근성 경고 발생
- **기능적 영향 없음** — SAPUI5 내부 포커스 관리의 잔여 동작
- 재진입 2회차에만 1회 발생하고 이후엔 발생하지 않음

---

## 6. myPage — 중첩 컴포넌트 onInit 우회

**파일**: `myPage/webapp/controller/MyMembership.controller.js`

### 문제

detailPage와 동일: portal의 `bypassed` 타겟으로 로드 시 `_onRouteMatched` 미호출 → 멤버십 데이터가 로드되지 않음.

추가 증상: `i18n key 'null'` assertion 오류 — 멤버십 데이터가 없어서 plan_code가 undefined, i18n 리소스에서 `status.null` 키를 찾지 못함.

### 해결

```javascript
onInit: function () {
    BaseController.prototype.onInit.call(this);
    this.i18n = this.getResourceBundle();
    this.getOwnerComponent().getRouter()
        .getRoute("MyMembership")
        .attachPatternMatched(this._onRouteMatched, this);

    // ★ portal 중첩 컴포넌트: bypassed만 타고 _onRouteMatched가 호출되지 않으므로 수동 로딩
    this._loadMembership();
},
```

- detailPage는 content_id가 필요해 복잡했지만, myPage는 파라미터가 없으므로 `_loadMembership()` 한 줄이면 충분
- `attachPatternMatched`도 유지 (PlanChange → MyMembership 정상 네비게이션 대응)

### 검증 (로그 기반)

```
[구독 변경]     PREMIUM → BASIC / active        ✅
[구독 취소]     BASIC   → BASIC / cancelled     ✅
[구독 재개]     BASIC   → BASIC / active        ✅
[플랜 재변경]   BASIC   → PREMIUM / active      ✅
```

---

## 7. OData 모델 fallback 패턴 (detailPage + myPage 공통)

### 문제

portal이 `lazy: true`로 detailPage/myPage 컴포넌트를 지연 로드할 때, `manifest.json`에 선언된 OData 모델이 자동 생성되지 않는 경우가 있다.

증상: `getModel("detail")` → `undefined`, OData 호출 불가

### 해결

모든 데이터 로딩 함수 시작 부분에 fallback 패턴 적용:

```javascript
var oModel = this.getView().getModel("모델명") || this.getOwnerComponent().getModel("모델명");
if (!oModel) {
    oModel = new sap.ui.model.odata.v2.ODataModel("/srv-api/odata/v2/서비스경로", {
        useBatch: false
    });
    this.getView().setModel(oModel, "모델명");
}
```

### 적용 위치

| 파일 | 함수 | 모델명 | OData URI |
|------|------|--------|-----------|
| `DetailMain.controller.js` | `_loadDetail` | `detail` | `/srv-api/odata/v2/detail/` |
| `MyMembership.controller.js` | `_loadMembership` | `myPage` | `/srv-api/odata/v2/ott/Mypage` |
| `MyMembership.controller.js` | `_callAction` | `myPage` | (기존 모델 사용) |
| `PlanChange.controller.js` | `_loadPlans` | `myPage` | `/srv-api/odata/v2/ott/Mypage` |
| `PlanChange.controller.js` | `onSubmitPlan` | `myPage` | (기존 모델 사용) |

---

## 8. myPage/PlanChange.controller.js — OData fallback 추가

**파일**: `myPage/webapp/controller/PlanChange.controller.js`

변경 전 (단독 실행 가정):

```javascript
_loadPlans: function () {
    var oModel = this.getView().getModel("myPage");
    // ...
},
```

변경 후:

```javascript
_loadPlans: function () {
    var oModel = this.getView().getModel("myPage") || this.getOwnerComponent().getModel("myPage");
    if (!oModel) {
        oModel = new sap.ui.model.odata.v2.ODataModel("/srv-api/odata/v2/ott/Mypage", {
            useBatch: false
        });
        this.getView().setModel(oModel, "myPage");
    }
    // ...
},
```

`onSubmitPlan`의 `callFunction` 호출부에도 동일 패턴 적용.

---

## 9. 전체 디버깅 타임라인

| 순서 | 시도 | 결과 |
|:--:|------|:--:|
| 1 | Menu.csv `menu_app_path` 수정 + `async: true` | 최초 진입 안 됨 (Component.js 404?) |
| 2 | detailPage onInit에서 해시 파싱 → `_onRouteMatched` 수동 호출 | ✅ 최초 진입 성공 |
| 3 | MainPage `onContentPress` → `HashChanger.setHash` | ✅ C001 데이터 로드 |
| 4 | C001 → MainPage → C002 테스트 | ❌ 이전 데이터 잔류 |
| 5 | `HashChanger.attachHashChanged` 추가 | ❌ 첫 로드부터 레이아웃 안 뜸 |
| 6 | 5번 롤백, `portal.navTo("OTT_DETAIL")` 시도 | ❌ `getParentComponent is not a function` |
| 7 | `setHash` → `window.location.hash` 변경 | ✅ 최초 진입 성공 |
| 8 | `view.addEventDelegate({ onAfterRendering })` | ❌ 재진입 시 onAfterRendering 미호출 |
| 9 | `window.addEventListener("hashchange")` | ✅ 최초 + 재진입 모두 성공! |
| 10 | 모든 `console.log` 제거 | ✅ |
| 11 | myPage `_loadMembership()` 수동 호출 | ✅ |
| 12 | myPage 구독 변경·취소·재개 전환 테스트 | ✅ 모든 상태 전환 정상 |
| 13 | myPage 로그 제거 | ✅ |
| 14 | `git pull origin dev` 후 8개 파일 검증 | ✅ 손상 없음 |

---

## 10. 핵심 원리 요약

### Portal 중첩 컴포넌트 3대 발등

| 문제 | 원인 | 해결 |
|------|------|------|
| `_onRouteMatched` 미호출 | bypassed target은 view만 렌더링, 라우트 이벤트 미발생 | onInit에서 수동 호출 |
| OData 모델 undefined | lazy load 시 manifest 모델 미전파 | `getModel \|\| getOwnerComponent().getModel \|\| new ODataModel` |
| `async: true` 누락 | 중첩 라우터 초기화 충돌 | manifest.json에 `"async": true` 추가 |

### detailPage 재진입 특수 과제

| 문제 | 원인 | 해결 |
|------|------|------|
| 이전 데이터 잔류 | portal이 Component 인스턴스를 파괴하지 않고 재사용 (hide/show) | `window.addEventListener("hashchange")`로 URL 변경 감지 → `_onRouteMatched` 재호출 |
| 생명주기 이벤트 미발생 | view 재렌더링 없음 → `onAfterRendering` 안 탐 | window 네이티브 이벤트로 우회 |
| 중복 로딩 방지 | 같은 content_id 연속 클릭 | `_lastContentId` 추적 |

### `_lastContentId` 패턴

```javascript
// onInit 내부
if (aMatch && aMatch[1] !== that._lastContentId) {
    that._lastContentId = aMatch[1];  // 갱신
    that._onRouteMatched(...);        // 로딩
}
// 같은 content_id면 skip → 불필요한 OData 중복 호출 방지
```

### HashChanger vs window.location.hash vs window.hashchange

| 방법 | 적합 상황 | 이슈 |
|------|-----------|------|
| `HashChanger.setHash` | SAPUI5 라우터 내부 제어 | 재진입 시 간헐적 불발 |
| `HashChanger.attachHashChanged` | 라우터 이벤트 리스닝 | onInit 등록 시 첫 로드 간섭 |
| `window.location.hash = "..."` | URL 설정 | ✅ 안정적 (네이티브) |
| `window.addEventListener("hashchange")` | URL 변경 감지 | ✅ 생명주기 무관, 항상 발화 |

---

## 11. 최종 파일별 변경 내역 (diff 요약)

### 1. `core_ott/db/data/com.cap.ott.core-Menu.csv`

```
- OTT_DETAIL  ; OTT ; 컨텐츠 상세 ; ... ; detailPage ; true ; true ; 110 ; sysmgt_ott/webapp ; ; ; ...
+ OTT_DETAIL  ; OTT ; 컨텐츠 상세 ; ... ; detailPage ; true ; true ; 110 ; sysmgt_ott/webapp ; /sysmgt_ott/detailPage/webapp ; Detail/:content_id: ; ...

- OTT_MYPAGE  ; OTT ; 마이페이지 ; ... ; myPage ; true ; true ; 120 ; sysmgt_ott/webapp ; ; ; ...
+ OTT_MYPAGE  ; OTT ; 마이페이지 ; ... ; myPage ; true ; true ; 120 ; sysmgt_ott/webapp ; /sysmgt_ott/myPage/webapp ; ; ...
```

### 2. `detailPage/webapp/manifest.json`

```diff
  "routing": {
      "config": {
+         "async": true,
          "bypassed": { "target": "DetailMain" }
      }
  }
```

### 3. `mainPage/webapp/view/Main.view.xml`

```diff
- <ColumnListItem>
+ <ColumnListItem type="Navigation" press=".onContentPress">
```

### 4. `mainPage/webapp/controller/Main.controller.js`

```diff
+ /** 컨텐츠 목록 행 클릭 → detailPage로 이동 */
+ onContentPress: function (oEvent) {
+     var contentId = oEvent.getSource().getBindingContext().getProperty("content_id");
+     if (contentId) {
+         window.location.hash = "detailPage/Detail/" + contentId;
+     }
+ },
```

### 5. `detailPage/webapp/controller/DetailMain.controller.js`

```diff
  onInit: function () {
      BaseController.prototype.onInit.call(this);
      this.i18n = this.getResourceBundle();
      this.getOwnerComponent().getRouter()
          .getRoute("DetailMain")
          .attachPatternMatched(this._onRouteMatched, this);

+     var that = this;
+     window.addEventListener("hashchange", function () {
+         if (that._bDestroyed) return;
+         var sHash = window.location.hash.replace(/^#/, "");
+         var aMatch = sHash.match(/detailPage\/Detail\/([^\/]+)/);
+         if (aMatch && aMatch[1] !== that._lastContentId) {
+             that._lastContentId = aMatch[1];
+             that._onRouteMatched({ ... });
+         }
+     });
+
+     var sHash = sap.ui.core.routing.HashChanger.getInstance().getHash();
+     var aMatch = sHash.match(/detailPage\/Detail\/([^\/]+)/);
+     if (aMatch) {
+         this._lastContentId = aMatch[1];
+         this._onRouteMatched({ ... });
+     }
  },
```

### 6. `myPage/webapp/controller/MyMembership.controller.js`

```diff
  onInit: function () {
      BaseController.prototype.onInit.call(this);
      this.i18n = this.getResourceBundle();
      this.getOwnerComponent().getRouter()
          .getRoute("MyMembership")
          .attachPatternMatched(this._onRouteMatched, this);
+     this._loadMembership();
  },
```

OData fallback도 추가:

```diff
  _loadMembership: function () {
-     var oModel = this.getView().getModel("myPage");
+     var oModel = this.getView().getModel("myPage") || this.getOwnerComponent().getModel("myPage");
+     if (!oModel) {
+         oModel = new sap.ui.model.odata.v2.ODataModel("/srv-api/odata/v2/ott/Mypage", { useBatch: false });
+         this.getView().setModel(oModel, "myPage");
+     }
```

### 7. `myPage/webapp/controller/PlanChange.controller.js`

```diff
  _loadPlans: function () {
-     var oModel = this.getView().getModel("myPage");
+     var oModel = this.getView().getModel("myPage") || this.getOwnerComponent().getModel("myPage");
+     if (!oModel) {
+         oModel = new sap.ui.model.odata.v2.ODataModel("/srv-api/odata/v2/ott/Mypage", { useBatch: false });
+         this.getView().setModel(oModel, "myPage");
+     }
```

`onSubmitPlan`에도 동일 패턴 적용.

### 8. DetailMain.controller.js `_loadDetail` — OData fallback

```diff
  _loadDetail: function (contentId) {
-     var oModel = this.getView().getModel("detail");
+     var oModel = this.getView().getModel("detail") || this.getOwnerComponent().getModel("detail");
+     if (!oModel) {
+         oModel = new sap.ui.model.odata.v2.ODataModel("/srv-api/odata/v2/detail/", { useBatch: false });
+         this.getView().setModel(oModel, "detail");
+     }
```

---

## 12. 재진입 시 `_onRouteMatched`이 불리는 전체 흐름도

```
MainPage → C001 클릭
  │  window.location.hash = "detailPage/Detail/C001"
  ▼
portal 라우터 → OTT_DETAIL 매칭 → detailPage Component load
  │  (첫 로드: 새 인스턴스)
  ▼
detailPage.onInit()
  │  ├─ attachPatternMatched("DetailMain", _onRouteMatched)
  │  ├─ window.addEventListener("hashchange", fnCheck)  ← 등록만, 이 시점엔 hashchange 안 탐
  │  └─ 해시 파싱: C001 → this._lastContentId = "C001" → _onRouteMatched({content_id: "C001"})
  ▼
_loadDetail("C001") → C001 데이터 표시 ✅

───── 사용자가 MainPage 타일 클릭 ─────

portal 라우터 → detailPage Component HIDE (aria-hidden=true) → mainPage 표시

───── MainPage에서 C002 클릭 ─────
  │  window.location.hash = "detailPage/Detail/C002"
  ▼
window "hashchange" 이벤트 발생 → fnCheck 실행
  │  hash = "detailPage/Detail/C002", match = "C002"
  │  "C002" !== this._lastContentId ("C001") → 통과!
  │  this._lastContentId = "C002"
  │  _onRouteMatched({content_id: "C002"})
  ▼
_loadDetail("C002") → C002 데이터 표시 ✅

───── MainPage에서 다시 C002 클릭 ─────
  │  window.location.hash = "detailPage/Detail/C002"
  ▼
"hashchange" → hash = "detailPage/Detail/C002", match = "C002"
  │  "C002" !== this._lastContentId ("C002")? → ❌ 같음 → skip (중복 방지)
```

---

---

## 13. 심화 이해 — 오늘의 Q&A로 보완된 개념들

> 아래 내용은 2026-07-08에 진행된 복습 세션에서 명확해진 개념들이다.

---

### 13-1. Menu.csv의 4개 핵심 컬럼 — 각각의 역할

Menu.csv 한 줄은 4개의 핵심 컬럼을 가진다:

```
OTT_DETAIL ; detailPage ; sysmgt_ott/webapp ; /sysmgt_ott/detailPage/webapp ; Detail/:content_id:
             menu_app_id    menu_repo_path       menu_app_path                    menu_route_path
```

| 컬럼 | 값 | 진짜 역할 |
|------|-----|------|
| `menu_app_id` | `detailPage` | **컴포넌트 이름**. portal이 이 이름으로 `detailPage.Component`를 찾음 |
| `menu_app_path` | `/sysmgt_ott/detailPage/webapp` | **URL 경로**. `sap.ui.loader.config`에 등록되어, `detailPage/어쩌고` → `/sysmgt_ott/detailPage/webapp/어쩌고` 로 변환 |
| `menu_repo_path` | `sysmgt_ott/webapp` | **캐시 관리용 폴더 위치**. `AppCacheBuster.register()`에 등록되어 브라우저 캐시 무효화 담당 |
| `menu_route_path` | `Detail/:content_id:` | **URL 패턴**. portal Router가 이 패턴에 매칭되는 해시를 감지 |

### 13-2. `menu_app_path`는 파일 시스템 경로가 아니라 URL 경로다

**오해하기 쉬운 포인트:**

```
파일 시스템 경로:  sysmgt_ott/webapp/detailPage/webapp/Component.js
menu_app_path:    /sysmgt_ott/detailPage/webapp
```

`menu_app_path`에는 `webapp`이 하나 빠져 있는 것처럼 보이지만, 이건 **URL 경로**이기 때문이다. approuter가 중간에서 번역해준다:

```
브라우저 요청 URL:    /sysmgt_ott/detailPage/webapp/Component.js
                            ↓
approuter xs-app.json:  /sysmgt_ott/ → sysmgt_ott/webapp/ 로 번역
                            ↓
실제 파일 경로:        sysmgt_ott/webapp/detailPage/webapp/Component.js
```

### 13-3. `menu_app_id` + `menu_app_path` = 하나의 매핑, 하나의 메커니즘

portal_ott는 Menu.csv를 읽고 UI5 모듈 로더에 이렇게 등록한다:

```javascript
sap.ui.loader.config({ 
    paths: { 
        "detailPage": "/sysmgt_ott/detailPage/webapp"
    } 
});
```

이 한 줄의 매핑이 **모든 요청에 일관되게 적용**된다:

```
최초 로드:    detailPage/Component.js           → /sysmgt_ott/detailPage/webapp/Component.js
뷰 로드:     detailPage/view/App.view.xml       → /sysmgt_ott/detailPage/webapp/view/App.view.xml
컨트롤러:    detailPage/controller/DetailMain.js → /sysmgt_ott/detailPage/webapp/controller/DetailMain.js
i18n:        detailPage/i18n/i18n.properties    → /sysmgt_ott/detailPage/webapp/i18n/i18n.properties
```

**"최초 로드"와 "내부 파일 로드"는 서로 다른 메커니즘이 아니다. 하나의 매핑이 일관되게 적용될 뿐이다.**

### 13-4. ⚠️ approuter ≠ SAPUI5 Router (완전히 다르다!)

이 두 가지는 이름만 비슷할 뿐 전혀 다른 존재다. 혼동하면 모든 게 꼬인다.

| | approuter | SAPUI5 Router |
|------|------|------|
| 정체 | Node.js HTTP 서버 프로세스 | 브라우저 안의 JavaScript 객체 |
| 하는 일 | 정적 파일 서빙 + API 프록시 | URL 해시(#) 관리 + 화면 전환 |
| 실행 위치 | **서버** (터미널에서 `npm run start:local`) | **브라우저** (사용자 PC 메모리) |
| port | 5000 | 없음 (메모리 상의 객체) |
| 언제 개입? | HTTP 요청이 들어올 때마다 | URL 해시가 변경될 때마다 |

**approuter의 개입 시점:**

```
① 누군가 정적 파일을 달라고 HTTP 요청 → 파일 전달
   "GET /sysmgt_ott/detailPage/webapp/Component.js" → 실제 파일 서빙

② 누군가 API를 호출 → 백엔드로 프록시
   "GET /srv-api/odata/v2/detail/..." → ott(8083)이나 core_ott(8084)로 전달
```

**approuter는 portal 진입이든 detailPage 진입이든 똑같이 동작한다. "따로 발동"하는 게 아니라, HTTP 요청이 있을 때마다 항상 같은 규칙으로 처리할 뿐이다.**

### 13-5. `lazy: true` + `async: true`의 관계

두 설정은 서로 다른 문제를 해결하지만, portal 중첩 구조에서는 둘 다 필수다.

| 설정 | 없으면? | 역할 |
|------|------|------|
| `lazy: true` (portal 쪽) | portal 로딩 시 모든 컴포넌트를 한꺼번에 로드 → 느려지고 화면 엉망 | "지금 말고, 필요할 때만 만들어라" |
| `async: true` (detailPage 쪽) | 두 SAPUI5 Router가 동시에 URL 해시를 초기화하려다 충돌 → `Router has not been initialized` 오류 | "portal Router가 먼저 초기화되고 끝나면 그때 초기화해라" |

**`lazy: true`는 portal의 Component.js에서, `async: true`는 detailPage의 manifest.json에서 각각 설정된다.**

### 13-6. `async: true` 충돌은 브라우저 안에서만 일어난다

`async: true`의 충돌 주체는 **approuter가 아니라 브라우저 메모리 안의 두 SAPUI5 Router 객체**다:

```
브라우저 메모리 안:
  ① portal의 Router 객체     ← portal_ott/Component.js가 만듦 (이미 실행 중)
  ② detailPage의 Router 객체  ← detailPage/Component.js가 막 생성됨

이 두 JavaScript 객체가 같은 URL 해시(#)를 동시에 관리하려고 하면서 충돌.
async: true는 이 충돌을 "portal 먼저, detailPage 나중에"로 순서를 조율한다.
```

**approuter는 이 충돌과 아무 관련이 없다. approuter는 그냥 파일 요청을 처리하는 배달부일 뿐이다.**

### 13-7. 왜 `window.location.hash`는 되고 `HashChanger.setHash()`는 안 됐나

| 방법 | 특징 | 재진입 시 |
|------|------|:--:|
| `HashChanger.setHash()` | SAPUI5가 **알아서 판단**. "이미 같은 상태니까 스킵" 가능 | ❌ C001→C002 전환 놓침 |
| `window.location.hash =` | 브라우저 **네이티브**. 무조건 변경, 무조건 이벤트 발생 | ✅ 확실하게 동작 |

`HashChanger`는 SAPUI5가 내부 최적화로 "똑같은 패턴이네, 굳이 다시 라우팅할 필요 없겠다"라고 판단할 수 있다. 반면 `window.location.hash`는 브라우저 레벨에서 아무 생각 없이 무조건 변경하고 이벤트를 발생시킨다.

**둘은 짝을 이뤄 동작한다:**

```
3단계: window.location.hash = "detailPage/Detail/C002"  → 해시 "쓰기" (MainPage)
5단계: window.addEventListener("hashchange", ...)       → 해시 변경 "감지" (detailPage)
```

### 13-8. `portalRouter.navTo()`가 실패한 진짜 이유

mainPage는 portal의 **rootView**로 로드된다. 즉, portal 안에 mainPage가 직접 렌더링되어 있을 뿐, portal과 mainPage는 컴포넌트 간 **부모-자식 관계가 아니다.**

```
portal Component
  └── Layout (rootView)
        └── <App id="app">
              ├── mainPage Component   ← portal과 부모-자식 관계 ❌
              ├── detailPage Component
              └── myPage Component
```

SAPUI5에서 `getParentComponent()`는 직접적인 부모-자식 관계에서만 존재한다. portal은 Router가 target으로 컴포넌트들을 관리할 뿐(`type: "Component"`, `usage: "..."`), 부모-자식 참조를 만들지 않는다.

```javascript
// 이렇게 하면 안 됨 (부모-자식 관계가 아니므로)
this.getOwnerComponent().getParentComponent().getRouter().navTo(...)
//                                ↑ undefined! ❌
```

**Router는 컴포넌트를 통해서만 접근 가능하다. mainPage는 portal 컴포넌트에 닿을 방법이 없어서 portal Router 자체를 호출할 수 없었다. 그래서 우회로로 `window.location.hash`를 선택한 것이다.**

### 13-9. 아키텍처적 고찰 — 더 나은 방법이 있었을까?

portal이 sub-path(`Detail/C001`)를 detailPage Router에게 위임해주는 것이 가장 깔끔하다. 그러나 SAPUI5 Router는 "하나의 앱이 전체 해시를 소유한다"고 가정하고 설계되어 있어, 부모-자식 Router 간의 위임(delegation) 메커니즘이 내장되어 있지 않다.

```
고려 가능한 대안들:

① portal 수정 → sub-path 위임 로직 추가
   ❌ _ott 원칙: 템플릿 절대 수정 금지

② componentData 활용 → detailPage Component.js init에서 직접 라우팅
   이론상 가능하지만, Component.js에서 Router를 건드리는 건 관례가 아님

③ bypassed 제거
   ❌ 단독 실행이 깨짐

④ onInit 수동 파싱 (우리가 한 방식)
   ✅ SAP 커뮤니티에서도 흔히 쓰는 실무적 우회책
```

**결론: SAPUI5 Router의 근본적 한계로 인해 완벽한 해결책은 없다. 우리 방식이 현실적인 최선이다.**

### 13-10. 데이터 요청은 원래부터 비동기 — `async: true`가 필요 없는 이유

`manifest.json`의 `async: true`는 **SAPUI5 Router 초기화 순서** 문제고, 데이터 요청은 **JavaScript 자체의 비동기 메커니즘**(Promise, 콜백)으로 전혀 다른 세계다.

```javascript
// readOdata → Promise (이미 비동기)
this.readOdata("/Contents('C001')")
    .then(function(oData) { /* 응답 오면 실행, UI 안 멈춤 */ });

// callFunction → 콜백 (이미 비동기)
oModel.callFunction("/submitReview", {...}, {
    success: function() { /* 응답 오면 실행, UI 안 멈춤 */ }
});
```

**데이터 요청은 애초에 설계 단계부터 비동기로 되어 있어서, portal 중첩 구조에서도 추가 설정 없이 안전하게 동작한다.**

### 13-11. `window.addEventListener("hashchange")`가 `HashChanger.attachHashChanged`보다 나은 이유

| 방법 | SAPUI5 생명주기 의존? | 재진입 시 발동? |
|------|:--:|:--:|
| `HashChanger.attachHashChanged` | ✅ 의존 | ❌ onInit 등록 시 첫 로드 간섭 |
| `view.onAfterRendering` | ✅ 의존 | ❌ 재진입 시 view 재렌더링 없음 |
| `window.addEventListener("hashchange")` | ❌ 무관 | ✅ **무조건 발동** |

portal이 컴포넌트를 숨겼다 보여주기만 할 때, SAPUI5 생명주기 이벤트(`onInit`, `onAfterRendering`, `patternMatched`)는 재발생하지 않는다. 하지만 **브라우저 네이티브 이벤트는 SAPUI5의 상태와 완전히 무관하게 항상 발동**한다. 그래서 `window.hashchange`만이 유일하게 신뢰할 수 있는 재진입 감지 수단이었다.

---

> **세션**: 2026-07-07 | **보완**: 2026-07-08 | **참고 문서**: `2026-07-07 OTT-approuter-메뉴경로정합-디버깅.md`, `2026-07-07 OTT-approuter-화면띄우기-전체과정.md`
