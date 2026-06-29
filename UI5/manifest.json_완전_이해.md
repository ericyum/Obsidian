# manifest.json 완전 이해

> **대상**: `detailPage/webapp/manifest.json`
> **역할**: SAP UI5 앱의 모든 설정을 한 곳에 선언하는 설계도. UI5가 이 파일을 읽고 앱을 구성한다.

---

## 전체 구조 (3단)

```json
{
    "_version": "1.59.0",
    "sap.app": { ... },    // ① 신원 + 데이터 출처
    "sap.ui":  { ... },    // ② 기기 지원
    "sap.ui5": { ... }     // ③ 모델, 뷰, 라우팅 등 핵심 설정
}
```

---

## ① `sap.app` — "나는 누구고, 어디서 데이터를 가져오는가"

```json
"sap.app": {
    "id": "detailPage",
    "type": "application",
    "applicationVersion": { "version": "1.0.0" },
    "dataSources": {
        "detailService": {
            "uri": "/srv-api/odata/v2/detail",
            "type": "OData",
            "settings": { "odataVersion": "2.0" }
        }
    }
}
```

### `id: "detailPage"` → namespace

이 값은 단순한 이름이 아니라 **앱의 namespace**다. 이걸 기준으로 모든 리소스 경로가 결정된다.

```
sap.app.id = "detailPage"
        │
        ▼ UI5 내부 규칙: manifest.json이 있는 폴더 = namespace 루트
        │
detailPage/webapp/   ← 이 폴더가 "detailPage" namespace의 루트
```

**경로 해석 규칙**: 점(`.`) → 디렉토리 구분자(`/`), 마지막 토큰에 `.view.xml` 또는 `.controller.js` 자동으로 붙음

```
"detailPage.view.App"  →  webapp/view/App.view.xml
"detailPage.view.DetailMain"  →  webapp/view/DetailMain.view.xml
"detailPage.i18n.i18n"  →  webapp/i18n/i18n.properties
```

> 💡 **왜 이런 방식을 쓰는가?** 앱을 어디에 배치하든 `detailPage`라는 이름만 유지되면 경로가 깨지지 않는다. Java 패키지와 동일한 원리.

### `dataSources` — 백엔드 연결 정보 (⚠️ 통신 발생 X)

| 항목 | 의미 |
|------|------|
| `detailService` | 이 연결의 **별칭**. 나중에 모델이 이 이름으로 참조 |
| `uri` | approuter를 경유하는 OData V2 엔드포인트 주소 |
| `odataVersion: "2.0"` | V2 OData → `callFunction()`은 Promise 미지원, 콜백 패턴 필수 |

> ⚠️ **이 시점에는 HTTP 요청이 전혀 발생하지 않는다.** 그냥 "이런 주소를 쓸 거야"라고 선언만 해둔 것.

```
dataSources  →  "커넥션 정보만 선언" (실제 통신 X)
model + preload  →  "이 시점에 실제 통신 시작"
```

---

## ② `sap.ui` — 기기 지원

```json
"sap.ui": {
    "technology": "UI5",
    "deviceTypes": { "desktop": true, "tablet": true, "phone": true }
}
```

데스크톱·태블릿·폰 전체 지원. `ResponsiveGridLayout`이 기기별로 `span="L12 M12 S12"` 같은 분기 처리할 때 이 설정을 참고한다.

---

## ③ `sap.ui5` — 본체 (모델 + 뷰 + 라우팅)

### 📚 의존성 라이브러리

```json
"dependencies": {
    "minUI5Version": "1.123.1",
    "libs": {
        "sap.m": {},
        "sap.ui.core": {},
        "sap.f": {},
        "sap.ui.layout": {},
        "sap.uxap": {}
    }
}
```

| 라이브러리 | 제공하는 주요 컨트롤 | 비유 |
|------------|---------------------|------|
| `sap.m` | Button, Input, Table, Text, Label | 기본 UI "원자재" |
| `sap.ui.core` | 코어 프레임워크 | 앱 작동의 기초 (필수) |
| `sap.f` | DynamicPage, GridList, FlexibleColumnLayout | Fiori "고급 UI 패턴" |
| `sap.ui.layout` | ResponsiveGridLayout, GridData, FlexBox | 순수 "배치 도구" (그리드·플렉스) |
| `sap.uxap` | ⭐ **ObjectPageLayout**, ObjectPageDynamicHeaderTitle | "Object Page" 패턴 (헤더+섹션 구조) |

#### 라이브러리별 상세

**`sap.m`** — 기본 위젯. 버튼, 입력창, 텍스트, 테이블 등. 모든 앱에 필수.

**`sap.ui.core`** — UI5 프레임워크 자체. MVC 패턴, 이벤트, 데이터 바인딩 등.

**`sap.f`** — SAP Fiori 디자인 시스템의 고급 UI 패턴. `DynamicPage`(스크롤 시 접히는 헤더), `GridList`(카드 그리드), `FlexibleColumnLayout`(반응형 다중 컬럼).

**`sap.ui.layout`** — 순수 배치·정렬 담당.
- `ResponsiveGridLayout`: 12컬럼 그리드. `span="L6 M12 S12"`로 화면 크기별 점유율 다르게 지정
- `GridData`: 위 그리드 안에서 각 항목의 컬럼 점유율 지정
- `FlexBox`: CSS Flexbox. `wrap="Wrap"`으로 자동 줄바꿈

**`sap.uxap`** — "물건 하나의 상세 정보를 보여주는" Fiori 정석 패턴.
- `ObjectPageLayout`: 루트 컨테이너. 헤더 + 섹션 구조. 스크롤 시 헤더가 접힘(snap)
- `ObjectPageDynamicHeaderTitle`: 지능형 헤더. expanded(펼침) ↔ snapped(접힘) 전환, 액션 버튼 영역 포함

```
uxap:ObjectPageLayout
├── headerTitle → ObjectPageDynamicHeaderTitle
│   ├── expanded: 제목 + 부가정보 + 액션 버튼 (펼쳐진 상태)
│   └── snapped:  제목만 한 줄 (스크롤 시 접힌 상태)
└── sections:
    ├── Section "기본 정보"
    ├── Section "출연진"
    └── Section "리뷰"
```

---

### 🗂️ 모델 3종 — ⚡ 여기서 실제 통신 시작

```json
"models": {
    "i18n": {
        "type": "sap.ui.model.resource.ResourceModel",
        "settings": { "bundleName": "detailPage.i18n.i18n" }
    },
    "detail": {
        "dataSource": "detailService",
        "preload": true,
        "settings": {
            "useBatch": false,
            "defaultUpdateMethod": "MERGE"
        }
    },
    "viewModel": {
        "type": "sap.ui.model.json.JSONModel"
    }
}
```

#### `i18n` — 국제화 텍스트 번들

- `bundleName: "detailPage.i18n.i18n"` → `webapp/i18n/i18n.properties`
- ResourceModel. 뷰에서 `{i18n>keyName}`, 컨트롤러에서 `this.i18n.getText("keyName")`으로 사용
- 현재 한글 45키 + 영문 45키

> `this.i18n`이 바로 사용 가능한 이유는 라이브러리의 BaseController에 `getResourceBundle()` 편의 메서드가 정의되어 있기 때문이다. (자세한 건 BaseController 문서 참조)

#### `detail` — OData V2 모델 (핵심!)

| 설정 | 값 | 의미 |
|------|-----|------|
| `dataSource` | `"detailService"` | 위에서 선언한 연결 정보에 바인딩. 이제 `detail`이란 이름으로 OData 모델 접근 가능 |
| `preload` | `true` | ⚡ **앱 시작 즉시 `$metadata` 요청 발생!** |
| `useBatch` | `false` | 모든 OData 요청을 **개별 HTTP 요청**으로 보냄 |
| `defaultUpdateMethod` | `"MERGE"` | 데이터 수정 시 PATCH(부분 업데이트) |

#### `preload: true`의 의미

```
preload: true  →  앱 시작 시점에 즉시 $metadata 요청
preload: false →  뷰 바인딩이 처음 평가될 때까지 요청 지연
```

`dataSources`는 "주소만 선언"이고, `model` + `preload: true`가 되어야 실제 HTTP 통신이 시작된다.

| 선언 | 통신 발생? |
|------|:--:|
| `dataSources` | ❌ 주소만 기억 |
| `model` + `dataSource` | ❌ 연결만 함 |
| `model` + **`preload: true`** | ✅ 이때 진짜 요청 시작 |
| 뷰 바인딩 평가 | ✅ 실제 데이터 요청 |

#### `useBatch: false` — 왜 껐는가?

OData V2에는 여러 요청을 **하나의 HTTP POST로 묶어서** 보내는 `$batch` 기능이 있다.

```
useBatch: true (배치 ON)
  POST /$batch  ← 3개 요청을 하나로 포장
  ├── GET Contents('C001')
  ├── GET Contents('C001')/Cast
  └── POST submitReview    ← ⚠️ callFunction이 깨짐!

useBatch: false (배치 OFF)
  GET  /detail/Contents('C001')        ← 개별 요청
  GET  /detail/Contents('C001')/Cast   ← 개별 요청
  POST /detail/submitReview            ← 개별 요청, 정상 작동!
```

> 💡 **비유**: 배치 ON = 택배 한 박스에 유리컵+책+냉동식품 → 냉동식품 녹음. 배치 OFF = 각각 따로 배송 → 각자 멀쩡히 도착.

`callFunction()`으로 보내는 액션(리뷰 등록·삭제·수정)은 OData 표준에 맞는 특별한 헤더·바디 구조가 필요한데, 배치로 묶으면 이 구조가 깨지거나 V2 어댑터가 제대로 처리하지 못한다. 그래서 일부러 배치를 꺼서 모든 요청을 개별 전송하는 것.

#### `defaultUpdateMethod: "MERGE"`

| 값 | HTTP 메서드 | 동작 |
|----|------------|------|
| `MERGE` | PATCH/MERGE | 변경된 필드만 전송 (부분 업데이트) |
| `PUT` | PUT | 전체 리소스를 덮어씀 |

#### `viewModel` — 클라이언트 상태 관리

순수 JSONModel. 초기값 빈 객체 `{}`. 컨트롤러에서 동적으로 데이터를 넣어서 쓴다:

```javascript
var oVM = this.getView().getModel("viewModel");
oVM.setProperty("/NewReview", { rating: 0, comment: "" });
oVM.setProperty("/HasReviewed", false);
```

---

### 🎨 CSS

```json
"resources": {
    "css": [{ "uri": "css/style.css" }]
}
```

`webapp/css/style.css` 자동 로드.

---

### 🧭 루트 뷰 + 라우팅

```json
"rootView": {
    "viewName": "detailPage.view.App",
    "type": "XML",
    "id": "App"
}
```

| 항목 | 의미 | 최종 경로 |
|------|------|-----------|
| `viewName` | 논리 경로 | `webapp/view/App.view.xml` |
| `id: "App"` | **뷰(View) 객체**의 ID | `<mvc:View id="App">` |

> 💡 `rootView.id: "App"`은 뷰 자체의 ID다. 내부 컨트롤의 ID와는 별개다. `App.view.xml` 안의 `<App id="app" />`은 라우팅의 `controlId`가 가리키는 대상이다.

```
실제 DOM:
<mvc:View id="App">           ← rootView.id = "App"
    <App id="app">            ← routing.controlId = "app"이 가리키는 대상
        <pages>
            <DetailMain />    ← 라우팅으로 여기 동적 삽입됨
        </pages>
    </App>
</mvc:View>
```

---

### 🧭 라우팅 전체 메커니즘

```json
"routing": {
    "config": {
        "routerClass": "sap.m.routing.Router",
        "viewPath": "detailPage.view",
        "controlId": "app",
        "controlAggregation": "pages",
        "transition": "slide",
        "bypassed": { "target": "DetailMain" }
    },
    "routes": [
        {
            "pattern": "Detail/{content_id}",
            "name": "DetailMain",
            "target": ["DetailMain"]
        }
    ],
    "targets": {
        "DetailMain": {
            "viewId": "DetailMain",
            "viewName": "DetailMain"
        }
    }
}
```

#### `config` — 라우터 동작 방식

| 설정 | 값 | 설명 |
|------|-----|------|
| `routerClass` | `"sap.m.routing.Router"` | `sap.m.App`의 `pages` aggregation에 뷰를 삽입하는 라우터 구현체 |
| `viewPath` | `"detailPage.view"` | 타겟 뷰를 찾을 기준 경로 → `webapp/view/` |
| `controlId` | `"app"` | 뷰를 삽입할 컨트롤. `App.view.xml`의 `<App id="app">` |
| `controlAggregation` | `"pages"` | `App` 컨트롤의 `pages` aggregation에 타겟 뷰를 동적 추가 |
| `transition` | `"slide"` | 페이지 전환 시 슬라이드 애니메이션 |
| `bypassed` | `"DetailMain"` | 해시 없는 URL로 접근 시 기본 타겟 |

#### `bypassed` — 해시 없음 폴백

```
http://localhost:5000/.../index.html            ← 해시 없음 → bypassed 발동 → DetailMain 표시
http://localhost:5000/.../index.html#/Detail/C001 ← 정상 매칭 → DetailMain 표시
http://localhost:5000/.../index.html#/아무거나   ← 매칭 실패 → bypassed 발동 → DetailMain 표시
```

무슨 일이 있어도 DetailMain은 보여준다는 안전장치.

#### `routes` — URL 패턴 매칭

```
"pattern": "Detail/{content_id}"
 ────── ─────────────
 고정      변수 (URL 파라미터)
```

```
http://localhost:5000/.../index.html#/Detail/C001
                                         └───────┘
                                      content_id = "C001"
```

| 필드 | 의미 |
|------|------|
| `pattern` | 매칭할 URL 해시 패턴. `{}` 안은 변수 |
| `name` | 이 라우트의 이름. 컨트롤러에서 `getRoute("DetailMain")`으로 참조 |
| `target` | 매칭 시 표시할 타겟. 배열이라 여러 타겟을 동시에 띄울 수도 있음 |

#### `targets` — 라우트가 가리키는 실제 뷰

| 필드 | 의미 | 최종 경로 |
|------|------|-----------|
| `viewName` | 뷰 파일 이름 | `viewPath` + `viewName` = `webapp/view/DetailMain.view.xml` |
| `viewId` | 이 타겟의 고정 ID | "DetailMain" — 뷰를 ID로 찾을 때 사용 (현재는 `this.getView()`를 직접 써서 거의 사용 안 함) |

#### `viewId`의 역사 (짧은 보충)

`viewId`는 라우터 캐시에서 뷰를 찾을 때 썼던 ID다. 컨트롤러에서 `router.getView("DetailMain")`으로 뷰를 가져오려고 했으나, `App.view.xml`에서 `controllerName`을 제거하면서 ViewFactory 충돌이 났다. 현재는 `this.getView()`로 바로 뷰를 가져와서 `viewId`에 의존하지 않는다. (자세한 건 DetailMain.controller.js 문서 참조)

---

## 🔄 전체 생명주기

```
1. index.html 로딩
      │
2. manifest.json 파싱
      ├─ sap.app.id = "detailPage" → namespace 등록
      ├─ dataSources "detailService" → /srv-api/odata/v2/detail (선언만)
      └─ sap.ui5 설정 읽기 시작
      │
3. 모델 생성
      ├─ i18n → ResourceModel → webapp/i18n/i18n.properties 로드
      ├─ detail → ODataModel V2 → preload: true → ⚡ $metadata 요청!
      └─ viewModel → 빈 JSONModel {}
      │
4. rootView 생성 → App.view.xml 렌더링
      │
5. 라우터 초기화 → 해시 기반으로 DetailMain 타겟 결정
      │
6. DetailMain.view.xml 로드 + DetailMain.controller.js 인스턴스 생성
      │
7. 컨트롤러 onInit() 호출 → 이벤트 핸들러 등록
      │
8. 뷰의 OData 바인딩 평가 → 실제 데이터 요청 시작
      GET /srv-api/odata/v2/detail/Contents('C001')
```

---

## 🎯 핵심 정리 (7줄)

| # | 포인트 |
|---|--------|
| 1 | `sap.app.id`는 namespace → 점(`.`) → 슬래시(`/`)로 경로 변환 (Java 패키지처럼) |
| 2 | `dataSources`는 주소만 선언, HTTP 통신은 **모델의 `preload: true` 시점**부터 |
| 3 | `useBatch: false` → V2 OData `callFunction()` 정상 작동을 위해 배치 모드 해제 |
| 4 | 5개 라이브러리: `sap.m`(원자재), `sap.ui.layout`(배치), `sap.f`(고급 패턴), `sap.uxap`(ObjectPage) |
| 5 | `rootView.id`는 뷰 자체 ID, `controlId`는 뷰 안의 `sap.m.App` 컨트롤 ID — 서로 다름 |
| 6 | `bypassed`는 해시 없는 URL 접근 시 폴백 타겟 지정 |
| 7 | 뷰+컨트롤러는 **앱 시작 시**가 아니라 **URL 라우트 매칭 시** 생성됨 |

---

> **작성일**: 2026-06-29
> **관련 기록**: `C:\Users\염정운\project\작성 내용\기록\압축_통합_기록_2026-06-22_to_2026-06-26.md`
