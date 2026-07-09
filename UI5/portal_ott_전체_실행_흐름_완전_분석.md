# portal_ott 전체 실행 흐름 완전 분석

> **작성일**: 2026-07-08
> **목적**: approuter_ott 로컬 실행 시, 브라우저에서 portal_ott에 접속한 후 모든 파일이 어떤 순서로 실행되는지 완전히 이해한다.
> **핵심 질문**: `index.html` → `sap-ui-core.js` → `ComponentSupport` → `Component.init()` → `manifest.json` → `UserSessionLoader`/`MenuLoader` → `Router.initialize()` → 첫 화면 렌더링까지의 전 과정

---

## 📁 관련 파일 위치

| 구성요소 | 경로 |
|----------|------|
| approuter | `cap-app/approuter_ott/` |
| approuter 로컬 설정 | `cap-app/approuter_ott/dev/xs-app.json` |
| approuter destination | `cap-app/approuter_ott/dev/default-env.json` |
| portal_ott | `cap-app/portal_ott/webapp/` |
| library_ott (공통 라이브러리) | `cap-app/library_ott/src/` |
| core_ott 백엔드 | `cap-node/core_ott/` (port 8084) |
| ott 백엔드 | `cap-node/ott/` (port 8083) |

---

## 0. 실행 전 준비 — 3개 터미널

```bash
# 터미널 1: OTT 비즈니스 백엔드 (port 8083)
cd cap-node/ott && npx cds watch --port 8083

# 터미널 2: core_ott 공통 백엔드 (port 8084) — UserManagement, Menu 등
cd cap-node/core_ott && npx cds watch --port 8084

# 터미널 3: approuter (port 5000) — 정적파일 서빙 + API 프록시
cd cap-app/approuter_ott && npm run start:local
```

### `start:local` 명령어 분석

```json
// package.json
"start:local": "node node_modules/@sap/approuter/approuter.js --workingDir ./dev --port 5000"
```

- `--workingDir ./dev` → `dev/` 폴더를 작업 디렉토리로 사용
- 즉, `dev/xs-app.json`을 라우팅 설정으로 읽고, `dev/default-env.json`을 destination 설정으로 읽음
- **approuter 자체의 `xs-app.json`(BTP 배포용)은 사용하지 않음!**

---

## 1. 브라우저 → HTTP 요청

```
사용자 주소창 입력: http://localhost:5000/portal_ott/index.html
```

브라우저는 `GET /portal_ott/index.html` HTTP 요청을 `localhost:5000`(approuter)으로 보냅니다.

---

## 2. approuter의 요청 처리 — `dev/xs-app.json` 라우트 매칭

approuter(port 5000)는 `dev/xs-app.json`의 `routes` 배열을 **위에서 아래로 순서대로** 검사합니다. **첫 번째로 매칭되는 라우트가 승리**합니다.

### 2.1 매칭 과정

```
요청 경로: /portal_ott/webapp/index.html

라우트 1: ^.*/srv-api/(.*ott-core/.*)$        → "srv-api" 없음 → ❌
라우트 2: ^.*/srv-api/(.*)$                    → "srv-api" 없음 → ❌
라우트 3: ^.*/erp-api/(.*)$                    → "erp-api" 없음 → ❌
라우트 4: ^.*/scim-api/(.*)$                   → "scim-api" 없음 → ❌
라우트 5: /common_ott.lib/(.*)                 → "common_ott.lib" 없음 → ❌
라우트 6: /common.lib/(.*)                     → "common.lib" 없음 → ❌
라우트 7: ^(.*)/template/(.*)$                 → "template" 없음 → ❌
라우트 8: ^(.*)/sysmgt/(.*)$                   → "sysmgt" 없음 → ❌
라우트 9: ^(.*)/board/(.*)$                    → "board" 없음 → ❌
라우트 10: ^(.*)/settlement/(.*)$              → "settlement" 없음 → ❌
라우트 11: ^(.*)/trendAnalysis/(.*)$           → "trendAnalysis" 없음 → ❌
라우트 12: ^/portal/(.*)$                      → "portal" ≠ "portal_ott" → ❌
라우트 13: ^(.*)/portal_ott/(.*)$              → ✅ 매칭!
라우트 14: ^/(.*)$                             → (여기까지 안 옴)
```

### 2.2 매칭된 라우트

```json
{
    "source": "^(.*)/portal_ott/(.*)$",
    "target": "$2",
    "localDir": "../../portal_ott/webapp",
    "authenticationType": "none",
    "csrfProtection": false,
    "cacheControl": "no-cache, no-store, must-revalidate"
}
```

### 2.3 정규표현식 캡처 그룹 해석

```
^(.*)/portal_ott/(.*)$
│     │              │
│     │              └─ 캡처 그룹 2 ($2): portal_ott/ 다음의 모든 것
│     └─ 캡처 그룹 1 ($1): /portal_ott 이전의 모든 것 (우리 경우 빈 문자열)
└─ 문자열 시작
```

실제 매칭:
```
입력: /portal_ott/index.html
$1 = ""                          (빈 문자열 — /portal_ott 앞에 아무것도 없음)
$2 = "index.html"         (portal_ott/ 이후)
```

### 2.4 `target` + `localDir` 조합

```json
"target": "$2",                           // → "index.html"
"localDir": "../../portal_ott/webapp"     // 디스크 경로
```

approuter가 디스크에서 읽는 실제 파일 경로:
```
{localDir}/{target}
= ../../portal_ott/webapp/index.html
```

### 2.5 최종 결과

approuter는 디스크 파일을 읽어서 HTTP 200 응답으로 브라우저에 반환합니다. `content-type: text/html`.

> 💡 `localDir`의 의미: "이 경로에 매칭되는 요청은 프록시(백엔드로 전달)하지 말고, 내 로컬 디스크에서 파일을 읽어서 바로 응답해라"

---

## 3. 브라우저가 index.html 파싱 시작

### 3.1 전체 index.html 코드

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Cache-control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <link rel="icon" href="/common_ott.lib/images/solum-favicon.png" >
    <title>Company</title>

    <!-- ★ 핵심: UI5 부트스트랩 -->
    <script
        id="sap-ui-bootstrap"
        src="https://ui5.sap.com/1.136/resources/sap-ui-core.js"
        data-sap-ui-theme="sap_fiori_3"
        data-sap-ui-resourceroots='{
            "portal": "./"
            ,"common_ott.lib": "/common_ott.lib"
        }'
        data-sap-ui-compatVersion="edge"
        data-sap-ui-async="true"
        data-sap-ui-frameOptions="trusted"
        data-sap-ui-appCacheBuster="./"
        data-sap-ui-componentName="app"
        data-sap-ui-libs="sap.m,sap.f,sap.ui.table,sap.uxap,sap.tnt,common_ott.lib"
        data-sap-ui-onInit="module:sap/ui/core/ComponentSupport"
    ></script>

    <script src="/common_ott.lib/thirdparty/sheetjs/xlsx.min.js"></script>
    <script src="/common_ott.lib/thirdparty/sheetjs/xlsx.full.min.js"></script>
</head>
<body class="sapUiBody sapUiSizeCompact" id="content2">
    <div
        data-sap-ui-component data-name="portal"
        data-id="container"
        data-settings='{"id" : "portal"}'
        data-handle-validation="true"
    ></div>
</body>
</html>
```

### 3.2 `<script src="...sap-ui-core.js">` — UI5 엔진 다운로드

브라우저가 CDN에서 `sap-ui-core.js`를 요청:
```
GET https://ui5.sap.com/1.136/resources/sap-ui-core.js
```
→ SAP CDN에서 UI5 1.136 버전 코어 라이브러리 다운로드

### 3.3 추가 스크립트 다운로드 (병렬)

```html
<script src="/common_ott.lib/thirdparty/sheetjs/xlsx.min.js"></script>
```

> ⚠️ **중요**: 이 `<script src="...">` 태그는 `data-sap-ui-resourceroots`의 영향을 **전혀 받지 않는다!**
> - `resourceroots` = **UI5 AMD 모듈 로더 전용** 설정 (`sap.ui.define()`, `sap.ui.require()`)
> - `<script src="...">` = **브라우저 HTML 파서**가 직접 처리 → 절대경로 그대로 HTTP GET 요청
> - 자세한 내용은 아래 [4.1.1절](#411-data-sap-ui-resourceroots-vs-script-src---완전히-다른-메커니즘) 참고

브라우저가 이 요청도 보냅니다:
```
GET /common_ott.lib/thirdparty/sheetjs/xlsx.min.js
```

approuter에서 `dev/xs-app.json` 라우트 매칭:
```json
{
    "source": "/common_ott.lib/(.*)",
    "target": "$1",
    "localDir": "../../library_ott/src",
    "authenticationType": "none"
}
```

정규표현식 매칭:
```
/common_ott.lib/thirdparty/sheetjs/xlsx.min.js
                  └────────── $1 ──────────────┘
```

디스크 경로:
```
../../library_ott/src/thirdparty/sheetjs/xlsx.min.js
```

---

## 4. UI5 코어 초기화 — `data-sap-ui-*` 속성 처리

`sap-ui-core.js`가 로드 완료되면, UI5 코어는 `<script id="sap-ui-bootstrap">` 태그의 `data-sap-ui-*` 속성들을 읽고 자동 설정합니다.

### 4.1 `data-sap-ui-resourceroots` — 모듈 경로 매핑 등록

```html
data-sap-ui-resourceroots='{
    "portal": "./"
    ,"common_ott.lib": "/common_ott.lib"
}'
```

UI5 코어 내부에서 다음과 같이 실행됩니다:

```javascript
// UI5 코어 내부 (자동 실행)
sap.ui.loader.config({
    paths: {
        "portal": "./",
        "common_ott.lib": "/common_ott.lib"
    }
});
```

이 매핑의 의미:

| UI5 네임스페이스 | → | URL로 변환 | → | 최종 HTTP 경로 |
|------------------|---|------------|---|----------------|
| `portal/Component.js` | → | `./Component.js` | → | `/portal_ott/webapp/Component.js` |
| `portal/view/Layout.view.xml` | → | `./view/Layout.view.xml` | → | `/portal_ott/webapp/view/Layout.view.xml` |
| `portal/controller/Layout.controller.js` | → | `./controller/Layout.controller.js` | → | `/portal_ott/webapp/controller/Layout.controller.js` |
| `common_ott/lib/controller/BaseController.js` | → | `/common_ott.lib/lib/controller/BaseController.js` | → | (approuter → `../../library_ott/src/lib/controller/BaseController.js`) |

> 💡 **`"./"` 의 기준점**: 현재 HTML 페이지의 URL 디렉토리. 즉 `/portal_ott/webapp/` 이 기준이 됩니다.

### 4.1.1 `data-sap-ui-resourceroots` vs `<script src="...">` — 완전히 다른 메커니즘

**둘 다 비슷한 URL(`/common_ott.lib/...`)로 요청하지만, 경로를 결정하는 방식이 완전히 다릅니다.**

#### 🅰 `data-sap-ui-resourceroots` = UI5 모듈 로더의 "주소록"

```html
data-sap-ui-resourceroots='{
    "portal": "./"
    ,"common_ott.lib": "/common_ott.lib"
}'
```

이 설정은 오직 **UI5의 AMD 모듈 로더**가 사용합니다. 즉, `sap.ui.define()`이나 `sap.ui.require()`로 모듈을 로드할 때만 작동합니다:

```javascript
// ✅ 이럴 때 resourceroots가 작동함
sap.ui.define([
    "common_ott/lib/util/MenuLoader",    // ← "common_ott.lib" → "/common_ott.lib" 로 변환
    "portal/controller/Home"             // ← "portal" → "./" 로 변환
], function(MenuLoader, Home) {
    // ...
});

// 변환 결과:
// "common_ott/lib/util/MenuLoader.js" → GET /common_ott.lib/lib/util/MenuLoader.js
// "portal/controller/Home.controller.js" → GET /portal_ott/webapp/controller/Home.controller.js
```

#### 🅱 `<script src="...">` = 브라우저 HTML 파서가 직접 요청

```html
<script src="/common_ott.lib/thirdparty/sheetjs/xlsx.min.js"></script>
```

이건 **순수 HTML `<script>` 태그**입니다. UI5 모듈 로더를 전혀 거치지 않습니다. 브라우저의 HTML 파서가 이 태그를 만나면:

1. `src` 속성의 값을 그대로 읽음 → `"/common_ott.lib/thirdparty/sheetjs/xlsx.min.js"`
2. 절대경로이므로 **아무 변환 없이** HTTP GET 요청을 보냄
3. `GET http://localhost:5000/common_ott.lib/thirdparty/sheetjs/xlsx.min.js`

#### 비교표

|                   | `resourceroots` 경로                                     | `<script src="...">` 경로                             |
| ----------------- | ------------------------------------------------------ | --------------------------------------------------- |
| **누가 처리?**        | UI5 AMD 모듈 로더 (`sap.ui.loader`)                        | 브라우저 HTML 파서                                        |
| **언제 작동?**        | `sap.ui.define()`, `sap.ui.require()` 호출 시             | HTML 파싱 중 `<script>` 태그 발견 시                        |
| **경로 변환?**        | **함** — 네임스페이스를 URL로 변환                                | **안 함** — `src` 값 그대로 요청                            |
| **예시**            | `"common_ott/lib/..."` → `GET /common_ott.lib/lib/...` | `"/common_ott.lib/..."` → `GET /common_ott.lib/...` |
| **approuter 도달?** | ✅ (변환된 URL도 같은 approuter 라우트에 매칭)                      | ✅ (절대경로가 approuter 라우트에 매칭)                         |

> 💡 **비유**: `resourceroots` = UI5 전용 내비게이션 앱. `"common_ott.lib"`라는 주소를 입력하면 `"/common_ott.lib"`로 안내해줌.
> `<script src="...">` = 브라우저가 직접 운전. 이미 전체 주소(절대경로)를 알고 있어서 내비 없이 바로 목적지로 감.
> **둘 다 최종 목적지(approuter)는 같지만, 가는 방식이 다르다!**

#### `data-sap-ui-libs`의 `common_ott.lib`도 마찬가지!

```html
data-sap-ui-libs="sap.m,sap.f,sap.ui.table,sap.uxap,sap.tnt,common_ott.lib"
```

`data-sap-ui-libs`에 적힌 라이브러리들도 UI5 내부적으로 **모듈 로더를 통해** `library.js`를 로드합니다. 즉, `common_ott.lib`도 `sap.ui.define()`처럼 `resourceroots`의 변환(`common_ott.lib` → `/common_ott.lib`)을 그대로 통과합니다.

> 📖 **자세한 4단계 분해**: 바로 아래 [4.2절](#42-data-sap-ui-libs--ui-라이브러리-preload)에서 `common_ott.lib` → `library.js` → `resourceroots` 변환 → approuter 파일 경로까지 **한 단계씩 완전 분해**해서 설명합니다.

`data-sap-ui-libs` vs `<script src>`의 흐름을 간단히 비교하면:

```
① data-sap-ui-libs="common_ott.lib"
   → UI5 코어가 common_ott.lib 라이브러리 preload 시작
   → UI5 모듈 로더 경유 → resourceroots 변환 적용
   → GET /common_ott.lib/library.js
   → approuter → ../../library_ott/src/library.js

② <script src="/common_ott.lib/thirdparty/sheetjs/xlsx.min.js"></script>
   → 브라우저 HTML 파서가 직접 처리
   → 경로 변환 없음 (이미 절대경로) → resourceroots 완전 무시
   → GET /common_ott.lib/thirdparty/sheetjs/xlsx.min.js
   → approuter → ../../library_ott/src/thirdparty/sheetjs/xlsx.min.js
```

### 4.2 `data-sap-ui-libs` — UI 라이브러리 preload

```html
data-sap-ui-libs="sap.m,sap.f,sap.ui.table,sap.uxap,sap.tnt,common_ott.lib"
```

#### 4.2.1 `data-sap-ui-libs`의 역할 — 라이브러리 선언

UI5가 이 라이브러리들의 `library.js` 파일을 **미리 다운로드**합니다:

| 라이브러리 | 다운로드 위치 | 용도 |
|-----------|-------------|------|
| `sap.m` | SAP CDN | 기본 UI 컨트롤 (Button, Text, Input, App, Page...) |
| `sap.f` | SAP CDN | Fiori 고급 패턴 (DynamicPage, FlexibleColumnLayout...) |
| `sap.ui.table` | SAP CDN | 데이터 테이블 |
| `sap.uxap` | SAP CDN | ObjectPage 패턴 (ObjectPageLayout...) |
| `sap.tnt` | SAP CDN | ToolPage, ToolHeader, SideNavigation |
| `common_ott.lib` | **우리 서버** | 공통 라이브러리 (BaseController, MenuLoader, UserSessionLoader...) |

`sap.m`, `sap.f` 같은 SAP 표준 라이브러리는 이미 `sap-ui-core.js`를 로드한 CDN 내에서 찾을 수 있기 때문에 문제가 없습니다. 하지만 **`common_ott.lib`는 우리 프로젝트의 커스텀 라이브러리**이므로, UI5가 이 파일을 어디서 찾아야 하는지 경로를 알려줘야 합니다. 바로 이때 `resourceroots`가 개입합니다.

#### 4.2.2 🔥 `common_ott.lib` 로드 4단계 — 완전 분해

`data-sap-ui-libs="common_ott.lib"`라고 적었을 때, 실제로 브라우저가 `library.js` 파일을 가져오기까지의 과정을 한 단계씩 따라가 보겠습니다.

---

**1단계: `common_ott.lib` → 모듈 경로로 변환**

UI5는 `data-sap-ui-libs`에 적힌 이름을 보고, "이 라이브러리의 매니페스트 파일을 로드해야 한다"고 판단합니다. 라이브러리 매니페스트는 항상 `library.js`라는 이름이므로, UI5는 내부적으로 다음과 같은 **모듈 경로**를 만듭니다:

```
common_ott.lib  →  common_ott/lib/library.js
                      ↑
                      점(.)을 슬래시(/)로 바꾸고,
                      마지막에 /library.js 를 붙인다
```

> 💡 UI5 네임스페이스는 Java 패키지처럼 점(`.`)으로 구분하지만, 파일 경로로 바꿀 때는 슬래시(`/`)로 변환됩니다.

---

**2단계: `resourceroots`의 영향을 받는다 — 모듈 로더 경유**

이 모듈 경로(`common_ott/lib/library.js`)를 실제로 가져오는 주체는 **UI5의 AMD 모듈 로더**(`sap.ui.loader`)입니다. `sap.ui.define()`이나 `sap.ui.require()`와 똑같은 메커니즘을 사용합니다.

따라서 `data-sap-ui-resourceroots`에 등록된 매핑이 **여기서 적용**됩니다. 이게 핵심입니다:

```
💡 data-sap-ui-libs 로 인한 요청은
   <script src="...">(HTML 파서 직접) 과 달리
   UI5 모듈 로더를 경유하므로 resourceroots의 영향을 받는다!
```

---

**3단계: `resourceroots` 변환 — 네임스페이스 → URL**

`resourceroots`에는 다음과 같이 등록되어 있습니다:

```json
"common_ott.lib": "/common_ott.lib"
```

이 매핑의 의미는: **모듈 경로에서 `common_ott.lib`(네임스페이스)로 시작하는 부분을 `/common_ott.lib`(URL 접두사)로 치환하라**는 것입니다.

실제 변환 과정을 시각화하면:

```
모듈 경로:     common_ott / lib / library.js
                 ↑               ↑
                 네임스페이스      나머지 (그대로 유지)
                 "common_ott.lib"
                 │
                 │  resourceroots 매핑에 따라 치환
                 ▼
URL 접두사:    /common_ott.lib

최종 URL:     /common_ott.lib / library.js
              ──────┬───────   ─────┬─────
              치환된 접두사       나머지 (그대로)
```

즉, 브라우저는 다음과 같은 HTTP 요청을 보내게 됩니다:

```
GET /common_ott.lib/library.js
```

---

**4단계: approuter가 실제 파일 경로로 변환**

브라우저가 `GET /common_ott.lib/library.js` 요청을 `localhost:5000`(approuter)으로 보내면, approuter는 `dev/xs-app.json`에서 이 경로와 일치하는 라우트를 찾습니다:

```json
{
    "source": "/common_ott.lib/(.*)",
    "target": "$1",
    "localDir": "../../library_ott/src",
    "authenticationType": "none"
}
```

정규표현식 매칭:

```
/common_ott.lib/library.js
└──────────────┬──────────┘
            $1 = "library.js"
```

최종 파일 경로:

```
{localDir} + {target}
= ../../library_ott/src + library.js
= ../../library_ott/src/library.js
```

approuter는 이 디스크 파일을 읽어서 브라우저에 HTTP 200 응답으로 반환합니다.

---

#### 4.2.3 전체 4단계 한눈에 보기

```
① data-sap-ui-libs="common_ott.lib"
   ↓  UI5가 모듈 경로로 변환
   common_ott/lib/library.js
   
② UI5 AMD 모듈 로더(sap.ui.loader)가 처리 시작
   ↓  resourceroots 매핑 적용!
   "common_ott.lib" → "/common_ott.lib"
   
③ 네임스페이스를 URL 접두사로 치환
   common_ott/lib/library.js
   ─────────                 (네임스페이스 → 치환)
            /library.js      (나머지 → 그대로)
   ════════════════════════
   /common_ott.lib/library.js  ← 최종 HTTP 요청 URL
   
④ 브라우저 → approuter(5000) → xs-app.json 라우트 매칭
   /common_ott.lib/(.*)  →  ../../library_ott/src/$1
   
   결과: ../../library_ott/src/library.js 파일 반환 🎉
```

#### 4.2.4 `<script src>`와의 결정적 차이 (재비교)

같은 `/common_ott.lib/...` 경로로 요청하지만, **가는 길이 완전히 다릅니다**:

|  | `data-sap-ui-libs="common_ott.lib"` | `<script src="/common_ott.lib/...">` |
|---|---|---|
| **1단계** | `common_ott.lib` → `common_ott/lib/library.js` (모듈 경로) | 변환 없음 (`src` 그대로) |
| **2단계** | UI5 모듈 로더 경유 ✅ | 브라우저 HTML 파서 직접 ❌ |
| **3단계** | `resourceroots` 변환 적용 ✅ | `resourceroots` 무시 ❌ |
| **4단계** | approuter에서 파일 경로로 변환 ✅ | approuter에서 파일 경로로 변환 ✅ |

> 🧠 **핵심**: 둘 다 4단계(approuter)는 같지만, `<script src>`는 1~3단계를 건너뛰고 **이미 완성된 절대 URL**로 바로 4단계로 직행합니다. `data-sap-ui-libs`는 1~3단계를 거쳐서 **네임스페이스를 URL로 조립**한 뒤 4단계로 갑니다.

#### 4.2.5 `library.js` 로드 후에는?

`library.js`는 라이브러리의 **매니페스트(선언 파일)**입니다. 이 파일 안에는 "이 라이브러리가 어떤 컨트롤과 모듈로 구성되어 있는지"가 정의되어 있습니다. `library.js` 로드가 끝나면 UI5는 이어서:

- **디버그 모드**: 개별 컨트롤 파일들을 필요할 때마다 하나씩 로드
- **프로덕션 모드**: `library-preload.js`(모든 모듈을 하나로 묶은 번들)를 로드

이후 `BaseController`, `MenuLoader`, `UserSessionLoader` 같은 개별 유틸리티 모듈도 `sap.ui.require()`를 통해 똑같은 `resourceroots` 변환 → approuter 과정을 거쳐 로드됩니다.

### 4.3 `data-sap-ui-async="true"` — 비동기 모드

```html
data-sap-ui-async="true"
```

모든 모듈 로딩을 비동기(AMD)로 처리. **절대 생략하면 안 됩니다** — 생략 시 동기 로딩으로 인해 앱이 깨집니다.

### 4.4 `data-sap-ui-onInit="module:sap/ui/core/ComponentSupport"` — 자동 컴포넌트 생성

```html
data-sap-ui-onInit="module:sap/ui/core/ComponentSupport"
```

UI5 초기화 완료 후, `ComponentSupport` 모듈이 자동으로 실행됩니다.

---

## 5. ComponentSupport — 컴포넌트 자동 생성

### 5.1 동작 방식

`ComponentSupport` 모듈은 다음을 수행합니다:

1. **DOM 스캔**: `document.querySelectorAll('[data-sap-ui-component]')` 로 모든 대상 div 검색
2. div 발견:

```html
<div
    data-sap-ui-component          ← 이 속성이 있으면 대상
    data-name="portal"             ← 로드할 컴포넌트 이름
    data-settings='{"id" : "portal"}'  ← 생성자에 전달할 설정
></div>
```

3. 자동 실행 (개발자가 직접 호출하지 않음):

```javascript
// ComponentSupport 내부 (의사코드)
sap.ui.require(["portal/Component"], function(ComponentClass) {
    new ComponentClass({
        id: "portal"   // data-settings에서 전달
    });
});
```

> 💡 `data-name="portal"` → `portal.Component` 로 변환됩니다. UI5는 `{name}.Component` 패턴으로 Component 클래스를 찾습니다.

4. `portal/Component.js` 로드:
```
→ sap.ui.loader 변환: "portal" → "./"
→ GET /portal_ott/webapp/Component.js
→ approuter localDir: ../../portal_ott/webapp/Component.js
```

#### 🔥 `data-name` vs `data-settings.id` — 완전히 다른 두 가지 역할

이 두 속성은 portal의 경우 **우연히 같은 값 `"portal"`** 이라서 헷갈리기 쉽지만, 하는 일이 완전히 다릅니다.

| | `data-name` | `data-settings.id` |
|---|---|---|
| **역할** | **어떤** Component 클래스를 로드할지 | 만들어진 인스턴스에 붙일 **ID** |
| **비유** | 붕어빵 틀 선택 (팥붕? 슈붕?) | 찍어낸 붕어빵 이름표 (1호, 2호...) |
| **파일 탐색?** | ✅ `resourceroots` 변환 → `portal/Component.js` | ❌ 파일 탐색에 전혀 관여 안 함 |
| **사용처** | ComponentSupport가 `.require()` 할 때 | `new ComponentClass()` 생성자 인자 |
| **결과물** | 로드된 JS 클래스 | DOM에 `<div id="portal">` |

##### 실제 내부 동작으로 보는 차이

```javascript
// ComponentSupport 내부 (의사코드)

// ① data-name → 어떤 파일을 require() 할지 결정
var sModuleName = "portal" + "/Component";  // "portal/Component"
sap.ui.require([sModuleName], function(ComponentClass) {
    //     ↑ 여기까지: data-name만 사용. 파일을 찾아서 클래스를 가져옴.

    // ② data-settings → 생성자에 그대로 전달
    var oSettings = JSON.parse('{"id" : "portal"}');
    new ComponentClass(oSettings);
    //                 ↑ 여기서 처음으로 data-settings.id 등장.
    //                   단순히 "인스턴스의 id 속성을 'portal'로 해줘" 일 뿐.
});
```

##### 왜 필요한가? — DOM에서의 자리

`data-settings.id`로 지정한 값은 최종 DOM에서 이 컴포넌트가 차지하는 `<div>`의 ID가 됩니다:

```html
<!-- index.html -->
<body id="content2">
  └─ <div id="portal">         ← ★ data-settings.id="portal" → 여기!
       └─ <div id="Layout">    ← rootView (manifest의 rootView.id)
            └─ <div id="app">  ← App 컨트롤 (manifest의 controlId)
                 └─ ...실제 페이지들...
```

##### 두 값이 꼭 같을 필요는 없다!

`data-name`과 `data-settings.id`는 **완전히 독립적**이므로 다른 값을 줘도 아무 문제 없이 동작합니다:

```html
<!-- 이렇게 해도 정상 작동함 -->
<div data-sap-ui-component
     data-name="detailPage"                      ← detailPage/Component.js 로드
     data-settings='{"id" : "myDetailApp"}'      ← DOM에는 id="myDetailApp"
></div>
```

> 🧠 **기억할 것**: `data-name`이 **"어떤 파일을"** 찾을지 결정하고, `data-settings.id`는 **"만들어진 결과물의"** DOM ID를 결정한다. portal이 둘 다 `"portal"`인 건 그냥 관례일 뿐이다.

---

## 6. Component.js 초기화 — `init()` 실행

### 6.1 전체 Component.js 코드

```javascript
sap.ui.define([
    "sap/ui/core/UIComponent",
    "sap/ui/core/BusyIndicator",
    "common_ott/lib/util/MenuLoader",
    "common_ott/lib/util/UserSessionLoader",
    "sap/ui/core/AppCacheBuster",
    "sap/ui/core/Theming",
],
function (UIComponent, BusyIndicator, MenuLoader, UserSessionLoader, AppCacheBuster, Theming) {
    "use strict";

    return UIComponent.extend("portal.Component", {
        metadata: {
            manifest: "json"      // ← manifest.json을 설계도로 사용
        },

        init: function () {
            // (A) 부모 init() — manifest 파싱 + 모델 생성 + rootView 렌더링
            UIComponent.prototype.init.apply(this, arguments);

            // (B) 라우터 이벤트 핸들러 등록
            const oRouter = this.getRouter();
            oRouter.attachBeforeRouteMatched(this.onBeforeRouteMatched, this);
            oRouter.attachRouteMatched(this.onRouteMatched, this);

            // (C) UserSessionLoader — 사용자 정보 비동기 로드
            new UserSessionLoader().attachEventOnce('ready', function(oEvent){
                let oModel = oEvent.getParameter("model");
                let oUserData = oModel.getData() || {};
                // 언어, 날짜, 숫자 포맷, 테마 설정
                const oCoreConfig = sap.ui.getCore().getConfiguration();
                const oFormatSettings = oCoreConfig.getFormatSettings();
                oFormatSettings.setDatePattern("short", oUserData.date_format_type_name || "yyyy-MM-dd");
                oFormatSettings.setDatePattern("medium", oUserData.date_format_type_name || "yyyy-MM-dd");
                oFormatSettings.setDatePattern("long", oUserData.date_format_type_name || "yyyy-MM-dd");
                oFormatSettings.setTimePattern("short", "HH:mm:ss");
                oFormatSettings.setTimePattern("medium", "HH:mm:ss");
                oFormatSettings.setTimePattern("long", "HH:mm:ss");
                let digitsGroup = oUserData.digits_format_type_name?.substr(3,1) || ",";
                let digitsDecimal = oUserData.digits_format_type_name?.substr(7,1) || ".";
                oFormatSettings.setNumberSymbol("group", digitsGroup);
                oFormatSettings.setNumberSymbol("decimal", digitsDecimal);
                oCoreConfig.setLanguage(oUserData.language_code || "EN");
                Theming.setTheme(oUserData.theme_code || "sap_fiori_3");
            }.bind(this));

            // (D) MenuLoader — 메뉴 정보 비동기 로드 → 라우트 동적 생성
            new MenuLoader().attachEventOnce('ready', (oEvent) => {
                const oModel = oEvent.getParameter("model");
                const oCore = sap.ui.getCore();
                if(!oCore.getModel("Menu")){
                    oCore.setModel(oModel, "Menu");
                }
                this._setRouteAndTarget(oModel.getData());
            });
        }
        // ... 나머지 메서드
    });
});
```

### 6.2 실행 순서 요약

`init()`이 호출되면 다음과 같은 순서로 진행됩니다:

| 순서 | 코드 | 설명 | 동기/비동기 |
|:--:|------|------|:--:|
| A | `UIComponent.prototype.init.apply(this, arguments)` | manifest 파싱 + 모델 생성 + rootView **트리거** + Router 생성 | 동기 (내부에서 비동기 시작은 함) |
| B | `oRouter.attachBeforeRouteMatched(...)` | 라우터 이벤트 리스너 등록 | 동기 |
| C | `new UserSessionLoader()` | 사용자 정보 로드 시작 | 비동기 |
| D | `new MenuLoader()` | 메뉴 정보 로드 시작 | 비동기 |

#### 🔥 핵심: `super.init()`은 Layout을 기다리지 않는다!

`super.init()`은 rootView 로딩을 **트리거만** 하고 즉시 리턴합니다. manifest.json의 `rootView.async: true` 때문에 Layout.view.xml과 Layout.controller.js의 로딩은 `super.init()`이 끝난 후에도 **백그라운드에서 계속 진행**됩니다.

```
Component.init() {
    super.init();    // ① rootView 로딩 "시작!" → 바로 리턴 (안 기다림!)
                     //    Router 객체 "생성!" → 동기
    
    this.getRouter() // ② Router는 이미 존재하므로 바로 사용 가능
        .attach...();
    
    new UserSessionLoader()  // ③ 비동기 시작
    new MenuLoader()         // ④ 비동기 시작
}  // init() 종료. Layout은 아직 로딩 중일 수도 있음.
```

#### Component의 남은 코드와 Layout 로딩은 독립적으로 진행된다

`Component.init()` 종료 직후부터 **두 갈래의 흐름**이 동시에 진행됩니다:

```
    Component 쪽                          Layout 쪽
    ─────────────                         ─────────
    UserSessionLoader → 응답 대기          Layout.view.xml 파싱 중...
    MenuLoader → 응답 대기                Layout.controller.js 로딩 중...
         │                                    │
         │                           Layout.controller.onInit()
         │                             → BaseController.onInit()
         │                               → 각종 Loader 시작
         │                             → setHeaderNavigation()
         │                                    │
         │                           Layout 렌더링 완료
         │                           <App id="app"> DOM에 존재!
         │                                    │
    MenuLoader 완료                            │
      → _setRouteAndTarget()                  │
        → Router.initialize()                 │
          → App에 Home 삽입 ←─────────────────┘
```

**이 두 흐름은 서로를 기다리지 않는다.** 각자 자기 할 일을 한다.

#### async Router의 suspend: App이 없으면 보류, 생기면 재개

`manifest.json`의 `routing.config.async: true` 덕분에, Router.initialize()는 `<App id="app">` 컨트롤이 아직 DOM에 없으면 **navigation을 보류(suspend)** 했다가 App이 생성되는 순간 **자동 재개**합니다:

```
MenuLoader 완료 → Router.initialize()
  ├─ byId("app") 존재? → 바로 Home 삽입 (일반적인 케이스)
  └─ byId("app") 없음? → navigation suspend (극단적 케이스)
       └─ Layout 파싱 완료 → App 생성 → 감지 → 자동 재개 → Home 삽입
```

#### App은 빈 컨테이너일 뿐 — 내부 데이터 로딩 상태는 신경 안 쓴다

Router.initialize()가 Home을 App에 삽입할 때, App은 **단순히 빈 바구니 역할**만 수행합니다. Home.view의 데이터가 로딩 중이든, Layout.controller.onInit()이 아직 안 끝났든, Router는 전혀 신경 쓰지 않고 그냥 Home을 App에 `addPage()` 합니다. 이후 Home.controller는 **자기 페이스대로** 데이터를 가져와서 UI를 점진적으로 채웁니다.

> 🧠 **핵심**: App이 존재하기만 하면, Router는 Layout.controller의 진행 상태와 **완전히 독립적으로** 작동한다. App 내부에 들어갈 view들의 데이터 로딩도 Router와는 무관하다.

---

## 7. [A] `super.init()` 상세 분석 — 가장 많은 일이 벌어지는 순간

`UIComponent.prototype.init()`는 SAP UI5 프레임워크 내부에서 다음과 같은 일을 순차적으로 수행합니다:

### 7.1 manifest.json 파싱

```json
{
    "_version": "1.59.0",
    "sap.app": {
        "id": "comportalott",        // ← 컴포넌트 공식 ID
        "type": "application",
        "dataSources": {
            "UserManagementService": {
                "uri": "/srv-api/odata/v2/ott-core/UserManagement",
                "type": "OData",
                "settings": { "odataVersion": "2.0" }
            }
        }
    },
    "sap.ui5": {
        "rootView": {
            "viewName": "portal.view.Layout",  // ← 첫 화면으로 보여줄 뷰
            "type": "XML",
            "async": true,
            "id": "Layout"
        },
        "models": {
            "i18n": {
                "type": "common_ott.lib.model.CoreResourceModel",
                "settings": { "bundleName": "common_ott.lib.i18n.i18n" }
            },
            "UserManagement": {
                "dataSource": "UserManagementService",
                "preload": true            // ← 즉시 데이터 로드 시작!
            }
        },
        "routing": {
            "config": {
                "routerClass": "sap.m.routing.Router",
                "viewType": "XML",
                "async": true,
                "controlAggregation": "pages",
                "controlId": "app",        // ← Layout.view.xml의 <App id="app">과 연결
                "clearControlAggregation": false
            },
            "routes": [],                  // ← 비어있음! MenuLoader가 나중에 채움
            "targets": {
                "Home": {
                    "type": "View",
                    "name": "Home",
                    "path": "portal.view",
                    "clearControlAggregation": true
                },
                "Search": { /* ... */ }
            }
        },
        "componentUsages": {}              // ← 비어있음! MenuLoader가 나중에 채움
    }
}
```

### 7.2 모델 생성

#### 7.2a i18n 모델

```json
"i18n": {
    "type": "common_ott.lib.model.CoreResourceModel",
    "settings": {
        "bundleName": "common_ott.lib.i18n.i18n"
    }
}
```

UI5가 `CoreResourceModel`을 생성하고, `common_ott/lib/i18n/i18n.properties` 파일을 로드합니다.

```
HTTP 요청: GET /common_ott.lib/i18n/i18n.properties
→ approuter localDir: ../../library_ott/src/i18n/i18n.properties
```

> 이 `i18n.properties`는 portal_ott뿐만 아니라 **모든 OTT 앱이 공유**하는 공통 번역 파일입니다. (예: "appTitle", 공통 버튼 레이블 등)

#### 7.2b UserManagement OData 모델 (`preload: true`)

```json
"UserManagement": {
    "dataSource": "UserManagementService",
    "preload": true
}
```

`preload: true`이므로 **모델 생성과 동시에 실제 HTTP 요청이 발생**합니다.

```
① $metadata 요청 (OData 모델 초기화):
   GET /srv-api/odata/v2/ott-core/UserManagement/$metadata

② 데이터 요청 (preload로 인한 자동 fetch):
   GET /srv-api/odata/v2/ott-core/UserManagement/UserSet?...
```

### 7.3 🔥 API 요청이 approuter를 통과하는 전체 과정 (핵심!)

#### 7.3.1 브라우저에서 HTTP 요청 발생

`UserManagement` OData 모델이 생성되면 브라우저는 다음 URL로 HTTP 요청을 보냅니다:

```
GET http://localhost:5000/srv-api/odata/v2/ott-core/UserManagement/$metadata
```

> 주의: 브라우저는 `http://localhost:5000`을 기준(base URL)으로 상대경로를 해석합니다. `index.html`이 `localhost:5000`에서 로드되었으므로 `/srv-api/...`는 `localhost:5000/srv-api/...`가 됩니다.

#### 7.3.2 approuter가 요청을 받고 라우트 매칭

approuter(port 5000)가 `GET /srv-api/odata/v2/ott-core/UserManagement/$metadata` 요청을 받습니다.

`dev/xs-app.json`의 routes를 **위→아래**로 검사:

```
요청: /srv-api/odata/v2/ott-core/UserManagement/$metadata

라우트 1: ^.*/srv-api/(.*ott-core/.*)$
```

정규표현식 매칭:
```
^               ← 문자열 시작
  .*            ← 아무 문자 0회 이상 (여기서는 빈 문자열 — 앞에 아무것도 없음)
  /srv-api/     ← 리터럴 매칭 "/srv-api/"
  (             ← 캡처 그룹 1 시작 ($1)
    .*          ← 아무 문자 0회 이상
    ott-core    ← 리터럴 "ott-core"
    /.*         ← "/" + 아무 문자
  )             ← 캡처 그룹 1 끝 → "odata/v2/ott-core/UserManagement/$metadata"
$               ← 문자열 끝

✅ 매칭 성공!
$1 = "odata/v2/ott-core/UserManagement/$metadata"
```

#### 7.3.3 destination 해석

매칭된 라우트:
```json
{
    "source": "^.*/srv-api/(.*ott-core/.*)$",
    "target": "$1",
    "destination": "onOttCoreService",
    "authenticationType": "none"
}
```

`destination: "onOttCoreService"` → approuter는 `dev/default-env.json`에서 이 이름을 찾습니다:

```json
{
    "destinations": [
        {
            "name": "onOttCoreService",
            "url": "http://localhost:8084",
            "forwardAuthToken": false
        },
        {
            "name": "onOttService",
            "url": "http://localhost:8083",
            "forwardAuthToken": false
        }
        // ...기타 destination
    ]
}
```

`onOttCoreService` → `http://localhost:8084`

#### 7.3.4 target 변환과 최종 프록시 URL

```
target: "$1" = "odata/v2/ott-core/UserManagement/$metadata"
destination URL = "http://localhost:8084"

최종 프록시 URL = destination URL + target
                = "http://localhost:8084/odata/v2/ott-core/UserManagement/$metadata"
```

#### 7.3.5 approuter가 백엔드로 요청 전달

approuter는 이제 **프록시** 역할을 합니다:

```
브라우저 ──GET──▶ approuter(5000) ──GET──▶ core_ott 백엔드(8084)
                 [중계자]                  [실제 데이터 소스]

① 브라우저 → approuter: GET /srv-api/odata/v2/ott-core/UserManagement/$metadata
② approuter → core_ott: GET /odata/v2/ott-core/UserManagement/$metadata
③ core_ott → approuter: 200 OK + $metadata XML 반환
④ approuter → 브라우저: 200 OK + $metadata XML 전달
```

#### 7.3.6 `authenticationType: "none"` 의 의미

로컬 개발 환경에서 `dev/xs-app.json`은 `"authenticationType": "none"`을 사용합니다. 이는 **인증 없이** 모든 요청을 통과시킨다는 뜻입니다.

운영 환경(`xs-app.json`)에서는 `"authenticationType": "xsuaa"`로 설정되어 XSUAA 인증이 필요합니다.

#### 7.3.7 전체 API 라우트 총정리

`dev/xs-app.json`에는 API 관련 라우트가 2개 있습니다:

| 우선순위 | source 패턴 | destination | 백엔드 |
|:--:|------|-------------|--------|
| 1 (먼저 검사) | `^.*/srv-api/(.*ott-core/.*)$` | `onOttCoreService` | core_ott (8084) |
| 2 | `^.*/srv-api/(.*)$` | `onOttService` | ott (8083) |

> 💡 **순서가 중요한 이유**: `/srv-api/odata/v2/ott-core/...` 요청은 두 패턴 모두에 매칭됩니다. 하지만 `ott-core` 패턴이 **먼저** 있으므로 core_ott(8084)로 전달됩니다. 만약 순서가 바뀌면 `(.*)$`가 먼저 매칭되어 ott(8083)로 잘못 전달됩니다.

### 7.4 rootView 로드 & 렌더링

```json
"rootView": {
    "viewName": "portal.view.Layout",
    "type": "XML",
    "async": true,
    "id": "Layout"
}
```

UI5가 `portal/view/Layout.view.xml`을 로드합니다:

```
→ sap.ui.loader 변환: "portal" → "./"
→ GET /portal_ott/webapp/view/Layout.view.xml
→ approuter localDir: ../../portal_ott/webapp/view/Layout.view.xml
```

---

## 8. Layout.view.xml 로드 & 렌더링

```xml
<mvc:View
    controllerName="portal.controller.Layout"
    xmlns="sap.m"
    xmlns:mvc="sap.ui.core.mvc"
    xmlns:tnt="sap.tnt"
    displayBlock="true"
>
    <tnt:ToolPage id="toolPage" >
        <tnt:header>
            <tnt:ToolHeader>
                <!-- 회사 로고 -->
                <Image src="/common_ott.lib/images/solum_logo.svg"
                       densityAware="false" decorative="false"
                       press="navToHome" width="7rem"/>

                <ToolbarSpacer />
                <!-- 검색창 -->
                <SearchField id="searchField" width="20%"
                    placeholder="{i18n>label.menuSearch}"
                    enableSuggestions="true"
                    search=".onSearch" suggest=".onSuggest"
                    suggestionItems="{ path: 'Menu>/',
                        filters: [{ path: 'menu_type_code', operator: 'EQ', value1: 'Menu' }]
                    }">
                    <SuggestionItem key="{Menu>menu_code}"
                        text="{ path:'Menu>menu_name', formatter: '.fnMenuNameI18n' }"
                        icon="{Menu>menu_icon}" />
                </SearchField>

                <ToolbarSpacer />
                <!-- 사용자명 표시 (User 모델이 아직 없으면 빈칸) -->
                <Label id="oDesplayName" design="Bold"
                    text="{User>/user_name} ({User>/business_partner_code})" />

                <!-- 설정 메뉴버튼 -->
                <MenuButton icon="sap-icon://action-settings"
                    buttonMode="Regular" useDefaultActionOnly="true">
                    <menu>
                        <Menu itemSelected="onMenuAction">
                            <MenuItem icon="sap-icon://person-placeholder"
                                text="{i18n>button.profile}" press="onPressUserProfile"/>
                            <MenuItem icon="sap-icon://log"
                                text="{i18n>button.logout}"
                                press="window.location.replace('/user/logout')"/>
                        </Menu>
                    </menu>
                </MenuButton>
            </tnt:ToolHeader>
        </tnt:header>

        <tnt:subHeader>
            <tnt:ToolHeader id="headerNavigation"/>  <!-- 동적 메뉴바가 들어갈 자리 -->
        </tnt:subHeader>

        <tnt:sideContent>
            <!-- 사이드바 (현재 비활성화) -->
        </tnt:sideContent>

        <tnt:mainContents>
            <!-- ★ 이 App이 라우터의 컨트롤. manifest의 controlId="app"과 일치 -->
            <App id="app" autoFocus="false" />
        </tnt:mainContents>
    </tnt:ToolPage>
</mvc:View>
```

### 8.1 Layout.view.xml 파싱 중 발생하는 일

1. `controllerName="portal.controller.Layout"` → `portal/controller/Layout.controller.js` 로드
2. `<Image src="/common_ott.lib/images/solum_logo.svg">` → 브라우저가 이미지 요청
   ```
   GET /common_ott.lib/images/solum_logo.svg
   → approuter localDir: ../../library_ott/src/images/solum_logo.svg
   ```
3. `<App id="app">` — manifest의 `routing.config.controlId: "app"`와 정확히 일치하는 ID. **라우터가 페이지를 삽입할 컨테이너**입니다.

---

## 9. Layout.controller.js — `onInit()` 실행

View가 로드되면 컨트롤러의 `onInit()`이 호출됩니다:

```javascript
onInit: function () {
    // (1) 부모 BaseController의 onInit → library_ott의 BaseController
    BaseController.prototype.onInit.call(this);
    // (2) 상단 메뉴바 설정 (MenuLoader 완료 후 실행됨)
    this.setHeaderNavigation();
}
```

### 9.1 `BaseController.prototype.onInit.call(this)` 상세 분석

portal의 `BaseController`는:
```javascript
// portal_ott/webapp/controller/BaseController.js
sap.ui.define([
    "common_ott/lib/controller/BaseController",
], function(BaseController) {
    "use strict";
    return BaseController.extend("portal.controller.App", {});
});
```

즉, `library_ott/src/controller/BaseController.js`의 `onInit()`이 실제로 실행됩니다.

#### 9.1.1 library_ott BaseController.onInit() 전체 흐름

```javascript
onInit : function ({FavoriteButton, PersoTable} = {}){
    const oCore = sap.ui.getCore();

    // ① UserSessionLoader — 사용자 세션 정보 로드
    new UserSessionLoader().attachEventOnce('ready', function(oEvent){
        let oModel = oEvent.getParameter("model");
        this.getView().setModel(oModel, "User");      // View에 User 모델 저장
        if(!oCore.getModel("User")){
            oCore.setModel(oModel, "User");            // Core에도 저장 (앱 간 공유)
        }
    }.bind(this));

    // ② CommonCodeLoader — 공통 코드 로드
    new CommonCodeLoader().attachEventOnce('ready', function(oEvent){
        let oModel = oEvent.getParameter("model");
        this.getView().setModel(oModel, "CommonCode");
        this._oCommonCode = oModel.getProperty("/");
        if(!oCore.getModel("CommonCode")){
            oCore.setModel(oModel, "CommonCode");
        }
    }.bind(this));

    // ③ MenuLoader — 메뉴 데이터 로드
    new MenuLoader().attachEventOnce('ready', (oEvent) => {
        const oModel = oEvent.getParameter("model");
        this.getView().setModel(oModel, "Menu");
        if(!oCore.getModel("Menu")){
            oCore.setModel(oModel, "Menu");
        }
    });

    // ④ MessageLoader — 메시지 로드
    new MessageLoader().attachEventOnce('ready', (oEvent) => {
        const oModel = oEvent.getParameter("model");
        this.getView().setModel(oModel, "Message");
        if(!oCore.getModel("Message")){
            oCore.setModel(oModel, "Message");
        }
    });

    // ⑤ MenuFunctionLoader — 메뉴별 기능 로드
    new MenuFunctionLoader().attachEventOnce('ready', (oEvent) => {
        const oModel = oEvent.getParameter("model");
        this.getView().setModel(oModel, "MenuFunction");
        if(!oCore.getModel("MenuFunction")){
            oCore.setModel(oModel, "MenuFunction");
        }
    });

    // ⑥ i18n 리소스 번들 참조 저장
    this.i18n = this.getResourceBundle();
}
```

**5개의 Loader가 모두 동시에(비동기로) 시작됩니다.**

### 9.2 🔥 API 요청 분석 — 각 Loader가 호출하는 백엔드 API

#### 9.2a UserSessionLoader — 사용자 정보

```javascript
// UserSessionLoader._fetchUserSession()
const oODataModel = new ODataModel("/srv-api/odata/v2/ott-core/UserManagement");
oODataModel.read("/UserSessionInfo", {
    success: resolve,
    error: reject
});
```

실제 HTTP 요청:
```
GET /srv-api/odata/v2/ott-core/UserManagement/UserSessionInfo?$format=json
```

approuter 처리:
```
① dev/xs-app.json 라우트 매칭:
   source: "^.*/srv-api/(.*ott-core/.*)$"
   $1 = "odata/v2/ott-core/UserManagement/UserSessionInfo?$format=json"

② destination 해석:
   "onOttCoreService" → "http://localhost:8084"

③ 최종 프록시:
   http://localhost:8084/odata/v2/ott-core/UserManagement/UserSessionInfo?$format=json

④ core_ott 백엔드가 UserSessionInfo 엔티티를 조회하여 사용자 정보 반환
   → { user_id: "U001", user_name: "홍길동", language_code: "KO", theme_code: "sap_fiori_3", ... }
```

#### 9.2b MenuLoader — 메뉴 정보

```javascript
// MenuLoader._fetchList()
const oODataModel = new ODataV4Model({
    serviceUrl: "/srv-api/odata/v4/ott-core/MenuManagement/"
});
const oBinding = oODataModel.bindList("/MenusRoleAppliedList");
oBinding.requestContexts();
```

실제 HTTP 요청:
```
GET /srv-api/odata/v4/ott-core/MenuManagement/MenusRoleAppliedList?...
```

approuter 처리:
```
① dev/xs-app.json 라우트 매칭:
   source: "^.*/srv-api/(.*ott-core/.*)$"
   $1 = "odata/v4/ott-core/MenuManagement/MenusRoleAppliedList?..."

② destination 해석:
   "onOttCoreService" → "http://localhost:8084"

③ 최종 프록시:
   http://localhost:8084/odata/v4/ott-core/MenuManagement/MenusRoleAppliedList?...

④ core_ott 백엔드가 Menu.csv 데이터를 권한 필터링하여 반환
```

#### 9.2c CommonCodeLoader — 공통 코드

```javascript
// V2 OData read
// serviceUrl: "/srv-api/odata/v2/ott-core/CommonCode"
```

approuter 처리: 동일한 `ott-core` 패턴 → core_ott(8084)

#### 9.2d MessageLoader — 메시지

동일한 core_ott(8084)로 라우팅

#### 9.2e MenuFunctionLoader — 메뉴 기능

동일한 core_ott(8084)로 라우팅

### 9.3 전체 API 요청 흐름도

```
브라우저 (localhost:5000에서 페이지 로드됨)
│
├─ manifest.json → UserManagement 모델 (preload: true)
│   └─ GET /srv-api/odata/v2/ott-core/UserManagement/$metadata
│       └─ approuter → onOttCoreService → http://localhost:8084/odata/v2/ott-core/UserManagement/$metadata
│
├─ BaseController.onInit() → UserSessionLoader
│   └─ GET /srv-api/odata/v2/ott-core/UserManagement/UserSessionInfo
│       └─ approuter → onOttCoreService → http://localhost:8084/odata/v2/ott-core/UserManagement/UserSessionInfo
│
├─ BaseController.onInit() → CommonCodeLoader
│   └─ GET /srv-api/odata/v2/ott-core/CommonCode/...
│       └─ approuter → onOttCoreService → http://localhost:8084/odata/v2/ott-core/CommonCode/...
│
├─ BaseController.onInit() → MenuLoader
│   └─ GET /srv-api/odata/v4/ott-core/MenuManagement/MenusRoleAppliedList
│       └─ approuter → onOttCoreService → http://localhost:8084/odata/v4/ott-core/MenuManagement/MenusRoleAppliedList
│
├─ BaseController.onInit() → MessageLoader
│   └─ GET /srv-api/odata/v2/ott-core/...  → approuter → core_ott(8084)
│
└─ BaseController.onInit() → MenuFunctionLoader
    └─ GET /srv-api/odata/v2/ott-core/...  → approuter → core_ott(8084)
```

> 💡 **핵심**: URL에 `ott-core`가 포함된 모든 API 요청은 `dev/xs-app.json`의 첫 번째 API 라우트(`(.*ott-core/.*)`)에 매칭되어 `onOttCoreService` destination(`http://localhost:8084`)으로 프록시됩니다.

---

## 10. [D] MenuLoader 완료 → `_setRouteAndTarget()` 실행

MenuLoader가 성공적으로 데이터를 받아오면, Component.init()에서 등록한 콜백이 실행됩니다:

```javascript
// Component.init() 내부
new MenuLoader().attachEventOnce('ready', (oEvent) => {
    const oModel = oEvent.getParameter("model");
    const oCore = sap.ui.getCore();
    if(!oCore.getModel("Menu")){
        oCore.setModel(oModel, "Menu");
    }
    this._setRouteAndTarget(oModel.getData());  // ← 여기!
});
```

### 10.1 MenuLoader가 반환하는 데이터 (Menu.csv 기반)

```javascript
[
    {
        menu_code: "OTT",
        parent_menu_code: null,
        menu_name: "OTT 서비스",
        menu_app_id: "OTT",
        menu_type_code: "Group",
        menu_repo_path: null,
        menu_app_path: null,
        menu_route_path: null
    },
    {
        menu_code: "OTT_MAIN",
        parent_menu_code: "OTT",
        menu_name: "메인",
        menu_app_id: "mainPage",
        menu_repo_path: "sysmgt_ott/webapp",
        menu_app_path: "/sysmgt_ott/mainPage/webapp",
        menu_route_path: ""               // ← 빈 문자열 (추가 경로 없음)
    },
    {
        menu_code: "OTT_DETAIL",
        parent_menu_code: "OTT",
        menu_name: "컨텐츠 상세",
        menu_app_id: "detailPage",
        menu_repo_path: "sysmgt_ott/webapp",
        menu_app_path: "/sysmgt_ott/detailPage/webapp",
        menu_route_path: "Detail/:content_id:"  // ← 내부 라우트 패턴
    },
    {
        menu_code: "OTT_MYPAGE",
        parent_menu_code: "OTT",
        menu_name: "마이페이지",
        menu_app_id: "myPage",
        menu_repo_path: "sysmgt_ott/webapp",
        menu_app_path: "/sysmgt_ott/myPage/webapp",
        menu_route_path: ""               // ← 빈 문자열
    }
    // ... OTT_BOARD, OTT_SETTLEMENT, OTT_TREND
]
```

### 10.2 `_setRouteAndTarget()` — 4단계 처리

#### 10.2a ① AppCacheBuster 등록

```javascript
const aRepoPaths = aMenuList.map(oTile => oTile.menu_repo_path).filter(Boolean);
const uniqueRepoPaths = [...new Set(aRepoPaths)];
// 결과: ["sysmgt_ott/webapp", "board/webapp", "settlement/webapp", "trendAnalysis/webapp"]

uniqueRepoPaths.forEach((path) => {
    AppCacheBuster.register(`/${path}`);
});
```

캐시 무효화를 위한 버전 토큰을 등록합니다. UI5 리소스 요청 시 `~버전~/` 형태의 경로로 변환됩니다.

#### 10.2b ② `sap.ui.loader.config()` — 하위 앱 네임스페이스 등록

```javascript
aMenuList.filter(o => !!o.menu_app_id).forEach(oTile => {
    const { menu_app_id: sTileId, menu_app_path } = oTile;

    if (menu_app_path) {
        const oResourceConfig = { [sTileId]: `${menu_app_path}` };
        sap.ui.loader.config({ paths: oResourceConfig });
    }
});
```

실행 결과:
```javascript
sap.ui.loader.config({ paths: { "mainPage": "/sysmgt_ott/mainPage/webapp" } });
sap.ui.loader.config({ paths: { "detailPage": "/sysmgt_ott/detailPage/webapp" } });
sap.ui.loader.config({ paths: { "myPage": "/sysmgt_ott/myPage/webapp" } });
sap.ui.loader.config({ paths: { "board": "/board" } });
sap.ui.loader.config({ paths: { "settlement": "/settlement" } });
sap.ui.loader.config({ paths: { "trendAnalysis": "/trendAnalysis" } });
```

> 💡 **이게 portal이 다른 앱의 파일을 찾을 수 있는 핵심 메커니즘입니다.** `detailPage/Component.js` → `sap.ui.loader`가 `/sysmgt_ott/detailPage/webapp/Component.js`로 변환 → approuter가 `localDir: ../../sysmgt_ott/webapp`에서 서빙.

#### 10.2c ③ 동적 Route 추가

```javascript
// OTT_MAIN 예시
oRouter.addRoute({
    pattern: "mainPage/",           // menu_route_path가 ""이므로
    name: "OTT_MAIN",
    target: {
        name: "OTT_MAIN",
        prefix: "mainPage"
    }
});

// OTT_DETAIL 예시
oRouter.addRoute({
    pattern: "detailPage/Detail/:content_id:",  // menu_route_path
    name: "OTT_DETAIL",
    target: {
        name: "OTT_DETAIL",
        prefix: "detailPage"
    }
});
```

#### 10.2d ④ 동적 Target + ComponentUsage 추가

```javascript
// 각 메뉴마다
oRouter.getTargets().addTarget(sMenuId, {
    viewId: sTileId,
    type: "Component",           // ← View가 아니라 Component를 로드!
    usage: sMenuId,
    title: menu_name,
    rootView: this.getAggregation("rootControl").getId()
});

oConfig.componentUsages[sMenuId] = {
    name: sTileId,               // "mainPage", "detailPage", "myPage" 등
    settings: {},
    componentData: {
        routePath: menu_route_path,
        MenuId: sMenuId,
        TileId: sTileId
    },
    lazy: true                   // ← 필요할 때만 로드 (초기 로딩 속도 향상)
};
```

### 10.3 `oRouter.initialize()` — 라우터 가동

```javascript
// _setRouteAndTarget()의 가장 마지막 줄
// 모든 라우트 정의가 끝난 후
this.oRouter.initialize();
```

#### 🔥 Router의 생명주기: 생성 ≠ 가동

Router는 `super.init()`에서 **생성**되지만, 이 시점에는 **시동이 꺼진(OFF) 상태**입니다. URL 해시가 바뀌어도 아무 일도 하지 않습니다. `initialize()`를 호출해야 비로소 **시동이 켜집니다(ON).**

```
super.init() 진행 중
  └─ Router 객체 생성 (메모리에만 존재. 시동 OFF)
       └─ URL 해시가 "detailPage/Detail/C001"이어도 → 완전히 무시

... 시간 흐름 ...

_setRouteAndTarget()
  ├─ routes[] 채움
  ├─ componentUsages{} 채움
  └─ Router.initialize() ← 시동 ON!
       └─ "자, 현재 해시가 뭐지?" → 읽는다 → 매칭 → target 실행
```

> 🧠 **비유**: Router 생성 = 자동차 조립 완료 (시동 OFF). `initialize()` = 시동 걸기. 시동 걸기 전에 내비 목적지를 입력해도 아무 일도 안 일어난다. 시동 거는 순간 내비가 "아까 입력된 목적지로 안내 시작!"

#### 왜 `_setRouteAndTarget()` 마지막에 호출하는가?

`manifest.json`의 초기 상태는 빈 껍데기입니다:

```json
"routes": [],
"componentUsages": {}
```

만약 `Router.initialize()`가 이 상태에서 호출되면, URL에 어떤 해시가 있어도 매칭되는 route가 없어서 아무것도 표시할 수 없습니다. 그래서 **반드시** `_setRouteAndTarget()`이 Menu.csv 데이터로 모든 route를 채운 후, 함수의 **마지막 줄**에서 initialize()를 호출합니다. 이 구조 덕분에 "routes가 비어 있어서 에러"는 원천적으로 불가능합니다.

#### URL 직통 접속도 안전하다

브라우저 주소창에 `http://localhost:5000/portal_ott/index.html#detailPage/Detail/C001`을 직접 입력해도 정상 작동합니다:

```
① index.html 로드 → Router 생성 (시동 OFF, 해시 무시)
② MenuLoader 완료 → _setRouteAndTarget() → routes 채움
③ Router.initialize() → 시동 ON!
   └─ "현재 해시가 detailPage/Detail/C001 이네?"
   └─ route 매칭 → OTT_DETAIL → detailPage Component 로드 ✅
```

Router가 시동 OFF 상태일 때는 해시 변경을 전혀 감지하지 않으므로, 시동을 켤 때까지 URL 해시는 그냥 무시됩니다. 시동 켜는 순간 **그 시점의 해시**를 읽고 첫 navigation을 실행합니다.

#### Router.initialize()가 하는 일 — 딱 두 가지

```javascript
// Router.initialize() 내부 (의사코드)
initialize: function() {
    // ① 현재 URL 해시를 읽고 매칭되는 target을 찾아 실행
    var sHash = window.location.hash.replace(/^#/, "");
    this._navigateToHash(sHash || "");  // 해시 없으면 기본 route("") → Home

    // ② 앞으로 URL 해시가 변경될 때마다 계속 감시
    window.addEventListener("hashchange", this._onHashChange.bind(this));
}
```

그게 전부입니다. "모든 데이터가 준비될 때까지 기다린다" 같은 건 없습니다.

#### Router는 target의 `type`만 보고 행동한다

Router가 매칭된 target을 처리할 때, 유일하게 보는 것은 target의 `type` 필드입니다:

```javascript
// Router 내부 target 처리 (의사코드)
function loadTarget(oTarget) {
    if (oTarget.type === "View") {
        // View target: xml 파일 하나만 로드해서 App에 바로 꽂는다. 끝.
        sap.ui.require([oTarget.path + "/" + oTarget.name + ".view.xml"], function(oView) {
            oApp.addPage(oView);
        });
    }
    
    else if (oTarget.type === "Component") {
        // Component target: Component를 통째로 생성해서 App에 꽂는다.
        // 생성된 Component는 자체 manifest, Router, 모델, 컨트롤러를 가진 완전한 독립 앱이다.
        sap.ui.component({
            name: oTarget.usage.componentName,
            lazy: oTarget.usage.lazy
        }).then(function(oComponent) {
            oApp.addPage(oComponent);
            // ← 이제부터 이 Component는 portal과 완전히 독립적으로 작동한다.
            //    portal은 더 이상 아무 관여도 하지 않는다.
        });
    }
}
```

| target type | Router의 역할 | 이후 |
|---|---|---|
| `"View"` (ex: Home) | View xml 로드 → App에 삽입 → **Router가 계속 관리** | 해당 View는 portal Router의 통제 아래에 있음 |
| `"Component"` (ex: detailPage) | Component 생성 → App에 삽입 → **portal은 손 뗌** | 생성된 Component가 자체 Router로 독립 작동 |

#### 🏢 portal은 건물주, detailPage는 독립 입점 가게

```
portal_ott (건물주)
├─ Layout.view.xml
│   ├─ [헤더] ← Layout.controller가 관리
│   ├─ [메뉴바] ← Layout.controller가 관리
│   └─ <App id="app"> ← 빈 공간 (임대 가능한 방)
│        │
│        ├─ Home.view ← portal Router가 직접 관리 (View target)
│        │   └─ Home.controller (portal Router의 통제 아래)
│        │
│        ├─ detailPage Component ← portal이 입점만 시켜줌 (Component target)
│        │   ├─ 자체 manifest.json  ← portal과 무관
│        │   ├─ 자체 Router         ← portal과 무관
│        │   ├─ 자체 OData 모델     ← portal과 무관
│        │   └─ 자체 컨트롤러       ← portal과 무관
│        │   ★ portal은 이 안에서 무슨 일이 일어나는지 전혀 모른다
│        │
│        └─ myPage Component ← 또 다른 독립 가게
│            └─ ... 자체 manifest, Router, 모델, 컨트롤러 ...
```

이게 바로 portal이 여러 앱을 통합하면서도 각 앱이 완전히 독립적으로 작동할 수 있는 핵심 설계입니다. portal은 단지 공간(App)을 제공하고, 헤더/메뉴바를 관리할 뿐, 각 앱의 내부 동작에는 전혀 관여하지 않습니다.

---

## 11. Home.view.xml 렌더링 — 첫 화면 표시

라우터가 `Home` 타겟을 `<App id="app">`에 렌더링합니다:

```xml
<!-- Home.view.xml -->
<mvc:View controllerName="portal.controller.Home">
    <Page title="..." showHeader="false">
        <content>
            <VBox>
                <Title text="데이터 현황 요약" />
                <HBox>
                    <GenericTile header="전체 자재" ... />
                    <GenericTile header="품목 승인 대기" ... />
                    <GenericTile header="품질 오류 항목" ... />
                </HBox>
                <!-- 차트, 리스트 등 -->
            </VBox>
        </content>
    </Page>
</mvc:View>
```

Home.controller.js의 `onInit()`도 함께 실행됩니다.

---

## 12. setHeaderNavigation() — 상단 메뉴바 생성

Layout.controller의 `onInit()`에서 호출한 `setHeaderNavigation()`도 MenuLoader 완료 후 실행됩니다:

```javascript
setHeaderNavigation: async function(){
    new MenuLoader().attachEventOnce('ready', async function (oEvent) {
        const oModel = oEvent.getParameter("model");
        const aMenuList = oModel.getData();
        const aFavoriteMenuList = await this.loadFavoriteMenu();
        const oMenuTree = this.buildTree(aMenuList, aFavoriteMenuList);

        // 상단메뉴바 동적생성
        this.createHeaderNavigation(oMenuTree);
    }.bind(this));
}
```

`buildTree()`가 메뉴를 부모-자식 트리 구조로 변환하고, `createHeaderNavigation()`이 `ToolHeader`에 `MenuButton`들을 동적으로 추가합니다.

---

## 13. 전체 실행 타임라인

> 💡 이 타임라인은 **일반적인 케이스**(Layout이 MenuLoader보다 먼저 완료되는 상황)를 기준으로 합니다. 두 흐름이 독립적이기 때문에 순서가 일부 바뀌어도 정상 작동합니다.

```
시간 →

0ms   │ sap-ui-core.js 다운로드 시작 (CDN)
      │
~200ms│ UI5 코어 초기화 완료
      │ ├─ resourceroots 등록: "portal" → "./", "common_ott.lib" → "/common_ott.lib"
      │ ├─ 라이브러리 preload: sap.m, sap.f, sap.uxap, sap.tnt, common_ott.lib
      │ └─ ComponentSupport 실행 → portal/Component.js 로드
      │
~400ms│ Component.init() 시작
      │ ├─ [A] super.init()                       ← 동기 블록
      │ │   ├─ manifest.json 파싱 (동기)
      │ │   ├─ i18n 모델 생성 (비동기 시작)
      │ │   ├─ UserManagement OData 모델 생성 (preload:true → $metadata 요청 시작)
      │ │   ├─ rootView(Layout.view.xml) 로딩 트리거 (async:true → 시작만!)
      │ │   └─ Router 객체 생성 (동기)
      │ │   ★ super.init() 리턴 — Layout 로딩은 백그라운드에서 계속 진행 중
      │ ├─ [B] Router 이벤트 리스너 등록 (동기)
      │ ├─ [C] UserSessionLoader 시작 (비동기)
      │ └─ [D] MenuLoader 시작 (비동기)
      │ ★ Component.init() 종료
      │
      │ ═══════════ 두 갈래 흐름이 독립적으로 진행 ═══════════
      │
      │ [Component 쪽 비동기]              [Layout 쪽 로딩]
~420ms│                                      Layout.view.xml 파싱
      │                                        ├─ ToolPage, ToolHeader, App 등
      │                                        │  컨트롤 객체들 생성
      │                                        │  ★ <App id="app"> 생성 완료!
      │                                        └─ Layout.controller.js 로드
      │
~450ms│                                      Layout.controller.onInit()
      │                                        ├─ BaseController.onInit()
      │                                        │  ├─ UserSessionLoader 시작
      │                                        │  ├─ CommonCodeLoader 시작
      │                                        │  ├─ MenuLoader 시작 (인스턴스 B)
      │                                        │  ├─ MessageLoader 시작
      │                                        │  └─ MenuFunctionLoader 시작
      │                                        ├─ this.i18n = getResourceBundle()
      │                                        └─ setHeaderNavigation()
      │                                           (MenuLoader ready 대기)
      │
~500ms│                                      Layout 렌더링 완료
      │                                        [헤더] [빈 App] [사이드바]
      │                                        ★ App은 이미 DOM에 존재
      │
      │ [Component]                          [Layout]
~700ms│ UserSessionLoader 완료                 Layout의 UserSessionLoader 완료
      │ ├─ 언어·날짜·테마 설정                  ├─ View/Core에 User 모델 저장
      │ └─ {User>/user_name} 업데이트            └─ {User>/user_name} 업데이트
      │
~800ms│ MenuLoader 완료 (인스턴스 A)            Layout의 MenuLoader 완료 (인스턴스 B)
      │ ├─ _setRouteAndTarget()                ├─ View/Core에 Menu 모델 저장
      │ │   ├─ AppCacheBuster 등록               └─ setHeaderNavigation()
      │ │   ├─ 하위 앱 sap.ui.loader.config()         → 상단 메뉴바 동적 생성
      │ │   ├─ 동적 Route + Target + Usage
      │ │   └─ Router.initialize()
      │ │       ├─ byId("app") → 있음! (Layout이 먼저 완료됨)
      │ │       └─ 바로 Home 타겟 → Home.view.xml 로드
      │ │           → oApp.addPage(oHomeView)
      │ │           ★ App은 단순 컨테이너. Home의 데이터 로딩 상태 무관
      │
~850ms│ Home.view.xml 렌더링
      │ └─ Home.controller.onInit() → OData 요청 시작 (타일 데이터)
      │
~900ms│ Home.controller 데이터 도착 → 타일 숫자 채워짐
      │
~1000ms│ 🎉 첫 화면 완전 표시!
       │   [로고] [OTT 서비스▼] ... [홍길동(U001)] [⚙]
       │   ┌─────────────────────────────────────┐
       │   │  데이터 현황 요약                      │
       │   │  [전체 자재] [품목 승인] [품질 오류]    │
       │   │  ...                                  │
       │   └─────────────────────────────────────┘
```

### 13.1 만약 순서가 바뀌면? — async Router의 suspend 메커니즘

극단적으로 MenuLoader가 Layout보다 먼저 완료되는 경우에도 UI5는 안전하게 동작합니다:

```
MenuLoader 완료 → Router.initialize()
  ├─ byId("app") → null (Layout XML 아직 파싱 안 됨)
  └─ ⚠️ async Router가 navigation을 suspend (보류)
       │
       │  ... Layout 파싱 진행 중 ...
       │
       └─ Layout 파싱 완료 → <App id="app"> 생성
            → Router가 감지 → 보류된 navigation 재개
            → Home.view.xml 로드 → App에 삽입
```

### 13.2 App이 이미 있을 때 Router.initialize() — 데이터 로딩과 무관

App이 존재하는 상태에서 Router.initialize()가 호출되면:

- Router는 App 컨트롤을 찾자마자 **즉시** Home을 삽입한다
- Layout.controller.onInit()이 아직 진행 중이어도, Home.view의 데이터가 로딩 중이어도 **전혀 상관하지 않는다**
- App은 단순한 컨테이너일 뿐이고, 각 view의 controller는 **독립적으로** 자기 데이터를 가져온다

```
<App id="app">                           ← 빈 바구니
  │
  ├─ Router가 Home 추가 → addPage()      ← 아무 조건 없이 그냥 넣음
  │   └─ Home.controller.onInit()         ← 얘는 알아서 자기 데이터 로드
  │       └─ readOdata() ⏳ 로딩 중...
  │
  └─ 한편 Layout.controller는            ← 완전히 독립적
      └─ setHeaderNavigation() 진행 중...
```

---

## 14. 핵심 요약: approuter API 프록시 흐름

```
┌──────────────────────────────────────────────────────────────────┐
│                        approuter (port 5000)                      │
│                                                                    │
│  dev/xs-app.json (라우트 규칙)           dev/default-env.json      │
│  ┌──────────────────────────┐         ┌──────────────────────┐    │
│  │ /srv-api/*ott-core/*     │────────▶│ onOttCoreService     │    │
│  │                          │         │  → localhost:8084    │    │
│  │ /srv-api/*               │────────▶│ onOttService         │    │
│  │                          │         │  → localhost:8083    │    │
│  │ /common_ott.lib/*        │──localDir: ../../library_ott/src   │
│  │ /portal_ott/*            │──localDir: ../../portal_ott/webapp │
│  │ /sysmgt_ott/*            │──localDir: ../../sysmgt_ott/webapp │
│  │ /*                       │──proxy→ ui (localhost:5001)        │
│  └──────────────────────────┘         └──────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘

브라우저 요청                    approuter 처리                       최종 목적지
─────────────                   ─────────────                       ──────────
/srv-api/...ott-core/...   →    onOttCoreService               →   localhost:8084 (core_ott)
/srv-api/...(그 외)...      →    onOttService                   →   localhost:8083 (ott)
/portal_ott/...             →    localDir에서 파일 읽기           →   디스크의 portal_ott/webapp/
/sysmgt_ott/...             →    localDir에서 파일 읽기           →   디스크의 sysmgt_ott/webapp/
/common_ott.lib/...         →    localDir에서 파일 읽기           →   디스크의 library_ott/src/
/board/...                  →    localDir에서 파일 읽기           →   디스크의 board/webapp/
```

---

## 15. 절대 잊지 말 것

| # | 핵심 |
|---|------|
| 1 | **approuter는 두 가지 역할**: ① 정적 파일 서빙(`localDir`), ② API 프록시(`destination`) |
| 2 | **`dev/xs-app.json`이 로컬 라우트 규칙** — routes는 위→아래 순서로 매칭, 첫 번째 매칭이 승리 |
| 3 | **`dev/default-env.json`이 destination 주소록** — `name`으로 참조, `url`이 실제 백엔드 주소 |
| 4 | **`srv-api` 경로에 `ott-core` 포함 여부로 백엔드 분기** — core_ott(8084) vs ott(8083) |
| 5 | **`data-sap-ui-resourceroots`는 UI5 모듈 경로의 "주소록"** — `"portal": "./"`는 현재 HTML 디렉토리 기준 |
| 6 | **`ComponentSupport`가 DOM을 스캔하여 Component 자동 생성** — 개발자가 `new` 호출 안 함 |
| 7 | **`preload: true` 시 OData 모델 생성 즉시 `$metadata` 요청 발생** |
| 8 | **UserSessionLoader, MenuLoader 등은 모두 `srv-api/ott-core/...` 경로로 core_ott(8084)에 요청** |
| 9 | **`super.init()`은 Layout을 기다리지 않는다** — `async:true` rootView는 트리거만 하고 즉시 리턴 |
| 10 | **Component와 Layout은 독립적으로 실행된다** — Router 등록 직후부터 두 갈래 흐름이 동시에 진행 |
| 11 | **Router는 생성 ≠ 가동** — `super.init()`에서 생성(OFF), `initialize()` 호출 시 가동(ON). OFF 상태에선 해시 무시 |
| 12 | **`_setRouteAndTarget()`이 routes 채운 후 마지막 줄에서 `initialize()` 호출** — 빈 routes로 매칭 시도할 일이 원천적으로 없다 |
| 13 | **`Router.initialize()`는 단순히 "현재 해시에 맞는 target을 App에 꽂는 기계"** — 데이터 준비를 기다리지 않는다 |
| 14 | **Router는 target의 `type`만 보고 View면 xml 로드, Component면 통째로 생성** — 그 이상의 일은 하지 않는다 |
| 15 | **portal은 건물주, detailPage/myPage는 독립 입점 가게** — 각자 자체 manifest, Router, 모델, 컨트롤러를 가진다 |
| 16 | **async Router는 App이 없으면 suspend, 생기면 재개** — `routing.config.async:true` 덕분에 안전 |
| 17 | **App은 빈 컨테이너** — 내부 view들의 데이터 로딩 상태와 무관하게 `addPage()` 가능 |
| 18 | **MenuLoader 완료 후에야 Router.initialize() 실행** — 메뉴 데이터로 동적 라우트 생성이 먼저 |
| 19 | **`lazy: true`** — 하위 앱(mainPage, detailPage 등)은 사용자가 메뉴 클릭 시점에 로드 |
| 20 | **각 view/component의 controller는 독립적** — 누가 누구를 기다리지 않는다. 점진적 렌더링(progressive rendering) |

---

> **문서 버전**: v1.0 · 2026-07-08
> **참고 기록**: `C:\Users\Eric\github\ott 기록\컴팩트_2026-07-08.md`, `컴팩트_2026-07-07.md`
> **관련 Obsidian 문서**: `2026-07-07-detailPage-myPage-portal-연동-전체-과정.md`
