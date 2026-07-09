# approuter_ott 전체 실행 흐름 완전 분석

> **작성일**: 2026-07-09
> **목적**: approuter_ott가 실행되어 `index.html` → `AuthCheck` → `portal_ott/index.html`까지 도달하는 전 과정을 한 줄 한 줄 완전히 이해한다.
> **핵심 질문**: `npm run start:local` → 브라우저 접속 → `location.href="/srv-api/odata/v4/frame/AuthCheck"` → `dev/xs-app.json` 라우트 매칭 → `dev/default-env.json` destination → `portal_ott/index.html` 도달까지의 전 과정

---

## 📁 관련 파일 위치

| 구성요소 | 경로 |
|----------|------|
| approuter 루트 | `cap-app/approuter_ott/` |
| approuter 진입점 (welcome) | `cap-app/approuter_ott/webapp/index.html` |
| approuter manifest | `cap-app/approuter_ott/webapp/manifest.json` |
| 로컬 라우트 규칙 | `cap-app/approuter_ott/dev/xs-app.json` |
| 로컬 destination 주소록 | `cap-app/approuter_ott/dev/default-env.json` |
| 운영 라우트 규칙 | `cap-app/approuter_ott/xs-app.json` |
| DEV 라우트 규칙 | `cap-app/approuter_ott/xs-app-dev.json` |
| QA 라우트 규칙 | `cap-app/approuter_ott/xs-app-qa.json` |
| 에러 페이지 | `cap-app/approuter_ott/error/404.html`, `error/500.html` |
| XSUAA 보안 설정 | `cap-app/approuter_ott/xs-security.json` |
| MTA 배포 설정 | `cap-app/approuter_ott/mta.yaml` |
| portal_ott | `cap-app/portal_ott/webapp/` |
| library_ott (공통 라이브러리) | `cap-app/library_ott/src/` |
| core_ott 백엔드 | `cap-node/core_ott/` (port 8084) |
| ott 백엔드 | `cap-node/ott/` (port 8083) |

---

## 0. approuter의 정체 — @sap/approuter

### 0.1 approuter란 무엇인가

`@sap/approuter`는 SAP BTP(Business Technology Platform) 환경에서 사용하는 **Node.js 기반 애플리케이션 라우터**다. 하나의 진입점에서:

1. **정적 파일 서빙** — HTML, JS, CSS, 이미지 등 (`localDir`)
2. **백엔드 API 프록시** — OData 서비스, REST API 등 (`destination`)
3. **사용자 인증** — XSUAA 연동 로그인/로그아웃 (`authenticationType`)
4. **세션 관리** — 세션 타임아웃, 쿠키 관리

의 네 가지 역할을 동시에 수행한다. 브라우저는 approuter 하나만 바라보면 되고, approuter가 뒤에서 모든 백엔드 서비스와 정적 파일을 중계해준다.

### 0.2 approuter의 두 가지 운영 모드

| 모드 | 설정 파일 | 인증 | 정적 파일 |
|------|----------|------|-----------|
| **로컬 개발** | `dev/xs-app.json` + `dev/default-env.json` | `"none"` (인증 없음) | `localDir` (디스크 직접 읽기) |
| **BTP 운영** | `xs-app.json` (프로젝트 루트) | `"xsuaa"` (XSUAA 인증) | `service: "html5-apps-repo-rt"` (HTML5 App Repo에서 서빙) |

> 💡 **핵심 차이**: 로컬에서는 `localDir`로 디스크에서 직접 파일을 읽지만, 운영에서는 `html5-apps-repo-rt` 서비스를 통해 BTP의 중앙 저장소에서 파일을 가져온다.

---

## 1. 실행 전 준비 — 3개의 터미널

```bash
# 터미널 1: OTT 비즈니스 백엔드 (port 8083)
cd cap-node/ott && npx cds watch --port 8083

# 터미널 2: core_ott 공통 백엔드 (port 8084) — UserManagement, Menu 등
cd cap-node/core_ott && npx cds watch --port 8084

# 터미널 3: approuter (port 5000) — 정적파일 서빙 + API 프록시
cd cap-app/approuter_ott && npm run start:local

# (선택) 터미널 4: ui5 serve (port 5001) — welcome 페이지 로컬 개발 서버
cd cap-app/approuter_ott && npm run start:ui
```

### 1.1 `start:local` 명령어 분석

```json
// package.json
"start:local": "node node_modules/@sap/approuter/approuter.js --workingDir ./dev --port 5000"
```

| 옵션 | 의미 |
|------|------|
| `node node_modules/@sap/approuter/approuter.js` | approuter 엔진을 Node.js로 직접 실행 |
| `--workingDir ./dev` | 작업 디렉토리를 `dev/`로 지정. 즉 `dev/xs-app.json`을 라우트 규칙으로, `dev/default-env.json`을 destination 설정으로 읽음 |
| `--port 5000` | HTTP 서버를 5000번 포트에 바인딩 |

> ⚠️ **중요**: `--workingDir ./dev` 때문에 approuter 루트에 있는 운영용 `xs-app.json`은 **완전히 무시된다!** 로컬 개발 중에는 오직 `dev/xs-app.json` + `dev/default-env.json`만 사용한다.

### 1.2 `start:ui` 명령어 (참고)

```json
"start:ui": "ui5 serve --port 5001"
```

UI5 툴링 서버를 5001번 포트로 실행한다. `webapp/` 디렉토리를 정적 파일 루트로 삼아 서빙한다. `dev/default-env.json`의 `"ui"` destination이 `http://localhost:5001`을 바라보기 때문에, approuter의 최하단 catch-all 라우트가 이 서버로 요청을 프록시한다.

---

## 2. 전체 실행 흐름 — 한눈에 보기

```
사용자가 브라우저에 http://localhost:5000 입력
│
├─ ① approuter: welcomeFile → /index.html
│     └─ route 매칭: ^/(.*)$ → destination "ui" → http://localhost:5001
│        └─ ui5 serve(5001) → webapp/index.html 반환
│
├─ ② 브라우저가 webapp/index.html 파싱
│     └─ <script>location.href="/srv-api/odata/v4/frame/AuthCheck"</script>
│        └─ 즉시 리다이렉트! → GET /srv-api/odata/v4/frame/AuthCheck
│
├─ ③ approuter: AuthCheck 요청 처리
│     └─ route 매칭: ^.*/srv-api/(.*)$ → destination "onOttService"
│        └─ 프록시: http://localhost:8083/odata/v4/frame/AuthCheck
│           └─ ott 백엔드(8083)가 인증 체크 수행 → 리다이렉트 응답
│
├─ ④ 브라우저: /portal_ott/index.html 로 리다이렉트
│     └─ approuter: route 매칭 → ^(.*)/portal_ott/(.*)$
│        └─ localDir: "../../portal_ott/webapp"
│           └─ 디스크에서 portal_ott/webapp/index.html 서빙
│
└─ ⑤ portal_ott SPA 로딩 시작
      └─ sap-ui-core.js → ComponentSupport → Component.init() → ...
```

---

## 3. ① welcomeFile — 첫 진입

### 3.1 사용자 요청

```
브라우저 주소창: http://localhost:5000
실제 HTTP 요청: GET /
```

### 3.2 `welcomeFile` 동작

```json
// dev/xs-app.json 최상단
"welcomeFile": "/index.html"
```

approuter는 루트 경로(`/`)로 요청이 들어오면 `welcomeFile`에 지정된 경로(`/index.html`)로 내부 재처리한다. 즉 `GET /`는 내부적으로 `GET /index.html`로 변환되어 라우트 매칭을 시작한다.

> 💡 **동작 방식**: 리다이렉트(302)가 아닌 **내부 재작성(internal rewrite)**이다. 브라우저 주소창은 그대로 `http://localhost:5000`을 유지하지만, approuter는 `/index.html` 요청을 처리한 것처럼 라우트 테이블을 검사한다.

### 3.3 `/index.html` 라우트 매칭

`dev/xs-app.json`의 `routes` 배열을 **위→아래** 순서로 검사. **첫 번째 매칭되는 라우트가 승리**한다.

```
요청 경로: /index.html

라우트  1: ^.*/srv-api/(.*ott-core/.*)$   → "srv-api" 없음 → ❌
라우트  2: ^.*/srv-api/(.*)$               → "srv-api" 없음 → ❌
라우트  3: ^.*/erp-api/(.*)$               → "erp-api" 없음 → ❌
라우트  4: ^.*/scim-api/(.*)$              → "scim-api" 없음 → ❌
라우트  5: /common_ott.lib/(.*)            → "common_ott.lib" 없음 → ❌
라우트  6: /common.lib/(.*)                → "common.lib" 없음 → ❌
라우트  7: ^(.*)/template/(.*)$            → "template" 없음 → ❌
라우트  8: ^(.*)/sysmgt/(.*)$              → "sysmgt" 없음 → ❌
라우트  9: ^(.*)/board/(.*)$               → "board" 없음 → ❌
라우트 10: ^(.*)/settlement/(.*)$          → "settlement" 없음 → ❌
라우트 11: ^(.*)/trendAnalysis/(.*)$       → "trendAnalysis" 없음 → ❌
라우트 12: ^/portal/(.*)$                  → "portal" ≠ index.html 시작 → ❌
라우트 13: ^(.*)/portal_ott/(.*)$          → "portal_ott" 없음 → ❌
라우트 14: ^(.*)/sysmgt_ott/(.*)$          → "sysmgt_ott" 없음 → ❌
라우트 15: ^/(.*)$                          → ✅ 매칭!
```

### 3.4 Catch-all 라우트 — `^/(.*)$`

```json
{
    "csrfProtection": false,
    "source": "^/(.*)$",
    "target": "/$1",
    "destination": "ui",
    "authenticationType": "none"
}
```

정규표현식 매칭:
```
^/(.*)$
│ │   │
│ │   └─ 캡처 그룹 1 ($1): / 다음의 모든 것 → "index.html"
│ └─ 리터럴 "/"
└─ 문자열 시작

$1 = "index.html"
```

### 3.5 Destination 해석

`destination: "ui"` → `dev/default-env.json`에서 찾는다:

```json
{
    "name": "ui",
    "url": "http://localhost:5001",
    "forwardAuthToken": false
}
```

### 3.6 target 변환 + 최종 프록시 URL

```
target: "/$1" = "/index.html"
destination URL = "http://localhost:5001"

최종 프록시 URL = destination URL + target
                = "http://localhost:5001/index.html"
```

### 3.7 approuter → ui5 serve 프록시

```
브라우저 ──GET /──▶ approuter(5000) ──GET /index.html──▶ ui5 serve(5001)
                    [중계자]                              [정적 파일 서버]

approuter는 요청을 그대로 ui5 serve에 전달하고,
ui5 serve는 webapp/index.html 파일을 읽어서 반환한다.
```

ui5 serve(5001)는 `ui5.yaml` 설정에 따라 `webapp/` 디렉토리를 정적 파일 루트로 사용한다. 따라서 `/index.html` 요청은 `webapp/index.html` 파일로 매핑된다.

---

## 4. ② `webapp/index.html` — AuthCheck 리다이렉트

### 4.1 전체 코드

```html
<!DOCTYPE html>
<html>
    <head>
        <script type="text/javascript"> 
            location.href="/srv-api/odata/v4/frame/AuthCheck";
        </script>
    </head>
</html>
```

### 4.2 동작 분석

이 HTML은 단 하나의 역할만 한다: **즉시 `/srv-api/odata/v4/frame/AuthCheck`로 리다이렉트**.

```javascript
location.href = "/srv-api/odata/v4/frame/AuthCheck";
```

- `location.href`는 브라우저의 현재 페이지 URL을 바꾼다
- `/srv-api/odata/v4/frame/AuthCheck`는 절대경로(상대경로 아님 — `/`로 시작)
- 기준 URL이 `http://localhost:5000`이므로 최종 URL은 `http://localhost:5000/srv-api/odata/v4/frame/AuthCheck`
- 브라우저는 이 URL로 **즉시 네비게이션**한다 (history에 새 엔트리 추가)

> 💡 **이 파일이 필요한 이유**: approuter의 시작점이지만, approuter 자체만으로는 인증 상태를 알 수 없다. ott 백엔드에 "이 사용자가 누구인지, 인증되었는지"를 물어봐야 한다. 이 리다이렉트가 그 물어보기 요청이다.

### 4.3 브라우저가 보내는 실제 HTTP 요청

```
GET /srv-api/odata/v4/frame/AuthCheck HTTP/1.1
Host: localhost:5000
```

---

## 5. ③ AuthCheck 요청 — approuter의 API 프록시

### 5.1 라우트 매칭

approuter(5000)가 `GET /srv-api/odata/v4/frame/AuthCheck` 요청을 받고, 다시 `dev/xs-app.json`의 routes를 **위→아래**로 검사한다.

```
요청 경로: /srv-api/odata/v4/frame/AuthCheck

라우트 1: ^.*/srv-api/(.*ott-core/.*)$
```

### 5.2 🔥 라우트 1 검사 — `ott-core` 패턴

```
정규표현식: ^.*/srv-api/(.*ott-core/.*)$

^               ← 문자열 시작
  .*            ← 아무 문자 0회 이상 (여기서는 빈 문자열)
  /srv-api/     ← 리터럴 매칭 "/srv-api/" ✅ 발견!
  (             ← 캡처 그룹 1 시작 ($1)
    .*          ← 아무 문자 0회 이상
    ott-core    ← 🔍 리터럴 "ott-core" 찾기...
```

`/srv-api/odata/v4/frame/AuthCheck` 안에 `ott-core`라는 문자열이 있는가? **없다!**

```
/srv-api/odata/v4/frame/AuthCheck
                    ↑ "frame" ≠ "ott-core"
```

→ ❌ **매칭 실패!** 라우트 1은 통과하지 못한다.

> 💡 라우트 1의 목적: `ott-core`를 포함하는 API 경로를 core_ott 백엔드(8084)로 보내기 위함. AuthCheck는 ott-core가 아니므로 ott 백엔드로 간다.

### 5.3 라우트 2 — 범용 `srv-api` 패턴

```
라우트 2: ^.*/srv-api/(.*)$
```

```
정규표현식: ^.*/srv-api/(.*)$

^               ← 문자열 시작
  .*            ← 아무 문자 0회 이상 (빈 문자열)
  /srv-api/     ← 리터럴 매칭 "/srv-api/" ✅ 발견!
  (             ← 캡처 그룹 1 시작 ($1)
    .*          ← 아무 문자 0회 이상 → "odata/v4/frame/AuthCheck"
  )             ← 캡처 그룹 1 끝
$               ← 문자열 끝
```

✅ **매칭 성공!**
```
$1 = "odata/v4/frame/AuthCheck"
```

매칭된 라우트:
```json
{
    "csrfProtection": false,
    "source": "^.*/srv-api/(.*)$",
    "destination": "onOttService",
    "cacheControl": "no-cache, no-store, must-revalidate",
    "target": "$1",
    "authenticationType": "none"
}
```

### 5.4 Destination 해석

```json
// dev/default-env.json
{
    "name": "onOttService",
    "url": "http://localhost:8083",
    "forwardAuthToken": false
}
```

### 5.5 최종 프록시 URL

```
target: "$1" = "odata/v4/frame/AuthCheck"
destination URL = "http://localhost:8083"

최종 프록시 URL = http://localhost:8083 + / + odata/v4/frame/AuthCheck
                = http://localhost:8083/odata/v4/frame/AuthCheck
```

### 5.6 프록시 흐름

```
브라우저                        approuter(5000)                 ott 백엔드(8083)
───────                        ────────────────                 ────────────────

① GET /srv-api/odata/v4/
     frame/AuthCheck ─────▶  ② xs-app.json 매칭
                              └─ route: ^.*/srv-api/(.*)$
                              └─ destination: onOttService
                              └─ target: odata/v4/frame/AuthCheck
                                                        
                              ③ GET /odata/v4/frame/AuthCheck ──▶
                                                                 ④ AuthCheck 처리
                                                                   └─ 세션 확인
                                                                   └─ 미인증 → 로그인 페이지로 리다이렉트
                                                                   └─ 인증됨 → portal_ott로 리다이렉트
                                                        
◀────────────────────────── ⑤ HTTP 302 Location: /portal_ott/index.html (또는 로그인 페이지)
```

### 5.7 `authenticationType: "none"` 의 의미

로컬 개발 환경에서는 `"authenticationType": "none"`으로 설정되어 있다:

```json
// dev/xs-app.json의 모든 라우트
"authenticationType": "none"
```

즉, approuter 자체는 **어떤 인증도 수행하지 않고** 모든 요청을 그대로 백엔드로 통과시킨다. 인증 로직은 백엔드(core_ott, ott)가 알아서 처리한다.

운영 환경(`xs-app.json`)에서는 `"authenticationType": "xsuaa"`로 설정되어 XSUAA 인증이 강제된다.

### 5.8 전체 API 라우트 우선순위

`dev/xs-app.json`에는 API 관련 라우트가 2개 있다:

| 우선순위 | source 패턴 | destination | 백엔드 | 설명 |
|:--:|------|-------------|--------|------|
| 1 (먼저 검사) | `^.*/srv-api/(.*ott-core/.*)$` | `onOttCoreService` | core_ott (8084) | ott-core 포함 API |
| 2 | `^.*/srv-api/(.*)$` | `onOttService` | ott (8083) | 그 외 모든 srv-api |

> 💡 **순서가 중요한 이유**: `/srv-api/odata/v2/ott-core/UserManagement/...` 같은 요청은 두 패턴 모두에 매칭된다. 하지만 `ott-core` 패턴이 **먼저** 있으므로 core_ott(8084)로 올바르게 전달된다. 만약 순서가 바뀌면 라우트 2의 `(.*)$`가 먼저 매칭되어 ott(8083)로 잘못 전달된다.

---

## 6. ④ AuthCheck 완료 → portal_ott로 리다이렉트

ott 백엔드(8083)의 `/odata/v4/frame/AuthCheck` 핸들러가 인증 상태를 확인한 후:

- **인증 성공 시**: `HTTP 302 Location: /portal_ott/index.html`
- **인증 실패 시**: 로그인 페이지로 리다이렉트 (로컬 개발에서는 보통 인증을 통과시킴)

브라우저는 302 응답을 받고 `/portal_ott/index.html`로 네비게이션한다.

### 6.1 `GET /portal_ott/index.html` 라우트 매칭

```
요청 경로: /portal_ott/index.html

라우트  1: ^.*/srv-api/(.*ott-core/.*)$   → "srv-api" 없음 → ❌
라우트  2: ^.*/srv-api/(.*)$               → "srv-api" 없음 → ❌
라우트  3: ^.*/erp-api/(.*)$               → "erp-api" 없음 → ❌
라우트  4: ^.*/scim-api/(.*)$              → "scim-api" 없음 → ❌
라우트  5: /common_ott.lib/(.*)            → "common_ott.lib" 없음 → ❌
라우트  6: /common.lib/(.*)                → "common.lib" 없음 → ❌
라우트  7: ^(.*)/template/(.*)$            → "template" 없음 → ❌
라우트  8: ^(.*)/sysmgt/(.*)$              → "sysmgt" 없음 → ❌
라우트  9: ^(.*)/board/(.*)$               → "board" 없음 → ❌
라우트 10: ^(.*)/settlement/(.*)$          → "settlement" 없음 → ❌
라우트 11: ^(.*)/trendAnalysis/(.*)$       → "trendAnalysis" 없음 → ❌
라우트 12: ^/portal/(.*)$                  → "portal" ≠ "portal_ott" → ❌
라우트 13: ^(.*)/portal_ott/(.*)$          → ✅ 매칭!
```

### 6.2 매칭된 라우트

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

### 6.3 정규표현식 캡처 그룹 해석

```
^(.*)/portal_ott/(.*)$
│     │              │
│     │              └─ 캡처 그룹 2 ($2): portal_ott/ 다음의 모든 것 → "index.html"
│     └─ 캡처 그룹 1 ($1): /portal_ott 이전의 모든 것 → "" (빈 문자열)
└─ 문자열 시작
```

### 6.4 `target` + `localDir` 조합

```json
"target": "$2",                            // → "index.html"
"localDir": "../../portal_ott/webapp"      // 디스크 경로
```

approuter가 디스크에서 읽는 실제 파일 경로:
```
{localDir}/{target}
= ../../portal_ott/webapp/index.html
= cap-app/portal_ott/webapp/index.html
```

### 6.5 `localDir`의 의미

> 💡 **`localDir`**: "이 경로에 매칭되는 요청은 프록시(백엔드로 전달)하지 말고, 내 로컬 디스크에서 파일을 읽어서 바로 HTTP 응답으로 반환해라"

approuter는 디스크 파일을 읽어서 `content-type: text/html`과 함께 브라우저에 HTTP 200 응답으로 반환한다. 이제 portal_ott SPA가 로딩되기 시작한다.

---

## 7. ⑤ portal_ott 로딩 — 이후 흐름

이 시점부터는 [portal_ott 전체 실행 흐름 완전 분석](https://...) 문서의 3절부터 이어진다:

1. `index.html` → `sap-ui-core.js` 다운로드 (CDN)
2. `data-sap-ui-resourceroots` 등록: `"portal": "./"`, `"common_ott.lib": "/common_ott.lib"`
3. `data-sap-ui-libs`로 라이브러리 preload (common_ott.lib도 포함)
4. `ComponentSupport` → `portal/Component.js` 자동 생성
5. `Component.init()` → manifest 파싱 → 모델 생성 → rootView 로딩 트리거
6. `UserSessionLoader`, `MenuLoader` 등 비동기 로더 시작
7. API 요청(`/srv-api/...`) → approuter → backends

이후 모든 `/srv-api/...ott-core/...` API 요청은 core_ott(8084)로, `/srv-api/...` (ott-core 미포함) 요청은 ott(8083)로, 정적 파일 요청은 해당 `localDir`에서 서빙된다.

---

## 8. ⚠️ approuter ≠ SAPUI5 Router — 이름은 비슷하지만 완전히 다르다

| | **approuter** (@sap/approuter) | **SAPUI5 Router** (sap.m.routing.Router) |
|---|---|---|
| **정체** | Node.js HTTP 서버 프로세스 | 브라우저 메모리 내 JS 객체 |
| **위치** | 서버 (터미널에서 실행) | 브라우저 (사용자 디바이스) |
| **하는 일** | HTTP 요청 라우팅 + 파일 서빙 + API 프록시 + 인증 | URL 해시(`#`) 기반 화면 전환 (SPA 내비게이션) |
| **설정 파일** | `xs-app.json` (라우트 규칙) | `manifest.json`의 `routing` 섹션 |
| **실행 시점** | 터미널에서 `npm start` | SPA가 브라우저에 로드된 후 |
| **포트** | 5000 (설정 가능) | 없음 (브라우저 내부) |

> 🧠 **기억할 것**: approuter는 "서버 앞의 교통정리 경찰관"이고, SAPUI5 Router는 "SPA 안의 내비게이션 안내원"이다. 둘 다 "라우터"라고 불리지만 역할과 위치가 완전히 다르다.

---

## 9. 🔥 `localDir` vs `destination` vs `service` — approuter의 3가지 처리 방식

approuter가 요청을 처리하는 방식은 크게 3가지다:

| 방식 | 설정 키 | 사용 환경 | 동작 |
|------|---------|----------|------|
| **로컬 파일 서빙** | `localDir` | 로컬 개발 (`dev/xs-app.json`) | 디스크에서 파일을 읽어 응답 |
| **백엔드 프록시** | `destination` | 로컬 + 운영 | `default-env.json`의 URL로 HTTP 요청 전달 |
| **HTML5 App Repo** | `service` | BTP 운영 (`xs-app.json`) | `html5-apps-repo-rt` 서비스에서 파일 조회 |

### 9.1 `localDir` — 로컬 파일 직접 서빙

```json
{
    "source": "^(.*)/portal_ott/(.*)$",
    "target": "$2",
    "localDir": "../../portal_ott/webapp"
}
```

- approuter가 **직접** 디스크 파일을 읽음
- `localDir` + `target` = 실제 파일 경로
- 운영 환경에서는 이 설정이 **존재하지 않음** (대신 `service` 사용)

### 9.2 `destination` — API 프록시

```json
{
    "source": "^.*/srv-api/(.*)$",
    "target": "$1",
    "destination": "onOttService"
}
```

- `default-env.json`에서 `destination.name`으로 URL을 찾음
- approuter가 HTTP 프록시 역할을 수행
- `authenticationType`에 따라 인증 토큰 추가 가능

### 9.3 `service` — BTP HTML5 App Repository (운영 only)

```json
{
    "source": "^.*/portal_ott/(.*)$",
    "target": "/comportalott-1.0.0/$1",
    "service": "html5-apps-repo-rt"
}
```

- `mta.yaml`에 정의된 `html5-apps-repo` 서비스를 통해 파일을 조회
- 파일이 BTP 중앙 저장소에 배포되어 있음
- `localDir` 대신 `service` 사용

---

## 10. 🔥 `dev/xs-app.json` vs `xs-app.json` (운영) — 전체 비교

### 10.1 로컬 개발 (`dev/xs-app.json`)

```json
{
    "welcomeFile": "/index.html",
    "authenticationMethod": "none",
    "routes": [
        { "source": "^.*/srv-api/(.*ott-core/.*)$", "destination": "onOttCoreService" },
        { "source": "^.*/srv-api/(.*)$",             "destination": "onOttService" },
        // ... erp-api, scim-api ...
        { "source": "/common_ott.lib/(.*)",           "localDir": "../../library_ott/src" },
        { "source": "/common.lib/(.*)",                "localDir": "../../library/src" },
        { "source": "^(.*)/portal_ott/(.*)$",         "localDir": "../../portal_ott/webapp" },
        { "source": "^(.*)/sysmgt_ott/(.*)$",         "localDir": "../../sysmgt_ott/webapp" },
        { "source": "^(.*)/board/(.*)$",               "localDir": "../../board/webapp" },
        { "source": "^(.*)/settlement/(.*)$",          "localDir": "../../settlement/webapp" },
        { "source": "^(.*)/trendAnalysis/(.*)$",       "localDir": "../../trendAnalysis/webapp" },
        { "source": "^/portal/(.*)$",                  "destination": "ui-portal" },
        { "source": "^/(.*)$",                         "destination": "ui" }
    ]
}
```

**특징**:
- `authenticationType: "none"` — 인증 없음
- `localDir` 사용 — 디스크 직접 읽기
- 모든 경로가 개별 `localDir`로 명시됨

### 10.2 BTP 운영 (`xs-app.json`)

```json
{
    "welcomeFile": "/index.html",
    "sessionTimeout": 60,
    "authenticationMethod": "route",
    "logout": {
        "logoutEndpoint": "/user/logout",
        "logoutPage": "/"
    },
    "routes": [
        { "source": "^.*/srv-api/(odata/v[24]/ott-core/.*)$", "destination": "onOttCoreService",
          "authenticationType": "xsuaa" },
        { "source": "^.*/srv-api/(.*)$",                       "destination": "onOttService",
          "authenticationType": "xsuaa" },
        { "source": "^.*/portal_ott/(.*)$",                    "service": "html5-apps-repo-rt",
          "target": "/comportalott-1.0.0/$1", "authenticationType": "xsuaa" },
        { "source": "/common_ott.lib/(.*)",                    "service": "html5-apps-repo-rt",
          "target": "/common_ottlib-1.0.0/$1", "authenticationType": "xsuaa" },
        // ... sysmgt_ott, board, settlement, trendAnalysis (모두 service + xsuaa) ...
        { "source": "(.*)",                                    "service": "html5-apps-repo-rt",
          "target": "/comportalott-1.0.0/$1", "authenticationType": "xsuaa" }
    ]
}
```

**특징**:
- `authenticationType: "xsuaa"` — XSUAA 인증 필수
- `service: "html5-apps-repo-rt"` — HTML5 App Repository에서 서빙
- `localDir` 없음 — 모든 정적 파일이 BTP 중앙 저장소에 배포됨
- `sessionTimeout: 60` — 세션 타임아웃 60분
- `logout` 설정 — 로그아웃 엔드포인트 + 로그아웃 후 이동할 페이지
- 라우트 수 적음 — portal_ott, common_ott.lib, sysmgt_ott 등만 명시

### 10.3 주요 차이점 요약

| 항목 | 로컬 (`dev/xs-app.json`) | 운영 (`xs-app.json`) |
|------|--------------------------|----------------------|
| 인증 | `"none"` | `"xsuaa"` |
| 정적 파일 | `localDir` (디스크) | `service` (HTML5 Repo) |
| 세션 관리 | 없음 | `sessionTimeout: 60` |
| 로그아웃 | 없음 | `/user/logout` → `/` |
| catch-all | `^/(.*)$` → `ui` destination | `(.*)` → portal_ott |
| 작업 디렉토리 | `--workingDir ./dev` | 기본 (프로젝트 루트) |

---

## 11. 전체 라우트 매칭 흐름도

```
                    ┌─── HTTP 요청이 approuter(5000)에 도착 ───┐
                    │                                          │
                    ▼                                          │
          ┌──────────────────┐                                │
          │ /srv-api/...     │─── ott-core 포함? ──▶ core_ott(8084)
          │                  │─── 그 외         ──▶ ott(8083)
          └──────────────────┘                                │
                    │                                          │
                    ▼                                          │
          ┌──────────────────┐                                │
          │ /erp-api/...     │──▶ onCloudConnector(9999)      │
          └──────────────────┘                                │
                    │                                          │
                    ▼                                          │
          ┌──────────────────┐                                │
          │ /scim-api/...    │──▶ onSCIMApi(9999)             │
          └──────────────────┘                                │
                    │                                          │
                    ▼                                          │
          ┌──────────────────┐                                │
          │ /common_ott.lib/ │──▶ localDir: library_ott/src   │
          └──────────────────┘                                │
                    │                                          │
                    ▼                                          │
          ┌──────────────────┐                                │
          │ /common.lib/     │──▶ localDir: library/src       │
          └──────────────────┘                                │
                    │                                          │
                    ▼                                          │
          ┌──────────────────┐                                │
          │ /portal_ott/     │──▶ localDir: portal_ott/webapp │
          └──────────────────┘                                │
                    │                                          │
                    ▼                                          │
          ┌──────────────────┐                                │
          │ /sysmgt_ott/     │──▶ localDir: sysmgt_ott/webapp │
          └──────────────────┘                                │
                    │                                          │
                    ▼                                          │
          ┌──────────────────┐                                │
          │ /board/          │──▶ localDir: board/webapp      │
          └──────────────────┘                                │
                    │                                          │
                    ▼                                          │
          ┌──────────────────┐                                │
          │ /settlement/     │──▶ localDir: settlement/webapp │
          └──────────────────┘                                │
                    │                                          │
                    ▼                                          │
          ┌──────────────────┐                                │
          │ /trendAnalysis/  │──▶ localDir: trendAnalysis/    │
          └──────────────────┘          webapp                │
                    │                                          │
                    ▼                                          │
          ┌──────────────────┐                                │
          │ /portal/         │──▶ destination: ui-portal      │
          │                  │    → http://localhost:5001      │
          └──────────────────┘                                │
                    │                                          │
                    ▼                                          │
          ┌──────────────────┐                                │
          │ / (그 외 모든 것) │──▶ destination: ui             │
          │                  │    → http://localhost:5001      │
          └──────────────────┘                                │
```

---

## 12. `default-env.json` — Destination 주소록

```json
{
    "destinations": [
        {
            "name": "onOttService",
            "url": "http://localhost:8083",
            "forwardAuthToken": false
        },
        {
            "name": "onOttCoreService",
            "url": "http://localhost:8084",
            "forwardAuthToken": false
        },
        {
            "name": "onCloudConnector",
            "url": "http://localhost:9999",
            "forwardAuthToken": false
        },
        {
            "name": "onSCIMApi",
            "url": "http://localhost:9999",
            "forwardAuthToken": false
        },
        {
            "name": "ui-portal",
            "url": "http://localhost:5001",
            "forwardAuthToken": false
        },
        {
            "name": "ui",
            "url": "http://localhost:5001",
            "forwardAuthToken": false
        }
    ]
}
```

| Destination | URL | 용도 |
|-------------|-----|------|
| `onOttService` | `http://localhost:8083` | OTT 비즈니스 API (컨텐츠, 정산, 구독자 등) |
| `onOttCoreService` | `http://localhost:8084` | OTT 공통 API (사용자, 메뉴, 공통코드 등) |
| `onCloudConnector` | `http://localhost:9999` | ERP 연동 (SAP Cloud Connector) |
| `onSCIMApi` | `http://localhost:9999` | SCIM 사용자 프로비저닝 API |
| `ui-portal` | `http://localhost:5001` | portal 경로 전용 UI 서버 |
| `ui` | `http://localhost:5001` | 범용 UI 서버 (welcome 페이지 등) |

> 💡 `ui-portal`과 `ui`는 둘 다 `http://localhost:5001`을 바라본다. 실제 사용 시나리오에 따라 분리할 수 있도록 이름만 다르게 해둔 것이다.

---

## 13. `package.json` — 스크립트 분석

### 13.1 주요 스크립트

```json
{
    "scripts": {
        "start": "node node_modules/@sap/approuter/approuter.js",
        "start:local": "node node_modules/@sap/approuter/approuter.js --workingDir ./dev --port 5000",
        "start:ui": "ui5 serve --port 5001",
        "build:cf": "mbt build",
        "build:dev": "node -e \"require('fs').copyFileSync('xs-app-dev.json','xs-app.json')\" && mbt build",
        "deploy:cf": "cf deploy mta_archives/com-approuterott_1.0.0.mtar"
    }
}
```

| 스크립트 | 설명 |
|----------|------|
| `start` | 운영 모드로 approuter 실행 (프로젝트 루트의 `xs-app.json` 사용) |
| `start:local` | **로컬 개발 모드**. `dev/xs-app.json` + `dev/default-env.json` 사용, port 5000 |
| `start:ui` | UI5 정적 파일 서버 실행 (port 5001, `webapp/` 서빙) |
| `build:cf` | MTA 빌드 (Cloud Foundry 배포용 아카이브 생성) |
| `build:dev` | DEV용 `xs-app-dev.json`을 `xs-app.json`으로 복사 후 MTA 빌드 |
| `deploy:cf` | Cloud Foundry에 배포 |

### 13.2 전체 의존성

```json
{
    "dependencies": {
        "@sap/approuter": "^20.0.1"
    }
}
```

approuter_ott의 유일한 런타임 의존성은 `@sap/approuter`다. 이 하나의 패키지가 HTTP 서버, 라우터, 프록시, 인증을 모두 제공한다.

---

## 14. `mta.yaml` — BTP 배포 구조

```yaml
ID: com-approuterott
_schema-version: "3.2"
version: 1.0.0

modules:
  - name: com-approuterott
    type: approuter.nodejs       # ← approuter 모듈 타입
    path: .
    properties:
      IAS_XSUAA_XCHANGE_ENABLED: true
      COOKIE_BACKWARD_COMPATIBILITY: true
    requires:
      - name: com-app-repo-runtime     # HTML5 앱 저장소
      - name: com-app-destination      # Destination 서비스
      - name: com-app-xsuaa            # XSUAA 인증 서비스
      - name: com-app-connectivity     # Cloud Connector 연결
```

approuter는 BTP에서 다음 4가지 BTP 서비스와 바인딩된다:

| 서비스 | 역할 |
|--------|------|
| `html5-apps-repo` (app-runtime) | HTML5 앱 파일 저장 및 서빙 — `xs-app.json`의 `service` 타겟 |
| `destination` (lite) | 백엔드 시스템 주소 관리 — `destination` 타겟의 실제 URL 제공 |
| `xsuaa` (application) | 사용자 인증 및 권한 관리 — `authenticationType: "xsuaa"` |
| `connectivity` (lite) | SAP Cloud Connector를 통한 온프레미스 시스템 연결 |

---

## 15. `xs-security.json` — XSUAA 보안 설정

```json
{
    "xsappname": "com-app-xsapp",
    "tenant-mode": "dedicated",
    "scopes": [
        {
            "name": "$XSAPPNAME.Admin",
            "description": "OTT 관리자 권한"
        }
    ],
    "role-templates": [
        {
            "name": "Admin",
            "description": "OTT 관리자",
            "scope-references": ["$XSAPPNAME.Admin"]
        }
    ],
    "role-collections": [
        {
            "name": "OTT_Settlement_Admin",
            "description": "OTT 관리자 롤 컬렉션",
            "role-template-references": ["$XSAPPNAME.Admin"]
        }
    ]
}
```

BTP XSUAA 인스턴스 생성 시 사용되는 보안 설정. Admin 스코프 하나만 정의되어 있고, `OTT_Settlement_Admin` 롤 컬렉션으로 관리자에게 할당한다.

---

## 16. 전체 실행 타임라인

```
시간 →

0ms    │ 사용자가 http://localhost:5000 입력
       │
~5ms   │ approuter(5000): welcomeFile → /index.html
       │ └─ route ^/(.*)$ → destination "ui" → http://localhost:5001
       │
~10ms  │ ui5 serve(5001) → webapp/index.html 반환
       │
~20ms  │ 브라우저: index.html 파싱
       │ └─ <script>location.href="/srv-api/odata/v4/frame/AuthCheck"</script>
       │ └─ 즉시 리다이렉트!
       │
~25ms  │ 브라우저: GET /srv-api/odata/v4/frame/AuthCheck → approuter(5000)
       │
~30ms  │ approuter 라우트 매칭:
       │ └─ route ^.*/srv-api/(.*)$ → destination "onOttService"
       │ └─ 프록시: http://localhost:8083/odata/v4/frame/AuthCheck
       │
~50ms  │ ott 백엔드(8083): AuthCheck 처리
       │ └─ 로컬 개발 → 인증 통과 → 302 /portal_ott/index.html
       │
~60ms  │ 브라우저: /portal_ott/index.html 로 이동
       │
~65ms  │ approuter 라우트 매칭:
       │ └─ route ^(.*)/portal_ott/(.*)$ → localDir: ../../portal_ott/webapp
       │ └─ 디스크에서 portal_ott/webapp/index.html 서빙
       │
~80ms  │ portal_ott index.html 로드
       │ └─ sap-ui-core.js 다운로드 시작 (CDN)
       │
~300ms │ UI5 코어 초기화
       │ └─ resourceroots, 라이브러리 preload
       │
~500ms │ portal_ott SPA 완전 로드
       │ 🎉 첫 화면 표시!
```

---

## 17. 핵심 요약: approuter API 프록시 흐름

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        approuter (port 5000)                              │
│                                                                           │
│  dev/xs-app.json (라우트 규칙)                 dev/default-env.json       │
│  ┌────────────────────────────────┐         ┌──────────────────────┐     │
│  │ /srv-api/*ott-core/*           │────────▶│ onOttCoreService     │     │
│  │                                │         │  → localhost:8084    │     │
│  │ /srv-api/*                     │────────▶│ onOttService         │     │
│  │                                │         │  → localhost:8083    │     │
│  │ /erp-api/*                     │────────▶│ onCloudConnector     │     │
│  │                                │         │  → localhost:9999    │     │
│  │ /scim-api/*                    │────────▶│ onSCIMApi            │     │
│  │                                │         │  → localhost:9999    │     │
│  │ /common_ott.lib/*              │──localDir: ../../library_ott/src     │
│  │ /common.lib/*                  │──localDir: ../../library/src         │
│  │ /portal_ott/*                  │──localDir: ../../portal_ott/webapp   │
│  │ /sysmgt_ott/*                  │──localDir: ../../sysmgt_ott/webapp   │
│  │ /board/*                       │──localDir: ../../board/webapp        │
│  │ /settlement/*                  │──localDir: ../../settlement/webapp   │
│  │ /trendAnalysis/*               │──localDir: ../../trendAnalysis/webapp│
│  │ /portal/*                      │────────▶│ ui-portal            │     │
│  │                                │         │  → localhost:5001    │     │
│  │ /* (catch-all)                 │────────▶│ ui                   │     │
│  │                                │         │  → localhost:5001    │     │
│  └────────────────────────────────┘         └──────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────┘

브라우저 요청                     approuter 처리                      최종 목적지
─────────────                    ─────────────                       ──────────
/ (welcomeFile)              →   ^/(.*)$ → ui destination        →   localhost:5001/index.html
                                  → webapp/index.html 반환
                                  → location.href AuthCheck 리다이렉트

/srv-api/...ott-core/...     →   onOttCoreService                 →   localhost:8084 (core_ott)
/srv-api/...(그 외)...        →   onOttService                     →   localhost:8083 (ott)
/portal_ott/...              →   localDir에서 파일 읽기             →   디스크의 portal_ott/webapp/
/sysmgt_ott/...              →   localDir에서 파일 읽기             →   디스크의 sysmgt_ott/webapp/
/common_ott.lib/...           →   localDir에서 파일 읽기             →   디스크의 library_ott/src/
/board/...                   →   localDir에서 파일 읽기             →   디스크의 board/webapp/
```

---

## 18. 절대 잊지 말 것

| # | 핵심 |
|---|------|
| 1 | **approuter는 3가지 역할**: ① 정적 파일 서빙(`localDir`), ② API 프록시(`destination`), ③ 사용자 인증(`authenticationType`) |
| 2 | **`dev/xs-app.json`이 로컬 라우트 규칙** — `--workingDir ./dev` 로 지정. routes는 위→아래 순서로 매칭, 첫 번째 매칭이 승리 |
| 3 | **`dev/default-env.json`이 destination 주소록** — `name`으로 참조, `url`이 실제 백엔드 주소 |
| 4 | **운영(`xs-app.json`)과 로컬(`dev/xs-app.json`)은 완전히 다르다** — 로컬: `localDir` + `authType:none`, 운영: `service` + `authType:xsuaa` |
| 5 | **`welcomeFile`은 루트(`/`) 진입 시 기본 파일 지정** — `/`로 들어오면 내부적으로 `/index.html`로 재처리 |
| 6 | **`webapp/index.html`의 유일한 역할은 AuthCheck 리다이렉트** — `location.href="/srv-api/odata/v4/frame/AuthCheck"` |
| 7 | **`srv-api` 경로에 `ott-core` 포함 여부로 백엔드 분기** — core_ott(8084) vs ott(8083). 순서가 중요! |
| 8 | **`localDir` = 디스크 직접 읽기** — 프록시하지 않고 approuter가 파일을 읽어서 바로 응답 |
| 9 | **`destination` = HTTP 프록시** — `default-env.json`의 URL로 요청을 그대로 전달 |
| 10 | **`service` = BTP HTML5 App Repository** — 운영 환경에서만 사용. `localDir` 대신 중앙 저장소에서 파일을 가져옴 |
| 11 | **approuter ≠ SAPUI5 Router** — 전자는 서버(Node.js HTTP), 후자는 브라우저 JS 객체 |
| 12 | **catch-all 라우트(`^/(.*)$`)는 항상 마지막에** — 모든 요청이 이 라우트에 도달하기 전에 더 구체적인 라우트가 처리해야 함 |
| 13 | **ui5 serve(5001)는 `webapp/` 디렉토리를 서빙** — approuter의 catch-all이 이 서버로 프록시 |
| 14 | **AuthCheck 완료 후 `302 Location: /portal_ott/index.html`** — ott 백엔드가 인증 확인 후 portal로 리다이렉트 |
| 15 | **하나의 approuter가 여러 백엔드 + 여러 정적 앱을 통합** — 브라우저는 approuter 하나만 바라보면 모든 리소스에 접근 가능 |

---

> **문서 버전**: v1.0 · 2026-07-09
> **참고 문서**: `portal_ott_전체_실행_흐름_완전_분석.md` (portal_ott 연동 분석)
> **분석 대상**:
> - `C:\Users\Eric\github\zen-ott-onboarding\cap-app\approuter_ott\webapp\index.html`
> - `C:\Users\Eric\github\zen-ott-onboarding\cap-app\approuter_ott\dev\xs-app.json`
> - `C:\Users\Eric\github\zen-ott-onboarding\cap-app\approuter_ott\dev\default-env.json`
> - `C:\Users\Eric\github\zen-ott-onboarding\cap-app\approuter_ott\xs-app.json` (운영)
> - `C:\Users\Eric\github\zen-ott-onboarding\cap-app\approuter_ott\package.json`
> - `C:\Users\Eric\github\zen-ott-onboarding\cap-app\approuter_ott\mta.yaml`
> - `C:\Users\Eric\github\zen-ott-onboarding\cap-app\approuter_ott\xs-security.json`
