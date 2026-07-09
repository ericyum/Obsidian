# AuthCheck 핸들러와 approuter index.html 의문 완전 해소

> **작성일**: 2026-07-09
> **목적**: `approuter_ott_전체_실행_흐름_완전_분석.md`를 읽고 생긴 5가지 의문을 완전히 해소한다.

---

## 📌 질문 1: AuthCheck 핸들러가 `core`의 `frame-service.cds`와 `FrameService.handler.ts`인가?

### 답: **맞다!** 정확히 그 파일들이다.

| 파일 | 경로 | 역할 |
|------|------|------|
| `frame-service.cds` | `cap-node/core/srv/cds/frame-service.cds` | `FrameService` 서비스 정의, `AuthCheck()` 함수 선언 |
| `FrameService.handler.ts` | `cap-node/core/srv/src/config/FrameService.handler.ts` | `AuthCheck`의 실제 동작(핸들러 로직) 구현 |

```cds
// frame-service.cds
namespace com.cap.core;

@impl: 'srv/src/config/FrameService.handler'
service FrameService { 
    @readonly
    function AuthCheck() returns Integer;
}
```

```typescript
// FrameService.handler.ts (핵심 부분)
export default class FrameService extends cds.ApplicationService {
  async init(): Promise<void> {
    this.on("AuthCheck", this.onAuthCheck);
    return super.init();
  }

  onAuthCheck(req: Request) {
    console.log("###############################onAuthCheck###################################");
    console.log(cds.context);
    req.http?.res.redirect('/portal/index.html');
    return 0;
  }
}
```

### ⚠️ 중요: 이 파일들은 `core`에 있다, `core_ott`에 있는 게 아니다!

```
cap-node/
├── core/           ← FrameService 여기 있음! (AuthCheck 포함)
│   └── srv/
│       ├── cds/frame-service.cds
│       ├── src/config/FrameService.handler.ts
│       └── server.ts              ← CDS_LOCAL_AUTH_BYPASS 로직
│
├── core_ott/       ← ❌ FrameService 없음
│   └── srv/
│       └── cds/    ← CodeManagement, MenuManagement, UserManagement, MessageManagement 만 있음
│
└── ott/            ← ❌ FrameService 없음
    └── srv/
        └── cds/    ← MainService, DetailService, MembershipService 등
```

> 💡 **분석 문서의 실수**: `approuter_ott_전체_실행_흐름_완전_분석.md`에서는 `core_ott`(8084)와 `ott`(8083) 두 백엔드만 언급했지만, 실제로 `core`(8080)라는 세 번째 백엔드가 필요하다. `FrameService`는 오직 `core`에만 존재하기 때문이다.

---

## 📌 질문 2: 인증 절차는 어디에 있는가?

### 답: **두 곳**에 나뉘어 있다.

### 2.1 `FrameService.handler.ts` — AuthCheck 엔드포인트 (인증 "통과문")

```typescript
onAuthCheck(req: Request) {
    console.log(cds.context);   // 현재 사용자 컨텍스트 로깅
    req.http?.res.redirect('/portal/index.html');  // 무조건 portal로 리다이렉트
    return 0;
}
```

로컬 개발 환경에서는 사실상 **아무 인증도 하지 않고** 무조건 `/portal/index.html`로 리다이렉트한다.

### 2.2 `core/srv/server.ts` — 로컬 개발 인증 우회

```typescript
// ⚠️ 로컬 dev 전용 — approuter(authenticationType:none)가 core로 인증 헤더를 안 보내서
// 인증 없는 요청이 401 나는 걸 우회.
if (process.env.CDS_LOCAL_AUTH_BYPASS === '1') {
  cds.User.default = new cds.User({ 
    id: 'admin1', 
    roles: ['Admin', 'authenticated-user'] 
  });
}
```

이 코드가 하는 일:
- 환경변수 `CDS_LOCAL_AUTH_BYPASS=1`이 설정되어 있으면
- 모든 요청을 `admin1` 사용자가 보낸 것처럼 처리
- 즉, CAP이 요청마다 401(Unauthorized)을 내지 않도록 **가짜 사용자**를 주입

### 2.3 운영 환경(BTP)에서는?

운영 환경(`xs-app.json`)에서는:

1. `authenticationType: "xsuaa"` → approuter가 XSUAA 로그인 페이지로 리다이렉트
2. 로그인 성공 → XSUAA 토큰 발급 → approuter가 모든 백엔드 요청에 토큰 첨부
3. 백엔드(`core`)의 `onAuthCheck`에 도달했을 때는 이미 인증된 사용자
4. `cds.context`에 실제 사용자 정보가 들어 있음

> 💡 **요약**: 로컬에서는 인증이 **우회**(bypass)되고, 운영에서는 **approuter + XSUAA**가 인증을 담당한다. `onAuthCheck` 자체는 실질적인 인증 로직을 수행하지 않고, 인증된 상태를 확인한 후 portal로 보내는 **라우팅 게이트** 역할이다.

---

## 📌 질문 3: `/portal/index.html` vs `/portal_ott/index.html` — 왜 다른가?

### 답: **코드와 분석 문서 사이에 불일치가 있다.**

### 3.1 실제 코드 (FrameService.handler.ts)

```typescript
req.http?.res.redirect('/portal/index.html');   // ← /portal/index.html 로 리다이렉트
```

### 3.2 분석 문서 (approuter_ott_전체_실행_흐름_완전_분석.md)

```
④ 브라우저: /portal_ott/index.html 로 리다이렉트    ← /portal_ott/index.html 이라고 기술
```

### 3.3 왜 이런 차이가 생겼는가?

| 가능성 | 설명 |
|--------|------|
| **원본(portal) 코드 잔재** | `core`는 원래 `portal`이라는 non-OTT 프로젝트의 공통 모듈이었다. OTT 버전(`portal_ott`)으로 마이그레이션하면서 redirect 경로를 `/portal_ott/index.html`로 바꾸지 않은 것 같다. |
| **분석 문서의 추론** | 분석 문서 작성자는 "AuthCheck 후 portal_ott가 로드되므로 redirect도 `/portal_ott/index.html`일 것"이라고 **의도대로** 추론했을 가능성이 크다. |

### 3.4 실제 동작 추적 — `/portal/index.html`로 리다이렉트되면 무슨 일이?

AuthCheck가 `/portal/index.html`로 리다이렉트 → approuter가 `dev/xs-app.json`에서 매칭:

```json
// 라우트 12번: ^/portal/(.*)$  (portal_ott보다 먼저 검사됨!)
{
    "source": "^/portal/(.*)$",
    "target": "/$1",                // → "/index.html"
    "destination": "ui-portal",     // → http://localhost:5001
}
```

프록시 결과: `http://localhost:5001/index.html` → ui5 serve가 `webapp/index.html` 서빙 → **AuthCheck 리다이렉트가 있는 approuter의 index.html이 다시 서빙됨**

```
/portal/index.html 요청
    ↓
approuter route ^/portal/(.*)$ → ui-portal (5001)
    ↓
webapp/index.html 서빙
    ↓
<script>location.href="/srv-api/odata/v4/frame/AuthCheck"</script>
    ↓
AuthCheck 다시 호출
    ↓
redirect('/portal/index.html')
    ↓
🔄 무한 루프!
```

### 3.5 🔥 결론: 코드에 버그가 있다

`FrameService.handler.ts`의 redirect 경로는 **`/portal_ott/index.html`** 이어야 한다. 현재 `/portal/index.html`은 `portal_ott`를 바라보지 않고 approuter의 welcome 페이지(자기 자신)로 돌아가는 **무한 루프**를 만든다.

고쳐야 할 코드:
```typescript
// 현재 (버그)
req.http?.res.redirect('/portal/index.html');

// 수정 (의도한 동작)
req.http?.res.redirect('/portal_ott/index.html');
```

수정 후 흐름:
```
AuthCheck → redirect('/portal_ott/index.html')
    ↓
approuter route ^(.*)/portal_ott/(.*)$ → localDir: ../../portal_ott/webapp
    ↓
portal_ott/webapp/index.html 서빙 ✅  (실제 portal SPA)
```

> 💡 **분석 문서가 틀린 게 아니다!** 분석 문서는 **의도된 동작**을 정확히 추론했다. 실제 코드가 의도와 다르게 작성되어 있을 뿐이다.

---

## 📌 질문 4: 실제로는 portal index로 접속하는데, approuter의 index.html이 AuthCheck를 하는 건가?

### 답: **아니.** 둘은 완전히 다른 파일이고, 역할도 다르다.

### 4.1 두 index.html 비교

| | approuter의 index.html | portal_ott의 index.html |
|---|---|---|
| **경로** | `cap-app/approuter_ott/webapp/index.html` | `cap-app/portal_ott/webapp/index.html` |
| **내용** | `<script>location.href="/srv-api/odata/v4/frame/AuthCheck"</script>` | `<script src="sap-ui-core.js">` + UI5 부트스트랩 |
| **크기** | 4줄 | 40줄+ |
| **라우트** | `^/(.*)$` → `ui` destination (catch-all) | `^(.*)/portal_ott/(.*)$` → `localDir` |
| **역할** | AuthCheck로 **즉시 리다이렉트**하는 트리거 | 실제 portal_ott **SPA 애플리케이션** |

### 4.2 approuter의 index.html은 "AuthCheck를 하는" 게 아니라 "AuthCheck를 **트리거**"할 뿐

```html
<!-- approuter의 index.html -->
<script>location.href="/srv-api/odata/v4/frame/AuthCheck";</script>
```

이 파일의 **유일한** 역할은 브라우저를 AuthCheck 엔드포인트로 보내는 것이다. AuthCheck 자체는 백엔드(`core`의 `FrameService`)가 수행한다.

### 4.3 사용자가 실제로 보는 전체 흐름

```
사용자: http://localhost:5000 입력
    ↓ (welcomeFile)
approuter: GET / → /index.html 변환 → catch-all route → ui5 serve
    ↓
ui5 serve: approuter/webapp/index.html 서빙
    ↓ (location.href)
브라우저: GET /srv-api/odata/v4/frame/AuthCheck
    ↓ (approuter 프록시)
core 백엔드: onAuthCheck() → redirect('/portal/index.html')  ⚠️ 버그: /portal_ott/index.html 여야 함
    ↓
approuter: route ^/portal/(.*)$ → ui-portal → 🔄 무한 루프 (버그)

[버그 수정 후 정상 흐름]
    ↓ (redirect('/portal_ott/index.html'))
approuter: route ^(.*)/portal_ott/(.*)$ → localDir
    ↓
portal_ott/webapp/index.html 서빙
    ↓
UI5 부트스트랩 → Component.init() → UserSessionLoader + MenuLoader → 첫 화면 🎉
```

---

## 📌 질문 5: portal 쪽의 UserSessionLoader나 MenuLoader가 실질적 인증을 하는가?

### 답: **아니.** 인증과는 전혀 무관하다. 둘 다 인증 **"이후"**의 데이터 로딩이다.

### 5.1 UserSessionLoader — 사용자 "설정" 로더

```javascript
// UserSessionLoader.js
_fetchUserSession : function() {
    const oODataModel = new ODataModel("/srv-api/odata/v2/ott-core/UserManagement");
    return new Promise((resolve, reject) => {
        oODataModel.read("/UserSessionInfo", { success: resolve, error: reject });
    });
}
```

하는 일:
- `/srv-api/odata/v2/ott-core/UserManagement/UserSessionInfo` 호출
- 가져오는 데이터: `language_code`, `date_format_type_name`, `digits_format_type_name`, `theme_code`
- 사용 목적: **UI 개인화 설정** (언어, 날짜 포맷, 숫자 포맷, 테마)

**인증과의 관계: ❌ 없음.** 이미 인증된 사용자의 선호 설정을 가져올 뿐이다.

### 5.2 MenuLoader — 메뉴/권한 로더

```javascript
// MenuLoader.js
_fetchList : function() {
    const oODataModel = new ODataV4Model({
        serviceUrl: "/srv-api/odata/v4/ott-core/MenuManagement/"
    });
    const oBinding = oODataModel.bindList("/MenusRoleAppliedList");
    return oBinding.requestContexts();
}
```

하는 일:
- `/srv-api/odata/v4/ott-core/MenuManagement/MenusRoleAppliedList` 호출
- 가져오는 데이터: 사용자의 역할(role)에 따라 접근 가능한 메뉴 목록
- 사용 목적: **동적 라우트 생성** (Component.js의 `_setRouteAndTarget()`)

**인증과의 관계: ❌ 없음.** 이미 인증된 사용자의 권한에 맞는 메뉴를 가져올 뿐이다. 인증 정보는 요청 헤더/세션에 이미 포함되어 있고, 백엔드가 알아서 필터링한다.

### 5.3 진짜 인증은 어디서?

```
┌─────────────────────────────────────────────────────────┐
│                    인증 계층 구조                         │
│                                                         │
│  [운영 환경 - BTP]                                       │
│  ┌──────────┐     ┌──────────┐     ┌───────────────┐   │
│  │ approuter │────▶│  XSUAA   │────▶│ core/ott      │   │
│  │ (xsuaa)   │     │ (OAuth)  │     │ (cds.User)    │   │
│  └──────────┘     └──────────┘     └───────────────┘   │
│   요청 가로챔        토큰 발급        토큰 검증 +         │
│   로그인 페이지       사용자 확인      AuthCheck 게이트    │
│                                                         │
│  [로컬 개발]                                             │
│  ┌──────────────────────────────────────────────────┐   │
│  │ core/srv/server.ts                                │   │
│  │ cds.User.default = { id: 'admin1', roles: [...] } │   │
│  │ → 모든 요청을 admin1로 우회                         │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  [그 이후] — 인증과 무관한 데이터 로딩                      │
│  ┌──────────────────┐  ┌──────────────────┐             │
│  │ UserSessionLoader │  │   MenuLoader      │             │
│  │ (언어/테마 설정)    │  │ (메뉴/권한 목록)    │             │
│  └──────────────────┘  └──────────────────┘             │
└─────────────────────────────────────────────────────────┘
```

---

## 📌 질문 6: approuter의 index.html의 쓸모는 무엇인가?

### 답: **"시작점" 역할. 단 하나의 진입점에서 인증 흐름을 트리거한다.**

### 6.1 approuter index.html이 없으면?

approuter의 `index.html`이 없으면:
- `http://localhost:5000`으로 접속했을 때 404 또는 welcomeFile 설정이 없어서 에러
- 어떤 URL로 시작해야 하는지 사용자가 알 수 없음
- 인증 흐름을 자동으로 시작할 수 없음

### 6.2 왜 이렇게 설계했는가?

```
approuter index.html                    portal_ott index.html
┌─────────────────┐                    ┌─────────────────────┐
│ "문지기"          │                    │ "건물"               │
│                  │                    │                     │
│ • 누구세요?       │  ──인증 완료──▶    │ • UI5 SPA            │
│ • 신원 확인하러    │                    │ • 메뉴 로딩           │
│   가세요          │                    │ • 라우터 초기화       │
│                  │                    │ • 첫 화면 렌더링      │
│ 4줄짜리 HTML      │                    │ 40줄짜리 HTML        │
│ 단 1개의 script   │                    │ sap-ui-core.js 포함  │
└─────────────────┘                    └─────────────────────┘
```

SAP BTP 아키텍처에서는 **approuter가 모든 것의 진입점**이다. 그래서:

1. 사용자는 approuter **하나만** 바라본다 (port 5000)
2. approuter의 index.html이 AuthCheck를 트리거
3. 인증되면 portal로 리다이렉트
4. 인증 안 되면 로그인 페이지로 리다이렉트

**approuter index.html은 "인증 게이트웨이 트리거"** 이고, **portal_ott index.html은 "실제 애플리케이션 부트스트랩"** 이다.

### 6.3 파일이 4줄인 이유

더 많은 코드가 필요하지 않다. 단 하나의 역할 — `location.href`로 AuthCheck로 보내기 — 만 수행하면 된다. 모든 실제 로직은 백엔드의 `FrameService.onAuthCheck()`에 있다.

---

## 🔥 전체 흐름도 (버그 수정 후)

```
사용자 브라우저: http://localhost:5000
│
├─① approuter welcomeFile → /index.html → catch-all → ui5 serve
│    └─ webapp/index.html 반환 (approuter의 index.html)
│
├─② <script>location.href="/srv-api/odata/v4/frame/AuthCheck"</script>
│    └─ 브라우저 즉시 리다이렉트
│
├─③ approuter: ^.*/srv-api/(.*)$ → onOttService(8083) 
│    ⚠️ 분석문서는 ott(8083)로 보낸다고 했지만,
│    실제 FrameService는 core(8080)에 있음 → 라우팅 불일치 가능성
│    └─ core 백엔드: FrameService.onAuthCheck()
│       ├─ 로컬: CDS_LOCAL_AUTH_BYPASS → admin1으로 우회
│       ├─ 운영: XSUAA 토큰 검증
│       └─ redirect('/portal_ott/index.html')  ← 버그 수정 후
│
├─④ approuter: ^(.*)/portal_ott/(.*)$ → localDir
│    └─ portal_ott/webapp/index.html 서빙 (실제 portal SPA)
│
├─⑤ portal_ott SPA 로딩
│    ├─ sap-ui-core.js → ComponentSupport → Component.init()
│    ├─ UserSessionLoader: 사용자 언어/테마/포맷 설정 로드 (인증❌)
│    ├─ MenuLoader: 역할 기반 메뉴 목록 로드 (인증❌)
│    └─ _setRouteAndTarget() → Router.initialize() → 첫 화면 🎉
```

---

## 🛠️ 발견된 버그 & 수정 제안

### 버그 1: AuthCheck redirect 경로

```typescript
// cap-node/core/srv/src/config/FrameService.handler.ts

// 현재 (버그 — 무한 루프 유발)
req.http?.res.redirect('/portal/index.html');

// 수정 제안
req.http?.res.redirect('/portal_ott/index.html');
```

### 버그 가능성 2: approuter 라우팅 불일치

`/srv-api/odata/v4/frame/AuthCheck`가 현재 `dev/xs-app.json`의 라우트에 의해 `ott`(8083)으로 전달됨. 하지만 `FrameService`는 `core`(8080)에 있음.

```json
// dev/default-env.json에 core destination 추가 제안
{
    "name": "onCoreService",
    "url": "http://localhost:8080",
    "forwardAuthToken": false
}
```

```json
// dev/xs-app.json에 frame 경로 라우트 추가 (ott-core 패턴 앞에!)
{
    "source": "^.*/srv-api/odata/v[0-9]+/frame/(.*)$",
    "destination": "onCoreService",
    "target": "odata/v4/frame/$1",
    "authenticationType": "none"
}
```

또는 `core`를 8080 대신 8084로 실행하고 `core_ott`의 서비스와 병합하는 방법도 있다.

---

## 📋 요약: 6가지 의문에 대한 답변

| # | 질문 | 답변 |
|---|------|------|
| 1 | AuthCheck 핸들러가 `frame-service.cds` + `FrameService.handler.ts`인가? | ✅ **맞다.** `cap-node/core/srv/cds/frame-service.cds` + `cap-node/core/srv/src/config/FrameService.handler.ts` |
| 2 | 인증 절차는 어디에? | `onAuthCheck()`는 인증 "게이트". 로컬은 `server.ts`의 `cds.User.default`로 우회, 운영은 XSUAA가 담당 |
| 3 | `/portal/index.html` vs `/portal_ott/index.html` | **코드는 `/portal/index.html`로 되어 있으나 이는 버그.** 분석 문서가 추론한 `/portal_ott/index.html`이 의도된 동작 |
| 4 | approuter index.html이 AuthCheck를 실제로 하는가? | ❌ **아니.** 백엔드로 리다이렉트만 트리거. 실제 AuthCheck는 `FrameService.onAuthCheck()`에서 수행 |
| 5 | UserSessionLoader/MenuLoader가 인증을 하는가? | ❌ **아니.** 둘 다 인증 "이후"의 데이터 로딩 (사용자 설정, 메뉴 목록). 진짜 인증은 XSUAA + cds.User |
| 6 | approuter index.html의 쓸모는? | **"인증 게이트웨이 트리거"**. 단일 진입점에서 인증 흐름을 자동 시작하는 4줄짜리 부트스트랩 |

---

> **문서 버전**: v1.0 · 2026-07-09
> **참고 문서**: 
> - `approuter_ott_전체_실행_흐름_완전_분석.md`
> - `portal_ott_전체_실행_흐름_완전_분석.md`
> **분석 대상 소스**:
> - `cap-node/core/srv/cds/frame-service.cds`
> - `cap-node/core/srv/src/config/FrameService.handler.ts`
> - `cap-node/core/srv/server.ts`
> - `cap-app/approuter_ott/dev/xs-app.json`
> - `cap-app/approuter_ott/dev/default-env.json`
> - `cap-app/approuter_ott/webapp/index.html`
> - `cap-app/portal_ott/webapp/index.html`
> - `cap-app/portal_ott/webapp/Component.js`
> - `cap-app/library_ott/src/util/UserSessionLoader.js`
> - `cap-app/library_ott/src/util/MenuLoader.js`
