# approuter_ott 전체 실행 흐름 완전 분석 (최종)

> **작성일**: 2026-07-10
> **목적**: `npm run start:local` → 브라우저 접속 → AuthCheck → portal_ott까지의 전 과정을 로컬과 BTP 배포 양쪽에서 완전히 이해한다.

---

## 📁 관련 파일 위치

| 구성요소 | 경로 |
|----------|------|
| approuter_ott 루트 | `cap-app/approuter_ott/` |
| 진입점 (welcome) | `cap-app/approuter_ott/webapp/index.html` |
| 로컬 라우트 규칙 | `cap-app/approuter_ott/dev/xs-app.json` |
| 로컬 destination 주소록 | `cap-app/approuter_ott/dev/default-env.json` |
| 운영 라우트 규칙 | `cap-app/approuter_ott/xs-app.json` |
| FrameService (AuthCheck) | `cap-node/core_ott/srv/cds/frame-service.cds` |
| FrameService 핸들러 | `cap-node/core_ott/srv/src/config/FrameService.handler.ts` |
| core_ott 서버 설정 | `cap-node/core_ott/srv/server.ts` |
| ott 서버 설정 | `cap-node/ott/srv/server.ts` |
| portal_ott | `cap-app/portal_ott/webapp/` |
| library_ott | `cap-app/library_ott/src/` |

---

## 0. approuter의 정체

`@sap/approuter`는 SAP BTP 환경의 **Node.js 기반 애플리케이션 라우터**다. 하나의 진입점(port 5000)에서:

1. **정적 파일 서빙** — HTML, JS, CSS (`localDir`)
2. **백엔드 API 프록시** — OData 서비스 (`destination`)
3. **사용자 인증** — XSUAA 연동 (BTP 운영 시)
4. **세션 관리** — 쿠키, 타임아웃

의 네 가지 역할을 수행한다.

### 0.1 두 가지 운영 모드

| 모드 | 설정 파일 | 인증 | 정적 파일 |
|------|----------|------|-----------|
| **로컬 개발** | `dev/xs-app.json` + `dev/default-env.json` | `authenticationType: "none"` | `localDir` (디스크 직접 읽기) |
| **BTP 운영** | `xs-app.json` (프로젝트 루트) | `authenticationType: "xsuaa"` | `service: "html5-apps-repo-rt"` |

> `dev/xs-app.json`은 `--workingDir ./dev` 옵션으로 지정된다. 운영용 `xs-app.json`은 완전히 무시된다.

---

## 1. 실행 방법

```bash
# 터미널 1: OTT 비즈니스 백엔드 (port 8083)
cd cap-node/ott && npx cds watch --port 8083

# 터미널 2: core_ott 공통 백엔드 (port 8084)
cd cap-node/core_ott && npx cds watch --port 8084

# 터미널 3: approuter_ott (port 5000)
cd cap-app/approuter_ott && npm run start:local
```

---

## 2. 로컬 개발: 전체 실행 흐름

```
브라우저 → http://localhost:5000
│
├─① welcomeFile → /index.html
│    └─ 라우트 매칭: ^/(.*)$ → localDir: "../webapp"
│       └─ approuter_ott/webapp/index.html 서빙
│
├─② index.html: location.href="/srv-api/odata/v4/frame/AuthCheck"
│    └─ 브라우저가 즉시 AuthCheck로 리다이렉트
│
├─③ approuter 라우트 매칭
│    └─ ^.*/srv-api/(.*frame/.*)$ → onOttCoreService (8084)
│       └─ 프록시: http://localhost:8084/odata/v4/frame/AuthCheck
│
├─④ core_ott(8084): FrameService.onAuthCheck()
│    └─ redirect('/portal_ott/index.html')
│    └─ ★ AuthCheck는 인증 검사를 하지 않는다. 단순 라우팅 게이트.
│
├─⑤ approuter 라우트 매칭
│    └─ ^(.*)/portal_ott/(.*)$ → localDir: "../../portal_ott/webapp"
│       └─ portal_ott/webapp/index.html 서빙
│
├─⑥ portal_ott SPA 로딩
│    └─ sap-ui-core.js → ComponentSupport → Component.init()
│    └─ UserSessionLoader, MenuLoader 등이 API 호출 시작
│
├─⑦ API 요청 (/srv-api/...ott-core/...)
│    └─ approuter → onOttCoreService (8084)
│       └─ CAP mocked auth → 최초 1회 Basic Auth 팝업
│          └─ U001 / 빈 비밀번호 입력 → 이후 세션 동안 자동 통과
│
└─⑧ portal_ott 첫 화면 표시 🎉
```

### 2.1 로컬 인증: Basic Auth 팝업

로컬 개발에서는 `authenticationType: "none"`이므로 approuter가 인증을 하지 않는다. 대신 CAP의 `mocked` auth 전략이 동작한다:

- API 요청 → CAP이 인증 정보 없음을 감지 → `401 + WWW-Authenticate: Basic`
- 브라우저가 Basic Auth 팝업 표시
- `U001` / 빈 비밀번호(또는 `admin1` 등 mock 계정) 입력
- 브라우저가 세션 동안 자격증명을 캐싱 → 이후 모든 API 호출 자동 통과

> **딱 한 번만** 팝업이 뜨고, 그 후로는 portal_ott가 정상 작동한다.

---

## 3. BTP 운영: 전체 실행 흐름

```
브라우저 → https://<app-url>
│
├─① welcomeFile → /index.html
│    └─ html5-apps-repo-rt에서 index.html 서빙
│
├─② index.html: location.href="/srv-api/odata/v4/frame/AuthCheck"
│
├─③ 🛑 approuter가 AuthCheck 요청을 가로챔!
│    └─ route: authenticationType: "xsuaa"
│    └─ "JWT 토큰이 없네? XSUAA 로그인 페이지로 가!"
│
├─④ XSUAA 로그인 페이지 → 사용자 인증
│    └─ 로그인 성공 → JWT 토큰 발급 → 원래 URL로 리다이렉트
│
├─⑤ AuthCheck 재요청 (이번엔 JWT 포함)
│    └─ approuter: JWT 확인 → 통과 → core_ott로 프록시
│    └─ core_ott: JWT 검증 → cds.context.user = 실제 사용자
│    └─ FrameService.onAuthCheck() → redirect('/portal_ott/index.html')
│
├─⑥ portal_ott SPA 로딩
│    └─ 모든 API 요청에 JWT 자동 첨부 → 401 없이 정상 통과
│    └─ cds.context.user에 실제 사용자 정보가 들어있음
│
└─⑦ portal_ott 첫 화면 표시 (로그인 팝업 없음!) 🎉
```

### 3.1 BTP 인증: XSUAA

| 단계 | 주체 | 동작 |
|:--:|------|------|
| 가로채기 | approuter | `authenticationType: "xsuaa"` → 미인증 요청 차단 |
| 로그인 | XSUAA | SAP BTP Identity Authentication 서비스 |
| 토큰 | XSUAA | JWT 발급, approuter가 모든 백엔드 요청에 첨부 |
| 검증 | CAP | JWT 서명 검증 → `cds.context.user` 자동 설정 |

---

## 4. dev/xs-app.json 라우트 구조

```
우선순위 (위→아래, 첫 매칭 승리)
│
├─ ① ^.*/srv-api/(.*ott-core/.*)$  →  onOttCoreService (8084)
│     core_ott의 공통 API (UserManagement, MenuManagement 등)
│
├─ ② ^.*/srv-api/(.*frame/.*)$     →  onOttCoreService (8084)
│     FrameService의 AuthCheck
│
├─ ③ ^.*/srv-api/(.*)$             →  onOttService (8083)
│     ott의 비즈니스 API (DetailService, MembershipService 등)
│
├─ ④ ^.*/erp-api/(.*)$             →  onCloudConnector (9999)
├─ ⑤ ^.*/scim-api/(.*)$            →  onSCIMApi (9999)
│
├─ ⑥ /common_ott.lib/(.*)          →  localDir: library_ott/src
├─ ⑦ /common.lib/(.*)              →  localDir: library/src
│
├─ ⑧ 정적 앱 localDir 라우트들
│     template, sysmgt, board, settlement, trendAnalysis,
│     portal_ott, sysmgt_ott
│
└─ ⑨ ^/(.*)$                       →  localDir: "../webapp"
      catch-all (approuter의 index.html 등)
```

### 4.1 핵심 포인트

| 원칙 | 설명 |
|------|------|
| `localDir` | 디스크에서 파일을 직접 읽어 응답. 프록시 없음. |
| `destination` | `default-env.json`의 URL로 HTTP 요청을 프록시. |
| `ott-core` vs `frame` | 둘 다 `core_ott`(8084)로 가지만 URL 패턴이 다르다. `ott-core`는 공통 API, `frame`은 AuthCheck. |
| catch-all | 맨 마지막에 위치. `../webapp` = `approuter_ott/webapp/` |

---

## 5. dev/default-env.json destinations

| Destination | URL | 용도 |
|-------------|-----|------|
| `onOttCoreService` | `http://localhost:8084` | core_ott 공통 API + AuthCheck |
| `onOttService` | `http://localhost:8083` | OTT 비즈니스 API |
| `onCapService` | `http://localhost:8080` | (원본 core, 현재 미사용) |
| `onCloudConnector` | `http://localhost:9999` | ERP 연동 |
| `onSCIMApi` | `http://localhost:9999` | SCIM 사용자 프로비저닝 |

---

## 6. 백엔드 구조

```
cap-node/
├── core/           ← 원본 템플릿 (절대 수정 금지, port 8080)
│   └── FrameService (AuthCheck) — 원본에는 있음
│
├── core_ott/       ← OTT 공통 백엔드 (port 8084)
│   ├── FrameService ← ★ 우리가 복사해온 AuthCheck
│   ├── UserManagement
│   ├── MenuManagement
│   ├── CodeManagement
│   └── MessageManagement
│
└── ott/            ← OTT 비즈니스 백엔드 (port 8083)
    ├── MainService
    ├── DetailService
    ├── MembershipService
    └── ...
```

---

## 7. FrameService (AuthCheck)

### 7.1 CDS 정의

```cds
namespace com.cap.ott.core;

@impl: 'srv/src/config/FrameService.handler'
service FrameService {
    @readonly
    function AuthCheck() returns Integer;
}
```

### 7.2 핸들러

```typescript
onAuthCheck(req: Request) {
    console.log(cds.context);
    req.http?.res.redirect('/portal_ott/index.html');
    return 0;
}
```

**AuthCheck는 인증을 하지 않는다.** 단지 인증된 사용자를 portal_ott로 안내하는 **라우팅 게이트**다.

- 로컬: CAP mocked auth가 사용자 주입 → AuthCheck는 그냥 redirect
- BTP: approuter+XSUAA가 인증 완료 → AuthCheck 도달 시 이미 `cds.context.user`에 실제 사용자 정보 존재

> 템플릿이 AuthCheck를 미리 만들어둔 이유: 나중에 `cds.context.user`를 확인해 "관리자면 admin으로, 일반 사용자면 portal로" 같은 **인가 로직**을 추가할 확장 포인트를 마련하기 위함이다.

---

## 8. 인증 아키텍처 비교

| | 로컬 개발 | BTP 운영 |
|---|---|---|
| **approuter 인증** | `authenticationType: "none"` → 전부 통과 | `authenticationType: "xsuaa"` → 미인증 차단 |
| **로그인 방식** | CAP mocked auth → Basic Auth 팝업 (1회) | XSUAA 로그인 페이지 |
| **사용자 정보** | `U001` 등 mock 계정 | 실제 BTP 사용자 |
| **API 인증** | Basic Auth 자격증명 (세션 캐싱) | JWT 토큰 (approuter 자동 첨부) |
| **AuthCheck 역할** | 그냥 redirect | 인증 완료 후 도달하는 환승역 |

---

## 9. approuter ≠ SAPUI5 Router

| | approuter | SAPUI5 Router |
|---|---|---|
| 정체 | Node.js HTTP 서버 | 브라우저 JS 객체 |
| 위치 | 서버 (터미널) | 브라우저 메모리 |
| 역할 | 파일 서빙 + API 프록시 + 인증 | URL 해시(#) 기반 SPA 화면 전환 |
| 설정 | `xs-app.json` | `manifest.json` routing |
| 포트 | 5000 | 없음 |

---

## 10. server.ts x-forwarded-for 패치

```typescript
// approuter가 로컬 dev에서 x-forwarded-for를 문자열 "undefined"로 보내서
// CAP이 이를 잘못된 값으로 판단해 $batch 요청이 500으로 실패하는 걸 방지
if (req.headers['x-forwarded-for'] === 'undefined') {
    delete req.headers['x-forwarded-for'];
}
```

로컬에서 approuter는 `x-forwarded-for: "undefined"`라는 잘못된 헤더를 전송한다. CAP이 이 값을 거부해 `$batch` 요청이 500으로 실패하는 것을 방지하기 위해 `core_ott`와 `ott` 양쪽 `server.ts`에 추가되어 있다.

---

## 11. 원본 템플릿 대비 변경 사항

| 항목 | 원본 (approuter) | OTT (approuter_ott) |
|------|-----------------|---------------------|
| AuthCheck 위치 | `core`(8080) | `core_ott`(8084) |
| FrameService redirect | `/portal/index.html` | `/portal_ott/index.html` |
| `/srv-api/` 라우트 | 단일 → `onCapService`(8080) | 3-way 분기 (ott-core/frame/그외) |
| `^/portal/(.*)$` 라우트 | 있음 | 제거 (`portal_ott`로 대체) |
| catch-all | `destination: "ui"`(5001) | `localDir: "../webapp"` |
| `ui-portal` destination | 있음 | 제거 (미사용) |
| `core_ott` server.ts | 없음 | x-forwarded-for 패치 추가 |
| `ott` server.ts | 없음 | x-forwarded-for 패치 추가 |

---

## 12. 핵심 요약

| # | 내용 |
|---|------|
| 1 | approuter는 **3가지 역할**: 정적 파일 서빙, API 프록시, 사용자 인증 |
| 2 | 로컬은 `dev/xs-app.json`, BTP는 `xs-app.json`. 둘은 완전히 다르다. |
| 3 | `welcomeFile`은 루트(`/`) 진입 시 `/index.html`로 내부 재작성 |
| 4 | `webapp/index.html`의 유일한 역할은 AuthCheck 리다이렉트 트리거 |
| 5 | AuthCheck는 **인증을 하지 않는다**. 라우팅 게이트일 뿐. |
| 6 | 로컬 인증 = CAP mocked auth + Basic Auth 팝업 (1회) |
| 7 | BTP 인증 = approuter가 XSUAA 로그인으로 가로챔 → JWT 발급 |
| 8 | `ott-core`와 `frame`은 둘 다 `core_ott`(8084)로 가지만 URL 패턴이 다르다 |
| 9 | `localDir` = 디스크 직접 읽기, `destination` = HTTP 프록시 |
| 10 | approuter ≠ SAPUI5 Router (서버 프로세스 vs 브라우저 JS 객체) |

---

> **문서 버전**: v2.0 · 2026-07-10
> **이전 버전 기반**: `approuter_ott_전체_실행_흐름_완전_분석.md` (v1.0, 2026-07-09), `AuthCheck_핸들러와_approuter_index_의문_해소.md` (v1.0, 2026-07-09)
