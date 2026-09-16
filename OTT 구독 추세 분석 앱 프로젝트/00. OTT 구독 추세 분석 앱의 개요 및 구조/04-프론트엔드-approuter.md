# 04 - 프론트엔드: AppRouter (진입점)

> **MTA ID:** `com-approuter`  
> **로컬 포트:** 5000 (라우터), 5001 (UI5 dev server)

---

## 1. AppRouter가 하는 일

**SAP Application Router**는 애플리케이션의 관문이다. 모든 HTTP 요청이 이곳을 거쳐간다.

```
사용자 → AppRouter → (인증) → (URL 분석) → 적절한 목적지로 전달
                                              ├── CAP 백엔드 (onCapService)
                                              ├── HTML5 Repository (UI5 앱)
                                              ├── Destination 서비스 (SAP ERP 등)
                                              └── SCIM ID 관리
```

---

## 2. 핵심 파일 설명

### xs-app.json / xs-app-dev.json / xs-app-qa.json - 라우팅 설정

**환경별 파일 구조:**
| 파일 | 환경 | 사용처 |
|---|---|---|
| `xs-app.json` | 운영(Production) | 기본 (mta.yaml 배포 시) |
| `xs-app-dev.json` | 개발(Development) | `npm run deploy:router` |
| `xs-app-qa.json` | QA | `npm run deploy:qarouter` |

**라우팅 규칙 분석:**

```json
// 1. CAP 백엔드 API (가장 높은 우선순위)
{ "source": "^.*/srv-api/(.*)$", "destination": "onCapService" }

// 2. 공통 UI5 라이브러리
{ "source": "/common.lib/(.*)", "service": "html5-apps-repo-rt" }

// 3. 각 UI5 앱
{ "source": "^.*/portal/(.*)$", "service": "html5-apps-repo-rt" }
{ "source": "^.*/sysmgt/(.*)$", "service": "html5-apps-repo-rt" }

// 4. 캐시버스팅 URL (~12345~/ 패턴)
{ "source": "^.*/portal/~[0-9]+~/(.*)$", "service": "html5-apps-repo-rt" }

// 5. 기본 (Fallback)
{ "source": "(.*)", "service": "html5-apps-repo-rt" }
```

**환경별 차이점:**
- **Dev**: `onCapService`, `onMDMService`, `onPlanService`, `onCloudConnector` destination 사용
- **QA**: `onCapQAService`, `onMDMQAService`, `onPlanQAService`, `onQACloudConnector` destination 사용 (QA 환경 전용)
- **Prod**: `onCapService`, `onMDMService`, `onPlanService`, `onCloudConnector`

> 즉, Dev/QA/Prod는 **Destination 이름**만 다르고 구조는 동일하다.

### xs-security.json - XSUAA 보안 설정

```json
{
  "xsappname": "com-app-xsapp",
  "tenant-mode": "dedicated",
  "oauth2-configuration": {
    "token-validity": 43200,           // 12시간
    "refresh-token-validity": 43200,   // 12시간
    "redirect-uris": [
      "https://*.cfapps.jp10.hana.ondemand.com/**",
      "https://*.jp10.applicationstudio.cloud.sap/**"
    ]
  }
}
```

**현재 상태:** scope, role-templates, role-collections가 모두 비어있다. 즉, 모든 인증된 사용자가 동일한 접근 권한을 가진다. 세분화된 권한이 필요하면 여기에 scope를 추가해야 한다.

### webapp/index.html
```html
<script type="text/javascript">
    location.href="/srv-api/odata/v4/frame/AuthCheck";
</script>
```

**흐름 설명:**
1. 사용자가 루트 URL(`/`)에 접속
2. AppRouter가 `index.html`을 서빙
3. JavaScript가 즉시 `/srv-api/odata/v4/frame/AuthCheck`로 리디렉션
4. AuthCheck 핸들러가 `/portal/index.html`로 리디렉션
5. 결과적으로 사용자는 포털 화면을 보게 된다

**왜 이런 방식을 쓰는가?** SAP BTP의 인증은 AppRouter 레벨에서 처리된다. 처음 접속 시 AppRouter가 XSUAA 로그인으로 리디렉션하고, 인증된 상태에서 `/srv-api/...` 경로로 접근하면 CAP 서버의 AuthCheck가 포털 URL로 다시 보내준다. 이를 통해 인증된 상태에서 포털에 진입할 수 있다.

### webapp/manifest.json - Welcome 앱
```json
{
  "sap.app": { "id": "comwelcome", "type": "application" },
  "sap.ui5": {
    "rootView": { "viewName": "welcome.view.Layout", "type": "XML" },
    "routing": { "routes": [], "targets": {} }
  }
}
```

`comwelcome`은 실질적인 UI가 없는 앱이다. 단순히 `index.html`에서 AuthCheck로 리디렉션하는 역할만 한다.

### mta.yaml - AppRouter 배포 설정
```yaml
modules:
  - name: com-approuter         # AppRouter 모듈
    type: approuter.nodejs
    requires:
      - com-app-repo-runtime    # HTML5 App Repo (런타임)
      - com-app-destination     # Destination 서비스
      - com-app-xsuaa           # XSUAA 인증
      - com-app-connectivity    # Connectivity (Cloud Connector)

  - name: comwelcome            # Welcome HTML5 모듈
    type: html5
    build-parameters:
      commands:
        - npm install
        - npm run build:ui      # ui5 build preload

resources:
  - com-app-repo-runtime (html5-apps-repo, app-runtime)
  - com-app-destination  (destination, lite)
  - com-app-xsuaa        (xsuaa, application)
  - com-app-connectivity (connectivity, lite)
```

**UI5 빌드 스크립트 (`ui5-deploy.yaml`):**
```yaml
builder:
  customTasks:
    - name: webide-extension-task-updateManifestJson
    - name: ui5-task-zipper
      configuration:
        archiveName: comwelcome
```

`ui5-task-zipper`는 빌드 결과물을 `.zip` 파일로 묶어준다. 이 zip 파일이 HTML5 Application Repository에 업로드된다.

---

## 3. AppRouter 에러 페이지

```
error/404.html → 400~404 상태 코드
error/500.html → 500~503 상태 코드
```

---

## 4. 로컬 개발 환경

```
dev/
├── default-env.json   # 로컬 환경 변수
└── xs-app.json        # 개발용 라우팅 설정 (xs-app-dev.json 복사본)
```

`npm run start:local` → AppRouter가 `./dev` 디렉토리를 workingDir로 사용하여 로컬에서 실행된다.

---

## 5. AppRouter 패키지 정보

```json
{
  "name": "approuter",
  "dependencies": { "@sap/approuter": "^20.0.1" },
  "scripts": {
    "start:local": "node node_modules/@sap/approuter/approuter.js --workingDir ./dev --port 5000",
    "start:ui": "ui5 serve --port 5001",
    "build:dev": "node -e \"require('fs').copyFileSync('xs-app-dev.json','xs-app.json')\" && mbt build"
  }
}
```

---

## 6. 현재 AppRouter 라우팅에 누락된 점

현재 `xs-app.json`(운영)과 `xs-app-dev.json`(개발)에는 **OTT 서비스 경로가 정의되어 있지 않다.** 즉, `cap-node/ott`의 TrendAnalysis, AnomalyAlert, FreeBoard 서비스는 현재 AppRouter를 통해 접근할 수 있는 경로가 없다.

**필요한 추가 작업:**
```json
// xs-app.json에 추가 필요
{
  "source": "^.*/ott-api/(.*)$",
  "target": "$1",
  "destination": "onOttService",
  "cacheControl": "no-cache, no-store, must-revalidate",
  "authenticationType": "xsuaa",
  "csrfProtection": false
}
```

그리고 BTP Cockpit → Destinations에 `onOttService` Destination도 생성해야 한다.
