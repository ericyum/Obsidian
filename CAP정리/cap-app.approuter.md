# cap-app/approuter 분석

## 1. 개요

`approuter`는 CAP 프로젝트의 진입점(Entry Point) 역할을 하는 중요한 모듈입니다. SAP BTP(Business Technology Platform) 환경에서 사용자 인증, 요청 라우팅, 정적 컨텐츠 제공 등 다양한 기능을 수행합니다. 이 문서는 `approuter` 모듈의 구조와 작동 방식을 상세히 설명하여 CAP 프로젝트를 처음 접하는 개발자도 쉽게 이해할 수 있도록 돕는 것을 목표로 합니다.

## 2. 주요 파일 및 역할

`approuter` 모듈의 핵심 기능은 여러 설정 파일에 의해 정의됩니다. 각 파일의 역할은 다음과 같습니다.

### 2.1. `mta.yaml`

`mta.yaml` 파일은 MTA(Multi-Target Application) 프로젝트의 배포 설명자 파일입니다. `approuter`와 관련된 모듈, 리소스, 종속성을 정의합니다.

```yaml
ID: com-approuter
_schema-version: "3.2"
version: 1.0.0

modules:
  - name: com-approuter
    type: approuter.nodejs
    path: .
    properties:
      IAS_XSUAA_XCHANGE_ENABLED: true
      COOKIE_BACKWARD_COMPATIBILITY: true #  쿠키 역호환성 허용
    requires:
      - name: com-app-repo-runtime
      - name: com-app-destination
      - name: com-app-xsuaa
      - name: com-app-connectivity

  # ... (other modules)

resources:
  - name: com-app-repo-runtime
    type: org.cloudfoundry.managed-service
    parameters:
      service-plan: app-runtime
      service: html5-apps-repo
      # ...
  - name: com-app-destination
    type: org.cloudfoundry.managed-service
    parameters:
      service: destination
      service-plan: lite
      # ...
  - name: com-app-xsuaa
    type: org.cloudfoundry.managed-service
    parameters:
      path: ./xs-security.json
      service-plan: application
      service: xsuaa
      # ...
  - name: com-app-connectivity
    type: org.cloudfoundry.managed-service
    parameters:
      service: connectivity
      service-plan: lite 
      # ...
```

*   **`modules`**:
    *   `com-approuter`: Node.js 기반의 앱라우터 모듈입니다. `requires` 섹션을 통해 BTP에서 제공하는 다양한 서비스(HTML5 앱 리포지토리, Destination, XSUAA, Connectivity)에 대한 종속성을 선언합니다.
*   **`resources`**:
    *   `approuter`가 필요로 하는 서비스 인스턴스를 생성합니다.
    *   `com-app-xsuaa`: 사용자 인증 및 권한 부여를 처리하는 XSUAA 서비스입니다. `xs-security.json` 파일을 참조하여 보안 설정을 구성합니다.
    *   `com-app-destination`: 백엔드 서비스의 주소를 안전하게 관리하는 Destination 서비스입니다.

### 2.2. `xs-security.json`

XSUAA 서비스의 상세 보안 설정을 정의하는 파일입니다.

```json
{
  "xsappname": "com-app-xsapp",
  "tenant-mode": "dedicated",
  "scopes": [],
  "role-templates": [],
  "role-collections": [],
  "attributes": [],
  "oauth2-configuration": {
    "token-validity": 43200, 
    "refresh-token-validity": 43200, 
    "redirect-uris": [
      "https://*.cfapps.jp10.hana.ondemand.com/**",
      "https://*.jp10.applicationstudio.cloud.sap/**",
      "https://*.authentication.jp10.hana.ondemand.com/**"
    ]
  },
  "foreign-scope-references": [
    "user_attributes"
  ]
}
```

*   `xsappname`: 이 보안 설정을 사용하는 애플리케이션의 고유 이름입니다.
*   `tenant-mode`: `dedicated`로 설정되어 있어, 각 테넌트가 독립적인 애플리케이션 인스턴스를 사용합니다.
*   `oauth2-configuration`: OAuth 2.0 인증 토큰의 유효 시간, 인증 후 리다이렉트될 URI 목록 등을 설정합니다.

### 2.3. `xs-app.json`

`approuter`의 핵심 동작인 라우팅 규칙을 정의하는 파일입니다.

```json
{
  "welcomeFile": "/index.html",
  "authenticationMethod": "route",
  "routes": [
    {
      "source": "^.*/srv-api/(.*)$",
      "target": "$1",
      "destination": "onCapService",
      "authenticationType": "xsuaa"
    },
    {
      "source": "^.*/mdm-api/(.*)$",
      "target": "$1",
      "destination": "onMDMService",
      "authenticationType": "xsuaa"
    },
    {
      "source": "/common.lib/(.*)",
      "target": "/commonlib-1.0.0/$1",
      "service": "html5-apps-repo-rt",
      "authenticationType": "xsuaa"
    },
    // ... (other routes)
  ]
}
```

*   **`welcomeFile`**: 사용자가 애플리케이션에 처음 접속했을 때 보여줄 기본 페이지를 지정합니다.
*   **`authenticationMethod`**: `route`로 설정되어 있어, 각 라우팅 규칙별로 인증 방법을 다르게 적용할 수 있습니다.
*   **`routes`**:
    *   `source`: 들어오는 요청의 URL 패턴을 정규식으로 정의합니다.
    *   `target`: `source` 패턴에 매칭된 요청을 어떻게 변환할지 정의합니다.
    *   `destination`: 요청을 전달할 목적지(백엔드 서비스)를 지정합니다. `mta.yaml`에 정의된 Destination 서비스에서 실제 주소를 찾아 요청을 전달합니다.
    *   `service`: `html5-apps-repo-rt`와 같이 BTP 내부 서비스를 지정할 수 있습니다. UI 애플리케이션과 같은 정적 컨텐츠를 제공할 때 사용됩니다.
    *   `authenticationType`: `xsuaa`로 설정된 경로는 XSUAA 서비스를 통해 인증된 사용자만 접근할 수 있습니다.

### 2.4. `package.json`

`approuter` 모듈 자체의 종속성을 관리하고 실행 스크립트를 정의하는 파일입니다.

```json
{
  "name": "approuter",
  "dependencies": {
    "@sap/approuter": "^20.0.1"
  },
  "scripts": {
    "start": "node node_modules/@sap/approuter/approuter.js",
    "build:ui": "ui5 build preload --clean-dest --config ui5-deploy.yaml  --include-task=generateCachebusterInfo",
    "build:cf": "mbt build",
    "deploy:cf": "cf deploy mta_archives/com-approuter_1.0.0.mtar"
  }
}
```

*   **`dependencies`**: `@sap/approuter` 패키지에 대한 의존성을 명시합니다. 이 패키지가 `approuter`의 실제 구현체입니다.
*   **`scripts`**: `start` (앱라우터 실행), `build` (MTA 빌드), `deploy` (배포) 등 다양한 개발 및 운영 작업을 위한 스크립트가 정의되어 있습니다.

## 3. 작동 흐름 및 메커니즘

`approuter`의 전체적인 작동 흐름은 다음과 같습니다.

1.  **사용자 요청**: 사용자가 웹 브라우저를 통해 애플리케이션 URL에 접속합니다.
2.  **`approuter` 수신**: BTP 환경의 `approuter`가 이 요청을 가장 먼저 수신합니다.
3.  **인증 확인**:
    *   `xs-app.json`의 라우팅 규칙에 `authenticationType: "xsuaa"`가 설정된 경우, `approuter`는 XSUAA 서비스와 통신하여 사용자가 인증되었는지 확인합니다.
    *   인증되지 않은 사용자는 SAP IDP(Identity Provider)의 로그인 페이지로 리다이렉트됩니다.
    *   로그인에 성공하면, XSUAA는 JWT(JSON Web Token)를 발급하고, `approuter`는 이 토큰을 이후의 백엔드 요청에 포함하여 전달합니다.
4.  **라우팅**:
    *   `approuter`는 요청 URL을 `xs-app.json`의 `routes` 배열에 정의된 `source` 패턴과 비교합니다.
    *   매칭되는 규칙을 찾으면, `destination` 또는 `service` 속성에 따라 요청을 적절한 곳으로 전달합니다.
        *   **Destination**: `onCapService`와 같은 Destination으로 요청을 보낼 경우, Destination 서비스에서 실제 백엔드 서비스의 URL을 조회하여 요청을 중계합니다. 이는 백엔드 서비스의 실제 주소를 숨기고 중앙에서 관리할 수 있게 해주는 중요한 메커니즘입니다.
        *   **Service**: UI 모듈과 같은 정적 파일을 요청하는 경우, `html5-apps-repo-rt` 서비스에서 해당 파일을 직접 제공합니다.
5.  **응답 반환**: 백엔드 서비스 또는 정적 파일 서비스로부터 받은 응답을 사용자에게 최종적으로 전달합니다.

### 특이사항: `index.html` 리다이렉션

이 프로젝트의 `approuter/webapp/index.html` 파일은 다음과 같이 구성되어 있습니다.

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

사용자가 `approuter`의 루트 URL에 접속하면, 이 `index.html`이 로드되어 즉시 `/srv-api/odata/v4/frame/AuthCheck` 경로로 리다이렉트됩니다. `xs-app.json`의 라우팅 규칙에 따라 이 경로는 `onCapService` Destination으로 전달되며, 이 과정에서 자연스럽게 XSUAA 인증 절차가 트리거됩니다. 즉, `approuter`는 단순히 진입점 역할만 하고, 실제 애플리케이션의 시작 및 인증 처리는 백엔드 서비스(`onCapService`)와의 상호작용을 통해 이루어지는 구조입니다.

## 4. 결론

`approuter`는 CAP 프로젝트의 문지기 역할을 수행하는 핵심 컴포넌트입니다. `mta.yaml`, `xs-security.json`, `xs-app.json` 등의 설정 파일을 통해 애플리케이션의 모듈과 리소스를 정의하고, 사용자 인증을 처리하며, 들어오는 모든 요청을 적절한 백엔드 서비스나 UI 애플리케이션으로 분배합니다. 이러한 구조를 이해하는 것은 CAP 기반의 클라우드 네이티브 애플리케이션을 개발하고 운영하는 데 있어 매우 중요합니다.
