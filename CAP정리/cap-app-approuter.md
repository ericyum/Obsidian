# cap-app/approuter 문서

## 1. 개요

`approuter`는 전체 웹 애플리케이션의 주요 진입점입니다. 이것은 사용자 인증을 처리하고 리버스 프록시 역할을 하는 표준 SAP 구성 요소로, 다양한 백엔드 서비스 및 프론트엔드 애플리케이션에 대한 요청을 전달합니다.

`approuter`의 주요 책임은 다음과 같습니다:
- **인증**: XSUAA 서비스를 통한 사용자 로그인 시행.
- **세션 관리**: 사용자 세션 및 타임아웃 처리.
- **라우팅**: URL 경로를 기반으로 적절한 마이크로서비스 또는 UI5 애플리케이션으로 요청 전달.
- **정적 콘텐츠 제공**: HTML5 애플리케이션 저장소에서 HTML5 애플리케이션(UI) 제공.

## 2. 구성 파일

`approuter`의 동작은 몇 가지 주요 구성 파일에 의해 정의됩니다.

### 2.1. `mta.yaml`

이 파일은 애플리케이션의 배포 설명자를 정의합니다. `approuter`의 경우 다음을 지정합니다:
- `approuter.nodejs` 모듈이라는 것.
- 기능에 중요한 종속 서비스:
    - **`com-app-xsuaa`**: 인증 및 권한 부여를 처리하기 위한 XSUAA 서비스 인스턴스.
    - **`com-app-destination`**: 백엔드 마이크로서비스의 URL을 조회하는 데 사용되는 대상 서비스.
    - **`com-app-repo-runtime`**: 컴파일된 UI5 애플리케이션이 저장되고 제공되는 HTML5 애플리케이션 저장소 서비스.
    - **`com-app-connectivity`**: 온프레미스 또는 기타 원격 시스템에 연결하기 위한 연결 서비스.

### 2.2. `xs-app.json`

이것은 라우팅 로직을 정의하는 핵심 구성 파일입니다.

#### 주요 속성:
- **`welcomeFile`**: `/index.html`. 로드될 기본 페이지.
- **`authenticationMethod`**: `route`. 각 라우트에 대해 개별적으로 인증이 처리됩니다.
- **`sessionTimeout`**: `60`분.
- **`logout`**: 로그아웃을 위한 엔드포인트를 정의합니다.

#### 라우팅 규칙 (`routes` 배열):

`routes` 배열은 이 파일에서 가장 중요한 부분입니다. 들어오는 요청 URL과 일치시킬 패턴 목록을 정의하고 요청을 전달할 위치를 지정합니다.

**백엔드 서비스 라우트:**
이러한 라우트는 API 호출을 백엔드 CAP 서비스로 전달합니다. `destination` 속성은 approuter에게 대상 서비스에서 URL을 조회하도록 지시합니다.

```json
[
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
  }
]
```
- `https://<app-url>/srv-api/Users`에 대한 요청은 `/Users` 경로와 함께 `onCapService` 대상(`cap-node/core` 서비스)으로 전달됩니다.
- `https://<app-url>/mdm-api/MaterialMasterSet`에 대한 요청은 `/MaterialMasterSet` 경로와 함께 `onMDMService` 대상(`cap-node/mdm` 서비스)으로 전달됩니다.

**UI5 애플리케이션 라우트:**
이러한 라우트는 다양한 프론트엔드 애플리케이션에 대한 정적 콘텐츠를 제공합니다. `service` 속성 `html5-apps-repo-rt`는 대상이 HTML5 애플리케이션 저장소에 있음을 나타냅니다.

```json
[
    {
      "source": "/common.lib/(.*)",
      "target": "/commonlib-1.0.0/$1",
      "service": "html5-apps-repo-rt"
    },
    {
      "source": "^.*/sysmgt/(.*)$",
      "target": "/comsysmgt-1.0.0/$1",
      "service": "html5-apps-repo-rt"
    },
    {
      "source": "^.*/mdm/(.*)$",
      "target": "/commdm-1.0.0/$1",
      "service": "html5-apps-repo-rt"
    },
    {
      "source": "^.*/portal/(.*)$",
      "target": "/comportal-1.0.0/$1",
      "service": "html5-apps-repo-rt"
    }
]
```
- `https://<app-url>/portal/index.html`과 같은 파일에 대한 요청은 HTML5 저장소에 저장된 `comportal-1.0.0` 애플리케이션에서 제공됩니다.

이 라우팅 구성은 프론트엔드 UI를 백엔드 서비스와 효과적으로 분리하여 독립적으로 개발, 배포 및 확장할 수 있도록 합니다.
