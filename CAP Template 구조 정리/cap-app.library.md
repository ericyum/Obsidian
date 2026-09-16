# cap-app/library (commonlib) 분석

## 1. 개요

`cap-app/library` 폴더는 `commonlib`이라는 이름의 UI5 공유 라이브러리 프로젝트입니다. 공유 라이브러리는 여러 UI5 애플리케이션에서 공통적으로 사용되는 UI 컨트롤, 유틸리티 기능, 모델 등을 중앙에서 관리하고 재사용하기 위해 사용됩니다. 이를 통해 코드 중복을 줄이고 애플리케이션의 일관성을 유지할 수 있습니다.

이 문서는 `commonlib`의 구조와 주요 기능, 그리고 다른 애플리케이션에서 이를 어떻게 활용할 수 있는지 상세히 설명합니다.

## 2. 주요 파일 및 역할

### 2.1. `mta.yaml`

`mta.yaml` 파일은 이 라이브러리가 BTP 환경에 어떻게 배포되는지를 정의합니다.

```yaml
_schema-version: "3.2"
ID: commonlib
description: BTP Launchpad - Shared Library
version: 0.0.1
modules:
- name: com-lib-content
  type: com.sap.application.content
  path: .
  requires:
  - name: com-lib-repo-host
    parameters:
      content-target: true
  build-parameters:
    build-result: resources
    requires:
    - name: commonlib
      artifacts:
      - commonlib.zip
      target-path: resources/

- name: commonlib
  type: html5
  path: .
  build-parameters:
    build-result: dist
    builder: custom
    commands:
    - npm install
    - npm run build:cf
    supported-platforms: []

resources:
  - name: com-lib-repo-host
    type: org.cloudfoundry.managed-service
    parameters:
      service: html5-apps-repo
      service-plan: app-host
      # ...
```

*   **`modules`**:
    *   `commonlib`: 라이브러리 자체를 빌드하는 HTML5 모듈입니다. `npm run build:cf` 스크립트를 통해 UI5 빌드 프로세스를 실행합니다.
    *   `com-lib-content`: 빌드된 라이브러리(`commonlib.zip`)를 `html5-apps-repo` 서비스에 배포하는 역할을 합니다.
*   **`resources`**:
    *   `com-lib-repo-host`: `app-host` 서비스 플랜을 사용하는 `html5-apps-repo` 서비스입니다. 빌드된 라이브러리의 정적 파일들이 이곳에 저장되어 다른 애플리케이션들이 참조할 수 있게 됩니다.

### 2.2. `src/library.js` 및 `src/manifest.json`

이 두 파일은 `commonlib`을 UI5 라이브러리로 정의하는 핵심 파일입니다.

**`src/library.js`**:
```javascript
sap.ui.getCore().initLibrary({
    name: "common.lib",
    version: "1.0.0",
    dependencies: [
        "sap.ui.core"
    ],
    // ...
    noLibraryCSS: true
});
```
*   `sap.ui.getCore().initLibrary`를 호출하여 `common.lib`이라는 네임스페이스로 라이브러리를 초기화합니다.

**`src/manifest.json`**:
```json
{
    "sap.app": {
        "id": "commonlib",
        "type": "library",
        "title": "{{title}}",
        "applicationVersion": {
            "version": "1.0.0"
        }
    },
    // ...
}
```
*   `"type": "library"`를 통해 이 프로젝트가 애플리케이션이 아닌 라이브러리임을 명시합니다.

## 3. 핵심 기능 및 재사용 컴포넌트

`commonlib`는 다양한 커스텀 컨트롤과 유틸리티를 제공하여 개발 생산성을 향상시킵니다.

### 3.1. 커스텀 컨트롤: `CodeValueHelp.js`

`src/control/ui/CodeValueHelp.js`는 매우 유연하고 강력한 "값 도움말" 다이얼로그를 제공하는 커스텀 컨트롤입니다.

```javascript
const CodeValueHelp = Control.extend("common.lib.control.ui.CodeValueHelp", {
    metadata: {
        properties: {
            title: { type: "string" },
            multiSelection: { type: "boolean", defaultValue: false },
            keyField: { type: "string", defaultValue: "code" },
            textField: { type: "string", defaultValue: "code_name" },
            // ... more properties
        },
        events: {
            apply: {},
            cancel: {}
        }
    },
    // ... implementation
});
```

*   **주요 기능**:
    *   **OData 연동**: OData 서비스에 바인딩하여 동적으로 데이터 목록을 가져올 수 있습니다.
    *   **단일/다중 선택**: `multiSelection` 속성을 통해 단일 선택 또는 다중 선택 모드로 작동합니다.
    *   **검색 및 필터링**: 사용자가 키워드를 입력하여 목록을 쉽게 필터링할 수 있습니다.
    *   **높은 재사용성**: 다양한 속성을 통해 다이얼로그의 제목, 크기, 동작 등을 쉽게 커스터마이징할 수 있어, 여러 화면에서 일관된 사용자 경험을 제공하는 값 도움말 기능을 구현할 수 있습니다.

*   **작동 메커니즘**:
    1.  컨트롤이 화면에 추가되고 OData 서비스와 바인딩됩니다.
    2.  사용자가 값 도움말 아이콘을 클릭하면 `open` 메소드가 호출되어 다이얼로그가 나타납니다.
    3.  `loadData` 메소드가 실행되어 OData 서비스를 호출하고, 사용자 검색어(`sKeyword`)를 포함한 필터를 구성하여 데이터를 요청합니다.
    4.  백엔드에서 받은 데이터를 다이얼로그 내부의 테이블에 표시합니다.
    5.  사용자가 특정 항목을 선택하고 '확인'을 누르면 `apply` 이벤트가 발생하며, 선택된 값들이 원래의 입력 필드에 반영됩니다.

### 3.2. 유틸리티: `UI5Util.js`

`src/util/UI5Util.js`는 UI5 개발 시 자주 사용되는 헬퍼 함수들을 모아놓은 정적 클래스입니다.

```javascript
const UI5Util = {
    /**
     * 바인딩된 프로퍼티 값 조회
     */
    getPropertyFromBinding(control, property, tableModel, itemPathPrefix, view) {
        // ...
    },

    /**
     * 부모 컨트롤 찾기 (타입 체크)
     */
    findParentAs(control, typeName) {
        // ...
    },

    /**
     * 테이블 정렬/필터 초기화
     */
    resetSortedAndFiltered(table) {
        // ...
    }
};
```

*   **주요 기능**:
    *   `getPropertyFromBinding`: 데이터 바인딩된 컨트롤의 실제 값을 가져옵니다. 포맷터가 적용된 경우에도 계산된 값을 정확히 반환해줍니다.
    *   `findParentAs`: 현재 컨트롤에서 시작하여 특정 타입의 부모 컨트롤을 찾을 때까지 상위로 탐색합니다.
    *   `resetSortedAndFiltered`: 테이블에 적용된 모든 정렬과 필터 조건을 초기화하여 원래 상태로 되돌립니다.
    *   이 외에도 컨트롤 탐색, 값 속성 조회 등 다양한 편의 기능을 제공하여 복잡한 UI 로직을 간결하게 작성할 수 있도록 돕습니다.

## 4. 라이브러리 활용 방법

다른 UI5 애플리케이션에서 `commonlib`를 사용하려면, 해당 애플리케이션의 `manifest.json` 파일에 다음과 같이 종속성을 추가해야 합니다.

```json
"sap.ui5": {
    "dependencies": {
        "minUI5Version": "1.95.0",
        "libs": {
            "sap.ui.core": {},
            "sap.m": {},
            // ...
        },
        "components": {},
        "library": [
            {
                "name": "common.lib",
                "lazy": false
            }
        ]
    },
    // ...
}
```

또한, `approuter`의 `xs-app.json`에 라이브러리 경로를 라우팅하는 규칙이 추가되어야 합니다.

```json
{
  "source": "/common.lib/(.*)",
  "target": "/commonlib-1.0.0/$1",
  "service": "html5-apps-repo-rt",
  "authenticationType": "xsuaa"
}
```

이렇게 설정하면, UI5 애플리케이션의 XML View나 Controller에서 `common.lib`의 컨트롤과 유틸리티를 `xmlns:custom="common.lib.control.ui"`와 같이 네임스페이스를 지정하여 사용할 수 있습니다.

## 5. 결론

`commonlib`는 커스텀 UI 컨트롤과 유틸리티 함수를 제공하여 CAP 프로젝트 내 여러 UI 애플리케이션의 개발 효율성과 코드 품질을 높이는 핵심적인 역할을 합니다. 복잡하고 반복적인 UI 로직을 라이브러리로 캡슐화하고, 이를 다른 애플리케이션에서 쉽게 가져다 쓸 수 있도록 함으로써, 개발자는 비즈니스 로직 구현에 더 집중할 수 있습니다.
