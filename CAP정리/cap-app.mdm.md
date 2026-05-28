# cap-app/mdm 분석

## 1. 개요

`cap-app/mdm` 폴더는 MDM(Master Data Management, 기준 정보 관리)과 관련된 여러 UI5 애플리케이션들을 하나로 묶어놓은 복합 프로젝트입니다. 일반적인 단일 UI5 애플리케이션과는 달리, 특정 비즈니스 기능(예: 자재 마스터 관리, 품목 이름 관리 등)을 수행하는 여러 개의 독립적인 미니 애플리케이션들이 `webapp` 폴더 내에 각각 존재하고 함께 배포되는 구조를 가집니다.

이 문서는 `cap-app/mdm`의 독특한 구조를 설명하고, 그 안에 포함된 개별 애플리케이션들의 공통적인 아키텍처와 작동 방식을 `materialMasterManagement`를 중심으로 상세히 분석합니다.

## 2. 프로젝트 구조: 애플리케이션의 집합

`cap-app/mdm`의 가장 큰 특징은 단일 애플리케이션이 아니라, 관련된 여러 애플리케이션의 집합체(Collection)라는 점입니다.

```
cap-app/mdm/
├── webapp/
│   ├── materialMasterManagement/  <- 자재 마스터 관리 앱
│   │   └── webapp/
│   │       ├── Component.js
│   │       ├── manifest.json
│   │       ├── controller/
│   │       └── view/
│   ├── materialItemNameManagement/  <- 품목 이름 관리 앱
│   │   └── webapp/
│   │       └── ...
│   ├── mdmCodeManagement/           <- MDM 공통 코드 관리 앱
│   │   └── webapp/
│   │       └── ...
│   └── ... (기타 여러 MDM 관련 앱들)
├── mta.yaml
├── package.json
└── ui5.yaml
```

*   **독립적인 하위 앱**: `webapp` 폴더 아래의 각 하위 폴더(예: `materialMasterManagement`)는 자신만의 `manifest.json`, `Component.js`, 뷰, 컨트롤러를 가진 완전한 형태의 UI5 애플리케이션입니다.
*   **통합 배포**: 프로젝트 루트의 `mta.yaml`과 `package.json`은 이 모든 하위 앱들을 하나의 배포 단위(`appmdm.zip`)로 묶어 BTP의 `html5-apps-repo`에 배포하는 역할을 합니다.
*   **라우팅**: 사용자가 BTP Launchpad에서 특정 MDM 기능의 타일을 클릭하면, `approuter`는 URL을 기반으로 `html5-apps-repo`에서 해당 하위 앱의 리소스(`Component.js` 등)를 찾아 로드해주는 방식으로 각 앱이 실행됩니다.

이러한 구조는 관련 기능들을 논리적으로 그룹화하면서도, 각 기능은 독립적으로 개발하고 유지보수할 수 있게 하는 유연성을 제공합니다.

## 3. 공통 아키텍처 분석 (materialMasterManagement 예시)

`cap-app/mdm` 내의 하위 앱들은 대부분 유사한 아키텍처를 공유합니다. `materialMasterManagement` 앱의 `webapp/manifest.json` 파일을 통해 공통적인 특징을 살펴보겠습니다.

```json
// materialMasterManagement/webapp/manifest.json
{
    "sap.app": {
        "id": "materialMasterManagement",
        "type": "application",
        "dataSources": {
            "mainService": {
                "uri": "/mdm-api/odata/v2/mdm/MaterialMasterRequest",
                "type": "OData",
                // ...
            }
        }
    },
    "sap.ui5": {
        "rootView": {
            "viewName": "materialMasterManagement.view.App",
            // ...
        },
        "models": {
            "i18n": {
                "type": "common.lib.model.CoreResourceModel",
                "settings": { "bundleName": "common.lib.i18n.i18n" }
            },
            "": {
                "dataSource": "mainService",
                "preload": true,
                // ...
            }
        },
        "routing": {
            "config": { /* ... */ },
            "routes": [
                { "pattern": "ObjectPage/:id:", "name": "ObjectPage", "target": ["ObjectPage"] },
                { "pattern": ":search:", "name": "MainList", "target": ["MainList"] }
            ],
            "targets": { /* ... */ }
        }
    }
}
```

### 3.1. 데이터 연동: OData 서비스

*   **`dataSources`**: 모든 하위 앱들은 백엔드와 OData V2 프로토콜로 통신합니다. `uri`가 `/mdm-api/...`로 시작하는 것을 볼 수 있는데, 이는 `approuter`의 `xs-app.json`에 정의된 라우팅 규칙을 통해 실제 백엔드 CAP 서비스(`onMDMService`)로 연결됩니다.
*   **`models`**: `dataSource`를 기반으로 OData 모델을 생성하여 뷰와 데이터를 바인딩합니다. `defaultBindingMode`를 `TwoWay`로 설정하여 사용자의 입력이 모델에 바로 반영되도록 하는 경우가 많습니다.

### 3.2. 공유 라이브러리(`common.lib`) 활용

*   **`models.i18n`**: 다국어 처리를 위한 리소스 모델(`i18n`)의 타입으로 `common.lib.model.CoreResourceModel`을 사용하고 있습니다. 이는 `cap-app/library`에서 개발한 공유 라이브러리의 기능을 직접 참조하는 부분으로, 공통 리소스(번역, 유틸리티 등)를 중앙에서 관리하고 있음을 보여줍니다.
*   XML View나 Controller에서도 `common.lib`에서 제공하는 커스텀 컨트롤(`CodeValueHelp` 등)이나 유틸리티(`UI5Util` 등)를 적극적으로 활용하여 개발 생산성을 높입니다.

### 3.3. 독립적인 라우팅

*   **`sap.ui5.routing`**: 각 하위 앱은 자신만의 내부 라우팅 로직을 가지고 있습니다. `materialMasterManagement`의 경우, 메인 리스트 화면(`MainList`)과 상세 정보 화면(`ObjectPage`) 간의 전환을 자체적으로 처리합니다. 이는 각 앱이 다른 앱에 영향을 주지 않고 독립적으로 작동할 수 있게 해주는 핵심 요소입니다.

## 4. 작동 흐름 및 메커니즘

1.  **실행**: 사용자가 Fiori Launchpad에서 '자재 마스터 관리' 타일을 클릭합니다.
2.  **`approuter`**: `approuter`는 해당 타일에 연결된 URL(예: `.../materialMasterManagement/index.html`) 요청을 수신합니다.
3.  **리소스 로딩**: `approuter`는 `html5-apps-repo`에서 `appmdm.zip` 내의 `materialMasterManagement` 애플리케이션 리소스를 찾아 브라우저에 제공합니다.
4.  **앱 초기화**: 브라우저는 `Component.js`를 로드하고 UI5 프레임워크가 앱을 초기화합니다.
5.  **데이터 요청**: 앱은 `manifest.json`에 정의된 `mainService` 데이터 소스를 통해 `/mdm-api/...` 경로로 데이터를 요청합니다.
6.  **서비스 호출**: `approuter`는 이 요청을 받아 `onMDMService` Destination에 매핑된 실제 CAP 서비스(Node.js)의 OData 엔드포인트를 호출합니다.
7.  **데이터 바인딩**: CAP 서비스가 반환한 데이터를 OData 모델을 통해 받아와 화면의 테이블이나 폼에 바인딩하여 사용자에게 보여줍니다.
8.  **내부 화면 전환**: 사용자가 목록에서 특정 자재를 클릭하면, 앱 내부의 라우터가 `ObjectPage` 경로로 URL을 변경하고 상세 화면으로 전환합니다. 이 모든 과정은 `materialMasterManagement` 앱 내에서 독립적으로 처리됩니다.

## 5. 결론

`cap-app/mdm`은 기준 정보 관리라는 큰 도메인 아래 관련된 여러 기능들을 독립적인 UI5 애플리케이션으로 각각 구현하고, 이를 하나의 패키지로 묶어 배포하는 효율적인 아키텍처를 채택하고 있습니다. 각 앱은 공유 라이브러리(`common.lib`)를 통해 코드와 리소스를 재사용하여 일관성을 유지하며, OData 서비스를 통해 CAP 백엔드와 통신하고, 자체적인 라우팅 로직을 통해 독립적으로 작동합니다. 이 구조는 대규모 비즈니스 애플리케이션을 기능 단위로 분할하여 관리하는 모범적인 사례를 보여줍니다.
