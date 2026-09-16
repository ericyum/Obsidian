# cap-app/sysmgt 분석

## 1. 개요

`cap-app/sysmgt` 폴더는 시스템 관리(System Management)에 필요한 다양한 기능들을 UI5 애플리케이션으로 구현하여 모아놓은 프로젝트입니다. `cap-app/mdm`과 동일한 아키텍처를 채택하여, 사용자 관리, 권한 관리, 메뉴 관리 등 각각의 독립적인 기능을 수행하는 미니 애플리케이션들을 하나의 패키지로 묶어 관리하고 배포합니다.

이 문서는 `sysmgt` 프로젝트의 전체적인 구조를 설명하고, 대표적인 하위 앱인 `userManagement`를 중심으로 개별 시스템 관리 앱들의 공통 아키텍처와 작동 방식을 분석합니다.

## 2. 프로젝트 구조: 시스템 관리 기능의 집합

`sysmgt` 프로젝트는 시스템 관리에 필요한 기능들을 독립적인 애플리케이션 단위로 분리하여 `webapp` 폴더 아래에 구성합니다.

```
cap-app/sysmgt/
├── webapp/
│   ├── userManagement/        <- 사용자 관리 앱
│   │   └── webapp/
│   │       ├── Component.js
│   │       ├── manifest.json
│   │       └── ...
│   ├── roleManagement/        <- 권한 관리 앱
│   │   └── webapp/
│   │       └── ...
│   ├── menuManagement/        <- 메뉴 관리 앱
│   │   └── webapp/
│   │       └── ...
│   └── ... (코드 관리, 메시지 관리 등)
├── mta.yaml
└── package.json
```

*   **모듈화된 기능**: 시스템 관리의 각 기능(사용자, 권한, 메뉴 등)이 개별 UI5 애플리케이션으로 구현되어 있어 기능별로 독립적인 개발 및 테스트, 유지보수가 용이합니다.
*   **통합 배포**: `mdm` 프로젝트와 마찬가지로, 루트의 `mta.yaml`은 이 모든 하위 앱들을 `comsysmgt.zip`이라는 단일 아카이브로 빌드하여 BTP의 `html5-apps-repo`에 배포합니다.
*   **실행 방식**: 사용자는 `portal` 앱이나 BTP Launchpad를 통해 특정 시스템 관리 메뉴를 선택하면, `approuter`가 해당 하위 앱의 리소스를 로드하여 실행시켜주는 방식으로 작동합니다.

## 3. 공통 아키텍처 분석 (userManagement 예시)

`sysmgt` 내의 모든 하위 앱들은 `mdm` 앱들과 마찬가지로 공통된 개발 패턴과 아키텍처를 따릅니다. `userManagement` 앱의 `webapp/manifest.json`을 통해 그 특징을 살펴보겠습니다.

```json
// userManagement/webapp/manifest.json
{
    "sap.app": {
        "id": "userManagement",
        "type": "application",
        "dataSources": {
            "mainService": {
                "uri": "/srv-api/odata/v2/core/UserManagement",
                // ...
            },
            "roleGroupService": {
                "uri": "/srv-api/odata/v2/core/RoleGroupManagement",
                // ...
            }
        }
    },
    "sap.ui5": {
        "models": {
            "i18n": { "type": "common.lib.model.CoreResourceModel", /* ... */ },
            "": { "dataSource": "mainService", /* ... */ },
            "roleGroupModel": { "dataSource": "roleGroupService", /* ... */ }
        },
        "routing": {
            "config": { /* ... */ },
            "routes": [
                { "pattern": "DetailList/:user_id:", "name": "DetailList", /* ... */ },
                { "pattern": ":search:", "name": "MainList", /* ... */ }
            ],
            "targets": {
                "MainFlexibleColumnLayout": { /* ... */ },
                "MainList": { /* ... */ },
                "DetailList": { /* ... */ }
            }
        }
    }
}
```

### 3.1. 백엔드 연동: 핵심 CAP 서비스(`core`)

*   **`dataSources`**: `userManagement` 앱은 `UserManagement`와 `RoleGroupManagement`라는 두 개의 데이터 소스를 사용합니다. 두 소스 모두 URI가 `/srv-api/odata/v2/core/...`로 시작하는데, 이는 이 앱들이 `cap-node/core` 모듈에서 제공하는 핵심 CAP 서비스와 직접적으로 통신함을 의미합니다. 즉, UI를 통해 이루어지는 모든 사용자 및 권한 그룹 관련 CRUD(Create, Read, Update, Delete) 작업은 `core` CAP 서비스의 비즈니스 로직을 통해 처리됩니다.
*   **`models`**: 각 데이터 소스를 기반으로 OData 모델을 생성하여 화면의 데이터와 동기화합니다.

### 3.2. Fiori UI 패턴 활용: `FlexibleColumnLayout`

*   **`routing.targets`**: `userManagement` 앱은 Fiori의 고급 레이아웃 중 하나인 `FlexibleColumnLayout`을 사용합니다. 이는 한 화면을 2개 또는 3개의 열로 분할하여 마스터-디테일(Master-Detail) 또는 마스터-디테일-디테일(Master-Detail-Detail) 구조의 화면을 효과적으로 구현할 수 있게 해주는 UI 패턴입니다.
*   **사용자 경험**: 라우팅 설정을 보면, 초기 화면(`MainList`)은 첫 번째 열(`beginColumnPages`)에 사용자 목록을 표시하고, 사용자가 특정 사용자를 선택하면 두 번째 열(`midColumnPages`)에 해당 사용자의 상세 정보(`DetailList`)가 나타나는 방식으로 동작합니다. 이를 통해 사용자는 컨텍스트를 잃지 않고 목록과 상세 정보를 한눈에 보며 작업할 수 있습니다.

### 3.3. `common.lib`를 통한 일관성 및 재사용성 확보

*   `userManagement` 앱 역시 `i18n` 모델 등에 `common.lib`의 리소스를 사용하여, 프로젝트 전반의 UI/UX 일관성을 유지하고 공통 기능의 재사용성을 높입니다.

## 4. 작동 흐름 및 메커니즘

1.  **진입**: 사용자가 포털에서 '사용자 관리' 메뉴를 클릭합니다.
2.  **앱 로딩**: `approuter`는 `comsysmgt.zip` 아카이브 내에서 `userManagement` 앱의 리소스를 찾아 브라우저에 로드합니다.
3.  **데이터 조회**: 앱이 초기화되면서, OData 모델은 `/srv-api/odata/v2/core/UserManagement` 엔드포인트를 호출하여 등록된 사용자 목록을 조회합니다.
4.  **화면 표시**: 조회된 사용자 목록을 `FlexibleColumnLayout`의 첫 번째 열(`MainList` 뷰)에 표시합니다.
5.  **상세 정보 조회**: 사용자가 목록에서 특정 사용자를 선택하면, 라우터는 `DetailList` 타겟으로 네비게이션을 트리거합니다. URL의 `user_id` 파라미터가 변경되고, OData 모델은 해당 `user_id`를 사용하여 특정 사용자의 상세 정보 및 할당된 권한 그룹 목록을 백엔드에 요청합니다.
6.  **상세 화면 표시**: 조회된 상세 정보를 `FlexibleColumnLayout`의 두 번째 열(`DetailList` 뷰)에 표시합니다.
7.  **데이터 변경**: 관리자가 상세 화면에서 사용자의 권한 그룹을 변경하고 저장하면, OData 모델은 `UPDATE` 또는 `CREATE`/`DELETE` 요청을 `/srv-api/...` 엔드포인트로 전송하여 백엔드 데이터베이스의 정보를 변경합니다.

## 5. 결론

`cap-app/sysmgt`는 `cap-app/mdm`과 동일한 "애플리케이션 집합" 아키텍처를 시스템 관리 기능에 적용한 모범 사례입니다. 각 관리 기능을 독립적인 앱으로 분리하여 개발 및 유지보수의 유연성을 확보하고, `FlexibleColumnLayout`과 같은 최신 Fiori UI 패턴을 적용하여 효율적인 사용자 경험을 제공합니다. 또한, 모든 데이터 처리는 `core` CAP 서비스를 통해 이루어지므로, UI와 백엔드 비즈니스 로직이 명확하게 분리된 구조를 가집니다.
