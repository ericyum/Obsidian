# cap-app/sysmgt 문서

## 1. 개요

`cap-app/sysmgt` 디렉토리에는 **시스템 관리** 전용 개별 UI5 애플리케이션 모음이 포함되어 있습니다. 각 하위 폴더는 시스템의 특정 관리 도메인을 관리하는 독립적인 애플리케이션입니다.

이러한 애플리케이션은 `cap-node/core` 백엔드 모듈에서 제공하는 기능에 대한 사용자 인터페이스 역할을 합니다.

시스템 관리 애플리케이션 모음에는 다음이 포함됩니다:
-   **`codeManagement`**: 코드 목록 및 해당 값 관리를 위함.
-   **`menuManagement`**: 애플리케이션 메뉴 구조 및 권한 정의를 위함.
-   **`messageManagement`**: 시스템 전체 메시지 관리를 위함.
-   **`roleGroupManagement`**: 역할 그룹 관리를 위함.
-   **`roleManagement`**: 개별 역할 및 해당 할당 관리를 위함.
-   **`userManagement`**: 사용자 계정 및 해당 역할 그룹 할당 관리를 위함.

이 모든 애플리케이션은 `userManagement` 애플리케이션에서 예시된 유사한 아키텍처 패턴을 따릅니다.

## 2. 예제 애플리케이션: `userManagement`

`userManagement` 애플리케이션을 통해 관리자는 사용자 정보를 검색, 보기, 생성 및 편집하고 사용자를 다른 역할 그룹에 할당할 수 있습니다.

### 2.1. 아키텍처 및 종속성

-   **독립 실행형 애플리케이션**: 자체 `Component.js`, `manifest.json`, 뷰 및 컨트롤러가 있는 완전하고 독립적인 UI5 애플리케이션입니다.
-   **백엔드 서비스**: `manifest.json`은 `cap-node/core` 백엔드의 OData 서비스, 특히 `UserManagement` 및 `RoleGroupManagement`에 대한 종속성을 선언합니다.
-   **`common.lib`에 대한 종속성**: 이 애플리케이션(`sysmgt`의 다른 모든 애플리케이션과 마찬가지로)은 중앙 `common.lib` 라이브러리에 크게 의존합니다.
    -   컨트롤러는 라이브러리의 `BaseController`를 확장합니다.
    -   i18n 텍스트에 라이브러리의 `CoreResourceModel`을 사용합니다.
    -   라이브러리의 `ExcelUtil`과 같은 유틸리티를 사용합니다.

### 2.2. 구조 및 로직

**`Component.js`:**
구성 요소의 `init` 함수는 `this.getRouter().initialize()`를 호출하는 최소한의 기능만 수행합니다. 모든 공통 초기화 로직(사용자 데이터 로딩, 공통 코드 등)은 상속받은 `BaseController`에 의해 자동으로 처리됩니다.

**`manifest.json`:**
-   애플리케이션의 데이터 소스, 모델 및 라우팅을 정의합니다.
-   라우팅은 항목 목록(`MainList`)과 세부 정보 보기(`DetailList`)를 나란히 표시하기 위한 표준 Fiori 패턴인 `FlexibleColumnLayout`을 사용하도록 구성됩니다.

**`MainList.controller.js`:**
이 컨트롤러는 기본 사용자 목록을 관리하고 공유 라이브러리의 이점을 보여줍니다.
-   **상속**: `BaseController`를 확장합니다.
-   **`onInit`**: 부모의 `onInit` 메소드(`BaseController.prototype.onInit.call(this)`)를 호출하여 모든 공통 모델의 로딩을 트리거합니다.
-   **간소화된 데이터 가져오기**: `BaseController`에서 상속된 `this.readOdata()` 도우미 함수를 사용하여 데이터를 가져오므로 코드가 단순화되고 가독성이 향상됩니다.
    ```javascript
    // _getRoleGroup 함수에서
    this.readOdata({
        model: oDataModel,
        path: "/RoleGroupSet"
    }).then(function(oReturnData){
        // ... 데이터 검색 성공 처리
    }).catch(function(){
        // ... 오류 처리
    });
    ```
-   **비즈니스 로직에 집중**: `BaseController` 및 라이브러리 유틸리티가 공통 작업을 처리하므로 이 컨트롤러는 사용자 검색 필터 구축 및 세부 정보 페이지로의 탐색 처리와 같은 사용자 관리에 특정한 로직에만 집중할 수 있습니다.

## 3. 결론

`sysmgt` 애플리케이션은 집중적이고 효율적인 관리자 UI 세트입니다. 중앙의 재사용 가능한 라이브러리(`common.lib`)가 기반을 제공하여 개별 애플리케이션이 가볍고 특정 도메인 로직에만 집중할 수 있도록 하는 건전한 아키텍처를 보여줍니다. 이 패턴은 이 디렉토리의 모든 애플리케이션에서 반복될 가능성이 높습니다.
