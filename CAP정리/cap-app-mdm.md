# cap-app/mdm 문서

## 1. 개요

`cap-app/mdm` 디렉토리에는 **마스터 데이터 관리**에 초점을 맞춘 포괄적인 UI5 애플리케이션 모음이 있습니다. `sysmgt` 모듈과 마찬가지로 이것은 개별 애플리케이션의 모음이며, 각 애플리케이션은 마스터 데이터의 특정 영역을 담당합니다.

이러한 애플리케이션은 `cap-node/mdm` 백엔드 서비스에 정의된 비즈니스 로직에 대한 사용자 인터페이스를 제공하여 사용자가 중요한 비즈니스 데이터를 생성, 관리 및 검색할 수 있도록 합니다.

마스터 데이터 관리 애플리케이션 모음에는 다음이 포함됩니다:
-   **`materialMasterManagement`**: 자재 마스터 레코드를 생성하고 관리하기 위한 핵심 애플리케이션.
-   **`materialMasterList`**: 자재 목록을 검색하고 보기 위함.
-   **`manageMakerInformation`**: 제조업체에 대한 데이터 처리 위함.
-   **`manageMaterialClassification`**: 자재 분류 계층 관리 위함.
-   **`makerSpecMappingManagement`**: 사양을 제조업체에 매핑하기 위함.
-   그리고 MDM의 특정 측면에 전념하는 많은 다른 애플리케이션들.

이 모음의 애플리케이션들은 모두 공통 아키텍처를 공유하며, 이는 `materialMasterManagement` 애플리케이션을 검토하여 이해할 수 있습니다.

## 2. 예제 애플리케이션: `materialMasterManagement`

이것은 자재 마스터 데이터 요청을 처리하기 위한 중앙 애플리케이션입니다. 사용자가 새로운 자재 생성 프로세스를 시작하고 목록 및 상세 객체 페이지를 통해 기존 자재를 관리할 수 있도록 합니다.

### 2.1. 아키텍처 및 종속성

-   **백엔드 서비스**: 애플리케이션은 `cap-node/mdm` 백엔드와 긴밀하게 통합되어 있습니다. `manifest.json`은 `MaterialMasterRequest`, `CommonCodeManagement` 및 `MaterialClassificationCode`를 포함한 여러 OData 서비스를 운영에 사용함을 보여줍니다.
-   **`common.lib` 기반**: `sysmgt` 앱과 마찬가지로 `common.lib` 라이브러리를 기반으로 구축되었습니다. 컨트롤러는 `BaseController`를 확장하고 라이브러리의 모델과 유틸리티를 사용하여 일관성을 보장하고 코드 중복을 줄입니다.

### 2.2. 구조 및 로직

**`manifest.json`:**
-   애플리케이션의 데이터 소스 및 모델을 정의합니다.
-   두 가지 주요 대상으로 라우팅을 구성합니다:
    -   **`MainList`**: 자재 마스터 요청 목록을 검색하고 표시하기 위한 뷰.
    -   **`ObjectPage`**: 단일 자재 마스터 레코드의 전체 세부 정보를 표시하는 데 사용되는 표준 Fiori 객체 페이지 뷰. 이는 복잡한 데이터를 보고 편집하는 풍부한 사용자 경험을 제공합니다.

**`MainList.controller.js`:**
이 컨트롤러는 기본 검색 및 목록 뷰를 관리합니다. 공통 아키텍처 패턴을 보여줍니다:
-   **`BaseController`를 확장**하고 부모 `onInit`을 호출하여 모든 공유 서비스가 로드되도록 합니다.
-   데이터 내보내기를 위해 **`ExcelUtil`**과 같은 라이브러리 구성 요소를 사용합니다.
-   컨트롤러의 코드는 자재 마스터 데이터와 관련된 **복잡한 비즈니스 로직**에 중점을 둡니다. 여기에는 다음이 포함됩니다:
    -   여러 필터를 사용한 복잡한 검색 쿼리 작성.
    -   계층적 값 도움말 처리 (예: 자재 분류 선택 시, 레벨 1의 선택이 레벨 2에서 사용 가능한 선택 항목을 필터링함).
    -   다양한 데이터 섹션의 완성도에 따라 자재 요청 상태를 결정하는 사용자 정의 로직 구현.
    ```javascript
    // onPressSearch의 복잡한 필터 로직 예시
    const sStatusKey = this.byId("crea_status_id").getSelectedKey();
    if (sStatusKey === "processing") {
        // [처리 중 조건] 상태 필드 중 하나라도 'R'(빨간색)인 경우
        aFilters.push(
            new Filter({ 
                filters: [
                    new Filter("BASIC_STATUS", "EQ", "R"),
                    new Filter("PLAN_STATUS", "EQ", "R"),
                    new Filter("AC_STATUS", "EQ", "R")
                ], 
                and: false 
            })
        );
    }
    ```

## 3. 결론

`mdm` 애플리케이션은 프론트엔드의 핵심 비즈니스 기능을 형성합니다. `sysmgt` 애플리케이션과 마찬가지로 `common.lib`에서 제공하는 견고하고 재사용 가능한 동일한 기반 위에 구축되었습니다. 그러나 마스터 데이터 관리와 관련된 복잡성과 규칙을 처리하기 위해 보다 정교하고 도메인별 UI 및 컨트롤러 로직을 포함하여 잘 구조화되고 확장 가능한 프론트엔드 아키텍처를 보여줍니다.
