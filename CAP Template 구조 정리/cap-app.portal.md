# cap-app/portal 분석

## 1. 개요

`cap-app/portal`은 CAP 프로젝트의 대문 역할을 하는 중앙 대시보드 애플리케이션입니다. 사용자가 로그인 후 가장 먼저 마주하게 되는 화면으로, 전체 시스템의 데이터 현황을 요약해서 보여주고, 다른 주요 업무 애플리케이션(예: 기준 정보 관리, 시스템 관리 등)으로 이동할 수 있는 관문(Gateway)의 역할을 수행합니다.

이 문서는 `portal` 애플리케이션의 구조, 주요 기능, 그리고 다른 모듈과의 상호작용 방식을 분석하여 전체 시스템의 중심 허브로서의 역할을 설명합니다.

## 2. 주요 파일 및 역할

### 2.1. `webapp/manifest.json`

`portal` 앱의 모든 설정이 집약된 핵심 파일입니다.

```json
{
    "sap.app": {
        "id": "comportal",
        "type": "application",
        "dataSources": {
			"UserManagementService": {
			  "uri": "/srv-api/odata/v2/core/UserManagement",
			  "type": "OData",
			  // ...
			}
        }
    },
    "sap.ui5": {
        "rootView": {
            "viewName": "portal.view.Layout",
            // ...
        },
        "models": {
            "i18n": {
                "type": "common.lib.model.CoreResourceModel",
                // ...
            },
            "UserManagement": {
                "dataSource": "UserManagementService",
                "preload": true
            }
        },
        "routing": {
            "config": { /* ... */ },
            "routes": [],
            "targets": {
                "Home": { /* ... */ },
                "Search": { /* ... */ }
            }
        }
    }
}
```

*   **`dataSources`**: `UserManagementService`라는 이름의 데이터 소스를 정의합니다. 이 서비스는 `/srv-api/...` 경로를 통해 `approuter`를 거쳐 핵심 CAP 서비스(`onCapService`)로 연결됩니다. 포털은 이 서비스를 통해 현재 로그인한 사용자의 이름과 같은 기본 정보를 가져옵니다.
*   **`rootView`**: 앱의 전체적인 레이아웃을 정의하는 `portal.view.Layout`을 최상위 뷰로 지정합니다.
*   **`models`**:
    *   `common.lib` 공유 라이브러리의 `CoreResourceModel`을 `i18n` 모델로 사용하여 다국어 처리를 합니다.
    *   `UserManagementService`를 기반으로 `UserManagement`라는 OData 모델을 생성하여 사용자 정보를 화면에 바인딩합니다.
*   **`routing`**: `routes` 배열이 비어있다는 점이 특징입니다. 이는 URL 해시(#)를 통한 화면 전환이 없음을 의미합니다. 대신, `targets`에 정의된 `Home`과 `Search` 뷰는 컨트롤러의 로직에 의해 프로그래밍 방식으로 화면에 표시됩니다.

### 2.2. `webapp/view/Layout.view.xml`

포털의 전체적인 UI 구조를 정의하는 뷰입니다.

```xml
<tnt:ToolPage id="toolPage">
    <tnt:header>
        <tnt:ToolHeader>
            <Image src="/common.lib/images/solum_logo.svg" />
            <ToolbarSpacer />
            <SearchField id="searchField" search=".onSearch" />
            <ToolbarSpacer />
            <Label text="{User>/user_name} ({User>/business_partner_code})" />
            <MenuButton icon="sap-icon://action-settings">
                <!-- 프로필, 로그아웃 메뉴 -->
            </MenuButton>
        </tnt:ToolHeader>
    </tnt:header>
    <tnt:mainContents>
        <App id="app" />
    </tnt:mainContents>
</tnt:ToolPage>
```

*   **`sap.tnt.ToolPage`**: Fiori 3 스타일의 표준 레이아웃을 사용하여 상단 헤더, (선택적) 사이드 네비게이션, 그리고 메인 컨텐츠 영역을 구성합니다.
*   **Header**: 헤더에는 로고, 메뉴 검색창, 로그인한 사용자 정보, 그리고 프로필/로그아웃 버튼이 위치하여 어느 화면에서든 공통적으로 접근할 수 있는 기능을 제공합니다.
*   **Main Contents**: `<App id="app" />` 컨트롤이 메인 컨텐츠 영역의 컨테이너 역할을 합니다. `manifest.json`에서 라우터의 타겟들이 이 `app` 컨트롤 내부에 표시되도록 설정되어 있습니다.

### 2.3. `webapp/view/Home.view.xml`

포털의 메인 화면으로, 다양한 정보를 요약하여 보여주는 대시보드 역할을 합니다.

```xml
<Page title="제조 마스터 데이터 관리 시스템 (MDM)" showHeader="false">
    <content>
        <!-- 데이터 현황 요약 타일 -->
        <GenericTile header="전체 자재" subheader="활성 상태" />
        <GenericTile header="품목 승인 대기" subheader="결재 필요" />

        <!-- 주요 관리 메뉴 목록 -->
        <List>
            <StandardListItem title="자재 마스터(Material Master) 관리" type="Navigation" />
            <StandardListItem title="BOM(Bill of Materials) 관리" type="Navigation" />
        </List>

        <!-- 최근 데이터 변경 이력 테이블 -->
        <Table items="{ path: '/MockRecentActivity' }">
            <!-- ... -->
        </Table>

        <!-- 데이터 시각화 차트 -->
        <vizControl:VizFrame vizType="column" />
        <vizControl:VizFrame vizType="line" />
    </content>
</Page>
```

*   **대시보드 UI**: `GenericTile`을 사용해 KPI 정보를, `List`를 사용해 다른 앱으로 이동할 수 있는 네비게이션 링크를, `Table`과 `VizFrame`(차트)을 사용해 데이터 현황을 시각적으로 제공합니다.
*   **정적/목업 데이터**: 뷰에 표시되는 데이터의 상당수가 현재는 하드코딩되거나 컨트롤러에 정의된 목업(Mock) 모델에 바인딩되어 있습니다. 실제 운영 환경에서는 이 부분들이 OData 서비스 호출을 통해 동적으로 채워지게 됩니다.

## 3. 작동 흐름 및 메커니즘

1.  **초기 진입**: 사용자가 `approuter`를 통해 시스템에 접속하면, 인증 과정을 거친 후 이 `portal` 애플리케이션으로 안내됩니다.
2.  **레이아웃 렌더링**: `Layout.view.xml`이 렌더링되어 포털의 기본 틀(헤더, 컨텐츠 영역 등)이 화면에 그려집니다. `Layout.controller.js`는 사용자 정보, 메뉴 목록 등을 OData 서비스로 요청하여 헤더 부분을 채웁니다.
3.  **홈 화면 표시**: `Layout.controller.js`는 라우터의 `navTo("Home")`과 같은 메소드를 호출하여 `Home.view.xml`을 메인 컨텐츠 영역(`<App id="app">`)에 표시합니다.
4.  **대시보드 데이터 로딩**: `Home.controller.js`는 OData 서비스를 호출하여 대시보드에 필요한 데이터(KPI, 변경 이력, 차트 데이터 등)를 가져와 모델에 설정하고, 뷰는 데이터 바인딩을 통해 자동으로 업데이트됩니다.
5.  **다른 앱으로 이동**: 사용자가 '주요 관리 메뉴' 목록에서 '자재 마스터 관리'를 클릭합니다.
6.  **화면 전환**: 해당 메뉴 항목의 `press` 이벤트 핸들러가 컨트롤러에서 실행됩니다. 컨트롤러는 `sap.ushell.services.CrossApplicationNavigation` 서비스(Fiori Launchpad 환경)나 `window.open` 등을 사용하여 해당 애플리케이션의 URL(예: `/mdm#materialMasterManagement`)로 이동시킵니다.
7.  **`approuter`의 재역할**: 이 새로운 URL 요청은 다시 `approuter`로 전달되고, `approuter`는 `mdm` 애플리케이션의 리소스를 사용자에게 제공하여 화면이 전환됩니다.

## 4. 결론

`cap-app/portal`은 단순히 예쁜 첫 화면이 아니라, 전체 시스템의 허브 역할을 하는 핵심적인 애플리케이션입니다. 사용자에게 시스템의 현재 상태에 대한 전반적인 뷰를 제공하고, 필요한 기능(애플리케이션)으로 쉽게 이동할 수 있는 시작점을 마련해줍니다. `common.lib`를 활용하여 일관된 UI/UX를 유지하고, OData 서비스를 통해 핵심 백엔드와 통신하며, 프로그래밍 방식의 화면 전환 로직을 통해 유연한 사용자 경험을 제공하는 잘 설계된 포털의 전형을 보여줍니다.
