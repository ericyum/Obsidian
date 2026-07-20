# 07 - 프론트엔드: System Management (시스템 관리)

> **MTA ID:** `comsysmgt`  
> **UI5 App ID:** `comsysmgt`  
> **구조:** 1개 umbrella 앱 + 6개 sub-apps

---

## 1. 개요

시스템 관리는 **관리자**가 시스템의 기반 데이터를 관리하는 UI다. 하나의 MTA로 패키징되지만, 내부에 6개의 독립적인 sub-application이 있다.

---

## 2. 파일 구조

```
sysmgt/webapp/
├── manifest.json                     # Umbrella manifest
│
├── codeManagement/                   # [1] 공통코드 관리
│   └── webapp/
│       ├── Component.js, Component-preload.js
│       ├── manifest.json, index.html
│       ├── controller/
│       │   ├── BaseController.js     # common.lib.BaseController 상속
│       │   ├── MainFlexibleColumnLayout.controller.js
│       │   ├── MainList.controller.js
│       │   └── DetailList.controller.js
│       ├── view/
│       │   ├── MainFlexibleColumnLayout.view.xml
│       │   ├── MainList.view.xml
│       │   └── DetailList.view.xml
│       ├── css/, xs-app.json
│       ├── package.json, ui5.yaml
│
├── menuManagement/                   # [2] 메뉴 관리
│   └── webapp/ (구조: codeManagement와 동일)
│
├── messageManagement/                # [3] 메시지 관리
│   └── webapp/ (구조: codeManagement와 동일)
│
├── roleManagement/                   # [4] 역할 관리
│   └── webapp/
│       ├── controller/
│       │   ├── App.controller.js     # FlexibleColumnLayout 컨트롤러
│       │   ├── BaseController.js
│       │   ├── MainList.controller.js      # 역할 목록
│       │   ├── DetailList.controller.js    # 역할별 메뉴/기능
│       │   └── DetailDetailList.controller.js  # 상세
│       ├── view/
│       │   ├── App.view.xml          # FlexibleColumnLayout
│       │   ├── MainList.view.xml
│       │   ├── DetailList.view.xml
│       │   ├── DetailDetailList.view.xml
│       │   ├── MainFlexibleColumnLayout.view.xml
│       │   └── fragment/
│       │       ├── AddUserDialog.fragment.xml
│       │       ├── AssignGroupDialog.fragment.xml
│       │       └── RoleGroupMapping.fragment.xml
│       ├── package.json, ui5.yaml
│
├── roleGroupManagement/              # [5] 역할그룹 관리
│   └── webapp/ (구조: codeManagement와 동일 + fragment 2개 추가)
│       └── view/fragment/
│           ├── AddGroupUserDialog.fragment.xml
│           └── GroupSyncDialog.fragment.xml
│
└── userManagement/                   # [6] 사용자 관리
    └── webapp/
        ├── controller/
        │   ├── App.controller.js     # FlexibleColumnLayout
        │   └── (Base, MainList, DetailList)
        ├── view/
        │   ├── App.view.xml
        │   ├── MainList.view.xml, DetailList.view.xml
        │   ├── MainFlexibleColumnLayout.view.xml
        │   └── fragment/
        │       ├── AddUserDialog.fragment.xml
        │       ├── AssignGroupDialog.fragment.xml
        │       └── RoleGroupMapping.fragment.xml
        └── package.json, ui5.yaml
```

---

## 3. 6개 Sub-App 개요

| Sub-App | DB 테이블 | CAP 서비스 | 설명 |
|---|---|---|---|
| **codeManagement** | CodeHeader, CodeItem | CodeManagement | 공통 코드 그룹 및 코드 관리 |
| **menuManagement** | Menu, MenuFunction, MenuLanguage, Favorite | MenuManagement | 메뉴 트리, 기능, 다국어, 즐겨찾기 관리 |
| **messageManagement** | Message | MessageManagement | 다국어 메시지 코드 관리 |
| **roleManagement** | Role, Role_Menu, Role_MenuFunction, Role_RoleGroup | RoleManagement | 역할 + 메뉴/기능 할당 + 역할그룹 매핑 |
| **roleGroupManagement** | RoleGroup, Role_RoleGroup, User_RoleGroup | RoleGroupManagement | 역할그룹 + 역할/사용자 할당 |
| **userManagement** | User, User_RoleGroup | UserManagement | 사용자 + 권한그룹 할당 |

---

## 4. 공통 UI 패턴

각 sub-app은 동일한 UI 패턴을 따른다. **Master-Detail** 구조:

### codeManagement 예시

```
MainList (왼쪽): 코드 그룹 목록
    │
    └── 클릭 → DetailList (오른쪽): 선택된 그룹의 코드 아이템 목록
```

### roleManagement 예시 (3-Column)

```
MainList (왼쪽): 역할 목록
    │
    └── 클릭 → DetailList (가운데): 역할별 메뉴 할당
                   │
                   └── 클릭 → DetailDetailList (오른쪽): 메뉴별 기능 할당
```

### userManagement / roleGroupManagement 예시 (FlexibleColumnLayout + Fragments)

```
MainList: 사용자/역할그룹 목록
    │
    └── 클릭 → DetailList: 할당된 그룹/사용자
                   │
                   └── Dialog (Fragment): 추가/할당 팝업
```

---

## 5. 각 Sub-App의 Controller 로직

### MainFlexibleColumnLayout.controller.js
FlexibleColumnLayout의 레이아웃(1-column, 2-column, 3-column)을 제어한다. 목록에서 항목을 선택하면 Detail 패널이 열리고, Detail에서 항목을 선택하면 DetailDetail 패널이 열리는 방식이다.

### MainList.controller.js
주요 목록 화면. OData 바인딩으로 데이터를 조회하고, 검색/필터/정렬 기능을 제공한다. `BaseController`의 `readOdata` 메서드를 사용한다.

### DetailList.controller.js
선택된 항목의 상세 정보를 표시하고 편집(수정/삭제/추가) 기능을 제공한다.

---

## 6. Fragment (팝업 다이얼로그)

| Fragment | 사용처 | 설명 |
|---|---|---|
| AddUserDialog | userManagement, roleManagement | 사용자 추가/할당 팝업 |
| AssignGroupDialog | userManagement, roleManagement | 그룹 할당 팝업 |
| RoleGroupMapping | userManagement, roleManagement | 역할그룹 매핑 관리 |
| AddGroupUserDialog | roleGroupManagement | 그룹에 사용자 추가 |
| GroupSyncDialog | roleGroupManagement | 그룹 동기화 |

---

## 7. 매니페스트 및 데이터 바인딩

각 sub-app은 자체 `manifest.json`에 OData DataSource를 정의하고, `Component.js`에서 모델을 초기화한다.

```json
// 예: codeManagement/manifest.json
{
  "sap.app": {
    "dataSources": {
      "CodeManagementService": {
        "uri": "/srv-api/odata/v2/core/CodeManagement",
        "type": "OData",
        "settings": { "odataVersion": "2.0" }
      }
    }
  },
  "sap.ui5": {
    "models": {
      "": { "dataSource": "CodeManagementService" }
    }
  }
}
```

---

## 8. 빌드 및 배포

각 sub-app은 개별 `package.json`과 `ui5.yaml`을 가지고 있으며, 상위 `sysmgt`의 `ui5-deploy.yaml`에서 모든 sub-app을 함께 빌드한다.

```yaml
# sysmgt/ui5-deploy.yaml
builder:
  customTasks:
    - name: ui5-task-zipper
      configuration:
        archiveName: comsysmgt
```

6개의 sub-app이 하나의 `comsysmgt.zip`으로 묶여 HTML5 Repository에 업로드된다.

```yaml
# sysmgt/mta.yaml
modules:
  - name: com-sysmgt-content
    type: com.sap.application.content
    requires:
      - name: com-sysmgt-repo-host
  - name: comsysmgt
    type: html5
    build-parameters:
      commands:
        - npm install
        - npm run build:cf
```
