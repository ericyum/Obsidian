# 06 - 프론트엔드: Portal (Launchpad)

> **네임스페이스:** `portal`  
> **MTA ID:** `comportal`  
> **UI5 App ID:** `comportal`

---

## 1. Portal이 하는 일

포털은 사용자가 로그인 후 가장 먼저 보게 되는 **메인 화면(Launchpad)** 이다. SAP Fiori Launchpad와 유사한 역할을 한다.

**주요 기능:**
- 🏠 홈 대시보드 (통계 타일, 차트)
- 📋 동적 메뉴 네비게이션 (상단 + 사이드)
- 🔍 메뉴 검색
- 👤 사용자 프로필 관리 (언어·테마·날짜서식 변경)
- 🧭 동적 라우팅 (메뉴 클릭 시 해당 UI5 앱 로드)

---

## 2. 파일 구조

```
portal/webapp/
├── Component.js                    # ★ 핵심: 동적 라우팅 설정
├── Component-preload.js            # 프리로드 (빌드 시 생성)
├── manifest.json                   # 앱 매니페스트
├── index.html                      # 진입점
├── favicon.ico
│
├── controller/
│   ├── BaseController.js           # common.lib.BaseController 상속
│   ├── Layout.controller.js        # ★ 레이아웃: 메뉴·검색·프로필
│   ├── Home.controller.js          # 홈 화면: 대시보드·차트
│   └── Search.controller.js        # 검색 결과 화면
│
├── view/
│   ├── Layout.view.xml             # ★ 메인 레이아웃 (ToolPage)
│   ├── Home.view.xml               # 홈 화면 (VizFrame 차트 포함)
│   ├── Search.view.xml             # 검색 결과 (GridContainer)
│   └── fragment/
│       └── UserProfile.fragment.xml # 사용자 프로필 설정 팝업
│
└── css/
    └── style.css                   # 스타일 시트
```

---

## 3. 핵심 흐름: Portal 로딩 시퀀스

```
1. index.html 로드
      ↓
2. Component.js init()
      │
      ├── [STEP 1] 사용자 세션 로드 (UserSessionLoader)
      │     GET /srv-api/odata/v2/core/UserManagement/UserSessionInfo
      │     응답 데이터:
      │       user_id, user_name, language_code, theme_code,
      │       date_format_type_name, digits_format_type_name, ...
      │     → UI5 Core 설정: 언어, 날짜/숫자 포맷, 테마
      │
      ├── [STEP 2] 권한 메뉴 로드 (MenuLoader)
      │     GET /srv-api/odata/v4/core/MenuManagement/MenusRoleAppliedList
      │     응답 데이터:
      │       [ { menu_code, menu_name, menu_app_id, menu_repo_path,
      │           menu_app_path, menu_route_path, parent_menu_code, ... } ]
      │     → Core Model "Menu"에 저장
      │
      └── [STEP 3] 동적 라우트/타겟 생성 (_setRouteAndTarget)
            │
            ├── AppCacheBuster 등록 (고유 repo 경로별)
            │
            ├── menu_app_id가 있는 메뉴만 처리:
            │   ├── UI5 loader 경로 설정: sap.ui.loader.config({ paths: { [id]: path } })
            │   ├── 동적 Route 추가: oRouter.addRoute({ pattern, name, target })
            │   ├── 동적 Target 추가: oRouter.getTargets().addTarget(name, { type: "Component", usage })
            │   └── componentUsages 동적 추가: manifest에 사용할 컴포넌트 등록
            │
            ├── 정적 Route 추가: "Home" (기본), "Search" (검색)
            │
            └── oRouter.initialize() → 라우팅 시작
      ↓
4. Layout.view.xml 렌더링
      ├── ToolHeader: 로고(이미지), 검색창(SearchField), 사용자명, 설정 버튼
      ├── SubHeader: 동적 상단 메뉴바 (MenuButton 그룹)
      ├── SideContent: 사이드 네비게이션 (SideNavigation)
      └── MainContents: App 컨테이너 → 현재 라우트의 View
```

---

## 4. Component.js 상세 분석

### 4.1 사용자 세션 처리
```javascript
new UserSessionLoader().attachEventOnce('ready', function(oEvent) {
    let oUserData = oModel.getData();
    // 날짜 포맷 설정
    oFormatSettings.setDatePattern("short", oUserData.date_format_type_name);
    // 그룹/소수점 구분자 설정
    let digitsGroup = oUserData.digits_format_type_name?.substr(3,1);  // ","
    let digitsDecimal = oUserData.digits_format_type_name?.substr(7,1); // "."
    oFormatSettings.setNumberSymbol("group", digitsGroup);
    oFormatSettings.setNumberSymbol("decimal", digitsDecimal);
    // 언어 설정
    oCoreConfig.setLanguage(oUserData.language_code || "EN");
    // 테마 설정
    Theming.setTheme(oUserData.theme_code || "sap_fiori_3");
});
```

**날짜서식 예시:** `digits_format_type_name`이 `"1,234,567.89"`라면, `substr(3,1)`은 `","`, `substr(7,1)`은 `"."` 이 된다. 이걸 보고 UI5가 숫자 포맷을 설정한다.

### 4.2 동적 메뉴 Route 생성 (_setRouteAndTarget)
```javascript
aMenuList.filter(o => !!o.menu_app_id).forEach(oTile => {
    const { menu_app_id, menu_code, menu_app_path, menu_route_path } = oTile;

    // 리소스 경로 설정 (Component 로드 위치)
    sap.ui.loader.config({ paths: { [menu_app_id]: menu_app_path } });

    // 라우트 패턴: "sysmgtcodeManagement/" 또는 "sysmgtcodeManagement/detail/{id}"
    const sPattern = `${menu_app_id}/${menu_route_path || ''}`;
    oRouter.addRoute({ pattern: sPattern, name: menu_code, target: { name: menu_code, prefix: menu_app_id } });

    // 타겟 추가 (Component 방식)
    oRouter.getTargets().addTarget(menu_code, {
        viewId: menu_app_id, type: "Component", usage: menu_code
    });

    // componentUsages 추가 (lazy loading)
    oConfig.componentUsages[menu_code] = {
        name: menu_app_id, settings: {}, componentData: { ... }, lazy: true
    };
});
```

**이것이 중요한 이유:** 포털은 메뉴가 몇 개일지, 어떤 UI5 앱이 연결될지 미리 알 수 없다. 그래서 **DB에서 가져온 메뉴 데이터를 기반으로 런타임에 동적으로 라우트를 생성**한다. `lazy: true`는 해당 메뉴를 실제로 클릭하기 전까지는 UI5 앱 컴포넌트를 로드하지 않는다는 뜻이다.

---

## 5. Layout.controller.js - 메뉴 · 검색 · 프로필

### 5.1 상단 메뉴바 생성 (createHeaderNavigation)

```javascript
oMenuTree.forEach(function(group) {
    if (group.menu_type_code === "Group") {
        let oMenuButton = new MenuButton({ text: group.menu_name, icon: group.menu_icon });
        let oMenu = new Menu();
        group.children.forEach(function(child) {
            let oMenuItem = new MenuItem({
                text: child.menu_name,
                icon: child.menu_icon,
                press: this.onSelectMenuRoute.bind(this, child.menu_code)
            });
            oMenu.addItem(oMenuItem);
        });
        oMenuButton.setMenu(oMenu);
        oToolHeader.addContent(oMenuButton);  // ToolHeader에 버튼 추가
    }
});
```

**구조:**
```
ToolHeader:
  [로고] [그룹1 ▼] [그룹2 ▼] [그룹3 ▼] [스페이서] [검색] [스페이서] [사용자명] [⚙]
                │
                ├── 메뉴항목1
                ├── 메뉴항목2
                └── 메뉴항목3
```

### 5.2 트리 빌드 (buildTree)
```javascript
buildTree: function(menuList, aFavoriteMenuList) {
    // 즐겨찾기 메뉴가 있으면 첫 번째 그룹으로 추가
    if (aFavoriteMenuList.length) {
        roots.push({
            menu_name: "{i18n>label.favoriteMenu}",
            menu_type_code: "Group",
            menu_icon: "sap-icon://bookmark",
            children: aFavoriteMenuList
        });
    }

    // parent_menu_code가 없는 메뉴를 루트로 식별
    menuList.forEach(item => {
        if (!item.parent_menu_code) {
            roots.push(map[item.menu_code]);
            addChildren(root, 1);  // 최대 3레벨까지 재귀적으로 자식 추가
        }
    });
}
```

### 5.3 메뉴 검색 (onSuggest, onSearch)
```
onSuggest → SearchField의 제안 목록을 Menu 모델에서 필터링
onSearch  → 제안 항목 클릭 시: 해당 메뉴로 라우팅
           → Enter 키 검색 시: /Search/{query} 경로로 라우팅
```

### 5.4 사용자 프로필 팝업 (onPressUserProfile)
```javascript
onPressUserProfileApply: function() {
    oUserManagementModel.update(
        `/UserSet(user_id='${oUserData.user_id}')`, 
        oSaveData,  // { language_code, theme_code, digits_format_type_code, date_format_type_code }
        { success: function(oReturnData) {
            // UI5 코어 설정 즉시 반영
            oFormatSettings.setDatePattern("short", oReturnData.date_format_type_name);
            oFormatSettings.setNumberSymbol("group", digitsGroup);
            oCoreConfig.setLanguage(oReturnData.language_code);
            Theming.setTheme(oReturnData.theme_code);
        }}
    );
}
```

---

## 6. Home.controller.js - 대시보드

현재는 **Mock 데이터**로 차트를 표시하고 있다.

```
대시보드 구성:
├── 데이터 현황 요약 (타일 3개)
│   ├── 전체 자재: 4,250 (Up)
│   ├── 품목 승인 대기: 12 (Critical)
│   └── 품질 오류 항목: 3 (Error)
│
├── 주요 관리 메뉴 (리스트)
│   └── 자재/BOM/공정/BP 관리
│
├── 최근 데이터 변경 이력 (테이블)
│
└── 차트 (SAP VizFrame)
    ├── 자재 유형별 현황 (세로 막대)
    └── 월별 신규 자재 등록 추이 (꺾은선)
```

**PDF 다운로드** 버튼도 구현되어 있다:
```javascript
onDownloadPDF: function() {
    fetch("/srv-api/odata/v2/core/FormService/TestPDF", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: { companyName: "Solum" } })
    })
    .then(response => response.blob())
    .then(blob => {
        // Blob → <a download> 태그로 강제 다운로드
    });
}
```

---

## 7. Search.controller.js - 검색 결과

```
onRouteMatched:
  → Menu 모델에서 menu_type_code === 'Menu' AND
    menu_name이 검색어를 포함하는 항목 필터링
  → GridContainer에 타일 형태로 표시
  → 결과 없으면 IllustratedMessage 표시

onTilePress:
  → 해당 메뉴 코드로 라우팅
```

---

## 8. manifest.json 주요 설정

```json
{
  "sap.app": { "id": "comportal" },
  "dataSources": {
    "UserManagementService": {
      "uri": "/srv-api/odata/v2/core/UserManagement",
      "type": "OData",
      "settings": { "odataVersion": "2.0" }
    }
  },
  "sap.ui5": {
    "models": {
      "i18n": {
        "type": "common.lib.model.CoreResourceModel",  // 동적 i18n
        "settings": { "bundleName": "common.lib.i18n.i18n" }
      },
      "UserManagement": {
        "dataSource": "UserManagementService",
        "preload": true
      }
    },
    "rootView": { "viewName": "portal.view.Layout", "type": "XML" },
    "componentUsages": {}  // 동적 추가됨
  }
}
```

**중요:** `componentUsages`는 빈 객체로 시작하지만, `Component.js`의 `_setRouteAndTarget`에서 동적으로 채워진다.
