# 05 - 프론트엔드: Library (공통 UI5 라이브러리)

> **네임스페이스:** `common.lib`  
> **MTA ID:** `commonlib`  
> **패키지명:** `commonlib`  
> **sapuxLayer:** `VENDOR`

---

## 1. 역할

모든 UI5 애플리케이션(portal, sysmgt 등)이 공통으로 사용하는 **공유 라이브러리**다. 커스텀 UI 컨트롤, 공통 모델, 유틸리티 클래스, i18n 관리 등을 제공한다.

---

## 2. 파일 구조

```
library/src/
├── library.js                         # 라이브러리 초기화
├── library-preload.js                 # 프리로드
├── manifest.json                      # 라이브러리 매니페스트
├── .library                           # UI5 라이브러리 메타데이터
│
├── control/                           # 커스텀 UI 컨트롤
│   ├── m/FavoriteButton.js            # 즐겨찾기 버튼
│   ├── m/PersonalizationManager.js    # 테이블 개인화
│   ├── m/variants/PersonalizationManager.js  # 개인화 변형
│   ├── tnt/NavigationListItem.js      # 확장된 네비게이션 아이템
│   ├── ui/CodeValueHelp.js            # 코드 값 도움말
│   ├── ui/Column.js                   # 확장 컬럼
│   ├── ui/Table.js                    # 확장 테이블
│   └── ui/ValueHelpDialog.js          # 값 도움말 다이얼로그
│
├── controller/
│   └── BaseController.js              # 모든 앱의 기본 컨트롤러 ★★★
│
├── model/
│   ├── CoreResourceModel.js           # 동적 i18n 리소스 모델
│   ├── JSONListScrollingModel.js      # 무한 스크롤 JSON 모델
│   ├── JSONWithReadModel.js           # 준비 완료 이벤트가 있는 JSON 모델
│   └── json/JSONListGrowableBinding.js # 증분 로딩 바인딩
│
├── util/
│   ├── CommonCodeLoader.js            # 공통 코드 로더 (코드 그룹별)
│   ├── ExcelUtil.js                   # Excel 파일 처리
│   ├── MenuLoader.js                  # 메뉴 트리 로더 ★★★
│   ├── MenuFunctionLoader.js          # 메뉴 기능 로더
│   ├── MessageLoader.js               # 동적 메시지 로더
│   ├── UI5Util.js                     # UI5 헬퍼 유틸
│   ├── UserSessionLoader.js           # 사용자 세션 로더 ★★★
│   ├── ValidateUtil.js                # 검증 유틸
│   └── ValidationHelper.js            # 검증 헬퍼
│
├── i18n/
│   ├── I18NModel.js                   # i18n 모델 헬퍼
│   ├── i18n.properties                # 기본 i18n
│   ├── i18n_en.properties             # 영어
│   ├── i18n_ko.properties             # 한국어
│   └── i18n_ko_KR.properties          # 한국어(대한민국)
│
├── thirdparty/sheetjs/                # Excel 라이브러리
│   ├── xlsx.min.js
│   ├── xlsx.full.min.js
│   ├── jszip.js
│   └── LICENSE
│
└── images/
    ├── logo_btp.webp
    ├── solum-favicon.png
    └── solum_logo.svg
```

---

## 3. 핵심 클래스 상세 분석

### 3.1 BaseController.js ★ 가장 중요

모든 UI5 앱의 컨트롤러가 상속받는 기본 클래스다.

```javascript
onInit({ FavoriteButton, PersoTable } = {}) {
    // 1. UserSessionLoader → 사용자 정보 (언어·날짜·숫자·테마)
    new UserSessionLoader().attachEventOnce('ready', (oEvent) => {
        this.getView().setModel(oModel, "User");
        sap.ui.getCore().setModel(oModel, "User");
    });

    // 2. CommonCodeLoader → 공통 코드 (LANGUAGE_CODE, MESSAGE_TYPE_CODE 등)
    new CommonCodeLoader().attachEventOnce('ready', (oEvent) => {
        this.getView().setModel(oModel, "CommonCode");
    });

    // 3. MenuLoader → 권한 메뉴 리스트
    new MenuLoader().attachEventOnce('ready', (oEvent) => {
        this.getView().setModel(oModel, "Menu");
    });

    // 4. MessageLoader → 다국어 메시지
    new MessageLoader().attachEventOnce('ready', (oEvent) => {
        this.getView().setModel(oModel, "Message");
    });

    // 5. MenuFunctionLoader → UI 기능 제어
    new MenuFunctionLoader().attachEventOnce('ready', (oEvent) => {
        // 현재 메뉴의 기능 설정 저장
        this._oMenuFunction = oModel.getProperty(`/${sMenuId}`) || {};
    });

    // 6. i18n 리소스 번들
    this.i18n = this.getResourceBundle();

    // 7. PersonalizationManager (테이블 개인화)
    if (this._oPersoTable) PersonalizationManager.init(this._oPersoTable);
}
```

**주요 메서드:**

| 메서드 | 설명 |
|---|---|
| `readOdata({ model, path, param })` | Promise 기반 OData read 유틸 |
| `fnDeepCopy(oData)` | 깊은 복사 (Date 제외) |
| `fnCellSelectorDepend(oTable)` | 테이블 셀 선택/복사 기능 |
| `fnMasterCodeValue(sCode, sGroupCode)` | 코드 → 코드명 조회 |
| `resetTableSettings(oTable)` | 테이블 정렬/필터 초기화 |
| `switchTablePersonalization(oTable, bEnable)` | 테이블 개인화 토글 |
| `onPressOpenPersoDialog(oEvt)` | 개인화 다이얼로그 열기 |

### 3.2 MenuLoader.js - 메뉴 로딩

```
MenuLoader()
  └── _initializeModel()
       └── _fetchList()
            └── GET /srv-api/odata/v4/core/MenuManagement/MenusRoleAppliedList
            └── → 권한별 메뉴 목록 반환
```

**중요 특징:**
- **싱글톤 패턴**: `oStaticModel`이 정적 변수. 한 번 로드되면 다시 API를 호출하지 않는다.
- **EventProvider 기반**: `ready` 이벤트를 발생시켜 구독자에게 알림.
- `attachEventOnce`를 사용하여 한 번만 이벤트 핸들러가 실행되도록 보장.

### 3.3 UserSessionLoader.js - 사용자 세션

```
UserSessionLoader()
  └── _fetchUserSession()
       └── GET /srv-api/odata/v2/core/UserManagement/UserSessionInfo
       └── → 사용자 언어·날짜서식·숫자서식·테마 등 세션 정보 반환
```

이 데이터를 기반으로 포털의 `Component.js`에서 UI5 코어의 언어, 날짜/숫자 포맷, 테마를 설정한다.

### 3.4 MessageLoader.js - 다국어 메시지

```
MessageLoader()
  └── _fetchMessageList()
       └── GET /srv-api/odata/v2/core/MessageManagement/MessageSet
           └── ?$filter=language_code eq 'KO'
       └── → { message_code: message_contents } 맵으로 변환
```

### 3.5 CoreResourceModel.js - 동적 i18n

```javascript
const MESSAGE_URL = "/srv-api/i18n.properties";

return Parent.extend("common.lib.model.CoreResourceModel", {
    constructor: function() {
        Parent.prototype.constructor.apply(this, arguments);
        this.enhance({ "url": MESSAGE_URL });
    }
});
```

**작동 방식:**
1. `CoreResourceModel`은 SAPUI5의 `ResourceModel`을 확장한 것이다.
2. 생성 시 `/srv-api/i18n.properties` URL을 사용하도록 `enhance()` 호출
3. 이 URL은 CAP core의 `server.ts`에서 동적으로 `.properties` 형식의 메시지를 제공한다.
4. DB의 `Message` 테이블에서 현재 언어에 맞는 메시지를 읽어 반환한다.

### 3.6 I18NModel.js - i18n 헬퍼

```javascript
I18NModel = {
    loadResourceModel: function() {
        return new ResourceModel({
            bundleName: "common.lib.i18n.i18n"  // 정적 .properties 파일
        });
    },
    getText: function(sKey, aArgs, bIgnoreKeyFallback) {
        return this.loadResourceModel().getResourceBundle().getText(sKey);
    }
};
```

`CoreResourceModel`이 동적 i18n(DB)을 처리하고, `I18NModel`은 정적 i18n(.properties 파일)을 처리한다.

---

## 4. 커스텀 UI 컨트롤

| 컨트롤 | 파일 | 설명 |
|---|---|---|
| FavoriteButton | `control/m/FavoriteButton.js` | 즐겨찾기 추가/제거 버튼 |
| PersonalizationManager | `control/m/PersonalizationManager.js` | 테이블 열/정렬 개인화 관리 |
| NavigationListItem | `control/tnt/NavigationListItem.js` | 새 탭 열기 지원하는 확장 네비게이션 아이템 |
| CodeValueHelp | `control/ui/CodeValueHelp.js` | 공통코드 값 도움말 다이얼로그 |
| Column | `control/ui/Column.js` | 필터/정렬 확장 컬럼 |
| Table | `control/ui/Table.js` | 확장 테이블 |
| ValueHelpDialog | `control/ui/ValueHelpDialog.js` | 값 도움말 다이얼로그 |

---

## 5. 배포 특이사항

```yaml
# mta.yaml
modules:
  - name: com-lib-content
    type: com.sap.application.content  # HTML5 Repository에 콘텐츠 업로드
    requires:
      - name: com-lib-repo-host
        parameters:
          content-target: true

  - name: commonlib
    type: html5
    build-parameters:
      builder: custom
      commands:
        - npm install
        - npm run build:cf   # ui5 build preload

resources:
  - name: com-lib-repo-host
    type: org.cloudfoundry.managed-service
    parameters:
      service: html5-apps-repo
      service-plan: app-host
```

**배포 순서:** 이 라이브러리는 다른 모든 UI5 앱보다 **먼저** 배포되어야 한다. portal, sysmgt 등이 `common.lib.*` 네임스페이스를 참조하기 때문이다.

**AppRouter 라우팅에서의 참조:**
```
/common.lib/(.*) → html5-apps-repo-rt:/commonlib-1.0.0/$1
```

즉, `/common.lib/images/solum_logo.svg`는 실제로 HTML5 Repository의 `/commonlib-1.0.0/images/solum_logo.svg`를 가리킨다.
