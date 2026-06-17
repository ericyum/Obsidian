# ZEN-OTT 기능3 & 기능6 — SAP UI5 프론트엔드 개발 명세서

> **작성일**: 2026-06-17
> **대상**: 기능3(구독자 추세 분석) + 기능6(자유 게시판) 백엔드에 대응하는 SAP UI5 프론트엔드
> **프로젝트 경로**: `cap-app/sysmgt/webapp/` 하위에 신규 모듈로 추가
> **UI5 버전**: 1.123.1 이상 (sap_horizon 테마)
> **OData 버전**: V2 (기존 cap-app 전역에서 V2 사용 중)

---

# 목차

1. [공통 아키텍처 이해](#1-공통-아키텍처-이해)
2. [기능3 — 구독자 추세 분석 대시보드](#2-기능3--구독자-추세-분석-대시보드)
3. [기능6 — 자유 게시판](#3-기능6--자유-게시판)
4. [공통 파일 템플릿](#4-공통-파일-템플릿)
5. [개발 순서](#5-개발-순서)

---

# 1. 공통 아키텍처 이해

## 1-1. cap-app 전체 구조

```
cap-app/
├── approuter/          # 앱 라우터 (정적 리소스 서빙 + 인증 프록시)
├── library/            # 공통 라이브러리 (BaseController, 커스텀 컨트롤, 유틸)
│   └── src/
│       ├── controller/BaseController.js    ← 모든 모듈 컨트롤러의 부모
│       ├── control/                        ← 커스텀 UI5 컨트롤
│       ├── model/                          ← CoreResourceModel 등
│       └── util/                           ← ValidationHelper, ExcelUtil 등
├── portal/             # 메인 포털 (레이아웃 + 홈 + 검색)
├── sysmgt/             # 시스템 관리 모듈들
│   └── webapp/
│       ├── codeManagement/     ← 참조 구현 (가장 완성도 높음)
│       ├── menuManagement/
│       ├── messageManagement/
│       ├── roleManagement/
│       ├── roleGroupManagement/
│       └── userManagement/
└── template/           # 신규 모듈 생성용 템플릿
```

## 1-2. 단일 모듈의 내부 구조

모든 모듈은 아래 패턴을 따릅니다 (`codeManagement` 기준):

```
[moduleName]/
├── package.json           # npm package (배포용)
├── ui5.yaml               # UI5 Tooling 설정
└── webapp/
    ├── Component.js        # UI5 컴포넌트 (라우터 초기화)
    ├── Component-preload.js # 빌드 산출물 (배포 최적화)
    ├── manifest.json       # 앱 디스크립터 (데이터소스, 라우팅, 모델)
    ├── index.html          # 진입점 (UI5 부트스트랩)
    ├── controller/
    │   ├── BaseController.js        # 모듈 전용 베이스 (common.lib.controller.BaseController 상속)
    │   ├── MainList.controller.js   # 목록 화면
    │   └── DetailList.controller.js # 상세 화면
    ├── view/
    │   ├── MainList.view.xml        # 목록 UI
    │   ├── DetailList.view.xml      # 상세 UI
    │   └── fragment/                # 재사용 프래그먼트
    └── css/
        └── style.css
```

## 1-3. 핵심 패턴 요약

### manifest.json 데이터소스

```json
"dataSources": {
    "ottTrendService": {
        "uri": "/srv-api/odata/v2/ott/trend/TrendAnalysis",
        "type": "OData",
        "settings": { "odataVersion": "2.0" }
    }
}
```

> **주의**: 기존 cap-app은 OData V2를 사용합니다. CAP 백엔드는 기본 V4이지만, `@sap/cds`는 V2 어댑터를 자동 제공하므로 `/odata/v4/` 대신 `/srv-api/odata/v2/` 경로 사용.

### 모델 계층

| 모델명 | 타입 | 용도 |
|--------|------|------|
| `""` (기본) | `sap.ui.model.odata.v2.ODataModel` | 백엔드 OData 바인딩 |
| `viewModel` | `sap.ui.model.json.JSONModel` | UI 상태 (검색조건, 편집모드, 레이아웃 등) |
| `i18n` | `common.lib.model.CoreResourceModel` | 다국어 리소스 |

### 컨트롤러 상속 계층

```
sap.ui.core.mvc.Controller
  └── common.lib.controller.BaseController  ← readOdata(), fnDeepCopy(), getResourceBundle()
        └── [module].controller.BaseController  ← onPressLayoutChange()
              └── [module].controller.MainList
              └── [module].controller.DetailList
```

### `readOdata()` 사용법

```js
this.readOdata({
    model: oDataModel,
    path: "/GroupTrendSet",
    param: {
        filters: aFilters,
        sorters: [new Sorter("analysis_month", true)],
        urlParameters: { "$expand": "group" }
    }
}).then(function(oReturnData) {
    // oReturnData.results → 배열
}).catch(function(oError) { ... });
```

---

# 2. 기능3 — 구독자 추세 분석 대시보드

## 2-1. 백엔드 API 참조

### 엔티티

#### SubscriberGroupSet (READ only)
| 필드 | 타입 | 설명 |
|------|------|------|
| `group_id` | String(30) | PK. 예: `SG_AGE_20S`, `SG_GENDER_M`, `SG_PLAN_PREMIUM` |
| `group_type` | String(30) | `AGE`, `GENDER`, `PLAN` |
| `group_name` | String(100) | 한글명. 예: `20대`, `남성`, `프리미엄` |

#### GroupTrendSet (READ only)
| 필드 | 타입 | 설명 |
|------|------|------|
| `trend_id` | UUID | PK |
| `group_id` | String(30) | FK → SubscriberGroup |
| `analysis_month` | String(7) | `YYYY-MM` |
| `total_members` | Integer | 전체 구독자 수 |
| `active_count` | Integer | 활성 구독자 수 |
| `churn_rate` | Decimal(5,2) | 이탈률 (%) |
| `avg_watch_seconds` | Integer64 | 1인당 평균 시청시간(초) |

#### ContentGroupStatSet (READ only)
| 필드 | 타입 | 설명 |
|------|------|------|
| `stat_id` | UUID | PK |
| `group_id` | String(30) | FK |
| `analysis_month` | String(7) | `YYYY-MM` |
| `genre` | String(100) | 장르명 (예: `액션`, `코미디`) |
| `watch_count` | Integer | 시청 횟수 |
| `total_seconds` | Integer64 | 총 시청시간(초) |

#### TrendAnomalySet (READ only + markAsRead 액션)
| 필드 | 타입 | 설명 |
|------|------|------|
| `anomaly_id` | UUID | PK |
| `group_id` | String(30) | FK |
| `anomaly_month` | String(7) | 발생 월 |
| `anomaly_type` | String(30) | `CHURN_SPIKE` / `WATCH_DROP` / `WATCH_SURGE` |
| `severity` | String(10) | `INFO` / `WARN` / `CRITICAL` |
| `change_rate` | Decimal(8,2) | 변화율 (%) |
| `alert_text` | String(500) | 한글 알림 메시지 |
| `is_read` | Boolean | 읽음 여부 |

### 액션

#### calculateTrends
```
POST /odata/v4/ott/trend/TrendAnalysis/calculateTrends
Content-Type: application/json
{ "analysisMonth": "2026-03" }

→ {
    "trend_count": 10,
    "anomaly_count": 3,
    "anomalies": [ ... ]
}
```

- **멱등성**: 동일 월 재호출 시 기존 데이터 삭제 후 재생성
- **권한**: `Admin` 롤 필요 (403)
- **에러**: `analysisMonth` 누락 시 400

#### markAsRead
```
POST /odata/v4/ott/trend/AnomalyAlert/markAsRead
Content-Type: application/json
{ "anomalyId": "uuid" }

→ 갱신된 TrendAnomaly 레코드
```

- **에러**: ID 누락 400, 존재하지 않는 ID 404

### OData V2 → V4 매핑 주의사항

| V4 | V2 |
|----|----|
| `group_id` | `group_id` (동일) |
| action 호출 | `POST .../calculateTrends` |
| 필터 `$filter=group_id eq '...'` | 동일 |
| 정렬 `$orderby=avg_watch_seconds desc` | 동일 |

## 2-2. 모듈 구조

```
sysmgt/webapp/trendAnalysis/
├── package.json
├── ui5.yaml
└── webapp/
    ├── Component.js
    ├── manifest.json
    ├── index.html
    ├── controller/
    │   ├── BaseController.js
    │   ├── Dashboard.controller.js          # 메인 대시보드
    │   └── AnomalyList.controller.js        # 이상 알림 팝업/리스트
    ├── view/
    │   ├── Dashboard.view.xml               # 대시보드 메인
    │   └── AnomalyList.view.xml             # 이상 알림 목록
    ├── fragment/
    │   ├── GroupTrendChart.fragment.xml      # 그룹별 추세 차트 (VizFrame)
    │   ├── GenreStats.fragment.xml           # 장르별 소비 통계
    │   └── AnomalyBadge.fragment.xml         # 헤더에 표시될 알림 배지
    └── css/
        └── style.css
```

## 2-3. 화면 설계

### 대시보드 메인 (Dashboard.view.xml)

```
┌──────────────────────────────────────────────────────┐
│  🔔 이상 알림 (3)          [분석월 선택▼] [분석실행] │  ← 헤더
├──────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │ 총 구독자 │ │ 활성 구독 │ │ 평균이탈률│ │ 평균시청│ │  ← KPI 카드 4개
│  │   47명   │ │   38명   │ │  12.5%   │ │ 8,230초│ │
│  └──────────┘ └──────────┘ └──────────┘ └────────┘ │
│                                                      │
│  ┌──────────────────────┐ ┌──────────────────────┐   │
│  │  그룹별 이탈률 (막대) │ │  그룹별 시청시간 (막대)│   │  ← 차트 2개
│  │  VizFrame           │ │  VizFrame            │   │
│  └──────────────────────┘ └──────────────────────┘   │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │  🔴 그룹별 추세 테이블 (GroupTrendSet)        │    │
│  │  ┌─────────────────────────────────────────┐ │    │
│  │  │ 그룹명 │ 인원 │ 활성 │ 이탈률 │ 시청시간 │ │    │
│  │  │ 20대  │  4  │  3  │  0%   │ 7,500   │ │    │
│  │  │ 40대  │  2  │  1  │ 50% 🔴│ 12,000  │ │    │
│  │  │ ...   │ ... │ ... │ ...   │ ...     │ │    │
│  │  └─────────────────────────────────────────┘ │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │  📊 장르별 소비 통계 (ContentGroupStatSet)     │    │
│  │  [그룹 필터▼]                                │    │
│  │  ┌─────────────────────────────────────────┐ │    │
│  │  │ 그룹  │ 장르   │ 시청수 │ 시청시간       │ │    │
│  │  │ 30대 │ 액션   │  12   │ 8,520         │ │    │
│  │  │ 30대 │ 스릴러 │   8   │ 3,300         │ │    │
│  │  └─────────────────────────────────────────┘ │    │
│  └──────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

### 이상 알림 목록 (AnomalyList.view.xml, 팝업 또는 페이지)

```
┌──────────────────────────────────────────────────────┐
│  이상 탐지 알림                           [모두 읽음]│
├──────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────┐  │
│  │ 🔴 CRITICAL | CHURN_SPIKE | 2026-04           │  │
│  │ SG_AGE_40S 그룹의 이탈률이 전월 대비 50%p     │  │
│  │ 급증했습니다. (현재: 50%)                     │  │
│  │                                     [읽음처리] │  │
│  └────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────┐  │
│  │ 🟡 WARN | WATCH_DROP | 2026-04                │  │
│  │ SG_PLAN_PREMIUM 그룹의 시청시간이 35% 감소... │  │
│  │                                     [읽음처리] │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

## 2-4. 파일별 구현 가이드

### 2-4-1. `manifest.json`

```json
{
    "_version": "1.59.0",
    "sap.app": {
        "id": "trendAnalysis",
        "type": "application",
        "applicationVersion": { "version": "1.0.0" },
        "dataSources": {
            "ottTrendService": {
                "uri": "/srv-api/odata/v2/ott/trend/TrendAnalysis",
                "type": "OData",
                "settings": { "odataVersion": "2.0" }
            },
            "ottAnomalyService": {
                "uri": "/srv-api/odata/v2/ott/trend/AnomalyAlert",
                "type": "OData",
                "settings": { "odataVersion": "2.0" }
            }
        }
    },
    "sap.ui": {
        "technology": "UI5",
        "deviceTypes": { "desktop": true, "tablet": true, "phone": true }
    },
    "sap.ui5": {
        "rootView": {
            "viewName": "trendAnalysis.view.Dashboard",
            "type": "XML",
            "async": true,
            "id": "Dashboard"
        },
        "dependencies": {
            "minUI5Version": "1.123.1",
            "libs": {
                "sap.m": {},
                "sap.f": {},
                "sap.ui.layout": {},
                "sap.viz": {}
            }
        },
        "models": {
            "i18n": {
                "type": "common.lib.model.CoreResourceModel",
                "settings": { "bundleName": "common.lib.i18n.i18n" }
            },
            "": {
                "dataSource": "ottTrendService",
                "preload": true,
                "settings": {
                    "defaultCountMode": "Inline",
                    "useBatch": true
                }
            },
            "anomalyModel": {
                "dataSource": "ottAnomalyService",
                "preload": true,
                "settings": {
                    "defaultCountMode": "Inline",
                    "useBatch": true
                }
            },
            "viewModel": {
                "type": "sap.ui.model.json.JSONModel"
            }
        },
        "resources": {
            "css": [{ "uri": "css/style.css" }]
        }
    }
}
```

> **주의**: 두 개의 OData 서비스(TrendAnalysis + AnomalyAlert)를 별도 데이터소스로 등록. 기본 `""` 모델은 TrendAnalysis, `anomalyModel`은 AnomalyAlert용.

### 2-4-2. `Component.js`

```js
sap.ui.define([
    "sap/ui/core/UIComponent"
], function(UIComponent) {
    "use strict";
    return UIComponent.extend("trendAnalysis.Component", {
        metadata: { manifest: "json" },
        init: function() {
            UIComponent.prototype.init.apply(this, arguments);
            // 라우팅 없음 — 단일 페이지 대시보드
        }
    });
});
```

### 2-4-3. `controller/BaseController.js`

```js
sap.ui.define([
    "common/lib/controller/BaseController"
], function(Controller) {
    "use strict";
    return Controller.extend("trendAnalysis.controller.BaseController", {
        // 모듈 공통 로직 (필요시 추가)
    });
});
```

### 2-4-4. `controller/Dashboard.controller.js` — 핵심 로직

```js
sap.ui.define([
    "./BaseController",
    "sap/ui/model/Filter",
    "sap/ui/model/Sorter",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel"
], function(BaseController, Filter, Sorter, MessageBox, MessageToast, JSONModel) {
    "use strict";

    return BaseController.extend("trendAnalysis.controller.Dashboard", {

        onInit: function() {
            // 1. viewModel 초기화
            var oViewModel = new JSONModel({
                analysisMonth: "2026-03",       // 기본 분석월
                trendCount: 0,
                anomalyCount: 0,
                isAnalyzing: false,
                kpi: {
                    totalMembers: 0,
                    activeCount: 0,
                    avgChurnRate: 0,
                    avgWatchSeconds: 0
                },
                groupFilter: "",                 // 장르통계 그룹 필터
                anomalyBadge: 0                  // 미읽음 알림 개수
            });
            this.getView().setModel(oViewModel, "viewModel");

            // 2. 알림 배지 갱신용 anomalyModel 준비
            this._oAnomalyModel = this.getOwnerComponent().getModel("anomalyModel");

            // 3. 초기 데이터 로드
            this._loadDashboard();
            this._loadAnomalyBadge();
        },

        // ===========================================
        // 대시보드 전체 데이터 로드
        // ===========================================
        _loadDashboard: function() {
            var oViewModel = this.getView().getModel("viewModel");
            var sMonth = oViewModel.getProperty("/analysisMonth");

            // GroupTrend + SubscriberGroup 동시 조회
            var oModel = this.getOwnerComponent().getModel();
            var aFilters = [new Filter("analysis_month", "EQ", sMonth)];

            this.readOdata({
                model: oModel,
                path: "/GroupTrendSet",
                param: {
                    urlParameters: { "$expand": "group" },
                    filters: aFilters,
                    sorters: [new Sorter("churn_rate", true)]  // 이탈률 높은 순
                }
            }).then(function(oData) {
                var aTrends = oData.results;

                // KPI 계산
                var iTotalMembers = 0, iActive = 0, fChurnSum = 0, iWatchSum = 0;
                aTrends.forEach(function(o) {
                    iTotalMembers += o.total_members || 0;
                    iActive += o.active_count || 0;
                    fChurnSum += o.churn_rate || 0;
                    iWatchSum += o.avg_watch_seconds || 0;
                });
                var iCnt = aTrends.length || 1;
                oViewModel.setProperty("/kpi", {
                    totalMembers: iTotalMembers,
                    activeCount: iActive,
                    avgChurnRate: (fChurnSum / iCnt).toFixed(1),
                    avgWatchSeconds: Math.round(iWatchSum / iCnt)
                });

                // 테이블 바인딩
                oViewModel.setProperty("/trendList", aTrends);

                // 차트 데이터 가공 (group_id → group_name)
                // ... VizFrame 데이터 설정
            }.bind(this));

            // ContentGroupStat 로드 (필터 가능)
            this._loadGenreStats(sMonth);
        },

        // ===========================================
        // 장르별 통계 로드
        // ===========================================
        _loadGenreStats: function(sMonth, sGroupId) {
            var oModel = this.getOwnerComponent().getModel();
            var aFilters = [new Filter("analysis_month", "EQ", sMonth)];
            if (sGroupId) {
                aFilters.push(new Filter("group_id", "EQ", sGroupId));
            }

            this.readOdata({
                model: oModel,
                path: "/ContentGroupStatSet",
                param: {
                    filters: aFilters,
                    sorters: [new Sorter("watch_count", true)]
                }
            }).then(function(oData) {
                this.getView().getModel("viewModel").setProperty("/genreStats", oData.results);
            }.bind(this));
        },

        // ===========================================
        // calculateTrends 액션 호출
        // ===========================================
        onPressAnalyze: function() {
            var oViewModel = this.getView().getModel("viewModel");
            var sMonth = oViewModel.getProperty("/analysisMonth");

            if (!sMonth) {
                MessageToast.show("분석 대상 월을 선택해주세요.");
                return;
            }

            oViewModel.setProperty("/isAnalyzing", true);
            var oView = this.getView();
            oView.setBusy(true);

            // OData V2에서 액션은 POST + Function Import 방식
            // CAP V2 어댑터: POST /calculateTrends with body
            var oModel = this.getOwnerComponent().getModel();
            var sPath = "/calculateTrends";

            // OData V2 action 호출
            oModel.callFunction(sPath, {
                method: "POST",
                urlParameters: {
                    "analysisMonth": sMonth
                }
            }).then(function(oResult) {
                // oResult는 액션 반환값 객체
                var oData = oResult;  // { trend_count, anomaly_count, anomalies }
                oViewModel.setProperty("/trendCount", oData.trend_count);
                oViewModel.setProperty("/anomalyCount", oData.anomaly_count);

                MessageToast.show(sMonth + " 분석 완료: " + oData.trend_count + "개 그룹, " + oData.anomaly_count + "건 이상 탐지");
                this._loadDashboard();
                this._loadAnomalyBadge();
                oView.setBusy(false);
                oViewModel.setProperty("/isAnalyzing", false);
            }.bind(this)).catch(function(oError) {
                MessageBox.error("분석 중 오류: " + (oError.message || "알 수 없는 오류"));
                oView.setBusy(false);
                oViewModel.setProperty("/isAnalyzing", false);
            }.bind(this));
        },

        // ===========================================
        // 분석월 변경
        // ===========================================
        onChangeMonth: function(oEvent) {
            var sMonth = oEvent.getParameter("value");
            this.getView().getModel("viewModel").setProperty("/analysisMonth", sMonth);
            this._loadDashboard();
        },

        // ===========================================
        // 그룹 필터 변경 (장르 통계)
        // ===========================================
        onChangeGroupFilter: function(oEvent) {
            var sGroupId = oEvent.getParameter("selectedItem").getKey();
            var sMonth = this.getView().getModel("viewModel").getProperty("/analysisMonth");
            this._loadGenreStats(sMonth, sGroupId || "");
        },

        // ===========================================
        // 이상 알림 배지 카운트
        // ===========================================
        _loadAnomalyBadge: function() {
            var aFilters = [new Filter("is_read", "EQ", false)];
            this.readOdata({
                model: this._oAnomalyModel,
                path: "/TrendAnomalySet",
                param: {
                    filters: aFilters,
                    urlParameters: { "$count": true }
                }
            }).then(function(oData) {
                this.getView().getModel("viewModel").setProperty("/anomalyBadge", oData.results.length);
            }.bind(this));
        },

        // ===========================================
        // 이상 알림 팝업 열기
        // ===========================================
        onPressAnomalyBadge: function() {
            // AnomalyList.view.xml을 Dialog로 로드
            if (!this._oAnomalyDialog) {
                this._oAnomalyDialog = sap.ui.xmlfragment(
                    "trendAnalysis.view.AnomalyList", this
                );
                this.getView().addDependent(this._oAnomalyDialog);
            }
            this._oAnomalyDialog.open();
            this._loadAnomalyList();
        },

        // ===========================================
        // 이상 알림 목록 로드 (Dialog 내부)
        // ===========================================
        _loadAnomalyList: function() {
            var aFilters = [new Filter("is_read", "EQ", false)];
            this.readOdata({
                model: this._oAnomalyModel,
                path: "/TrendAnomalySet",
                param: {
                    filters: aFilters,
                    sorters: [new Sorter("anomaly_month", true), new Sorter("severity", false)]
                }
            }).then(function(oData) {
                this.getView().getModel("viewModel").setProperty("/anomalyList", oData.results);
            }.bind(this));
        },

        // ===========================================
        // markAsRead 액션
        // ===========================================
        onPressMarkAsRead: function(oEvent) {
            var oBindingContext = oEvent.getSource().getBindingContext("viewModel");
            var sAnomalyId = oBindingContext.getProperty("anomaly_id");

            this._oAnomalyModel.callFunction("/markAsRead", {
                method: "POST",
                urlParameters: { "anomalyId": sAnomalyId }
            }).then(function() {
                MessageToast.show("읽음 처리되었습니다.");
                this._loadAnomalyList();
                this._loadAnomalyBadge();
            }.bind(this));
        },

        // ===========================================
        // 모든 알림 읽음 처리
        // ===========================================
        onPressMarkAllAsRead: function() {
            var aAnomalyList = this.getView().getModel("viewModel").getProperty("/anomalyList") || [];
            var that = this;
            var iTotal = aAnomalyList.length;
            var iDone = 0;

            if (iTotal === 0) return;

            aAnomalyList.forEach(function(oAnomaly) {
                that._oAnomalyModel.callFunction("/markAsRead", {
                    method: "POST",
                    urlParameters: { "anomalyId": oAnomaly.anomaly_id }
                }).then(function() {
                    iDone++;
                    if (iDone >= iTotal) {
                        MessageToast.show(iTotal + "건 모두 읽음 처리되었습니다.");
                        that._loadAnomalyList();
                        that._loadAnomalyBadge();
                    }
                });
            });
        }

    });
});
```

### 2-4-5. `view/Dashboard.view.xml`

```xml
<mvc:View
    controllerName="trendAnalysis.controller.Dashboard"
    xmlns:mvc="sap.ui.core.mvc"
    xmlns="sap.m"
    xmlns:f="sap.f"
    xmlns:form="sap.ui.layout.form"
    xmlns:layout="sap.ui.layout"
    xmlns:core="sap.ui.core"
    height="100%"
>
    <f:DynamicPage
        toggleHeaderOnTitleClick="true"
        headerExpanded="true"
        backgroundDesign="Standard"
    >
        <!-- ===== 헤더 영역 ===== -->
        <f:title>
            <f:DynamicPageTitle>
                <f:heading>
                    <HBox>
                        <Title text="구독자 추세 분석 대시보드"/>
                        <layoutData>
                            <FlexItemData alignSelf="Center" styleClass="sapUiTinyMarginBegin"/>
                        </layoutData>
                    </HBox>
                </f:heading>
                <f:actions>
                    <!-- 🔔 알림 배지 버튼 -->
                    <Button
                        icon="sap-icon://bell"
                        text="{= ${viewModel>/anomalyBadge} > 0 ? '(' + ${viewModel>/anomalyBadge} + ')' : ''}"
                        type="{= ${viewModel>/anomalyBadge} > 0 ? 'Emphasized' : 'Ghost'}"
                        press="onPressAnomalyBadge"
                    />
                    <!-- 분석월 선택 -->
                    <DatePicker
                        value="{viewModel>/analysisMonth}"
                        displayFormat="yyyy-MM"
                        valueFormat="yyyy-MM"
                        change="onChangeMonth"
                        width="12rem"
                    />
                    <!-- 분석 실행 버튼 -->
                    <Button
                        text="분석 실행"
                        type="Emphasized"
                        icon="sap-icon://refresh"
                        press="onPressAnalyze"
                        enabled="{= !${viewModel>/isAnalyzing}}"
                    />
                </f:actions>
            </f:DynamicPageTitle>
        </f:title>

        <!-- ===== 컨텐츠 영역 ===== -->
        <f:content>
            <ScrollContainer height="100%" horizontal="false" vertical="true">
                <VBox class="sapUiContentPadding">

                    <!-- ── KPI 카드 4개 ── -->
                    <HBox justifyContent="SpaceBetween" class="sapUiMediumMarginBottom">
                        <GenericTile header="{viewModel>/kpi/totalMembers}" subheader="총 구독자" size="M"/>
                        <GenericTile header="{viewModel>/kpi/activeCount}" subheader="활성 구독자" size="M"/>
                        <GenericTile header="{viewModel>/kpi/avgChurnRate}%" subheader="평균 이탈률" size="M"
                            headerColor="{= ${viewModel>/kpi/avgChurnRate} > 20 ? 'Critical' : 'Good'}"/>
                        <GenericTile header="{viewModel>/kpi/avgWatchSeconds}초" subheader="평균 시청시간" size="M"/>
                    </HBox>

                    <!-- ── 그룹별 추세 테이블 ── -->
                    <Panel headerText="그룹별 월간 추세" expandable="true" expanded="true">
                        <Table
                            id="trendTable"
                            items="{viewModel>/trendList}"
                            growing="true"
                            growingThreshold="10"
                        >
                            <columns>
                                <Column><Text text="그룹명"/></Column>
                                <Column hAlign="End"><Text text="전체 인원"/></Column>
                                <Column hAlign="End"><Text text="활성"/></Column>
                                <Column hAlign="End"><Text text="이탈률"/></Column>
                                <Column hAlign="End"><Text text="평균 시청(초)"/></Column>
                            </columns>
                            <items>
                                <ColumnListItem>
                                    <cells>
                                        <Text text="{viewModel>group/group_name}"/>
                                        <Text text="{viewModel>total_members}"/>
                                        <Text text="{viewModel>active_count}"/>
                                        <ObjectStatus
                                            text="{viewModel>churn_rate}%"
                                            state="{
                                                path: 'viewModel>churn_rate',
                                                formatter: '.churnStateFormatter'
                                            }"
                                        />
                                        <Text text="{viewModel>avg_watch_seconds}"/>
                                    </cells>
                                </ColumnListItem>
                            </items>
                        </Table>
                    </Panel>

                    <!-- ── 장르별 소비 통계 ── -->
                    <Panel headerText="장르별 콘텐츠 소비 통계" expandable="true" expanded="true">
                        <headerToolbar>
                            <Toolbar>
                                <Label text="그룹 필터:"/>
                                <Select change="onChangeGroupFilter" width="15rem">
                                    <items>
                                        <core:Item key="" text="전체 그룹"/>
                                        <core:Item key="SG_AGE_10S" text="10대"/>
                                        <core:Item key="SG_AGE_20S" text="20대"/>
                                        <core:Item key="SG_AGE_30S" text="30대"/>
                                        <core:Item key="SG_AGE_40S" text="40대"/>
                                        <core:Item key="SG_AGE_50S" text="50대"/>
                                        <core:Item key="SG_GENDER_M" text="남성"/>
                                        <core:Item key="SG_GENDER_F" text="여성"/>
                                        <core:Item key="SG_PLAN_BASIC" text="베이직"/>
                                        <core:Item key="SG_PLAN_STANDARD" text="스탠다드"/>
                                        <core:Item key="SG_PLAN_PREMIUM" text="프리미엄"/>
                                    </items>
                                </Select>
                            </Toolbar>
                        </headerToolbar>
                        <Table
                            id="genreTable"
                            items="{viewModel>/genreStats}"
                        >
                            <columns>
                                <Column><Text text="그룹"/></Column>
                                <Column><Text text="장르"/></Column>
                                <Column hAlign="End"><Text text="시청 횟수"/></Column>
                                <Column hAlign="End"><Text text="총 시청시간(초)"/></Column>
                            </columns>
                            <items>
                                <ColumnListItem>
                                    <cells>
                                        <Text text="{viewModel>group_id}"/>
                                        <Text text="{viewModel>genre}"/>
                                        <Text text="{viewModel>watch_count}"/>
                                        <Text text="{viewModel>total_seconds}"/>
                                    </cells>
                                </ColumnListItem>
                            </items>
                        </Table>
                    </Panel>

                </VBox>
            </ScrollContainer>
        </f:content>
    </f:DynamicPage>
</mvc:View>
```

### 2-4-6. `view/AnomalyList.view.xml` (프래그먼트로 구현 — Dialog)

```xml
<core:FragmentDefinition
    xmlns="sap.m"
    xmlns:core="sap.ui.core"
>
    <Dialog
        title="이상 탐지 알림"
        contentWidth="700px"
        contentHeight="500px"
        resizable="true"
    >
        <headerToolbar>
            <Toolbar>
                <ToolbarSpacer/>
                <Button text="모두 읽음" press="onPressMarkAllAsRead" type="Ghost"/>
            </Toolbar>
        </headerToolbar>
        <List
            items="{viewModel>/anomalyList}"
            noDataText="이상 알림이 없습니다."
        >
            <CustomListItem>
                <VBox class="sapUiSmallMargin">
                    <HBox justifyContent="SpaceBetween">
                        <ObjectStatus
                            text="{viewModel>anomaly_type}"
                            state="{
                                path: 'viewModel>severity',
                                formatter: '.severityStateFormatter'
                            }"
                            icon="{
                                path: 'viewModel>severity',
                                formatter: '.severityIconFormatter'
                            }"
                        />
                        <Text text="{viewModel>anomaly_month}"/>
                    </HBox>
                    <Text text="{viewModel>alert_text}" class="sapUiSmallMarginTopBottom"/>
                    <HBox justifyContent="End">
                        <Button
                            text="읽음 처리"
                            press="onPressMarkAsRead"
                            type="Default"
                            icon="sap-icon://accept"
                        />
                    </HBox>
                </VBox>
            </CustomListItem>
        </List>
    </Dialog>
</core:FragmentDefinition>
```

### 2-4-7. Formatter 함수 (Dashboard.controller.js에 추가)

```js
// severity → 상태 아이콘
severityIconFormatter: function(sSeverity) {
    switch(sSeverity) {
        case "CRITICAL": return "sap-icon://alert";
        case "WARN": return "sap-icon://warning";
        case "INFO": return "sap-icon://hint";
        default: return "sap-icon://information";
    }
},

// severity → ObjectStatus state
severityStateFormatter: function(sSeverity) {
    switch(sSeverity) {
        case "CRITICAL": return "Error";
        case "WARN": return "Warning";
        case "INFO": return "Information";
        default: return "None";
    }
},

// churn_rate → 색상
churnStateFormatter: function(fChurn) {
    if (fChurn >= 50) return "Error";
    if (fChurn >= 20) return "Warning";
    return "Success";
}
```

---

# 3. 기능6 — 자유 게시판

## 3-1. 백엔드 API 참조

### 엔티티

#### PostSet (CRUD)
| 필드 | 타입 | 설명 |
|------|------|------|
| `post_id` | UUID | PK (자동 생성, CREATE 시 제외) |
| `title` | String(200) | **필수**. 게시글 제목 |
| `content` | String(5000) | **필수**. 게시글 내용 |
| `author_user_id` | String(10) | FK → Users. **CREATE 시 자동 주입됨, 클라이언트에서 보내지 않음** |
| `view_count` | Integer | 조회수. **READ 시 서버에서 자동 증가** |
| `deleted_flag` | Boolean | 소프트 삭제 플래그. **DELETE 시 true로 설정** |
| `createdAt` | DateTime | 생성일시 (managed) |
| `modifiedAt` | DateTime | 수정일시 (managed) |

> **핵심 로직**:
> - `author_user_id`는 백엔드 `before CREATE` 핸들러에서 자동 주입 → 클라이언트는 `title`, `content`만 전송
> - 단건 READ(`$filter=post_id eq '...'` 또는 `/PostSet('uuid')`) 시 `view_count` 자동 +1
> - UPDATE/DELETE 시 작성자 본인 확인 → 타인이 시도하면 **403**
> - DELETE는 실제 삭제가 아닌 `deleted_flag = true` (소프트 삭제)

#### CommentSet (CRUD)
| 필드 | 타입 | 설명 |
|------|------|------|
| `comment_id` | UUID | PK (자동 생성) |
| `post_id` | UUID | FK → Post. **필수** |
| `content` | String(2000) | **필수**. 댓글 내용 |
| `author_user_id` | String(10) | FK → Users. **CREATE 시 자동 주입** |
| `createdAt` | DateTime | 생성일시 |
| `modifiedAt` | DateTime | 수정일시 |

> **핵심 로직**:
> - `author_user_id` 자동 주입 (Post와 동일)
> - UPDATE/DELETE 시 작성자 본인 확인 → 타인 시도 시 **403**
> - DELETE는 물리 삭제 (소프트 삭제 아님)

### 게시글 READ with 댓글 expand

```
GET /PostSet?$expand=comments
GET /PostSet('uuid')?$expand=comments
```

### 에러 응답

| 상황 | HTTP | 메시지 |
|------|------|--------|
| 미인증 | 401 | "인증된 사용자만 게시글을 작성할 수 있습니다." |
| 타인 수정 | 403 | "본인이 작성한 게시글만 수정/삭제할 수 있습니다." |
| 없는 글 | 404 | "게시글을 찾을 수 없습니다." |

## 3-2. 모듈 구조

```
sysmgt/webapp/freeBoard/
├── package.json
├── ui5.yaml
└── webapp/
    ├── Component.js
    ├── manifest.json
    ├── index.html
    ├── controller/
    │   ├── BaseController.js
    │   ├── PostList.controller.js          # 게시글 목록
    │   ├── PostDetail.controller.js        # 게시글 상세 + 댓글
    │   └── PostCreate.controller.js        # 게시글 작성 (다이얼로그 or 페이지)
    ├── view/
    │   ├── PostList.view.xml               # 게시글 목록
    │   ├── PostDetail.view.xml             # 게시글 상세 + 댓글
    │   └── PostCreate.view.xml             # 게시글 작성
    ├── fragment/
    │   ├── CommentItem.fragment.xml         # 댓글 아이템
    │   └── CommentCreate.fragment.xml       # 댓글 작성 입력창
    └── css/
        └── style.css
```

## 3-3. 화면 설계

### 게시글 목록 (PostList.view.xml)

```
┌──────────────────────────────────────────────────────┐
│  자유 게시판                           [글쓰기]      │
├──────────────────────────────────────────────────────┤
│  검색: [            ] 🔍                             │
├──────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────┐  │
│  │ 📌 [U001의 첫 번째 게시글]          👁️ 15      │  │
│  │    작성자: Kim Minjun | 2026-06-17 10:30       │  │
│  │    댓글 3개                                    │  │
│  ├────────────────────────────────────────────────┤  │
│  │ 📌 [U002의 게시글]                  👁️ 8       │  │
│  │    작성자: Lee Seoyeon | 2026-06-17 09:15      │  │
│  │    댓글 0개                                    │  │
│  └────────────────────────────────────────────────┘  │
│                            << < 1 2 3 > >>          │
└──────────────────────────────────────────────────────┘
```

### 게시글 상세 + 댓글 (PostDetail.view.xml)

```
┌──────────────────────────────────────────────────────┐
│  ← 목록               [수정] [삭제]                  │
├──────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────┐  │
│  │  U001의 첫 번째 게시글                         │  │
│  │                                                │  │
│  │  안녕하세요! 자유 게시판 테스트입니다.         │  │
│  │  이 글은 U001이 작성했습니다.                  │  │
│  │                                                │  │
│  │  작성자: Kim Minjun | 2026-06-17 10:30 | 👁️ 16│  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ── 댓글 2개 ──────────────────────────────────────  │
│  ┌────────────────────────────────────────────────┐  │
│  │ U002: 좋은 글이네요!                           │  │
│  │ 2026-06-17 11:00                    [수정][삭제]│  │
│  ├────────────────────────────────────────────────┤  │
│  │ U003: 저도 동의합니다.                         │  │
│  │ 2026-06-17 11:30                    [수정][삭제]│  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ── 댓글 작성 ────────────────────────────────────  │
│  ┌────────────────────────────────────────────────┐  │
│  │ [                                               ]│  │
│  │                                    [댓글 등록]  │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

## 3-4. 파일별 구현 가이드

### 3-4-1. `manifest.json`

```json
{
    "_version": "1.59.0",
    "sap.app": {
        "id": "freeBoard",
        "type": "application",
        "applicationVersion": { "version": "1.0.0" },
        "dataSources": {
            "ottBoardService": {
                "uri": "/srv-api/odata/v2/ott/board/FreeBoard",
                "type": "OData",
                "settings": { "odataVersion": "2.0" }
            }
        }
    },
    "sap.ui": {
        "technology": "UI5",
        "deviceTypes": { "desktop": true, "tablet": true, "phone": true }
    },
    "sap.ui5": {
        "rootView": {
            "viewName": "freeBoard.view.PostList",
            "type": "XML",
            "async": true,
            "id": "PostList"
        },
        "dependencies": {
            "minUI5Version": "1.123.1",
            "libs": {
                "sap.m": {},
                "sap.f": {},
                "sap.ui.layout": {}
            }
        },
        "models": {
            "i18n": {
                "type": "common.lib.model.CoreResourceModel",
                "settings": { "bundleName": "common.lib.i18n.i18n" }
            },
            "": {
                "dataSource": "ottBoardService",
                "preload": true,
                "settings": {
                    "defaultCountMode": "Inline",
                    "useBatch": true
                }
            },
            "viewModel": {
                "type": "sap.ui.model.json.JSONModel"
            }
        },
        "routing": {
            "config": {
                "routerClass": "sap.m.routing.Router",
                "viewType": "XML",
                "controlId": "app",
                "controlAggregation": "pages",
                "transition": "slide",
                "async": true
            },
            "routes": [
                {
                    "pattern": "",
                    "name": "PostList",
                    "target": "PostList"
                },
                {
                    "pattern": "detail/{post_id}",
                    "name": "PostDetail",
                    "target": "PostDetail"
                }
            ],
            "targets": {
                "PostList": {
                    "type": "View",
                    "viewType": "XML",
                    "id": "PostList",
                    "name": "PostList",
                    "path": "freeBoard.view",
                    "clearControlAggregation": true
                },
                "PostDetail": {
                    "type": "View",
                    "viewType": "XML",
                    "id": "PostDetail",
                    "name": "PostDetail",
                    "path": "freeBoard.view",
                    "clearControlAggregation": true
                }
            }
        },
        "resources": {
            "css": [{ "uri": "css/style.css" }]
        }
    }
}
```

### 3-4-2. `Component.js`

```js
sap.ui.define([
    "sap/ui/core/UIComponent"
], function(UIComponent) {
    "use strict";
    return UIComponent.extend("freeBoard.Component", {
        metadata: { manifest: "json" },
        init: function() {
            UIComponent.prototype.init.apply(this, arguments);
            this.getRouter().initialize();
        }
    });
});
```

### 3-4-3. `controller/BaseController.js`

```js
sap.ui.define([
    "common/lib/controller/BaseController"
], function(Controller) {
    "use strict";
    return Controller.extend("freeBoard.controller.BaseController", {
        // 현재 로그인 사용자 ID 반환
        getCurrentUserId: function() {
            var oUserModel = this.getView().getModel("User");
            return oUserModel ? oUserModel.getProperty("/userId") : null;
        },
        // 작성자 본인 확인 (UI 표시용)
        isAuthor: function(sAuthorId) {
            return this.getCurrentUserId() === sAuthorId;
        }
    });
});
```

### 3-4-4. `controller/PostList.controller.js`

```js
sap.ui.define([
    "./BaseController",
    "sap/ui/model/Filter",
    "sap/ui/model/Sorter",
    "sap/m/MessageToast"
], function(BaseController, Filter, Sorter, MessageToast) {
    "use strict";

    return BaseController.extend("freeBoard.controller.PostList", {

        onInit: function() {
            BaseController.prototype.onInit.call(this, {});

            var oViewModel = this.getOwnerComponent().getModel("viewModel");
            oViewModel.setProperty("/postList", []);
            oViewModel.setProperty("/searchText", "");

            // 라우팅 매치
            this.getOwnerComponent().getRouter()
                .getRoute("PostList")
                .attachPatternMatched(this._onPatternMatched, this);
        },

        _onPatternMatched: function() {
            this._loadPosts();
        },

        // 게시글 목록 로드 (삭제된 글 제외)
        _loadPosts: function() {
            var oModel = this.getOwnerComponent().getModel();
            var oViewModel = this.getOwnerComponent().getModel("viewModel");
            var sSearch = oViewModel.getProperty("/searchText") || "";
            var aFilters = [];

            // 소프트 삭제 제외
            aFilters.push(new Filter("deleted_flag", "EQ", false));

            // 검색어
            if (sSearch) {
                aFilters.push(new Filter({
                    filters: [
                        new Filter("title", "Contains", sSearch),
                        new Filter("content", "Contains", sSearch)
                    ],
                    and: false  // OR 조건
                }));
            }

            this.readOdata({
                model: oModel,
                path: "/PostSet",
                param: {
                    filters: aFilters,
                    sorters: [new Sorter("createdAt", true)],  // 최신순
                    urlParameters: {
                        "$expand": "comments",
                        "$count": true
                    }
                }
            }).then(function(oData) {
                // 댓글 개수 계산
                var aList = oData.results.map(function(oPost) {
                    oPost.commentCount = (oPost.comments && oPost.comments.results)
                        ? oPost.comments.results.length : 0;
                    return oPost;
                });
                oViewModel.setProperty("/postList", aList);
            }.bind(this));
        },

        // 검색
        onPressSearch: function(oEvent) {
            var sQuery = oEvent.getParameter("query") || oEvent.getParameter("value") || "";
            this.getOwnerComponent().getModel("viewModel").setProperty("/searchText", sQuery);
            this._loadPosts();
        },

        // 글쓰기 → 새 창 또는 이동
        onPressNewPost: function() {
            // PostCreate를 Dialog로 열기
            if (!this._oCreateDialog) {
                this._oCreateDialog = sap.ui.xmlfragment(
                    "freeBoard.view.PostCreate", this
                );
                this.getView().addDependent(this._oCreateDialog);
            }
            this._oCreateDialog.open();
        },

        // 게시글 상세 이동
        onPressPostItem: function(oEvent) {
            var sPostId = oEvent.getSource().getBindingContext("viewModel").getProperty("post_id");
            this.getOwnerComponent().getRouter().navTo("PostDetail", {
                post_id: sPostId
            });
        }

    });
});
```

### 3-4-5. `controller/PostDetail.controller.js`

```js
sap.ui.define([
    "./BaseController",
    "sap/m/MessageBox",
    "sap/m/MessageToast"
], function(BaseController, MessageBox, MessageToast) {
    "use strict";

    return BaseController.extend("freeBoard.controller.PostDetail", {

        onInit: function() {
            BaseController.prototype.onInit.call(this, {});
            this.getOwnerComponent().getRouter()
                .getRoute("PostDetail")
                .attachPatternMatched(this._onPatternMatched, this);
        },

        _onPatternMatched: function(oEvent) {
            var sPostId = oEvent.getParameter("arguments").post_id;
            this._sPostId = sPostId;
            this._loadPostDetail(sPostId);
        },

        // 게시글 + 댓글 로드 (단건 → view_count 자동 증가)
        _loadPostDetail: function(sPostId) {
            var oModel = this.getOwnerComponent().getModel();
            var oViewModel = this.getOwnerComponent().getModel("viewModel");

            // OData V2에서 key로 조회
            oModel.read("/PostSet('" + sPostId + "')", {
                urlParameters: { "$expand": "comments" },
                success: function(oPost) {
                    oViewModel.setProperty("/currentPost", oPost);
                    oViewModel.setProperty("/comments", oPost.comments?.results || []);
                    // 소프트 삭제 확인
                    if (oPost.deleted_flag) {
                        MessageToast.show("삭제된 게시글입니다.");
                    }
                },
                error: function() {
                    MessageToast.show("게시글을 불러올 수 없습니다.");
                }
            });
        },

        // 수정 모드 전환
        onPressEdit: function() {
            var oPost = this.getView().getModel("viewModel").getProperty("/currentPost");
            if (!this.isAuthor(oPost.author_user_id)) {
                MessageToast.show("본인이 작성한 게시글만 수정할 수 있습니다.");
                return;
            }
            this.getView().getModel("viewModel").setProperty("/editMode", true);
        },

        // 수정 저장
        onPressSave: function() {
            var oModel = this.getOwnerComponent().getModel();
            var oViewModel = this.getView().getModel("viewModel");
            var oPost = oViewModel.getProperty("/currentPost");

            oModel.update("/PostSet('" + oPost.post_id + "')", {
                title: oPost.title,
                content: oPost.content
            }, {
                success: function() {
                    MessageToast.show("수정되었습니다.");
                    oViewModel.setProperty("/editMode", false);
                },
                error: function(oErr) {
                    if (oErr.statusCode === 403) {
                        MessageToast.show("본인이 작성한 게시글만 수정할 수 있습니다.");
                    } else {
                        MessageToast.show("수정 중 오류가 발생했습니다.");
                    }
                }
            });
        },

        // 삭제 (소프트 삭제)
        onPressDelete: function() {
            var oPost = this.getView().getModel("viewModel").getProperty("/currentPost");
            if (!this.isAuthor(oPost.author_user_id)) {
                MessageToast.show("본인이 작성한 게시글만 삭제할 수 있습니다.");
                return;
            }

            MessageBox.confirm("정말 삭제하시겠습니까?", {
                onClose: function(sAction) {
                    if (sAction !== MessageBox.Action.OK) return;

                    var oModel = this.getOwnerComponent().getModel();
                    oModel.remove("/PostSet('" + oPost.post_id + "')", {
                        success: function() {
                            MessageToast.show("삭제되었습니다.");
                            this.getOwnerComponent().getRouter().navTo("PostList");
                        }.bind(this),
                        error: function(oErr) {
                            if (oErr.statusCode === 403) {
                                MessageToast.show("본인이 작성한 게시글만 삭제할 수 있습니다.");
                            }
                        }.bind(this)
                    });
                }.bind(this)
            });
        },

        // 댓글 작성
        onPressAddComment: function() {
            var oModel = this.getOwnerComponent().getModel();
            var sContent = this.byId("commentInput").getValue();

            if (!sContent || !sContent.trim()) {
                MessageToast.show("댓글 내용을 입력해주세요.");
                return;
            }

            oModel.create("/CommentSet", {
                post_id: this._sPostId,
                content: sContent.trim()
            }, {
                success: function() {
                    MessageToast.show("댓글이 등록되었습니다.");
                    this.byId("commentInput").setValue("");
                    this._loadPostDetail(this._sPostId);  // 목록 갱신
                }.bind(this),
                error: function() {
                    MessageToast.show("댓글 등록 중 오류가 발생했습니다.");
                }.bind(this)
            });
        },

        // 댓글 삭제
        onPressDeleteComment: function(oEvent) {
            var oComment = oEvent.getSource().getBindingContext("viewModel").getObject();
            if (!this.isAuthor(oComment.author_user_id)) {
                MessageToast.show("본인이 작성한 댓글만 삭제할 수 있습니다.");
                return;
            }

            var oModel = this.getOwnerComponent().getModel();
            oModel.remove("/CommentSet('" + oComment.comment_id + "')", {
                success: function() {
                    MessageToast.show("댓글이 삭제되었습니다.");
                    this._loadPostDetail(this._sPostId);
                }.bind(this),
                error: function(oErr) {
                    if (oErr.statusCode === 403) {
                        MessageToast.show("본인이 작성한 댓글만 삭제할 수 있습니다.");
                    }
                }.bind(this)
            });
        },

        // 목록으로 이동
        onPressNavBack: function() {
            this.getOwnerComponent().getRouter().navTo("PostList");
        }

    });
});
```

### 3-4-6. `view/PostList.view.xml`

```xml
<mvc:View
    controllerName="freeBoard.controller.PostList"
    xmlns:mvc="sap.ui.core.mvc"
    xmlns="sap.m"
    xmlns:f="sap.f"
    height="100%"
>
    <f:DynamicPage
        toggleHeaderOnTitleClick="true"
        headerExpanded="true"
    >
        <f:title>
            <f:DynamicPageTitle>
                <f:heading>
                    <Title text="자유 게시판"/>
                </f:heading>
            </f:DynamicPageTitle>
        </f:title>
        <f:header>
            <f:DynamicPageHeader pinnable="true">
                <f:content>
                    <SearchField
                        value="{viewModel>/searchText}"
                        search="onPressSearch"
                        placeholder="제목 또는 내용 검색"
                        width="100%"
                    />
                </f:content>
            </f:DynamicPageHeader>
        </f:header>
        <f:content>
            <List
                items="{viewModel>/postList}"
                noDataText="등록된 게시글이 없습니다."
                growing="true"
                growingThreshold="20"
                growingScrollToLoad="true"
            >
                <CustomListItem type="Active" press="onPressPostItem">
                    <VBox class="sapUiSmallMargin">
                        <HBox justifyContent="SpaceBetween">
                            <Title text="{viewModel>title}" level="H3"/>
                            <ObjectStatus
                                text="👁️ {viewModel>view_count}"
                                state="Information"
                            />
                        </HBox>
                        <HBox class="sapUiTinyMarginTop">
                            <Text text="작성자: {viewModel>author_user_id}"/>
                            <Text text=" | " class="sapUiTinyMarginBeginEnd"/>
                            <Text text="{
                                path: 'viewModel>createdAt',
                                type: 'sap.ui.model.odata.type.DateTime',
                                formatOptions: { pattern: 'yyyy-MM-dd HH:mm' }
                            }"/>
                            <Text text=" | 댓글 {viewModel>commentCount}개" class="sapUiTinyMarginBegin"/>
                        </HBox>
                    </VBox>
                </CustomListItem>
            </List>
        </f:content>
        <f:footer>
            <OverflowToolbar>
                <ToolbarSpacer/>
                <Button
                    text="글쓰기"
                    type="Emphasized"
                    icon="sap-icon://create"
                    press="onPressNewPost"
                />
            </OverflowToolbar>
        </f:footer>
    </f:DynamicPage>
</mvc:View>
```

### 3-4-7. `view/PostDetail.view.xml`

```xml
<mvc:View
    controllerName="freeBoard.controller.PostDetail"
    xmlns:mvc="sap.ui.core.mvc"
    xmlns="sap.m"
    xmlns:f="sap.f"
    xmlns:core="sap.ui.core"
    height="100%"
>
    <Page showNavButton="true" navButtonPress="onPressNavBack">
        <headerContent>
            <Button icon="sap-icon://edit" press="onPressEdit"
                visible="{= ${viewModel>/currentPost/author_user_id} === ${viewModel>/currentPost/author_user_id} && !${viewModel>/editMode}}"/>
            <Button icon="sap-icon://delete" press="onPressDelete"
                visible="{= !${viewModel>/editMode}}"/>
        </headerContent>
        <content>
            <ScrollContainer height="100%" horizontal="false" vertical="true">
                <VBox class="sapUiContentPadding">

                    <!-- 게시글 본문 -->
                    <Panel>
                        <VBox>
                            <!-- 제목 -->
                            <Title
                                text="{viewModel>/currentPost/title}"
                                visible="{= !${viewModel>/editMode}}"
                                level="H2"
                            />
                            <Input
                                value="{viewModel>/currentPost/title}"
                                visible="{viewModel>/editMode}"
                            />
                            <!-- 본문 -->
                            <Text
                                text="{viewModel>/currentPost/content}"
                                visible="{= !${viewModel>/editMode}}"
                                class="sapUiSmallMarginTop"
                            />
                            <TextArea
                                value="{viewModel>/currentPost/content}"
                                visible="{viewModel>/editMode}"
                                rows="10"
                                growing="true"
                                class="sapUiSmallMarginTop"
                            />
                            <!-- 메타 정보 -->
                            <HBox class="sapUiSmallMarginTop" justifyContent="SpaceBetween">
                                <Text text="작성자: {viewModel>/currentPost/author_user_id} | {
                                    path: 'viewModel>/currentPost/createdAt',
                                    type: 'sap.ui.model.odata.type.DateTime',
                                    formatOptions: { pattern: 'yyyy-MM-dd HH:mm' }
                                }"/>
                                <Text text="조회수: {viewModel>/currentPost/view_count}"/>
                            </HBox>
                            <!-- 수정 저장/취소 버튼 -->
                            <HBox visible="{viewModel>/editMode}" justifyContent="End" class="sapUiSmallMarginTop">
                                <Button text="취소" press="onPressEditCancel" type="Ghost"/>
                                <Button text="저장" press="onPressSave" type="Emphasized" class="sapUiTinyMarginBegin"/>
                            </HBox>
                        </VBox>
                    </Panel>

                    <!-- 댓글 영역 -->
                    <Panel headerText="{= '댓글 (' + ${viewModel>/comments}.length + ')'}"
                        class="sapUiMediumMarginTop"
                    >
                        <List
                            items="{viewModel>/comments}"
                            noDataText="아직 댓글이 없습니다."
                        >
                            <CustomListItem>
                                <VBox>
                                    <HBox justifyContent="SpaceBetween">
                                        <Text text="{viewModel>author_user_id}" class="sapUiTinyMarginEnd"/>
                                        <Text text="{
                                            path: 'viewModel>createdAt',
                                            type: 'sap.ui.model.odata.type.DateTime',
                                            formatOptions: { pattern: 'yyyy-MM-dd HH:mm' }
                                        }"/>
                                    </HBox>
                                    <Text text="{viewModel>content}" class="sapUiSmallMarginTop"/>
                                    <HBox justifyContent="End" class="sapUiTinyMarginTop">
                                        <Button
                                            icon="sap-icon://delete"
                                            press="onPressDeleteComment"
                                            type="Transparent"
                                            tooltip="삭제"
                                            visible="{= ${viewModel>/author_user_id} === ${viewModel>/currentPost/author_user_id}}"
                                        />
                                    </HBox>
                                </VBox>
                            </CustomListItem>
                        </List>
                    </Panel>

                    <!-- 댓글 작성 -->
                    <Panel headerText="댓글 작성" class="sapUiMediumMarginTop">
                        <TextArea
                            id="commentInput"
                            placeholder="댓글을 입력하세요..."
                            rows="3"
                            maxLength="2000"
                            width="100%"
                        />
                        <HBox justifyContent="End" class="sapUiSmallMarginTop">
                            <Button
                                text="댓글 등록"
                                type="Emphasized"
                                icon="sap-icon://add"
                                press="onPressAddComment"
                            />
                        </HBox>
                    </Panel>

                </VBox>
            </ScrollContainer>
        </content>
    </Page>
</mvc:View>
```

### 3-4-8. `view/PostCreate.view.xml` (프래그먼트 — Dialog)

```xml
<core:FragmentDefinition
    xmlns="sap.m"
    xmlns:core="sap.ui.core"
>
    <Dialog
        title="새 게시글 작성"
        contentWidth="600px"
        contentHeight="400px"
    >
        <content>
            <VBox class="sapUiContentPadding">
                <Label text="제목" required="true"/>
                <Input
                    id="newPostTitle"
                    placeholder="제목을 입력하세요"
                    maxLength="200"
                    width="100%"
                />
                <Label text="내용" required="true" class="sapUiMediumMarginTop"/>
                <TextArea
                    id="newPostContent"
                    placeholder="내용을 입력하세요"
                    maxLength="5000"
                    rows="10"
                    growing="true"
                    width="100%"
                    height="200px"
                />
            </VBox>
        </content>
        <beginButton>
            <Button text="취소" press="onPressCreateCancel" type="Ghost"/>
        </beginButton>
        <endButton>
            <Button text="등록" press="onPressCreatePost" type="Emphasized"/>
        </endButton>
    </Dialog>
</core:FragmentDefinition>
```

### 3-4-9. `index.html`

```html
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="X-UA-Compatible" content="IE=edge"/>
    <meta charset="utf-8"/>
    <title>ZEN-OTT 자유 게시판</title>
    <script
        id="sap-ui-bootstrap"
        src="https://openui5.hana.ondemand.com/resources/sap-ui-core.js"
        data-sap-ui-theme="sap_horizon"
        data-sap-ui-libs="sap.m"
        data-sap-ui-compatVersion="edge"
        data-sap-ui-async="true"
        data-sap-ui-frameOptions="trusted"
        data-sap-ui-onInit="module:sap/ui/core/ComponentSupport"
    ></script>
</head>
<body class="sapUiBody">
    <div data-sap-ui-component
        data-name="freeBoard"
        data-id="container"
        data-settings='{"id": "app"}'
    ></div>
</body>
</html>
```

---

# 4. 공통 파일 템플릿

모든 신규 모듈에 공통으로 필요한 파일:

### `package.json`

```json
{
    "name": "[moduleName]",
    "version": "1.0.0",
    "description": "ZEN-OTT [moduleName] UI Module",
    "scripts": {
        "start": "ui5 serve",
        "build": "ui5 build"
    }
}
```

### `ui5.yaml`

```yaml
specVersion: "2.6"
type: application
metadata:
  name: [moduleName]
```

---

# 5. 개발 순서

## 기능3 (TrendAnalysis)

1. `sysmgt/webapp/trendAnalysis/` 디렉토리 생성
2. `package.json`, `ui5.yaml`, `index.html` 작성 (템플릿 기반)
3. `manifest.json` 작성 (데이터소스 2개 등록)
4. `Component.js` 작성
5. `controller/BaseController.js` 작성
6. `controller/Dashboard.controller.js` 작성 (핵심 로직)
7. `view/Dashboard.view.xml` 작성 (UI)
8. `view/AnomalyList.view.xml` 프래그먼트 작성
9. `css/style.css` 작성
10. 테스트: `cds watch --port 8083` 구동 후 브라우저에서 확인

## 기능6 (FreeBoard)

1. `sysmgt/webapp/freeBoard/` 디렉토리 생성
2. 공통 파일 작성
3. `manifest.json` 작성 (라우팅 포함)
4. `Component.js` 작성
5. `controller/BaseController.js` (getCurrentUserId, isAuthor 추가)
6. `controller/PostList.controller.js` 작성
7. `controller/PostDetail.controller.js` 작성 (가장 복잡 — CRUD + 댓글)
8. `view/PostList.view.xml` 작성
9. `view/PostDetail.view.xml` 작성
10. `view/PostCreate.view.xml` 프래그먼트 작성
11. `css/style.css` 작성
12. 테스트

---

## 주의사항 체크리스트

- [ ] OData V2 경로는 `/srv-api/odata/v2/ott/...` (V4의 `/odata/v4/ott/...` 가 아님)
- [ ] `author_user_id`는 클라이언트에서 **절대 보내지 않음** (백엔드 자동 주입)
- [ ] Post DELETE는 소프트 삭제 → 목록에서 `deleted_flag eq false` 필터 필수
- [ ] 댓글 DELETE는 물리 삭제 → 삭제 확인 필요
- [ ] 본인 작성 확인은 `User` 모델의 `userId` 와 `author_user_id` 비교
- [ ] CAP V2에서 액션은 `oModel.callFunction("/actionName", { method: "POST", urlParameters: {...} })`
- [ ] `readOdata()`는 Promise 기반 → `.then().catch()` 체이닝
- [ ] `viewModel`은 모든 UI 상태를 JSONModel로 관리
- [ ] XML 뷰에서 `{viewModel>/path}` 로 바인딩
