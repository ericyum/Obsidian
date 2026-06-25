# ott 풀스택 분리 작업 — 최종 산출물

> **작성일**: 2026-06-23
> **담당**: 염정운
> **상태**: ✅ 완료
> **목적**: OTT 프로젝트 프론트엔드·백엔드 전체를 `_ott` 접미사 폴더로 완전 독립. 템플릿 원본(core, library, approuter, portal, sysmgt)과 완전 분리.

---

## 0. 최종 아키텍처 — _ott 풀스택

```
zen-ott-onboarding/
│
├── cap-node/
│   ├── core/              ← 원본 (손 안 댐) ✅
│   ├── ott/               ← OTT 비즈니스 백엔드 (기존)
│   └── core_ott/          ← [신규] OTT 공통 백엔드 ★
│
├── cap-app/
│   ├── approuter/         ← 원본 (손 안 댐) ✅
│   ├── portal/            ← 원본 (손 안 댐) ✅
│   ├── sysmgt/            ← 원본 (손 안 댐) ✅
│   ├── library/           ← 원본 (손 안 댐) ✅
│   │
│   ├── approuter_ott/     ← OTT 진입점 / 라우팅
│   ├── portal_ott/        ← OTT 포털 레이아웃
│   ├── library_ott/       ← [신규] OTT 공통 라이브러리 ★
│   └── sysmgt_ott/        ← OTT 기능 모듈 (detailPage, myPage)
│
└── Obsidian/OTT project/
    └── core_ott_백엔드_분리_작업_계획서.md  ← 본 문서
```

---

## 1. 요청 흐름 (최종)

```
브라우저
  → approuter_ott (인증 + 라우팅)
      │
      ├─ /srv-api/ott-core/* → core_ott (port 8084)  ★ 신규
      │   ├─ MenuManagement      → 메뉴, 즐겨찾기
      │   ├─ UserManagement      → 사용자 세션
      │   ├─ CodeManagement      → 공통 코드
      │   └─ MessageManagement   → 다국어 메시지
      │
      ├─ /srv-api/* → ott 백엔드 (port 8083)
      │   ├─ /odata/v4/detail    → DetailService (기능2)
      │   ├─ /odata/v4/main      → MainService (기능1)
      │   ├─ /odata/v4/ott/Mypage → MembershipService (기능4)
      │   ├─ /odata/v4/ott/Admin → SettlementService (기능5)
      │   └─ ...기능3,6 서비스들
      │
      ├─ /portal_ott/*    → portal_ott (포털 UI)
      ├─ /sysmgt_ott/*    → sysmgt_ott (기능 모듈)
      ├─ /common_ott.lib/* → library_ott (공통 라이브러리)
      └─ (.*)             → portal_ott (기본 진입점)
```

---

## 2. 신규 생성 모듈

### 2.1 `cap-node/core_ott/` (37개 파일)

공통 백엔드 서비스 — core에서 OTT에 필요한 부분만 추출.

| 구성 요소 | 파일 | 설명 |
|----------|------|------|
| 설정 | `.cdsignore`, `.cdsrc.json`, `package.json`, `mta.yaml`, `tsconfig.json` 등 8종 | 독립 CAP 프로젝트 |
| DB 모델 | `db/cds/` 14종 | Menu, User, Role, Code, Message 등 |
| 시드 데이터 | `db/data/` 5종 CSV | OTT 메뉴 3건 + 역할/그룹 |
| 서비스 | `srv/cds/` 4종 | MenuManagement, UserManagement, CodeManagement, MessageManagement |
| 핸들러 | `srv/server.ts`, `User.handler.ts` | TypeScript 로직 |

**네임스페이스**: `com.cap.ott.core`
**OData 경로**: `ott-core/MenuManagement`, `ott-core/UserManagement`, `ott-core/CodeManagement`, `ott-core/MessageManagement`
**로컬 포트**: 8084
**HDI 컨테이너**: `com-ott-core-hdi-container`

### 2.2 `cap-app/library_ott/` (42개 파일)

OTT 전용 공통 라이브러리 — library에서 복사 후 API 경로만 변경.

| 변경 파일 | 변경 내용 |
|----------|----------|
| `MenuLoader.js` | `core/MenuManagement` → `ott-core/MenuManagement` |
| `MenuFunctionLoader.js` | `core/MenuManagement` → `ott-core/MenuManagement` |
| `UserSessionLoader.js` | `core/UserManagement` → `ott-core/UserManagement` |
| `CommonCodeLoader.js` | `core/CodeManagement` → `ott-core/CodeManagement` |
| `MessageLoader.js` | `core/MessageManagement` → `ott-core/MessageManagement` |
| `FavoriteButton.js` | `core/MenuManagement` → `ott-core/MenuManagement` |
| `PersonalizationManager.js` | `core/Personalization` → `ott-core/Personalization` |

**UI5 네임스페이스**: `common_ott.lib` (기존 `common.lib`과 완전 분리)

---

## 3. CSV 시드 데이터

### `com.cap.ott.core-Menu.csv`

```
menu_code;parent_menu_code;menu_name;menu_app_id;menu_app_path;menu_route_path;menu_icon;menu_type_code;menu_level_number;menu_use_flag;menu_sort_number;menu_repo_path
OTT;;OTT 서비스;OTT;;;sap-icon://tv;Group;1;true;100;
OTT_DETAIL;OTT;컨텐츠 상세;detailPage;./sysmgt_ott/webapp/;Detail/{content_id};sap-icon://detail-view;Menu;2;true;110;sysmgt_ott/webapp
OTT_MYPAGE;OTT;마이페이지;myPage;./sysmgt_ott/webapp/;;sap-icon://customer;Menu;2;true;120;sysmgt_ott/webapp
```

### `com.cap.ott.core-Role.csv`
```
role_code;role_name;role_description;use_flag
OTT_USER;OTT 사용자;OTT 서비스 기본 사용자 역할;true
```

### `com.cap.ott.core-RoleGroup.csv`
```
role_group_code;role_group_name;role_group_description;use_flag
OTT_USER_GROUP;OTT 사용자 그룹;OTT 서비스 사용자 그룹;true
```

### `com.cap.ott.core-Role_Menu.csv`
```
role_code;menu_code;use_flag
OTT_USER;OTT;true
OTT_USER;OTT_DETAIL;true
OTT_USER;OTT_MYPAGE;true
```

### `com.cap.ott.core-Role_RoleGroup.csv`
```
role_code;role_group_code
OTT_USER;OTT_USER_GROUP
```

---

## 4. 수정 파일 총괄

### 4.1 portal_ott (7개 파일)

| 파일 | 변경 |
|------|------|
| `index.html` | resource root `common.lib` → `common_ott.lib`, libs, script src |
| `Component.js` | import `common/lib/` → `common_ott/lib/` |
| `BaseController.js` | import `common/lib/` → `common_ott/lib/` |
| `Layout.controller.js` | import `common/lib/` → `common_ott/lib/`, API 경로 `ott-core/` |
| `manifest.json` | model type, bundleName, resource js, dataSource URI |
| `ui5.yaml` | serveStatic `/common.lib` → `/common_ott.lib` |
| `Layout.view.xml` | image src `/common.lib/` → `/common_ott.lib/` |

### 4.2 sysmgt_ott (detailPage 4개 + myPage 4개 + 루트 2개 = 10개 파일)

| 파일 | 변경 |
|------|------|
| `detailPage/.../BaseController.js` | import `common/lib/` → `common_ott/lib/` |
| `detailPage/.../manifest.json` | model type, bundleName |
| `detailPage/ui5.yaml` | serveStatic 경로 |
| `myPage/.../BaseController.js` | import `common/lib/` → `common_ott/lib/` |
| `myPage/.../manifest.json` | model type, bundleName |
| `myPage/ui5.yaml` | serveStatic 경로 |
| `sysmgt_ott/ui5.yaml` | serveStatic 경로 |
| `sysmgt_ott/ui5-local.yaml` | serveStatic 경로 |

### 4.3 approuter_ott (5개 파일)

| 파일 | 변경 |
|------|------|
| `xs-app.json` | `ott-core` 라우트 추가 + catch-all → `onOttService` |
| `xs-app-dev.json` | `ott-core` 라우트 추가 + catch-all → `onOttService` |
| `xs-app-qa.json` | `ott-core` 라우트 추가 + catch-all → `onOttService` |
| `dev/xs-app.json` | `ott-core` 라우트, catch-all → `onOttService`, `common_ott.lib` localDir |
| `dev/default-env.json` | `onOttCoreService` → `localhost:8084` destination 추가 |

---

## 5. 원본 무변경 검증

| 원본 폴더 | OTT 흔적 | 상태 |
|----------|:--:|:--:|
| `cap-node/core/` | 0건 | ✅ |
| `cap-app/library/` | 0건 | ✅ |
| `cap-app/approuter/` | 0건 | ✅ |
| `cap-app/portal/` | 0건 | ✅ |
| `cap-app/sysmgt/` | 0건 | ✅ |

---

## 6. OTT 풀스택 완전성 검증

### 6.1 _ott 모듈 간 의존 관계

```
approuter_ott ─────────────────────────────────────────┐
  │ 라우팅: ott-core/* → core_ott                     │
  │ 라우팅: srv-api/* → ott                           │
  │ 서빙: portal_ott, sysmgt_ott, library_ott         │
  └───────────────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     portal_ott     sysmgt_ott     library_ott
     (포털 UI)     (기능 모듈)    (공통 유틸)
          │              │              │
          └──────────────┴──────────────┘
                         │
               import common_ott/lib/*
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
     core_ott (ott-core/*)         ott (srv-api/*)
     공통 백엔드 서비스           비즈니스 백엔드
```

### 6.2 비-_ott 모듈 참조 여부

| 검사 항목 | 결과 |
|----------|:--:|
| portal_ott의 API 호출 경로 | 전부 `ott-core/` ✅ |
| library_ott의 API 호출 경로 | 전부 `ott-core/` (7/7) ✅ |
| sysmgt_ott의 API 호출 경로 | OTT 비즈니스 서비스만 ✅ |
| approuter_ott의 destination | `onOttCoreService` + `onOttService`만 ✅ |
| core_ott CDS의 namespace 참조 | `com.cap.ott.core`만 사용 ✅ |
| Home.controller.js test code | `core/FormService`, `core/Attachment` (dead code) ⚠️ |

> ⚠️ `Home.controller.js`의 `core/FormService`, `core/Attachment` 참조는 템플릿 테스트 코드로, OTT 기능과 무관. 필요 시 추후 제거 가능.

---

## 7. 작업 이력

| # | 작업 | 파일 수 |
|---|------|:--:|
| 1 | `core_ott/` 폴더 생성 + 설정 파일 복사 | 8 |
| 2 | `db/cds/` 14종 모델 복사 + namespace 변경 | 14 |
| 3 | `db/data/` 5종 CSV 시드 데이터 작성 | 5 |
| 4 | `srv/cds/` 4종 서비스 복사 + namespace/경로 변경 | 4 |
| 5 | `srv/server.ts`, `User.handler.ts` 복사 + namespace 변경 | 2 |
| 6 | `db/index.cds`, `srv/index.cds` 작성 | 2 |
| 7 | `library_ott/` 생성 (library 복사) | 42 |
| 8 | `library_ott/` 7개 파일 `core/` → `ott-core/` 변경 | 7 |
| 9 | `portal_ott` 7개 파일 `common.lib` → `common_ott.lib` 변경 | 7 |
| 10 | `sysmgt_ott` 10개 파일 `common.lib` → `common_ott.lib` 변경 | 10 |
| 11 | `approuter_ott` 5개 파일 라우팅/destination 수정 | 5 |
| 12 | 원본 롤백 (library 7개 파일) | 7 |
| **총계** | | **113** |

---

## 8. 남은 작업

- [ ] `core_ott`에 `npm install` 실행
- [ ] `core_ott` + `ott` 동시 기동 테스트 (`cds watch`)
- [ ] approuter_ott 통해 E2E 테스트
- [ ] MTA 배포 설정 — `core_ott`를 `mta.yaml`에 모듈로 추가
- [ ] `Home.controller.js` dead code 정리 (선택)
