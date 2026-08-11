# OTT approuter로 화면 띄우기 — 전체 과정

> 2026-07-06. BAS 로컬에서 approuter_ott(포트 5000) 하나로 런치패드 → 자유게시판/정산 화면이 뜰 때까지, 요청이 어떤 경로로 흐르는지 처음부터 끝까지. 디버깅이 아니라 작동 원리.

## 0. 한눈에 보는 그림

```
브라우저
  │  https://port5000-.../portal_ott/index.html
  ▼
approuter_ott (포트 5000)  ← 정적 파일 서빙 + OData 프록시
  │  xs-app.json 라우트 매칭
  ├── /portal_ott/   → localDir cap-app/portal_ott/webapp   (런치패드 파일 직접)
  ├── /board/        → localDir cap-app/board/webapp         (board 앱 파일 직접)
  ├── /settlement/   → localDir cap-app/settlement/webapp    (settlement 앱 파일 직접)
  ├── /common_ott.lib/ → localDir cap-app/library_ott/src    (공용 라이브러리)
  └── /srv-api/odata/v2/ott-core/...  → destination onOttCoreService (localhost:8084)
      /srv-api/odata/v2/ott/...       → destination onOttService    (localhost:8083)

core_ott (포트 8084)  ← 메뉴·사용자·권한 데이터 (CAP + sqlite/HANA)
ott     (포트 8083)  ← 게시판·정산 비즈니스 데이터 (CAP)
```

approuter가 **문지기**다. 브라우저가 요청하면 xs-app.json 규칙에 따라 (a) 내부 폴더에서 정적 파일을 직접 주거나, (b) 뒷단 백엔드(8084/8083)로 요청을 전달한다. 브라우저는 5000번만 본다.

## 1. 구성 요소

| 이름 | 포트 | 역할 |
|---|---|---|
| **approuter_ott** | 5000 | 유일한 입구. 정적 파일 서빙 + 백엔드 프록시 |
| **portal_ott** | (approuter가 서빙) | 런치패드. 메뉴를 불러 타일을 그리고, 타일 클릭 시 해당 앱을 띄움 |
| **board** | (approuter가 서빙) | 자유게시판 앱. 목록·상세·작성 |
| **settlement** | (approuter가 서빙) | 정산 관리 앱. 월별 정산 실행·조회 |
| **common_ott.lib** (library_ott) | (approuter가 서빙) | 공용 라이브러리. BaseController, i18n, 로더 |
| **core_ott** | 8084 | 메뉴·사용자·권한·공통코드. `ott-core` 서비스 |
| **ott** | 8083 | 게시판·정산 비즈니스. `ott` 서비스 |

portal_ott/board/settlement는 **독립된 앱**이 아니라 approuter가 폴더에서 직접 갖다 주는 UI5 앱이다. 백엔드는 core_ott(권한·메뉴)와 ott(비즈니스) 둘로 나뉜다.

## 2. 접속 첫 단계 — 런치패드 띄우기

브라우저가 `https://port5000-.../portal_ott/index.html` 치면:

1. approuter가 xs-app 라우트를 찾는다.
2. `^(.*)/portal_ott/(.*)$` 매칭 → `localDir: ../../portal_ott/webapp`, `target: $2`
   - `$2 = index.html` → 파일 `cap-app/portal_ott/webapp/index.html` 반환
3. 브라우저가 index.html 안의 `sap-ui-core.js`를 로드 → portal_ott의 `Component.js` 요청
   - `/portal_ott/Component.js` → 같은 라우트 → `portal_ott/webapp/Component.js` 반환
4. portal_ott Component가 부팅되며 **메뉴를 불러온다** → 다음 단계.

> 왜 `/index.html`(루트)은 502였나: 루트는 catch-all `^/(.*)$` → destination `ui`(포트 5001, ui5 serve)로 가는데, 5001을 안 띄웠기 때문. `/portal_ott/index.html`은 localDir에서 직접 주니까 5001이 필요 없다. **입구는 항상 `/portal_ott/index.html`**.

## 3. 메뉴 불러오기 — core_ott(8084)

portal_ott Component가 부팅되면 공용 라이브러리(common_ott.lib)의 `MenuLoader`가 동작한다:

1. OData 요청: `GET /srv-api/odata/v2/ott-core/MenusRoleAppliedList`
2. approuter 라우트 매칭: `^.*/srv-api/(.*ott-core/.*)$` → destination `onOttCoreService` (localhost:8084)
3. 8084(core_ott)가 "이 사용자에게 허용된 메뉴 목록"을 계산해서 반환.
   - 계산 로직: `user_id = $user` 조건으로 User → User_RoleGroup → RoleGroup → Role → Role_Menu → Menu 6단계 조인. 백엔드가 다 계산하고 프론트는 받기만 한다.
4. 받은 메뉴 목록으로 portal_ott가 타일을 그린다.

이때 메뉴 한 줄이 곧 타일 하나다. 메뉴 데이터의 핵심 필드:

| 필드 | 예시 | 쓰임 |
|---|---|---|
| `menu_code` | `OTT_BOARD` | 타일 식별자. 라우트 이름·Component usage 키로 쓰임 |
| `menu_app_id` | `board` | UI5 Component namespace. `sap.app.id`와 일치 |
| `menu_app_path` | `/board` | Component 파일을 어디서 받을지 URL 경로 |
| `menu_repo_path` | `board/webapp` | AppCacheBuster용. 배포 빌드 때 캐시 경로 |
| `menu_route_path` | (빈) | 앱 **내부** 라우트의 시작 경로. board는 빈 = 목록부터 |
| `menu_icon`, `menu_name` | feedback, 자유게시판 | 타일 표시 |

## 4. 타일 클릭 — board 앱 띄우기

사용자가 "자유게시판" 타일을 누르면, portal_ott Component.js가 **동적으로 라우트를 하나 만든다**:

```js
// menu_app_path 로더에 등록 → board 네임스페이스를 /board 경로로 해석
sap.ui.loader.config({ paths: { "board": "/board" } });

// 라우트 패턴: ${menu_app_id}/${menu_route_path} = "board/"
// 라우트 이름 = menu_code = "OTT_BOARD"
// Component usage 등록: 이름 board, MenuId 주입
componentUsages["OTT_BOARD"] = {
  name: "board",
  componentData: { routePath: "", MenuId: "OTT_BOARD", TileId: "board" },
  lazy: true
};
```

라우터가 `board/` 패턴으로 이동하면, 등록된 Component usage가 `board` Component를 **지연 로드**한다:

1. UI5가 `board/Component.js`를 요청 → `sap.ui.loader.config` 덕분에 `/board/Component.js`로 번역
2. approuter: `^(.*)/board/(.*)$` 매칭 → `localDir: ../../board/webapp`, `$2 = Component.js`
   - 파일 = `cap-app/board/webapp/Component.js` 반환 ✓
3. board Component가 로드되며 **내부 라우터를 초기화**.

### 핵심 정합 규칙 (여기가 제일 중요)

> **`menu_app_path`는 approuter `localDir`이 가리키는 폴더까지의 URL 경로여야 한다.**

board 앱 파일은 `cap-app/board/webapp/` 안에 있다. approuter 라우트의 `localDir`이 이미 `../../board/webapp`까지 가리키고 있다. 그러므로:
- `menu_app_path = /board` → 요청 `/board/Component.js` → `$2 = Component.js` → `board/webapp/` + `Component.js` ✓

만약 `menu_app_path = /board/webapp`로 주면:
- 요청 `/board/webapp/Component.js` → `$2 = webapp/Component.js` → `board/webapp/` + `webapp/Component.js` = `board/webapp/webapp/Component.js` → **404** (한 단계 중복)

기존 sysmgt_ott는 `sysmgt_ott/webapp/mainPage/webapp/`처럼 한 단계 더 깊어서 `/sysmgt_ott/mainPage/webapp`까지 써야 맞았다. board/settlement는 portal_ott처럼 한 단계 얕아서 `/board`에서 멈춰야 한다. **구조가 다르면 app_path도 다르다.**

## 5. board 내부 라우터 — 중첩 라우팅

board Component가 로드되면 자기 라우터를 켠다. 이때 **부모(portal_ott)의 라우트 prefix를 상속**받는다.

board manifest 안의 라우트:
- `MainList` — 패턴 `""` (빈 = 시작 화면, 목록)
- `DetailList` — 패턴 `"detail/{post_id}"` (상세)

부모 prefix가 `board/`이므로, 실제 URL은:
- 목록: `.../portal_ott/index.html#/board/`
- 상세: `.../portal_ott/index.html#/board/detail/OTT_POST_001`

board 안에서 `getRouter().navTo("DetailList", {post_id: "..."})` 하면 URL이 `#/board/detail/...`로 바뀌고, 같은 board Component 안에서 beginColumn(목록) → midColumn(상세)으로 화면이 전환된다. **페이지 이동이 아니라 같은 FCL 안에서 컬럼만 추가**되는 식. 라우터를 쓴다고 해서 새 Component가 로드되는 건 아니다.

## 6. 데이터 불러오기 — ott(8083)

board 화면이 뜨면 목록 데이터를 가져온다:

1. board manifest에 선언된 OData 모델: `/srv-api/odata/v2/ott/board/FreeBoard`
2. approuter 라우트 매칭: `ott-core`가 아니니 첫 라우트엔 안 걸리고, `^.*/srv-api/(.*)$` → destination `onOttService` (localhost:8083)
3. 8083(ott)이 게시글 목록을 반환 → board가 표에 바인딩.

상세 화면: `GET /srv-api/odata/v2/ott/board/FreeBoard('OTT_POST_001')?$expand=comments` → 같은 경로로 8083.

정산(settlement)도 같은 흐름. 다만 정산 액션(`runSettlement`)은 `POST /srv-api/odata/v2/ott/Admin/runSettlement` → 8083.

> **백엔드 분리 요점**: `/srv-api/.../ott-core/...`는 8084(메뉴·사용자·권한), `/srv-api/.../ott/...`는 8083(비즈니스). approuter가 URL 패턴으로 나눠준다. 프론트는 둘 다 `/srv-api/`로만 부르면 된다.

## 7. common_ott.lib — 공용 뼈대

board/settlement의 Controller는 `common_ott.lib.controller.BaseController`를 상속한다. 이 라이브러리는 approuter가 `/common_ott.lib/` → `cap-app/library_ott/src`에서 서빙한다.

BaseController가 onInit에서 자동으로 깔아주는 것:
- **UserSessionLoader** — `/srv-api/odata/v2/ott-core/UserManagement` → 본인 정보(User 모델)
- **MenuLoader** — 메뉴 목록
- **CommonCodeLoader** — 공통코드
- **MessageLoader** — 메시지
- **MenuFunctionLoader** — 메뉴별 기능 권한

그래서 board/settlement Controller는 User 모델을 별도 로드 안 해도 `this.getModel("User")`로 접근한다. `getComponentData().MenuId`(`OTT_BOARD`)로 "이 앱이 어떤 메뉴에서 왔는지"도 안다.

## 8. 전체 요청 흐름 요약

```
[1] 브라우저 → /portal_ott/index.html
    approuter → portal_ott/webapp/index.html (정적)

[2] portal_ott Component 부팅 → MenuLoader → /srv-api/.../ott-core/MenusRoleAppliedList
    approuter → 8084(core_ott) → 메뉴 목록 반환 → 타일 그림

[3] 타일 클릭 → portal_ott가 동적 라우트 "board/" 생성 + Component usage 등록
    라우터 이동 → board Component 지연 로드

[4] /board/Component.js 요청
    approuter → board/webapp/Component.js (정적, menu_app_path=/board 덕분)

[5] board Component 부팅 → 내부 라우터(부모 prefix board/ 상속) → MainList 화면
    + BaseController가 UserSessionLoader → 8084 (본인정보, 404여도 조용)

[6] board 목록 데이터 → /srv-api/.../ott/board/FreeBoard
    approuter → 8083(ott) → 게시글 목록 → 표 바인딩

[7] 행 클릭 → navTo("DetailList",{post_id}) → URL #/board/detail/...
    같은 board 안에서 상세 컬럼 추가 → /FreeBoard('...')?$expand=comments → 8083
```

## 9. 메뉴 시드 한 줄 요약

`Menu.csv` 한 행이 타일 하나. board/settlement 행의 핵심:
```
OTT_BOARD     ; board     ; board/webapp   ; /board        ; (빈 route)
               app_id      repo_path         app_path
OTT_SETTLEMENT; settlement; settlement/webapp; /settlement  ; (빈 route)
```
- `menu_app_id` = UI5 Component 이름
- `menu_app_path` = approuter localDir이 가리키는 폴더까지의 경로 (★ 정합의 핵심)
- `menu_repo_path` = AppCacheBuster용 (배포 빌드)
- `menu_route_path` = 앱 내부 시작 라우트

## 10. 로컬에서 띄울 때 실제 하는 일

BAS 터미널 3개 띄워 각각 실행:

### ① core_ott (포트 8084, 메뉴·권한)
```bash
cd cap-node/core_ott
npm run watch
```
- `cds watch --port 8084`
- 메뉴 목록(MenusRoleAppliedList), 사용자, 공통코드 담당
- sqlite in-memory + mocked auth(admin1 등)

### ② ott (포트 8083, 게시판·정산)
```bash
cd cap-node/ott
CDS_LOCAL_AUTH_BYPASS=1 npm run watch
```
- `CDS_LOCAL_AUTH_BYPASS=1` **필수** — admin1 + Admin 롤 강제 주입. 안 넣으면 정산 `@(requires:'Admin')`이 403
- 게시판(FreeBoard), 정산(SettlementService) 비즈니스 데이터

### ③ approuter_ott (포트 5000, 입구)
```bash
cd cap-app/approuter_ott
npm run start:local
```
- `--workingDir ./dev --port 5000` → **dev/xs-app.json** 사용 (배포용 루트 xs-app.json 아님)
- 로컬 destination: `dev/default-env.json` → onOttCoreService(8084), onOttService(8083)
- `listening on port: 5000` 뜨면 OK

### ④ 접속
```
https://port5000-workspaces-ws-mrnba.jp10.applicationstudio.cloud.sap/portal_ott/index.html
```
- **반드시 `/portal_ott/index.html`** (루트 `/index.html` 아님 — 5001 ui5 serve 안 띄워서 502)
- port preview로 5000 추적 후 이 경로로 진입

세 개(8084, 8083, 5000)가 다 떠 있어야 화면이 온전히 뜬다. 하나라도 빠지면 그 단계에서 멈춘다 — 8084 빠지면 메뉴가 안 뜨고, 8083 빠지면 타일은 눌리되 데이터가 안 오고, 5000 빠지면 접속 자체가 안 된다.


## 11. 검증 체크리스트

| 항목 | 확인 |
|---|---|
| 메뉴 뜸 | 런치패드에 자유게시판·정산 타일 보임 (8084 정상) |
| 타일 클릭 → 앱 로드 | Console에 `Component.js` 404 없어야 (menu_app_path 정합) |
| 목록 데이터 | 게시글/정산이력 표에 데이터 뜸 (8083 정상) |
| 읽기 | 무조건 됨 (8083 bypass) |
| **write(글쓰기·정산실행)** | Basic auth 헤더 제거 후 CSRF 막힐 수 있음 — CF 배포 전 확인 |

## 12. 자주 틀리는 것

- **루트 `/index.html` 502** → `/portal_ott/index.html` 쓸 것. 5001 안 띄워도 됨.
- **타일 무한 로딩 + `board/Component.js` 404** → `menu_app_path`가 `/board/webapp`이면 한 단계 중복. `/board`로 시드 수정.
- **정산 403** → 8083에 `CDS_LOCAL_AUTH_BYPASS=1` 안 넣음.
- **잘못된 xs-app.json 보는 중** → 로컬 기동은 `dev/xs-app.json` (배포용 루트 xs-app.json은 정규식 다름).

## 13. CF 배포 전 제거/복구 (dev 머지 전)

- board/settlement manifest `Basic auth` 헤더 → 이미 제거(cb2e40e)
- approuter_ott `dev/default-env.json` `onOttService.forwardAuthToken` → CF에선 true 복구 (로컬 bypass용 false). 단 default-env.json은 .gitignore라 커밋 안 됨 — CF 환경 destination 설정에서 처리.