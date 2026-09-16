# OTT approuter 메뉴 경로 정합 — BAS 로컬 화면 띄우기 디버깅

> 2026-07-06. BAS에서 approuter_ott(5000)로 portal_ott → board/settlement 타일 진입 시 무한 로딩이 떴던 원인과 해결. 핵심은 `menu_app_path` 시드 값과 approuter `localDir`의 정합.

## 1. 증상 흐름

1. `https://port5000-.../index.html` → **502 bad gateway**
2. `/portal_ott/index.html`로 직접 진입 → 런치패드 메뉴 정상 노출(타일 보임)
3. 자유게시판·정산 타일 클릭 → **무한 로딩**
4. Console 핵심:
    
    ```
    /board/webapp/Component.js  → 404
    failed to load 'board/Component.js' from /board/webapp/Component.js: script load error
    The following error occurred while displaying routing target with name 'OTT_BOARD'
    ```
    
    나머지(AppCacheBuster 404, i18n key 누락, list bindings assertion)는 전부 부수 현상.

## 2. 502의 원인 (1단계)

approuter_ott `dev/xs-app.json`:

- `welcomeFile: "/index.html"`
- catch-all `^/(.*)$` → destination `ui` (localhost:5001, `ui5 serve`)

`/index.html`이 catch-all을 타서 5001로 가는데, 5001을 안 띄워서 502.  
**해결**: 5001을 굳이 띄우지 않고 `/portal_ott/index.html`로 직접 진입. 이 경로는 localDir(`../../portal_ott/webapp`)에서 approuter가 직접 서빙하므로 5001 불필요.

## 3. 무한 로딩의 원인 (2단계) — menu_app_path 정합

### 구조 비교

|앱|실제 Component.js 경로|approuter localDir|시드 menu_app_path|
|---|---|---|---|
|sysmgt_ott/mainPage (기존, 정상)|`sysmgt_ott/webapp/mainPage/webapp/Component.js`|`../../sysmgt_ott/webapp`|`/sysmgt_ott/mainPage/webapp`|
|board (신규, 404)|`board/webapp/Component.js`|`../../board/webapp`|`/board/webapp` ← 틀림|
|settlement (신규, 404)|`settlement/webapp/Component.js`|`../../settlement/webapp`|`/settlement/webapp` ← 틀림|
|portal_ott (참조, 정상)|`portal_ott/webapp/Component.js`|`../../portal_ott/webapp`|(메뉴 아님, 직접 접속)|

### 정합 원리

approuter xs-app 라우트:

```json
{ "source": "^(.*)/board/(.*)$", "target": "$2", "localDir": "../../board/webapp" }
```

- URL `/board/<X>` → `$2 = <X>` → `localDir + <X>` 파일 서빙.

portal_ott가 메뉴 로드 후 Component usage를 띄울 때:

```js
sap.ui.loader.config({ paths: { [menu_app_id]: menu_app_path } });
// board → paths.board = "/board/webapp"
```

이후 Component 로드 URL = `menu_app_path + "/Component.js"`.

**board (틀린 경우)**:

- menu_app_path = `/board/webapp`
- 로드 URL = `/board/webapp/Component.js`
- approuter: `$2 = webapp/Component.js`
- 파일 = `board/webapp/` + `webapp/Component.js` = **`board/webapp/webapp/Component.js`** → 404

`localDir`이 이미 `board/webapp`까지 가리키는데, `menu_app_path`가 또 `/board/webapp`를 붙여서 한 단계가 중복.

**sysmgt_ott/mainPage (맞는 경우)**:

- menu_app_path = `/sysmgt_ott/mainPage/webapp`
- 로드 URL = `/sysmgt_ott/mainPage/webapp/Component.js`
- approuter: `$2 = mainPage/webapp/Component.js`
- 파일 = `sysmgt_ott/webapp/` + `mainPage/webapp/Component.js` = **`sysmgt_ott/webapp/mainPage/webapp/Component.js`** → ✓

sysmgt_ott는 `webapp/` 밑에 `mainPage/webapp/`가 또 있어서(한 단계 더 깊음) 경로가 맞았음.

### 핵심 규칙

> **menu_app_path의 첫 세그먼트(/X)는 approuter source의 `(.*)/X/(.*)`와 매칭되고, 그 뒤 경로는 localDir 디렉토리 내부의 상대 경로여야 한다.**

board/settlement는 portal_ott와 같은 단순 구조(`X/webapp/Component.js`)라, localDir이 `../../X/webapp`일 때 app_path는 `/X`에서 멈춰야 한다. 기존 sysmgt_ott처럼 `/X/하위/...`까지 달면 안 됨 — board엔 하위 단계가 없으니까.

## 4. 해결

시드 `Menu.csv` 두 행 수정 (menu_app_path만, menu_repo_path는 AppCacheBuster용이라 유지):

```
OTT_BOARD     ; ... ; /board/webapp  → /board
OTT_SETTLEMENT; ... ; /settlement/webapp → /settlement
```

커밋 `451b042` (feature/hsan). pull → 8084(core_ott) 재기동 → 메뉴 재로드 → 브라우저 새로고침 → 타일 클릭 시 Component.js 정상 로드.

## 5. 다음 단계로 넘긴 것

Component.js 로드가 풀리면 board 내부 OData(`/srv-api/odata/v2/ott/board/FreeBoard`)로 진입. 이건 `^.*/srv-api/(.*)$` → onOttService(ott 8083)로 가므로 **8083(ott)이 같이 떠 있어야** 한다.

## 6. 비밀 점검 (커밋 전)

- `default-env.json`(destination URL + UAA secret) — `.gitignore **/default-env.json` 제외 ✓
- `core_ott/srv/server.ts` — 추적되지만 secret/bypass 값 없음 ✓
- `core_ott/.cdsrc.json` — `credentials:{url:":memory:"}` 뿐, 시크릿 아님 ✓
- `xs-security.json` — scope 정의, 원래 공개용 ✓
- board/settlement manifest `Basic YWRtaW4xOg==`(admin1:빈비번) — 커밋됨, 로컬 mock 자격증명이라 실비밀 아님. **main 머지 전 제거 예정**.

## 7. 한 줄 요약

board/settlement는 portal_ott형 단순 구조라 `menu_app_path`를 localDir 접두사까지만(`/board`) 주면 끝 — `webapp` 한 단계가 중복되어 Component.js를 못 찾았던 것.