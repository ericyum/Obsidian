# meeting-whisper 프로젝트 구조

## 전체 조감도

이 레포는 **1개 MTA(Multi-Target Application)로 배포되는 4개 런타임 모듈 + 폐기된 legacy**로 구성됩니다. 핵심 원리는 **"하나의 레포, 여러 런타임"**입니다 — 브라우저(JS), CAP(Node.js), Worker(Python), Approuter가 각자 다른 buildpack으로 배포되지만 소스는 한곳에서 관리됩니다.

```
meeting-whisper/
├── 📁 배포 설정 (루트)
├── 📁 approuter/       ← ① 진입점·인증
├── 📁 app/meeting-ui/  ← ② 브라우저 UI (approuter가 서빙)
├── 📁 cap/             ← ③ 백엔드 두뇌 (데이터·권한·API·요약)
├── 📁 workers/         ← ④ 전사 전용 Python 워커
├── 📁 scripts/         운영 도구
├── 📁 deploy/ docs/    배포 매니페스트·아키텍처 문서
├── 📁 tests/           테스트 (Python + JS 혼재)
└── 📁 legacy/fastapi/  폐기된 로컬 PoC
```

배포 시 요청 흐름은 이 순서입니다:

```
브라우저 → ① approuter → ② UI(정적) / ③ CAP(/api) → ④ Worker → AI Core
```

---

## 루트 — 배포·설정의 중심

|파일|역할|
|---|---|
|`mta.yaml`|**가장 중요.** 4개 모듈 + 서비스 바인딩 + 모든 운영 환경변수 정의|
|`xs-security.json`|XSUAA scope/role 정의 (`MeetingUser`/`Worker`/`MeetingAdmin`)|
|`server.js`|CAP 부트스트랩 커스터마이징 — 바이너리 업로드 라우트 장착 + 정적 파일 서빙|
|`package.json`|npm 스크립트(빌드·배포·로컬 실행), CAP mock 사용자 정의|
|`.env.example` / `requirements.txt` / `pytest.ini`|환경변수 템플릿, Python 의존성, 테스트 설정|
|`meeting_whisper.code-workspace`|VS Code 멀티루트 워크스페이스|

---

## ① approuter/ — 진입점과 인증

사용자가 실제로 접속하는 유일한 주소입니다. **코드가 아니라 설정**만 있습니다.

```
approuter/
├── xs-app.json          라우팅 규칙 (핵심)
├── package.json         @sap/approuter 의존성
└── dev/                 로컬 개발용 설정
```

`xs-app.json`이 URL을 목적지로 라우팅합니다:

|경로 패턴|목적지|용도|
|---|---|---|
|`/user-api/*`|approuter 내장|현재 사용자 정보|
|`/api/*`|`srv-api` (CAP)|공개 API|
|`/api-binary/*`|`srv-api` (CAP)|대용량 오디오 업로드|
|`/internal/*`|`srv-api` (CAP)|Worker 콜백 (XSUAA 인증)|
|`/*`|`resources/` 로컬|정적 UI|

**모든 경로가 `authenticationType: xsuaa`**입니다. 즉 로그인 없이는 UI 파일 한 장도 못 받습니다.

---

## ② app/meeting-ui/ — 브라우저 UI

**빌드 도구·프레임워크 없는 순수 ES 모듈 SPA**입니다. 빌드 시 `copy-web-assets.js`가 approuter의 `resources/`로 복사됩니다.

```
app/meeting-ui/
├── index.html           단일 페이지 골격
├── styles.css           전체 스타일 (327줄)
├── site.webmanifest / favicon.svg   PWA 메타
├── vendor/pretendard/   한글 폰트 (번들된 유일한 외부 자산)
└── js/
    ├── app.js       2,973줄 — 전체 오케스트레이션·라우팅·뷰·녹음
    ├── api.js       623줄 — CAP API 호출 래퍼
    ├── ui.js        248줄 — DOM 렌더링 헬퍼
    ├── recorder.js  102줄 — MediaRecorder 래퍼
    ├── session.js   233줄 — 세션 만료 경고·연장·keep-alive
    └── drafts.js    102줄 — IndexedDB 녹음 draft 복구
```

**모듈 간 역할 분담**:

- `app.js`가 사실상 전체 컨트롤러 — 목록/생성/상세 3개 뷰, 라우팅, 실시간 전사, 참여자 칩 UI를 모두 담습니다.
- `api.js`는 서버 통신을 격리 — `useCapApi()` 분기로 CAP과 legacy FastAPI 양쪽을 지원합니다(`api.js:427`).
- `session.js`·`drafts.js`는 신뢰성 기능(세션·크래시 복구)을 별도 분리.

---

## ③ cap/ — 백엔드 두뇌

가장 밀도 높은 영역입니다. **데이터 모델 + 2개 서비스 + 라이브러리**로 나뉩니다.

```
cap/
├── db/
│   ├── schema.cds        7개 엔티티 데이터 모델
│   └── package.json      HDI deployer 설정
└── srv/
    ├── meeting-service.cds/.js       공개 API (@requires MeetingUser)
    ├── worker-internal-service.cds/.js  Worker 콜백 API (@requires Worker)
    ├── ai-usage-service.cds          AI 사용량 조회 서비스    
    └── lib/                          비즈니스 로직 (여기가 핵심)
```

### 서비스 계층 — 두 개의 분리된 API

|서비스|경로|권한|누가 호출|대표 액션|
|---|---|---|---|---|
|`MeetingService`|`/api`|`MeetingUser`|브라우저|`enqueueTranscription`, `saveSummary`, `runReview`, `deleteMeeting`|
|`WorkerInternalService`|`/internal`|`Worker`|Python Worker|`claimTranscription`, `getAudio`, `saveTranscript`, `heartbeatTranscription`|

**설계 의도**: 사용자용과 Worker용 API를 물리적으로 분리하고 role도 다릅니다. Worker는 `getAudio`/`saveTranscript` 같은 내부 동작만 가능하고, 사용자는 이 경로에 접근할 role이 없습니다.

### lib/ — 로직 라이브러리 (12개 모듈)

책임별로 잘 쪼개져 있습니다:

```
lib/
├── ai-core.js               1,031줄 ★ 요약·검토·AI Core 호출·JSON 검증/복구
├── meeting-note.js          transcript/note 검증·정규화·Markdown 변환
├── transcription-dispatcher.js  DB 기반 전사 큐 + Worker dispatch
├── cleanup.js               60초 주기 장애 복구 (stale 정리)
├── audio-store.js           오디오 저장·권한·Worker 전달 payload
├── object-store.js          S3 호환 Object Store (signed URL)
├── audio.js                 오디오 메타 유틸
├── binary-upload-route.js   대용량 바이너리 업로드 Express 라우트
├── ai-usage-log.js          AiCoreUsageLogs 기록
├── worker-client.js         CAP → Worker HTTP 호출
└── status.js                상태 상수(STATUS enum)
```

### 3계층 구조

lib/ 내부 의존성을 따라가면 **명확한 3계층**으로 나뉩니다.

#### 계층 0 — 순수 유틸 (의존성 없음, 잎사귀 노드)

로컬 `require`가 하나도 없는 기반 모듈들입니다. 누구에게도 의존하지 않아 순환 위험이 없습니다.

|모듈|노출|성격|
|---|---|---|
|`status.js`|`STATUS`|상태 문자열 상수. **가장 많이 참조됨** (5곳)|
|`audio.js`|`isExpired`, `toBase64Content`, `normalizeAudioContent`,   <br>`getMaxAudioBytes`, `getAudioExpiresAt`|오디오 인코딩·만료 계산|
|`object-store.js`|`putAudioObject`, `getAudioObjectUrl`, `deleteAudioObject`,   <br>`createAudioObjectKey`, `isObjectStoreEnabled`|S3 호환 스토리지 (AWS SDK 래퍼)|
|`meeting-note.js`|`normalizeNote`, `noteToMarkdown`, `parseTranscriptContent`,   <br>`validateTranscriptContent`, `transcriptToChunks`, `fallbackSummary`|transcript/note 검증·정규화·chunking|
|`ai-usage-log.js`|`logAiCoreUsage`, `extractAiCoreUsage/ModelName/RequestId`|AI 호출 로그 기록·파싱|
|`worker-client.js`|`enqueueTranscription`, `workerToken`|CAP→Worker HTTP (env만 읽음)|

#### 계층 1 — 조합 모듈 (계층 0을 묶음)

|모듈|의존하는 계층 0|역할|
|---|---|---|
|`ai-core.js` ★|`meeting-note` + `ai-usage-log`|요약·검토 전체|
|`audio-store.js`|`audio` + `object-store` + `status`|오디오 저장·권한·Worker payload|
|`transcription-dispatcher.js`|`status`|DB 큐 엔진|
|`binary-upload-route.js`|`audio-store`|대용량 업로드 Express 라우트|

**`ai-core.js`의 의존 방향이 핵심입니다** (`ai-core.js:1-11`):

- `meeting-note`에서 `transcriptToChunks`(12,000자 분할), `normalizeNote`(스키마 정규화), `fallbackSummary`(LLM 없을 때 대체)를 가져옵니다.
- `ai-usage-log`에서 응답 파싱·기록 함수를 가져옵니다.
- 즉 ai-core는 **"AI 호출 오케스트레이션"에 집중**하고, transcript 다루기와 usage 기록은 계층 0에 위임합니다. 1,031줄이지만 책임 경계는 지켜져 있습니다.

#### 계층 2 — 특수 조합: cleanup.js

`cleanup.js`만 **계층 1 모듈(transcription-dispatcher)에 의존**하는 유일한 케이스입니다 (`cleanup.js:1-3`):

```
const { deleteObjectStoreRows } = require("./audio-store");        // 계층1
const { STATUS } = require("./status");                            // 계층0
const { DISPATCH_STATUS } = require("./transcription-dispatcher"); // 계층1
```

`DISPATCH_STATUS`를 가져와 stale dispatch를 정리하고, `audio-store`로 만료 오디오를 삭제합니다. **읽기 방향이 한 방향(cleanup → dispatcher)**이라 순환이 없습니다.

### 핵심 관찰

#### 1. 순환 의존성이 전혀 없다

require 방향이 **계층 2 → 1 → 0 단방향**으로 깔끔합니다. 이것이 가능한 이유는 두 가지 상수 모듈을 분리했기 때문입니다:

- **`status.js`** — `STATUS`를 별도 파일로 뺐기에, 여러 모듈이 상태 상수를 공유해도 서로를 import할 필요가 없습니다.
- **`transcription-dispatcher.js`가 `DISPATCH_STATUS`를 export** — cleanup이 dispatcher의 상수만 필요로 하고 로직은 안 건드립니다.

만약 `DISPATCH_STATUS`가 cleanup 안에 정의됐다면 dispatcher↔cleanup 순환이 생겼을 것입니다. 이 분리가 의도적 설계입니다.

#### 2. 두 서비스의 의존 패턴이 대칭적

||meeting-service (사용자)|worker-internal-service (Worker)|
|---|---|---|
|공통|`ai-core`, `audio-store`, `meeting-note`, `status`|동일|
|고유|`cleanup`, `worker-client`, `transcription-dispatcher` (큐 시작·dispatch)|`ai-usage-log`, `audio`(base64 변환)|

**두 서비스 모두 `summarizeMeeting`을 호출**합니다 — meeting-service는 사용자의 `runSummary` 액션에서, worker-internal은 transcript 저장 직후 백그라운드 요약에서. 요약 로직이 ai-core 한곳에 있어 중복이 없습니다.

#### 3. 가장 많이 참조되는 모듈 = 안정성의 축

- **`status.js`** (5곳): status, audio-store, cleanup, dispatcher, 두 서비스 — 가장 변경에 민감. STATUS enum 값 변경 시 전체 파급.
- **`audio-store.js`** (3곳): 두 서비스 + cleanup + binary-upload — 오디오 저장 방식 변경의 중심.
- **`meeting-note.js`** (3곳): ai-core + 두 서비스 — transcript 스키마 변경의 중심.

이 3개가 사실상 **변경 파급이 가장 큰 핫스팟**입니다. 리팩터링 시 여기부터 테스트를 확보해야 합니다.

#### 4. 외부 결합 지점

lib 내부는 깨끗하지만, 외부 세계와 닿는 3개 모듈이 실질적 위험 지점입니다:

- `object-store.js` → AWS SDK / Object Store 자격증명
- `worker-client.js` → Worker HTTP (타임아웃·429 처리)
- `ai-core.js` → AI Core 인증·토큰 캐싱·네트워크

이들은 모두 **env 기반으로 우아한 비활성화**(URL/자격증명 없으면 fallback)를 구현해, 로컬 개발에서 외부 의존 없이 동작하도록 설계되어 있습니다.

---

## ④ workers/transcription/ — 전사 전용 Python 워커

CAP과 완전히 독립된 FastAPI 앱. **STT만** 담당합니다.

```
workers/transcription/
├── Dockerfile           (Kyma 시절 잔재, 현재 CF는 python_buildpack)
├── requirements.txt     Python 의존성
├── README.md
└── app/
    ├── main.py          689줄 ★ FastAPI 엔드포인트·동시성·heartbeat·콜백    
    ├── aicore_stt.py    1,140줄 ★ AI Core STT·chunk 분할·4중 검증·병합    
    ├── transcriber.py   299줄 — 엔진 분기 (aicore/whisperx/fast)    
    ├── cap_client.py    195줄 — Worker → CAP 인증·호출    
    └── config.py        110줄 — 환경변수 설정(Pydantic)
```

**포인트**: `transcriber.py:19-46`는 `TRANSCRIPTION_ENGINE`으로 3개 엔진(`aicore`/`whisperx`/`fast`)을 분기합니다. 운영은 `aicore`만 쓰지만, whisperX/faster-whisper 경로가 코드에 남아 있습니다 — ADR-001에서 "fallback/reference로 남긴다"고 한 부분입니다.

---

## 보조 영역

### scripts/ — 운영 도구 (19개)

- **JS 운영**: `ops-retry-transcription.js`(강제 재전사), `export-remote-ai-usage.js`(사용량 export), `ops-mark-meeting-failed.js`, `rerun-summary-from-transcript.js`
- **PowerShell smoke**: `cap-smoke.ps1`, `cap-worker-smoke.ps1`, `cap-browser-e2e.ps1`
- **Kyma 잔재**: `kyma-worker-deploy.ps1`, `kyma-worker-image.ps1` (미사용)
- **마이그레이션**: `migrate-legacy-sqlite-to-hana.js`

### deploy/ + docs/

```
deploy/
├── OPERATIONS.md        운영 절차
├── cf/README.md         Cloud Foundry 배포 (현재)
└── kyma/                Kyma 매니페스트 9개 (폐기, 참조용)

docs/transcription-architecture-migration.md   아키텍처 전환 요약
```

**주의**: `deploy/kyma/`는 폐기된 경로입니다. Vault Runbook이 "Kyma 문서를 운영 절차로 쓰지 말라"고 경고하는 대상입니다.

### tests/ — 혼재 구조

```
Python 26개: 21개 → legacy/fastapi | 5개 → workers (현재)
JS 3개: ai-core-summary / transcript-validation / transcription-dispatcher
```

`pytest.ini`의 `pythonpath=legacy/fastapi` 때문에 `npm test`는 주로 폐기 코드를 검증하고, JS 3개는 실행 스크립트가 없습니다.

### legacy/fastapi/ — 폐기된 원형

2026-06 로컬 PoC(FastAPI + SQLite + WhisperX). 현재 운영 경로가 아니지만 초기 파이프라인 설계(diarize, speakers, summarize, review)를 담고 있어 참조 가치는 있습니다.