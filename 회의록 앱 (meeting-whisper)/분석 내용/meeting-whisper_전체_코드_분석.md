# Meeting Whisper — 전체 코드 분석

> **분석일**: 2026-07-20
> **기준 소스**: `meeting_whisper` `0182231` (`main`)
> **분석 범위**: 전체 프로젝트 코드 구조, 데이터 모델, API, 프론트엔드, Worker, 배포

---

## 1. 프로젝트 개요

**Meeting Whisper**는 브라우저 녹음 또는 음성 파일을 받아 SAP AI Core로 전사·요약하고, 권한이 적용된 회의록을 편집·공유하는 SAP BTP 내부 서비스입니다.

### 현재 운영 아키텍처

```text
Browser (Vanilla JS)
  → CF Approuter / XSUAA
  → CAP srv (Node.js)
     → HANA Cloud (DB)
     → Object Store (임시 오디오 저장)
     → TranscriptionDispatches (queue)
  → CF STT Worker (Python/FastAPI)
     → SAP AI Core STT (gemini-2.5-flash)
  → CAP internal saveTranscript
     → background summary/review
     → SAP AI Core Orchestration (claude-4.7-opus)
```

### 기술 스택

| 계층 | 기술 |
|------|------|
| **프론트엔드** | Vanilla JS + HTML/CSS (SPA, UI5 아님) |
| **백엔드** | SAP CAP (Node.js) + HANA Cloud |
| **Worker** | Python + FastAPI |
| **인증** | XSUAA + Approuter |
| **전사** | SAP AI Core `gemini-2.5-flash` |
| **요약** | SAP AI Core Orchestration `claude-4.7-opus` |
| **저장소** | Object Store (오디오 임시) + HANA (메타데이터) |
| **배포** | Cloud Foundry (MTA) |

---

## 2. 전체 폴더 구조

```
meeting-whisper/
├── mta.yaml                          # ★ MTA 배포 정의 (전체 오케스트레이션)
├── package.json                      # 루트 npm 스크립트
├── server.js                         # CAP 서버 진입점 (binary upload, static)
├── xs-security.json                  # XSUAA 인증 설정
├── .env.example                      # 환경변수 예시
│
├── app/
│   └── meeting-ui/                   # ★ 프론트엔드 (Vanilla JS SPA)
│       ├── index.html                # 진입점 HTML (모든 view 포함)
│       ├── styles.css                # 전체 CSS
│       ├── favicon.svg
│       ├── site.webmanifest
│       ├── vendor/                   # 외부 폰트
│       │   └── pretendard/
│       └── js/                       # JavaScript 모듈
│           ├── app.js                # 앱 진입점, 라우팅, 이벤트 바인딩
│           ├── api.js                # CAP API 통신 레이어
│           ├── ui.js                 # UI 렌더링 (목록, 상세, 편집)
│           ├── recorder.js           # 브라우저 녹음 (MediaRecorder)
│           ├── drafts.js             # 로컬 드래프트 저장/복구
│           └── session.js            # 세션 관리 (만료 경고, keep-alive)
│
├── approuter/                        # Approuter (Node.js)
│   ├── package.json
│   ├── xs-app.json                   # BTP 라우팅 규칙
│   ├── dev/
│   │   ├── xs-app.json               # 로컬 개발용 라우팅
│   │   └── default-env.example.json  # 로컬 Destination 예시
│   └── .cfignore
│
├── cap/                              # CAP 백엔드
│   ├── db/
│   │   ├── schema.cds                # ★ 데이터 모델 (8개 엔티티)
│   │   └── package.json
│   └── srv/
│       ├── meeting-service.cds       # ★ 공개 API (MeetingService)
│       ├── meeting-service.js        # API 구현 (1000+ 라인)
│       ├── worker-internal-service.cds # ★ Worker 전용 Internal API
│       ├── worker-internal-service.js  # Internal API 구현
│       ├── ai-usage-service.cds      # AI 사용량 조회 서비스
│       └── lib/                      # 핵심 라이브러리
│           ├── ai-core.js            # AI Core Orchestration (요약·검토)
│           ├── ai-usage-log.js       # AI 사용량 로깅
│           ├── audio-store.js        # 오디오 저장·권한 관리
│           ├── audio.js              # 오디오 메타데이터 추출
│           ├── binary-upload-route.js # 바이너리 업로드 라우트
│           ├── cleanup.js            # 만료 오디오 정리
│           ├── meeting-note.js       # Transcript/Note 검증·변환·정규화
│           ├── object-store.js       # Object Store 연동
│           ├── status.js             # Meeting 상태 상수
│           ├── transcription-dispatcher.js # ★ Queue/Dispatch 관리
│           └── worker-client.js      # Worker HTTP 클라이언트
│
├── workers/
│   └── transcription/                # ★ STT Worker (Python)
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── README.md
│       └── app/
│           ├── __init__.py
│           ├── main.py               # ★ FastAPI 서버 + Job orchestration
│           ├── config.py             # 설정 (Pydantic Settings)
│           ├── cap_client.py         # CAP HTTP 클라이언트
│           ├── transcriber.py        # 전사 엔진 진입점
│           └── aicore_stt.py         # AI Core STT 호출·chunk·검증
│
├── deploy/                           # 배포 참고자료
│   ├── OPERATIONS.md
│   ├── cf/README.md
│   └── kyma/                         # Kyma (현재 미사용)
│
├── scripts/                          # 운영·테스트 스크립트
│   ├── export-remote-ai-usage.js     # AI usage 로컬 내보내기
│   ├── ops-retry-transcription.js    # 운영자 강제 재전사
│   ├── ops-list-meeting-metrics.js   # 회의별 메트릭 조회
│   ├── ops-mark-meeting-failed.js    # 회의 실패 처리
│   ├── ops-restore-audio-object.js   # 오디오 복구
│   ├── rerun-summary-from-transcript.js # transcript → summary 재실행
│   ├── backfill_diarization.py       # 화자분리 소급 적용
│   ├── copy-web-assets.js            # UI 에셋 복사
│   ├── cap-smoke.ps1                 # CAP smoke 테스트
│   ├── cap-worker-smoke.ps1          # Worker smoke 테스트
│   └── reprocess_job.py              # Job 재처리
│
└── tests/                            # 테스트
    ├── test_worker_*.py              # Worker API/전사 테스트
    ├── test_*.py                     # CAP API/모델 테스트
    ├── ai-core-summary.test.js       # AI Core 요약 테스트
    ├── transcript-validation.test.js # 전사 검증 테스트
    ├── transcription-dispatcher.test.js # Dispatch 테스트
    └── cap-browser-e2e.js            # E2E 브라우저 테스트
```

---

## 3. 데이터 모델 (`cap/db/schema.cds`)

### 3.1 엔티티 관계도

```
Meetings (중심)
├── audio: Composition of one AudioObjects
├── transcript: Composition of one Transcripts
├── summary: Composition of one Summaries
├── reviewItems: Composition of many ReviewItems
└── (연결됨) TranscriptionDispatches.meeting → Meetings

AiCoreUsageLogs (독립 - meetingId로 연결)
```

### 3.2 Meetings

```cds
entity Meetings : cuid, managed {
  owner          : String(255);       // 생성자 ID (XSUAA user_id)
  ownerName      : String(160);       // 생성자 표시명
  title          : String(160);       // 회의 제목
  category       : String(80);        // 회의 카테고리
  participants   : LargeString;       // 참여자 JSON 배열
  sharedWith     : LargeString;       // 공유자 JSON 배열
  accessTokens   : LargeString;       // 접근 토큰 (owner+participants+sharedWith)
  status         : JobStatus;         // created → audio_uploaded → queued → transcribing → summarizing → done/failed
  percent        : Integer;           // 진행률 (0-100)
  durationSec    : Integer;           // 오디오 길이 (초)
  processingSec  : Decimal(9,1);      // 전체 처리 시간
  summarySec     : Decimal(9,1);      // 요약 생성 시간
  minSpeakers    : Integer;           // 최소 화자 수
  maxSpeakers    : Integer;           // 최대 화자 수
  diarize        : Boolean;           // 화자분리 여부
  whisperModelSize : String(40);      // 전사 모델 크기
  error          : LargeString;       // 오류 메시지
}
```

### 3.3 상태 흐름 (`JobStatus`)

```
created → audio_uploaded → queued → transcribing → summarizing → done
                                                          ↘ failed
```

### 3.4 TranscriptionDispatches

Worker에 전사 작업을 전달하는 Queue/Dispatch 관리 엔티티:

| 필드 | 설명 |
|------|------|
| `status` | `queued` → `dispatching` → `dispatched` → `failed` |
| `attempts` | 시도 횟수 |
| `maxAttempts` | 최대 시도 (기본 5) |
| `nextAttemptAt` | 다음 시도 시각 (exponential backoff) |
| `lockedAt` | Worker가 작업을 가져간 시각 |
| `workerRunId` | Worker 실행 ID (중복 방지) |
| `workerStage` | Worker 현재 단계 |
| `workerHeartbeatAt` | 마지막 heartbeat 시각 |
| `workerMessage` | Worker 진행 메시지 |
| `payload` | 전사 요청 페이로드 (JSON) |
| `lastError` | 마지막 오류 메시지 |

### 3.5 Transcripts, Summaries, ReviewItems

| 엔티티 | 저장 데이터 |
|--------|------------|
| **Transcripts** | 전사 JSON (segments 배열), 언어 |
| **Summaries** | 요약 JSON + Markdown |
| **ReviewItems** | 검토 후보 (이름·용어·날짜·어색표현) |
| **AudioObjects** | 오디오 메타데이터 + content / objectKey |
| **AiCoreUsageLogs** | AI Core 호출별 토큰·모델·지연시간 |

---

## 4. CAP 백엔드

### 4.1 서비스 구조

```
MeetingService        @path: '/api'       @requires: 'MeetingUser'
WorkerInternalService  @path: '/internal'  @requires: 'Worker'
```

### 4.2 MeetingService — 공개 API

| Action | 용도 | 권한 |
|--------|------|------|
| `uploadAudio(meetingId, mimeType, size, content)` | 오디오 업로드 | MeetingUser |
| `enqueueTranscription(meetingId)` | 전사 대기열 등록 | MeetingUser |
| `retryMeeting(meetingId)` | 실패 회의 재시도 | MeetingUser |
| `getMeetingStatus(meetingId)` | 상태·진행률·queue position 조회 | MeetingUser |
| `runSummary(meetingId)` | 요약 수동 재실행 | MeetingUser |
| `runReview(meetingId)` | 검토 후보 재생성 | MeetingUser |
| `saveTranscript(meetingId, content)` | 전사본 저장 (사용자 편집) | MeetingUser |
| `saveSummary(meetingId, content)` | 요약본 저장 | MeetingUser |
| `updateMeetingTitle(meetingId, title)` | 제목 수정 | MeetingUser |
| `updateMeetingCategory(meetingId, category)` | 카테고리 수정 | MeetingUser |
| `updateMeetingAccess(meetingId, participants, sharedWith)` | 접근 권한 수정 | MeetingUser |
| `deleteMeeting(meetingId)` | 회의 삭제 | MeetingUser (생성자만) |
| `deleteAudio(meetingId)` | 오디오 삭제 | MeetingUser |
| `cleanupExpiredAudio()` | 만료 오디오 정리 | **MeetingAdmin** |

### 4.3 WorkerInternalService — Worker 전용 Internal API

| Action | 용도 |
|--------|------|
| `claimTranscription(meetingId, workerRunId)` | 작업 할당 요청 |
| `heartbeatTranscription(meetingId, workerRunId, stage, message, percent)` | 진행 상황 보고 |
| `getAudio(meetingId)` | 오디오 다운로드 (signed URL 또는 base64) |
| `saveTranscript(meetingId, workerRunId, language, content)` | 전사 결과 저장 |
| `logAiUsage(...)` | AI Core 사용량 기록 |
| `failTranscription(meetingId, workerRunId, error)` | 실패 보고 |

### 4.4 핵심 라이브러리 (`cap/srv/lib/`)

| 파일 | 역할 | 핵심 기능 |
|------|------|----------|
| **ai-core.js** | AI Core LLM 호출 | 요약(summarizeMeeting), 검토(reviewMeeting), AI usage 로깅 |
| **transcription-dispatcher.js** | Queue/Dispatch 관리 | `createTranscriptionDispatch()`, polling loop, backoff, stale 감지 |
| **worker-client.js** | Worker HTTP 통신 | `enqueueTranscription()`, dispatch 요청 전송 |
| **meeting-note.js** | Transcript/Note 처리 | 검증, 정규화, JSON↔Markdown 변환 |
| **audio-store.js** | 오디오 CRUD + 권한 | 업로드, Object Store 연동, 접근 제어 |
| **object-store.js** | Object Store 연동 | signed URL 생성, 업로드/다운로드 |
| **cleanup.js** | 만료 오디오 정리 | 주기적 정리 작업 |
| **status.js** | Meeting 상태 상수 | STATUS enum |

### 4.5 요약/검토 파이프라인

```
transcript 저장 완료
  ↓
summarizeMeeting() (background task)
  ├─ 긴 transcript → 12,000자 chunk 분할
  ├─ chunk별 partial summary 생성 (AI Core Orchestration template)
  ├─ 16,000자 단위 reduce 결합
  └─ 최종 meeting note 1개 저장 (JSON + Markdown)
  ↓
reviewMeeting()
  ├─ 저장된 summary를 대상으로
  ├─ AI Core direct orchestration (claude-4.7-opus)
  └─ ReviewItems 생성 (이름·용어·날짜·표현)
```

### 4.6 Dispatcher 동작 방식

```
startTranscriptionDispatcher()
  └─ setInterval(5초) polling
      └─ processTranscriptionDispatchQueue()
          ├─ SELECT queued rows (nextAttemptAt ≤ now)
          ├─ limit 2개 처리
          └─ dispatchOne()
              ├─ UPDATE status → 'dispatching'
              ├─ POST /transcribe to Worker
              ├─ Worker busy(429) → 다시 queued, backoff
              ├─ 성공 → status = 'dispatched'
              └─ 실패 → backoff 또는 maxAttempts 초과 시 failed
```

---

## 5. STT Worker (`workers/transcription/`)

### 5.1 Worker 개요

- **타입**: Python FastAPI 앱
- **배포**: CF Python Buildpack
- **동시성**: 2개 job 동시 처리 (`WORKER_CONCURRENCY=2`)
- **전사 방식**: SAP AI Core `gemini-2.5-flash`

### 5.2 API 엔드포인트

| Endpoint | Method | 설명 |
|----------|--------|------|
| `/health` | GET | 헬스 체크 |
| `/transcribe` | POST | 전사 요청 (202 Accepted → background task) |

### 5.3 `/transcribe` 처리 흐름

```
POST /transcribe (meetingId, workerRunId, expectedDurationSec)
  ↓
① 인증: TRANSCRIPTION_WORKER_TOKEN 검증
  ↓
② 동시성 체크: Semaphore로 최대 2개 제한
  - full → 429 (Worker is busy)
  ↓
③ CAP claimTranscription() 호출
  ↓
④ Background task 실행:
  ├─ Stage: downloading (25%)
  │   └─ CAP getAudio() → signed URL 또는 base64
  ├─ Stage: transcribing (30% → 74%)
  │   ├─ 오디오 분할 판단 (720초 초과 → chunk)
  │   ├─ AI Core STT 호출 (gemini-2.5-flash)
  │   │   ├─ 실패 → 같은 chunk 2회 재시도
  │   │   └─ 계속 실패 → 6분 단위 소분할
  │   ├─ Progress callback → heartbeat (약 30초 간격)
  │   └─ chunk 병합 (timestamp offset)
  ├─ Stage: saving (75%)
  │   ├─ Transcript 검증 (JSON·segments·timestamp)
  │   └─ CAP saveTranscript() 호출
  └─ 완료 → Semaphore 해제
```

### 5.4 주요 모듈

| 파일 | 역할 |
|------|------|
| **main.py** | FastAPI 서버, job orchestration, 검증 |
| **config.py** | Pydantic Settings (환경변수) |
| **cap_client.py** | CAP HTTP 클라이언트 (claim, heartbeat, save, fail) |
| **transcriber.py** | 전사 엔진 진입점 (run_transcription) |
| **aicore_stt.py** | AI Core STT 호출, chunk 분할, 검증, 재시도 |

---

## 6. 프론트엔드 (`app/meeting-ui/`)

### 6.1 기술 스택

- **Vanilla JS** (UI5 아님, 순수 JavaScript SPA)
- CSS 직접 작성
- Pretendard 폰트
- 브라우저 API: `MediaRecorder`, `SpeechRecognition`, `localStorage`

### 6.2 View 구조 (SPA)

```html
index.html (모든 view가 하나의 HTML에)
├── #view-list      목록 화면 (검색·필터·카드 목록)
├── #view-entry     새 회의록 작성 (카테고리·제목·참여자·파일)
├── #view-recording 녹음 화면 (타이머·파형·실시간 임시 전사)
└── #view-result    결과 화면 (요약 탭·전사 탭·다운로드)
```

### 6.3 JS 모듈 역할

| 파일 | 역할 | 핵심 기능 |
|------|------|----------|
| **app.js** | 앱 진입점 | View 라우팅, 이벤트 바인딩, 회의 생성/조회 플로우 |
| **api.js** | API 통신 | CAP action 호출, 세션 터치, 401 감지, polling |
| **ui.js** | UI 렌더링 | 목록 카드 렌더링, 상세 화면(요약/전사 탭), 편집·수정 |
| **recorder.js** | 녹음 | MediaRecorder, 타이머, 파형, 임시 전사(SpeechRecognition) |
| **drafts.js** | 로컬 저장 | localStorage 기반 드래프트 저장·복구 |
| **session.js** | 세션 관리 | 만료 경고(10분 전), 수동 연장, 녹음 중 keep-alive(30분) |

### 6.4 권한 모델 (프론트엔드)

| 사용자 | 열람 | 편집 | 접근관리 | 삭제 |
|--------|:---:|:---:|:---:|:---:|
| 생성자 | ✅ | ✅ | ✅ | ✅ |
| 참여자 | ✅ | ✅ | ❌ | ❌ |
| 공유자 | ✅ | ❌ | ❌ | ❌ |

---

## 7. 배포 구조 (`mta.yaml`)

### 7.1 MTA 모듈 구성

```
meeting-whisper (MTA)
├── meeting-whisper-srv          ← CAP Node.js 서버 (512M/1G)
├── meeting-whisper-stt-worker   ← Python STT Worker (2G/2G)
├── meeting-whisper-db-deployer  ← HANA HDI Deployer
└── meeting-whisper-approuter    ← Approuter (256M/512M)
```

### 7.2 BTP 서비스

| 서비스 | 타입 | 용도 |
|--------|------|------|
| `zen-meeting-whisper` | HANA HDI | 데이터베이스 |
| `meeting-whisper-auth` | XSUAA | 사용자·Worker 인증 |
| `tf-objectstore` | Object Store (existing) | 오디오 write |
| `objectstoreRead` | Object Store (existing) | 오디오 read |
| `default_aicore` | AI Core (existing) | STT + LLM |

### 7.3 모듈 간 연결 (MTA Dependency Wiring)

```
meeting-whisper-srv
  ├── provides: srv-api → srv-url: ${default-url}
  ├── provides: (to worker) TRANSCRIPTION_WORKER_URL ← worker-api.worker-url

meeting-whisper-stt-worker
  ├── provides: worker-api → worker-url: ${default-url}
  └── → CAP srv로 callback (XSUAA + TRANSCRIPTION_WORKER_TOKEN)

meeting-whisper-approuter
  ├── requires: srv-api → destination "srv-api"
  └── → 모든 트래픽을 CAP srv로 프록시
```

### 7.4 주요 환경변수

| 변수 | 값 | 설명 |
|------|-----|------|
| `TRANSCRIPTION_ENGINE` | `aicore` | 전사 엔진 |
| `AICORE_STT_MODEL` | `gemini-2.5-flash` | STT 모델 |
| `AICORE_STT_CHUNK_SECONDS` | `720` | 12분 단위 chunk |
| `AICORE_STT_MAX_ATTEMPTS` | `2` | chunk 재시도 |
| `WORKER_CONCURRENCY` | `2` | 동시 처리 job |
| `TRANSCRIPTION_DISPATCH_LIMIT` | `2` | Dispatch 동시 처리 |
| `CAP_LLM_MODEL` | `anthropic--claude-4.7-opus` | 검토용 LLM |
| `CAP_LLM_SUMMARY_CHUNK_CHARS` | `12000` | 요약 chunk 크기 |
| `SESSION_TIMEOUT` | `180` | 세션 만료 (분) |
| `MAX_AUDIO_BYTES` | `157286400` | 최대 업로드 (≈150MiB) |

---

## 8. 인증 구조

### 8.1 Role Collection

```
xs-security.json
├── MeetingUser  ← 일반 사용자 (회의 생성·조회·편집)
├── MeetingAdmin ← 운영자 (cleanup, admin actions)
└── Worker       ← STT Worker (service-to-service)
```

### 8.2 인증 흐름

```
브라우저 → Approuter (XSUAA 로그인) → CAP srv
CAP → Worker: TRANSCRIPTION_WORKER_TOKEN (shared bearer)
Worker → CAP: XSUAA binding + Worker scope
```

---

## 9. 데이터 흐름 (End-to-End)

### 9.1 회의 생성부터 완료까지

```
① 사용자: 카테고리·제목·참여자 입력 + 녹음/파일 업로드
② 브라우저: POST /api/uploadAudio (binary)
③ CAP: AudioObjects 저장 (DB 또는 Object Store)
④ Meeting status → audio_uploaded
⑤ CAP: createTranscriptionDispatch() → queued
⑥ CAP Dispatcher: polling → Worker POST /transcribe
⑦ Worker: claim → download audio → AI Core STT → 검증 → saveTranscript
⑧ CAP: saveTranscript() → Transcripts 저장 → background task
⑨ CAP background: summarizeMeeting() → AI Core Orchestration template
⑩ CAP: Summaries 저장 (JSON + Markdown)
⑪ CAP: reviewMeeting() → AI Core direct orchestration → ReviewItems 저장
⑫ Meeting status → done, percent → 100
⑬ CAP cleanup: 오디오 정리
```

### 9.2 장시간 파일 처리

```
오디오 > 720초 (12분)
  ↓
Worker에서 chunk 분할
  ├─ 12분 단위 WAV chunk
  ├─ 각 chunk → AI Core STT (2회 재시도)
  ├─ 계속 실패 → 6분 소분할
  └─ timestamp offset 병합

긴 transcript (> 12,000자)
  ↓
CAP에서 요약 chunk 분할
  ├─ 12,000자 ordered chunk
  ├─ chunk별 partial summary
  ├─ 16,000자 단위 reduce 결합
  └─ 최종 summary 1개
```

---

## 10. 핵심 설계 결정

| 결정 | 이유 |
|------|------|
| **Kyma → CF Worker 전환** | GPU 없는 CPU Whisper 성능 불안정, base64 메모리 부담 |
| **Object Store 도입** | 큰 오디오 파일을 DB에 직접 저장하지 않고 signed URL로 전달 |
| **전사·요약 분리** | Worker는 transcript만 저장, CAP이 background로 summary 생성 |
| **Dispatch Queue** | Worker busy 시 429 반환, CAP이 backoff로 재시도 |
| **Semaphore 동시성** | Worker instance 1개에서 2개 job만 동시 처리 |
| **Template vs Direct** | 요약은 template(moдель 고정), 검토는 direct(claude-4.7-opus) |
| **Vanilla JS** | UI5 대신 순수 JS로 빠른 개발·가벼운 번들 |
| **session.js** | Approuter 세션 180분 + 브라우저 localStorage 기반 만료 경고 |

---

## 11. 현재 제약사항

| 제약 | 상세 |
|------|------|
| **자동 화자분리** | `DIARIZE=false`, 비활성화 |
| **배포 환경** | CF `dev` space 단일 (staging/prod 분리 안 됨) |
| **최대 업로드** | 150MiB, ALAC M4A 등 초과 파일은 별도 압축 필요 |
| **과거 transcript** | Raw JSON 형태 저장된 과거 데이터 자동 교정 안 함 |
| **Outlook 내보내기** | `mailto:` 기반, URL 길이 제한 |
| **장시간 관측** | 60~110분 파일의 처리시간·비용·chunk 경계 품질 관측 중 |

---

## 12. Kyma 관련 기록

- Kyma CPU Whisper Worker는 2026-07-02에 Cloud Foundry + AI Core로 전환됨
- `deploy/kyma/`에 Kyma manifest가 보존되어 있으나 현재 운영 경로 아님
- `legacy/fastapi/`에 초기 로컬 FastAPI 프로토타입 존재
- 자세한 전환 배경은 [ADR-001](../meeting-whisper-vault/02_기술_설계/ADR-001_CF_AI_Core_STT_전환.md) 참조

---

> **관련 Vault**: `meeting-whisper-vault/`
> **코드 레포**: `meeting_whisper` `0182231`
