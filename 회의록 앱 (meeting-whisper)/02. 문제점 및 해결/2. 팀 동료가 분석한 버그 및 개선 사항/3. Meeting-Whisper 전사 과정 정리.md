  
전사는 **두 개의 완전히 분리된 경로**로 이루어집니다. Vault가 강조하는 핵심은 "화면에 보이는 실시간 전사"와 "실제 저장되는 최종 전사"가 서로 다른 파이프라인이라는 점입니다.

---

## A. 브라우저 실시간 임시 전사 (버려지는 경로)

> Vault: _"SpeechRecognition → 실시간 임시 전사 → 화면 미리보기와 draft에만 저장 → 서버·최종 전사·요약에 전달하지 않음"_

**코드**: `app.js:1561` `startLiveTranscript()`

- 브라우저 내장 `webkitSpeechRecognition`을 `ko-KR`, `continuous`, `interimResults`로 실행합니다 (`app.js:1570-1573`).
- 결과는 `liveFinalText`/`liveInterimText` 변수에 쌓여 화면에만 그려지고(`app.js:1609-1610`), `saveLiveFinalToDraft()`가 이를 **브라우저 로컬 draft**(`currentDraft.liveText`)에만 저장합니다 (`app.js:1632`).

**"createJob 요청에 포함되지 않는다"의 코드 증거** — 이게 가장 중요합니다:

`api.js:425`의 `createJob()` 시그니처를 보면 인자에 `blob`(오디오)·`title`·`participants`·`duration`은 있지만 **`liveFinalText`를 받는 자리가 아예 없습니다.** 실제 서버 요청 본문(`api.js:428-441`)도:

```
body: JSON.stringify({ 
  title, category, participants, sharedWith, 
  durationSec: Math.round(duration || 0),   // ← 임시 전사 텍스트 없음  
  minSpeakers: null, maxSpeakers: null, diarize,
})
```

즉 사용자가 실시간 전사 화면을 아무리 고쳐도 서버로 갈 통로 자체가 없습니다. 마이크는 **동시에 두 곳**으로 들어갑니다 — `MediaRecorder`(`recorder.js:17`)는 오디오 Blob을 만들고, `SpeechRecognition`은 화면 미리보기를 만듭니다. **최종 전사에 쓰이는 건 오직 앞의 Blob**입니다.

---

## B. 최종 전사 파이프라인 (실제로 저장되는 경로)

> Vault: _"최종 전사·요약의 입력은 업로드된 원본 오디오와, 그 오디오에서 생성한 AI Core STT 전사본이다"_

`createJob`의 실제 3단계 (`api.js:442-445`):

```
const meetingId = meeting.ID;
await uploadCapAudio(meetingId, blob, ext);   // 1) 오디오 업로드
await postAction("enqueueTranscription", { meetingId });  // 2) 큐 등록
```

### 1단계 — 큐 등록 (CAP)

**코드**: `meeting-service.js:353` `queueMeetingTranscription()`

- 오디오가 업로드됐는지 확인(`AudioObjects`) → 없으면 409 (`:355-356`).
- 기존 `Transcripts`/`Summaries`/`ReviewItems`를 삭제하고 상태를 `queued`(10%)로 초기화 (`:358-365`).
- **여기서 Vault의 `durationSec → expectedDurationSec` 전달이 일어납니다** (`:366-373`):
    
    ```
    await createTranscriptionDispatch(cds, {  meetingId,  expectedDurationSec: meeting.durationSec,  // ← 핵심  ...});
    ```
    

### 2단계 — Dispatch 큐 → Worker 호출 (CAP)

**코드**: `transcription-dispatcher.js`

- `createTranscriptionDispatch`가 `TranscriptionDispatches` row를 `queued`로 INSERT (`:57`).
- 5초 폴링 루프(`startTranscriptionDispatcher`)가 due한 row를 집어 `dispatchOne`으로 Worker에 POST.
- **Vault "payload에 값 없으면 Meeting 재조회"의 코드**(`:130-135`) — 오래된 큐 payload에 `expectedDurationSec`가 없으면 그 자리에서 Meeting을 다시 SELECT해 채웁니다. 긴 파일 오판 방지의 2차 안전장치입니다.
- Worker가 바쁘면(`429`/`WORKER_BUSY`) row를 다시 `queued`로 되돌리고 Meeting을 `queued`로 유지 (`:161-178`) — Vault의 "Worker busy 시 queued 유지"와 일치.

### 3단계 — Worker 접수와 동시성 제한

**코드**: `main.py:102` `POST /transcribe`

- `WORKER_CONCURRENCY`(=2)만큼의 `asyncio.Semaphore` slot (`main.py:37-44`). slot이 다 차면 **429 반환** (`:122-126`) → 위의 CAP busy 처리로 연결됩니다.
- slot 확보 후 `claim_transcription`으로 이 실행을 등록(중복 방어)하고, 실제 전사는 `background_tasks`로 넘긴 뒤 즉시 `202 accepted` 응답 (`:127-148`).

### 4단계 — 오디오 다운로드 → 길이 판단 → chunk → AI Core STT

**코드**: `main.py:184` `_process_transcription` → `aicore_stt.py:44` `run_aicore_transcription`

Worker는 Object Store signed URL로 오디오를 임시 파일에 다운로드(`main.py:195`, `_download_temp_audio`)한 뒤 STT를 돌립니다. Vault 항목별 코드 위치:

|Vault 규칙|코드|
|---|---|
|**"측정 또는 예상 길이가 720초 넘으면 분할"**|`aicore_stt.py:743` `_should_chunk_audio` — `_duration_hint(측정, 예상)`의 **max**가 `chunk_seconds`(720) 초과 시 True (`:756-757`). "파일 크기 무관"이지만 inline 바이트 초과 시에도 분할(`:753-755`).|
|WAV chunk 생성|`aicore_stt.py:760` `_split_audio_chunks` — PyAV로 16kHz mono s16 WAV로 리샘플해 720초 단위 분할|
|STT 호출|`aicore_stt.py:87` `_run_single_aicore_transcription` — AI Core completion endpoint에 오디오를 base64 file로 첨부 (`:362-393`)|
|**"잘못된 STT 응답은 같은 chunk 최대 2회 요청"**|`aicore_stt.py:102` `for attempt in range(1, max_attempts+1)` — `AICORE_STT_MAX_ATTEMPTS`(=2). **재시도 시 이전 검증 오류를 프롬프트에 넣어 교정 요청**(`:354-360`)|
|**"계속 실패한 12분 chunk는 약 6분 단위로 소분할"**|`aicore_stt.py:223` `_transcribe_chunk_with_fallback` — `fallback_seconds = max(60, chunk_duration/2)`(`:242`). 720초/2 = 360초(약 6분). 60초 이하면 더 안 쪼개고 실패 (`:243-244`)|

### 5단계 — 병합·검증 (Worker 측)

> Vault: _"JSON 구조, 발화 text, timestamp 순서·범위와 raw JSON 중첩 검증"_ / _"검증 실패를 하나의 raw text segment로 저장하지 않는다"_

이게 이 프로젝트에서 **가장 방어적으로 짜인 부분**이고, Vault가 강조하는 이유입니다. 검증이 **4중**으로 걸려 있습니다:

**① STT 응답 파싱 직후** — `aicore_stt.py:284` `content_to_transcript`  → `_normalize_segments`(`:571`):

- 완전한 JSON object인지 / `segments` 배열인지 (`:294-299`)
- timestamp가 유한한 0 이상 숫자인지(`:706` `_timestamp_value`), `end >= start`인지(`:587`), 역순이 아닌지(`:591`)
- 오디오 길이 초과 검사(`:605-625`)
- **"raw JSON 중첩 차단"** — segment 하나가 통째로 transcript JSON을 담고 있으면 거부 (`:635-636` `_looks_like_raw_transcript_json`)

**② chunk 병합 시** — `aicore_stt.py:923` `_merge_chunk_transcripts`: 각 chunk의 timestamp에 `chunk.start_sec` offset을 더해 전체 타임라인 복원, 병합 후에도 monotonic·raw JSON 재검증 (`:947-952`).

**③ callback 직전** — `main.py:237` `_validate_transcript_for_storage`: 저장 전 마지막 게이트. useful segment가 0이면 거부(`:275-276`).

검증이 최종 실패하면 `TranscriptionResponseError`가 올라가 `main.py:225-227`에서 `_mark_failed`로 이어지고 — **잘못된 결과를 raw text로 저장하지 않고 meeting을 실패 처리**합니다. (원본 오디오는 재전사를 위해 유지)

### 6단계 — CAP saveTranscript 재검증 (Vault 마지막 단계)

> Vault: _"CAP saveTranscript 재검증"_

Worker가 검증을 통과한 transcript를 CAP internal로 콜백하면 **CAP이 한 번 더 검증**합니다.

**코드**: `worker-internal-service.js:248` `saveTranscript`:

- `assertCurrentWorkerRun`으로 현재 run인지 확인(오래된/다른 run 콜백 거부 → 중복 방어) (`:252`)
- **④ `validateTranscriptContent(content, { durationSec })`** (`:259`) — `meeting-note.js:39`의 JS판 검증(segments 배열, timestamp, raw JSON 중첩 `:112`)이 Python 검증을 그대로 재현. 실패 시 **422로 거부** (`:260-262`).
- 통과하면 `Transcripts` 저장 → Meeting을 `summarizing`(80%)으로 → `startWorkerSummaryTask`로 **백그라운드 요약 시작** (`:264-280`). 여기서 전사 저장과 요약의 HTTP 수명이 분리됩니다(ADR-001의 핵심 이점).

---

## 전체 그림 (코드 기준)

```
[A] SpeechRecognition (app.js:1561) → 화면·로컬 draft only ─────╳ 서버로 안 감

[B] MediaRecorder Blob (recorder.js)
→ createJob (api.js:425) ── POST /api/Meetings + uploadCapAudio + enqueueTranscription
 → queueMeetingTranscription (meeting-service.js:353) durationSec→expectedDurationSec
  → createTranscriptionDispatch → TranscriptionDispatches(queued)
   → dispatchOne (transcription-dispatcher.js:109) [429면 재큐]
    → Worker POST /transcribe (main.py:102) [semaphore 2슬롯, 초과 429]
     → claim → download → run_aicore_transcription (aicore_stt.py:44)
      → _should_chunk_audio(>720s) → split → AI Core STT (2회 재시도)
       → 실패 chunk는 ~360s 소분할 fallback
      → 검증① normalize → 검증② merge
     → 검증③ _validate_transcript_for_storage (main.py:237)
    → cap.save_transcript 콜백
  → 검증④ validateTranscriptContent (worker-internal-service.js:259) [실패=422]
   → Transcripts 저장 → summarizing(80%) → 백그라운드 요약
```