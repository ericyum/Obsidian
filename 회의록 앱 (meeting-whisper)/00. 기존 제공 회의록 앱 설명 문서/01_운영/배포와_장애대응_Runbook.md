# 배포와 장애대응 Runbook

최종 갱신: 2026-07-20

기준 소스: `meeting_whisper` `0182231`

현재 Cloud Foundry 운영 경로의 배포, 권한, secret rotation, 장애 확인과 rollback 기준이다. 실제 secret·token·service key JSON은 문서와 Git에 남기지 않는다.

## 현재 런타임

```text
CF API: https://api.cf.jp10.hana.ondemand.com
Org/Space: org-build-tf-joule-1on6yytw / dev

Apps:
  meeting-whisper-approuter
  meeting-whisper-srv
  meeting-whisper-stt-worker

Services:
  zen-meeting-whisper       HANA HDI
  meeting-whisper-auth      XSUAA
  objectstore               write binding -> tf-objectstore
  objectstoreRead           read binding
  meeting-whisper-aicore    -> default_aicore
```

주요 운영값:

| 항목 | 값 |
| --- | --- |
| STT engine | `aicore` |
| STT model | `gemini-2.5-flash` |
| Worker memory/disk | `2G` / `2G` |
| Worker concurrency | `2` |
| CAP dispatch limit/backoff | `2` / `10000ms` |
| Audio chunk | `720s` |
| Upload/STT max | `157286400` bytes |
| Diarization | `false` |
| Summary template | `meeting-whisper-summary-ko-v1 / orchestration / 0.0.1` |
| Direct LLM model | `anthropic--claude-4.7-opus` (`CAP_LLM_MODEL`) |

실제 값은 `mta.yaml`을 최종 기준으로 확인한다.

Summary template와 direct LLM model은 서로 다른 설정이다. 요약 모델은 AI Core의 등록 template가 결정하고, 현재 direct 경로를 사용하는 요약 검토는 `CAP_LLM_MODEL`을 `model.name`으로 사용한다. 따라서 `CAP_LLM_MODEL` 변경은 요약 template의 모델을 변경하지 않는다. 배포 후 수동 환경변수 변경 이력이 있다면 `meeting-whisper-srv`의 런타임 값도 확인하되, 전체 환경 출력에 포함된 credential은 저장하거나 공유하지 않는다.

## 배포

### 사전 확인

```powershell
git status --short
git branch --show-current
cf target
npm.cmd install
```

배포 전에 최소 검증을 수행한다.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_worker_api.py tests/test_worker_cap_client.py tests/test_worker_audio_payload.py tests/test_worker_aicore_transcriber.py
npx cds compile cap/db/schema.cds cap/srv/meeting-service.cds cap/srv/worker-internal-service.cds --to sql
npm.cmd run cap:build
```

### MTA 배포

```powershell
npm.cmd run build:mtar
cf deploy mta_archives/meeting-whisper_0.1.0.mtar
```

이미 검증된 MTAR를 배포할 때:

```powershell
npm.cmd run deploy:cf:mtar
```

### 배포 후 확인

```powershell
cf apps
cf services
cf app meeting-whisper-srv
cf app meeting-whisper-approuter
cf app meeting-whisper-stt-worker
```

```powershell
Invoke-RestMethod https://org-build-tf-joule-1on6yytw-dev-meeting-whisper-stt-worker.cfapps.jp10.hana.ondemand.com/health
```

마지막으로 Approuter에서 로그인하고 짧은 파일의 업로드 → 전사 → 요약 완료를 확인한다.

## Role Collection

`xs-security.json`의 role template:

- `MeetingUser`: 일반 사용자
- `MeetingAdmin`: 운영/admin action
- `Worker`: Worker -> CAP internal service-to-service 호출

설정 원칙:

1. 일반 사용자/그룹에는 `MeetingUser`를 할당한다.
2. 운영자에게만 필요 시 `MeetingAdmin`을 추가한다.
3. `Worker`는 사람에게 할당하지 않는다.
4. 변경 후 Approuter에서 다시 로그인하고 `/user-api/currentUser`와 회의 목록 접근을 확인한다.

## Secret rotation

한 번에 한 종류만 교체하고 smoke가 끝나기 전에 이전 credential을 폐기하지 않는다.

### CAP -> Worker token

```powershell
cf set-env meeting-whisper-srv TRANSCRIPTION_WORKER_TOKEN <new-token>
cf set-env meeting-whisper-stt-worker TRANSCRIPTION_WORKER_TOKEN <new-token>
cf restage meeting-whisper-srv
cf restage meeting-whisper-stt-worker
```

두 앱의 값은 항상 같아야 한다. 설정 후 짧은 전사 smoke를 실행한다.

### XSUAA, AI Core, Object Store

- raw credential 복사보다 service binding 갱신을 우선한다.
- Worker에 `meeting-whisper-auth` binding이 있는지 확인한다.
- Object Store binding 이름은 `objectstore`, `objectstoreRead`를 유지한다.
- binding 변경 후 대상 앱을 restage하고 `401/403` 또는 storage 오류가 없는지 확인한다.

## 장애 확인 순서

1. UI에서 meeting ID, title, status, percent와 오류 문구를 확보한다.
2. CAP `getMeetingStatus`의 dispatch 상태·시도 횟수·queue position·Worker stage를 확인한다.
3. `TranscriptionDispatches`의 `status`, `attempts`, `lockedAt`, `lastError`를 확인한다.
4. 앱과 Worker 상태·로그를 확인한다.

```powershell
cf app meeting-whisper-srv
cf app meeting-whisper-stt-worker
cf logs meeting-whisper-srv --recent
cf logs meeting-whisper-stt-worker --recent
cf logs meeting-whisper-approuter --recent
```

### 증상별 판단

| 증상 | 우선 확인 |
| --- | --- |
| `queued`, Worker busy/`429` | queue position과 backoff; 추가 수동 dispatch 금지 |
| 오래된 `dispatching` | 실제 Worker 실행 여부 확인 후 stale reset/retry |
| CAP -> Worker `401/403` | 양쪽 `TRANSCRIPTION_WORKER_TOKEN` 일치 |
| Worker -> CAP `401/403` | XSUAA binding, Worker scope, CAP direct internal route |
| Object Store 오류 | write/read binding과 signed URL |
| AI Core STT 오류 | service binding, model, quota, chunk 로그 |
| `No space left on device` | Worker disk, 동시 처리 수와 임시 chunk |
| transcript validation 오류 | 원본 보존 여부 확인 후 재전사 |
| summary만 실패 | transcript를 기준으로 요약 재실행 |

운영 스크립트와 주의사항은 [세션과 운영도구](세션과_운영도구.md)를 따른다.

## Rollback

이전 정상 MTAR가 있으면 그대로 재배포한다. 없으면 이전 정상 commit에서 다시 build한다.

```powershell
cf deploy mta_archives/meeting-whisper_0.1.0.mtar
```

환경변수만 문제라면 이전 값을 복원하고 관련 앱을 restage한다.

```powershell
cf set-env meeting-whisper-srv <KEY> <previous-value>
cf restage meeting-whisper-srv
cf restage meeting-whisper-approuter
cf restage meeting-whisper-stt-worker
```

HDI schema/data rollback은 자동으로 하지 않는다. DB 변경은 forward migration 또는 별도 백업/복구 절차로 판단한다.

## 운영에서 하지 않을 것

- 사용자를 CAP srv direct route로 안내하지 않는다.
- 운영 secret을 Vault, source, shell history나 공유 채팅에 기록하지 않는다.
- Worker busy 상태에서 dispatch row를 중복 생성하지 않는다.
- 원본 존재 여부를 확인하지 않고 transcript/summary를 삭제하지 않는다.
- 테스트하지 않은 MTAR를 dev 밖의 환경으로 바로 승격하지 않는다.
- Kyma 문서를 현재 운영 배포 절차로 사용하지 않는다.

## 코드 레포의 기준 문서

- `README.md`
- `mta.yaml`
- `.env.example`
- `deploy/OPERATIONS.md`
- `deploy/cf/README.md`
