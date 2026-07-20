# 운영

현재 운영 기준은 Cloud Foundry의 Approuter, CAP srv, STT Worker와 SAP AI Core다. Kyma Worker는 운영 경로가 아니다.

## 문서

- [배포와 장애대응 Runbook](배포와_장애대응_Runbook.md)
- [세션과 운영도구](세션과_운영도구.md)

## 매일 확인할 최소 항목

```powershell
cf target
cf apps
cf services
cf app meeting-whisper-srv
cf app meeting-whisper-approuter
cf app meeting-whisper-stt-worker
```

Worker health:

```powershell
Invoke-RestMethod https://org-build-tf-joule-1on6yytw-dev-meeting-whisper-stt-worker.cfapps.jp10.hana.ondemand.com/health
```

문제가 있으면 다음 로그를 먼저 본다.

```powershell
cf logs meeting-whisper-srv --recent
cf logs meeting-whisper-stt-worker --recent
cf logs meeting-whisper-approuter --recent
```

## 운영 원칙

- 사용자는 Approuter URL로만 접속한다.
- Worker internal callback은 CAP srv 직접 route를 사용한다.
- `TRANSCRIPTION_WORKER_TOKEN`은 CAP srv와 Worker에 같은 값을 설정한다.
- 실제 secret, token과 service key JSON은 Vault와 Git에 남기지 않는다.
- HDI schema rollback은 자동으로 처리하지 않는다.
- 원본 오디오가 없는 회의는 운영 스크립트만으로 재전사할 수 없다.
