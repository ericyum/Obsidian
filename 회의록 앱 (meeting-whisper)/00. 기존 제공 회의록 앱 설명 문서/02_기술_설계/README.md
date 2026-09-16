# 기술 설계

현재 코드의 구현 세부를 이해할 때 필요한 문서만 유지한다.

## 문서

- [ADR-001 CF AI Core STT 전환](ADR-001_CF_AI_Core_STT_전환.md)
  - Kyma CPU Whisper를 중단하고 CF Worker + SAP AI Core STT를 선택한 이유
- [AI Core LLM과 프롬프트](AI_Core_LLM과_프롬프트.md)
  - 요약·검토 호출 방식, template, schema와 category guidance
- [장시간 전사와 요약](장시간_전사와_요약.md)
  - 오디오 chunk, STT 검증·복구, 계층형 요약

## 소스에서 바로 확인할 위치

```text
cap/srv/lib/ai-core.js                  요약·검토·AI usage
cap/srv/lib/meeting-note.js             transcript/note 검증과 변환
cap/srv/lib/transcription-dispatcher.js queue와 Worker dispatch
cap/srv/worker-internal-service.js      Worker callback과 background summary
workers/transcription/app/aicore_stt.py AI Core STT, chunk와 검증
workers/transcription/app/main.py       Worker API와 concurrency
mta.yaml                                운영 모듈·서비스·환경값
```

Vault 문서와 코드가 다르면 코드를 기준으로 문서를 갱신한다.
