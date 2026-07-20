# ADR-001: CF Worker + SAP AI Core STT 전환

상태: Accepted

결정일: 2026-07-02

최종 검증: 2026-07-20

## 결정

운영 전사 경로를 Kyma CPU Whisper Worker에서 Cloud Foundry Python Worker + SAP AI Core STT로 전환한다.

```text
Before
CAP srv -> Kyma Worker -> CPU Whisper/WhisperX

After
CAP srv -> CF STT Worker -> SAP AI Core gemini-2.5-flash
```

Kyma manifest와 local Whisper 경로는 코드 레포에 fallback/reference로 남지만 현재 운영 배포에는 사용하지 않는다.

## 배경

Kyma Worker는 CAP과 전사 실행 환경을 분리하는 구조를 검증하는 데는 성공했다. 그러나 운영 전사로는 다음 문제가 있었다.

- GPU 없는 CPU Whisper의 긴 파일 처리시간을 예측하기 어려움
- WhisperX/torch image와 메모리·임시 디스크 부담
- base64 또는 큰 HTTP payload의 CAP/Worker 메모리 부담
- Kyma 임시/free plan의 지속성
- 화자분리·모델 선택 조합에 따른 성능 편차
- 전사 저장과 요약 완료가 같은 처리 수명에 묶이는 문제

## 선택한 구조

```text
Browser
  -> CF Approuter / XSUAA
  -> CAP srv
     -> HANA Cloud
     -> Object Store temporary audio
     -> dispatch queue
  -> CF STT Worker
     -> SAP AI Core STT
  -> CAP internal saveTranscript
     -> background summary
```

역할:

- CAP: 권한, metadata, Object Store, queue, 상태, transcript/summary 저장
- Worker: job claim, 오디오 다운로드·분할, AI Core STT, transcript callback
- AI Core: 실제 STT model 실행

## 운영 기준

| 항목 | 값 |
| --- | --- |
| Worker app | `meeting-whisper-stt-worker` |
| Engine | `TRANSCRIPTION_ENGINE=aicore` |
| Model | `AICORE_STT_MODEL=gemini-2.5-flash` |
| Worker instance/concurrency | `1 / 2` |
| Worker memory/disk | `2G / 2G` |
| Chunk | `720s` |
| Inline threshold | `25165824` bytes |
| Max audio | `157286400` bytes |
| Diarization | `false` |

## 보안과 통신

- CAP -> Worker는 shared bearer `TRANSCRIPTION_WORKER_TOKEN`을 사용한다.
- Worker -> CAP은 XSUAA binding과 Worker scope를 사용한다.
- callback URL은 CAP srv direct `/internal` route다.
- 사용자는 Worker와 CAP direct route를 호출하지 않고 Approuter만 사용한다.

## 성능 근거

| 구조 | 파일 | 관측 결과 |
| --- | --- | --- |
| Kyma CPU Whisper | 54분 27초 | 8분 경과 후에도 완료되지 않았고 안정적 완료 기준 확보 실패 |
| CF Worker + AI Core STT | 11분 14초 | 전사 약 1분 10초, 요약 약 9.9초 |

실제 장시간 성능은 음성 품질, chunk 수, AI Core 상태에 따라 달라지므로 60~110분 파일을 계속 관측한다.

## 결과

긍정적 결과:

- CAP, Worker와 AI model의 책임이 명확해짐
- 무거운 Whisper runtime을 운영 Worker에서 제거
- 전사 저장 후 background summary로 HTTP 수명 단축
- Worker concurrency와 CAP queue를 독립 조정 가능
- AI Core token/latency usage 기록 가능

감수한 제약:

- AI Core 서비스 상태와 quota에 의존
- model 변경은 사용자 UI가 아니라 운영 env 변경 필요
- 자동 화자분리는 기본 제공하지 않음
- 긴 파일의 chunk 경계와 비용을 계속 관측해야 함

## 후속 설계

- 길이 기반 강제 분할과 응답 검증: [장시간 전사와 요약](장시간_전사와_요약.md)
- 과거 검증 요약: [Kyma 검증과 전환 요약](../90_과거이력/Kyma_검증과_전환_요약.md)
