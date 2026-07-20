# Kyma 검증과 CF AI Core STT 전환 요약

이 문서는 현재 판단에 영향을 준 Kyma 검증 결과만 보존한다. 당시 일별 작업 로그, Docker 기초 설명과 상세 구현 계획은 Vault Git 이력에서 확인할 수 있다.

## 당시 목표

2026-06 Meeting Whisper는 다음 구조를 검증했다.

```text
Browser -> CF Approuter -> CAP srv -> Object Store
                                      -> Kyma Worker
                                         -> CPU Whisper/WhisperX
                                      <- transcript callback
```

Kyma Worker는 사용자 요청과 분리된 전사 실행 환경을 제공하고, CAP은 HANA·권한·queue·요약을 담당하는 구성이었다.

## 확인한 성과

- Docker image와 Kyma Deployment/Service/APIRule 구성
- CAP -> Worker shared bearer token 검증
- Worker -> CAP internal callback과 XSUAA client credentials 검증
- Object Store signed URL 기반 오디오 전달
- Worker busy 시 `429`, CAP queue/retry 처리
- 사용자 UI의 dispatch 오류와 장시간 대기 상태 표시

## 운영 경로로 채택하지 않은 이유

1. GPU 없는 CPU Whisper는 긴 회의의 완료 시간을 안정적으로 보장하기 어려웠다.
2. WhisperX/torch image가 크고 메모리·임시 디스크 요구량이 높았다.
3. 50~60분 오디오에서 OOM, 502와 긴 처리시간 리스크가 있었다.
4. Kyma 임시/free plan의 지속성과 운영 리소스가 불확실했다.
5. 모델·화자분리 조합이 사용자 선택에 노출되면 품질과 처리시간을 예측하기 어려웠다.

대표 관측값:

| 구조 | 파일 | 결과 |
| --- | --- | --- |
| Kyma + CPU Whisper | 54분 27초 | 8분 경과 후에도 완료되지 않았고 안정적인 완료 기준을 확보하지 못함 |
| CF Worker + AI Core STT | 11분 14초 | 전사 약 1분 10초, 요약 약 9.9초 |

## 전환 결정

2026-07-02부터 운영 전사를 다음 구조로 전환했다.

```text
CAP srv -> CF STT Worker -> SAP AI Core STT
```

- 기본 엔진: `TRANSCRIPTION_ENGINE=aicore`
- 기본 모델: `AICORE_STT_MODEL=gemini-2.5-flash`
- 기본 화자분리: `DIARIZE=false`
- Worker instance: 1개, concurrency slot: 2개
- 원본 전달: Object Store
- 전사 저장 후 요약: CAP background task

## 남긴 교훈

- 사용자 UI/API와 장시간 AI 처리는 런타임 책임을 분리한다.
- 큰 오디오는 base64 JSON 대신 Object Store와 signed URL을 사용한다.
- Worker 요청 접수, claim, heartbeat, callback을 별도 상태로 기록한다.
- 전사 저장과 요약 완료를 같은 HTTP 수명에 묶지 않는다.
- 장시간 입력은 크기뿐 아니라 예상 재생시간으로도 분할 여부를 판단한다.
- 잘못된 STT JSON을 text fallback으로 저장하지 않고 재시도·검증한다.

현재 결정의 상세는 [ADR-001](../02_기술_설계/ADR-001_CF_AI_Core_STT_전환.md)을 따른다.
