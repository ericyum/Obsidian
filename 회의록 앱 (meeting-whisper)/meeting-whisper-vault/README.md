# Meeting Whisper Vault

최종 정리: 2026-07-20

기준 소스: `meeting_whisper` `0182231` (`main`)

이 Vault는 Meeting Whisper의 **현재 구조를 빠르게 이해하고 운영하기 위한 문서**다. 구현 세부의 최종 기준은 코드이며, Vault에는 신규 참여자와 운영자에게 필요한 설명·의사결정·복구 절차만 유지한다.

## 처음 보는 사람의 읽는 순서

1. [프로젝트 개요](00_시작하기/프로젝트_개요.md) — 서비스 목적, 현재 아키텍처, 데이터 흐름
2. [사용자 가이드](00_시작하기/사용자_가이드.md) — 실제 화면과 사용자 동작
3. [운영 README](01_운영/README.md) — 배포·장애 대응·세션·운영 도구
4. [기술 설계 README](02_기술_설계/README.md) — STT 전환, LLM, 장시간 처리
5. [변경 이력](03_변경이력/CHANGELOG.md) — 현재 구조가 만들어진 핵심 마일스톤

Kyma 기반 전사 구조가 왜 폐기됐는지만 확인하려면 [Kyma 검증과 전환 요약](90_과거이력/Kyma_검증과_전환_요약.md)을 본다.

## 현재 구조 한눈에 보기

```text
Browser
  -> CF Approuter / XSUAA
  -> CAP srv
     -> HANA Cloud
     -> Object Store
     -> TranscriptionDispatches
  -> CF STT Worker
     -> SAP AI Core STT, gemini-2.5-flash
  -> CAP internal saveTranscript
     -> background summary/review
     -> SAP AI Core Orchestration
```

현재 운영의 핵심은 다음과 같다.

- 사용자 UI, CAP, STT Worker는 Cloud Foundry에서 운영한다.
- 전사는 SAP AI Core `gemini-2.5-flash`, 요약은 AI Core Orchestration template를 사용한다.
- 원본 오디오는 Object Store에 임시 저장하고 처리 후 정리한다.
- 긴 오디오와 긴 전사문은 내부 분할하지만 사용자에게는 전사본 1개와 요약본 1개를 제공한다.
- Kyma/CPU Whisper 문서는 현재 운영 기준이 아니다.

## 폴더 구조

```text
00_시작하기/    신규 참여자와 사용자용 문서
01_운영/        배포, 장애 대응, 세션과 운영 스크립트
02_기술_설계/   현재 기술 설계와 중요한 의사결정
03_변경이력/    핵심 마일스톤 요약
90_과거이력/    현재 판단에 필요한 최소 과거 이력
```

## 문서와 코드의 우선순위

내용이 다르면 다음 순서를 따른다.

1. `meeting_whisper` 현재 소스와 테스트
2. `mta.yaml`, `.env.example`, CDS/Pydantic 계약
3. 이 Vault의 `00_시작하기`, `01_운영`, `02_기술_설계`
4. `03_변경이력`, `90_과거이력`

API 계약을 Vault에 별도 복제하지 않는다. 현재 계약은 다음 소스를 직접 확인한다.

- 공개 API: `cap/srv/meeting-service.cds`
- Worker internal API: `cap/srv/worker-internal-service.cds`
- Worker 요청 모델: `workers/transcription/app/main.py`
- 데이터 모델: `cap/db/schema.cds`

## 문서 유지 원칙

- 현재 동작을 바꾸는 소스 변경에는 관련 Vault 문서와 기준 commit을 함께 갱신한다.
- 배포값은 실제 secret이 아닌 변수 이름과 의미만 기록한다.
- 일일 작업 로그, 임시 계획, 코드와 중복되는 API schema는 Vault에 추가하지 않는다.
- 장기적으로 가치가 있는 결정은 ADR 또는 `CHANGELOG.md` 한 줄로 남긴다.
- 제거된 과거 문서가 필요하면 Vault Git 이력에서 복구한다.
