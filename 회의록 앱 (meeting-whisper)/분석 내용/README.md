# Meeting Whisper — 코드 분석

> **분석일**: 2026-07-20
> **기준 소스**: `meeting_whisper` `0182231`
> **목적**: 프로젝트 전체 코드 구조와 설계를 철저히 이해하기 위한 분석 문서

---

## 읽는 순서

1. **[전체 코드 분석](meeting-whisper_전체_코드_분석.md)** — 프로젝트 구조, 데이터 모델, End-to-End 흐름, 설계 결정
2. **[CAP 백엔드](1. CAP 백엔드 — 데이터 모델 & API 분석.md)** — 데이터 모델, API, 라이브러리, Dispatcher
3. **[프론트엔드](2. 프론트엔드 — Vanilla JS SPA 분석.md)** — View 구조, JS 모듈, 세션·녹음·편집
4. **[STT Worker](3. STT Worker — Python FastAPI 분석.md)** — Job lifecycle, chunk·검증·재시도
5. **[배포](4. 배포 — MTA & Cloud Foundry 분석.md)** — MTA 구조, 서비스, 인증 흐름, ZEN-OTT와 비교

---

## 프로젝트 한눈에

| 항목 | 내용 |
|------|------|
| **목적** | 음성 파일 → AI Core 전사 → 요약 → 회의록 공유 |
| **프론트엔드** | Vanilla JS SPA |
| **백엔드** | SAP CAP (Node.js) + HANA Cloud |
| **Worker** | Python FastAPI |
| **전사** | SAP AI Core `gemini-2.5-flash` |
| **요약** | SAP AI Core Orchestration (`claude-4.7-opus`) |
| **저장소** | Object Store (오디오) + HANA (메타데이터) |
| **인증** | XSUAA + Approuter |
| **배포** | Cloud Foundry (단일 MTA, 4개 모듈) |

---

## ZEN-OTT와의 주요 차이점

| 항목 | ZEN-OTT | Meeting Whisper |
|------|---------|-----------------|
| 프론트엔드 | UI5 (SAP 표준) | Vanilla JS |
| html5-apps-repo | 사용 | 미사용 |
| MTA 구조 | 여러 개의 분리된 MTA | 단일 MTA |
| 백엔드 연결 | BTP Destination 수동 설정 | MTA provides/requires 자동 |
| Worker | 없음 | Python 별도 앱 |
| 배포 복잡도 | 높음 | 낮음 |

---

## 관련 Vault

- `meeting-whisper-vault/` — 공식 운영·설계 문서
- [프로젝트 개요](../meeting-whisper-vault/00_시작하기/프로젝트_개요.md)
- [사용자 가이드](../meeting-whisper-vault/00_시작하기/사용자_가이드.md)
- [운영 README](../meeting-whisper-vault/01_운영/README.md)
