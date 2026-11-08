---
tags: [삼성물산, IntegrationSuite, MCP, OpenAPI, CSRF]
검토일: 2026-11-08
---

# Integration Suite 실행 구조

[[삼성 결산 업무 AI 자동화 프로젝트/03 제공 자료 구조/00 제공 자료 전체 지도|00 제공 자료 전체 지도]] · [[삼성 결산 업무 AI 자동화 프로젝트/02 구현 인수인계와 SAP 시스템 이해|02 구현 인수인계와 SAP 시스템 이해]] · [[삼성 결산 업무 AI 자동화 프로젝트/03 제공 자료 구조/05 Integration Suite/02 패키지와 내부 파일 안내|02 패키지와 내부 파일 안내]]

**이 폴더는 JWD가 선택한 SAP 업무 API를 MCP 도구로 호출할 수 있게 만든 Integration Suite 패키지의 내보내기다.** Python 앱처럼 실행하는 소스 폴더가 아니라, IS에 가져와 설정·배포할 아티팩트 묶음이다.

## 1. 실제 자료 형태

```text
05_Integration_Suite/
├─ APL15 Business Actions.zip       패키지 원본 ZIP
└─ APL15 Business Actions/          ZIP을 풀어 둔 10개 파일
   ├─ <32자리 ID>_content × 6       실제로는 각각 ZIP 아티팩트
   ├─ resources.cnt                base64 JSON 리소스 목록
   ├─ contentmetadata.md           base64 속성 텍스트
   ├─ ExportInformation.info       패키지명·내보낸 시점
   └─ hash                         내보내기 해시 메타데이터
```

2026-11-08 읽기 점검에서 원본 ZIP의 **10개 파일이 해제본 10개와 바이트 단위로 동일**함을 확인했다. `_content` 여섯 개는 확장자가 없지만 ZIP으로 열리며, 내부에는 합계 **49개 파일**이 있다. 이 비교는 실제 IS 배포 버전·활성화 상태의 검증은 아니다.

내보내기 정보의 시점은 **2026-10-13 06:45:58 UTC**다. 원본 파일·아티팩트·내부 파일 지도는 [[삼성 결산 업무 AI 자동화 프로젝트/03 제공 자료 구조/05 Integration Suite/02 패키지와 내부 파일 안내|02 패키지와 내부 파일 안내]]에 있다.

## 2. OpenAPI, MCP, iFlow의 관계

![[삼성 결산 업무 AI 자동화 프로젝트/다이어그램/04 IS 실행 경로.png|950]]

편집: [[삼성 결산 업무 AI 자동화 프로젝트/다이어그램/04 IS 실행 경로.excalidraw|04 IS 실행 경로.excalidraw]]

| 구성 | 역할 | 대표 파일 |
|---|---|---|
| **OpenAPI** | HTTP 경로·operationId·입력·출력·실행 제한 정의 | `apidefinition/apidefinition.json` |
| **API 아티팩트** | API를 IS의 호출·인증·연계 흐름에 연결 | `apiartifact_default_a1.iflw` |
| **MCP Server 아티팩트** | AI가 볼 Tool 이름·schema와 실제 소스 API를 연결 | `mcpspecification.json`, `mcpsourcemapping.json`, `mcpartifact_default_a1.iflw` |
| **CSRF Gateway iFlow** | token·cookie를 얻고 SAP POST를 수행 | `scenarioflows/integrationflow/*.iflw` |
| **Groovy 스크립트** | 은행 계좌/지급 요청 입력 검증·조회 결과 해석·draft·응답 정규화 | Bank Transfer iFlow 내부 `script/` |
| **플랫폼 설정** | 실제 대상·인증·Credential·배포 | iFlow 파라미터, IS Security Material 등 |

MCP Server를 생성했다고 SAP 업무 API가 새로 생기는 것은 아니다. SAP 측 서비스와 권한이 있어야 하며, IS에서는 해당 API를 호출할 대상·인증·흐름을 맞춰야 한다.

## 3. 현재 제공 계약의 세 도구

| MCP Tool | API operationId | HTTP 경로 | 실제 처리 |
|---|---|---|---|
| `post_depreciation_simulation` | `scheduleDepreciationSimulation` | `POST /depreciation/simulation` | 감가상각 **TEST** 작업 등록 |
| `post_foreign_currency_valuation_run` | `runForeignCurrencyValuation` | `POST /foreign-currency-valuation/run` | 외화평가 `T`/`U` 실행 |
| `post_bank_transfers_payment_requests` | `createBankTransferPaymentRequest` | `POST /bank-transfers/payment-requests` | 지급 요청 draft 생성·활성화 |

### 감가상각

제공 계약에서 **`TestRun`은 항상 true**다. 해당 도구는 시뮬레이션이며, 실제 감가상각 기표까지 이 계약으로 가능하다고 설명하지 않는다. 고유한 RequestID로 요청하고 결과를 후속 조회해야 한다.

### 외화평가

**`RunMode=T`는 테스트, `U`는 실제 업데이트**다. 실제 기표는 사용자의 명시적인 요청이 있어야 하며 애매한 지시에서 U를 추정하지 않도록 설명되어 있다. 선택 조건은 `ToSelections.results`에 전달한다.

### 계좌이체 요청

이 도구는 **draft를 만들고 활성화할 뿐, 지급 요청을 release하거나 실제 지급하지 않는다.** 출금·입금 House Bank/Account, 통화, 금액, 기준일이 필요하다. 회사코드·내부 계좌 식별자·지급 방법은 매핑이 유일할 때만 해석하고 누락·모호한 값을 추정하지 않는 계약이다.

이는 **자금 업무 예제**다. 같은 패키지에 있다고 삼성 결산 PoC의 핵심 대상에 자동 포함하지 않는다.

> [!warning] 타임아웃과 중복 실행
> 세 도구의 설명은 모호한 실패·타임아웃 뒤 자동 재시도를 경계한다. SAP에 작업/요청이 이미 생성됐을 수 있기 때문이다. 통신 응답이 없었다는 이유로 곧바로 새 요청을 반복하지 말고 기존 결과를 확인해야 한다.

## 4. CSRF Gateway가 하는 일

영상 **30:00–36:00**의 핵심은 SAP OData의 세션·CSRF 요구를 iFlow로 감싸는 것이다.

![[삼성 결산 업무 AI 자동화 프로젝트/첨부/V2-3504-CSRF-iFlow.png|950]]

*영상 2, 35:03.96. FCV Gateway의 token fetch와 POST 준비 흐름.*

| 단계 | 역할 |
|---|---|
| Prepare CSRF Fetch | 원본 요청을 보관하고 token 요청 준비 |
| Fetch CSRF Token | SAP에 GET으로 token·cookie 요청 |
| Restore POST Body | 보관했던 실제 요청 body 복구 |
| Create AFAB/FCV | 같은 세션 맥락으로 실제 SAP 작업 요청 POST |
| 응답 반환 | RequestID·접수/오류 등 결과를 호출자에게 전달 |

**CSRF token은 “이 세션에서 의도된 변경 요청인가”를 확인하는 보조 장치다. 누가 어느 회사의 결산을 실행할 수 있는지에 대한 업무 권한을 대신하지 않는다.**

현재 iFlow에는 Credential 참조 `APL_PCE_ACTIONS_01_BASIC`이 보인다. 이는 참조 이름이며 비밀번호가 아니다. 실제 Security Material과 연결 인증은 IS에 별도 존재해야 한다. 문서에는 값·실제 접속 URL을 복제하지 않았다.

## 5. 패키지의 여섯 아티팩트는 모두 결산 핵심 기능이 아니다

| 아티팩트 | 결산 관련성 | 읽을 때 주의할 점 |
|---|---|---|
| APL15 AFAB Simulation CSRF Gateway | 직접 관련 | 감가상각 TEST 연계 |
| APL15 FCV Simulation CSRF Gateway | 직접 관련 | 이름에 Simulation이 있어도 상위 계약의 T/U 의미를 확인 |
| Finance Simulation Actions API | 직접 관련 | 감가상각·외화평가와 은행 요청 세 경로의 OpenAPI |
| FINANCE_SIMULATION_ACTIONS_MCP | 직접 관련 | 세 도구와 API 매핑 |
| APL15 Bank Transfer Payment Request CSRF Gateway | 자금 예제 | draft 생성·활성화이며 지급 아님 |
| APL15 Business Actions CSRF Gateway | 별도 예제/공통명 | 실제 흐름에 **Create Planned Order**가 보임. 이름만 보고 결산 전용 Gateway로 판단하지 않음 |

AFAB/FCV 등은 내부 `.iflw` 파일명이 똑같이 `APL15 Business Actions CSRF Gateway.iflw`다. **파일명보다 컨테이너의 리소스 메타데이터·노드 내용으로 역할을 판별**해야 한다.

## 6. 최종 결과와 업무 상태 반영은 별도 계약이다

![[삼성 결산 업무 AI 자동화 프로젝트/다이어그램/03 요청과 결과 상태.png|950]]

| 단계 | 확인할 증거 |
|---|---|
| 요청 접수 | RequestID·입력 조건·TEST/UPDATE 모드·응답 |
| SAP 작업 진행 | 동일 ID의 상태·Job 식별·오류 |
| 작업 완료 | 최종 결과·로그·스풀·실제 기표이면 전표 등의 업무 증거 |
| 대시보드 완료 반영 | RAP Task/Cycle 연결, Custom 테이블 상태·이력, Fiori 재조회 |

IS에서 응답을 받았다고 대시보드 상태가 저절로 바뀌는 것은 아니다. PPTX의 **JWD 결과 → S/4 CBO 자동 업데이트** 과제와 RAP의 **외부 완료 콜백 pending**을 함께 봐야 한다.

## 7. 변경 위치를 찾는 순서

| 바꾸려는 것 | 먼저 볼 파일/구성 | 함께 확인 |
|---|---|---|
| 입력 필드·경로·모드 | API/MCP 내부 `apidefinition.json` | MCP Tool schema·설명·source mapping |
| CSRF·SAP 호출 | AFAB/FCV `.iflw` | 파라미터·Security Material·SAP 서비스 |
| 은행 요청 검증/응답 | 활성 참조 Groovy 파일 | iFlow script 참조·draft 준비/활성화 |
| 인증·실제 대상 | IS 런타임 설정·Credential·파라미터 | Destination/Cloud Connector/SAP 권한 |
| 권한 통제 | 로그인 → IS → SAP 실제 처리 사용자 | PFCG·허용/거부 테스트 |

이번에는 아티팩트와 파일을 읽고 ZIP/해제본을 대조했다. **IS Import·배포·API 호출·지급 요청 생성·SAP 기표·권한 테스트는 수행하지 않았다.**
