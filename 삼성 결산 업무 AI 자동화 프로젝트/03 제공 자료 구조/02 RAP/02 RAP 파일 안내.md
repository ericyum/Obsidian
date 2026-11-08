# RAP 파일 안내

[[삼성 결산 업무 AI 자동화 프로젝트/00 프로젝트 길잡이|00 프로젝트 길잡이]] · [[삼성 결산 업무 AI 자동화 프로젝트/03 제공 자료 구조/00 제공 자료 전체 지도|00 제공 자료 전체 지도]] · [[삼성 결산 업무 AI 자동화 프로젝트/03 제공 자료 구조/02 RAP/01 RAP 백엔드 구조와 데이터 흐름|01 RAP 백엔드 구조와 데이터 흐름]]

> [!summary] 안내 기준
> 풀린 RAP 인계본의 **185개 파일**을 실제 파일명 그대로 안내한다. 별도 ZIP 1개를 포함하면 `02_RAP`의 물리 파일은 186개다.
> ZIP 내부는 중복 집계하지 않았다. 표와 코드 경로는 원본을 찾기 위한 것이며 SAP 배포·실행 결과는 아니다.

## 파일 집계와 경로 규칙

모든 아래 경로는 저장소의 `02_RAP/` 기준 상대 경로다.
공통 접두사 `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/` 아래에 핵심 객체가 있다.
상위 패키지·Gateway 프로젝트도 빠짐없이 포함한다.
파일 목록은 객체별 정확한 파일 세트이며, 한 CDS 이름이 Data/Behavior/Access Controls에 각각 등장하면 서로 다른 ABAP 객체 유형이다.

| 영역 | 논리 객체 | 파일 |
|---|---:|---:|
| 패키지 식별 | 2 | 2 |
| 기타 Gateway 프로젝트 | 1 | 1 |
| 권한 객체·기본값 | 2 | 2 |
| Database Tables | 7 | 14 |
| CDS Data Definitions | 31 | 62 |
| CDS Behavior Definitions | 4 | 8 |
| CDS Metadata Extensions | 6 | 12 |
| CDS Access Controls | 2 | 4 |
| Service Definitions | 3 | 6 |
| Service Bindings | 3 | 3 |
| Gateway V4 Service Groups | 1 | 1 |
| Classes | 10 | 70 |
| **풀린 객체 합계** | **72** | **185** |
| 원본 ZIP | 보관물 1개 | 1 |

## 직렬화 파일을 읽는 법

동일 객체를 여러 파일로 나눈 것이므로 각 파일을 서로 다른 기능으로 세지 않는다.

| 파일 패턴 | 역할과 주의 |
|---|---|
| `*.tabl.ddic` / `*.tabl.json` | 테이블 필드 정의 / 객체 속성 |
| `*.ddls.acds` / `*.ddls.json` | CDS select·projection 정의 / 객체 속성 |
| `*.bdef.abdl` / `*.bdef.json` | behavior 동작·필드·관계 선언 / 객체 속성 |
| `*.ddlx.acds` / `*.ddlx.json` | UI annotation 확장 / 객체 속성 |
| `*.dcls.acds` / `*.dcls.json` | 조회 권한 DCL / 객체 속성 |
| `*.srvd.acds` / `*.srvd.json` | 서비스 expose 목록 / 객체 속성 |
| `*.srvb.json` | 서비스 binding type·category·버전·정의 연결 |
| `*.jsonc` | 패키지·Gateway·권한 객체의 식별 메타데이터. 일부는 VS Code 미지원 안내만 포함 |
| `*.clas.abap` | 전역 클래스 정의/구현. 일반 classrun의 본문이 주로 여기 있음 |
| `*.clas.implementations.abap` | local helper 구현. **ZBP behavior pool의 실제 로직**을 여기서 읽음 |
| `*.clas.definitions.abap` | local helper 정의 include. 제공본은 안내 주석 수준 |
| `*.clas.macros.abap` | macro include. 제공본은 안내 주석 수준 |
| `*.clas.testclasses.abap` | ABAP Unit include. 제공본은 안내 주석이며 실행 가능한 테스트 클래스가 아님 |
| `*.clas.json` | 클래스 설명·언어 등 메타데이터 |
| `*.clas.texts.en.properties` | 영어 text symbol 리소스. 제공본 10개는 빈 파일 |

예를 들어 `ZBP_I_CLS_CYCLE`은 7개 파일로 직렬화되지만 하나의 behavior pool이다.
`ZCL_CLS_P1_SELFTEST.clas.abap`는 실행 가능한 조회 점검 코드지만, 빈 `testclasses` include와 혼동하지 않는다.

## 원본 보관물

- `ZAPL_FINANCE.zip` — 압축 인계본. 이미 풀린 디렉터리 분석과 중복 집계하지 않음.

## 객체별 전체 파일 목록

아래 각 객체의 코드 블록에는 해당 디렉터리의 **모든 직접 파일**이 들어 있다.
각 디렉터리의 역할은 바로 위 설명을 따른다.

### ZAPL_FINANCE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/`

상위 SAP 패키지. Finance RAP 객체의 패키지 이름과 설명을 보존한다.

```text
zapl_finance.devck.jsonc
```

### ZAFAB_JOB

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/Others/GW Service Builder Projects/ZAFAB_JOB/`

Gateway Service Builder 프로젝트의 객체 식별 메타데이터. Job 구현·서비스 처리 클래스는 이 디렉터리에 없다.

```text
zafab_job.iwpr.jsonc
```

### ZAPL_FI_CLOSE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/`

결산 Dashboard RAP 하위 패키지. 테이블·CDS·behavior·서비스 구현의 중심이다.

```text
zapl_fi_close.devck.jsonc
```

### 499DD1F96DB09C950E0BBDEED45D04HT

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Authorizations/Authorization Defaults (TADIR)/499DD1F96DB09C950E0BBDEED45D04HT/`

설정 서비스 그룹 ZUI_CLS_CONFIG_O4에 연결된 Authorization Default(TADIR) 객체 메타데이터.

```text
499dd1f96db09c950e0bbdeed45d04ht.sush.jsonc
```

### Z_FICLOSE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Authorizations/Authorization Objects/Z_FICLOSE/`

결산 권한 객체 식별 메타데이터. BUKRS/ACTVT를 쓰는 실제 검사는 behavior와 DCL에서 읽는다.

```text
z_ficlose.susob.jsonc
```

### ZUI_CLS_CONFIG_O4

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Business Services/Gateway V4 Service Groups/ZUI_CLS_CONFIG_O4/`

설정 서비스 OData V4 UI 바인딩. Gateway 서비스 그룹 객체는 별도의 식별 메타데이터다.

```text
zui_cls_config_o4.g4ba.jsonc
```

### ZAPI_CLS_INSIGHT_O4

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Business Services/Service Bindings/ZAPI_CLS_INSIGHT_O4/`

Insight 서비스 OData V4 Web API 바인딩. destination·기술 사용자·접근성은 여기서 검증되지 않았다.

```text
zapi_cls_insight_o4.srvb.json
```

### ZUI_CLS_CONFIG_O4

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Business Services/Service Bindings/ZUI_CLS_CONFIG_O4/`

설정 서비스 OData V4 UI 바인딩. Gateway 서비스 그룹 객체는 별도의 식별 메타데이터다.

```text
zui_cls_config_o4.srvb.json
```

### ZUI_CLS_DASH_O4

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Business Services/Service Bindings/ZUI_CLS_DASH_O4/`

Dashboard 서비스 OData V4 UI 바인딩. JSON 존재만으로 publish 상태는 확인되지 않는다.

```text
zui_cls_dash_o4.srvb.json
```

### ZAPI_CLS_INSIGHT

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Business Services/Service Definitions/ZAPI_CLS_INSIGHT/`

Insight Web API 서비스 정의. 결산 데이터와 MCP 재무 조회 총 11 entity를 expose한다. 읽기 전용 MCP 정책과 서비스 전체 권한은 별도 검증 대상이다.

```text
zapi_cls_insight.srvd.acds
zapi_cls_insight.srvd.json
```

### ZUI_CLS_CONFIG

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Business Services/Service Definitions/ZUI_CLS_CONFIG/`

설정 서비스 정의. Templates/Tasks/Dependencies 세 entity alias를 expose한다.

```text
zui_cls_config.srvd.acds
zui_cls_config.srvd.json
```

### ZUI_CLS_DASHBOARD

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Business Services/Service Definitions/ZUI_CLS_DASHBOARD/`

UI 서비스 정의. 대시보드·실행 상태·이력·이슈·값 도움말 총 15 entity를 expose한다.

```text
zui_cls_dashboard.srvd.acds
zui_cls_dashboard.srvd.json
```

### ZC_CLS_DASHAGENT

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Access Controls/ZC_CLS_DASHAGENT/`

조회 DCL. CompanyCode를 Z_FICLOSE/BUKRS, ACTVT=03에 연결한다. 실제 PFCG role/사용자 할당과 신원 전파 검증 결과는 포함하지 않는다.

```text
zc_cls_dashagent.dcls.acds
zc_cls_dashagent.dcls.json
```

### ZC_CLS_DASHEXEC

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Access Controls/ZC_CLS_DASHEXEC/`

조회 DCL. CompanyCode를 Z_FICLOSE/BUKRS, ACTVT=03에 연결한다. 실제 PFCG role/사용자 할당과 신원 전파 검증 결과는 포함하지 않는다.

```text
zc_cls_dashexec.dcls.acds
zc_cls_dashexec.dcls.json
```

### ZC_CLS_CYCLE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Behavior Definitions/ZC_CLS_CYCLE/`

Cycle projection behavior: 인터페이스의 생성·수정·업무/차수 액션·이슈 해결 액션을 서비스에서 사용한다. Cycle delete는 use하지 않는다.

```text
zc_cls_cycle.bdef.abdl
zc_cls_cycle.bdef.json
```

### ZC_CLS_TEMPLATE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Behavior Definitions/ZC_CLS_TEMPLATE/`

Template projection behavior: Template/Task/Dependency CRUD와 association 생성 동작을 소비 계층에 허용한다.

```text
zc_cls_template.bdef.abdl
zc_cls_template.bdef.json
```

### ZI_CLS_CYCLE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Behavior Definitions/ZI_CLS_CYCLE/`

Cycle managed behavior: 차수·업무 인스턴스·이력·이슈의 영속 테이블·CRUD·액션·검증·잠금·권한 종속 관계를 선언한다.

```text
zi_cls_cycle.bdef.abdl
zi_cls_cycle.bdef.json
```

### ZI_CLS_TEMPLATE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Behavior Definitions/ZI_CLS_TEMPLATE/`

Template managed behavior: Template/Task/Dependency의 영속 테이블·CRUD·managed UUID·ETag·잠금·권한 종속 관계를 선언한다.

```text
zi_cls_template.bdef.abdl
zi_cls_template.bdef.json
```

### ZC_CLS_ASSIGNEEVH

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZC_CLS_ASSIGNEEVH/`

업무 인스턴스와 차수에서 담당자/회사코드별 후보를 읽는 값 도움말.

```text
zc_cls_assigneevh.ddls.acds
zc_cls_assigneevh.ddls.json
```

### ZC_CLS_COMPANYCODESVH

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZC_CLS_COMPANYCODESVH/`

기존 결산 차수의 distinct 회사코드 값 도움말. SAP 전체 회사코드 마스터와 동일하지 않다.

```text
zc_cls_companycodesvh.ddls.acds
zc_cls_companycodesvh.ddls.json
```

### ZC_CLS_CYCLE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZC_CLS_CYCLE/`

Cycle transactional_query root projection. TaskInstance 관계를 서비스용 객체로 redirect한다.

```text
zc_cls_cycle.ddls.acds
zc_cls_cycle.ddls.json
```

### ZC_CLS_DASHAGENT

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZC_CLS_DASHAGENT/`

AGENT 모드 한정 업무 집계. 회사코드 PFCG 조회 DCL이 별도로 있다.

```text
zc_cls_dashagent.ddls.acds
zc_cls_dashagent.ddls.json
```

### ZC_CLS_DASHCYCLE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZC_CLS_DASHCYCLE/`

차수별 집계. 전체/완료(SKIPPED 포함)/자동화/Agent/오류 업무 수와 진행률을 계산한다.

```text
zc_cls_dashcycle.ddls.acds
zc_cls_dashcycle.ddls.json
```

### ZC_CLS_DASHEXEC

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZC_CLS_DASHEXEC/`

실행 모드별 집계. 회사코드 PFCG 조회 DCL이 별도로 있다. 상태·분류·담당자·우선순위까지 group by한다.

```text
zc_cls_dashexec.ddls.acds
zc_cls_dashexec.ddls.json
```

### ZC_CLS_DASHTASK

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZC_CLS_DASHTASK/`

대시보드 업무별 행 조회. Agent/자동화 구분·집계 보조 숫자·TaskInstance 탐색을 제공한다. analytical cube가 아니다.

```text
zc_cls_dashtask.ddls.acds
zc_cls_dashtask.ddls.json
```

### ZC_CLS_EXECUTIONMODEVH

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZC_CLS_EXECUTIONMODEVH/`

기존 업무 인스턴스의 실행 모드와 표시 텍스트 값 도움말.

```text
zc_cls_executionmodevh.ddls.acds
zc_cls_executionmodevh.ddls.json
```

### ZC_CLS_MCP_OPENITEM

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZC_CLS_MCP_OPENITEM/`

미결 항목 상세 소비용 조회. 회계문서 키·계정·거래처·기일·금액·통화·담당자를 노출한다.

```text
zc_cls_mcp_openitem.ddls.acds
zc_cls_mcp_openitem.ddls.json
```

### ZC_CLS_MCP_OPENITEMAGG

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZC_CLS_MCP_OPENITEMAGG/`

회사코드·계정·담당자·회사코드 통화별 미결 항목 수와 금액 합계.

```text
zc_cls_mcp_openitemagg.ddls.acds
zc_cls_mcp_openitemagg.ddls.json
```

### ZC_CLS_MCP_SUPPINV

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZC_CLS_MCP_SUPPINV/`

공급업체 송장 상태 A/B에 한정한 PendingSupplierInvoices 조회.

```text
zc_cls_mcp_suppinv.ddls.acds
zc_cls_mcp_suppinv.ddls.json
```

### ZC_CLS_PRIORITYVH

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZC_CLS_PRIORITYVH/`

기존 업무 마스터의 우선순위와 표시 텍스트 값 도움말.

```text
zc_cls_priorityvh.ddls.acds
zc_cls_priorityvh.ddls.json
```

### ZC_CLS_STATHIST

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZC_CLS_STATHIST/`

StatusHistory 소비용 projection. 이력 조회와 관계 탐색을 노출한다.

```text
zc_cls_stathist.ddls.acds
zc_cls_stathist.ddls.json
```

### ZC_CLS_TASKCATEGORYVH

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZC_CLS_TASKCATEGORYVH/`

기존 업무 마스터의 분류 값 도움말.

```text
zc_cls_taskcategoryvh.ddls.acds
zc_cls_taskcategoryvh.ddls.json
```

### ZC_CLS_TASKDEP

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZC_CLS_TASKDEP/`

Dependency 소비용 projection. 설정 서비스에서 업무 선행관계를 노출한다.

```text
zc_cls_taskdep.ddls.acds
zc_cls_taskdep.ddls.json
```

### ZC_CLS_TASKINST

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZC_CLS_TASKINST/`

TaskInstance 소비용 projection. 화면·API가 볼 필드와 History/Issue/Cycle 관계를 제공한다.

```text
zc_cls_taskinst.ddls.acds
zc_cls_taskinst.ddls.json
```

### ZC_CLS_TASKISSUE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZC_CLS_TASKISSUE/`

TaskIssue 소비용 projection. 이슈 조회 및 behavior 소비 모델이다.

```text
zc_cls_taskissue.ddls.acds
zc_cls_taskissue.ddls.json
```

### ZC_CLS_TASKMASTER

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZC_CLS_TASKMASTER/`

TaskMaster 소비용 projection. Template·Dependency 관계를 서비스용 객체로 redirect한다.

```text
zc_cls_taskmaster.ddls.acds
zc_cls_taskmaster.ddls.json
```

### ZC_CLS_TASKSTATUSVH

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZC_CLS_TASKSTATUSVH/`

기존 업무 인스턴스의 상태 코드와 표시 텍스트 값 도움말.

```text
zc_cls_taskstatusvh.ddls.acds
zc_cls_taskstatusvh.ddls.json
```

### ZC_CLS_TEMPLATE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZC_CLS_TEMPLATE/`

Template 소비용 root projection. 서비스용 Task projection으로 composition을 redirect한다.

```text
zc_cls_template.ddls.acds
zc_cls_template.ddls.json
```

### ZC_CLS_TEMPLATEVH

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZC_CLS_TEMPLATEVH/`

ActiveYn=Y인 결산 템플릿 값 도움말.

```text
zc_cls_templatevh.ddls.acds
zc_cls_templatevh.ddls.json
```

### ZI_CLS_CYCLE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZI_CLS_CYCLE/`

Cycle root 인터페이스. Template 이름 등 기준 정보와 TaskInstance composition을 제공한다.

```text
zi_cls_cycle.ddls.acds
zi_cls_cycle.ddls.json
```

### ZI_CLS_DASHTASKBASE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZI_CLS_DASHTASKBASE/`

TaskInstance·Cycle·TaskMaster를 조인한 공통 대시보드 fact view. 업무 행의 계획/실제/상태 정보를 모은다.

```text
zi_cls_dashtaskbase.ddls.acds
zi_cls_dashtaskbase.ddls.json
```

### ZI_CLS_MCP_OPENITEM

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZI_CLS_MCP_OPENITEM/`

표준 회계항목/전표 CDS를 조인하고 ClearingJournalEntry가 빈 항목을 읽는 MCP 인터페이스. authorizationCheck는 NOT_REQUIRED다.

```text
zi_cls_mcp_openitem.ddls.acds
zi_cls_mcp_openitem.ddls.json
```

### ZI_CLS_MCP_SUPPINV

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZI_CLS_MCP_SUPPINV/`

I_SupplierInvoiceAPI01 기반 공급업체 송장 인터페이스. authorizationCheck는 NOT_REQUIRED다.

```text
zi_cls_mcp_suppinv.ddls.acds
zi_cls_mcp_suppinv.ddls.json
```

### ZI_CLS_STATHIST

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZI_CLS_STATHIST/`

StatusHistory 인터페이스. TaskInstance parent와 Cycle을 연결한다.

```text
zi_cls_stathist.ddls.acds
zi_cls_stathist.ddls.json
```

### ZI_CLS_TASKDEP

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZI_CLS_TASKDEP/`

TaskDep 인터페이스. Task parent, Template, 선행 Task association을 제공한다.

```text
zi_cls_taskdep.ddls.acds
zi_cls_taskdep.ddls.json
```

### ZI_CLS_TASKINST

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZI_CLS_TASKINST/`

TaskInstance 인터페이스. Cycle/TaskMaster를 연결하고 상태 텍스트·criticality·업무 플래그를 계산하며 History/Issue composition을 제공한다.

```text
zi_cls_taskinst.ddls.acds
zi_cls_taskinst.ddls.json
```

### ZI_CLS_TASKISSUE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZI_CLS_TASKISSUE/`

TaskIssue 인터페이스. TaskInstance parent, Cycle, 선택적 StatusHistory와 이슈 표시 criticality를 제공한다.

```text
zi_cls_taskissue.ddls.acds
zi_cls_taskissue.ddls.json
```

### ZI_CLS_TASKMASTER

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZI_CLS_TASKMASTER/`

TaskMaster 인터페이스. Template parent와 Dependency composition을 연결한다.

```text
zi_cls_taskmaster.ddls.acds
zi_cls_taskmaster.ddls.json
```

### ZI_CLS_TEMPLATE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZI_CLS_TEMPLATE/`

Template root 인터페이스. TaskMaster를 composition으로 연결한다.

```text
zi_cls_template.ddls.acds
zi_cls_template.ddls.json
```

### ZME_CLS_CYCLE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Metadata Extensions/ZME_CLS_CYCLE/`

Cycle 화면 annotation. Overview·자식 Tasks 섹션, Open/Close/Reopen/Generate 액션을 배치한다.

```text
zme_cls_cycle.ddlx.acds
zme_cls_cycle.ddlx.json
```

### ZME_CLS_DASHCYCLE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Metadata Extensions/ZME_CLS_DASHCYCLE/`

대시보드 차수 집계의 선택 필드·목록 열 annotation.

```text
zme_cls_dashcycle.ddlx.acds
zme_cls_dashcycle.ddlx.json
```

### ZME_CLS_DASHTASK

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Metadata Extensions/ZME_CLS_DASHTASK/`

대시보드 필터·값 도움말·Priority/Agent/Dday/Upcoming별 목록 qualifier annotation.

```text
zme_cls_dashtask.ddlx.acds
zme_cls_dashtask.ddlx.json
```

### ZME_CLS_STATHIST

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Metadata Extensions/ZME_CLS_STATHIST/`

상태 이력의 순번·유형·이전/다음 상태·변경자·시간 목록 annotation.

```text
zme_cls_stathist.ddlx.acds
zme_cls_stathist.ddlx.json
```

### ZME_CLS_TASKINST

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Metadata Extensions/ZME_CLS_TASKINST/`

업무 상세의 Overview·계획/실제·실행·이력·이슈 섹션과 여섯 업무 액션 annotation.

```text
zme_cls_taskinst.ddlx.acds
zme_cls_taskinst.ddlx.json
```

### ZME_CLS_TASKISSUE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Metadata Extensions/ZME_CLS_TASKISSUE/`

이슈 상세와 심각도/상태 criticality 및 ResolveIssue 액션 annotation.

```text
zme_cls_taskissue.ddlx.acds
zme_cls_taskissue.ddlx.json
```

### ZCLS_CYCLE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Dictionary/Database Tables/ZCLS_CYCLE/`

회사코드·연도·기간·원장별 결산 차수와 기준일·담당자·열림/닫힘 시간을 저장한다.

```text
zcls_cycle.tabl.ddic
zcls_cycle.tabl.json
```

### ZCLS_STATHIST

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Dictionary/Database Tables/ZCLS_STATHIST/`

업무 상태 변경 이력: 순번·이벤트·이전/다음 상태·변경자·이유·실행 참조를 저장한다.

```text
zcls_stathist.tabl.ddic
zcls_stathist.tabl.json
```

### ZCLS_TASKDEP

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Dictionary/Database Tables/ZCLS_TASKDEP/`

업무와 선행 업무의 관계: 필요 상태·lag·필수/활성 여부를 저장한다. 관계 저장 자체가 자동 선행조건 실행 엔진을 뜻하지 않는다.

```text
zcls_taskdep.tabl.ddic
zcls_taskdep.tabl.json
```

### ZCLS_TASKINST

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Dictionary/Database Tables/ZCLS_TASKINST/`

차수별 업무 실행 스냅샷: 상태·실행 모드·시간·결과·오류·완료/승인자를 저장한다.

```text
zcls_taskinst.tabl.ddic
zcls_taskinst.tabl.json
```

### ZCLS_TASKISSUE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Dictionary/Database Tables/ZCLS_TASKISSUE/`

업무별 이슈: 유형·심각도·SAP 객체 참조·담당자·기한·해결 정보를 저장한다.

```text
zcls_taskissue.tabl.ddic
zcls_taskissue.tabl.json
```

### ZCLS_TASKMASTER

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Dictionary/Database Tables/ZCLS_TASKMASTER/`

템플릿 아래 업무 정의: 실행 유형·기본 담당자·계획 시간·우선순위·검토/승인 조건·Fiori 탐색 정보를 저장한다.

```text
zcls_taskmaster.tabl.ddic
zcls_taskmaster.tabl.json
```

### ZCLS_TEMPLATE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Dictionary/Database Tables/ZCLS_TEMPLATE/`

결산 템플릿: 명칭·유형·버전·유효기간·활성 여부를 저장한다.

```text
zcls_template.tabl.ddic
zcls_template.tabl.json
```

### ZBP_I_CLS_CYCLE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Source Code Library/Classes/ZBP_I_CLS_CYCLE/`

차수·업무·이슈 behavior pool. 상태 전이, 업무 생성, 회사코드 권한, 중복 차수 검증, 상태 이력 추가를 구현한다. Agent 요청은 상태/이력 기록 단계다.

```text
zbp_i_cls_cycle.clas.abap
zbp_i_cls_cycle.clas.definitions.abap
zbp_i_cls_cycle.clas.implementations.abap
zbp_i_cls_cycle.clas.json
zbp_i_cls_cycle.clas.macros.abap
zbp_i_cls_cycle.clas.testclasses.abap
zbp_i_cls_cycle.clas.texts.en.properties
```

### ZBP_I_CLS_TEMPLATE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Source Code Library/Classes/ZBP_I_CLS_TEMPLATE/`

템플릿 behavior pool. 생성/수정/삭제의 Z_FICLOSE 권한을 검사한다. BUKRS는 DUMMY다.

```text
zbp_i_cls_template.clas.abap
zbp_i_cls_template.clas.definitions.abap
zbp_i_cls_template.clas.implementations.abap
zbp_i_cls_template.clas.json
zbp_i_cls_template.clas.macros.abap
zbp_i_cls_template.clas.testclasses.abap
zbp_i_cls_template.clas.texts.en.properties
```

### ZCL_CLS_DIAG_EXECMODE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Source Code Library/Classes/ZCL_CLS_DIAG_EXECMODE/`

MANUAL/AUTOMATIC/AGENT 이외 실행 모드를 조회하고 마스터 기준 예상 모드를 출력한다.

```text
zcl_cls_diag_execmode.clas.abap
zcl_cls_diag_execmode.clas.definitions.abap
zcl_cls_diag_execmode.clas.implementations.abap
zcl_cls_diag_execmode.clas.json
zcl_cls_diag_execmode.clas.macros.abap
zcl_cls_diag_execmode.clas.testclasses.abap
zcl_cls_diag_execmode.clas.texts.en.properties
```

### ZCL_CLS_DIAG_FA030

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Source Code Library/Classes/ZCL_CLS_DIAG_FA030/`

FA030/SIMULATION 한 행을 MANUAL/NOT_STARTED로 보정한다. DIAG 명칭이지만 실제 UPDATE/COMMIT을 수행한다.

```text
zcl_cls_diag_fa030.clas.abap
zcl_cls_diag_fa030.clas.definitions.abap
zcl_cls_diag_fa030.clas.implementations.abap
zcl_cls_diag_fa030.clas.json
zcl_cls_diag_fa030.clas.macros.abap
zcl_cls_diag_fa030.clas.testclasses.abap
zcl_cls_diag_fa030.clas.texts.en.properties
```

### ZCL_CLS_DIAG_LEGACY_STATE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Source Code Library/Classes/ZCL_CLS_DIAG_LEGACY_STATE/`

정규 상태 목록 밖의 행을 조회하고 마스터 플래그·실제 시간·최근 상태 이력을 출력한다.

```text
zcl_cls_diag_legacy_state.clas.abap
zcl_cls_diag_legacy_state.clas.definitions.abap
zcl_cls_diag_legacy_state.clas.implementations.abap
zcl_cls_diag_legacy_state.clas.json
zcl_cls_diag_legacy_state.clas.macros.abap
zcl_cls_diag_legacy_state.clas.testclasses.abap
zcl_cls_diag_legacy_state.clas.texts.en.properties
```

### ZCL_CLS_FIX_LEGACY_STATE

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Source Code Library/Classes/ZCL_CLS_FIX_LEGACY_STATE/`

고정 UUID별 보정 계획과 최신 이력·마스터 매핑·버전을 검사한 뒤 스냅샷 상태/모드를 보정한다. 일반 마이그레이션으로 재사용하지 않는다.

```text
zcl_cls_fix_legacy_state.clas.abap
zcl_cls_fix_legacy_state.clas.definitions.abap
zcl_cls_fix_legacy_state.clas.implementations.abap
zcl_cls_fix_legacy_state.clas.json
zcl_cls_fix_legacy_state.clas.macros.abap
zcl_cls_fix_legacy_state.clas.testclasses.abap
zcl_cls_fix_legacy_state.clas.texts.en.properties
```

### ZCL_CLS_MIGRATE_P1

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Source Code Library/Classes/ZCL_CLS_MIGRATE_P1/`

MONTH→MONTHLY, APP→FIORI_APP, JOB→BACKGROUND_JOB, AGENT→AI_AGENT 정규화 UPDATE/COMMIT 도구.

```text
zcl_cls_migrate_p1.clas.abap
zcl_cls_migrate_p1.clas.definitions.abap
zcl_cls_migrate_p1.clas.implementations.abap
zcl_cls_migrate_p1.clas.json
zcl_cls_migrate_p1.clas.macros.abap
zcl_cls_migrate_p1.clas.testclasses.abap
zcl_cls_migrate_p1.clas.texts.en.properties
```

### ZCL_CLS_P1_SELFTEST

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Source Code Library/Classes/ZCL_CLS_P1_SELFTEST/`

조회 기반 데이터/코드/상태/대시보드 점검과 PASS/WARN/FAIL 출력 classrun. 실행 결과와 ABAP Unit 테스트는 별개다.

```text
zcl_cls_p1_selftest.clas.abap
zcl_cls_p1_selftest.clas.definitions.abap
zcl_cls_p1_selftest.clas.implementations.abap
zcl_cls_p1_selftest.clas.json
zcl_cls_p1_selftest.clas.macros.abap
zcl_cls_p1_selftest.clas.testclasses.abap
zcl_cls_p1_selftest.clas.texts.en.properties
```

### ZCL_CLS_RUNTIME_SEED

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Source Code Library/Classes/ZCL_CLS_RUNTIME_SEED/`

과거 및 실행일 기준 현재 기간의 데모 차수·업무 상태·이력·이슈 생성 classrun. 실제 Agent 실행 결과가 아니다.

```text
zcl_cls_runtime_seed.clas.abap
zcl_cls_runtime_seed.clas.definitions.abap
zcl_cls_runtime_seed.clas.implementations.abap
zcl_cls_runtime_seed.clas.json
zcl_cls_runtime_seed.clas.macros.abap
zcl_cls_runtime_seed.clas.testclasses.abap
zcl_cls_runtime_seed.clas.texts.en.properties
```

### ZCL_CLS_SEED_DATA

**경로:** `ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Source Code Library/Classes/ZCL_CLS_SEED_DATA/`

기본 템플릿·업무·의존성 샘플 생성 classrun. 고정 UUID 중복 시 중단하며 직접 INSERT/COMMIT/ROLLBACK을 사용한다.

```text
zcl_cls_seed_data.clas.abap
zcl_cls_seed_data.clas.definitions.abap
zcl_cls_seed_data.clas.implementations.abap
zcl_cls_seed_data.clas.json
zcl_cls_seed_data.clas.macros.abap
zcl_cls_seed_data.clas.testclasses.abap
zcl_cls_seed_data.clas.texts.en.properties
```

## 찾아가기 예시

**Agent 버튼 이후 무엇이 구현됐는지:**
`…/Core Data Services/Behavior Definitions/ZC_CLS_CYCLE/zc_cls_cycle.bdef.abdl` → `…/ZI_CLS_CYCLE/zi_cls_cycle.bdef.abdl` → `…/Source Code Library/Classes/ZBP_I_CLS_CYCLE/zbp_i_cls_cycle.clas.implementations.abap`를 읽는다.
구현은 AGENT_RUNNING 및 AGENT_START 이력 기록이며 외부 runtime callback pending이 명시된다.

**집계 숫자가 다를 때:**
`…/Core Data Services/Data Definitions/ZC_CLS_DASHCYCLE/zc_cls_dashcycle.ddls.acds`와 `…/ZC_CLS_DASHEXEC/zc_cls_dashexec.ddls.acds`의 완료 조건과 group by 단위를 비교한다.
DashCycle 완료는 SKIPPED를 포함하고 DashExec 완료는 COMPLETED만 포함한다.

**권한 질문이 생겼을 때:**
Access Controls의 두 DCL → ZBP handler의 AUTHORITY-CHECK → 실제 SAP role/사용자·BTP 신원 설정 순으로 확인한다.
현재 문서는 마지막 단계의 운영 상태를 확인하지 않았으며 사용자 권한 전파는 후속 과제로 남아 있다.

> [!warning] 실행 주의
> 이 문서는 소스 읽기 안내다. Seed·migration·FIX·DIAG_FA030은 SAP DB에 쓰므로 인계 문서를 읽는 과정에서 실행할 필요가 없다.
> 원본 보존을 위해 소스 수정·배포·테스트 실행은 하지 않았다.

## 원본 위치

원본 루트: `C:\Users\ericy\github\POC\ASPN\02_RAP`.
[원본 폴더](<file:///C:/Users/ericy/github/POC/ASPN/02_RAP/>)
[테이블 객체 폴더](<file:///C:/Users/ericy/github/POC/ASPN/02_RAP/ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Dictionary/Database%20Tables/>)
[CDS 객체 폴더](<file:///C:/Users/ericy/github/POC/ASPN/02_RAP/ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core%20Data%20Services/>)
[클래스 객체 폴더](<file:///C:/Users/ericy/github/POC/ASPN/02_RAP/ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Source%20Code%20Library/Classes/>)
