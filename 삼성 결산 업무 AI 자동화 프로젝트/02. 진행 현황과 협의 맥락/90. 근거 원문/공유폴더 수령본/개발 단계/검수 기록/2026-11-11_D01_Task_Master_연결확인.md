---
title: "D01 Task Master 연결 확인"
project: poc-samsung
author: jycha
type: validation
status: review
created: 2026-11-11
updated: 2026-11-11
summary: "마스터 미조회 2개·영향 업무 18건 중 완료 16건 확인. 활성 seed·이력·인수 자료 대조 후 원래 업무 대응은 미확정"
related:
  - "[[02_개발단계/검수기록/2026-11-11_R-01_Fiori_기준동작]]"
  - "[[01_계획단계/Fiori_수정사항과_적용계획]]"
tags:
  - project/poc-samsung
  - type/validation
  - phase/development
  - status/review
  - tech/fiori
  - tech/rap
  - tech/btp
---

# D01 Task Master 연결 확인

**대시보드 41건 / 전체 Task 43건의 차이는 업무 마스터 연결 누락과 두 조회의 조인 방식 차이로 설명된다.** 추가 2건이 참조하는 TaskUuid는 Config 서비스에서도 조회되지 않는다. 정상 대조 업무의 마스터는 같은 서비스에서 조회된다.

문제 UUID 두 개를 참조하는 기간별 업무는 **2026년 1~9월 각 2건, 총 18건**이다. 이 기록은 원인 확인이며 데이터 복구 완료 기록은 아니다. 마스터가 처음부터 누락됐는지, 삭제됐는지, 이전 UUID가 남았는지는 아직 확정하지 않았다.

## 실행 기준

| 필드 | 값 |
|---|---|
| case_id / stage_id | D01-01~06 / RV01 읽기 점검 |
| requirement_ids | R-01, F-03, F-10; 마스터 무결성과 실제 조회 원천 확인 |
| wbs_ids | 고객 엑셀 과제 0·1.2.4 관련. 이번 확인만으로 해당 과제 완료·공수 확정 처리하지 않음 |
| source_sha | `381385e1c9679a0bd859374e00053f804ccfc03d`; 기존 docs 변경은 보존, 앱 소스 변경 없음 |
| deployed_version | SAP 활성 CDS 7개를 ADT GET으로 읽어 저장소와 공백 제거 후 내용 일치 확인. CF 배포 앱 전체 버전 동일성은 미검증 |
| environment_alias | LOCAL_FIORI_SAP_READ — 기존 로컬 AppRouter·Destination·CF SSH 경로 |
| fixture_version | 2026-11-11 실제 조회 시점. 동결/격리 fixture 없음 |
| role_alias | LOCAL_NOAUTH_DEST_BASIC — 기존 Destination 인증, SAP 실제 사용자·client 번호는 별도 확인하지 않음 |
| approval_id | 사용자 채팅 “진행하자”에 따른 추가 2건의 마스터 연결 읽기 점검. 데이터 복구/삭제 승인 아님 |
| executed_at / executor_role | 2026-11-11 17:30 KST 전후 / Codex 읽기 API·활성 소스 점검 |
| expected | 기간별 업무의 TaskUuid가 마스터에 대응하며 같은 컨텍스트의 대시보드·전체 목록이 일관됨 |
| actual | Config Tasks에서 문제 마스터 2개 미조회. 같은 UUID 참조 업무 총 18건. 활성 CDS의 INNER JOIN / association 차이 확인 |
| result | FAIL — D01 데이터 연결 결함 재현. 원인 조사 수행 완료, 복구·재검증 미실시 |
| trace_reference | 아래 D01-01~06. 로컬 재현 요청: 소스 저장소 `03_Fiori_App/http/fiori-master-integrity.http` |
| side_effects | GET 조회만 수행. 업무 Action, SQL 쓰기, 클래스 실행, 서비스 활성화·배포 미실시. 접속/조회 로그는 생성될 수 있음 |
| rollback_reference | 시스템/업무 데이터 변경 없어 복구 작업 없음 |
| reviewer / reviewed_at | 검토자 미지정 / 미검토 |

## 확인한 요청과 결과

공통 컨텍스트는 회사 `1710`, 템플릿 `MONTHLY CLOSING TEMPLATE`, 연도 `2026`, 기간 `009`, 원장 `0L`이다. TemplateUuid·CycleUuid는 [[02_개발단계/검수기록/2026-11-11_R-01_Fiori_기준동작|R-01 기준 기록]]과 같다. 문제 마스터는 문서에서 M01·M02로 부르며, 정확한 UUID는 로컬 HTTP 재현 파일에서 관리한다. 실명·담당자 ID·업무 결과 원문은 기록하지 않는다.

| case_id | 요청 / 비교 | 실제 결과 |
|---|---|---|
| D01-01 | DashboardTasks / TaskInstances를 회사·템플릿·연도·기간·원장·CycleUuid 모두 같은 조건으로 조회 | 41 / 43건. UUID 집합 차이는 2건. 두 업무는 서로 다른 M01·M02를 참조, ExecutionMode=AGENT이고 코드·이름·분류·우선순위가 빈 값 |
| D01-02 | 별도 Config 서비스 `$metadata`와 `Tasks` 읽기 | HTTP 200. Tasks·Templates·Dependencies 노출. UI manifest의 mainService와 별도 서비스이며 기존 Destination 경로로 읽기 가능 |
| D01-03 | `Tasks`를 M01·M02의 TaskUuid만으로 조회. 템플릿·ActiveYn 제한 없음. 각 키 단건 조회도 수행 | 목록 HTTP 200 / 0건, 단건 각각 HTTP 404. 다른 템플릿·비활성 조건 때문에 요청에서 숨긴 결과가 아님 |
| D01-04 | 정상 대조군과 전체 템플릿 마스터 조회 | 정상 업무 INV030의 TaskUuid 단건 조회 성공, 동일 템플릿·코드 검색 1건. 해당 템플릿 전체 마스터는 41건, 모두 ActiveYn=Y. count와 반환 수 일치, nextLink 없음 |
| D01-05 | M01·M02 참조 TaskInstances를 기간 제한 없이 조회 | count 18 / 반환 18 / nextLink 없음. 각 마스터 UUID를 9건씩 참조. 2026년 1~9월 매 기간 2건, 회사·템플릿 각 1종 |
| D01-06 | SAP 활성 CDS 7개와 서비스 정의 2개를 ADT의 `source/main` GET으로 조회 | 모두 HTTP 200. 아래 원천과 조인 구조 확인. CDS 7개는 저장소와 공백 제거 후 내용 일치 |

Config 서비스 경로는 `/sap/opu/odata4/sap/zui_cls_config_o4/srvd/sap/zui_cls_config/0001/`이다. 이는 추정 URL만 적은 것이 아니라 실제 메타데이터와 Tasks 조회로 확인한 경로다.

재현 파일의 14개 GET도 확인했다. 06·07은 위 결함을 재현하는 404, 나머지는 200이었다. ADT의 11~14번은 파일에 적은 `Accept: text/plain`을 적용해야 하며, 점검 도구에서 이 헤더를 생략했을 때는 400이 나와 헤더를 반영한 뒤 200을 확인했다. 재현 요청의 상태 일치는 기존 결함의 해소를 뜻하지 않는다.

## 활성 SAP에서 확인한 조회 구조

| 활성 객체 | 실제 관계 / 의미 |
|---|---|
| ZUI_CLS_DASHBOARD | DashboardTasks → ZC_Cls_DashTask, TaskInstances → ZC_Cls_TaskInst |
| ZUI_CLS_CONFIG | Tasks → ZC_Cls_TaskMaster |
| ZC_Cls_TaskMaster → ZI_Cls_TaskMaster | 마스터 projection → `zcls_taskmaster`. 확인한 소스에는 WHERE 필터 없음, authorizationCheck는 NOT_REQUIRED |
| ZC_Cls_TaskInst → ZI_Cls_TaskInst | 기간별 업무 projection → `zcls_taskinst`. TaskUuid로 `_TaskMaster` association 연결, 코드·이름 등을 해당 관계에서 가져옴 |
| ZC_Cls_DashTask → ZI_Cls_DashTaskBase | TaskInst에 Cycle과 TaskMaster를 INNER JOIN. Master의 TaskUuid와 일치하지 않는 업무는 행 자체가 제외됨 |
| ZI_Cls_Cycle | `zcls_cycle` 조회 |

즉, 전체 목록에는 기간별 업무 43건이 남지만 마스터가 연결되지 않는 2건의 표시 필드는 비어 있다. 대시보드는 연결되는 41건만 반환한다. 이는 프런트의 단순 건수 표시 문제를 고치는 것으로 해결되는 결함이 아니다.

또한 **현재 확인한 핵심 조회 경로는 기존 zcls_* 테이블 기반**이다. R-01 때 미확인이었던 원천을 이번 활성 소스 확인으로 보완했다. ZC01_*로 이관됐다고 볼 수 없으며, ZC01 데이터를 다른 경로에서 사용 중인지까지 조사한 것은 아니다.

확인한 CDS는 ZI_Cls_TaskMaster, ZC_Cls_TaskMaster, ZI_Cls_TaskInst, ZC_Cls_TaskInst, ZI_Cls_Cycle, ZI_Cls_DashTaskBase, ZC_Cls_DashTask다. OData 응답만으로 판단하지 않고 활성 코드와 함께 대조했다. SAP client 간 원본 테이블 비교나 삭제·이관 이력 조회는 수행하지 않았다.

## 후속: 누락된 두 업무의 원본 추적

2026-11-11 17:35~17:47 KST 전후에 M01·M02의 원래 코드·이름을 추가로 추적했다. **조회 가능한 자료만으로 두 UUID의 원래 업무를 특정하지 못했다.** 일정 시각이나 Agent라는 공통점만으로 감가상각·외화평가 등의 업무에 대응시키지 않는다.

이번 실행의 case_id는 D01-07~13이다. 위 실행 기준의 stage_id·요구사항·WBS·source_sha·접속 경로·권한 별칭·검토 상태를 그대로 적용한다. 사용자의 후속 “진행하자”에 따라 읽기 조사만 수행했으며, 아래 활성 클래스 3개의 내용도 저장소와 공백 제거 후 일치했다. 기대값은 원래 UUID→업무 코드·이름의 확인이고, 실제 결과는 대응 근거 미확보다. **원본 대응 확인 결과는 BLOCKED**, 기존 D01 결함은 FAIL로 유지한다. 데이터·앱 소스 변경이 없으므로 rollback 대상도 없다.

| case_id | 조회·대조 범위 | 확인한 사실 / 한계 |
|---|---|---|
| D01-07 | 문제 UUID 참조 TaskInstances 18건의 상태·계획·실적·버전 | 1~8월 16건은 `에이전트 수행 완료`, 실제 시작·종료 및 실행 참조·결과가 존재. 9월 2건은 `미처리`이며 실제 수행 값 없음. 모두 VersionNo=1. 완료 기록을 삭제·재생성하는 복구는 사용하지 않음 |
| D01-08 | 활성 `ZCL_CLS_SEED_DATA`, `ZCL_CLS_RUNTIME_SEED`, `ZCL_CLS_MIGRATE_P1`을 ADT GET으로 읽기 | 세 소스 모두 문제 UUID 없음. 현재 마스터 41개 UUID·업무 코드는 seed의 41개와 모두 일치. 누락된 원래 업무를 현재 seed에서 골라 복원할 근거 없음 |
| D01-09 | ADT가 반환한 seed 과거 버전 | 반환된 1개 버전 `00000`의 updated는 `2026-09-16T02:59:42Z`. 41개 코드가 있고 문제 UUID 없음. 조회 가능한 한 버전의 확인이며 전체 변경·삭제 이력을 조사한 것은 아님 |
| D01-10 | Config Dependencies에서 M01·M02를 선행/후행 UUID로 검색 | HTTP 200 / count 0 / 반환 0. 의존관계에서 원래 업무의 단서 미확보 |
| D01-11 | 영향받는 18개 TaskInstanceUuid로 StatusHistory 직접 필터 | HTTP 200 / count 0 / 반환 0 / nextLink 없음. 해당 18건의 상태이력에서는 단서 미확보. R-01 D02의 정상 대조 업무 이력 2건/association 0건 관찰은 별도로 유지 |
| D01-12 | 인수 원본 Excel 5개에서 두 UUID를 하이픈 유무 정규화하여 검색 | 구현 요구사항 v0.1, 결산 Activity Flow 수정본, ZC01 테이블 정의 v1.1, ZC01 insert 샘플 v1.1, ZFCV_JOB_SRV_POST_Parameters에서 일치 없음. 새 ZC01 샘플의 업무 ID를 기존 UUID에 임의 대응할 근거 없음. 원본 파일 수정 없음 |
| D01-13 | 저장소 텍스트·Git 텍스트 이력·현재 소스 ZIP과 중첩 UI5 ZIP·기존 vault 문서 검색 | 두 UUID를 원래 TaskCode·TaskName으로 연결할 근거 없음. 현재 runtime seed와 RAP GenerateTaskInstances는 기존 마스터를 읽어 그 UUID로 업무를 생성하므로, 현재 소스만으로 두 UUID의 유입 경로는 설명되지 않음. 과거 SAP 변경·삭제 이력은 별도 확인 필요 |

M01의 계획은 각 월 말일 22:00 KST 시작·120분, M02는 다음 달 1일 17:00 KST 시작·120분이다. 현재 마스터에 같은 offset·시작시각의 항목이 없어 일정만으로도 대응하지 못했다. 실행 참조·결과는 메모리에서 업무 코드 및 감가상각·외화평가·미결항목 관련 표현과 대조했으나 단서를 찾지 못했다. 결과 원문·담당자 ID는 저장하지 않았다.

seed는 템플릿이 이미 있으면 조기 종료하며, runtime seed는 현재 활성 마스터 기준으로 기간별 업무를 만든다. **이 클래스들을 실행해 원본을 복원할 수 있다고 판단하지 않는다.** 이번 조사에서는 클래스 실행, 업무 Action, 쓰기 요청, 활성화·배포를 하지 않았다.

로컬 HTTP 파일에는 15~20번 GET을 추가했다. 15번은 완료 이력 보존 기준, 16번은 의존관계, 17번은 현재 UUID·코드, 18~20번은 활성 클래스 내용을 확인하는 요청이다. 모두 HTTP 200을 확인했다. 과거 버전 내용은 ADT 클래스 리소스가 제공한 `includes/main/versions` 링크와 해당 entry의 content 링크를 따라 조회했다. 버전 주소를 다른 시스템에도 같은 값으로 가정하지 않는다.

### FI/RAP·원 개발사에 확인할 내용

> 현재 연결된 SAP와 동일 client 기준으로, 로컬 `fiori-master-integrity.http`의 `masterA`(M01)·`masterB`(M02)에 해당하는 원본 `ZCLS_TASKMASTER` 행 또는 **UUID → TaskCode·TaskName 대응표**를 확인해 주세요. 2026년 1~9월 총 18개 업무가 참조하며, 그중 16개는 완료 상태입니다. UUID를 교체한 이력이 있다면 이전·신규 UUID 대응과 원본 마스터 속성도 필요합니다. 삭제·이관·Transport 또는 초기 적재 이력이 있다면 함께 확인해 주세요.

현재 응답만으로 마스터 누락과 UUID 교체 후 참조 잔존을 구별할 수 없다. 위 대응이 확보되면 원본 UUID의 마스터 복원과 기존 참조 정정 중 적합한 방식을 결정하고, 18건의 상태·실적·결과 보존을 확인한다. 이 자료 확보는 과제 0의 서비스·데이터 이관 검증에 필요한 준비 사항이며, 0번 이관이나 UI 개선 완료로 처리하지 않는다.

## 복구 전에 확정할 사항

| 확인할 것 | 이유 / 담당 |
|---|---|
| M01·M02가 원래 어떤 업무였는지 | 이름·코드가 없어 현재 응답만으로 원래 업무를 복원할 수 없음. FI/RAP 담당이 원본 마스터·인수/이관 자료·동일 client 데이터를 대조 |
| 마스터 누락인지 UUID 교체 후 이전 참조 잔존인지 | 누락이면 원래 UUID·정상 속성의 마스터 복구, UUID 교체라면 검증된 대응에 따른 참조 정정이 후보. 새 UUID를 임의 생성하거나 이름만 추정하면 연결이 복구되지 않을 수 있음 |
| 기존 업무의 상태·이력·결과 보존 | 영향 범위는 9월 2건이 아니라 총 18건. 업무 재생성·삭제·seed 실행을 복구안으로 바로 사용하지 않음 |
| 0번 ZC01 이관과 이번 정정의 적용 환경 | 현재는 기존 테이블 경로다. 기존 환경 정정과 신규 ZC01 서비스 이관의 검수 기준을 구분 |

원본 업무 대응이 확정되기 전에는 마스터 복원과 참조 정정 중 어느 쪽이 맞는지 결정할 수 없다. 이번 단계에서 데이터 변경은 하지 않았다.

## 복구 후 재검증 기준

- 같은 컨텍스트의 DashboardTasks와 TaskInstances의 UUID 집합·건수가 합의된 정상 대상과 일치해야 한다. 정상 건수를 41 또는 43으로 미리 확정하지 않는다.
- 해당 업무의 코드·이름·분류·우선순위와 Agent 집계가 원본 기준에 맞아야 한다.
- 영향받는 1~9월 전체에서 연결을 다시 검사하고, 정상 41개 업무와 기존 상태·이력·업무 결과가 보존됐는지 확인한다.
- 상세 진입·담당자 필터·진행률·우선순위·타임라인을 같은 조건으로 회귀한다. R-01의 D02/D03은 별도 결함으로 남긴다.

[[01_계획단계/Fiori_수정사항과_적용계획|Fiori 수정사항과 적용 계획]]
