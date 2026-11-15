---
title: "Fiori Dashboard 분석 — 이해와 확인 질문"
project: poc-samsung
author: jycha
type: analysis
status: review
created: 2026-11-12
updated: 2026-11-13
summary: "대시보드 조회 차이와 화면별 완료 판단 현황, 기존 개발 의도·PoC 수정 필요 여부를 확인할 질문 정리"
related:
  - "[[01_계획단계/Fiori_수정사항과_적용계획]]"
  - "[[02_개발단계/검수기록/2026-11-11_R-01_Fiori_기준동작]]"
  - "[[02_개발단계/검수기록/2026-11-11_D01_Task_Master_연결확인]]"
  - "[[02_개발단계/검수기록/2026-11-12_S01_Fiori_상태판정_실데이터]]"
tags:
  - project/poc-samsung
  - type/analysis
  - phase/planning
  - status/review
  - tech/fiori
  - tech/ui5
  - tech/rap
  - tech/joule
---

## 메시지·메일용 확인 질문

기존 환경에 `ZCLS_*` 테이블 기반 OData 서비스에 연결된 Fiori Dashboard 버전을 기준으로 분석하여, 궁금한 부분을 정리했습니다.

1. 동일한 2026년 9월 조건에서 대시보드 41건·전체 Task 43건으로 조회되는데, PoC에서 이 차이를 유지해도 되는지와 신규 `ZC01_*` 전환 시 조회 기준의 변경 여부를 확인 부탁드립니다.
2. ‘에이전트 수행 완료’를 에이전트 팀·진행률에서는 완료로, 우선순위에서는 미종결로 처리하는 것이 의도된 동작인지, 시연 전에 기준을 조정해야 하는지 확인 부탁드립니다.
3. 과제 **1.2.4 「Custom Fiori App. 오류사항 수정」의 ‘전월 마지막 주부터 캘린더 표시’ 요구**(내부 관리 ID `F-02`, 상세 PPT 24쪽)와 관련해, 7월 조회 시 6/28~7/4 달력에 일정이 표시되지 않으므로 다른 카드는 7월 기준을 유지하면서 캘린더에 포함할 전월 업무 범위와 조회 기준을 어떻게 정할지 협의 부탁드립니다.
4. Task 상세 화면 상단의 업무 처리 버튼과 ✨ 버튼을 이번 PoC 시연에서 사용하는지, 사용한다면 어떤 동작·결과를 보여주면 되는지 확인 부탁드립니다.

---

Fiori Dashboard를 실제 SAP 데이터에 연결해 확인하면서, 기존 개발 의도와 PoC 시나리오 관점에서 몇 가지 확인할 사항을 정리했습니다.

PoC인 만큼 전반적인 고도화보다는 시나리오에서 보여줄 기능이 정상 동작하는 것을 우선으로 보려고 합니다.

각 시나리오 마다 어떻게 보여주고 시연할지 몰라서
**현재 현상 중 의도된 동작과 시연 전에 수정해야 할 부분을 구분하려는 목적**으로 작성해보았습니다.


# Autonomous Closing Dashboard

는 “선택한 달의 결산 업무를 얼마나 끝냈고, 무엇을 더 해야 하는지” 보는 업무 관리 화면입니다.

| 카드·영역               | 쉽게 말하면                               | 포함되는 업무                                     |
| ------------------- | ------------------------------------ | ------------------------------------------- |
| **에이전트 팀**          | “Agent 담당 업무들은 지금 어떤 상태인가?”          | Agent 업무만                                   |
| **결산 진행 현황 — 자동화율** | “전체 업무 중 Agent·자동 처리와 수동 처리의 비중은?”   | 전체 업무를 실행 방식으로 분류                           |
| **결산 진행 현황 — 진행률**  | “전체 중 얼마나 완료했고, 검토·오류는 얼마나 남았나?”     | 전체 업무를 상태로 분류                               |
| **결산 D-Day**        | “**결산 기준일까지 며칠 남았고, 주의할 업무는** 무엇인가?” | 기준일 정보 + 중요한 미완료 건수 + 오류·주의·기한 경과 업무 일부     |
| **우선순위 Task**       | “아직 **끝나지 않은 중요한 업무**부터 보자”          | Agent·수동 구분 없이 High/Critical 미완료 업무 중 상위 5건 |
| **예정 Task**         | “**계획 시작일이 있는 미완료 업무**를 보자”          | **Agent·수동 구분 없이 상위 5건**. 현재는 과거 날짜도 포함     |
| **일정 타임라인**         | “각 업무를 언제 하기로 계획했나?”                 | 계획 시작일이 있는 업무를 날짜별로 표시                      |
| **상단 전체 Task 목록**   | “요약 말고 업무를 한 건씩 모두 보자”               | 같은 조회 조건의 기간별 업무. 당시 43건                    |


1. **대시보드와 전체 Task 목록의 조회 대상·건수 차이**

	회사·결산기간 조건 9월 조회 기준
	- `/TaskInstances`의 2건은 연결된 업무 마스터를 찾지 못해 이름·코드 등이 비어 있습니다.
	- `/DashboardTasks`는 마스터까지 연결되는 업무만 가져오기 때문에 그 2건이 빠집니다.

	같은 참조 문제가 **2026년 1~9월 총 18건**에 있습니다.

	1.1 두 화면의 조회 대상과 건수가 다른 것은 의도된 설계인가요?

	1.2 PoC에서는 현재 차이가 있어도 괜찮을까요?

	1.3 현재 확인한 CDS는 기존 `ZCLS_*` 테이블 기반입니다. 과제 0에서 신규 `ZC01_*` 기반 OData로 교체할 예정이며, 이때 조회 대상이나 데이터 구조도 달라지는 설계인가요?


    ```
    /sap/opu/odata4/sap/zui_cls_dash_o4/srvd/sap/zui_cls_dashboard/0001/
    + DashboardTasks   → 대시보드 업무
    + TaskInstances    → 전체 목록·상세의 업무
    ```

    |조회 대상|Agent|Manual|합계|
    |---|---|---|---|
    |`/DashboardTasks`|12|29|**41**|
    |`/TaskInstances`|14|29|**43**|

    핵심 소스는 [zi_cls_dashtaskbase.ddls.acds 9행](<../../../../../Desktop/Project/poc-samsung/02_RAP/ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZI_CLS_DASHTASKBASE/zi_cls_dashtaskbase.ddls.acds>)입니다.
    ```
    as select from ZI_Cls_TaskInst as ti
    inner join ZI_Cls_Cycle as cy
        on cy.CycleUuid = ti.CycleUuid
    inner join ZI_Cls_TaskMaster as tm
        on tm.TaskUuid = ti.TaskUuid
    ```

    마지막 조건이 핵심입니다.

    ```
    tm.TaskUuid = ti.TaskUuid
    ```


2. **화면별 완료 판단 기준 차이**

    같은 `/DashboardTasks` 응답을 사용하지만, 화면 영역마다 완료 판단 기준이 다릅니다.

    실제 8월 CASH010은 `Status='에이전트 수행 완료'`, `IsCompleted=false`, `IsPriorityOpen=true`, `Priority='HIGH'`로 반환됩니다. 현행 코드에서는 아래처럼 처리됩니다.

    | 화면 영역 | 판단 기준 | CASH010 처리 결과 |
    |---|---|---|
    | 에이전트 팀·전체 완료 건수/진행률 | 한글 Status를 `COMPLETED`로 변환 | 완료에 포함 |
    | 일정 타임라인 | 한글 Status를 완료 상태로 변환 | Agent 완료로 분류 |
    | 우선순위 Task·D-Day 미결 Critical | 원본 Status와 우선순위 기준 | 미종결 업무에 포함 |
    | 우선순위 상태 문구의 색상 | 원본 Status 기준 | 기본 상태색(`None`) |

    우선순위 카드의 왼쪽 아이콘은 `sap-icon://alert`로 고정되어 있으며, 상태에 따라 바뀌지 않습니다([Dashboard.view.xml 464·479~480행][view]).

    서버의 `IsCompleted`도 원본 Status가 영문 `COMPLETED` 또는 `SKIPPED`인지로 계산하므로, 한글 `에이전트 수행 완료`는 현재 `false`입니다. 우선순위 카드는 `IsCompleted`를 직접 사용하지 않고 원본 상태·우선순위·`IsPriorityOpen`으로 판단합니다.


    **완료 판단 기준의 차이는 확인됐지만, 업무상 오류인지는 미확정입니다. 각 영역에서 말하는 ‘완료’의 의미를 확인한 뒤 수정 범위를 결정해야 합니다.**

    2.1 ‘에이전트 수행 완료’는 전체 업무의 최종 완료인가요, 검토·승인 전의 에이전트 실행 단계 완료인가요?

    2.2 에이전트 팀·진행률에는 완료로 포함하면서 우선순위·미결 건수에는 남기는 것이 의도된 동작인가요?

    2.3 이번 PoC에서는 현재 기준을 유지해 설명하면 될까요, 아니면 시연 전에 특정 집계·표시 기준을 수정해야 할까요?


    미종결에서 제외하는 Status
    - COMPLETED: 최종 완료
    - SKIPPED: 수행 대상에서 제외
    - 그 외는 미종결로 포함 (에이전트 수행완료  )

    | 원본 `Status` | 상태 문구의 색상 |
    |---|---|
    | `NOT_STARTED`, `READY`, `REOPENED` | 주의색 |
    | `IN_PROGRESS`, `AGENT_RUNNING` | 정보색 |
    | `REVIEW_REQUIRED` | 주의색 |
    | `APPROVAL_REQUIRED` | 정보색 |
    | `FAILED`, `BLOCKED` | 오류색 |
    | 한글 상태·빈값 등 인식하지 못하는 값 | 기본색(`None`) |



    **근거:** [LegacyStatusCompatibility.js 6~27행][legacy]의 상태 변환, [DashboardMetrics.js 156~190행][metrics]의 완료·미결 집계, [DashboardState.js 118~147행][state]의 카드 선정, [ZI_Cls_TaskInst 43~86행][taskinst-cds]의 서버 판정.

    상세 조회·계산 재현 결과는 [[02_개발단계/검수기록/2026-11-12_S01_Fiori_상태판정_실데이터]]를 참고합니다. 실제 응답과 현행 함수로 확인했으며, 이번 문서 정리에서 앱·데이터는 변경하지 않았습니다.

3. **타임라인의 표시 주간과 데이터 조회 범위**

    과제 **1.2.4 / F-02**의 요구사항은 **‘선택한 결산 월의 전월 마지막 주부터 표시’**입니다(상세 PPT 24쪽).

    현재 Fiori에서 **2026년 7월을 조회하면 캘린더는 전월 마지막 주인 6/28~7/4로 표시되지만, 현재 데이터에서는 해당 주에 일정이 나오지 않습니다.** 날짜 예시는 일요일 주 시작 기준입니다.

    이는 `/DashboardTasks` OData 호출 시 `$filter`에 **`FiscalYear eq '2026' and FiscalPeriod eq '007'`과 선택한 7월의 `CycleUuid` 조건**을 함께 전달하여 **7월 결산 업무만 조회하기 때문**입니다. 해당 주에 예정된 업무라도 6월 결산에 속하면 조회 대상에서 제외됩니다.

    **실제 조회 확인 — 2026-11-13, 회사 1710·동일 템플릿·원장 0L**

    | 조회한 결산 월 | 반환된 업무 | 계획 기간이 6/28~7/4와 겹치는 업무 |
    |---|---:|---:|
    | 6월 | 41건 | 37건 |
    | 7월 | 41건 | 0건 |

    두 요청 모두 HTTP 200으로 정상 조회됐습니다. **7월 업무 조회가 실패한 것이 아니라, 달력은 전월 마지막 주를 보여주면서 데이터는 선택한 결산 월로 제한하는 상태**입니다. 현재는 주간을 이동해도 해당 주에 맞춰 다른 결산 월의 업무를 추가 조회하지 않습니다.

    **캘린더 외의 카드·진행률은 선택한 7월 결산 데이터를 기준으로 표시하고 있습니다. 캘린더에서 전월 마지막 주의 업무까지 보여주려면, 다른 영역은 7월 기준을 유지하면서 캘린더에 한해 전월 업무를 포함하도록 조회 범위를 분리·확장할지 합의가 필요해 보입니다.**

    **근거:** [Dashboard.controller.js 438~445·603~618행][controller]의 선택 기간·CycleUuid 조회 조건, [TimelineHelper.js 52~73·109~112행][timeline]의 초기 주간 계산, [Fiori 가이드 11.1~11.2절][fiori-guide]. 표시 시작일과 데이터 포함 범위를 구분해 확인하며, 수정 범위는 아직 확정하지 않았습니다.

4. **Task 상세 화면 상단 버튼의 시연 사용 여부**

    Task 상세 화면 오른쪽 상단에 `Start Task`, `Complete Work`, `Complete Review`, `Approve`, `Reopen`, `Request AI Agent`와 **✨ 버튼**이 있습니다. 확인한 화면에서는 업무 처리 버튼들이 비활성화되어 있고, ✨ 버튼은 활성화되어 보이지만 이전 점검에서 클릭 후 눈에 보이는 반응이 없었습니다.

    [Fiori 인수 가이드 14.2절][fiori-guide]에는 업무 시작·완료·검토·승인·재오픈·Agent 실행 요청의 기능과 활성 조건이 설명돼 있습니다. 다만 **이번 PoC 시연에서 어떤 버튼을 직접 사용하는지**는 확인한 자료에서 명시되지 않았으며, 별도 ✨ 버튼의 정확한 용도도 확인이 필요합니다.

    **상세 화면 오른쪽 상단의 버튼들은 이번 PoC 시연에서 실제 사용하는 기능인가요? 사용한다면 어떤 버튼을 눌러 어떤 동작·결과를 보여주는지 확인 부탁드립니다.**

## 이 문서의 확인 범위와 연결 자료

- 1번과 4번의 업무 건수·버튼 관찰은 **2026-11-11 점검 당시 근거**를 보존합니다. **2번은 2026-11-12 실제 8·9월 GET과 현행 계산 함수 재현으로 추가 확인**했으며 S01 기록을 따릅니다. 이번에 브라우저 표시·버튼 실행을 새로 검증하지는 않았습니다.
- 3번의 6·7월 비교는 **2026-11-13 로컬 AppRouter를 통한 실제 GET** 결과입니다. `FiscalYear='2026'`, `FiscalPeriod='006'/'007'`로 각각 조회했고, 두 응답 모두 41건·단일 Cycle·다음 페이지 없음이었습니다. `PlannedStartAt/PlannedEndAt`을 한국 시간으로 해석해 6/28 00:00 이상~7/5 00:00 미만 구간과의 겹침을 계산했습니다. 브라우저를 새로 조작하거나 앱·업무 데이터를 변경하지 않았습니다.
- 받은 가이드는 인수 소스 기준 설명입니다. 버튼의 실제 외부 실행 성공이나 현재 배포 환경의 권한까지 증명하는 자료로 보지 않습니다.
- 여기의 질문과 변경 후보는 미결 사항입니다. 개발 범위·공수·완료를 확정한 것이 아니며, 결정과 진행 상태는 [[01_계획단계/Fiori_수정사항과_적용계획]]에서 관리합니다.
- 실제 조회 근거: [[02_개발단계/검수기록/2026-11-11_R-01_Fiori_기준동작]], [[02_개발단계/검수기록/2026-11-11_D01_Task_Master_연결확인]].

[legacy]: <../../../../../Desktop/Project/poc-samsung/03_Fiori_App/APL1.5_ZAPL_FINANCE_1.0.13_Partner_Source_r2_20261015/financial-close-dashboard/webapp/ext/dashboard/LegacyStatusCompatibility.js>
[metrics]: <../../../../../Desktop/Project/poc-samsung/03_Fiori_App/APL1.5_ZAPL_FINANCE_1.0.13_Partner_Source_r2_20261015/financial-close-dashboard/webapp/ext/dashboard/DashboardMetrics.js>
[state]: <../../../../../Desktop/Project/poc-samsung/03_Fiori_App/APL1.5_ZAPL_FINANCE_1.0.13_Partner_Source_r2_20261015/financial-close-dashboard/webapp/ext/dashboard/DashboardState.js>
[semantics]: <../../../../../Desktop/Project/poc-samsung/03_Fiori_App/APL1.5_ZAPL_FINANCE_1.0.13_Partner_Source_r2_20261015/financial-close-dashboard/webapp/ext/dashboard/TaskSemantics.js>
[timeline]: <../../../../../Desktop/Project/poc-samsung/03_Fiori_App/APL1.5_ZAPL_FINANCE_1.0.13_Partner_Source_r2_20261015/financial-close-dashboard/webapp/ext/dashboard/TimelineHelper.js>
[controller]: <../../../../../Desktop/Project/poc-samsung/03_Fiori_App/APL1.5_ZAPL_FINANCE_1.0.13_Partner_Source_r2_20261015/financial-close-dashboard/webapp/ext/dashboard/Dashboard.controller.js>
[view]: <../../../../../Desktop/Project/poc-samsung/03_Fiori_App/APL1.5_ZAPL_FINANCE_1.0.13_Partner_Source_r2_20261015/financial-close-dashboard/webapp/ext/dashboard/Dashboard.view.xml>
[tasklist]: <../../../../../Desktop/Project/poc-samsung/03_Fiori_App/APL1.5_ZAPL_FINANCE_1.0.13_Partner_Source_r2_20261015/financial-close-dashboard/webapp/ext/tasklist/TaskList.controller.js>
[timeline-view]: <../../../../../Desktop/Project/poc-samsung/03_Fiori_App/APL1.5_ZAPL_FINANCE_1.0.13_Partner_Source_r2_20261015/financial-close-dashboard/webapp/ext/dashboard/Timeline.fragment.xml>
[manifest]: <../../../../../Desktop/Project/poc-samsung/03_Fiori_App/APL1.5_ZAPL_FINANCE_1.0.13_Partner_Source_r2_20261015/financial-close-dashboard/webapp/manifest.json>
[task-annotation]: <../../../../../Desktop/Project/poc-samsung/02_RAP/ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Metadata Extensions/ZME_CLS_TASKINST/zme_cls_taskinst.ddlx.acds>
[behavior]: <../../../../../Desktop/Project/poc-samsung/02_RAP/ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Source Code Library/Classes/ZBP_I_CLS_CYCLE/zbp_i_cls_cycle.clas.implementations.abap>
[taskinst-cds]: <../../../../../Desktop/Project/poc-samsung/02_RAP/ZAPL_FINANCE/ZAPL_FINANCE/ZAPL_FI_CLOSE/Core Data Services/Data Definitions/ZI_CLS_TASKINST/zi_cls_taskinst.ddls.acds>
[fiori-guide]: <../80_산출물/01_인수원본/APL_Financial_Close_Partner_Visual_Guides_20261111/APL_FINANCE_1.0.13_Fiori_OData_CDS_Data_Mapping_Partner_Guide_20261111.md>
[joule-guide]: <../80_산출물/01_인수원본/APL_Financial_Close_Partner_Visual_Guides_20261111/APL_Financial_Close_Skill_MCP_S4_End_to_End_Partner_Guide_v2_20261111.md>
