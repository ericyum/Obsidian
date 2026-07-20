# OTT Module 4·5 — 프론트엔드 작업 가이드 (백엔드 기준)

> **작성일**: 2026-06-17
> **백엔드 담당**: 나영일 (기능4 마이페이지 + 기능5 관리자·정산)
> **대상**: 이 백엔드로 프론트엔드를 구현할 동료
> **목적**: 어떤 화면을 만들고, 어떤 기능을 넣고, 어떤 API를 호출할지 정리

---

## 0. 백엔드 연결 기본

| 항목                | 값                                                                                    |
| ----------------- | ------------------------------------------------------------------------------------ |
| 프로토콜              | **OData V4** (`sap.ui.model.odata.v4.ODataModel`)                                    |
| Base 경로           | `/odata/v4/ott/Mypage` (기능4) · `/odata/v4/ott/Admin` (기능5)                           |
| 인증                | Mypage = `authenticated-user` **+ 본인 스코프**(로그인 사용자 = 데이터 주인) / Admin = **`Admin` 롤** |
| 개발 테스트 계정 (비번 빈칸) | `admin1`(Admin 롤) / `U001`~`U010`(구독 회원) / `U011`(미구독) / `user1`·`user2`(일반)         |
| 핵심 원칙             | **액션에 user_id를 보내지 않음** — 로그인 사용자 기준으로 자동 처리                                         |

> 로컬에서 백엔드만 단독 기동: `npm run ott` → `http://localhost:8083`. 동작 확인용 `cap-node/ott/test.http`(REST Client) 참고.

---

## 1. 마이페이지 (기능4) — 회원용, 본인 데이터만

### MY-01. 내 멤버십 (홈)

- **목적**: 로그인 회원이 자기 구독 상태를 조회
- **데이터**: `GET /Mypage/UserSet?$expand=plan` → **본인 1행만** 반환 (user_id 몰라도 됨)
- **화면 구성**: 멤버십 카드
  - 이름(`user_name`) / 플랜명(`plan/name`) / 월 요금(`plan/price`) / 동시시청(`plan/maxStreams`) / 상태 배지(`subscription_status`) / 기간(`subscription_start_date`~`subscription_end_date`)
- **상태별 버튼 분기 (핵심 로직)**:

  | `subscription_status` | 노출 버튼 |
  |---|---|
  | `active` | **플랜 변경**(→MY-02) · **해지**(→MY-03) |
  | `cancelled` / `suspended` | **재개**(→MY-03) |
  | `none` (미구독) | **구독 시작**(→MY-02) |

### MY-02. 플랜 목록 · 구독/변경

- **목적**: 플랜 비교 후 신규 구독 또는 변경
- **데이터**: `GET /Mypage/PlanSet` → 플랜 3개(BASIC/STANDARD/PREMIUM) 카드 (가격·동시시청수 비교)
- **사용자 동작 → API**:

  | 상황 | 버튼 | 호출 |
  |---|---|---|
  | 미구독자(none) | "구독 시작" | `POST /Mypage/subscribe` `{ "plan": "BASIC" }` |
  | 기존 구독자(active) | "이 플랜으로 변경" | `POST /Mypage/changePlan` `{ "plan": "PREMIUM" }` |

- 현재 플랜은 카드에 하이라이트 (MY-01에서 받은 `subscription_plan` 재사용 — 추가 호출 불필요)

### MY-03. 해지 / 재개 (확인 모달)

- **목적**: 해지·재개 의사 확인 후 실행
- **동작 → API** (둘 다 body 없음):
  - 해지 확정 → `POST /Mypage/cancel`
  - 재개 확정 → `POST /Mypage/resume`
- 완료 후 MY-01 재조회로 상태 배지 갱신

### 에러 처리 (서버가 한글 메시지 반환 → 토스트로 그대로 표시)

| 상황 | 응답 |
|---|---|
| 미구독 아닌데 subscribe | 400 "미구독(none) 상태에서만 신규 구독이 가능합니다" |
| active 아닌데 changePlan | 400 "active 상태에서만 플랜 변경이 가능합니다" |
| 같은 플랜으로 changePlan | 400 "이미 PREMIUM 플랜을 사용 중입니다" |
| 없는 플랜 코드 | 400 "존재하지 않는 플랜입니다" |
| 이미 해지/active 인데 cancel/resume | 400 |

---

## 2. 관리자 · 정산 (기능5) — 관리자(`Admin` 롤)용

### AD-01. 정산 실행 · 조회 (메인)

- **목적**: 월 정산 실행 + 결과 확인
- **사용자 동작 → API**:

  | 동작 | 호출 | 결과 |
  |---|---|---|
  | 월 선택 후 "정산 실행" | `POST /Admin/runSettlement` `{ "month": "2026-03" }` | "2026-03 정산 완료: 파트너 3곳, 총 지급액 73,830원" (멱등 — 재실행 OK) |
  | 결과 조회 (헤더+디테일) | `GET /Admin/SettlementRunSet('2026-03')?$expand=items($expand=partner($select=name))` | 헤더(월 매출·전체시청) + 파트너별 행 |
  | 월별 이력/합계 | `GET /Admin/SettlementByMonth?$orderby=month desc` | 월·매출·지급합계·파트너수 |

- **화면 구성**:
  - 상단: 월 선택(Select/DatePicker) + "정산 실행" 버튼
  - **요약(헤더)**: 월 매출(`totalRevenue`), 전체 시청(`totalWatchSeconds`)
  - **테이블(items)**: 파트너명(`partner/name`) · 시청시간(`partnerWatchSeconds`) · 배분매출(`sharedRevenue`) · 분배율(`rsRate`) · **지급액(`settlementAmount`)**

- ⚠️ **꼭 주의**: 월 매출은 **헤더(`SettlementRuns`)의 `totalRevenue` 1개**를 사용. 디테일(`Settlements`) 행에는 매출 컬럼이 없음(헤더/디테일 분리 구조) → **디테일을 sum하면 안 됨.**

### AD-02. 파트너 관리 

- **데이터**: `GET /Admin/PartnerSet` → CP 목록·분배율
- **동작**: 분배율 수정 `PATCH /Admin/PartnerSet('P001')` `{ "rsRate": 0.75 }` → **다음 정산부터 적용**(과거 정산은 스냅샷이라 불변)

---

## 3. 화면 레이아웃 (와이어프레임)

각 화면의 대략적 배치와 요소가 어떤 데이터/API에 묶이는지.
### MY-01. 내 멤버십 (홈)

![[Pasted image 20260618085610.png]]

```
┌────────────────────────────────────────────┐
│ 👤 마이페이지                               │  ← 상단 바
├────────────────────────────────────────────┤
│ 김민준 님                     [ 구독중·active ]│  ← user_name + 상태 배지(status)
│                                              │
│ ┌ 플랜 ──┐ ┌ 월 요금 ┐ ┌ 동시시청 ┐ ┌ 기간 ─┐ │  ← plan/name·price·maxStreams + 기간
│ │PREMIUM │ │17,900원 │ │   4    │ │~26-01 │ │
│ └────────┘ └────────┘ └────────┘ └───────┘ │
│                                              │
│ [ 플랜 변경 ]   [ 해지 ]                      │  ← 상태별로 버튼이 바뀜(아래)
└────────────────────────────────────────────┘
   상태별 버튼:  active → 변경·해지  /  cancelled·suspended → 재개  /  none → 구독 시작
```
- 카드 전체 = `UserSet?$expand=plan`의 본인 1행에 바인딩. 버튼 노출은 `subscription_status`로 분기.

### MY-02. 플랜 목록 · 구독/변경

![[Pasted image 20260618085654.png]]

```
┌─ BASIC ───┐  ┌─ STANDARD ─┐  ╔═ PREMIUM ══╗   ← 현재 플랜은 강조 + "현재 플랜" 배지
│ 9,900원/월 │  │ 13,900원/월 │  ║ [현재 플랜] ║
│ 동시시청 1 │  │ 동시시청 2  │  ║ 17,900원/월 ║
│ [  선택  ] │  │ [  선택  ]  │  ║ 동시시청 4  ║
└───────────┘  └────────────┘  ║ [ 사용 중 ]  ║
                                ╚════════════╝
   미구독자: "구독 시작" → subscribe   /   기존 구독자: "선택" → changePlan
```
- 카드 3장 = `PlanSet` 리스트 바인딩. 현재 플랜은 MY-01의 `subscription_plan`과 비교해 강조.

### MY-03. 해지 / 재개 (확인 모달)

![[Pasted image 20260618085733.png]]

```
            ┌──────────────────────────────┐
            │              ⚠                │
            │   구독을 해지하시겠어요?       │
            │   재개로 다시 이용할 수 있어요 │
            │                              │
            │   [ 취소 ]      [ 해지 ]      │
            └──────────────────────────────┘
   취소 → 닫기   /   해지 → POST cancel   ( 재개 모달이면 → POST resume )
```
- MY-01 위에 뜨는 오버레이. 확정 후 MY-01 재조회로 상태 배지 갱신.

### AD-01. 정산 관리 (관리자)

![[Pasted image 20260618085801.png]]

```
┌────────────────────────────────────────────┐
│ 🧮 정산 관리                                │
├────────────────────────────────────────────┤
│ [ 2026-03 ▼ ]   [ ▶ 정산 실행 ]              │  ← 월 선택 + runSettlement
│                                              │
│ ┌ 월 매출 ────┐   ┌ 전체 시청 ──┐            │  ← 요약 = SettlementRuns 헤더값
│ │ 111,200원  │   │  56,100초   │            │
│ └────────────┘   └────────────┘             │
│                                              │
│ 파트너         시청(초)   분배율    지급액     │  ← 테이블 = items(파트너별 디테일)
│ Studio Alpha   34,140    70%    47,369원     │
│ Studio Beta    18,480    60%    21,978원     │
│ Studio Gamma    3,480    65%     4,483원     │
└────────────────────────────────────────────┘
   매출은 헤더값 1개 사용(디테일 sum 금지)  ·  회원 계정 호출 시 403
```
- 요약 = `SettlementRunSet('{월}')`의 헤더, 테이블 = 그 `items`($expand). 실행은 `runSettlement`.

### AD-02. 파트너 관리

![[Pasted image 20260618085818.png]]

```
┌────────────────────────────────────────────┐
│ 🤝 파트너 관리                              │
├────────────────────────────────────────────┤
│ 파트너          분배율(rsRate)      동작      │
│ Studio Alpha    [ 0.70 ]         [ 저장 ]    │  ← 인라인 편집
│ Studio Beta     [ 0.60 ]         [ 저장 ]    │
│ Studio Gamma    [ 0.65 ]         [ 저장 ]    │
└────────────────────────────────────────────┘
   분배율 수정 → PATCH PartnerSet('P001')  ·  다음 정산부터 적용(과거 정산 불변)
```
- 목록 = `PartnerSet`. rsRate 인라인 수정 후 저장 → `PATCH`. (정산 입력값 관리 화면)

---

## 4. 프론트가 알아야 할 도메인 규칙

| 규칙 | 의미 |
|---|---|
| **구독 상태 4종** | `active` / `cancelled` / `suspended` / `none` → MY-01 버튼 분기 기준 |
| **액션은 본인만** | user_id를 안 보냄. 로그인 사용자가 곧 대상 → "다른 사용자 선택" UI 없음 |
| **Admin은 롤 필요** | 회원 계정으로 Admin API 호출 시 403 → 관리자 화면은 권한 분기 |
| **정산 매출 표시** | 헤더 `totalRevenue` 사용 (디테일 sum 금지) |
| **결제 없음** | subscribe 시 즉시 active (결제 단계 화면 불필요), 구독기간 자동(오늘+1년) |

---

## 5. 개발 테스트 시나리오 (mocked 계정)

```
U011 로그인 → MY-01에 "구독 시작" 노출 → subscribe(BASIC) → active 전환 확인
U001 로그인 → 플랜 변경 / 해지 / 재개 흐름
admin1 로그인 → AD-01 정산 실행 → 결과 테이블
              (Studio Alpha 47,369 / Beta 21,978 / Gamma 4,483원)
U001 로 Admin API 호출 → 403 (권한 분기 확인)
```

---

## 6. 화면 ↔ API 한눈 요약

| 화면 | 조회(GET) | 실행(POST/PATCH) |
|---|---|---|
| MY-01 멤버십 | `/Mypage/UserSet?$expand=plan` | — |
| MY-02 플랜 | `/Mypage/PlanSet` | `/Mypage/subscribe` 또는 `/Mypage/changePlan` |
| MY-03 해지/재개 | — | `/Mypage/cancel`, `/Mypage/resume` |
| AD-01 정산 | `/Admin/SettlementRunSet('{월}')?$expand=items`, `/Admin/SettlementByMonth` | `/Admin/runSettlement` |
| AD-02 파트너 | `/Admin/PartnerSet` | `PATCH /Admin/PartnerSet('{id}')` |

> 참고 문서: 백엔드 아키텍처는 `02_개발단계/Module4_5_서비스_아키텍처_및_커스텀로직.md`, 엔티티는 `01_계획단계/OTT-통합-Entity설계 06-15갱신.md`.
