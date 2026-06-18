# OTT 프로젝트 데이터 정의 (백엔드: 기능4 마이페이지 + 기능5 관리자/정산)

  

본 문서는 `cap-node/ott`의 데이터 모델을 정의한다. 팀 공용 `DATA_SPEC.md`(사용자/컨텐츠/시청기록 3개 테이블)를 그대로 채택하고, 기능4·5에 필요한 3개 테이블을 추가하여 **총 6개 테이블**로 구성한다.

  

- 스키마 파일: `cap-node/ott/db/cds/*-model.cds` (namespace `com.cap.ott`, 엔티티당 1파일)

- 시드 데이터: `cap-node/ott/db/data/com.cap.ott-*.csv` (네임스페이스 접두사 필수)

- DB: 개발 = SQLite in-memory(시드 자동 적재), 운영 = HANA (`.cdsrc.json` 프로파일)

- 인코딩: 한글 값이 포함된 CSV(`com.cap.ott-SubscriptionPlans.csv`)는 **UTF-8 BOM** — 엑셀에서 바로 열어도 깨지지 않음

  

---

  

## 1. 사용자 테이블 (`com.cap.ott-Users.csv`) — 팀 공용

  

회원과 멤버십 상태를 한 행에 담는다(평탄화). 마이페이지(기능4)의 본체이자, 정산(기능5)의 매출 산출 기초.

**팀 공용 스펙과 동일** — 컬럼·CSV 헤더 변경 없음.

  

| 컬럼명 | 타입 | 설명 | 비고 |

|---|---|---|---|

| `user_id` | String(10) | 사용자 고유 식별자 | PK (예: U0001) |

| `user_name` | String(100) | 사용자명 | 화면 표시용 |

| `subscription_plan` | String(20) | 구독 플랜 코드 | `BASIC` / `STANDARD` / `PREMIUM` → `SubscriptionPlans.code` 참조 |

| `subscription_status` | String(20) | 구독 상태 | `active` / `cancelled` / `suspended` |

| `subscription_start_date` | Date | 구독 시작일 | YYYY-MM-DD |

| `subscription_end_date` | Date | 구독 종료일 | YYYY-MM-DD |

  

**주요 활용**:

- 마이페이지 멤버십 조회·관리: `changePlan` / `cancel` / `resume` 액션이 이 테이블의 구독 컬럼을 갱신한다.

- 정산 월 매출 산출: `subscription_status = 'active'` 사용자의 플랜 가격 합계.

  

---

  

## 2. 컨텐츠 테이블 (`com.cap.ott-Contents.csv`) — 팀 공용 + 확장

  

컨텐츠 카탈로그. 정산에서 시청시간을 파트너(CP)별로 묶는 배분 단위.

팀 공용 스펙에 **`partner_partner_id` 컬럼 1개를 확장**했다 (정산용 CP 매핑, 팀 합의 필요 사항).

  

| 컬럼명 | 타입 | 설명 | 비고 |

|---|---|---|---|

| `content_id` | String(10) | 컨텐츠 고유 식별자 | PK (예: C0001) |

| `title` | String(200) | 컨텐츠 제목 | |

| `genre` | String(100) | 장르 | 복수일 경우 `/` 구분 |

| `tags` | String(500) | 태그 | 파이프(`\|`) 구분 |

| `content_type` | String(20) | 컨텐츠 유형 | `MOVIE` / `SERIES` / `FEATURE` / `ANIMATION` |

| `avg_rating` | Decimal(2,1) | 평균 평점 | 0.0 ~ 5.0 |

| `rating_count` | Integer | 평가 수 | |

| `partner_partner_id` | String(10) | 콘텐츠 제공자(CP) | **확장 컬럼**, FK → `Partners.partner_id` (예: P0001) |

  

**주요 활용**:

- 정산: `partner_partner_id`로 시청기록을 CP별로 집계한다 (파트너 미지정 콘텐츠는 배분 제외).

- 메인/상세 페이지(기능1·2)의 활용은 팀 공용 스펙과 동일.

  

---

  

## 3. 시청 기록 테이블 (`com.cap.ott-ViewingHistory.csv`) — 팀 공용 + 형식 조정

  

사용자-컨텐츠 간 N:M 교차 테이블이자 **정산 배분의 기준**(시청 초).

팀 공용 스펙과 데이터는 동일하며, CAP 적재를 위해 **형식 2가지만 조정**했다:

① FK 헤더 `user_id` → `user_user_id`, `content_id` → `content_content_id` (CAP FK 명명 규칙)

② `watch_datetime`을 ISO 형식으로 (`2026-03-10 20:15:00` → `2026-03-10T20:15:00Z`)

  

| 컬럼명 | 타입 | 설명 | 비고 |

|---|---|---|---|

| `history_id` | String(10) | 기록 고유 식별자 | PK (예: H00001) |

| `user_user_id` | String(10) | 사용자 ID | FK → `Users.user_id` |

| `content_content_id` | String(10) | 컨텐츠 ID | FK → `Contents.content_id` |

| `watch_datetime` | DateTime | 시청 일시 | ISO 8601, 정산 월('YYYY-MM')을 여기서 파생 |

| `watch_duration_seconds` | Integer | 실제 시청 시간(초) | **정산 배분 기준** |

| `completion_percentage` | Decimal(5,2) | 시청 완료율(%) | 0.00 ~ 100.00 |

| `device_type` | String(20) | 시청 기기 | `SmartTV` / `Smartphone` / `Tablet` / `PC` |

  

**주요 활용**:

- 정산: `watch_datetime`의 월별로 `watch_duration_seconds`를 파트너별 합산 → 매출 배분 비율 산출.

- 구독자 추이 분석(기능3)의 활용은 팀 공용 스펙과 동일.

  

---

  

## 4. 구독 플랜 테이블 (`com.cap.ott-SubscriptionPlans.csv`) — 신규

  

플랜 카탈로그. 마이페이지 "구독 플랜 관리" 화면의 본체이며, 정산의 매출 단가이기도 하다.

  

| 컬럼명 | 타입 | 설명 | 비고 |

|---|---|---|---|

| `code` | String(20) | 플랜 코드 | PK, `BASIC` / `STANDARD` / `PREMIUM` — `Users.subscription_plan`과 조인 |

| `name` | String(100) | 플랜 표시명 | 베이직 / 스탠다드 / 프리미엄 |

| `price` | Integer | 월 구독료(원) | 9,900 / 13,900 / 17,900 — **정산 매출 계산에 사용** |

| `maxStreams` | Integer | 동시 시청 수 | 1 / 2 / 4 — 플랜 비교 표시용 (강제 로직 없음) |

| (+ `createdAt`/`createdBy`/`modifiedAt`/`modifiedBy`) | | CAP `managed` 감사 필드 | 자동 관리 |

  

**주요 활용**:

- 플랜 목록/비교 화면, `changePlan` 시 유효 플랜 검증.

- 정산 월 매출 = Σ(active 사용자의 `price`).

  

---

  

## 5. 파트너 테이블 (`com.cap.ott-Partners.csv`) — 신규

  

콘텐츠 제공자(CP) 마스터. RS(Revenue Share) 정산의 분배율을 보유한다.

  

| 컬럼명 | 타입 | 설명 | 비고 |

|---|---|---|---|

| `partner_id` | String(10) | 파트너 고유 식별자 | PK (예: P0001 — 다른 테이블 키 스타일과 통일) |

| `name` | String(100) | 파트너명 | Studio Alpha / Beta / Gamma |

| `rsRate` | Decimal(5,4) | 분배율 | 0.0000 ~ 1.0000 (예: 0.70 = 70%) |

| (+ managed 감사 필드) | | | 자동 관리 |

  

**주요 활용**:

- 정산 실행 시 `rsRate`를 읽어 지급액 계산. 계산 시점의 값이 `Settlements`에 스냅샷 저장된다.

  

---

  

## 6. 정산 결과 테이블 (`Settlements`) — 신규, 시드 없음

  

월별·파트너별 RS 정산 결과. **정산 실행(`POST /odata/v4/ott/Admin/runSettlement`)으로만 생성**되며, 같은 월 재실행 시 삭제 후 재생성(멱등).

  

| 컬럼명 | 타입 | 설명 | 비고 |

|---|---|---|---|

| `ID` | UUID | 정산 고유 식별자 | PK (CAP `cuid`) |

| `month` | String(7) | 정산 월 | 'YYYY-MM' |

| `partner_partner_id` | String(10) | 파트너 | FK → `Partners.partner_id` |

| `totalRevenue` | Integer | 월 매출 스냅샷(원) | Σ(active 사용자 × 플랜 가격) — 파트너마다 중복 저장이므로 sum 금지 |

| `totalWatchSeconds` | Integer64 | 월 전체 시청(초) | |

| `partnerWatchSeconds` | Integer64 | 해당 파트너 시청(초) | |

| `rsRate` | Decimal(5,4) | 분배율 스냅샷 | 이후 `Partners.rsRate`가 바뀌어도 과거 정산 불변 |

| `sharedRevenue` | Integer | 배분된 매출(원) | totalRevenue × (파트너 시청 / 전체 시청), 원 단위 버림 |

| `settlementAmount` | Integer | 최종 지급액(원) | sharedRevenue × rsRate, 원 단위 버림 |

| (+ managed 감사 필드) | | | 자동 관리 |

  

**주요 활용**:

- 관리자 정산 화면: 월별/파트너별 지급액 조회 (`GET /odata/v4/ott/Admin/SettlementSet`).

  

---

  

## 테이블 간 관계

  

> 시각화 버전: [docs/erd.svg](docs/erd.svg)

  

```

                          (팀 공용 3개)

Users (1) ─────< (N) ViewingHistory (N) >───── (1) Contents

  │                                                  │

  │ (N)                                              │ (N)

  ▼                                                  ▼

SubscriptionPlans (1)                          Partners (1)

  · 코드(subscription_plan)로 조인                  · 분배율(rsRate) 보유

  · 가격 = 정산 매출 단가                            │

                                                     ▼ (1)─<(N)

                                               Settlements

                                                 · 월별·파트너별 정산 결과

                                                 · rsRate/매출 스냅샷 보존

```

  

- `ViewingHistory`는 `Users`와 `Contents`의 **N:M 교차 테이블** (팀 공용 스펙과 동일).

- `Users → SubscriptionPlans`: UUID가 아닌 **플랜 코드**로 연결된다 (`subscription_plan` = `code`).

- `Contents → Partners`: 정산을 위한 확장 관계. 하나의 파트너가 여러 콘텐츠를 보유.

- `Settlements`는 실행 시점의 계산 결과를 **스냅샷**으로 저장하므로, 원본(Users/Plans/Partners)이 변해도 과거 정산은 변하지 않는다.

  

## 정산 공식 (참고)

  

```

월 매출          = Σ (subscription_status='active' 사용자의 plan.price)

sharedRevenue    = 월 매출 × (파트너 시청시간 / 전체 시청시간)

settlementAmount = sharedRevenue × rsRate          ※ 반올림은 원 단위 버림(floor)

```

  

시드 기준 2026-03 기대값: 매출 111,200원, 전체 시청 56,100초

→ Studio Alpha **47,369원** · Studio Beta **21,978원** · Studio Gamma **4,483원**