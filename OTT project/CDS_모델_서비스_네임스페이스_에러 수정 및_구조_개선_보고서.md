# CDS 모델·서비스 구조 개선 보고서

> **작성일**: 2026-06-18
> **대상**: ZEN-OTT 프로젝트 `cap-node/ott/`
> **범위**: db/cds/ 폴더 구조화, 서비스 파일명 변경, VS Code 언어 서버 오류 해결, TypeScript 패턴 수정

---

## 1. 변경 배경

### 1-1. VS Code CDS 언어 서버 오류 — 왜 런타임은 멀쩡한데 VS Code만 빨간줄인가?

**증상**: 서버는 잘 돌아가는데 VS Code 에디터에서 아래 같은 오류가 표시됨

```
FreeBoard-service.cds(15, 52): Artifact "com.cap.ott.Post" has not been found
```

**원인**: CDS에는 같은 파일을 읽는 두 가지 방식(컴파일러)이 존재하는데, 이 둘의 **파일 탐색 전략이 서로 다르기 때문**입니다.

#### 🔵 `cds watch` 런타임 — "책을 처음부터 순서대로 읽는" 방식

런타임은 루트 `index.cds`에서 출발해 `using from`을 만날 때마다 해당 파일로 점프하면서 **모든 파일을 하나의 연쇄 체인으로 읽어나갑니다**. 마치 책 목차를 1페이지부터 순서대로 읽듯이:

```
📄 index.cds (루트)
    ↓ using from './db'
📄 db/index.cds
    ↓ using from './cds/Core-model.cds'              ← Users, Contents 등이 여기서 정의됨
    ↓ using from './cds/board/Freeboard-model.cds'    ← Post, Comment 가 여기서 정의됨
    ↓ ...
    ↓ using from './srv'
📄 srv/index.cds
    ↓ using from './cds/board/FreeBoard-service'      ← Post, Comment 를 사용함
```

`FreeBoard-service.cds`를 읽는 시점에는 이미 `Freeboard-model.cds`가 앞에서 로드되어 `Post`, `Comment`가 "알고 있는" 상태이므로 아무 문제가 없습니다.

#### 🔴 VS Code CDS 언어 서버 — "파일 한 장만 뽑아서 검사하는" 방식

반면 VS Code에 내장된 CDS 언어 서버는 **지금 열려있는 파일 하나만 단독으로 검사**합니다. `index.cds` 체인을 타고 올라가서 앞뒤 문맥을 파악하지 않습니다. 사전에서 단어 하나만 뽑아놓고 "이 단어, 우리 사전에 정의된 적 없는데요?"라고 따지는 것과 같습니다.

```
📄 FreeBoard-service.cds (이 파일만 단독 검사)

    service com.cap.ott.FreeBoard {
        entity PostSet    as projection on com.cap.ott.Post;     ← ❓ "Post"? 들어본 적 없는데?
        entity CommentSet as projection on com.cap.ott.Comment;  ← ❓ "Comment"? 들어본 적 없는데?
    }
```

이 파일 안에는 `Post`나 `Comment`가 어디에 정의됐는지 알려주는 `using from`이 없기 때문에 **"Artifact has not been found"** 오류가 발생합니다.

#### 영향을 받은 모든 cross-file 참조

같은 원리로 다음 엔티티들도 각각의 파일에서 동일한 오류가 발생했습니다:

`Post`, `Comment`, `Users`, `SubscriptionPlans`, `Partners`, `Contents`, `SubscriberGroup`

#### 해결: 각 파일에 "이거 어디 있는지" 직접 알려주기

각 파일 상단에 `using from`을 추가하여, 언어 서버가 단독 검사할 때도 참조 대상을 찾을 수 있게 했습니다 (→ [2-4절](#2-4-vs-code-언어-서버-오류-해결--using-from-지시문-추가) 참조).

> 💡 **핵심 교훈**: `cds watch`가 성공한다고 CDS 문법이 완벽한 것은 아닙니다. VS Code 언어 서버는 더 엄격하게 각 파일의 **자급자족성**(self-contained)을 요구합니다. 모든 cross-file 참조는 해당 파일 안에 `using from`으로 명시하는 습관을 들이세요.

### 1-2. 유지보수성 개선

- `db/cds/`에 기능별 폴더(`trend/`, `board/`)를 만들어 `srv/cds/`와 대칭 구조로 개선
- 서비스 파일명이 연계된 db 모델 파일명을 유추할 수 있도록 `AnomalyAlert-service.cds` → `TrendAnomaly-service.cds`로 변경

---

## 2. 변경 상세

### 2-1. db/cds/ 폴더 구조화

```
Before                              After
db/cds/                             db/cds/
├── Core-model.cds                  ├── Core-model.cds           (기능1,2 공통)
├── Trend-model.cds                 ├── MypageAdmin-model.cds    (기능4,5)
├── MypageAdmin-model.cds           ├── trend/
├── Freeboard-model.cds             │   └── Trend-model.cds      (기능3)
                                    └── board/
                                        └── Freeboard-model.cds  (기능6)
```

- `db/cds/Trend-model.cds` → `db/cds/trend/Trend-model.cds`
- `db/cds/Freeboard-model.cds` → `db/cds/board/Freeboard-model.cds`

### 2-2. 서비스 파일명 변경

| Before | After | 사유 |
|--------|-------|------|
| `srv/cds/trend/AnomalyAlert-service.cds` | `srv/cds/trend/TrendAnomaly-service.cds` | 연계된 db 모델 `Trend-model.cds`의 키워드 "Trend" 포함 → 파일명만으로 db↔srv 매칭 가능 |

### 2-3. 서브디렉토리 간 CDS 참조 규칙 (중요 발견)

`db/cds/` 아래 서브디렉토리(예: `board/`, `trend/`)에 있는 `.cds` 파일이 **다른 디렉토리**의 엔티티를 참조할 때는 `ott.` prefix가 필요합니다.

```
db/cds/
├── Core-model.cds          ← entity Users { ... }
├── board/
│   └── Freeboard-model.cds ← Association to ott.Users  ← ott. prefix 필수!
```

- `ott.Users`는 `namespace com.cap.ott;` 내에서 `com.cap.ott.Users`로 올바르게 해석됩니다.
- 동일 디렉토리 내 파일 간 참조나, 루트 `db/cds/`의 파일 간 참조는 prefix 불필요.
- Core-model.cds(`ott.SubscriptionPlans`), MypageAdmin-model.cds(`ott.Contents`)도 이 규칙에 따라 기존 prefix가 올바릅니다.

**→ `ott.` prefix는 제거하지 않고 그대로 유지합니다.**

### 2-4. VS Code 언어 서버 오류 해결 — `using from` 지시문 추가

VS Code CDS 언어 서버는 `index.cds`의 `using from` 체인을 따라가지 않으므로, **각 파일에 직접 `using from`을 명시**하여 cross-file 참조를 해결했습니다.

#### 모델 파일 (db/cds/)

| 파일 | 추가된 지시문 | 참조 대상 | 비고 |
|------|-------------|----------|------|
| `board/Freeboard-model.cds` | `using { Users } from '../Core-model';` | Core-model의 Users | 단방향 |
| `Core-model.cds` | `using from './MypageAdmin-model';` | MypageAdmin-model 전체 | 순환 참조 — 중괄호 없는 `using from` |
| `MypageAdmin-model.cds` | `using from './Core-model';` | Core-model 전체 | 순환 참조 — 중괄호 없는 `using from` |
| `trend/Trend-model.cds` | (불필요) | 외부 참조 없음 | — |

#### 서비스 파일 (srv/cds/)

| 파일 | 추가된 지시문 | 참조 대상 |
|------|-------------|----------|
| `board/FreeBoard-service.cds` | `using from '../../../db/cds/board/Freeboard-model';` | Freeboard-model |
| `trend/TrendAnalysis-service.cds` | `using from '../../../db/cds/trend/Trend-model';` | Trend-model |
| `trend/TrendAnomaly-service.cds` | `using from '../../../db/cds/trend/Trend-model';` | Trend-model |

> **순환 참조 해결**: `Core-model` ↔ `MypageAdmin-model`은 서로의 엔티티를 참조하는 순환 의존성입니다. 중괄호 없는 `using from './file'` 문법을 사용하여 CAP의 다중 패스(multi-pass) 해결 방식으로 처리했습니다.

### 2-5. `db/index.cds` 로딩 순서 최적화

Freeboard-model은 Core-model의 Users를 참조하므로, Core-model 바로 다음에 로드하도록 배치했습니다.

```cds
using from './cds/Core-model.cds';              // 1. 공통 마스터 (Users, Contents 등)
using from './cds/board/Freeboard-model.cds';    // 2. Core 바로 다음 (Users 참조)
using from './cds/trend/Trend-model.cds';        // 3. 독립적 (외부 참조 없음)
using from './cds/MypageAdmin-model.cds';        // 4. 순환 참조 (마지막)
```

### 2-6. `srv/index.cds` 갱신

```cds
// === Module 3: 구독자 추세 분석 ===
using from './cds/trend/TrendAnalysis-service';
using from './cds/trend/TrendAnomaly-service';   // ← 파일명 변경 반영
```

### 2-7. TypeScript `return req.error()` 패턴 수정 (6곳)

`return req.error(...)`가 TypeScript `Error` 타입을 반환하여 `void` 시그니처와 충돌하는 문제 해결.

**패턴**: `return req.error(...)` → `req.error(...); return;`

| 파일 | 위치 | 건수 |
|------|------|:--:|
| `FreeBoard.handler.ts` | `verifyPostAuthor` — 404, 403 | 2 |
| `FreeBoard.handler.ts` | `verifyCommentAuthor` — 404, 403 | 2 |
| `AnomalyAlert.handler.ts` | `markAsRead` — 404 | 1 |
| `TrendAnalysis.handler.ts` | `calculateTrends` — 400 | 1 |

---

## 3. 최종 프로젝트 구조

```
cap-node/ott/
├── index.cds                              (using from './db'; using from './srv';)
├── .cdsrc.json
│
├── db/
│   ├── index.cds                          ✅ 경로 갱신 + 로딩 순서 최적화
│   ├── cds/
│   │   ├── Core-model.cds                 ✅ using from './MypageAdmin-model'
│   │   ├── MypageAdmin-model.cds          ✅ using from './Core-model'
│   │   ├── trend/
│   │   │   └── Trend-model.cds            (기능3 · 이동)
│   │   └── board/
│   │       └── Freeboard-model.cds        ✅ using { Users } from '../Core-model' (기능6 · 이동)
│   └── data/                              (CSV 13종)
│
├── srv/
│   ├── index.cds                          ✅ TrendAnomaly-service로 갱신
│   ├── cds/
│   │   ├── trend/
│   │   │   ├── TrendAnalysis-service.cds  ✅ @(requires: 'Admin') + using from Trend-model
│   │   │   └── TrendAnomaly-service.cds   ✅ renamed + @(requires: 'Admin') + using from
│   │   └── board/
│   │       └── FreeBoard-service.cds      ✅ using from Freeboard-model
│   └── src/feature/
│       ├── trend/
│       │   ├── TrendAnalysis.handler.ts   ✅ (group_id fix + JOIN fix + req.error/return)
│       │   └── AnomalyAlert.handler.ts    ✅ (anomaly_id fix + req.error/return)
│       └── board/
│           └── FreeBoard.handler.ts       ✅ (req.error/return 분리 4곳)
│
└── test/
    ├── calculator.test.ts
    ├── Module36Test.http
    └── run-all-tests.mjs                  ✅ (55 cases)
```

---

## 4. 검증 결과

- ✅ `cds watch --port 8083` 정상 기동 (14개 CDS 파일, 7개 서비스)
- ✅ FreeBoard PostSet: 200/201, TrendAnalysis: 200, AnomalyAlert: 200
- ✅ **CDS 컴파일 에러 0건** (ERROR 없음, WARNING 없음)
- ✅ **VS Code 언어 서버 오류 0건** (모든 파일에 `using from` 명시)
- ✅ CSV 13종 in-memory DB 시딩 정상
- ✅ `db/index.cds` 로딩 순서 최적화
- ✅ `AnomalyAlert-service.cds` → `TrendAnomaly-service.cds` 파일명 변경 완료
- ✅ TypeScript `void` 타입 오류 0건 (`req.error`/`return` 분리 적용)

---

## 5. 수정 파일 총괄

| # | 파일 | 변경 내용 |
|---|------|----------|
| 1 | `db/cds/trend/Trend-model.cds` | **이동** (db/cds/ → db/cds/trend/) |
| 2 | `db/cds/board/Freeboard-model.cds` | **이동** + `using { Users } from '../Core-model';` |
| 3 | `db/cds/Core-model.cds` | `using from './MypageAdmin-model';` |
| 4 | `db/cds/MypageAdmin-model.cds` | `using from './Core-model';` |
| 5 | `db/index.cds` | 경로 갱신 + 로딩 순서 최적화 |
| 6 | `srv/cds/trend/TrendAnomaly-service.cds` | **파일명 변경** + `@(requires: 'Admin')` + `using from` |
| 7 | `srv/cds/trend/TrendAnalysis-service.cds` | `@(requires: 'Admin')` + `using from` |
| 8 | `srv/cds/board/FreeBoard-service.cds` | `using from` |
| 9 | `srv/index.cds` | `TrendAnomaly-service`로 참조 갱신 |
| 10 | `srv/src/feature/trend/TrendAnalysis.handler.ts` | 그룹ID 파싱 `SG_` prefix (3곳) + JOIN 쿼리 (2곳) + `req.error`/`return` 분리 |
| 11 | `srv/src/feature/trend/AnomalyAlert.handler.ts` | `ID`→`anomaly_id` (3곳) + `req.error`/`return` 분리 |
| 12 | `srv/src/feature/board/FreeBoard.handler.ts` | `req.error`/`return` 분리 (4곳: 404×2 + 403×2) |

> **참고**: Core-model, MypageAdmin-model, Freeboard-model의 `ott.` prefix는 서브디렉토리 간 참조에 필요한 올바른 CDS 문법이므로 **변경하지 않음**.
