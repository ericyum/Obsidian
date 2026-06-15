# 08 - CI/CD 및 배포 가이드

> **파이프라인 설정:** `.pipeline/config.yml`  
> **가이드 문서:** `CICD_GUIDE.txt`  
> **빌드 도구:** MBT (Cloud MTA Build Tool) + Node.js 22  
> **CI/CD 서비스:** SAP Continuous Integration and Delivery

---

## 1. CI/CD 파이프라인 개요

```
GitHub Push → Webhook → SAP CI/CD Service → Build → Malware Scan → Deploy to CF
```

**설정 파일:** `.pipeline/config.yml`

```yaml
general:
  buildTool: "mta"
  productiveBranch: "master"        # master 브랜치만 배포

service:
  buildToolVersion: "MBTJ11N22"     # Node.js 22 + MTA Build Tool

stages:
  Build:                            # MTA 빌드 (mbt build)
  Additional Unit Tests:            # 현재 비활성화
  Malware Scan:                     # 자동 수행
  Acceptance:                       # 현재 비활성화 (E2E 테스트)
  Compliance:                       # 현재 비활성화 (SonarQube)
  Release:                          # CF 배포
    cloudFoundryDeploy: true
    cfApiEndpoint: "https://api.cf.jp10.hana.ondemand.com"
    cfCredentialsId: "cf-credentials"
    mtaDeployParameters: "-f --version-rule ALL"
```

**현재 파이프라인의 상태:**
- ✅ Build: 활성화 (mbt build)
- ❌ Lint: 비활성화 (npmExecuteLint: false)
- ❌ Unit Test: 비활성화 (npmExecuteScripts: false)
- ✅ Malware Scan: 자동
- ❌ E2E Test: 비활성화
- ❌ SonarQube: 비활성화 (sonarExecuteScan: false)
- ✅ CF Deploy: 활성화 (master 브랜치만)

---

## 2. 빌드 프로세스

### CAP 모듈 빌드 (`build:cf`)
```bash
# 1. CDS 소스 컴파일 (CDS → HANA SQL + OData EDMX + TypeScript)
npm run build:cds       # cds build --production

# 2. 생성된 원본 TypeScript 정리
npm run cleanup:ts      # rimraf gen/srv/srv/**/*.ts

# 3. TypeScript → JavaScript 컴파일
npm run build:ts        # tsc -p tsconfig.cdsbuild.json

# 결과물: gen/srv/ (배포 가능한 Node.js 앱)
#         gen/db/  (HANA DB deployer)
```

### UI5 모듈 빌드 (`build:ui`)
```bash
npm run build:ui        # ui5 build preload --clean-dest

# 결과물: dist/ (Component-preload.js 포함)
```

### MTA 빌드 (`build:mbt`)
```bash
mbt build               # mta.yaml 기반 .mtar 패키지 생성

# 결과물: mta_archives/<module>_<version>.mtar
```

---

## 3. 배포 순서 (9개 Job)

| 순서 | Job 이름 | MTA 경로 | 설명 |
|---|---|---|---|
| 1 | deploy-cap-core | `cap-node/core/mta.yaml` | 공통 핵심 서비스 |
| 2 | deploy-cap-mdm | `cap-node/mdm/mta.yaml` | MDM 서비스 |
| 3 | deploy-cap-plan | `cap-node/plan/mta.yaml` | 계획 서비스 |
| 4 | deploy-library | `cap-app/library/mta.yaml` | 공통 UI5 라이브러리 |
| 5 | deploy-portal | `cap-app/portal/mta.yaml` | 포털 UI |
| 5 | deploy-mdm-ui | `cap-app/mdm/mta.yaml` | MDM UI |
| 5 | deploy-sysmgt | `cap-app/sysmgt/mta.yaml` | 시스템 관리 UI |
| 5 | deploy-dp | `cap-app/dp/mta.yaml` | 수요계획 UI |
| 6 | deploy-approuter | `cap-app/approuter/mta.yaml` | AppRouter |

> 동일 순서(5)의 Job들은 서로 독립적이므로 동시에 실행 가능하다.

---

## 4. 크리덴셜(자격증명) 설정

SAP CI/CD Service의 **Credentials** 탭에서 등록:

| 이름 | 타입 | 내용 |
|---|---|---|
| `cf-credentials` | Basic Auth | CF 배포용 사용자 ID/비밀번호 (SpaceDeveloper 역할 필요) |
| `github-credentials` | Basic Auth | GitHub Personal Access Token |

---

## 5. 환경별 배포 전략

| 환경 | Branch | CF Space | 설명 |
|---|---|---|---|
| **Dev** | `develop` | dev | 개발자 Push 시 자동 배포 |
| **QA** | `release/qa` | qa | QA 테스트용 |
| **Prod** | `master` | prod | 수동 승인 후 운영 배포 |

**AppRouter 환경별 파일:**
- `xs-app.json` → 운영 (Production) 라우팅
- `xs-app-dev.json` → 개발 (Dev) 라우팅
- `xs-app-qa.json` → QA 라우팅

```bash
# 개발 환경 배포 (xs-app-dev.json → xs-app.json 복사 후 빌드)
npm run deploy:router    # → deploy:mta (dev)

# QA 환경 배포
npm run deploy:qarouter  # → deploy:qa
```

---

## 6. BTP 서비스 인스턴스 (사전 생성 필요)

| 서비스 | Plan | 이름 |
|---|---|---|
| HANA HDI Container | hdi-shared | `com-hdi-container-${space}` |
| XSUAA | application | `com-app-xsuaa-${space}` |
| Destination | lite | `com-app-destination-${space}` |
| Connectivity | lite | `com-app-connectivity-${space}` |
| HTML5 App Repo (Runtime) | app-runtime | `com-app-repo-runtime-${space}` |
| HTML5 App Repo (Host) | app-host | `com-welcome-repo-host-${space}` |

---

## 7. 주요 트러블슈팅

| 문제 | 원인 | 해결 |
|---|---|---|
| npm install 실패 | package-lock.json 불일치 | 로컬에서 `npm install` 후 lock 파일 커밋 |
| MBT Build 실패 | Node 버전 불일치 | `MBTJ11N22` 선택 (Node 22) |
| TypeScript 컴파일 오류 | 타입 에러 | `npx tsc --noEmit` 로컬 확인 |
| CF Deploy 실패 (할당량) | 메모리 초과 | Space Quota 증설 요청 |
| CF Deploy 실패 (서비스 없음) | 인스턴스 미생성 | `cf create-service` 수동 생성 |
| Approuter 404 | Destination 미등록 | BTP Cockpit → Destinations 확인 |

---

## 8. 향후 개선 항목 (로드맵)

### 단기 (우선 적용)
- [ ] ESLint 설정 (`npmExecuteLint: true`)
- [ ] 단위 테스트 (Jest/Mocha) 및 `npmExecuteScripts: true` 활성화

### 중기
- [ ] E2E 테스트 (wdi5 등) 및 Acceptance 단계 활성화
- [ ] Branch 전략 도입 (develop → dev, master → prod)

### 장기
- [ ] SonarQube 연동 (`sonarExecuteScan: true`)
- [ ] TMS(Transport Management Service) 연동

---

## 9. 로컬 개발 vs 실제 배포 비교

| 항목 | 로컬 개발 | 실제 배포 |
|---|---|---|
| DB | SQLite (in-memory) | HANA (HDI Container) |
| 인증 | Mocked (`cds.requires.auth.kind: "mocked"`) | XSUAA (OAuth 2.0) |
| OData 어댑터 | v4 (직접) | v2 + v4 (adapter) |
| 파일 스토리지 | 로컬 | S3 (Object Store) |
| 메일 | 모의 | SAP Destination → SMTP |
| 환경변수 | `default-env.json` | VCAP_SERVICES (CF 자동 주입) |
