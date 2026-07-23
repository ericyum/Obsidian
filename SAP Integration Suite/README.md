# SAP Integration Suite 가이드

SAP 직원이 업무 관점에서 빠르게 판단할 수 있도록 만든 입문 자료입니다. 제품 기능은 SAP 공식 Feature Scope Description(FSD, 2025-09-04)와 Help Portal을 우선으로 검토했고, 가격과 플랜은 2026-07-21에 확인한 SAP 공식 가격 페이지를 기준으로 합니다.

> **먼저 알아둘 점**: 이 문서는 설계와 판단을 돕는 자료입니다. 실제 사용 가능 기능, 무료 체험, 단가는 계약, 리전, 계정 상태에 따라 달라질 수 있으므로 구매 또는 운영 전에는 SAP Discovery Center와 SAP 담당자에게 확인해야 합니다.

## 권장 읽기 경로

### 1. 도입 판단 경로

| 순서 | 문서 | 답하는 질문 |
|---|---|---|
| 1 | [1. 정의](1.%20Integration%20Suite%20정의.md) | IS는 무엇이며, 무엇을 대체하지 않는가? |
| 2 | [2. 기능](2.%20Integration%20Suite%20기능.md) | 어떤 통합 방식과 capability가 있는가? |
| 3 | [3. 사용 이유](3.%20Integration%20Suite%20사용%20이유.md) | 어떤 현장 문제가 생기면 IS를 고려하는가? |
| 4 | [4. S/4HANA OData 직접 연계와 IS 비교](4.%20S4HANA%20OData%20%EC%A7%81%EC%A0%91%20%EC%97%B0%EA%B3%84%EC%99%80%20Integration%20Suite%20%EB%B9%84%EA%B5%90.md) | 내 Fiori/S/4HANA 구조에 중간 통합 계층이 필요한가? |
| 5 | [5. 대체 가능성](5.%20Integration%20Suite%20대체%20가능성.md) | 직접 연계·기존 플랫폼·전문 서비스와 비교하면 어떤가? |
| 6 | [6. 가격과 플랜](6.%20Integration%20Suite%20가격과%20플랜.md) | 플랜별 기능과 공개 목록가는 얼마인가? |
| 6-1 | [시나리오별 비용 비교와 TCO 판단](6-1.%20시나리오별%20비용%20비교와%20TCO%20판단.md) | 우리 개발·운영비와 비교하면 어느 방식이 타당한가? |
| 7 | [7. 무료 체험과 도입](7.%20무료%20체험과%20도입%20가이드.md) | 작은 PoC를 어떻게 안전하게 시작하는가? |

### 2. 구현 경로

| 순서 | 문서 | 시작 조건 |
|---|---|---|
| 8-1 | [iFlow 설계와 배포](8-1.%20iFlow%20설계와%20배포.md) | Cloud Integration으로 첫 연계를 직접 만들고 싶을 때 |
| 8-2 | [API Management로 API 노출과 테스트](8-2.%20API%20Management%EB%A1%9C%20iFlow%20API%20%EB%85%B8%EC%B6%9C%EA%B3%BC%20%ED%85%8C%EC%8A%A4%ED%8A%B8.md) | 8-1에서 배포한 iFlow를 API Proxy로 보호·공개할 때 |

> **읽는 법**: 도입 여부가 아직 불확실하면 1~7과 6-1만 읽어도 됩니다. 8-1·8-2는 도입/PoC 방향이 정해진 뒤 따라 하는 실습 문서입니다.

## 한 장 요약

SAP Integration Suite는 SAP BTP에서 제공되는 iPaaS(Integration Platform as a Service)입니다. SAP 및 비SAP 애플리케이션, API, 이벤트, B2B 파트너를 연결하고 이를 중앙에서 설계·운영·모니터링합니다. 단순히 데이터를 옮기는 도구 하나가 아니라, 통합 방식별 기능을 한 서비스군으로 제공하는 플랫폼입니다.

![Integration Suite 아키텍처](assets/integration-suite-architecture.svg)

## 출처 원칙과 기존 문서 검토 결과

- **우선 출처**: SAP FSD, SAP Help Portal, SAP 제품/가격/Discovery Center 페이지
- **보조 출처**: 제공된 국내 블로그·파트너 글은 용어 이해에만 참고했습니다. 이 문서의 가격, 할인율, 포함 기능, 경쟁 제품 비교의 근거로 사용하지 않았습니다.
- **제거한 주장**: 기존 문서의 월별 달러 가격 범위, 3년 TCO 절감률, 특정 할인율, 무료 체험 일수, 플랜별 포함량처럼 공식 공개 자료로 검증되지 않은 수치·단정은 모두 삭제했습니다.

## 핵심 공식 자료

- [SAP Integration Suite 제품 개요](https://www.sap.com/korea/products/technology-platform/integration-suite.html)
- [SAP Integration Suite 가격](https://www.sap.com/korea/products/technology-platform/integration-suite/pricing.html)
- [SAP Discovery Center 서비스 카탈로그](https://discovery-center.cloud.sap/serviceCatalog/integration-suite?region=all)
- [SAP Help Portal - Integration Suite](https://help.sap.com/docs/integration-suite/sap-integration-suite/what-is-sap-integration-suite?locale=en-US)
- [Help Portal - capability 간 상호작용](https://help.sap.com/docs/integration-suite/sap-integration-suite/how-integration-suite-capabilities-interact?locale=en-US)
- [Help Portal - 배포 옵션과 런타임](https://help.sap.com/docs/integration-suite/sap-integration-suite/runtimes?locale=en-US)
- [Help Portal - capability 활성화](https://help.sap.com/docs/integration-suite/sap-integration-suite/activating-and-managing-capabilities?locale=en-US)
- `FSD_IntegrationSuite.pdf` - SAP Feature Scope Description, 2025-09-04
