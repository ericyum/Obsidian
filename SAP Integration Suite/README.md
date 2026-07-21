# SAP Integration Suite 가이드

SAP 직원이 업무 관점에서 빠르게 판단할 수 있도록 만든 입문 자료입니다. 제품 기능은 SAP 공식 Feature Scope Description(FSD, 2025-09-04)와 Help Portal을 우선으로 검토했고, 가격과 플랜은 2026-07-21에 확인한 SAP 공식 가격 페이지를 기준으로 합니다.

> **먼저 알아둘 점**: 이 문서는 설계와 판단을 돕는 자료입니다. 실제 사용 가능 기능, 무료 체험, 단가는 계약, 리전, 계정 상태에 따라 달라질 수 있으므로 구매 또는 운영 전에는 SAP Discovery Center와 SAP 담당자에게 확인해야 합니다.

## 읽는 순서

| 문서 | 답하는 질문 |
|---|---|
| [1. 정의](1.%20Integration%20Suite%20정의.md) | SAP Integration Suite는 정확히 무엇인가? |
| [2. 기능](2.%20Integration%20Suite%20기능.md) | 어떤 기능이 있고, 언제 쓰는가? |
| [3. 사용 이유](3.%20Integration%20Suite%20사용%20이유.md) | 어떤 문제를 해결하며 왜 고려하는가? |
| [4. 대체 가능성](4.%20Integration%20Suite%20대체%20가능성.md) | 우리 상황에서 꼭 필요한가? 더 저렴한 선택지는? |
| [5. 가격과 플랜](5.%20Integration%20Suite%20가격과%20플랜.md) | 무엇이 비용을 만들고 플랜은 어떻게 다른가? |
| [6. 무료 체험과 도입](6.%20무료%20체험과%20도입%20가이드.md) | 테스트는 무료인가? 무엇부터 확인해야 하는가? |
| [7. 사용 방법](7.%20Integration%20Suite%20사용%20방법.md) | 첫 iFlow를 만들고 API로 노출하려면 어떻게 하는가? |

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
