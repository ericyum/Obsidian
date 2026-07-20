# SAP Integration Suite — 종합 조사 보고서

> **작성일**: 2026-07-20
> **목적**: SAP Integration Suite에 대한 전방위적 이해 (정의·기능·사용이유·대체·가격·BTP 플랜)

---

## 빠른 참조

| 질문 | 답변 |
|------|------|
| **무엇인가?** | SAP BTP의 iPaaS. 기업의 모든 통합(API·EDI·이벤트·프로세스)을 관리하는 클라우드 플랫폼 |
| **왜 사용하나?** | SAP 생태계와의 원천 통합, 2,000+ Pre-built 콘텐츠, 중앙 거버넌스, Low-code 민첩성, TCO 절감 |
| **대체 가능한가?** | 가능하나 SAP 연동 시 비용·개발 기간이 2~5배 증가. MuleSoft, Boomi, 오픈소스 등이 대안 |
| **얼마인가?** | Standard: $7~12K/월(Tenant), Enterprise: $12~20K/월(Tenant) + 사용량 과금 |
| **어떤 BTP 플랜?** | CPEA (기업) 또는 PAYG (중소기업). Free Tier는 불가, Trial 30일 가능 |

---

## 읽는 순서

1. **[정의](1. Integration Suite 정의.md)** — 역사(PI→PO→CPI→IS), iPaaS 개념, 아키텍처, 5대 통합 패턴
2. **[기능](2. Integration Suite 기능.md)** — 6개 Capability 상세 (Cloud Integration, API Mgmt, Event Mesh, Trading Partner Mgmt, Integration Advisor, Edge Integration Cell)
3. **[사용 이유](3. Integration Suite 사용 이유.md)** — 8가지 이유 (SAP 생태계, Low-code, TCO, Clean Core, AI 시대 대비 등)
4. **[대체 가능성](4. Integration Suite 대체 가능성.md)** — 경쟁사(MuleSoft/Boomi/Azure), 오픈소스, Point-to-Point, 의사결정 매트릭스, 숨은 비용
5. **[가격 정책](5. Integration Suite 가격 정책.md)** — Standard vs Enterprise, 과금 모델, 시나리오별 비용 예시, 할인 전략
6. **[BTP 요금제](6. BTP 요금제별 사용 가능 여부.md)** — CPEA vs PAYG vs Free Tier, Trial, 번들링
7. **[플랜별 과금 상세](7. 각 플랜별 과금 체계 상세.md)** — Standard/Enterprise 상세 가격, 개별 Capability 구매, 3년 TCO 비교

---

## 한눈에 보는 Integration Suite

```
SAP Integration Suite = Cloud Integration (iFlow)
                      + API Management
                      + Event Mesh
                      + Trading Partner Management (B2B/EDI)
                      + Integration Advisor (AI)
                      + Edge Integration Cell (하이브리드)

지원 통합: SAP ↔ SAP, SAP ↔ Non-SAP, On-prem ↔ Cloud, B2B/EDI
배포: Cloud (Multi-tenant), Edge Cell (On-prem), API Gateway
과금: Tenant 구독 + 사용량(Messages/API Calls/Data Transfer)
```

---

## 출처

- SAP 공식 사이트: https://www.sap.com/korea/products/technology-platform/integration-suite.html
- SAP Discovery Center: https://discovery-center.cloud.sap/
- SAP Help Portal: https://help.sap.com/docs/integration-suite/
- Gartner Magic Quadrant for iPaaS
- Forrester Wave: Integration Platforms
