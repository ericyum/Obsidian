# cap-node/mdm 모듈 문서

## 1. 개요

`cap-node/mdm` 모듈은 애플리케이션의 마스터 데이터 관리(MDM) 기능을 담당합니다. 자재 및 제조 관련 데이터에 중점을 두고 핵심 비즈니스 엔티티의 생성, 유효성 검사 및 관리를 처리합니다.

이 모듈은 다음에 대한 데이터 구조 및 비즈니스 로직을 정의합니다.
- 자재 마스터, 플랜트, 판매 및 회계와 같은 다양한 뷰 포함.
- 제조업체 정보.
- 자재 분류 및 속성.
- 검증 및 유효성 검사 규칙.
- S/4HANA와 같은 외부 시스템과의 통합.

## 2. 데이터 모델 (엔티티)

데이터 모델은 `cap-node/mdm/db/`에 정의되어 있습니다. 중앙 엔티티는 `MaterialMaster`입니다.

- **MaterialMaster**: 자재의 핵심 엔티티입니다. 일반 데이터를 포함하고 보다 구체적인 데이터 뷰에 연결됩니다.
    - **MaterialMasterBasicInfo**: 기본 자재 정보.
    - **MaterialMasterPlantInfo**: 플랜트별 데이터.
    - **MaterialMasterSalesOrganizationInfo**: 판매별 데이터.
    - **MaterialMasterAccountingInfo**: 회계별 데이터.
    - **MaterialMasterMakerInfo**: 자재 제조업체에 대한 정보.
    - **MaterialMasterSpecInfo**: 사양 세부 정보.
- **MakerInformation**: 제조업체에 대한 데이터를 저장합니다.
- **MaterialClassificationCode / Item**: 자재에 대한 분류 계층을 관리합니다.
- **MaterialAttributes**: 자재에 대한 사용자 정의 속성을 정의합니다.
- **VerificationRulesHeader / Item**: 데이터 유효성 검사 규칙을 저장합니다.
- **CommonCodeHeader / Item**: MDM별 코드 목록을 관리합니다.

### 예: `MaterialMaster` 엔티티 (`db/cds/MaterialMaster-model.cds`)

```cds
namespace com.cap.mdm;

using managed from '@sap/cds/common';

@cds.autoexpose
entity MaterialMaster {
    key REQ_NUM                     : String(20)      @title : '요청번호';
        MATERIAL                    : String(40)      @title : '자재코드';
        MATL_DESC                   : String(100)     @title : '자재내역';
        MATL_TYPE                   : String(4)       @title : '자재유형';
        PROD_HIER                   : String(18)      @title : '제품계층구조';
        CREA_STATUS                 : String(1)       @title : '품의상태';
        // ... 많은 다른 필드들

        // 자식 엔티티에 대한 구성
        plant_item                  : Composition of many mdm.MaterialMasterPlantInfo on ...;
        maker_item                  : Composition of many mdm.MaterialMasterMakerInfo on ...;
        spec_info                   : Composition of many mdm.MaterialMasterSpecInfo on ...;
        // ... 등등
}
```

## 3. 서비스

서비스는 `cap-node/mdm/srv/`에 정의되어 있습니다. MDM 데이터와 상호 작용하기 위한 API를 제공하고 중요한 비즈니스 로직을 포함합니다.

- **MaterialMaster-service**: 자재 마스터 데이터를 생성하고 관리하기 위한 기본 서비스.
- **MaterialMasterRequest-service**: 새 자재에 대한 요청을 처리합니다.
- **MakerInformationMangement-service**: 제조업체 데이터를 관리합니다.
- **VerificationRules-service**: 유효성 검사 규칙을 관리합니다.
- **ExternalS4hana-service**: S/4HANA 시스템과의 통합을 위한 서비스를 노출합니다. 이것은 `srv/external/`에 정의되어 있습니다.
- ... 및 데이터 모델에 해당하는 기타 서비스.

### 예: `MaterialMaster.handler.ts`의 비즈니스 로직

MDM 모듈의 서비스 핸들러는 비즈니스 로직이 풍부합니다. `MaterialMaster.handler.ts`는 다음 로직을 포함하는 대표적인 예입니다.

#### 1. 자동 ID 및 코드 생성
핸들러는 새 자재가 생성될 때 고유한 `REQ_NUM`(요청 번호)와 구조화된 `MATERIAL` 코드를 자동으로 생성합니다. 동시 요청이 있는 경우에도 중복 ID가 없는지 확인하기 위해 `Mutex`를 사용합니다.

```typescript
// 요청 번호 생성을 위한 단순화된 로직
private async _generateRequestNumber(tx: any): Promise<string> {
    const today = new Date();
    const dateKey = today.toISOString().slice(0, 10).replace(/-/g, ''); // YYYYMMDD
    const prefix = `RN${dateKey}`;

    // 오늘의 최대 ID를 찾아 시퀀스를 증가시킵니다.
    const result = await tx.run(
        SELECT.one.from('com.cap.mdm.MaterialMaster')
            .columns('max(REQ_NUM) as maxId')
            .where({ REQ_NUM: { like: `${prefix}%` } })
    );

    let seq = 1;
    if (result?.maxId) {
        const lastSeq = parseInt(result.maxId.replace(prefix, ''), 10);
        if (!isNaN(lastSeq)) seq = lastSeq + 1;
    }

    return `${prefix}${String(seq).padStart(3, '0')}`;
}
```

#### 2. 데이터 완전성 상태 계산
서비스는 `MaterialMaster` 및 `plant_item`과 같은 자식 엔티티의 필수 필드가 채워졌는지 여부에 따라 다른 상태 필드(예: `BASIC_STATUS`, `PLAN_STATUS`)를 계산합니다. 이는 사용자가 마스터 데이터의 완전성을 이해하는 데 도움이 됩니다.

```typescript
// 상태 계산을 위한 단순화된 로직
private _calculateStatus<T extends object>(data: T, mandatoryFields: (keyof T)[]): string {
    if (!data) return '';

    // 필수 필드가 비어 있는지 확인
    for (const field of mandatoryFields) {
        const val = data[field];
        if (val === null || val === undefined || val === '') {
            return 'R'; // 불완전한 경우 '빨간색'
        }
    }
    return 'G'; // 완료된 경우 '녹색'
}
```

이 모듈은 고품질 마스터 데이터를 관리하는 애플리케이션의 목적에 중심적인 역할을 합니다.
