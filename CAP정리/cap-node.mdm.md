# cap-node/mdm 분석

## 1. 개요

`cap-node/mdm` 모듈은 프로젝트의 기준 정보 관리(Master Data Management)를 담당하는 핵심 CAP 백엔드 서비스입니다. 자재 마스터, 제조사 정보, 품목 분류 등 복잡한 기준 정보 데이터를 생성하고 관리하는 비즈니스 로직을 수행합니다. `cap-app/mdm`에 포함된 여러 UI 애플리케이션들은 이 `mdm` 모듈이 제공하는 OData API를 통해 백엔드와 통신합니다.

이 문서는 `mdm` 모듈의 데이터 모델, 서비스, 커스텀 로직 및 외부 시스템 연동 방식을 분석하여, 복잡한 비즈니스 요구사항을 CAP를 통해 어떻게 구현하는지 상세히 설명합니다.

## 2. 데이터 모델 및 서비스

### 2.1. 데이터 모델: `MaterialMaster-model.cds`

`mdm` 모듈의 데이터 모델은 실제 비즈니스의 복잡성을 반영하여 깊고 풍부한 구조를 가집니다.

```cds
namespace com.cap.mdm;
using managed from '@sap/cds/common';

@cds.autoexpose
entity MaterialMaster {
    key REQ_NUM: String(20);
    MATCLASS: String(4);
    MATERIAL: String(40);
    // ... 수많은 자재 마스터 속성들
    plant_item: Composition of many mdm.MaterialMasterPlantInfo;
    maker_item: Composition of many mdm.MaterialMasterMakerInfo;
    // ... 등등 다수의 Composition 관계
}
extend MaterialMaster with managed;
```

*   **복합 엔티티**: `MaterialMaster` 엔티티는 자재의 기본 정보뿐만 아니라, `Composition`을 통해 플랜트, 제조사, 사양, 회계 등 수많은 하위 정보들을 포함하는 복합적인 구조를 가집니다. 이는 하나의 '자재 마스터 생성 요청'이 여러 하위 테이블의 데이터를 동시에 포함할 수 있음을 의미합니다.
*   **가상 필드**: `virtual REF_MATERIAL`과 같은 가상 필드를 사용하여, 데이터베이스에는 저장되지 않지만 '파생 채번'과 같은 특정 비즈니스 로직을 수행할 때 필요한 임시 값을 서비스 단계에서 처리할 수 있도록 합니다.

### 2.2. 서비스 정의: `MaterialMaster-service.cds`

`mdm` 서비스는 단순한 데이터 노출을 넘어, UI가 소비하기 좋은 형태로 데이터를 가공하여 제공하는 데 중점을 둡니다.

```cds
@(path: 'mdm/MaterialMaster')
@impl: 'srv/src/feature/materialMaster/MaterialMaster.handler'
service MaterialMasterService {
    // ... 단순 프로젝션들
    entity MaterialMasterSet as projection on MaterialMaster;

    // 복합 View 정의
    entity MaterialMasterView as select from MaterialMasterSet as master 
        left join CommonCodeItem as ZSECTICode on ...
        left join MaterialClassificationCode as matClasslevel1 on ...
    {
        key REQ_NUM,
        master.ZSECTI,
        ZSECTICode.ZFIELD05 as ZSECTINA, // 코드 '명칭'을 함께 제공
        ZCAT_L1,
        matClasslevel1.ZCLNAME as ZCAT_L1NA, // 분류 '명칭'을 함께 제공
        // ...
    };
}
```

*   **`left join`을 활용한 View**: 서비스 정의에서 `select from ...` 구문을 사용하여 여러 엔티티를 `left join`한 `MaterialMasterView`와 같은 복합 View를 만듭니다.
*   **UI 최적화**: 이 View는 `ZSECTI` 같은 '코드' 값만 반환하는 것이 아니라, `CommonCodeItem` 테이블과 조인하여 해당 코드의 '명칭'(`ZSECTINA`)까지 함께 반환합니다. 이를 통해 UI에서는 코드 값과 명칭을 한 번의 OData 호출로 가져올 수 있어, 여러 번의 API 호출을 하는 비효율을 막고 성능을 크게 향상시킵니다. 이는 CAP를 활용한 서비스 설계의 핵심적인 최적화 기법입니다.

## 3. 커스텀 비즈니스 로직: `MaterialMaster.handler.ts`

`mdm` 모듈의 백미는 복잡한 비즈니스 규칙을 구현한 TypeScript 핸들러에 있습니다.

```typescript
export default class MaterialMasterHandler extends cds.ApplicationService {
    private readonly _idLock = new Mutex(); // 동시성 제어를 위한 Mutex

    async init(): Promise<void> {
        // CREATE 이벤트가 발생하기 전에 채번 및 상태 계산 로직 실행
        this.before(['CREATE', 'UPDATE'], 'MaterialMasterSet', this.handleAfterReadMaster);
        await super.init();
    }

    private handleAfterReadMaster = async (req: Request) => {
        const data = req.data as MaterialMasterSet;
        const unlock = await this._idLock.lock(); // Mutex 잠금 시작
        try {
            // 1. 요청번호 (REQ_NUM) 생성
            data.REQ_NUM = await this._generateRequestNumber(tx);
            // 2. 자재코드 (MATERIAL) 생성
            data.MATERIAL = await this._generateMaterialCode(tx, data, req);
            // 3. 데이터 완성도에 따른 상태(R, Y, G) 계산
            row.BASIC_STATUS = this._calculateStatus(row, mandatoryFields, allFields);
        } finally {
            unlock(); // Mutex 잠금 해제
        }
    };
    // ... 채번 및 상태계산 로직 구현 ...
}
```

*   **커스텀 채번 로직**: CAP이 기본 제공하는 UUID 대신, 비즈니스 규칙에 맞는 채번 로직을 구현합니다.
    *   `_generateRequestNumber`: `RN+생성일자+순번` 형식의 요청 번호를 생성합니다.
    *   `_generateMaterialCode`: 분류코드, 순번, 버전을 조합하여 복잡한 규칙의 자재 코드를 생성합니다. 특히 '신규 채번'과 '파생 채번' 로직을 분기하여 처리합니다.
*   **동시성 제어 (Concurrency Control)**:
    *   `Mutex`라는 잠금(Lock) 메커니즘을 구현하여 사용합니다.
    *   만약 두 명의 사용자가 거의 동시에 동일한 분류의 자재 생성을 요청할 경우, 동일한 채번을 시도하여 충돌이 발생할 수 있습니다. `Mutex`는 이처럼 중요한 채번 로직이 한 번에 하나의 요청에 의해서만 실행되도록 보장하여 데이터 정합성을 지키는 핵심적인 역할을 합니다.
*   **동적 상태 계산**: `_calculateStatus`와 같은 함수를 통해, `MaterialMaster`와 그 하위 엔티티들(`plant_item` 등)의 데이터가 필수 규칙에 맞게 입력되었는지를 종합적으로 판단하여 '완료(G)', '진행중(Y)', '미흡(R)'과 같은 상태 값을 동적으로 계산하여 부여합니다.

## 4. 외부 시스템 연동

`package.json` 파일은 `mdm` 모듈이 외부 시스템과 연동됨을 보여줍니다.

```json
// package.json
"cds": {
    "requires": {
      "zapi_mm_create_o4": {
        "kind": "odata",
        "model": "srv/external/zapi_mm_create_o4",
        "credentials": {
          "destination": "onCloudConnector"
        }
      }
    }
}
```

*   **외부 OData 서비스 정의**: `zapi_mm_create_o4`라는 이름의 외부 OData 서비스를 정의합니다. 이는 SAP S/4HANA와 같은 ERP 시스템의 자재 생성 API일 가능성이 높습니다.
*   **Destination 서비스 활용**: 이 서비스를 호출할 때 `onCloudConnector`라는 이름의 Destination을 사용하도록 설정되어 있습니다. BTP의 Destination 서비스와 Connectivity 서비스를 통해, CAP 애플리케이션은 클라우드 환경에서 온프레미스(On-premise)에 있는 ERP 시스템의 API를 안전하고 안정적으로 호출할 수 있습니다.
*   **메커니즘**: `mdm` 앱에서 자재 생성 요청이 최종 '승인'되면, 핸들러 로직 내에서 CAP의 서비스 객체를 통해 `zapi_mm_create_o4` 서비스를 호출하여 ERP 시스템에 실제 자재 마스터를 생성하는 로직이 실행될 것입니다.

## 5. 결론

`cap-node/mdm` 모듈은 단순한 데이터 서비스를 넘어, 복잡한 비즈니스 규칙과 외부 시스템 연동까지 처리하는 고도화된 백엔드 서비스의 결정체입니다. 효율적인 서비스 설계를 통해 UI의 성능을 최적화하고, TypeScript 핸들러를 통해 채번, 동시성 제어, 상태 관리 등 핵심 비즈니스 로직을 견고하게 구현하며, Destination 서비스를 통해 다른 시스템과 유연하게 통합되는 모습을 통해 CAP의 진정한 강력함을 보여주는 모듈입니다.
