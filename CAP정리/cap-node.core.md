# cap-node/core 분석

## 1. 개요

`cap-node/core` 모듈은 전체 CAP 프로젝트의 백엔드 시스템에서 핵심 기반을 담당하는 CAP(Cloud Application Programming Model) 서비스입니다. 사용자, 권한, 메뉴, 공통 코드 등 여러 애플리케이션에서 공통적으로 필요로 하는 데이터를 관리하고 비즈니스 로직을 처리합니다. `cap-app/portal`과 `cap-app/sysmgt` UI 애플리케이션들이 직접적으로 데이터를 주고받는 대상이 바로 이 `core` 모듈입니다.

이 문서는 `core` 모듈을 데이터 모델(DB), 서비스 정의(Service), 그리고 커스텀 로직(Handler)의 세 가지 관점에서 심층적으로 분석하여, CAP를 활용한 백엔드 개발의 전형적인 구조와 메커니즘을 설명합니다.

## 2. 데이터 모델 (DB) - `db/cds/`

데이터의 청사진인 데이터베이스 스키마는 CDS(Core Data Services)를 사용하여 `db/cds/` 폴더에 파일 단위로 정의됩니다.

### `User-model.cds` & `Role-model.cds`

```cds
// User-model.cds
namespace com.cap.core;
using managed from '@sap/cds/common';

entity User {
    key user_id: String(255);
    user_name: String(240);
    email_address: String(255);
    // ... other user attributes
    user_role_group: Composition of many core.User_RoleGroup on user_role_group.user_id = user_id;
}
extend User with managed;

// Role-model.cds
namespace com.cap.core;
using managed from '@sap/cds/common';

entity Role {
    key role_code: String(30);
    role_name: String(30);
    // ...
    role_role_group: Composition of many core.Role_RoleGroup on role_role_group.role_code = role_code;
}
extend Role with managed;
```

*   **엔티티(Entity) 정의**: `entity` 키워드를 사용하여 데이터베이스 테이블과 매핑될 엔티티를 정의합니다. `User` 엔티티는 사용자 정보를, `Role` 엔티티는 시스템의 역할(권한) 정보를 담습니다.
*   **관계 정의**:
    *   `Composition of many`: `User`와 `RoleGroup` 간의 다대다(N:M) 관계를 `User_RoleGroup`이라는 중간 엔티티를 통해 정의합니다. `Composition`을 사용함으로써 `User_RoleGroup`의 생명주기가 `User`에 종속되도록 합니다.
    *   `Association to`: `User` 엔티티에서 `CodeItem` 엔티티로의 `Association`은 외래 키(Foreign Key) 관계를 설정하여, 언어 코드나 날짜 형식 같은 값을 공통 코드 테이블에서 참조하도록 하여 데이터의 정합성을 유지합니다.
*   **`managed` Aspect**: `extend User with managed` 구문은 CAP의 강력한 기능 중 하나입니다. 이를 통해 해당 엔티티에 `createdAt`, `createdBy`, `modifiedAt`, `modifiedBy` 필드가 자동으로 추가되어, 모든 데이터 변경 이력을 손쉽게 추적할 수 있습니다.

## 3. 서비스 노출 (Service) - `srv/cds/`

데이터베이스에 정의된 모델을 외부(UI 애플리케이션 등)에서 사용할 수 있도록 OData API로 노출하는 역할을 합니다.

### `UserManagement-service.cds`

```cds
using { com.cap.core.User as User } from '../../db/cds/User-model';

namespace com.cap.core;

@(path: 'core/UserManagement')
@impl: 'srv/src/feature/user/User.handler'
service UserManagement {

    entity UserSet as projection on User {
        key user_id,
        user_name,
        // ...
        language_code_item.code_name as language_name
    };

    @(restrict: [
        { grant: 'READ', where: 'user_id = $user' }
    ])
    entity UserSessionInfo as select from UserSet { /* ... */ };
}

annotate UserManagement with @(requires: 'authenticated-user');
```

*   **`service` 정의**: `UserManagement`라는 이름의 OData 서비스를 정의합니다.
*   **`@path`**: 이 서비스를 `.../core/UserManagement`라는 URL 경로로 노출시킵니다. `approuter`는 `/srv-api` 요청을 이 CAP 서비스의 URL로 매핑해주므로, UI는 최종적으로 `/srv-api/odata/v2/core/UserManagement` 주소로 이 서비스를 호출하게 됩니다.
*   **`projection on`**: 데이터베이스 엔티티(`User`)를 그대로 노출하지 않고, `projection`을 통해 API의 스펙을 별도로 정의합니다. 이를 통해 필요한 필드만 선택하거나, 연관된 엔티티의 필드를 가져와(예: `language_name`) 구조를 단순화하는 등 유연한 API 설계가 가능합니다.
*   **`@impl`**: 서비스의 커스텀 로직을 구현한 TypeScript 파일(`User.handler.ts`)을 지정합니다. 선언적인 서비스 정의와 실제 로직 구현을 분리하는 CAP의 핵심적인 설계 사상입니다.
*   **선언적 권한 관리**:
    *   `@requires: 'authenticated-user'`: 서비스 전체에 인증된 사용자만 접근할 수 있도록 강제합니다.
    *   `@restrict: ... 'user_id = $user'`: `UserSessionInfo` 엔티티에 대해, 사용자가 자신의 정보만 조회할 수 있도록 데이터 레벨의 접근 제어를 설정합니다. `$user`는 CAP가 자동으로 채워주는 현재 로그인 사용자의 ID입니다.

## 4. 커스텀 비즈니스 로직 (Handler) - `srv/src/`

단순한 CRUD를 넘어선 복잡한 비즈니스 로직은 TypeScript 핸들러 클래스에 구현됩니다.

### `feature/user/User.handler.ts`

```typescript
import cds, { Request } from '@sap/cds';
import type { User } from '#cds-models/com/cap/core';

export default class UserHandler extends cds.ApplicationService {

  async init(): Promise<void> {
    // 'UserSet' 엔티티에 대한 CREATE 이벤트가 발생하기 '전(before)'에
    // 'beforeUserCreate' 메소드를 실행하도록 등록합니다.
    this.before('CREATE', 'UserSet', this.beforeUserCreate);
    await super.init();
  }

  /**
   * CREATE UserSet 실행 전 유효성 검증
   */
  async beforeUserCreate(req: Request) {
    const data = req.data as User;
    const db = cds.transaction(req);

    // 입력된 이메일 주소와 동일한 사용자가 DB에 있는지 확인합니다.
    const result = await db.run(
      SELECT.from('com.cap.core.User').where({ email_address: data.email_address })
    );

    // 중복된 이메일이 있다면, 에러를 발생시켜 CREATE 작업을 중단시킵니다.
    if (result.length > 0) {
      req.error(999, '중복된 이메일입니다.');
    }
  }
}
```

*   **이벤트 기반 아키텍처**: CAP 서비스는 이벤트 기반으로 동작합니다. `CREATE`, `READ`, `UPDATE`, `DELETE`와 같은 OData 요청이 들어오면, 각 요청에 해당하는 이벤트가 발생합니다.
*   **이벤트 핸들러 등록**: `init()` 메소드에서 `this.before()`, `this.on()`, `this.after()` 등의 메소드를 사용하여 특정 이벤트에 대한 핸들러 함수를 등록할 수 있습니다. 위 예제는 `CREATE` 이벤트가 발생하기 `before`에 특정 로직을 수행하도록 설정한 것입니다.
*   **비즈니스 로직 구현**: `beforeUserCreate` 메소드는 `UserSet` 엔티티가 생성되기 전에 '이메일 중복 체크'라는 비즈니스 규칙을 수행합니다. 데이터베이스를 조회하여 중복이 발견되면 `req.error()`를 통해 요청을 중단시키고 UI에 오류 메시지를 전달합니다.

## 5. 결론

`cap-node/core` 모듈은 CAP의 핵심 사상을 잘 보여주는 모범적인 백엔드 서비스입니다. CDS를 통해 데이터 모델, 서비스 API, 권한을 선언적으로 명확하게 정의하고, 복잡한 비즈니스 로직은 TypeScript 핸들러에 위임하여 코드의 가독성과 유지보수성을 극대화합니다. 이러한 구조를 통해 개발자는 반복적인 CRUD 구현에서 벗어나, 중복 체크와 같은 핵심 비즈니스 로직에 집중할 수 있습니다.
