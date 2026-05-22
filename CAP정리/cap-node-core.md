# cap-node/core 모듈 문서

## 1. 개요

`cap-node/core` 모듈은 CAP 애플리케이션 백엔드의 기본 부분입니다. 시스템 관리, 사용자 관리 및 기타 교차 애플리케이션 문제에 대한 핵심 기능을 제공합니다.

이 모듈은 다음을 처리하기 위한 공통 데이터 구조(엔티티) 및 서비스를 정의합니다.
- 사용자 및 역할
- 메뉴 및 권한 부여
- 코드 목록 (예: 드롭다운용)
- 첨부 파일
- 메시징/메일링

## 2. 데이터 모델 (엔티티)

데이터 모델은 `cap-node/core/db/`에 정의되어 있습니다. 주요 엔티티는 다음과 같습니다.

- **User**: 이름, 이메일, 언어 기본 설정 및 테마 설정과 같은 사용자 프로필 정보를 저장합니다. 역할 그룹과 관계가 있습니다.
- **Role**: 역할을 정의합니다.
- **RoleGroup**: 사용자에게 할당할 수 있는 역할 모음입니다.
- **Menu**: 탐색 메뉴 구조를 저장합니다.
- **MenuFunction**: 메뉴 내의 특정 기능 또는 작업을 나타냅니다.
- **CodeHeader / CodeItem**: 코드 목록 및 해당 값(예: 국가, 상태, 유형)을 관리하기 위한 일반적인 구조입니다.
- **Message**: 시스템 또는 사용자 메시지를 저장하고 관리하기 위함입니다.
- **Attachment**: 파일 업로드를 처리하기 위한 일반적인 엔티티입니다.
- **Favorite**: 사용자의 즐겨찾는 메뉴 또는 항목을 관리하기 위함입니다.
- **Associations**: `User_RoleGroup`, `Role_Menu` 등과 같이 다른 엔티티를 연결하기 위해 여러 엔티티가 존재합니다.

### 예: `User` 엔티티 (`db/cds/User-model.cds`)

```cds
namespace com.cap.core;

using managed from '@sap/cds/common';

entity User {
    key user_id                             : String(255)       @title : '사용자ID';
        user_name                           : String(240)       @title : '사용자명';
        email_address                       : String(255)       @title : '이메일';
        language_code                       : String(4)         @title : '언어코드';
        theme_code                          : String(30)        @title : '테마코드';
        use_flag                            : Boolean           @title : '사용여부';
        // ... 다른 필드들
        user_role_group                     : Composition of many core.User_RoleGroup
                                                on  user_role_group.user_id = user_id;
}
```

## 3. 서비스

서비스는 `cap-node/core/srv/`에 정의되어 있습니다. 각 서비스는 데이터 모델을 노출하고 해당 도메인에 대한 비즈니스 로직을 포함합니다.

- **UserManagement-service**: 사용자와 역할 할당을 관리합니다.
- **RoleManagement-service**: 역할을 관리합니다.
- **RoleGroupManagement-service**: 역할 그룹을 관리합니다.
- **MenuManagement-service**: 메뉴 구조를 관리합니다.
- **CodeManagement-service**: 코드 목록에 대한 액세스를 제공합니다.
- **MessageManagement-service**: 메시지를 관리합니다.
- **Attachment-service**: 파일 업로드 및 다운로드를 처리합니다.
- **FormService-service**: 양식(아마도 PDF)을 생성하는 데 사용됩니다.
- **Mail-service**: 이메일을 보내기 위함입니다.
- **frame-service**: 일반적인 프레임워크 서비스입니다.

### 예: `User.handler.ts`의 비즈니스 로직

서비스 핸들러에는 사용자 지정 비즈니스 로직이 포함되어 있습니다. 예를 들어, `srv/src/feature/user/User.handler.ts`는 두 사용자가 동일한 이메일 주소를 가질 수 없도록 합니다.

```typescript
import cds, { Request } from '@sap/cds';
import type { User } from '#cds-models/com/cap/core';

export default class UserHandler extends cds.ApplicationService {

  async init(): Promise<void> {
    // UserSet 엔티티에 대한 CREATE 이벤트에 'before' 핸들러 등록
    this.before('CREATE', 'UserSet', this.beforeUserCreate);
    await super.init();
  }

  /**
   * 사용자 생성 전 유효성 검사.
   * 중복 이메일을 확인합니다.
   */
  async beforeUserCreate(req: Request) {
    const data = req.data as User;
    const db = cds.transaction(req);

    // 동일한 이메일을 가진 사용자가 이미 존재하는지 확인
    const result = await db.run(
      SELECT.from('com.cap.core.User').where({ email_address: data.email_address })
    );

    // 사용자가 발견되면 오류 발생
    if (result.length > 0) {
      req.error(999, '중복된 이메일입니다.'); // "Duplicate email."
    }
  }
}
```
이는 `core` 모듈의 서비스에 대한 사용자 지정 유효성 검사 로직이 어떻게 구현되는지 보여줍니다.
