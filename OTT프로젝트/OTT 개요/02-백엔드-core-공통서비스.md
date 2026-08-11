# 02 - 백엔드: cap-node/core (공통 핵심 서비스)

> **네임스페이스:** `com.cap.core`  
> **패키지명:** `@com/cap-ott-core`  
> **포트:** 8080 (로컬 개발)  
> **MTA ID:** `com-cap-ott-core`

---

## 1. 이 모듈이 하는 일

**공통 인프라 서비스**로서, 이 프로젝트 전체의 기반이 되는 기능을 제공한다:

- 👤 **사용자 관리** (UserManagement): 사용자 CRUD + 세션 정보 + 권한그룹 연동
- 🔑 **역할 관리** (RoleManagement): 역할·역할그룹·메뉴할당·기능제어 관리
- 📋 **메뉴 관리** (MenuManagement): 메뉴 트리 + 다국어 + 즐겨찾기 + 권한 필터링
- 🏷️ **공통 코드 관리** (CodeManagement): 코드그룹/코드아이템 (언어·날짜서식·통화 등)
- ✉️ **메시지 관리** (MessageManagement): 다국어 메시지 (DB 기반 i18n)
- 📧 **메일 서비스** (MailService): SAP Destination 경유 SMTP 메일 발송
- 📄 **PDF 서비스** (FormService): PDFKit 기반 PDF 생성 (박스 라벨 등)
- 🔐 **인증 체크** (FrameService): AuthCheck 함수 (리디렉션)

---

## 2. 파일 하나하나 상세 설명

### 2.1 설정 파일들

#### `package.json`
```json
{
  "name": "@com/cap-ott-core",
  "dependencies": {
    "@sap/cds": "^9.3.0",        // CAP 프레임워크
    "@cap-js/hana": "^2.2.0",    // HANA DB 어댑터
    "@sap/xssec": "^4.8.0",      // XSUAA 인증
    "nodemailer": "^8.0.1",       // 이메일 전송
    "pdfkit": "^0.17.2",          // PDF 생성
    "@aws-sdk/client-s3": "^3.879.0" // S3 파일 업로드
  },
  "scripts": {
    "watch": "cds watch --port 8080",      // 개발 서버
    "build:cf": "npm run build:cds && npm run cleanup:ts && npm run build:ts",
    "deploy:mta": "npm run build:mbt && npm run deploy:cf"
  }
}
```

**중요 포인트:**
- `build:cf` 스크립트는 세 단계로 이루어진다:
  1. `cds build --production` → CDS 모델을 HANA SQL로 컴파일
  2. `cleanup:ts` → `gen/srv/srv/**/*.ts` 파일 정리
  3. `build:ts` (tsc) → TypeScript → JavaScript 컴파일
- `odata-v2-adapter`를 사용해 OData v2로 API를 제공한다 (SAPUI5 호환성)
- Subpath import `#cds-models/*`를 통해 생성된 타입을 불러온다

#### `mta.yaml` - 배포 디스크립터
```yaml
modules:
  - name: com-cap-ott-core     # Node.js 앱 모듈
    type: nodejs
    path: gen/srv               # 빌드 결과물 경로
    requires:
      - com-hdi-container       # HANA DB 사용
      - com-cap-ott-core-destination  # Destination 서비스
      - com-cap-ott-core-xsuaa  # XSUAA 인증

  - name: com-ott-db-deployer   # DB Deployer 모듈
    type: hdb
    path: gen/db
    requires:
      - com-hdi-container
```

**중요:** `${space}` 변수가 서비스 이름 뒤에 붙는다. 예: `com-hdi-container-dev`, `com-hdi-container-prod`. 이렇게 하면 동일한 BTP Subaccount 내에서 Dev/Prod 환경을 분리할 수 있다.

#### `default-env.json` - 로컬 개발용 환경 설정
로컬에서 `cds watch` 할 때 사용할 HANA DB 연결 정보가 담겨 있다. 실제 BTP 환경의 credentials를 담고 있으므로 보안에 주의해야 한다. (`host`, `port`, `user`, `password`, `certificate`)

#### `.cdsrc.json` - CAP 빌드 설정
```json
{
  "build": {
    "target": "gen",
    "tasks": [
      { "for": "hana", "src": "db", "options": { "model": ["db", "db/cds", "srv"] } },
      { "for": "nodejs" },
      { "for": "typescript" }
    ]
  },
  "hana": { "deploy-format": "hdbtable" },
  "requires": { "db": { "kind": "hana" } }
}
```

**설명:**
- `build.target`: 빌드 결과물이 `gen/` 폴더에 생성된다
- `hana.deploy-format: "hdbtable"`: HANA 배포 시 `.hdbtable` 파일 형식 사용
- `requires.db.kind: "hana"`: 기본 DB로 HANA를 사용 (로컬 SQLite가 아닌)

---

### 2.2 DB 모델 (db/cds/) - 13개 엔티티

#### User-model.cds - 사용자 테이블
```
Entity: User (com.cap.core.User)
주요 필드:
  - user_id (PK): 사용자ID
  - user_name: 사용자명
  - language_code: 언어코드 → CodeItem.LANGUAGE_CODE
  - email_address: 이메일
  - user_status_code: 사용자상태코드
  - use_flag: 사용여부 (Boolean)
  - user_role_group: → User_RoleGroup (Composition)
```

**연관 관계:**
- `User_RoleGroup`과 Composition 관계 (사용자가 삭제되면 권한그룹 매핑도 함께 삭제)
- `CodeItem`과 Association (언어·날짜서식·숫자서식·사용자구분)

#### Role-model.cds - 역할 테이블
```
Entity: Role
주요 필드:
  - role_code (PK): 역할 코드
  - role_name: 역할 명
  - use_flag: 사용 여부

연관:
  - role_menu (Composition): Role_Menu
  - role_role_group (Composition): Role_RoleGroup
  - role_menu_function (Composition): Role_MenuFunction
```

#### Menu-model.cds - 메뉴 테이블
```
Entity: Menu (가장 복잡한 엔티티)
주요 필드:
  - menu_code (PK): 메뉴 코드 (15자)
  - parent_menu_code: 상위 메뉴 코드 (자기 참조)
  - menu_app_id: 연결된 UI5 앱 ID
  - menu_repo_path: UI5 레포지토리 경로
  - menu_app_path: 앱 내부 경로
  - menu_route_path: 라우트 경로
  - menu_parameter: 파라미터
  - menu_icon: 아이콘 (sap-icon://...)
  - menu_type_code: 메뉴 유형 (Group/Menu/Link)
  - menu_level_number: 메뉴 레벨 (트리 깊이)

연관:
  - menu_function (Composition): MenuFunction
  - menu_language (Composition): MenuLanguage
  - role_menu (Composition): Role_Menu
  - role_menu_function (Composition): Role_MenuFunction
```

**중요 개념:** 메뉴는 트리 구조다. `parent_menu_code`로 자기 참조를 하며, `menu_level_number`로 깊이를 나타낸다. `menu_app_id`와 `menu_app_path`를 통해 실제 SAPUI5 앱과 연결된다.

#### CodeHeader + CodeItem - 공통 코드 관리
```
CodeHeader (코드 그룹):
  code_group (PK): 예) 'LANGUAGE_CODE', 'MESSAGE_TYPE_CODE'
  code_group_name: 코드 그룹 이름
  code_group_type: 'cap' (기본값, CAP에서 직접 관리)

CodeItem (코드 상세):
  code_group + code (복합 PK)
  code_name: localized String → 다국어 지원
  attribute_1~10: 확장 속성 (최대 10개까지 임의 데이터 저장 가능)
```

**중요:** `CodeItem.code_name`은 `localized String` 타입이다. 즉, 한 필드가 언어별로 다른 값을 가질 수 있다. CAP 내부에서 `_texts` 테이블로 분리되어 관리된다.

#### Favorite-model.cds - 즐겨찾기
```
Entity: Favorite
  user_id (PK): @cds.on.insert: $user → 자동으로 현재 사용자 ID
  menu_code (PK): 메뉴 코드
  sort_number: 정렬 번호
```

#### 연관 관계 매핑 테이블들
| 테이블 | 관계 | 설명 |
|---|---|---|
| User_RoleGroup | User ↔ RoleGroup | 사용자-역할그룹 매핑 |
| Role_RoleGroup | Role ↔ RoleGroup | 역할-역할그룹 매핑 |
| Role_Menu | Role ↔ Menu | 역할별 메뉴 접근 권한 + use_flag |
| Role_MenuFunction | Role ↔ Menu ↔ MenuFunction | 역할별 UI 기능 사용 여부 |
| MenuFunction | Menu ↔ function_code | 메뉴별 사용 가능한 기능 정의 |
| MenuLanguage | Menu ↔ language_code | 메뉴명 다국어 |
| Message | message_code + language_code | 다국어 메시지 |

---

### 2.3 서비스 정의 (srv/cds/) - 9개 서비스

#### FrameService
```cds
@impl: 'srv/src/config/FrameService.handler'
service FrameService {
    @readonly
    function AuthCheck() returns Integer;
}
```
**기능:** 인증 체크 + 포털로 리디렉션. AppRouter의 `index.html`에서 호출한다.

#### UserManagement
```cds
@(path: 'core/UserManagement')
service UserManagement {
    entity UserSet as projection on User;
    entity UserWithRoles as select from User left join User_RoleGroup;
    @(restrict: [{ grant: 'READ', where: 'user_id = $user' }])
    entity UserSessionInfo as select from UserSet ...;
}
```
**중요 기능:**
- `UserSet`: 기본 사용자 CRUD (핸들러에서 이메일 중복 체크)
- `UserWithRoles`: 사용자 + 권한그룹 목록 (role_group_codes를 쉼표로 합침)
- `UserSessionInfo`: **로그인한 사용자 본인**의 정보만 조회 (`user_id = $user` 제한)

#### RoleManagement
```cds
@(path: 'core/RoleManagement')
service RoleManagement {
    entity RoleSet, RoleGroupSet, RoleGroupMapSet, RoleMenuSet, RoleMenuFunctionSet;
    view RoleMenuList(role_code: String);
    view RoleMenuFunctionList(role_code: String, menu_code: String);
}
```
**중요 기능:**
- `RoleMenuList`: 특정 역할에 할당된/미할당된 메뉴 목록을 LEFT OUTER JOIN으로 조회
- `RoleMenuFunctionList`: 특정 역할-메뉴 조합에 대한 기능별 사용 여부 목록
- 입력 파라미터 `role_code`, `menu_code`로 필터링

#### RoleGroupManagement
```cds
@(path: 'core/RoleGroupManagement')
service RoleGroupManagement {
    entity RoleGroupSet, RoleRoleGroupSet, UserRoleGroupSet;
    view UnassignedRoleList(role_group_code: String);
    view UnassignedUserList(role_group_code: String);
}
```
**중요:** `UnassignedRoleList`, `UnassignedUserList`는 아직 할당되지 않은 역할/사용자를 조회하는 View. UI에서 "할당 가능한 역할/사용자" 팝업에 사용된다.

#### MenuManagement (가장 복잡한 서비스)
```cds
@(path: 'core/MenuManagement')
service MenuManagement {
    entity MenuSet, MenuFunctionSet, MenuLanguageSet, FavoriteSet;

    @(restrict: [{ grant: 'READ', where: 'user_id = $user' }])
    view MenusRoleAppliedList;    // 사용자 권한에 따른 메뉴 목록

    @(restrict: [{ grant: 'READ', where: 'user_id = $user' }])
    view FavoriteRoleAppliedList; // 사용자 즐겨찾기 메뉴

    @(restrict: [{ grant: 'READ', where: 'user_id = $user' }])
    view MenuFunctionAppliedList; // 사용자 권한에 따른 UI 기능 제어
}
```

**핵심 비즈니스 로직:**

1. **MenusRoleAppliedList**: `User → User_RoleGroup → Role_RoleGroup → Role → Role_Menu → Menu` 경로를 INNER JOIN하여 사용자에게 허용된 메뉴만 조회한다. `menu_use_flag = true` AND `role_menu.use_flag = true` 조건을 적용한다.

2. **FavoriteRoleAppliedList**: 위와 동일한 경로 + `Favorite` 테이블을 추가로 JOIN하여 즐겨찾기로 등록된 메뉴만 조회.

3. **MenuFunctionAppliedList**: 각 사용자가 가진 모든 권한그룹의 역할을 통해 접근 가능한 기능들을 `ROW_NUMBER()`로 집계하여, `role_menu_function.use_flag`가 있으면 우선, 없으면 `menu_function.default_use_flag`를 사용한다. 이를 통해 **"권한별로 특정 UI 기능을 켜고 끌 수 있다"**.

#### CodeManagement
```cds
service CodeManagement {
    entity CodeMasterSet as projection on CodeHeader;
    entity CodeSet as projection on CodeItem;
    entity CodeTexts as projection on CodeItem.texts;  // localized text
}
```

#### MessageManagement
```cds
service MessageManagement {
    entity MessageSet;
    view MessageList(language_code: String);
}
```
**중요:** `MessageList` View는 `ROW_NUMBER()` + `PARTITION BY message_code, message_type_code`를 사용하여, 요청한 언어가 있으면 우선, 없으면 다른 언어를 보여준다.

#### MailService
```cds
service MailService {
    type MailOptions {
        sender: String;
        to: many String;      // 다중 수신자 (배열)
        cc: many String;
        bcc: many String;
        subject: String;
        text: LargeString;
        html: LargeString;
    }
    action sendMail(options: MailOptions) returns String;
}
```

#### FormService
```cds
service FormService {
    type Item { companyName: String(30); }
    action TestPDF(data: Item);
}
```

---

### 2.4 핸들러 (비즈니스 로직)

#### server.ts - 커스텀 CDS 서버
```
cds.on('bootstrap') → Express 미들웨어 추가
  - x-username 헤더 로깅

cds.on('served') → 커스텀 엔드포인트 추가
  - GET /i18n.properties → generateI18n("KO")
  - GET /i18n_:lang.properties → generateI18n(lang)

generateI18n(lang):
  1. DB 연결
  2. SELECT message_code, message_contents FROM Message WHERE language_code = lang
  3. .properties 형식 문자열로 변환
  4. 반환 (text/plain)
```

**이것이 왜 중요한가?** 일반 SAPUI5 앱은 정적 `.properties` 파일에서 i18n을 읽지만, 이 프로젝트는 DB에 저장된 동적 메시지를 사용한다. `CoreResourceModel`이 이 URL을 읽어서 i18n 모델을 만든다.

#### FrameService.handler.ts - 인증 체크
```typescript
onAuthCheck(req: Request) {
    console.log("###onAuthCheck###");
    req.http?.res.redirect('/portal/index.html');  // 포털로 리디렉션
    return 0;
}
```

#### User.handler.ts - 사용자 생성 전 이메일 중복 체크
```typescript
beforeUserCreate(req: Request) {
    const { email_address } = req.data as User;
    const result = await db.run(
        SELECT.from('com.cap.core.User').where({ email_address })
    );
    if (result.length > 0) {
        req.error(999, '중복된 이메일입니다.');
    }
}
```

#### Mail.handler.ts - 이메일 발송
```typescript
on('sendMail', async (req) => {
    const { options } = req.data;  // OData 페이로드에서 추출
    const mailOptions = {
        from: options.sender,      // sender → from 매핑
        to: options.to,
        cc, bcc, subject, text, html
    };
    const emailService = EmailService.getInstance();  // 싱글톤
    const info = await emailService.sendMail(mailOptions);
    return info.messageId;         // Message ID 반환
});
```

#### EmailService.ts - SMTP 메일 전송
```typescript
async sendMail(options) {
    // 1. SAP BTP Destination 'onMailServer' 조회
    const dest = await getDestination({ destinationName: 'onMailServer' });

    // 2. Destination 속성에서 SMTP 설정 추출
    //    host, port, user, password, SSL 여부

    // 3. Nodemailer Transporter 생성
    // 4. 메일 발송
}
```

**중요:** SAP BTP에서 메일 서버 정보를 직접 코드에 하드코딩하지 않는다. 대신 `Destination` 서비스를 사용한다. BTP Cockpit에서 `onMailServer`라는 Destination을 생성하고 SMTP 호스트·포트·계정을 설정하면, 코드에서는 Destination 이름만 참조한다. 이렇게 하면 환경(Dev/Prod)별로 다른 메일 서버를 사용할 수 있다.

#### FormService.handler.ts + BoxLabel.ts - PDF 생성
```typescript
onTestPDF(req) {
    const { res } = req.http;
    const doc = new PDFDocument({ margin: 0, size: [283, 227] });
    const builder = new BoxLabelBuilder(doc);
    doc.pipe(res);                    // PDF를 HTTP 응답으로 스트리밍
    await builder.generatePDF(data);
    doc.end();

    // ★ 핵심: 응답 전송 완료까지 대기
    await new Promise((resolve) => {
        res.on('finish', resolve);
    });
}
```

**PDF 생성 흐름:**
1. 클라이언트가 `POST /FormService/TestPDF` 호출
2. `FormService.handler.ts`에서 PDFDocument 생성
3. `BoxLabelBuilder`가 표 그리기 (drawTableCell 사용)
4. PDF 스트림을 HTTP 응답으로 직접 전송 (파일 다운로드)
5. 클라이언트(브라우저)에서 Blob으로 받아서 `<a download>` 태그로 저장

**PDFBuilder.drawTableCell()** 유틸리티:
- 텍스트/바코드 모두 지원 (string은 텍스트, Buffer는 이미지)
- 멀티라인 텍스트 (줄바꿈 처리)
- 수직 정렬 (top/middle/bottom)
- 배경색, 테두리, 패딩, 자간 설정 가능
- 용지 크기: 283×227 (소형 라벨 크기)

---

### 2.5 전체 데이터 권한 모델 정리

```
사용자(User)
  └── User_RoleGroup (N:M)
       └── RoleGroup
            └── Role_RoleGroup (N:M)
                 └── Role
                      ├── Role_Menu → Menu (메뉴 접근 권한 + use_flag)
                      └── Role_MenuFunction → MenuFunction (UI 기능 제어)

권한 계산 방식:
1. 사용자 → 소속된 RoleGroup 목록 조회
2. RoleGroup → 할당된 Role 목록 조회
3. Role → Role_Menu → Menu (use_flag=true인 것만)
4. Role → Role_MenuFunction → MenuFunction (use_flag 우선, 아니면 default_use_flag)

이 결과로:
- 어떤 메뉴가 보이는지
- 메뉴 내에서 어떤 버튼/기능이 활성화되는지
가 결정된다.
```
