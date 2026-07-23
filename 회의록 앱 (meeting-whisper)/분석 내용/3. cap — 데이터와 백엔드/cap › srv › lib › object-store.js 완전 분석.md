# object-store.js 완전 분석

> 소스: `cap/srv/lib/object-store.js`  
> 역할: VCAP 바인딩을 해석해 S3 호환 Object Store 업로드·서명 URL·삭제를 수행한다.

## 1. 이 파일이 있는 이유

VCAP 바인딩을 해석해 S3 호환 Object Store 업로드·서명 URL·삭제를 수행한다.

이 파일은 **3. cap — 데이터와 백엔드** 영역에 속한다. 독립적으로 보기보다 아래의 호출자·의존 대상·데이터 계약과 함께 읽어야 한다.

## 2. 파일 개요

| 항목 | 값 |
|---|---|
| 경로 | `cap/srv/lib/object-store.js` |
| 형식 | `javascript` |
| 전체 줄 수 | 201 |
| 주석 줄 수 | 0 |
| 주요 심볼 수 | 14 |
| 환경 변수 참조 수 | 10 |

## 3. 의존성

- `node:crypto`
- `@aws-sdk/client-s3`
- `@aws-sdk/s3-request-presigner`

## 4. 주요 선언과 책임

| 선언 | 읽을 때 확인할 점 |
|---|---|
| `boolFromEnv` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `intFromEnv` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `objectStoreConfig` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `normalizeEndpoint` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `vcapObjectStoreCredentials` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `parseJsonEnv` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `isObjectStoreEnabled` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `requireObjectStoreConfig` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `assertObjectStoreAudioHandoffReady` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `objectStoreClient` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `createAudioObjectKey` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `putAudioObject` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `getAudioObjectUrl` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |
| `deleteAudioObject` | 입력, 상태 변경, 반환값과 실패 경로를 확인한다. |

## 5. 환경 변수

- `OBJECT_STORE_BUCKET`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `OBJECT_STORE_ENDPOINT`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `OBJECT_STORE_REGION`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `OBJECT_STORE_ACCESS_KEY_ID`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `OBJECT_STORE_SECRET_ACCESS_KEY`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `OBJECT_STORE_ROLE`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `OBJECT_STORE_PREFIX`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `OBJECT_STORE_READ_VCAP_NAME`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `OBJECT_STORE_WRITE_VCAP_NAME`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.
- `OBJECT_STORE_VCAP_NAME`: 배포 환경과 로컬 실행 값이 달라질 수 있으므로 `mta.yaml`, `.env.example`과 함께 확인한다.

## 7. 코드 흐름 상세

### 7.1 `boolFromEnv`

- 위치: 7~12행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.2 `intFromEnv`

- 위치: 13~17행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.3 `objectStoreConfig`

- 위치: 18~35행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.4 `normalizeEndpoint`

- 위치: 36~46행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.5 `vcapObjectStoreCredentials`

- 위치: 47~87행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.6 `parseJsonEnv`

- 위치: 88~97행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.7 `isObjectStoreEnabled`

- 위치: 98~101행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.8 `requireObjectStoreConfig`

- 위치: 102~118행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.9 `assertObjectStoreAudioHandoffReady`

- 위치: 119~129행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.10 `objectStoreClient`

- 위치: 130~153행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.11 `createAudioObjectKey`

- 위치: 154~159행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.12 `putAudioObject`

- 위치: 160~170행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.13 `getAudioObjectUrl`

- 위치: 171~183행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

### 7.14 `deleteAudioObject`

- 위치: 184~201행
- 의미: 이 블록의 전제 조건, 외부 호출, DB/메모리 상태 변경, 예외 처리 순서가 핵심이다.

## 8. 변경 시 영향 범위
+
- 공개 계약, 상태명, 환경 변수, URL을 바꾸면 호출 측과 테스트를 함께 수정해야 한다.
- 저장 데이터의 형식이 바뀌면 기존 HANA 행과 진행 중인 작업의 호환성을 확인한다.
- 인증·소유권 검사는 편의상 우회하면 안 된다. Approuter 인증과 CAP 역할 검사는 서로 다른 계층이다.

## 9. 관련 문서

- [[00. 전체 구조와 책임 지도]]
- [[01. 녹음부터 회의록까지 End-to-End]]
- [[02. 데이터 모델과 상태 전이]]
- [[03. 인증·권한·신뢰 경계]]
