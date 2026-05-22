# cap-app/template 문서

## 1. 개요

`cap-app/template` 디렉토리에는 작동하는 실행 애플리케이션이 포함되어 있지 않습니다. 대신, 이 프로젝트 내에서 새로운 UI5 애플리케이션을 생성하기 위한 **보일러플레이트** 또는 **템플릿** 역할을 합니다.

개발자가 새로운 UI 모듈(예: `sysmgt` 또는 `mdm` 컬렉션에 추가할 새로운 관리 화면)을 생성해야 할 때, 이 `template` 폴더를 시작점으로 복사할 수 있습니다. 이를 통해 새로운 애플리케이션이 처음부터 프로젝트의 기본 규칙과 종속성 요구 사항을 준수하도록 보장합니다.

## 2. 구조 및 목적

이 디렉토리의 구조는 최소한이며, 기본 UI5 애플리케이션 모듈에 필요한 필수 파일만 포함합니다.

-   **`webapp/manifest.json`**: 애플리케이션 설명자 파일입니다. 템플릿에는 가장 기본적인 설정만 포함되어 있습니다.
-   **기타 구성 파일**: `mta.yaml`, `package.json` 및 `ui5-deploy.yaml`과 같은 파일은 필요에 따라 조정할 수 있는 새 애플리케이션을 빌드하고 배포하는 데 필요한 구성을 제공합니다.

결정적으로 템플릿에는 다음이 **없습니다**:
-   `*.view.xml` 파일 (UI가 정의되지 않음).
-   `*.controller.js` 파일 (로직이 구현되지 않음).
-   `manifest.json`의 데이터 소스(`dataSources`) 또는 라우팅(`routing`) 구성.

### `manifest.json` 콘텐츠

`webapp/manifest.json`의 내용은 일반적인 시작점으로서의 역할을 강조합니다.

```json
{
	"_version": "1.59.0",
	"sap.app": {
		"id": "template",
		"type": "application",
		"applicationVersion": {
			"version": "1.0.0"
		},
        "title": "{{appTitle}}",
        "description": "{{appDescription}}",
        "resources": "resources.json",
        "ach": "ach"
	},
	"sap.ui": {
        "technology": "UI5",
        "icons": {
            "icon": "",
            "favIcon": "",
            "phone": "",
            "phone@2": "",
            "tablet": "",
            "tablet@2": ""
        },
        "deviceTypes": {
            "desktop": true,
            "tablet": true,
            "phone": true
        }
	},
	"sap.ui5": {
        "flexEnabled": false,
        "dependencies": {
			"minUI5Version": "1.123.1",
            "libs": {
				"sap.ui.core": {},
				"sap.tnt": {},
				"sap.m": {},
                "sap.ui.layout": {}
            }
        },
        "contentDensities": {
            "compact": true,
            "cozy": true
        }
    }
}
```

## 3. 사용 방법

개발자는 일반적으로 다음 단계를 따라 새 애플리케이션을 만듭니다.

1.  **복사 및 이름 바꾸기**: 전체 `cap-app/template` 디렉토리를 복사하고 새 애플리케이션의 목적을 반영하도록 이름을 바꿉니다(예: `cap-app/mdm/newFeature`).
2.  **`manifest.json` 업데이트**: `manifest.json`의 `id`를 `"template"`에서 고유 식별자(예: `"newFeature"`)로 변경합니다.
3.  **애플리케이션 정의**: `manifest.json`에 `rootView`, `dataSources` 및 `routing` 구성을 추가합니다.
4.  **UI 구현**: `webapp` 폴더 내에 필요한 `View` 및 `Controller` 파일을 만듭니다.
5.  **개발**: 필요에 따라 `common.lib`의 `BaseController`를 확장하여 애플리케이션의 기능을 빌드합니다.
6.  **배포**: 새 애플리케이션을 포함하도록 배포 구성을 업데이트합니다.
