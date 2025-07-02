
# Python 개발 환경 설정 최종 요약

이 문서는 Python 프로젝트 개발을 위한 환경 설정 과정을 종합적으로 정리한 것입니다. VS Code, Anaconda/Miniconda, 가상 환경, 그리고 `pip` 명령어를 활용하는 방법을 상세히 다룹니다.

## 1. 개발 도구 및 환경 관리 시스템 이해

- **VS Code (Visual Studio Code):**
    
    - 다양한 프로그래밍 언어를 다룰 수 있는 **만능 작업대** 역할을 하는 코드 편집기입니다.
        
    - **Python 확장 설치의 중요성:** VS Code에서 Python 코드를 효율적으로 다루려면 Python 확장을 설치해야 합니다. 이 확장은 코드 자동 완성, 디버깅, 구문 강조 등 Python 개발에 특화된 다양한 기능들을 제공합니다.
        
- **Anaconda / Miniconda:**
    
    - **Python 파일별 버전 관리 (가상 환경):** Anaconda 또는 Miniconda는 Python 버전과 라이브러리들을 프로젝트별로 격리하여 관리할 수 있게 해주는 강력한 도구입니다. 이를 통해 여러 프로젝트 간의 의존성 충돌을 방지할 수 있습니다.
        
    - **Miniconda:** Anaconda의 작은 버전으로, 핵심 기능(conda, Python)만 포함하여 가볍게 시작할 수 있습니다.
        

## 2. Conda를 이용한 가상 환경 관리

Miniconda 설치 후 Anaconda Prompt를 실행하면 기본적으로 `(base)` 환경이 활성화됩니다.

- **(base) C:\Users\SBA>** 와 같이 프롬프트가 표시됩니다.
    

### 2.1. 가상 환경 생성

프로젝트를 위한 독립적인 Python 환경을 만듭니다.

```
# 새로운 'pyqt5'라는 이름의 가상 환경을 생성하고 Python 3.10 버전을 설치합니다.
(base) C:\Users\SBA>conda create -n pyqt5 python=3.10
```

### 2.2. 가상 환경 진입 (활성화)

생성한 가상 환경을 사용하려면 해당 환경을 활성화해야 합니다.

```
# 생성한 'pyqt5' 환경을 활성화합니다.
(base) C:\Users\SBA>conda activate pyqt5
```

- 프롬프트가 `(base)`에서 `(pyqt5)`로 변경되는 것을 확인할 수 있습니다. 이제 이 환경에서 실행하는 모든 `pip` 명령은 `pyqt5` 가상 환경 내에 패키지를 설치합니다.
    

### 2.3. 가상 환경에서 나가기 (비활성화)

작업을 마친 후 가상 환경에서 나올 때 사용합니다.

```
# 현재 활성화된 가상 환경에서 나갑니다.
(pyqt5) C:\Users\SBA>conda deactivate
```

### 2.4. 설치된 가상 환경 목록 보기

현재 시스템에 생성된 모든 가상 환경 목록을 확인합니다.

```
# 현재 시스템에 생성된 모든 가상 환경 목록을 출력합니다.
(base) C:\Users\SBA>conda env list
```

### 2.5. 가상 환경 삭제 (선택 사항)

더 이상 필요 없는 가상 환경은 삭제할 수 있습니다.

```
# 'pyqt5'라는 이름의 가상 환경을 삭제합니다.
(base) C:\Users\SBA>conda env remove -n pyqt5
```

## 3. Pip를 이용한 패키지 관리

가상 환경이 활성화된 상태에서 `pip`를 사용하여 Python 패키지를 설치하고 관리합니다.

### 3.1. 설치된 패키지 목록 출력

현재 활성화된 가상 환경에 설치된 패키지 목록을 확인합니다.

```
# 현재 활성화된 가상 환경에 설치된 패키지 목록을 출력합니다.
(pyqt5) C:\Users\SBA>pip list
```

- **예시 출력:**
    
    ```
    Package    Version
    ---------- -------
    pip        25.1
    setuptools 78.1.1
    wheel      0.45.1
    ```
    
    - 이 목록을 통해 `pyqt5` 모듈이 아직 설치되지 않은 상태임을 확인할 수 있습니다.
        

### 3.2. 패키지 설치 (`pip install`)

프로젝트에 필요한 특정 패키지를 설치합니다.

```
# 활성화된 'pyqt5' 환경에 PyQt5 라이브러리를 설치합니다.
(pyqt5) C:\Users\SBA\github\pyqt5_example>pip install pyqt5
```

- **설치 과정 출력 예시:**
    
    ```
    Collecting pyqt5
      Using cached PyQt5-5.15.11-cp38-abi3-win_amd64.whl.metadata (2.1 kB)
    Collecting PyQt5-sip<13,>=12.15 (from pyqt5)
      Downloading PyQt5_sip-12.17.0-cp313-cp313-win_amd64.whl.metadata (492 bytes)
    Collecting PyQt5-Qt5<5.16.0,>=5.15.2 (from pyqt5)
      Using cached PyQt5_Qt5-5.15.2-py3-none-win_amd64.whl.metadata (552 bytes)
    Using cached PyQt5-5.15.11-cp38-abi3-win_amd64.whl (6.9 MB)
    Using cached PyQt5_Qt5-5.15.2-py3-none-win_amd64.whl (50.1 MB)
    Downloading PyQt5_sip-12.17.0-cp313-cp313-win_amd64.whl (58 kB)
    Installing collected packages: PyQt5-Qt5, PyQt5-sip, pyqt5
    Successfully installed PyQt5-Qt5-5.15.2 PyQt5-sip-12.17.0 pyqt5-5.15.11
    ```
    
    - 이 출력은 `pip`가 `PyQt5`와 그 의존성 패키지들을 찾고, 다운로드하고, 성공적으로 설치했음을 보여줍니다. 각 패키지의 버전 정보도 함께 표시됩니다.
        

### 3.3. 현재 환경의 의존성 기록 (`pip freeze > requirements.txt`)

현재 활성화된 Python 환경에 설치된 모든 패키지와 그 버전을 `requirements.txt` 파일에 기록합니다. 이 파일은 프로젝트의 "설치 설명서" 역할을 하며, 다른 환경에서 동일한 개발 환경을 재현하는 데 사용됩니다.

```
# 현재 활성화된 Python 환경에 설치된 모든 패키지와 그 버전을 'requirements.txt' 파일로 내보냅니다.
(pyqt5) C:\Users\SBA\github\pyqt5_example>pip freeze > requirements.txt
```

- **`pip freeze`**: 현재 활성화된 Python 환경에 설치된 모든 패키지(이름과 정확한 버전 포함)를 `패키지명==버전` 형식으로 나열합니다.
    
- **`>`**: 리다이렉션 연산자입니다. `pip freeze` 명령의 표준 출력을 화면에 표시하는 대신, `requirements.txt`라는 파일로 저장하라는 의미입니다.
    
- **`requirements.txt` 파일 내용 예시:**
    
    ```
    PyQt5==5.15.11
    PyQt5-Qt5==5.15.2
    PyQt5_sip==12.17.0
    # ... (다른 설치된 패키지들도 포함될 수 있습니다)
    ```
    
    이 파일은 프로젝트의 의존성을 명확하게 기록하며, Git과 같은 버전 관리 시스템에 커밋하여 다른 개발자와 공유할 수 있습니다.
    

### 3.4. 기록된 의존성으로 패키지 설치 (`pip install -r requirements.txt`)

`requirements.txt` 파일에 명시된 모든 패키지들을 현재 활성화된 환경에 설치합니다.

```
# 'requirements.txt' 파일에 나열된 모든 Python 패키지들을 현재 활성화된 환경에 설치합니다.
(pyqt5) C:\Users\SBA\github\pyqt5_example>pip install -r requirements.txt
```

- **주요 목적:**
    
    - **환경 재현 (Reproducibility):** 다른 컴퓨터나 새로운 가상 환경에서 프로젝트를 시작할 때, 이 명령 하나로 필요한 모든 라이브러리를 정확한 버전으로 설치하여 개발 당시의 환경을 완벽하게 재현할 수 있습니다.
        
    - **협업 (Collaboration):** 팀원들이 동일한 개발 환경을 쉽게 구축하고 유지할 수 있도록 합니다.
        
    - **배포 (Deployment) 및 CI/CD:** 프로젝트를 서버에 배포하거나 자동화된 빌드 시스템에서 의존성을 설치할 때 사용됩니다.
        
- **참고:** `pip install pyqt5`를 통해 이미 패키지가 설치된 환경에서 바로 이 명령을 실행한다면, `pip`는 대부분 "Requirement already satisfied" 메시지를 출력할 것입니다. 이는 패키지가 이미 설치되어 있어 추가적인 작업이 필요 없다는 의미입니다.
    

## 4. VS Code와 가상 환경 연동

새로운 가상 환경을 만든 후에는 VS Code에서 해당 환경을 사용하도록 설정해야 합니다.

### 4.1. VS Code 인터프리터 선택

```
# 1. VS Code에서 `Ctrl + Shift + P`를 눌러 Command Pallet 창을 엽니다.
# 2. 'Python: Select Interpreter'를 검색하여 선택합니다.
# 3. 나타나는 목록에서 방금 생성했던 'pyqt5' 가상 환경을 선택합니다.
# 4. 그 후에 VS Code의 터미널 창을 다시 열면, 가상 환경이 활성화된 상태로 실행됩니다.
```

## 5. PyQt5 예제 실행

`PyQt5` 모듈이 설치되고 가상 환경이 VS Code에 잘 연동되었는지 확인하기 위해 간단한 `PyQt5` 창을 띄워봅니다.

### 5.1. `blank_window.py` 파일 작성

다음 코드를 `blank_window.py`라는 파일로 저장합니다.

```
import sys
from PyQt5.QtWidgets import QApplication, QWidget

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('My First Application') # 창 제목 설정
        self.move(300, 300) # 창을 화면의 (300, 300) 위치로 이동
        self.resize(400, 200) # 창 크기를 너비 400, 높이 200으로 설정
        self.show() # 창을 화면에 표시

if __name__ == '__main__':
    app = QApplication(sys.argv) # QApplication 객체 생성 (모든 PyQt5 앱의 필수 요소)
    ex = MyApp() # MyApp 클래스의 인스턴스 생성
    sys.exit(app.exec_()) # 이벤트 루프 시작 및 종료 시 시스템 종료
```

### 5.2. PyQt5 애플리케이션 실행

VS Code 터미널에서 다음 명령어를 실행하여 PyQt5 창을 띄웁니다.

```
# 'pyqt5' 가상 환경이 활성화된 상태에서 PyQt5 애플리케이션을 실행합니다.
(pyqt5) C:\Users\SBA\github\pyqt5_example>python blank_window.py
```

- 이 명령을 실행하면 "My First Application"이라는 제목의 빈 창이 나타날 것입니다.
    

## 최종 워크플로우 요약

1. **Miniconda 설치 및 Anaconda Prompt 실행.**
    
2. **가상 환경 생성:** `conda create -n my_project_env python=3.10`
    
3. **가상 환경 활성화:** `conda activate my_project_env`
    
4. **VS Code에서 인터프리터 선택:** `Ctrl + Shift + P` -> `Python: Select Interpreter` -> `my_project_env` 선택.
    
5. **필요한 패키지 설치:** `pip install pyqt5` (필요한 다른 패키지들도 설치)
    
6. **현재 환경 의존성 기록:** `pip freeze > requirements.txt` (이 파일을 Git에 커밋)
    
7. **PyQt5 앱 실행 테스트:** `python your_app.py`
    
8. **다른 환경에서 프로젝트 설정 시:** `conda activate my_project_env` (또는 새로 생성) 후 `pip install -r requirements.txt`
    

이러한 과정을 통해 Python 개발 환경을 체계적으로 설정하고 관리할 수 있습니다.