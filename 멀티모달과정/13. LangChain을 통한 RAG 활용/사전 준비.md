
1. Git 설치하기 (이미 설치 되어 있음.)

2. PowerShell은 관리자 권한으로 실행

3. 명령어 git을 입력하여 출력이 정상적으로 실행되는지 확인
![[Pasted image 20250813131001.png]]

4. 다음의 명령어를 입력하여 Policy 를 적용
```
PS C:\Users\SBA> Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
```

### 명령어의 구성 요소

- **`Set-ExecutionPolicy`**: 실행 정책을 설정하는 PowerShell의 기본 명령어
    
- **`RemoteSigned`**: 실행 정책의 이름으로, 이 정책은 다음과 같은 규칙을 적용한다.
    
    - **로컬(Local)에서 작성된 스크립트**: 컴퓨터에서 직접 작성하거나 생성한 스크립트는 서명이 없어도 실행할 수 있다.
        
    - **원격(Remote)에서 다운로드한 스크립트**: 인터넷에서 다운로드한 스크립트는 신뢰할 수 있는 발행자의 디지털 서명이 있어야만 실행할 수 있다.
        
    - 이 정책은 보안과 편리함 사이의 균형을 잘 맞춰주기 때문에 많이 사용된다.
        
- **`-Scope CurrentUser`**: 이 정책이 적용될 범위를 지정한다. `CurrentUser`는 현재 로그인한 사용자에게만 정책을 적용하겠다는 의미입니다. 시스템 전체에 영향을 주지 않아 더 안전하다.
    
- **`-Force`**: 정책 변경 확인 메시지(예/아니오)를 건너뛰고 바로 적용하겠다는 의미한다.
    

### 왜 이 명령어를 사용해야 하는가?

기본적으로 PowerShell은 보안을 위해 스크립트 실행을 막아두는 경우가 많다. `Set-ExecutionPolicy` 명령어를 사용하면 `pyenv-win`과 같이 인터넷에서 다운로드한 도구들이 필요한 스크립트 파일을 실행할 수 있게 허용하는 것이다.

따라서 `pyenv-win`을 설치하는 과정에서 이 명령어를 사용한 것은, `pyenv`가 정상적으로 동작하는 데 필요한 PowerShell 스크립트들이 실행될 수 있도록 길을 열어주는 작업이라고 이해하면 된다.

5. 적용이 완료된 후 **Windows PowerShell** 을 껐다가 키기.

6. pyenv 클론 하기

```
PS C:\Users\SBA> git clone https://github.com/pyenv-win/pyenv-win.git "$env:USERPROFILE\.pyenv"
```

`"$env:USERPROFILE\.pyenv"`는 복제할 리포지토리의 위치를 명시적으로 지정한 것이다.

**`$env:USERPROFILE`**: 이것은 Windows PowerShell 환경 변수로, 사용자의 프로필 디렉터리(예: `C:\Users\사용자이름`)를 가리킨다.

7. 환경 변수 추가

각각 `\.pyenv\pyenv-win\`를 경로로 하는 `PYENV`, `PYENV_ROOT`, `PYENV_HOME`라는 이름의 환경 변수를 등록
```
PS C:\Users\SBA> [System.Environment]::SetEnvironmentVariable('PYENV', $env:USERPROFILE + "\.pyenv\pyenv-win\", "User") 
PS C:\Users\SBA> [System.Environment]::SetEnvironmentVariable('PYENV_ROOT', $env:USERPROFILE + "\.pyenv\pyenv-win\", "User") 
PS C:\Users\SBA> [System.Environment]::SetEnvironmentVariable('PYENV_HOME', $env:USERPROFILE + "\.pyenv\pyenv-win\", "User")
```

기존에 있는 PATh 환경 변수에 `\.pyenv\pyenv-win\bin`과 `\.pyenv\pyenv-win\shims` 경로를 추가
```
PS C:\Users\SBA> [System.Environment]::SetEnvironmentVariable('PATH', $env:USERPROFILE + "\.pyenv\pyenv-win\bin;" + $env:USERPROFILE + "\.pyenv\pyenv-win\shims;" + [System.Environment]::GetEnvironmentVariable('PATH', "User"), "User")
```

- `PYENV`, `PYENV_ROOT`, `PYENV_HOME` 같은 변수들은 `pyenv` 프로그램이 자신의 파일들이 어디에 있는지 스스로 찾을 수 있도록 돕습니다.
    
- `PATH` 환경 변수에 `bin`과 `shims` 경로를 추가하는 것은 매우 중요합니다.
    
    - `bin` 폴더에는 `pyenv` 명령어(예: `pyenv install`, `pyenv global`)의 실행 파일이 들어 있습니다. 이 경로를 `PATH`에 추가해야 터미널 어디에서든 `pyenv` 명령어를 입력할 수 있게 됩니다.
        
    - `shims` 폴더에는 `python`, `pip` 등의 명령어를 가로채서 `pyenv`가 관리하는 파이썬 버전을 대신 실행시켜주는 일종의 "프록시(proxy)" 파일이 들어 있습니다. 이 덕분에 사용자가 `python`이라고 입력하면 `pyenv`가 설정한 특정 버전의 파이썬이 실행되는 것입니다.

이제 pyenv 명령어를 통해서 손 쉽계 python 버전을 설치하고 적용할 수 있다.

8. pyenv를 이용해서 3.11버전의 python을 설치 및 확인

python 3.11 버전 설치하기
```
PS C:\Users\SBA> pyenv install 3.11
```

3.11 버전의 python을 기본 파이썬 버전으로 설정
```
PS C:\Users\SBA> pyenv global 3.11
```

현재 파이썬 버전이 3.11인지 확인하기
```
PS C:\Users\SBA> python --version
Python 3.11.9
```

9. Poetry 패키지 관리 도구를 설치

```
PS C:\Users\SBA> pip3 install poetry==1.8.5
```

10. 랭체인 클론 하기

```
PS C:\Users\SBA> cd github

PS C:\Users\SBA\github> git clone https://github.com/teddylee777/langchain-kr.git
```

11. 클론한 폴더로 이동한 뒤 파이썬 가상환경 설정
```
PS C:\Users\SBA\github> cd langchain-kr

PS C:\Users\SBA\github\langchain-kr> poetry shell
```

`poetry shell` 명령어를 `langchain-kr` 프로젝트 폴더에서 실행하면, `poetry`는 현재 `pyenv`를 통해 활성화된 파이썬 버전(3.11)을 기반으로 새로운 가상 환경을 만든다.

그럼 다음처럼 파이썬 버전이 3.11인 독립된 가상 환경 속으로 들어가게 된다.

```
(langchain-kr-py3.11) C:\Users\SBA\github\langchain-kr>
```

12. 파이썬 패키지 일괄 업데이트
```
(langchain-kr-py3.11) C:\Users\SBA\github\langchain-kr> poetry update
```

이 명령어는 `poetry`라는 도구를 사용해서 **특정 프로젝트의 "의존성(dependencies)"을 업데이트**하는 명령어입니다.

- 이 명령어는 `poetry`로 관리되는 프로젝트 폴더(예: `langchain-kr` 폴더) 안에서 실행됩니다.
    
- `poetry update`는 프로젝트에 있는 `pyproject.toml` 파일의 내용을 읽습니다. 이 파일에는 프로젝트가 필요로 하는 라이브러리(예: `langchain-core`, `openai` 등)와 버전 정보가 명시되어 있습니다.
    
- `poetry`는 해당 파일의 버전 제약 조건(예: `^0.1.0`는 `0.1.0` 이상 `0.2.0` 미만을 의미)에 맞춰, 현재 설치된 라이브러리들보다 더 최신 버전이 있는지 확인하고 설치합니다.
    
- 업데이트된 모든 라이브러리의 정확한 버전은 `poetry.lock` 파일에 기록됩니다.
     
- 마지막으로, 이 `poetry.lock` 파일에 명시된 모든 패키지들을 **프로젝트의 가상 환경**에 설치합니다.

쉽게 말해, 이것은 **"이 프로젝트가 사용하는 모든 재료(라이브러리)들을 최신 버전으로 바꿔줘"** 라는 의미입니다.

13. VS Code의 '확장'에서 각각 python과 jupyter를 검색해서 설치하고 껐다가 재실행

14. 설치가 다 되었으며, 우측 상단 "select kernel"에서 python environment를 클릭해서 설치한 가상환경이 뜨는 지를 확인