# huggingface에서 하는 API KEY 발급 받는 방법

1. huggingface 마이페이지의 Access Tokens에서 Create New Token을 클릭
![[Pasted image 20250820161134.png]]

2. 토큰의 이름을 입력하고 create 클릭
![[Pasted image 20250820161204.png]]

3. 그러면 다음과 같은 화면이 뜨는데 이것이 토큰이다.

![[Pasted image 20250820161340.png]]







# ollama에서 모델 설치하는 방법

1. ollama 홈페이지에서 ollama를 다운 받고 설치를 한다.
https://ollama.com/

2. 원하는 모델이 있으면 해당 모델을 ollama 홈페이지에 검색. 그러면 모델을 다운 받는 코드를 보여준다.
https://ollama.com/library/gemma3:270m

3. 모델을 다운 받는다. (예시는 power shell이다.)
```
PS C:\windows\system32> ollama run gemma3:270m
PS C:\windows\system32> ollama run jmpark333/eeve
```

![[Pasted image 20250820112818.png]]



# serpapi API KEY 발급

1. 그냥 단순히 serp API 홈페이지에서 회원가입을 한 후 화면 좌측의 'api key'를 누르면 된다.
https://serpapi.com/

![[Pasted image 20250820162754.png]]

![[Pasted image 20250820143422.png]]


# wget설치

1. 이전에 만들었던 랭체인 가상환경을 킨 뒤 
![[Pasted image 20250820162820.png]]

2. 가상환경 안에서 다음 명령어를 실행한다. 

이 명령어는 스크립트를 통해서 설치해야하는 것들을 읽고 그것들을 설치하는 것을 컴퓨터가 알아서 해도 되도록 허용하는 것이다.
```
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

```
choco install wget
```

이것을 통해서 wget 명령어로 보다 쉽게 특정 URL에 있는 파일을 다운 받을 수 있다.
