MS Copilot <- GitHub를 소유
GitHub Copilot을 만들면서

calude code(PC의 로컬 공간에서 코드를 생성, 터미널 기반)

ChatGPT
캔버스 환경(하나의 파일을 요청하고 수정하는데 적합)
Codex(GitHub와 연동하여 코드생성의 결과물을 GitHub에 올려준다.)
파일의 읽고 쓰는 공간을 GitHub공간으로 확장

Gemini
캔버스 환경(하나의 파일을 요청하고 수정하는데 적합)
jules(GitHub와 연동해서 코드생성)
Gemini CLI

Git: 버전관리 소프트웨어 -> 리눅스 커널(오픈소스)
회사 내부 서버에 Git을 설치하고 사용한다.
GitHub : Git을 사용하는 사이트(무료, 유료로 저장 공간 제공)

![[Pasted image 20250701093817.png]]

GitHub에 레포지터리를 생성하고 관리하는 방법
1. 리모트 레포지터리를 먼서 생성
2. 로컬(PC)에 레포지터리를 복사(clone)
3. 로컬에서 파일을 생성
4. 다음과 순서에 따라 리모트 레포지터리에 올림

C:\Users\SBA\github\github_ex>git add test.txt (add를 통해서 올리고)

C:\Users\SBA\github\github_ex>git config --global user.email "ericyum9196@gmail.com"

C:\Users\SBA\github\github_ex>git config --global user.name "ericyum" (초기에만 둘을 통해 인증을 함)

C:\Users\SBA\github\github_ex>git commit -m "test" (commit, 반드시 메세지(--m)를 해야함.)

C:\Users\SBA\github\github_ex>git push (push를 하면 리모트 레포지터리에 적용)

다음은 주로 사용하는 git 명령어
git fetch
git pull
git status (push가 안될 경우에는 이 명령어로 상태를 확인할 수 있다.)

git add
git commit --m "메세지"
git push






# jules와 google cli
jules -> github와 연동해서 ai로 코드를 생성하고 푸시해줌 (https://jules.google/)
1. github계정과 연동하여 작업할 원격 레포지토리와 연결한다.
2. 원하는 사항을 적고 그에 해당하는 파일을 만들어달라고 한다.
3. 만들어 줄 텐데, 그 후 푸시해달라고 한다.
4. 원격 레포지토리로 돌아가서 새로고침을 해보면 새로운 브랜치가 생성된 것을 알 수 있으며 그 브랜치로 푸시가 된 것을 볼 수 있다.

google cli -> cmd에서 jules와 동일한 것을 해줌
jules와 다르게 로컬에 있는 이미지도 곧잘 분석해서 그에 맞는 파일을 생성한다. 

google cli를 사용하는 방법
## 빠른 시작
npx https://github.com/google-gemini/gemini-cli 
## 또는 전역 설치
npm install -g @google/gemini-cli
gemini