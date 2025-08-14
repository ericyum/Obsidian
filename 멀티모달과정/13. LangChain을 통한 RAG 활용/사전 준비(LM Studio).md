
# LM Studio란?

LM Studio는 **컴퓨터에 직접 대규모 언어 모델(LLM)을 다운로드하고 실행할 수 있게 해주는 데스크톱 애플리케이션**입니다.

쉽게 말해, 인터넷을 통해 OpenAI의 GPT나 Google의 Gemini 같은 모델에 접속하는 대신, 여러분의 PC에서 AI 모델을 직접 실행하고 대화할 수 있도록 도와주는 도구입니다.

### 주요 특징 및 장점

1. **쉬운 사용성 (GUI 기반):**
    
    - 복잡한 코드나 터미널 명령어 없이도, 직관적인 그래픽 사용자 인터페이스(GUI)를 통해 모델을 검색하고, 다운로드하고, 실행할 수 있습니다. AI 모델을 처음 다뤄보는 사람들도 쉽게 접근할 수 있습니다.
        
2. **완전한 로컬 실행:**
    
    - 한번 모델을 다운로드하면, 인터넷 연결 없이도 모델을 사용할 수 있습니다.
        
    - 모든 데이터 처리가 로컬 PC에서 이루어지므로, 민감한 정보를 외부 API로 전송할 필요가 없어 **높은 개인정보 보호**가 가능합니다.
        
    - API 사용료가 발생하지 않아 **비용 부담이 전혀 없습니다.**
        
3. **다양한 모델 지원 (GGUF):**
    
    - Hugging Face 등에서 공개된 수많은 오픈 소스 모델들을 쉽게 찾아볼 수 있습니다.
        
    - 특히 `GGUF`라는 압축 파일 형식을 지원하는데, 이는 CPU와 GPU를 효율적으로 활용하여 일반 PC에서도 LLM을 부드럽게 실행할 수 있도록 최적화된 형식입니다.
        
4. **내장된 채팅 인터페이스:**
    
    - 다운로드한 모델과 즉시 대화해볼 수 있는 채팅창이 내장되어 있습니다. 모델의 성능을 바로 테스트하거나 개인 비서처럼 활용할 수 있습니다.
        
5. **로컬 서버 API 기능:**
    
    - 가장 중요한 기능 중 하나입니다. 실행 중인 모델을 마치 OpenAI API처럼 사용할 수 있도록 로컬 서버를 열어줍니다.
        
    - 이를 통해 기존에 OpenAI API를 사용하도록 만들어진 애플리케이션이나 코드(예: LangChain, LlamaIndex 등)를 수정 없이 LM Studio에서 실행 중인 모델과 연동할 수 있습니다.
        

**요약하자면,** LM Studio는 "로컬에서 AI 모델을 실행하는 것"이라는 복잡한 작업을 매우 간단하게 만들어주는 올인원(All-in-One) 도구입니다. 개발자뿐만 아니라 일반 사용자들도 비용과 개인정보 부담 없이 AI 모델을 직접 체험하고 활용할 수 있게 해줍니다.









# 설치 방법

1. 아래의 사이트에서 다운로드를 받은 뒤 설치한다.

https://lmstudio.ai/

---

2. 개발자 모드로 선택

![[Pasted image 20250814111239.png]]

---

3. 모델 **Gemma3n E4b, exaone-4.0-1.2b** 검색 및 설치

model search에서 Gemma3n E4b, exaone-4.0-1.2b 검색 및 설치 한다.

아래 이미지에 보이는 화면 좌측의 '돋보기' 아이콘을 클릭한다.
![[Pasted image 20250814162920.png]]

그 후에 아래 이미지와 같은 창이 뜨는데 여기에서 Gemma3와 exaone이라고 검색한 후 해당 모델을 설치 한다.
![[Pasted image 20250814163051.png]]

Gemma3n E4b 모델은 RAM이 부족한 일반적인 데스크탑에서도 능히 돌릴 수 있을 정도로 용량이 가볍다. 
exaone-4.0-1.2b 모델은 LG에서 개발을 했기 때문에 한글 최적화가 되어 있으며, Gemma보다 더 가벼워서 더 빠르게 출력할 수 있다.

추가 정보)
**`Gemma3n E4b`, `exaone-4.0-1.2b` 모델 둘 다 텍스트 처리 전용 모델**이며, 이미지를 이해하고 처리하는 기능(멀티모달 기능)을 지원하지 않는다.

`llava-v1.5-7b-llamafile` 또는 `gemma-3-12b` 모델을 통해서 이미지를 처리할 수 있습니다.

---

4. GPU가 감지 되었는지 확인하기

다시 한번 '돋보기' 아이콘을 눌러서 Mission Control -> Hardware에 들어가면 다음과 같은 화면이 뜬다. 여기의 'GPUs'에서 `1 GPU detected with CUDA` 와 같이 뜨는 지를 확인한다.

![[Pasted image 20250814130852.png]]

`화면 우측 하단 톱니바퀴`를 누르는 방법으로도 해당 창을 띄울 수 있다.

![[Pasted image 20250814163408.png]]

---

5. 질문을 해서 정상적으로 작동이 되는지 확인하기

![[Pasted image 20250814131116.png]]

'채팅'에서 질문을 하면 위의 이미지 처럼 나온다. 또한 위 이미지의 맨 아랫 부분에서 볼 수 있듯이 토큰 몇개를 썼는지를 알 수 있다.

---

5. LM Studio가 사용하는 IP 및 포트 확인하기

LM Studio의 `Developer`탭 에서 IP과 포트를 확인할 수 있다.

![[Pasted image 20250814133554.png]]


![[Pasted image 20250814133606.png]]
우측 상단에 이렇게 되어있는 것을 볼 수 있다.

IP주소 `127.0.0.1`은 접속하고자 하는 컴퓨터의 IP이며, 포트 번호 `1234`는 LM Studio가 기본적(default)으로 차지하는 포트 번호이다. 또한 `127.0.0.1`은 `내 컴퓨터`의 IP주소이다.

---

6. VsCode에서 실행할 수 있도록 설정 하기
```python
# 패키지 임포트

from langchain_openai import OpenAI, ChatOpenAI

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.schema import HumanMessage, SystemMessage

from langchain.prompts import PromptTemplate

from langchain.chains import LLMChain

import os

  

# LMStudio 서버 설정

#LMSTUDIO_BASE_URL = "http://localhost:1234/v1"

# LMSTUDIO_BASE_URL = "http://172.16.2.144:1234/v1"

LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"

LMSTUDIO_API_KEY = "not-needed"  # LMStudio는 API 키가 필요없음

MODEL_NAME = "gemma-3-27b"  # LMStudio에 로드된 모델 이름

  

print("설정 완료!")
```

---

7. 설정한 사항을 바탕으로 LLM객체를 생성하기

```python
# LMStudio용 OpenAI 호환 LLM 생성

# 객체 생성

llm = OpenAI(

    base_url=LMSTUDIO_BASE_URL,

    api_key=LMSTUDIO_API_KEY,

    model=MODEL_NAME,

    temperature=0.1,  # 창의성 수준 (0.0 ~ 1.0)

    max_tokens=4096,

)

  

# 질의내용

question = "대한민국의 수도는 어디인가요?"

  

# 질의

print(f"답변: {llm.invoke(question)}")
```

---

8. 그런데 만약 외부 IP로 진행 하고 싶다면?

Developer화면에서 settings에서 "로컬 네트워크에서 제공" 목록을 체크한다.
![[Pasted image 20250814133840.png]]

![[Pasted image 20250814133855.png]]

그러면 다음과 같이 외부에서도 접근할 수 있게 된다.

![[Pasted image 20250814133924.png]]