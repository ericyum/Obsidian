# Llama-cpp

이 노트북은 LangChain 내에서 Llama-cpp 임베딩을 사용하는 방법에 대해 설명합니다.

lama-cpp-python 패키지를 최신 버전으로 업그레이드하는 pip 명령어입니다.

```python
# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()
```

```python
# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH08-Embeddings")
```

```python
# 설치
# !pip install -qU llama-cpp-python
```

```python
!pip install llama-cpp-python --force-reinstall --no-cache-dir
```

Visual Studio Build Tools 설치:  
 - Visual Studio 다운로드 페이지로 이동하여 Visual Studio Build Tools를 다운로드하고 실행하세요.  
 - 설치 관리자에서 'C++를 사용한 데스크톱 개발' 워크로드를 반드시 선택해야 해요.  
 - 오른쪽 설치 세부 정보에서 **'MSVC v143 - VS 2022 C++ x64/x86 빌드 도구'**와 'Windows 11 SDK' 또는 **'Windows 10 SDK'**가 선택되어 있는지 확인하세요.  
 - 설치를 완료하세요.  
  

환경 변수 설정 확인:  
 - Visual Studio Build Tools를 설치하면 필요한 컴파일러 경로가 자동으로 설정되는 경우가 많아요.  
 - 만약 문제가 계속 발생한다면, **'x64 Native Tools Command Prompt for VS 2022'**를 실행하여 pip install 명령을 실행해 보세요. 이 프롬프트는 필요한 환경 변수를 자동으로 설정해 줘요.  
  

재시도:  
 - 모든 설치와 설정이 끝난 후, 다시 pip install llama-cpp-python 명령을 실행하여 설치를 시도해 보세요.  

- `LlamaCppEmbeddings` 클래스를 `langchain_community.embeddings` 모듈에서 임포트합니다.

`LlamaCppEmbeddings`는 LLaMA 모델을 사용하여 텍스트 임베딩을 생성하는 클래스입니다. 이 클래스는 LangChain 커뮤니티에서 제공하는 확장 기능 중 하나로, C++로 구현된 LLaMA 모델을 활용하여 빠르고 효율적인 임베딩 생성을 지원합니다.

```python
from langchain_community.embeddings import LlamaCppEmbeddings
```

- `LlamaCppEmbeddings` 클래스를 사용하여 임베딩 모델을 초기화합니다.
- `model_path` 매개변수를 통해 사전 학습된 LLaMA 모델 파일("ggml-model-q4_0.bin")의 경로를 지정합니다.

ggml-model-q4_0.bin 파일을 다운로드 받는 코드

```python
!pip install hf_xet
```

```python
from huggingface_hub import hf_hub_download

repo_id = "bartowski/Llama-3.2-3B-Instruct-GGUF"
filename = "Llama-3.2-3B-Instruct-Q8_0.gguf"

try:
    # huggingface-hub 라이브러리를 사용하여 파일 다운로드
    filepath = hf_hub_download(repo_id=repo_id, filename=filename)

    print(f"File downloaded successfully to: {filepath}")

except Exception as e:
    print(f"An error occurred: {e}")
```

```python
modelFilePath = "C:/Users/SBA/.cache/huggingface/hub/models--bartowski--Llama-3.2-3B-Instruct-GGUF/snapshots/5ab33fa94d1d04e903623ae72c95d1696f09f9e8/Llama-3.2-3B-Instruct-Q8_0.gguf"
```

```python
# LlamaCpp 임베딩 모델을 초기화하고, 모델 경로를 지정합니다.
# llama = LlamaCppEmbeddings(model_path=\"/path/to/model/ggml-model-q4_0.bin\")
llama = LlamaCppEmbeddings(model_path=modelFilePath)
```

- `text` 변수에 "This is a test document."라는 문자열을 할당합니다.

```python
text = "This is a test document."  # 테스트용 문서 텍스트를 정의합니다.
```

`llama.embed_query(text)`는 입력된 텍스트를 임베딩 벡터로 변환하는 함수입니다.

- `text` 매개변수로 전달된 텍스트를 임베딩 모델에 입력하여 벡터 표현을 생성합니다.
- 생성된 임베딩 벡터는 `query_result` 변수에 저장됩니다.

이 함수는 텍스트를 벡터 공간에 매핑하여 의미적 유사성을 계산하거나 검색에 활용할 수 있는 벡터 표현을 얻는 데 사용됩니다.

```python
# 텍스트를 임베딩하여 쿼리 결과를 생성합니다.
query_result = llama.embed_query(text)
```

`llama.embed_documents([text])` 함수를 호출하여 `text` 문서를 임베딩합니다.

- `text` 문서를 리스트 형태로 `embed_documents` 함수에 전달합니다.
- `llama` 객체의 `embed_documents` 함수는 문서를 벡터 표현으로 변환합니다.
- 변환된 벡터 표현은 `doc_result` 변수에 저장됩니다.

```python
# 텍스트를 임베딩하여 문서 결과를 생성합니다.
doc_result = llama.embed_documents([text])
```
