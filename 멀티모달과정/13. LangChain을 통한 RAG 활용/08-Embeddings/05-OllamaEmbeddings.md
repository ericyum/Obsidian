# Ollama

Ollama는 로컬 환경에서 대규모 언어 모델(LLM)을 쉽게 실행할 수 있게 해주는 오픈소스 프로젝트입니다. 이 도구는 다양한 LLM을 간단한 명령어로 다운로드하고 실행할 수 있게 해주며, 개발자들이 AI 모델을 자신의 컴퓨터에서 직접 실험하고 사용할 수 있도록 지원합니다. Ollama는 사용자 친화적인 인터페이스와 빠른 성능으로, AI 개발 및 실험을 더욱 접근하기 쉽고 효율적으로 만들어주는 도구입니다.

- [공식 웹사이트/설치](https://ollama.com/)

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
texts = [
    "안녕, 만나서 반가워.",
    "LangChain simplifies the process of building applications with large language models",
    "랭체인 한국어 튜토리얼은 LangChain의 공식 문서, cookbook 및 다양한 실용 예제를 바탕으로 하여 사용자가 LangChain을 더 쉽고 효과적으로 활용할 수 있도록 구성되어 있습니다. ",
    "LangChain은 초거대 언어모델로 애플리케이션을 구축하는 과정을 단순화합니다.",
    "Retrieval-Augmented Generation (RAG) is an effective technique for improving AI responses.",
]
```

**지원되는 임베딩 모델 확인**

- https://ollama.com/library

`ollama pull nomic-embed-text`

```python
from langchain_community.embeddings import OllamaEmbeddings

ollama_embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    # model="chatfire/bge-m3:q8_0" # BGE-M3
)
```

`Query` 를 임베딩 합니다.

```python
# 쿼리 임베딩
embedded_query = ollama_embeddings.embed_query("LangChain 에 대해서 상세히 알려주세요.")
# 임베딩 차원 출력
len(embedded_query)
```

문서를 임베딩 합니다.

```python
# 문서 임베딩
embedded_documents = ollama_embeddings.embed_documents(texts)
```

유사도 계산 결과를 출력합니다.

```python
import numpy as np

# 질문(embedded_query): LangChain 에 대해서 알려주세요.
similarity = np.array(embedded_query) @ np.array(embedded_documents).T

# 유사도 기준 내림차순 정렬
sorted_idx = (np.array(embedded_query) @ np.array(embedded_documents).T).argsort()[::-1]

# 결과 출력
print("[Query] LangChain 에 대해서 알려주세요.\n====================================")
for i, idx in enumerate(sorted_idx):
    print(f"[{i}] 유사도: {similarity[idx]:.3f} | {texts[idx]}")
    print()
```
