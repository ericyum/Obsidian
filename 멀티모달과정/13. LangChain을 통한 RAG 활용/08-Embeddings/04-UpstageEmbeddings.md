# Upstage

Upstage는 인공지능(AI) 기술, 특히 대규모 언어 모델(LLM)과 문서 AI 분야에 특화된 국내 스타트업입니다.

**API 키 발급**

- API 키 발급은 [Upstage API Console](https://console.upstage.ai/api-keys)에서 가능합니다.

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
    "Retrieval-Augmented Generation (RAG) is an effective technique for improving AI responses."
]
```

**지원되는 임베딩 모델 확인**

- https://developers.upstage.ai/docs/apis/embeddings

모델 정보

| Model                              | Release date | Context Length | Description                                                                                         |
|------------------------------------|--------------|----------------|-----------------------------------------------------------------------------------------------------|
| solar-embedding-1-large-query      | 2024-05-10   | 4000           | 4k 컨텍스트 제한이 있는 Solar-base Query Embedding 모델입니다. 이 모델은 검색 및 재정렬과 같은 정보 탐색 작업에서 사용자의 질문을 임베딩하는 데 최적화되어 있습니다. |
| solar-embedding-1-large-passage    | 2024-05-10   | 4000           | 4k 컨텍스트 제한이 있는 Solar-base Passage Embedding 모델입니다. 이 모델은 검색할 문서나 텍스트를 임베딩하는 데 최적화되어 있습니다.                                           |

```python
from langchain_upstage import UpstageEmbeddings

# 쿼리 전용 임베딩 모델
query_embeddings = UpstageEmbeddings(model="solar-embedding-1-large-query")

# 문장 전용 임베딩 모델
passage_embeddings = UpstageEmbeddings(model="solar-embedding-1-large-passage")
```

`Query` 를 임베딩 합니다.

```python
# 쿼리 임베딩
embedded_query = query_embeddings.embed_query("LangChain 에 대해서 상세히 알려주세요.")
# 임베딩 차원 출력
len(embedded_query)
```

문서를 임베딩 합니다.

```python
# 문서 임베딩
embedded_documents = passage_embeddings.embed_documents(texts)
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


```