# Jina Reranker

이 노트북은 문서 압축 및 `retrieval`을 위해 **Jina Reranker**를 사용하는 방법을 보여줍니다.
- [API 키 발급](https://jina.ai/reranker)
`.env` 파일에 아래와 같이 추가합니다.
```
JINA_API_KEY="YOUR_JINA_API_KEY"
```

**JinaAI사이트에 회원 가입을 한 후, API KEY를 발급 받아 .env파일에 등록해야한다.**

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
logging.langsmith("Reranker")
```

## Jina Reranker

간단한 예시를 위한 데이터를 로드하고 retriever 를 생성합니다.

```python
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )
```

```python
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# 문서 로드
documents = TextLoader("./data/appendix-keywords.txt").load()

# 텍스트 분할기 초기화
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# 문서 분할
texts = text_splitter.split_documents(documents)

# 검색기 초기화
retriever = FAISS.from_documents(texts, OpenAIEmbeddings()).as_retriever(
    search_kwargs={"k": 10}
)

# 질의문
query = "Word2Vec 에 대해서 설명해줘."

# 문서 검색
docs = retriever.invoke(query)

# 문서 출력
pretty_print_docs(docs)
```

## JinaRerank를 사용한 재정렬 수행

이제 `Jina Reranker`를 압축기로 사용하여 기본 `retriever`를 `ContextualCompressionRetriever`로 감싸봅시다.

```python
from ast import mod
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import JinaRerank

# JinaRerank 압축기 초기화
compressor = JinaRerank(model="jina-reranker-v2-base-multilingual", top_n=3)

# 문서 압축 검색기 초기화
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

# 관련 문서 검색 및 압축
compressed_docs = compression_retriever.invoke("Word2Vec 에 대해서 설명해줘.")
```

```python
# 압축된 문서의 보기 좋게 출력
pretty_print_docs(compressed_docs)
```