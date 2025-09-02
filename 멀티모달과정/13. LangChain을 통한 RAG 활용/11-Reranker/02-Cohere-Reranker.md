# Cohere reranker

>[Cohere](https://cohere.ai/about)는 기업이 인간-기계 상호작용을 개선할 수 있도록 돕는 자연어 처리 모델을 제공하는 캐나다의 스타트업입니다.

이 노트북은 `retriever`에서 [Cohere의 rerank endpoint](https://docs.cohere.com/docs/reranking)를 사용하는 방법을 보여줍니다. 

**Cohere사이트에 회원 가입을 한 후, API KEY를 발급 받아 .env파일에 등록해야한다.**

```python
# 설치
# !pip install -qU cohere
```

```python
# langchain-cohere 버전 업그레이드
# !pip install --upgrade cohere langchain-cohere
```

**Output:**

```
Requirement already satisfied: cohere in c:\users\sba\appdata\local\pypoetry\cache\virtualenvs\langchain-kr-us6bdj1p-py3.11\lib\site-packages (5.17.0)
Requirement already satisfied: langchain-cohere in c:\users\sba\appdata\local\pypoetry\cache\virtualenvs\langchain-kr-us6bdj1p-py3.11\lib\site-packages (0.4.5)
...
```

## Cohere API 키 설정

- [API 키 발급](https://dashboard.cohere.com/api-keys)
- `.env` 파일에 `COHERE_API_KEY` 키 값에 API 키를 넣어주세요.

**참고**
- [공식 도큐먼트](https://docs.cohere.com/docs/the-cohere-platform?_gl=1*st323v*_gcl_au*MTA2ODUyNDMyNy4xNzE4MDMzMjY2*_ga*NTYzNTI5NDYyLjE3MTgwMzMyNjc.*_ga_CRGS116RZS*MTcyMTk4NzMxMi4xMS4xLjE3MjE5ODczNjIuMTAuMC4w)
- [Reranker Model 리스트](https://docs.cohere.com/docs/rerank-2)

```python
# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()
```

**Output:**

```
True
```

```python
# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("Reranker")
```

**Output:**

```
LangSmith 추적을 시작합니다.
[프로젝트명]
Reranker
```

## 사용법

```python
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )
```

**Cohere 다국어 지원 모델**

- Embedding: `embed-multilingual-v3.0`, `embed-multilingual-light-v3.0`, `embed-multilingual-v2.0`
- Reranker: `rerank-multilingual-v3.0`, `rerank-multilingual-v2.0`

```python
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings

# 문서 로드
documents = TextLoader("./data/appendix-keywords.txt", encoding="utf-8").load()

# 텍스트 분할기 초기화
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# 문서 분할
texts = text_splitter.split_documents(documents)

# 검색기 초기화
retriever = FAISS.from_documents(
    texts, CohereEmbeddings(model="embed-multilingual-v3.0")
).as_retriever(search_kwargs={"k": 10})

# 질의문
query = "Word2Vec 에 대해서 알려줘!"

# 문서 검색
docs = retriever.invoke(query)

# 문서 출력
pretty_print_docs(docs)
```

**Output:**

```
Document 1:\n\nCrawling\n\n정의: 크롤링은 자동화된 방식으로 웹 페이지를 방문하여 데이터를 수집하는 과정입니다. 이는 검색 엔진 최적화나 데이터 분석에 자주 사용됩니다.\n예시: 구글 검색 엔진이 인터넷 상의 웹사이트를 방문하여 콘텐츠를 수집하고 인덱싱하는 것이 크롤링입니다.\n연관키워드: 데이터 수집, 웹 스크래핑, 검색 엔진\n\nWord2Vec\n\n정의: Word2Vec은 단어를 벡터 공간에 매핑하여 단어 간의 의미적 관계를 나타내는 자연어 처리 기술입니다. 이는 단어의 문맥적 유사성을 기반으로 벡터를 생성합니다.\n예시: Word2Vec 모델에서 "왕"과 "여왕"은 서로 가까운 위치에 벡터로 표현됩니다.\n연관키워드: 자연어 처리, 임베딩, 의미론적 유사성\nLLM (Large Language Model)\n----------------------------------------------------------------------------------------------------\nDocument 2: ... (and so on)
```

## CohereRerank을 사용한 재정렬

이제 기본 `retriever`를 `ContextualCompressionRetriever`로 감싸보겠습니다. Cohere 재정렬 엔드포인트를 사용하여 반환된 결과를 재정렬하는 `CohereRerank`를 추가할 것입니다.
CohereRerank에서 모델 이름을 지정하는 것이 필수임을 유의하십시오!

```python
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# 문서 재정렬 모델 설정
compressor = CohereRerank(model="rerank-multilingual-v3.0")

# 문맥 압축 검색기 설정
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

# 압축된 문서 검색
compressed_docs = compression_retriever.invoke("Word2Vec 에 대해서 알려줘!")

# 압축된 문서 출력
pretty_print_docs(compressed_docs)
```

**Output:**

```
Document 1:\n\nCrawling\n\n정의: 크롤링은 자동화된 방식으로 웹 페이지를 방문하여 데이터를 수집하는 과정입니다. 이는 검색 엔진 최적화나 데이터 분석에 자주 사용됩니다.\n예시: 구글 검색 엔진이 인터넷 상의 웹사이트를 방문하여 콘텐츠를 수집하고 인덱싱하는 것이 크롤링입니다.\n연관키워드: 데이터 수집, 웹 스크래핑, 검색 엔진\n\nWord2Vec\n\n정의: Word2Vec은 단어를 벡터 공간에 매핑하여 단어 간의 의미적 관계를 나타내는 자연어 처리 기술입니다. 이는 단어의 문맥적 유사성을 기반으로 벡터를 생성합니다.\n예시: Word2Vec 모델에서 "왕"과 "여왕"은 서로 가까운 위치에 벡터로 표현됩니다.\n연관키워드: 자연어 처리, 임베딩, 의미론적 유사성\nLLM (Large Language Model)\n----------------------------------------------------------------------------------------------------\nDocument 2: ... (and so on)
```