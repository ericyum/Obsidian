# 벡터스토어 기반 검색기(VectorStore-backed Retriever)

**VectorStore 지원 검색기** 는 vector store를 사용하여 문서를 검색하는 retriever입니다.

Vector store에 구현된 **유사도 검색(similarity search)** 이나 **MMR** 과 같은 검색 메서드를 사용하여 vector store 내의 텍스트를 쿼리합니다.

아래의 코드를 실행하여 VectorStore 를 생성합니다.

```python
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()
```

```text
True
```

```python
# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH10-Retriever")
```

```text
LangSmith 추적을 시작합니다.
[프로젝트명]
CH10-Retriever
```

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# TextLoader를 사용하여 파일을 로드합니다.
loader = TextLoader("./data/appendix-keywords.txt", encoding="utf-8")

# 문서를 로드합니다.
documents = loader.load()

# 문자 기반으로 텍스트를 분할하는 CharacterTextSplitter를 생성합니다. 청크 크기는 300이고 청크 간 중복은 없습니다.
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)

# 로드된 문서를 분할합니다.
split_docs = text_splitter.split_documents(documents)

# OpenAI 임베딩을 생성합니다.
embeddings = OpenAIEmbeddings()

# 분할된 텍스트와 임베딩을 사용하여 FAISS 벡터 데이터베이스를 생성합니다.
db = FAISS.from_documents(split_docs, embeddings)
```

### VectorStore에서 VectorStoreRetriever 초기화(as_retriever)

`as_retriever` 메서드는 VectorStore 객체를 기반으로 VectorStoreRetriever를 초기화하고 반환합니다. 이 메서드를 통해 다양한 검색 옵션을 설정하여 사용자의 요구에 맞는 문서 검색을 수행할 수 있습니다.

**매개변수(parameters)**

- `**kwargs`: 검색 함수에 전달할 키워드 인자
  - `search_type`: 검색 유형 ("similarity", "mmr", "similarity_score_threshold")
  - `search_kwargs`: 추가 검색 옵션
    - `k`: 반환할 문서 수 (기본값: 4)
    - `score_threshold`: similarity_score_threshold 검색의 최소 유사도 임계값
    - `fetch_k`: MMR 알고리즘에 전달할 문서 수 (기본값: 20)
    - `lambda_mult`: MMR 결과의 다양성 조절 (0-1 사이, 기본값: 0.5)
    - `filter`: 문서 메타데이터 기반 필터링

**반환값(return)**

- `VectorStoreRetriever`: 초기화된 VectorStoreRetriever 객체

**참고**

- 다양한 검색 전략 구현 가능 (유사도, MMR, 임계값 기반)
- MMR (Maximal Marginal Relevance) 알고리즘으로 검색 결과의 다양성 조절 가능
- 메타데이터 필터링으로 특정 조건의 문서만 검색 가능
- `tags` 매개변수를 통해 검색기에 태그 추가 가능

**주의사항**

- `search_type`과 `search_kwargs`의 적절한 조합 필요
- MMR 사용 시 `fetch_k`와 `k` 값의 균형 조절 필요
- `score_threshold` 설정 시 너무 높은 값은 검색 결과가 없을 수 있음
- 필터 사용 시 데이터셋의 메타데이터 구조 정확히 파악 필요
- `lambda_mult` 값이 0에 가까울수록 다양성이 높아지고, 1에 가까울수록 유사성이 높아짐

```python
# 데이터베이스를 검색기로 사용하기 위해 retriever 변수에 할당
retriever = db.as_retriever()
```

### Retriever의 invoke()

`invoke` 메서드는 Retriever의 주요 진입점으로, 관련 문서를 검색하는 데 사용됩니다. 이 메서드는 동기적으로 Retriever를 호출하여 주어진 쿼리에 대한 관련 문서를 반환합니다.

**매개변수(parameters)**

- `input`: 검색 쿼리 문자열
- `config`: Retriever 구성 (Optional[RunnableConfig])
- `**kwargs`: Retriever에 전달할 추가 인자

**반환값(return)**

- `List[Document]`: 관련 문서 목록

```python
# 관련 문서를 검색
docs = retriever.invoke("임베딩(Embedding)은 무엇인가요?")

for doc in docs:
    print(doc.page_content)
    print("=========================================================")
```

```text
정의: 임베딩은 단어나 문장 같은 텍스트 데이터를 저차원의 연속적인 벡터로 변환하는 과정입니다. 이를 통해 컴퓨터가 텍스트를 이해하고 처리할 수 있게 합니다.
예시: "사과"라는 단어를 [0.65, -0.23, 0.17]과 같은 벡터로 표현합니다.
연관키워드: 자연어 처리, 벡터화, 딥러닝

Token
=========================================================
정의: Word2Vec은 단어를 벡터 공간에 매핑하여 단어 간의 의미적 관계를 나타내는 자연어 처리 기술입니다. 이는 단어의 문맥적 유사성을 기반으로 벡터를 생성합니다.
예시: Word2Vec 모델에서 "왕"과 "여왕"은 서로 가까운 위치에 벡터로 표현됩니다.
연관키워드: 자연어 처리, 임베딩, 의미론적 유사성
LLM (Large Language Model)
=========================================================
Semantic Search

정의: 의미론적 검색은 사용자의 질의를 단순한 키워드 매칭을 넘어서 그 의미를 파악하여 관련된 결과를 반환하는 검색 방식입니다.
예시: 사용자가 "태양계 행성"이라고 검색하면, "목성", "화성" 등과 같이 관련된 행성에 대한 정보를 반환합니다.
연관키워드: 자연어 처리, 검색 알고리즘, 데이터 마이닝

Embedding
=========================================================
정의: 크롤링은 자동화된 방식으로 웹 페이지를 방문하여 데이터를 수집하는 과정입니다. 이는 검색 엔진 최적화나 데이터 분석에 자주 사용됩니다.
예시: 구글 검색 엔진이 인터넷 상의 웹사이트를 방문하여 콘텐츠를 수집하고 인덱싱하는 것이 크롤링입니다.
연관키워드: 데이터 수집, 웹 스크래핑, 검색 엔진

Word2Vec
=========================================================
```

### Max Marginal Relevance (MMR)

`MMR(Maximal Marginal Relevance)` 방식은 쿼리에 대한 관련 항목을 검색할 때 검색된 문서의 **중복** 을 피하는 방법 중 하나입니다. 

단순히 가장 관련성 높은 항목들만을 검색하는 대신, MMR은 쿼리에 대한 **문서의 관련성** 과 이미 선택된 **문서들과의 차별성을 동시에 고려** 합니다.

- `search_type` 매개변수를 `"mmr"` 로 설정하여 **MMR(Maximal Marginal Relevance)** 검색 알고리즘을 사용합니다.
- `k`: 반환할 문서 수 (기본값: 4)
- `fetch_k`: MMR 알고리즘에 전달할 문서 수 (기본값: 20)
- `lambda_mult`: MMR 결과의 다양성 조절 (0~1, 기본값: 0.5, 0: 유사도 점수만 고려, 1: 다양성만 고려)

```python
# MMR(Maximal Marginal Relevance) 검색 유형을 지정
retriever = db.as_retriever(
    search_type="mmr", search_kwargs={"k": 2, "fetch_k": 10, "lambda_mult": 0.6}
)

# 관련 문서를 검색합니다.
docs = retriever.invoke("임베딩(Embedding)은 무엇인가요?")

# 관련 문서를 검색
for doc in docs:
    print(doc.page_content)
    print("=========================================================")
```

```text
정의: 임베딩은 단어나 문장 같은 텍스트 데이터를 저차원의 연속적인 벡터로 변환하는 과정입니다. 이를 통해 컴퓨터가 텍스트를 이해하고 처리할 수 있게 합니다.
예시: "사과"라는 단어를 [0.65, -0.23, 0.17]과 같은 벡터로 표현합니다.
연관키워드: 자연어 처리, 벡터화, 딥러닝

Token
=========================================================
Semantic Search

정의: 의미론적 검색은 사용자의 질의를 단순한 키워드 매칭을 넘어서 그 의미를 파악하여 관련된 결과를 반환하는 검색 방식입니다.
예시: 사용자가 "태양계 행성"이라고 검색하면, "목성", "화성" 등과 같이 관련된 행성에 대한 정보를 반환합니다.
연관키워드: 자연어 처리, 검색 알고리즘, 데이터 마이닝

Embedding
=========================================================
```

### 유사도 점수 임계값 검색(similarity_score_threshold)

유사도 점수 임계값을 설정하고 해당 임계값 이상의 점수를 가진 문서만 반환하는 검색 방법을 설정할 수 있습니다.

임계값을 적절히 설정함으로써 **관련성이 낮은 문서를 필터링** 하고, 질의와 **가장 유사한 문서만 선별** 할 수 있습니다.

- `search_type` 매개변수를 `"similarity_score_threshold"` 로 설정하여 유사도 점수 임계값을 기준으로 검색을 수행합니다.

- `search_kwargs` 매개변수에 `{"score_threshold": 0.8}`를 전달하여 유사도 점수 임계값을 0.8로 설정합니다. 이는 검색 결과의 **유사도 점수가 0.8 이상인 문서만 반환됨** 을 의미합니다.

```python
retriever = db.as_retriever(
    # 검색 유형을 "similarity_score_threshold 으로 설정
    search_type="similarity_score_threshold",
    # 임계값 설정
    search_kwargs={"score_threshold": 0.8},
)

# 관련 문서를 검색
for doc in retriever.invoke("Word2Vec 은 무엇인가요?"):
    print(doc.page_content)
    print("=========================================================")
```

```text
정의: Word2Vec은 단어를 벡터 공간에 매핑하여 단어 간의 의미적 관계를 나타내는 자연어 처리 기술입니다. 이는 단어의 문맥적 유사성을 기반으로 벡터를 생성합니다.
예시: "왕"과 "여왕"은 서로 가까운 위치에 벡터로 표현됩니다.
연관키워드: 자연어 처리, 임베딩, 의미론적 유사성
LLM (Large Language Model)
=========================================================
```

### top_k 설정

검색 시 사용할 `k` 와 같은 검색 키워드 인자(kwargs)를 지정할 수 있습니다.

`k` 매개변수는 검색 결과에서 반환할 상위 결과의 개수를 나타냅니다.

- `search_kwargs`에서 `k` 매개변수를 1로 설정하여 검색 결과로 반환할 문서의 수를 지정합니다.

```python
# k 설정
retriever = db.as_retriever(search_kwargs={"k": 1})

# 관련 문서를 검색
docs = retriever.invoke("임베딩(Embedding)은 무엇인가요?")

# 관련 문서를 검색
for doc in docs:
    print(doc.page_content)
    print("=========================================================")
```

```text
정의: 임베딩은 단어나 문장 같은 텍스트 데이터를 저차원의 연속적인 벡터로 변환하는 과정입니다. 이를 통해 컴퓨터가 텍스트를 이해하고 처리할 수 있게 합니다.
예시: "사과"라는 단어를 [0.65, -0.23, 0.17]과 같은 벡터로 표현합니다.
연관키워드: 자연어 처리, 벡터화, 딥러닝

Token
=========================================================
```

## 동적 설정(Configurable)

- 검색 설정을 동적으로 조정하기 위해 `ConfigurableField` 를 사용합니다.
- `ConfigurableField` 는 검색 매개변수의 고유 식별자, 이름, 설명을 설정하는 역할을 합니다.
- 검색 설정을 조정하기 위해 `config` 매개변수를 사용하여 검색 설정을 지정합니다.
- 검색 설정은 `config` 매개변수에 전달된 딕셔너리의 `configurable` 키에 저장됩니다.
- 검색 설정은 검색 쿼리와 함께 전달되며, 검색 쿼리에 따라 동적으로 조정됩니다.

```python
from langchain_core.runnables import ConfigurableField

# k 설정
retriever = db.as_retriever(search_kwargs={"k": 1}).configurable_fields(
    search_type=ConfigurableField(
        id="search_type",
        name="Search Type",
        description="The search type to use",
    ),
    search_kwargs=ConfigurableField(
        # 검색 매개변수의 고유 식별자를 설정
        id="search_kwargs",
        # 검색 매개변수의 이름을 설정
        name="Search Kwargs",
        # 검색 매개변수에 대한 설명을 작성
        description="The search kwargs to use",
    ),
)
```

아래는 동적 검색설정을 적용한 예시입니다.

```python
# 검색 설정을 지정. Faiss 검색에서 k=3로 설정하여 가장 유사한 문서 3개를 반환
config = {"configurable": {"search_kwargs": {"k": 3}}}

# 관련 문서를 검색
docs = retriever.invoke("임베딩(Embedding)은 무엇인가요?", config=config)

# 관련 문서를 검색
for doc in docs:
    print(doc.page_content)
    print("=========================================================")
```

```text
정의: 임베딩은 단어나 문장 같은 텍스트 데이터를 저차원의 연속적인 벡터로 변환하는 과정입니다. 이를 통해 컴퓨터가 텍스트를 이해하고 처리할 수 있게 합니다.
예시: "사과"라는 단어를 [0.65, -0.23, 0.17]과 같은 벡터로 표현합니다.
연관키워드: 자연어 처리, 벡터화, 딥러닝

Token
=========================================================
정의: Word2Vec은 단어를 벡터 공간에 매핑하여 단어 간의 의미적 관계를 나타내는 자연어 처리 기술입니다. 이는 단어의 문맥적 유사성을 기반으로 벡터를 생성합니다.
예시: "왕"과 "여왕"은 서로 가까운 위치에 벡터로 표현됩니다.
연관키워드: 자연어 처리, 임베딩, 의미론적 유사성
LLM (Large Language Model)
=========================================================
Semantic Search

정의: 의미론적 검색은 사용자의 질의를 단순한 키워드 매칭을 넘어서 그 의미를 파악하여 관련된 결과를 반환하는 검색 방식입니다.
예시: 사용자가 "태양계 행성"이라고 검색하면, "목성", "화성" 등과 같이 관련된 행성에 대한 정보를 반환합니다.
연관키워드: 자연어 처리, 검색 알고리즘, 데이터 마이닝

Embedding
=========================================================
```

```python
# 검색 설정을 지정. score_threshold 0.8 이상의 점수를 가진 문서만 반환
config = {
    "configurable": {
        "search_type": "similarity_score_threshold",
        "search_kwargs": {
            "score_threshold": 0.8,
        },
    }
}

# 관련 문서를 검색
docs = retriever.invoke("Word2Vec 은 무엇인가요?", config=config)

# 관련 문서를 검색
for doc in docs:
    print(doc.page_content)
    print("=========================================================")
```

```text
정의: Word2Vec은 단어를 벡터 공간에 매핑하여 단어 간의 의미적 관계를 나타내는 자연어 처리 기술입니다. 이는 단어의 문맥적 유사성을 기반으로 벡터를 생성합니다.
예시: "왕"과 "여왕"은 서로 가까운 위치에 벡터로 표현됩니다.
연관키워드: 자연어 처리, 임베딩, 의미론적 유사성
LLM (Large Language Model)
=========================================================
```

```python
# 검색 설정을 지정. mmr 검색 설정.
config = {
    "configurable": {
        "search_type": "mmr",
        "search_kwargs": {"k": 2, "fetch_k": 10, "lambda_mult": 0.6},
    }
}

# 관련 문서를 검색
docs = retriever.invoke("Word2Vec 은 무엇인가요?", config=config)

# 관련 문서를 검색
for doc in docs:
    print(doc.page_content)
    print("=========================================================")
```

```text
정의: Word2Vec은 단어를 벡터 공간에 매핑하여 단어 간의 의미적 관계를 나타내는 자연어 처리 기술입니다. 이는 단어의 문맥적 유사성을 기반으로 벡터를 생성합니다.
예시: "왕"과 "여왕"은 서로 가까운 위치에 벡터로 표현됩니다.
연관키워드: 자연어 처리, 임베딩, 의미론적 유사성
LLM (Large Language Model)
=========================================================
정의: 크롤링은 자동화된 방식으로 웹 페이지를 방문하여 데이터를 수집하는 과정입니다. 이는 검색 엔진 최적화나 데이터 분석에 자주 사용됩니다.
예시: 구글 검색 엔진이 인터넷 상의 웹사이트를 방문하여 콘텐츠를 수집하고 인덱싱하는 것이 크롤링입니다.
연관키워드: 데이터 수집, 웹 스크래핑, 검색 엔진

Word2Vec
=========================================================
```

## Upstage 임베딩과 같이 Query & Passage embedding model 이 분리된 경우

기본 retriever는 쿼리와 문서에 대해 동일한 임베딩 모델을 사용합니다.

하지만 쿼리와 문서에 대해 서로 다른 임베딩 모델을 사용하는 경우가 있습니다.

이러한 경우에는 쿼리 임베딩 모델을 사용하여 쿼리를 임베딩하고, 문서 임베딩 모델을 사용하여 문서를 임베딩합니다.

이렇게 하면 쿼리와 문서에 대해 서로 다른 임베딩 모델을 사용할 수 있습니다.

```python
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_upstage import UpstageEmbeddings

# TextLoader를 사용하여 파일을 로드합니다.
loader = TextLoader("./data/appendix-keywords.txt", encoding="utf-8")

# 문서를 로드합니다.
documents = loader.load()

# 문자 기반으로 텍스트를 분할하는 CharacterTextSplitter를 생성합니다. 청크 크기는 300이고 청크 간 중복은 없습니다.
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)

# 로드된 문서를 분할합니다.
split_docs = text_splitter.split_documents(documents)

# Upstage 임베딩을 생성합니다. 문서용 모델을 사용합니다.
doc_embedder = UpstageEmbeddings(model="solar-embedding-1-large-passage")

# 분할된 텍스트와 임베딩을 사용하여 FAISS 벡터 데이터베이스를 생성합니다.
db = FAISS.from_documents(split_docs, doc_embedder)
```

아래는 쿼리용 Upstage 임베딩을 생성하고, 쿼리 문장을 벡터로 변환하여 벡터 유사도 검색을 수행하는 예시입니다.

```python
# 쿼리용 Upstage 임베딩을 생성합니다. 쿼리용 모델을 사용합니다.
query_embedder = UpstageEmbeddings(model="solar-embedding-1-large-query")

# 쿼리 문장을 벡터로 변환합니다.
query_vector = query_embedder.embed_query("임베딩(Embedding)은 무엇인가요?")

# 벡터 유사도 검색을 수행하여 가장 유사한 2개의 문서를 반환합니다.
db.similarity_search_by_vector(query_vector, k=2)
```

```text
[Document(id='e08972c7-cb59-4283-af16-031040dfb6c1', metadata={'source': './data/appendix-keywords.txt'}, page_content='정의: 임베딩은 단어나 문장 같은 텍스트 데이터를 저차원의 연속적인 벡터로 변환하는 과정입니다. 이를 통해 컴퓨터가 텍스트를 이해하고 처리할 수 있게 합니다.\n예시: "사과"라는 단어를 [0.65, -0.23, 0.17]과 같은 벡터로 표현합니다.\n연관키워드: 자연어 처리, 벡터화, 딥러닝\n\nToken'),
 Document(id='c36c9c62-40a9-488c-bda2-023bc07fa7f8', metadata={'source': './data/appendix-keywords.txt'}, page_content='정의: Word2Vec은 단어를 벡터 공간에 매핑하여 단어 간의 의미적 관계를 나타내는 자연어 처리 기술입니다. 이는 단어의 문맥적 유사성을 기반으로 벡터를 생성합니다.\n예시: Word2Vec 모델에서 "왕"과 "여왕"은 서로 가까운 위치에 벡터로 표현됩니다.\n연관키워드: 자연어 처리, 임베딩, 의미론적 유사성\nLLM (Large Language Model)')]
```