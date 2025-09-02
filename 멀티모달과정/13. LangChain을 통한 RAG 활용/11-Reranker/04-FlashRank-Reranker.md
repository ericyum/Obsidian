# FlashRank reranker

>[FlashRank](https://github.com/PrithivirajDamodaran/FlashRank)는 기존 검색 및 `retrieval` 파이프라인에 재순위를 추가하기 위한 초경량 및 초고속 Python 라이브러리입니다. 이는 SoTA `cross-encoders`를 기반으로 합니다.
이 노트북은 문서 압축 및 `retrieval`을 위해 [flashrank](https://github.com/PrithivirajDamodaran/FlashRank)를 사용하는 방법을 보여줍니다.
## 환경설정

```python
# 설치
# !pip install -qU flashrank
```

```python
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"Document {i+1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
                for i, d in enumerate(docs)
            ]
        )
    )
```

## FlashrankRerank

간단한 예시를 위한 데이터를 로드하고 retriever 를 생성합니다.

```python
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# 문서 로드
documents = TextLoader("./data/appendix-keywords.txt", encoding="utf-8").load()

# 텍스트 분할기 초기화
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# 문서 분할
texts = text_splitter.split_documents(documents)

# 각 텍스트에 고유 ID 추가
for idx, text in enumerate(texts):
    text.metadata["id"] = idx
    
# 검색기 초기화
retriever = FAISS.from_documents(
    texts, OpenAIEmbeddings()
).as_retriever(search_kwargs={"k": 10})

# 질의문
query = "Word2Vec 에 대해서 설명해줘."

# 문서 검색
docs = retriever.invoke(query)

# 문서 출력
pretty_print_docs(docs)
```

**Output:**

```
Document 1:

Crawling

정의: 크롤링은 자동화된 방식으로 웹 페이지를 방문하여 데이터를 수집하는 과정입니다. 이는 검색 엔진 최적화나 데이터 분석에 자주 사용됩니다.
예시: 구글 검색 엔진이 인터넷 상의 웹사이트를 방문하여 콘텐츠를 수집하고 인덱싱하는 것이 크롤링입니다.
연관키워드: 데이터 수집, 웹 스크래핑, 검색 엔진

Word2Vec

정의: Word2Vec은 단어를 벡터 공간에 매핑하여 단어 간의 의미적 관계를 나타내는 자연어 처리 기술입니다. 이는 단어의 문맥적 유사성을 기반으로 벡터를 생성합니다.
예시: Word2Vec 모델에서 "왕"과 "여왕"은 서로 가까운 위치에 벡터로 표현됩니다.
연관키워드: 자연어 처리, 임베딩, 의미론적 유사성
LLM (Large Language Model)
Metadata: {'source': './data/appendix-keywords.txt', 'id': 5}
----------------------------------------------------------------------------------------------------
Document 2:

Semantic Search

...
```

이제 기본 `retriever`를 `ContextualCompressionRetriever`로 감싸고, `FlashrankRerank`를 압축기로 사용해 봅시다.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_openai import ChatOpenAI

# LLM 초기화
llm = ChatOpenAI(temperature=0)

# 문서 압축기 초기화
compressor = FlashrankRerank(model="ms-marco-MultiBERT-L-12")

# 문맥 압축 검색기 초기화
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

# 압축된 문서 검색
compressed_docs = compression_retriever.invoke(
    "Word2Vec 에 대해서 설명해줘."
)

# 문서 ID 출력
print([doc.metadata["id"] for doc in compressed_docs])
```

**Output:**

```
INFO:flashrank.Ranker:Downloading ms-marco-MultiBERT-L-12...
ms-marco-MultiBERT-L-12.zip: 100%|██████████| 98.7M/98.7M [00:00<00:00, 104MiB/s] 
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
[4, 0, 13]
```

reranker 가 적용된 후 결과를 비교합니다.

```python
# 문서 압축 결과 출력
pretty_print_docs(compressed_docs)
```

**Output:**

```
Document 1:

HuggingFace

정의: HuggingFace는 자연어 처리를 위한 다양한 사전 훈련된 모델과 도구를 제공하는 라이브러리입니다. 이는 연구자와 개발자들이 쉽게 NLP 작업을 수행할 수 있도록 돕습니다.
예시: HuggingFace의 Transformers 라이브러리를 사용하여 감정 분석, 텍스트 생성 등의 작업을 수행할 수 있습니다.
연관키워드: 자연어 처리, 딥러닝, 라이브러리

Digital Transformation

정의: 디지털 변환은 기술을 활용하여 기업의 서비스, 문화, 운영을 혁신하는 과정입니다. 이는 비즈니스 모델을 개선하고 디지털 기술을 통해 경쟁력을 높이는 데 중점을 둡니다.
예시: 기업이 클라우드 컴퓨팅을 도입하여 데이터 저장과 처리를 혁신하는 것은 디지털 변환의 예입니다.
연관키워드: 혁신, 기술, 비즈니스 모델

Crawling
Metadata: {'id': 4, 'relevance_score': 0.99972415, 'source': './data/appendix-keywords.txt'}
----------------------------------------------------------------------------------------------------
Document 2:

Semantic Search

...
```

경량 모델이라서 우선 순위가 그렇게 높게 나타나지 않는 문제점이 있다.
