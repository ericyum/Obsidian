# RAG 기본 구조 이해하기

## 1. 사전작업(Pre-processing) - 1~4 단계

![rag-1.png](./assets/rag-1.png)

![rag-1-graphic](./assets/rag-graphic-1.png)

사전 작업 단계에서는 데이터 소스를 Vector DB (저장소) 에 문서를 로드-분할-임베딩-저장 하는 4단계를 진행합니다.

- 1단계 문서로드(Document Load): 문서 내용을 불러옵니다.
- 2단계 분할(Text Split): 문서를 특정 기준(Chunk) 으로 분할합니다.
- 3단계 임베딩(Embedding): 분할된(Chunk) 를 임베딩하여 저장합니다.
- 4단계 벡터DB 저장: 임베딩된 Chunk 를 DB에 저장합니다.

## 2. RAG 수행(RunTime) - 5~8 단계

![rag-2.png](./assets/rag-2.png)

![](./assets/rag-graphic-2.png)

- 5단계 검색기(Retriever): 쿼리(Query) 를 바탕으로 DB에서 검색하여 결과를 가져오기 위하여 리트리버를 정의합니다. 리트리버는 검색 알고리즘이며(Dense, Sparse) 리트리버로 나뉘게 됩니다. Dense: 유사도 기반 검색, Sparse: 키워드 기반 검색
- 6단계 프롬프트: RAG 를 수행하기 위한 프롬프트를 생성합니다. 프롬프트의 context 에는 문서에서 검색된 내용이 입력됩니다. 프롬프트 엔지니어링을 통하여 답변의 형식을 지정할 수 있습니다.
- 7단계 LLM: 모델을 정의합니다.(GPT-3.5, GPT-4, Claude, etc..)
- 8단계 Chain: 프롬프트 - LLM - 출력 에 이르는 체인을 생성합니다.

## 실습에 활용한 문서

소프트웨어정책연구소(SPRi) - 2023년 12월호

- 저자: 유재흥(AI정책연구실 책임연구원), 이지수(AI정책연구실 위촉연구원)
- 링크: https://spri.kr/posts/view/23669
- 파일명: `SPRI_AI_Brief_2023년12월호_F.pdf`

_실습을 위해 다운로드 받은 파일을 `data` 폴더로 복사해 주시기 바랍니다_

## 환경설정

API KEY 를 설정합니다.

```python
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()
```

**Output:**

```
True
```

LangChain으로 구축한 애플리케이션은 여러 단계에 걸쳐 LLM 호출을 여러 번 사용하게 됩니다. 이러한 애플리케이션이 점점 더 복잡해짐에 따라, 체인이나 에이전트 내부에서 정확히 무슨 일이 일어나고 있는지 조사할 수 있는 능력이 매우 중요해집니다. 이를 위한 최선의 방법은 [LangSmith](https://smith.langchain.com)를 사용하는 것입니다.

LangSmith가 필수는 아니지만, 유용합니다. LangSmith를 사용하고 싶다면, 위의 링크에서 가입한 후, 로깅 추적을 시작하기 위해 환경 변수를 설정해야 합니다.

```python
# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install -qU langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH12-RAG")
```

**Output:**

```
LangSmith 추적을 시작합니다.
[프로젝트명]
CH12-RAG
```

## RAG 기본 파이프라인(1~8단계)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
```

아래는 기본적인 RAG 구조 이해를 위한 뼈대코드(skeleton code) 입니다.

각 단계별 모듈의 내용을 앞으로 상황에 맞게 변경하면서 문서에 적합한 구조를 찾아갈 수 있습니다.

(각 단계별로 다양한 옵션을 설정하거나 새로운 기법을 적용할 수 있습니다.)

```python
# 단계 1: 문서 로드(Load Documents)
loader = PyMuPDFLoader("data/SPRI_AI_Brief_2023년12월호_F.pdf")
docs = loader.load()
print(f"문서의 페이지수: {len(docs)}")
```

**Output:**

```
문서의 페이지수: 23
```

페이지의 내용을 출력합니다.

```python
print(docs[10].page_content)
```

**Output:**

```
SPRi AI Brief |  
2023-12월호
8
코히어, 데이터 투명성 확보를 위한 데이터 출처 탐색기 공개
n 코히어와 12개 기관이  광범위한 데이터셋에 대한 감사를 통해 원본 데이터 출처, 재라이선스 상태, 
작성자 등 다양한 정보를 제공하는 ‘데이터 출처 탐색기’ 플랫폼을 출시
n 대화형 플랫폼을 통해 개발자는 데이터셋의 라이선스 상태를 쉽게 파악할 수 있으며 데이터셋의 
구성과 계보도 추적 가능
...
```

`metadata` 를 확인합니다.

```python
docs[10].__dict__
```

**Output:**

```json
{'id': None,
 'metadata': {'producer': 'Hancom PDF 1.3.0.542',
  'creator': 'Hwp 2018 10.0.0.13462',
  'creationdate': '2023-12-08T13:28:38+09:00',
  'source': 'data/SPRI_AI_Brief_2023년12월호_F.pdf',
  'file_path': 'data/SPRI_AI_Brief_2023년12월호_F.pdf',
  'total_pages': 23,
  'format': 'PDF 1.4',
  'title': '',
  'author': 'dj',
  'subject': '',
  'keywords': '',
  'moddate': '2023-12-08T13:28:38+09:00',
  'trapped': '',
  'modDate': "D:20231208132838+09'00'",
  'creationDate': "D:20231208132838+09'00'",
  'page': 10},
 'page_content': '...',
 'type': 'Document'}
```

```python
# 단계 2: 문서 분할(Split Documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)
print(f"분할된 청크의수: {len(split_documents)}")
```

**Output:**

```
분할된 청크의수: 72
```

```python
# 단계 3: 임베딩(Embedding) 생성
embeddings = OpenAIEmbeddings()
```

```python
# 단계 4: DB 생성(Create DB) 및 저장
# 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
```

```python
for doc in vectorstore.similarity_search("구글"):
    print(doc.page_content)
```

**Output:**

```
저해상도 이미지의 고해상도 전환도 지원
n IT 전문지 테크리퍼블릭(TechRepublic)은 온디바이스 AI가 주요 기술 트렌드로 부상했다며, 
2024년부터 가우스를 탑재한 삼성 스마트폰이 메타의 라마(Llama)2를 탑재한 퀄컴 기기 및 구글 
어시스턴트를 적용한 구글 픽셀(Pixel)과 경쟁할 것으로 예상
...
```

```python
# 단계 5: 검색기(Retriever) 생성
# 문서에 포함되어 있는 정보를 검색하고 생성합니다.
retriever = vectorstore.as_retriever()
```

검색기에 쿼리를 날려 검색된 chunk 결과를 확인합니다.

```python
# 검색기에 쿼리를 날려 검색된 chunk 결과를 확인합니다.
retriever.invoke("삼성전자가 자체 개발한 AI 의 이름은?")
```

**Output:**

```
[Document(id='f8fece64-3c84-4187-af0a-d8a06df5d9e4', metadata={...}, page_content='...'), ...]
```

```python
# 단계 6: 프롬프트 생성(Create Prompt)
# 프롬프트를 생성합니다.
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

#Context: 
{context}

#Question:
{question}

#Answer:"""
)
```

```python
# 단계 7: 언어모델(LLM) 생성
# 모델(LLM) 을 생성합니다.
llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
```

```python
# 단계 8: 체인(Chain) 생성
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

생성된 체인에 쿼리(질문)을 입력하고 실행합니다.

```python
# 체인 실행(Run Chain)
# 문서에 대한 질의를 입력하고, 답변을 출력합니다.
question = "삼성전자가 자체 개발한 AI 의 이름은?"
response = chain.invoke(question)
print(response)
```

**Output:**

```
삼성전자가 자체 개발한 AI의 이름은 '삼성 가우스'입니다.
```

## 전체 코드

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 단계 1: 문서 로드(Load Documents)
loader = PyMuPDFLoader("data/SPRI_AI_Brief_2023년12월호_F.pdf")
docs = loader.load()

# 단계 2: 문서 분할(Split Documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)

# 단계 3: 임베딩(Embedding) 생성
embeddings = OpenAIEmbeddings()

# 단계 4: DB 생성(Create DB) 및 저장
# 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# 단계 5: 검색기(Retriever) 생성
# 문서에 포함되어 있는 정보를 검색하고 생성합니다.
retriever = vectorstore.as_retriever()

# 단계 6: 프롬프트 생성(Create Prompt)
# 프롬프트를 생성합니다.
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

#Context: 
{context}

#Question:
{question}

#Answer:"""
)

# 단계 7: 언어모델(LLM) 생성
# 모델(LLM) 을 생성합니다.
llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

# 단계 8: 체인(Chain) 생성
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

```python
# 체인 실행(Run Chain)
# 문서에 대한 질의를 입력하고, 답변을 출력합니다.
question = "삼성전자가 자체 개발한 AI 의 이름은?"
response = chain.invoke(question)
print(response)
```

**Output:**

```
삼성전자가 자체 개발한 AI의 이름은 '삼성 가우스'입니다.
```