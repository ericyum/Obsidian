# 이전 대화를 기억하는 Chain 생성방법

이 내용을 이해하기 위한 사전 지식
- `RunnableWithMessageHistory`: [https://wikidocs.net/235581](https://wikidocs.net/235581)

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
logging.langsmith("CH12-RAG")
```

**Output:**

```
LangSmith 추적을 시작합니다.
[프로젝트명]
CH12-RAG
```

## 1. 일반 Chain 에 대화기록 추가

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


# 프롬프트 정의
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 Question-Answering 챗봇입니다. 주어진 질문에 대한 답변을 제공해주세요.",
        ),
        # 대화기록용 key 인 chat_history 는 가급적 변경 없이 사용하세요!
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "#Question:\n{question}"),  # 사용자 입력을 변수로 사용
    ]
)

# llm 생성
llm = ChatOpenAI()

# 일반 Chain 생성
chain = prompt | llm | StrOutputParser()
```

대화를 기록하는 체인 생성(`chain_with_history`)

```python
# 세션 기록을 저장할 딕셔너리
store = {}


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    print(f"[대화 세션ID]: {session_ids}")
    if session_ids not in store:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,  # 세션 기록을 가져오는 함수
    input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
    history_messages_key="chat_history",  # 기록 메시지의 키
)
```

첫 번째 질문 실행

```python
chain_with_history.invoke(
    # 질문 입력
    {"question": "나의 이름은 테디입니다."},
    # 세션 ID 기준으로 대화를 기록합니다.
    config={"configurable": {"session_id": "abc123"}},
)
```

**Output:**

```
[대화 세션ID]: abc123
'안녕하세요, 테디님! 어떤 도움이 필요하신가요?'
```

이어서 질문 실행

```python
chain_with_history.invoke(
    # 질문 입력
    {"question": "내 이름이 뭐라고?"},
    # 세션 ID 기준으로 대화를 기록합니다.
    config={"configurable": {"session_id": "abc123"}},
)
```

**Output:**

```
[대화 세션ID]: abc123
'당신의 이름은 테디입니다.'
```

## 2. RAG + RunnableWithMessageHistory

먼저 일반 RAG Chain 을 생성합니다. 단, 6단계의 prompt 에 `{chat_history}` 를 꼭 추가합니다.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# 단계 1: 문서 로드(Load Documents)
loader = PDFPlumberLoader("data/SPRI_AI_Brief_2023년12월호_F.pdf")
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

#Previous Chat History:
{chat_history}

#Question: 
{question} 

#Context: 
{context} 

#Answer:"""
)

# 단계 7: 언어모델(LLM) 생성
# 모델(LLM) 을 생성합니다.
llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

# 단계 8: 체인(Chain) 생성
chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

대화를 저장할 함수 정의

```python
# 세션 기록을 저장할 딕셔너리
store = {}


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    print(f"[대화 세션ID]: {session_ids}")
    if session_ids not in store:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


# 대화를 기록하는 RAG 체인 생성
rag_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,  # 세션 기록을 가져오는 함수
    input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
    history_messages_key="chat_history",  # 기록 메시지의 키
)
```

첫 번째 질문 실행

```python
rag_with_history.invoke(
    # 질문 입력
    {"question": "삼성전자가 만든 생성형 AI 이름은?"},
    # 세션 ID 기준으로 대화를 기록합니다.
    config={"configurable": {"session_id": "rag123"}},
)
```

**Output:**

```
[대화 세션ID]: rag123
"삼성전자가 만든 생성형 AI 이름은 '삼성 가우스'입니다."
```

이어진 질문 실행

```python
rag_with_history.invoke(
    # 질문 입력
    {"question": "이전 답변을 영어로 번역해주세요."},
    # 세션 ID 기준으로 대화를 기록합니다.
    config={"configurable": {"session_id": "rag123"}},
)
```

**Output:**

```
[대화 세션ID]: rag123
'The name of the generative AI created by Samsung Electronics is "Samsung Gauss."'
```
