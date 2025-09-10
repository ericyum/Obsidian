# Agentic RAG

이번 챕터에서는 문서 검색을 통해 최신 정보에 접근하여 검색 결과를 가지고 답변을 생성하는 에이전트를 만들어 보겠습니다.

질문에 따라 문서를 검색하여 답변하거나, 인터넷 검색 도구를 활용하여 답변하는 에이전트를 만들어 보겠습니다.

**참고**

- RAG 를 수행하되, Agent 를 활용하여 RAG 를 수행한다면 이를 **Agentic RAG** 라고 부릅니다.

## 도구(Tools)

Agent 가 활용할 도구를 정의하여 Agent 가 추론(reasoning)을 수행할 때 활용하도록 만들 수 있습니다.

Tavily Search 는 그 중 대표적인 **검색 도구** 입니다. 검색을 통해 최신 정보에 접근하여 검색 결과를 가지고 답변을 생성할 수 있습니다. 도구는 이처럼 검색 도구 뿐만아니라 Python 코드를 실행할 수 있는 도구, 직접 정의한 함수를 실행하는 도구 등 다양한 종류와 방법론을 제공합니다.

### 웹 검색도구: Tavily Search

LangChain에는 Tavily 검색 엔진을 도구로 쉽게 사용할 수 있는 내장 도구가 있습니다.

Tavily Search 를 사용하기 위해서는 API KEY를 발급 받아야 합니다.

- [Tavily Search API 발급받기](https://app.tavily.com/sign-in)

발급 받은 API KEY 를 다음과 같이 환경변수에 등록 합니다.

`.env` 파일에 다음과 같이 등록합니다.

- `TAVILY_API_KEY=발급 받은 Tavily API KEY 입력`

```python
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()
```

```python
# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install -qU langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH15-Agentic-RAG")
```

```python
# TavilySearchResults 클래스를 langchain_community.tools.tavily_search 모듈에서 가져옵니다.
from langchain_community.tools.tavily_search import TavilySearchResults

# TavilySearchResults 클래스의 인스턴스를 생성합니다
# k=6은 검색 결과를 6개까지 가져오겠다는 의미입니다
search = TavilySearchResults(k=6)
```

`search.invoke` 함수는 주어진 문자열에 대한 검색을 실행합니다.

`invoke()` 함수에 검색하고 싶은 검색어를 넣어 검색을 수행합니다.

```python
# 검색 결과를 가져옵니다.
search.invoke("판교 카카오 프렌즈샵 아지트점의 전화번호는 무엇인가요?")
```

### 문서 기반 문서 검색 도구: Retriever

우리의 데이터에 대해 조회를 수행할 retriever도 생성합니다.

**실습에 활용한 문서**

소프트웨어정책연구소(SPRi) - 2023년 12월호

- 저자: 유재흥(AI정책연구실 책임연구원), 이지수(AI정책연구실 위촉연구원)
- 링크: https://spri.kr/posts/view/23669
- 파일명: `SPRI_AI_Brief_2023년12월호_F.pdf`

_실습을 위해 다운로드 받은 파일을 `data` 폴더로 복사해 주시기 바랍니다_

이 코드는 웹 기반 문서 로더, 문서 분할기, 벡터 저장소, 그리고 OpenAI 임베딩을 사용하여 문서 검색 시스템을 구축합니다.

여기서는 PDF 문서를 `FAISS` DB 에 저장하고 조회하는 retriever 를 생성합니다.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader

# PDF 파일 로드. 파일의 경로 입력
loader = PyPDFLoader("data/SPRI_AI_Brief_2023년12월호_F.pdf")

# 텍스트 분할기를 사용하여 문서를 분할합니다.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# 문서를 로드하고 분할합니다.
split_docs = loader.load_and_split(text_splitter)

# VectorStore를 생성합니다.
vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())

# Retriever를 생성합니다.
retriever = vector.as_retriever()
```

이 함수는 `retriever` 객체의 `invoke()` 를 사용하여 사용자의 질문에 대한 가장 **관련성 높은 문서** 를 찾는 데 사용됩니다.

```python
# 문서에서 관련성 높은 문서를 가져옵니다.
retriever.invoke("삼성전자가 개발한 생성형 AI 관련 내용을 문서에서 찾아줘")
```

이제 우리가 검색을 수행할 인덱스를 채웠으므로, 이를 에이전트가 제대로 사용할 수 있는 도구로 쉽게 변환할 수 있습니다.

`create_retriever_tool` 함수로 `retriever` 를 도구로 변환합니다.

**즉, retriever를 tool로 등록한 것**

```python
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    name="pdf_search",  # 도구의 이름을 입력합니다.
    description="use this tool to search information from the PDF document",  # 도구에 대한 설명을 자세히 기입해야 합니다!!
)
```

### Agent 가 사용할 도구 목록 정의

이제 두 가지를 모두 만들었으므로, Agent 가 사용할 도구 목록을 만들 수 있습니다.

`tools` 리스트는 `search`와 `retriever_tool`을 포함합니다. 

```python
# tools 리스트에 search와 retriever_tool을 추가합니다.
tools = [search, retriever_tool]
```

## Agent 생성

이제 도구를 정의했으니 에이전트를 생성할 수 있습니다. 

먼저, Agent 가 활용할 LLM을 정의하고, Agent 가 참고할 Prompt 를 정의합니다.

**참고**
- 멀티턴 대화를 지원하지 않는다면 "chat_history" 를 제거해도 좋습니다.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# LLM 정의
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Prompt 정의
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. "
            "Make sure to use the `pdf_search` tool for searching information from the PDF document. "
            "If you can't find the information from the PDF document, use the `search` tool for searching information from the web.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
```

다음으로는 Tool Calling Agent 를 생성합니다.

```python
from langchain.agents import create_tool_calling_agent

# tool calling agent 생성
agent = create_tool_calling_agent(llm, tools, prompt)
```

마지막으로, 생성한 `agent` 를 실행하는 `AgentExecutor` 를 생성합니다.

**참고**

- `verbose=False` 로 설정하여 중간 단계 출력을 생략하였습니다.

```python
from langchain.agents import AgentExecutor

# AgentExecutor 생성
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
```

## 에이전트 실행하기

이제 몇 가지 질의에 대해 에이전트를 실행할 수 있습니다!

현재 이러한 모든 질의는 **상태(Stateless) 가 없는** 질의입니다(이전 상호작용을 기억하지 않습니다).

`agent_executor` 객체의 `invoke` 메소드는 딕셔너리 형태의 인자를 받아 처리합니다. 이 예제에서는 `input` 키에 `hi!` 값을 할당한 딕셔너리를 인자로 전달하고 있습니다. 이는 일반적으로 AI 에이전트, 함수 실행기, 또는 명령 처리기 등의 객체에서 입력을 처리하기 위해 사용됩니다.

```python
from langchain_teddynote.messages import AgentStreamParser

# 각 단계별 출력을 위한 파서 생성
agent_stream_parser = AgentStreamParser()
```

```python
# 질의에 대한 답변을 스트리밍으로 출력 요청
result = agent_executor.stream(
    {"input": "2024년 프로야구 플레이오프 진출한 5개 팀을 검색하여 알려주세요."}
)

for step in result:
    # 중간 단계를 parser 를 사용하여 단계별로 출력
    agent_stream_parser.process_agent_steps(step)
```

`agent_executor` 객체의 `invoke` 메소드를 사용하여, 질문을 입력으로 제공합니다.

```python
# 질의에 대한 답변을 스트리밍으로 출력 요청
result = agent_executor.stream(
    {"input": "삼성전자가 자체 개발한 생성형 AI 관련된 정보를 문서에서 찾아주세요."}
)

for step in result:
    # 중간 단계를 parser 를 사용하여 단계별로 출력
    agent_stream_parser.process_agent_steps(step)
```

## 이전 대화내용 기억하는 Agent

이전의 대화내용을 기억하기 위해서는 `RunnableWithMessageHistory` 를 사용하여 `AgentExecutor` 를 감싸줍니다.

`RunnableWithMessageHistory` 에 대한 자세한 내용은 아래 링크를 참고해 주세요.

**참고**
- [RunnableWithMessageHistory](https://wikidocs.net/254682)

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# session_id 를 저장할 딕셔너리 생성
store = {}


# session_id 를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in store:  # session_id 가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


# 채팅 메시지 기록이 추가된 에이전트를 생성합니다.
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # 대화 session_id
    get_session_history,
    # 프롬프트의 질문이 입력되는 key: "input"
    input_messages_key="input",
    # 프롬프트의 메시지가 입력되는 key: "chat_history"
    history_messages_key="chat_history",
)
```

```python
# 질의에 대한 답변을 스트리밍으로 출력 요청
response = agent_with_chat_history.stream(
    {"input": "삼성전자가 개발한 생성형 AI 관련된 정보를 문서에서 찾아주세요."},
    # session_id 설정
    config={"configurable": {"session_id": "abc123"}},
)

# 출력 확인
for step in response:
    agent_stream_parser.process_agent_steps(step)
```

```python
response = agent_with_chat_history.stream(
    {"input": "이전의 답변을 영어로 번역해 주세요."},
    # session_id 설정
    config={"configurable": {"session_id": "abc123"}},
)

# 출력 확인
for step in response:
    agent_stream_parser.process_agent_steps(step)
```

## Agent 템플릿

다음은 전체 템플릿 코드 입니다.

```python
# 필요한 모듈 import
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.document_loaders import PyMuPDFLoader
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_teddynote.messages import AgentStreamParser

########## 1. 도구를 정의합니다 ##########

### 1-1. Search 도구 ###
# TavilySearchResults 클래스의 인스턴스를 생성합니다
# k=6은 검색 결과를 6개까지 가져오겠다는 의미입니다
search = TavilySearchResults(k=6)

### 1-2. PDF 문서 검색 도구 (Retriever) ###
# PDF 파일 로드. 파일의 경로 입력
loader = PyMuPDFLoader("data/SPRI_AI_Brief_2023년12월호_F.pdf")

# 텍스트 분할기를 사용하여 문서를 분할합니다.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# 문서를 로드하고 분할합니다.
split_docs = loader.load_and_split(text_splitter)

# VectorStore를 생성합니다.
vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())

# Retriever를 생성합니다.
retriever = vector.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    name="pdf_search",  # 도구의 이름을 입력합니다.
    description="use this tool to search information from the PDF document",  # 도구에 대한 설명을 자세히 기입해야 합니다!!
)

### 1-3. tools 리스트에 도구 목록을 추가합니다 ###
# tools 리스트에 search와 retriever_tool을 추가합니다.
tools = [search, retriever_tool]

########## 2. LLM 을 정의합니다 ##########
# LLM 모델을 생성합니다.
llm = ChatOpenAI(model="gpt-4o", temperature=0)

########## 3. Prompt 를 정의합니다 ##########

# Prompt 를 정의합니다 - 이 부분을 수정할 수 있습니다!
# Prompt 정의
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. "
            "Make sure to use the `pdf_search` tool for searching information from the PDF document. "
            "If you can't find the information from the PDF document, use the `search` tool for searching information from the web.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

########## 4. Agent 를 정의합니다 ##########

# 에이전트를 생성합니다.
# llm, tools, prompt를 인자로 사용합니다.
agent = create_tool_calling_agent(llm, tools, prompt)

########## 5. AgentExecutor 를 정의합니다 ##########

# AgentExecutor 클래스를 사용하여 agent와 tools를 설정하고, 상세한 로그를 출력하도록 verbose를 True로 설정합니다.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

########## 6. 채팅 기록을 수행하는 메모리를 추가합니다. ##########

# session_id 를 저장할 딕셔너리 생성
store = {}


# session_id 를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in store:  # session_id 가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


# 채팅 메시지 기록이 추가된 에이전트를 생성합니다.
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # 대화 session_id
    get_session_history,
    # 프롬프트의 질문이 입력되는 key: "input"
    input_messages_key="input",
    # 프롬프트의 메시지가 입력되는 key: "chat_history"
    history_messages_key="chat_history",
)

########## 7. Agent 파서를 정의합니다. ##########
agent_stream_parser = AgentStreamParser()
```

```python
########## 8. 에이전트를 실행하고 결과를 확인합니다. ##########

# 질의에 대한 답변을 출력합니다.
response = agent_with_chat_history.stream(
    {"input": "구글이 앤스로픽에 투자한 금액을 문서에서 찾아줘"},
    # 세션 ID를 설정합니다.
    # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않습니다
    config={"configurable": {"session_id": "abc123"}},
)

for step in response:
    agent_stream_parser.process_agent_steps(step)
```

```python
########## 8. 에이전트를 실행하고 결과를 확인합니다. ##########

# 질의에 대한 답변을 출력합니다.
response = agent_with_chat_history.stream(
    {"input": "이전의 답변을 영어로 번역해 주세요"},
    # 세션 ID를 설정합니다.
    # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않습니다
    config={"configurable": {"session_id": "abc123"}},
)

for step in response:
    agent_stream_parser.process_agent_steps(step)
```

```python
########## 8. 에이전트를 실행하고 결과를 확인합니다. ##########

# 질의에 대한 답변을 출력합니다.
response = agent_with_chat_history.stream(
    {
        "input": "2024년 프로야구 플레이오프 진출 5개팀을 검색해서 알려주세요. 한글로 답변하세요"
    },
    # 세션 ID를 설정합니다.
    # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않습니다
    config={"configurable": {"session_id": "abc456"}},
)

for step in response:
    agent_stream_parser.process_agent_steps(step)
```

```python
########## 8. 에이전트를 실행하고 결과를 확인합니다. ##########

# 질의에 대한 답변을 출력합니다.
response = agent_with_chat_history.stream(
    {"input": "이전의 답변을 SNS 게시글 형태로 100자 내외로 작성하세요."},
    # 세션 ID를 설정합니다.
    # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않습니다
    config={"configurable": {"session_id": "abc456"}},
)

for step in response:
    agent_stream_parser.process_agent_steps(step)
```

```python
########## 8. 에이전트를 실행하고 결과를 확인합니다. ##########

# 질의에 대한 답변을 출력합니다.
response = agent_with_chat_history.stream(
    {"input": "이전의 답변에 한국 시리즈 일정을 추가하세요."},
    # 세션 ID를 설정합니다.
    # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않습니다
    config={"configurable": {"session_id": "abc456"}},
)

for step in response:
    agent_stream_parser.process_agent_steps(step)
```
