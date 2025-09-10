# Toolkits 활용 Agent

LangChain 프레임워크를 사용하는 가장 큰 이점은 3rd-party integration 되어 있는 다양한 기능들입니다.

그 중 Toolkits 는 다양한 도구를 통합하여 제공합니다.

아래 링크에서 다양한 Tools/Toolkits 를 확인할 수 있습니다.

**참고**

- [Agent Toolkits](https://api.python.langchain.com/en/latest/community/agent_toolkits.html)

- [Tools](https://python.langchain.com/docs/integrations/tools/)

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
logging.langsmith("CH15-Agent-Projects")
```

먼저, 임시 폴더인 `tmp` 를 생성합니다.

```python
import os

if not os.path.exists("tmp"):
    os.mkdir("tmp")
```

## FileManagementToolkit

`FileManagementToolkit` 는 로컬 파일 관리를 위한 도구 모음입니다. 

### 주요 구성 요소

**파일 관리 도구들**

- `CopyFileTool`: 파일 복사
  
- `DeleteFileTool`: 파일 삭제

- `FileSearchTool`: 파일 검색

- `MoveFileTool`: 파일 이동

- `ReadFileTool`: 파일 읽기

- `WriteFileTool`: 파일 쓰기

- `ListDirectoryTool`: 디렉토리 목록 조회

**설정**

- `root_dir`: 파일 작업의 루트 디렉토리 설정 가능

- `selected_tools`: 특정 도구만 선택적으로 사용 가능


**동적 도구 생성**

- `get_tools` 메서드로 선택된 도구들의 인스턴스 생성


이 `FileManagementToolkit`은 로컬 파일 관리 작업을 자동화하거나 AI 에이전트에게 파일 조작 능력을 부여할 때 유용하게 사용할 수 있습니다. 단, 보안 측면에서 신중한 접근이 필요합니다.

```python
# FileManagementToolkit을 가져옵니다. 이 도구는 파일 관리 작업을 수행하는 데 사용됩니다.
from langchain_community.agent_toolkits import FileManagementToolkit

# 'tmp'라는 이름의 디렉토리를 작업 디렉토리로 설정합니다.
working_directory = "tmp"

# FileManagementToolkit 객체를 생성합니다.
# root_dir 매개변수에 작업 디렉토리를 지정하여 모든 파일 작업이 이 디렉토리 내에서 이루어지도록 합니다.
toolkit = FileManagementToolkit(root_dir=str(working_directory))

# toolkit.get_tools() 메서드를 호출하여 사용 가능한 모든 파일 관리 도구를 가져옵니다.
# 이 도구들은 파일 복사, 삭제, 검색, 이동, 읽기, 쓰기, 디렉토리 목록 조회 등의 기능을 제공합니다.
available_tools = toolkit.get_tools()

# 사용 가능한 도구들의 이름을 출력합니다.
print("[사용 가능한 파일 관리 도구들]")
for tool in available_tools:
    print(f"- {tool.name}: {tool.description}")
```

```python
# 도구 중 일부만 지정하여 선택하는 것도 가능합니다
tools = FileManagementToolkit(
    root_dir=str(working_directory),
    selected_tools=["read_file", "file_delete", "write_file", "list_directory"],
).get_tools()
tools
```

```python
read_tool, delete_tool, write_tool, list_tool = tools

# 파일 쓰기
write_tool.invoke({"file_path": "example.txt", "text": "Hello World!"})
```

```python
# 파일 목록 조회
print(list_tool.invoke({}))
```

```python
# 파일 삭제
print(delete_tool.invoke({"file_path": "example.txt"}))
```

```python
# 파일 목록 조회
print(list_tool.invoke({}))
```

```python
# 필요한 모듈과 클래스를 임포트합니다.
from langchain.tools import tool
from typing import List, Dict
from langchain_teddynote.tools import GoogleNews


# 최신 뉴스 검색 도구를 정의합니다.
@tool
def latest_news(k: int = 5) -> List[Dict[str, str]]:
    """Look up latest news"""
    # GoogleNews 객체를 생성합니다.
    news_tool = GoogleNews()
    # 최신 뉴스를 검색하고 결과를 반환합니다. k는 반환할 뉴스 항목의 수입니다.
    return news_tool.search_latest(k=k)


# FileManagementToolkit을 사용하여 파일 관리 도구들을 가져옵니다.
tools = FileManagementToolkit(
    root_dir=str(working_directory),
).get_tools()

# 최신 뉴스 검색 도구를 tools 리스트에 추가합니다.
tools.append(latest_news)

# 모든 도구들이 포함된 tools 리스트를 출력합니다.
tools
```

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_teddynote.messages import AgentStreamParser

# session_id 를 저장할 딕셔너리 생성
store = {}

# 프롬프트 생성
# 프롬프트는 에이전트에게 모델이 수행할 작업을 설명하는 텍스트를 제공합니다. (도구의 이름과 역할을 입력)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. "
            "Make sure to use the `latest_news` tool to find latest news. "
            "Make sure to use the `file_management` tool to manage files. ",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# LLM 생성
llm = ChatOpenAI(model="gpt-4o-mini")

# Agent 생성
agent = create_tool_calling_agent(llm, tools, prompt)

# AgentExecutor 생성
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True,
)


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

agent_stream_parser = AgentStreamParser()
```

```python
result = agent_with_chat_history.stream(
    {
        "input": "최신 뉴스 5개를 검색하고, 각 뉴스의 제목을 파일명으로 가지는 파일을 생성하고(.txt), "
        "파일의 내용은 뉴스의 내용과 url을 추가하세요. "
    },
    config={"configurable": {"session_id": "abc123"}},
)

print("Agent 실행 결과:")
for step in result:
    agent_stream_parser.process_agent_steps(step)
```

`tmp` 폴더 내부를 확인해보면 아래와 같이 파일이 생성된 것을 확인할 수 있습니다.

![](./assets/toolkits-01.png)

```python
result = agent_with_chat_history.stream(
    {
        "input": "이전에 생성한 파일 제목 맨 앞에 제목에 어울리는 emoji를 추가하여 파일명을 변경하세요. "
        "파일명도 깔끔하게 변경하세요. "
    },
    config={"configurable": {"session_id": "abc123"}},
)

print("Agent 실행 결과:")
for step in result:
    agent_stream_parser.process_agent_steps(step)
```

`tmp` 폴더 내부를 확인해보면 아래와 같이 파일명이 변경된 것을 확인할 수 있습니다.

![](./assets/toolkits-02.png)

```python
result = agent_with_chat_history.stream(
    {
        "input": "이전에 생성한 모든 파일을 `news` 폴더를 생성한 뒤 해당 폴더에 모든 파일을 복사하세요. "
        "내용도 동일하게 복사하세요. "
    },
    config={"configurable": {"session_id": "abc123"}},
)

print("Agent 실행 결과:")
for step in result:
    agent_stream_parser.process_agent_steps(step)
```

`tmp` 폴더 내부를 확인해보면 아래와 같이 `news` 폴더가 생성되고 파일이 복사된 것을 확인할 수 있습니다.

![](./assets/toolkits-03.png)

```python
result = agent_with_chat_history.stream(
    {"input": "news 폴더 안에 있는 모든 .txt 파일을 삭제하세요."},
    config={"configurable": {"session_id": "abc123"}},
)

print("Agent 실행 결과:")
for step in result:
    agent_stream_parser.process_agent_steps(step)
```

`tmp` 폴더 내부를 확인해보면 아래와 같이 `news` 폴더를 제외한 모든 파일이 삭제된 것을 확인할 수 있습니다.

![](./assets/toolkits-04.png)
