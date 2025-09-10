## Iteration 기능과 사람 개입(Human-in-the-loop)

`iter()` 메서드는 에이전트의 실행 과정을 단계별로 반복할 수 있게 해주는 반복자(iterator)를 생성합니다.

중간 과정에서 사용자의 입력을 받아 계속 진행할지 묻는 기능을 제공합니다. 이를 `Human-in-the-loop` 라고 합니다.

```python
# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()
```

```python
# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install -qU langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH15-Agents")
```

먼저, 도구(tool) 를 정의합니다.

```python
# !pip install --upgrade langchain langchain-core
```

```python
from langchain.agents import tool


@tool
def add_function(a: float, b: float) -> float:
    """Adds two numbers together."""
    return a + b
```

다음으로는 `add_function` 을 사용하여 덧셈 계산을 수행하는 Agent 를 정의합니다.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor

# 도구 정의
tools = [add_function]

# LLM 생성
gpt = ChatOpenAI(model="gpt-4o-mini")

# prompt 생성
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant.",
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Agent 생성
gpt_agent = create_tool_calling_agent(gpt, tools, prompt)

# AgentExecutor 생성
agent_executor = AgentExecutor(
    agent=gpt_agent,
    tools=tools,
    verbose=False,
    max_iterations=10,
    handle_parsing_errors=True,
)
```

### AgentExecutor의 iter()

이 메서드는 AgentExecutor의 실행 과정을 단계별로 반복할 수 있게 해주는 반복자(iterator)를 생성합니다.

**함수 설명**
`iter()` 는 에이전트가 최종 출력에 도달하기까지 거치는 단계들을 순차적으로 접근할 수 있는 `AgentExecutorIterator` 객체를 반환합니다.

**주요 기능**
- **단계별 실행 접근**: 에이전트의 실행 과정을 단계별로 살펴볼 수 있습니다.

**흐름 정리**

`"114.5 + 121.2 + 34.2 + 110.1"` 의 덧셈 계산을 수행하기 위해서는 단계별로 계산이 수행되게 됩니다.

1. 114.5 + 121.2 = 235.7
2. 235.7 + 34.2 = 270.9
3. 270.9 + 110.1 = 381.0

이러한 계산 과정을 단계별로 살펴볼 수 있습니다.

이때, 

단계별로 계산 결과를 사용자에게 보여주고, 사용자가 계속 진행할지 묻습니다. (**Human-in-the-loop**)

사용자가 'y'가 아닌 다른 입력을 하면 반복 중단됩니다.

```python
# 계산할 질문 설정
question = "114.5 + 121.2 + 34.2 + 110.1 의 계산 결과는?"

# agent_executor를 반복적으로 실행
for step in agent_executor.iter({"input": question}):
    if output := step.get("intermediate_step"):
        action, value = output[0]
        if action.tool == "add_function":
            # Tool 실행 결과 출력
            print(f"\nTool Name: {action.tool}, 실행 결과: {value}")
        # 사용자에게 계속 진행할지 묻습니다.
        _continue = input("계속 진행하시겠습니다? (y/n)?:\n") or "Y"
        # 사용자가 'y'가 아닌 다른 입력을 하면 반복 중단
        if _continue.lower() != "y":
            break

# 최종 결과 출력
if "output" in step:
    print(step["output"])
```

```python
output[0]
```

```