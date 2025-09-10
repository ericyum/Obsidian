# OpenAI 외 도구 호출 에이전트(Tool Calling Agent)

OpenAI 외에도 `Anthropic`, `Google Gemini`, `Together.ai`, `Ollama`, `Mistral`과 같은 더 광범위한 공급자 구현을 지원합니다.

이번 챕터에서는 다양한 LLM 을 사용하여 도구 호출 에이전트를 생성하고 실행하는 방법을 살펴보겠습니다.

**참고 링크**

- [LangChain 공식 도큐먼트](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/tool_calling/)

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
logging.langsmith("CH15-Agents")
```

```python
from langchain.tools import tool
from typing import List, Dict
from langchain_teddynote.tools import GoogleNews


# 도구 정의
@tool
def search_news(query: str) -> List[Dict[str, str]]:
    """Search Google News by input keyword"""
    news_tool = GoogleNews()
    return news_tool.search_by_keyword(query, k=5)


print(f"도구 이름: {search_news.name}")
print(f"도구 설명: {search_news.description}")
```

```python
# tools 정의
tools = [search_news]
```

## Agent 용 프롬프트 생성

- `chat_history` : 이전 대화 내용을 저장하는 변수 (멀티턴을 지원하지 않는다면, 생략 가능합니다.)
- `agent_scratchpad` : 에이전트가 임시로 저장하는 변수
- `input` : 사용자의 입력

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent

# 프롬프트 생성
# 프롬프트는 에이전트에게 모델이 수행할 작업을 설명하는 텍스트를 제공합니다. (도구의 이름과 역할을 입력)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. "
            "Make sure to use the `search_news` tool for searching keyword related news.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
```

## Tool Calling 을 지원하는 다양한 LLM 목록

실습 진행을 위해서는 아래 내용을 설정해야 합니다.

**Anthropic**

- [Anthropic API 키 발급 관련](https://console.anthropic.com/settings/keys)
- `.env` 파일 내 `ANTHROPIC_API_KEY` 에 발급받은 키를 설정하세요

**Gemini**

- [Gemini API 키 발급 관련](https://aistudio.google.com/app/apikey?hl=ko)
- `.env` 파일 내 `GOOGLE_API_KEY` 에 발급받은 키를 설정하세요

**Together AI**

- [Together AI API 키 발급 관련](https://api.together.ai/)
- `.env` 파일 내 `TOGETHER_API_KEY` 에 발급받은 키를 설정하세요

**Ollama**

- [Ollama Tool Calling 지원 모델 리스트](https://ollama.com/search?c=tools)
- [이번 실습에 사용할 llama3.1 모델](https://ollama.com/library/llama3.1)
- 터미널 창에 `ollama pull llama3.1` 명령어를 입력하여 모델을 다운로드 받습니다.
- 이전에 Ollama 를 사용하지 않았다면, [Ollama](https://wikidocs.net/233805) 를 참고해 주세요.

langchain-ollama 설치를 한 뒤 진행해 주세요.

```python
# !pip install -qU langchain-ollama==0.1.3
```

```python
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import os

# GPT-4o-mini
gpt = ChatOpenAI(model="gpt-4o-mini")

# # Claude-3-5-sonnet
# claude = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)

# Gemini-1.5-pro-latest
gemini = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

# # Llama-3.1-70B-Instruct-Turbo
# llama = ChatOpenAI(
#     base_url="https://api.together.xyz/v1",
#     api_key=os.environ["TOGETHER_API_KEY"],
#     model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
# )

# # Llama-3.1
# ollama = ChatOllama(model="llama3.1", temperature=0)

# # Qwen2.5 7B (한글 성능 괜찮은 편)
# qwen = ChatOllama(
#     model="qwen2.5:latest",
# )
```

LLM 기반으로 Agent 를 생성합니다.

```python
from langchain.agents import create_tool_calling_agent

# Agent 생성
gpt_agent = create_tool_calling_agent(gpt, tools, prompt)
# claude_agent = create_tool_calling_agent(claude, tools, prompt)
gemini_agent = create_tool_calling_agent(gemini, tools, prompt)
# llama_agent = create_tool_calling_agent(llama, tools, prompt)
# ollama_agent = create_tool_calling_agent(ollama, tools, prompt)
# qwen_agent = create_tool_calling_agent(qwen, tools, prompt)
```

## AgentExecutor 생성 후 실행 및 결과 확인


```python
from langchain.agents import AgentExecutor

# gpt_agent 실행
agent_executor = AgentExecutor(
    agent=gpt_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

result = agent_executor.invoke({"input": "AI 투자와 관련된 뉴스를 검색해 주세요."})

print("Agent 실행 결과:")
print(result["output"])
```

다양한 llm을 사용하여 에이전트를 실행합니다.

다음은 입력받은 llm을 사용하여 Agent 를 생성하고 실행하여 결과를 출력하는 함수입니다.

```python
def execute_agent(llm, tools, input_text, label):
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    result = executor.invoke({"input": input_text})
    print(f"[{label}] 결과입니다.")
    if isinstance(result["output"], list) and len(result["output"]) > 0:
        for item in result["output"]:
            if "text" in item:
                print(item["text"])
    elif isinstance(result["output"], str):
        print(result["output"])
    else:
        print(result["output"])
```

각 llm 별로 에이전트를 생성하고 실행하여 결과를 출력합니다.

```python
query = (
    "AI 투자와 관련된 뉴스를 검색하고, 결과를 Instagram 게시글 형식으로 작성해 주세요."
)
```

```python
# gpt
execute_agent(gpt, tools, query, "gpt")
```

```python
# # claude
# execute_agent(claude, tools, query, "claude")
```

```python
# gemini
execute_agent(gemini, tools, query, "gemini")
```

```python
# # llama3.1 70B (Together.ai)
# execute_agent(
#     llama,
#     tools,
#     "Search AI related news and write it in Instagram post format",
#     "llama3.1 70B",
# )
```

```python
# # llama3.1 8B (ollama)
# execute_agent(ollama, tools, query, "llama3.1(Ollama)")
```

```python
# # qwen2.5 7B (ollama)
# query = "AI 투자와 관련된 뉴스를 검색하고, 결과를 Instagram 게시글 형식으로 작성해 주세요. 한글로 답변하세요!"

# execute_agent(qwen, tools, query, "qwen2.5(Ollama)")
```
