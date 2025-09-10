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
logging.langsmith("CH15-React-Agent")
```

## React Agent

![](assets/agent.png)

```python
from langchain_openai import ChatOpenAI
from langchain_teddynote.tools.tavily import TavilySearch
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# 메모리 설정
memory = MemorySaver()

# 모델 설정
model = ChatOpenAI(model_name="gpt-4o-mini")
```

## 도구 설정

### 웹 검색

```python
from langchain_teddynote.tools.tavily import TavilySearch


web_search = TavilySearch(
    topic="general",  # 뉴스 주제 (general 또는 news)
    max_results=5,  # 최대 검색 결과
    include_answer=False,
    include_raw_content=False,
    include_images=False,
    format_output=False,  # 결과 포맷팅
)

web_search.name = "web_search"
web_search.description = (
    "Use this tool to search on the web for any topic other than news."
)
```

```python
result = web_search.search("SK AI SUMMIT 2024 관련된 정보를 찾아줘")
print(result)
```

### 파일 관리

```python
from langchain_community.agent_toolkits import FileManagementToolkit

# 'tmp'라는 이름의 디렉토리를 작업 디렉토리로 설정합니다.
working_directory = "tmp"

# FileManagementToolkit 객체를 생성합니다.
file_management_tools = FileManagementToolkit(
    root_dir=str(working_directory),
).get_tools()
```

```python
# 파일 관리 도구 출력
file_management_tools
```

## Retriever 도구

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PDFPlumberLoader

# PDF 파일 로드. 파일의 경로 입력
loader = PDFPlumberLoader("data/SPRI_AI_Brief_2023년12월호_F.pdf")

# 텍스트 분할기를 사용하여 문서를 분할합니다.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# 문서를 로드하고 분할합니다.
split_docs = loader.load_and_split(text_splitter)

# VectorStore를 생성합니다.
vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())

# Retriever를 생성합니다.
pdf_retriever = vector.as_retriever()
```

```python
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.prompts import PromptTemplate

# PDF 문서를 기반으로 검색 도구 생성
retriever_tool = create_retriever_tool(
    pdf_retriever,
    "pdf_retriever",
    "Search and return information about SPRI AI Brief PDF file. It contains useful information on recent AI trends. The document is published on Dec 2023.",
    document_prompt=PromptTemplate.from_template(
        "<document><context>{page_content}</context><metadata><source>{source}</source><page>{page}</page></metadata></document>"
    ),
)
```

도구 목록을 정의합니다.

```python
tools = [web_search, *file_management_tools, retriever_tool]
tools
```

## 에이전트 생성

```python
from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(model, tools, checkpointer=memory)
```

에이전트를 시각화 합니다.

```python
from langchain_teddynote.graphs import visualize_graph

visualize_graph(agent_executor)
```

## 출력 함수 정의

```python
from langchain_teddynote.messages import stream_graph
```

```python
# Config 설정
config = {"configurable": {"thread_id": "abc123"}}
inputs = {"messages": [("human", "안녕? 내 이름은 테디야")]}

# 그래프 스트림
stream_graph(agent_executor, inputs, config, node_names=["agent"])
```

```python
config = {"configurable": {"thread_id": "abc123"}}
inputs = {"messages": [("human", "내 이름이 뭐라고?")]}

# 그래프 스트림
stream_graph(agent_executor, inputs, config, node_names=["agent"])
```

```python
config = {"configurable": {"thread_id": "abc123"}}
inputs = {
    "messages": [
        ("human", "AI Brief 보고서에서 Anthropic 투자 관련된 정보를 요약해줘.")
    ]
}
stream_graph(agent_executor, inputs, config, node_names=["agent", "tools"])
```

```python
config = {"configurable": {"thread_id": "abc123"}}
inputs = {
    "messages": [
        (
            "human",
            "한강 작가의 노벨상 수상 관련된 뉴스를 검색하고 보고서 형식에 맞게 작성해줘",
        )
    ]
}
stream_graph(agent_executor, inputs, config, node_names=["agent", "tools"])
```

```python
instruction = """
당신의 임무는 `보도자료`를 작성하는 것입니다.
----
다음의 내용을 순서대로 처리해 주세요.
1. `한강 작가의 노벨상 수상` 관련된 뉴스를 검색해 주세요.
2. 노벨상 수상 관련 뉴스를 바탕으로 보고서 / 보드자료 작성해 주세요.
3. 단, 중간에 요점 정리를 위한 markdown 테이블 형식 요약을 적극 활용해 주세요.
4. 출력 결과를 파일로 저장해 주세요. (파일 이름은 "agent_press_release.md")
"""
```

```python
config = {"configurable": {"thread_id": "abc123"}}
inputs = {"messages": [("human", instruction)]}
stream_graph(agent_executor, inputs, config, node_names=["agent", "tools"])
```
