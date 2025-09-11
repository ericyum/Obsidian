

# LangGraph 챗봇 구축

먼저 `LangGraph`를 사용하여 간단한 챗봇을 만들어 보겠습니다. 이 챗봇은 사용자 메시지에 직접 응답할 것입니다. 비록 간단하지만, `LangGraph`로 구축하는 핵심 개념을 설명할 것입니다. 이 섹션이 끝나면 기본적인 챗봇을 구축하게 될 것입니다.
`StateGraph`를 생성하는 것으로 시작하십시오. `StateGraph` 객체는 챗봇의 구조를 "상태 기계(State Machine)"로 정의합니다. 
`nodes`를 추가하여 챗봇이 호출할 수 있는 `llm`과 함수들을 나타내고, `edges`를 추가하여 봇이 이러한 함수들 간에 어떻게 전환해야 하는지를 지정합니다.


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
logging.langsmith("CH17-LangGraph-Modules")
```


## Step-by-Step 개념 이해하기!



### STEP 1. 상태(State) 정의


```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    # 메시지 정의(list type 이며 add_messages 함수를 사용하여 메시지를 추가)
    messages: Annotated[list, add_messages]
```


### STEP 2. 노드(Node) 정의



다음으로 "`chatbot`" 노드를 추가합니다. 
노드는 작업의 단위를 나타내며, 일반적으로 정규 **Python** 함수입니다.


```python
from langchain_openai import ChatOpenAI

# LLM 정의
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 챗봇 함수 정의
def chatbot(state: State):
    # 메시지 호출 및 반환
    return {"messages": [llm.invoke(state["messages"])]}
```


### STEP 3. 그래프(Graph) 정의, 노드 추가


```python
# 그래프 생성
graph_builder = StateGraph(State)

# 노드 이름, 함수 혹은 callable 객체를 인자로 받아 노드를 추가
graph_builder.add_node("chatbot", chatbot)
```


**참고**

- `chatbot` 노드 함수는 현재 `State`를 입력으로 받아 "messages"라는 키 아래에 업데이트된 `messages` 목록을 포함하는 사전(TypedDict) 을 반환합니다. 

- `State`의 `add_messages` 함수는 이미 상태에 있는 메시지에 llm의 응답 메시지를 추가합니다. 



### STEP 4. 그래프 엣지(Edge) 추가

다음으로, `START` 지점을 추가하세요. `START`는 그래프가 실행될 때마다 **작업을 시작할 위치** 입니다.


```python
# 시작 노드에서 챗봇 노드로의 엣지 추가
graph_builder.add_edge(START, "chatbot")
```



마찬가지로, `END` 지점을 설정하십시오. 이는 그래프 흐름의 종료(끝지점) 를 나타냅니다.


```python
# 그래프에 엣지 추가
graph_builder.add_edge("chatbot", END)
```


### STEP 5. 그래프 컴파일(compile)



마지막으로, 그래프를 실행할 수 있어야 합니다. 이를 위해 그래프 빌더에서 "`compile()`"을 호출합니다. 이렇게 하면 상태에서 호출할 수 있는 "`CompiledGraph`"가 생성됩니다.


```python
# 그래프 컴파일
graph = graph_builder.compile()
```


### STEP 6. 그래프 시각화



이제 그래프를 시각화해봅시다.


```python
from langchain_teddynote.graphs import visualize_graph

# 그래프 시각화
visualize_graph(graph)
```


### STEP 7. 그래프 실행



이제 챗봇을 실행해봅시다!



### LangGraph 스트림 처리 과정

---

`graph.stream()` 메서드를 사용한 데이터 처리 과정은 다음과 같습니다.

1.  **`event` 생성**: `graph.stream()`은 그래프 내의 각 노드가 실행을 완료할 때마다 **`event`**라는 딕셔너리를 반환합니다. 이 딕셔너리는 해당 노드가 상태를 어떻게 업데이트했는지에 대한 정보를 담고 있습니다.

2.  **`event` 객체의 구조**: `event` 객체는 `{\"노드 이름\": 상태 업데이트 내용}` 형식으로 구성됩니다.

    ```json
    {
      \"chatbot\": {
        \"messages\": [
          { \"content\": \"새로운 챗봇 응답\", \"type\": \"ai\" }
        ]
      }
    }
    ```

3.  **`event.values()` 순회**: `for value in event.values():` 구문은 `event` 딕셔너리의 **값(value)** 부분만 가져와서 순차적으로 처리합니다.

    * **단일 노드**: 노드가 하나만 있다면, `event.values()`는 하나의 딕셔너리(`{\"messages\": [...]}`)만 포함합니다.
    * **병렬 노드**: 여러 노드가 동시에 실행되는 복잡한 구조라면, `event.values()`에 여러 노드의 출력값이 포함됩니다.

4.  **최신 메시지 추출**: `value[\"messages\"][-1].content` 코드는 메시지 리스트의 가장 마지막 항목(`[-1]`)을 선택해 `content` 속성에 담긴 텍스트를 출력합니다.

    * `add_messages` 함수 덕분에 메시지 리스트의 마지막 항목은 항상 가장 최근에 생성된 응답입니다.


```python
question = "서울의 유명한 맛집 TOP 10 추천해줘"

# 그래프 이벤트 스트리밍
for event in graph.stream({"messages": [("user", question)]}):
    # 이벤트 값 출력
    for value in event.values():
        print("Assistant:", value["messages"][-1].content)
```


자! 여기까지가 가장 기본적인 챗봇 구축이었습니다. 

아래는 이전 과정을 정리한 전체 코드입니다.



## 전체 코드


```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_teddynote.graphs import visualize_graph


###### STEP 1. 상태(State) 정의 ######
class State(TypedDict):
    # 메시지 정의(list type 이며 add_messages 함수를 사용하여 메시지를 추가)
    messages: Annotated[list, add_messages]


###### STEP 2. 노드(Node) 정의 ######
# LLM 정의
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 챗봇 함수 정의
def chatbot(state: State):
    # 메시지 호출 및 반환
    return {"messages": [llm.invoke(state["messages"])]}


###### STEP 3. 그래프(Graph) 정의, 노드 추가 ######
# 그래프 생성
graph_builder = StateGraph(State)

# 노드 이름, 함수 혹은 callable 객체를 인자로 받아 노드를 추가
graph_builder.add_node("chatbot", chatbot)

###### STEP 4. 그래프 엣지(Edge) 추가 ######
# 시작 노드에서 챗봇 노드로의 엣지 추가
graph_builder.add_edge(START, "chatbot")

# 그래프에 엣지 추가
graph_builder.add_edge("chatbot", END)

###### STEP 5. 그래프 컴파일(compile) ######
# 그래프 컴파일
graph = graph_builder.compile()

###### STEP 6. 그래프 시각화 ######
# 그래프 시각화
visualize_graph(graph)

###### STEP 7. 그래프 실행 ######
question = "서울의 유명한 맛집 TOP 10 추천해줘"

# 그래프 이벤트 스트리밍
for event in graph.stream({"messages": [("user", question)]}):
    # 이벤트 값 출력
    for value in event.values():
        print(value["messages"][-1].content)
```
```