# 스트리밍 모드

---

`graph`의 전체 상태를 **스트리밍**하는 방법

---

LangGraph는 여러 스트리밍(Streaming) 모드를 지원합니다. 

주요 모드는 다음과 같습니다

- `values`: 이 스트리밍 모드는 그래프의 값들을 스트리밍합니다. 이는 각 노드가 호출된 후의 **그래프의 전체 상태**를 의미합니다.
- `updates`: 이 스트리밍 모드는 그래프의 업데이트 내용을 스트리밍합니다. 이는 각 노드가 호출된 후의 **그래프 상태에 대한 업데이트**를 의미합니다.
- `messages`: 이 스트리밍 모드는 각 노드의 메시지를 스트리밍합니다. 이때 **LLM 의 토큰 단위의 출력 스트리밍** 도 가능합니다.

---

## 환경설정

---

## 그래프 정의하기

이 가이드에서는 간단한 에이전트를 사용하겠습니다.

---

## 노드의 단계별 출력

**스트리밍 모드**
- `values`: 각 단계의 현재 상태 값 출력
- `updates`: 각 단계의 상태 업데이트만 출력 (기본값)
- `messages`: 각 단계의 메시지 출력

여기서 스트리밍의 의미는 LLM 출력시 토큰 단위로 스트리밍하는 개념이 아니라, 단계별로 출력하는 의미를 가집니다.

---

### `stream_mode = "values"`

`values` 모드는 각 단계의 현재 상태 값을 출력합니다.

**참고**

`chunk.items()`

- `key`: State 의 key 값
- `value`: State 의 key 에 대한하는 value

---

#### 동기(Synchronous) 방식의 스트리밍

- `chunk` 는 dictionary 형태(key: State 의 key, value: State 의 value)

---

#### 비동기(Asynchronous) 방식의 스트리밍

**참고**

- `astream()` 메서드는 비동기 스트림 처리를 통해 그래프를 실행하고 값 모드로 청크 단위 응답을 생성합니다.
- `async for` 문을 사용하여 비동기 스트림 처리를 수행합니다.


---

최종 결과만 확인하고 싶은 경우, 다음과 같이 처리 합니다.

---

### `stream_mode = "updates"`

`updates` 모드는 각 단계에 대한 업데이트된 State 만 내보냅니다. 

- 출력은 노드 이름을 key 로, 업데이트된 값을 values 으로 하는 `dictionary` 입니다.

**참고**

`chunk.items()`

- `key`: 노드(Node) 의 이름
- `value`: 해당 노드(Node) 단계에서의 출력 값(dictionary). 즉, 여러 개의 key-value 쌍을 가진 dictionary 입니다.

---

#### 동기(Synchronous) 방식의 스트리밍

---

#### 비동기(Asynchronous) 방식의 스트리밍

---

### `stream_mode = "messages"`

`messages` 모드는 각 단계에 대한 메시지를 스트리밍합니다.

**참고**

- `chunk` 는 두 개의 요소를 가진 tuple 입니다.
  - `chunk_msg`: 실시간 출력 메시지
  - `metadata`: 노드 정보

---

#### 동기(Synchronous) 방식의 스트리밍

---

#### 비동기(Asynchronous) 방식의 스트리밍

---

## 특정 노드에 대한 출력 스트리밍

**참고**

- `metadata["langgraph_node"]` 를 통해 특정 노드에서 출력된 메시지만 출력할 수 있습니다.

---

특정 노드(Node) 에 대해서 출력하고 싶은 경우, stream_mode="messages" 를 통해 설정할 수 있습니다.

`stream_mode="messages"` 설정시, (`chunk_msg`, `metadata`) 형태로 메시지를 받습니다.
- `chunk_msg` 는 실시간 출력 메시지, 
- `metadata` 는 노드 정보를 의미합니다.

`metadata["langgraph_node"]` 를 통해 특정 노드에서 출력된 메시지만 출력할 수 있습니다.

(예시) chatbot 노드에서 출력된 메시지만 출력하는 경우

`metadata["langgraph_node"] == "chatbot"`

---

metadata 를 출력하면 노드 정보를 확인할 수 있습니다.

---

## 사용자 정의 `tag` 필터링 된 스트리밍

LLM 의 출력이 여러 군데에서 발생하는 경우, 특정 노드에서 출력된 메시지만 출력하고 싶은 경우가 있습니다.

이러한 경우, `tags` 를 추가하여 출력하고 싶은 노드만 선별할 수 있습니다.

llm 에 tags 를 추가하는 방법은 다음과 같습니다. `tags` 는 리스트 형태로 추가할 수 있습니다.

`llm.with_config(tags=["WANT_TO_STREAM"])`


이를 통해 이벤트를 더 정확하게 필터링하여 해당 모델에서 발생한 이벤트만 유지할 수 있습니다.

---

아래 예시는 `WANT_TO_STREAM` 태그가 있는 경우만 출력하는 예시입니다.

---

## 도구 호출에 대한 스트리밍 출력

- `AIMessageChunk`: 실시간 토큰 단위의 출력 메시지
- `tool_call_chunks`: 도구 호출 청크. 만약 `tool_call_chunks` 가 존재하는 경우, 도구 호출 청크를 누적하여 출력합니다. (도구 토큰은 이 속성을 보고 판단하여 출력)

---

## Subgraphs 스트리밍 출력

이번에는 Subgraphs 를 통해 스트리밍 출력을 확인하는 방법을 알아보겠습니다.

Subgraphs 는 그래프의 일부를 서브그래프로 정의하는 기능입니다.

**흐름**

- Subgraphs 에서는 기존의 최신 뉴스를 검색하는 기능을 재사용합니다.
- Parent Graph 에서는 검색된 최신 뉴스를 바탕으로 SNS 포스트를 생성하는 기능을 추가합니다.

---

그래프를 시각화합니다.


---

### Subgraphs 출력을 '생략' 하는 경우


---

### Subgraphs 출력도 '포함' 하는 경우

**참고**

- `subgraphs=True` 를 통해 Subgraphs 의 출력도 포함할 수 있습니다.
- `(namespace, chunk)` 형태로 출력됩니다.

---

#### Subgraphs 안에서 LLM 출력 토큰 단위 스트리밍

---

**참고**

- `kind` 는 이벤트 종류를 나타냅니다.
- 이벤트 종류는 [StreamEvent 타입별 정리](https://wikidocs.net/265576) 에서 확인하세요!

---

#### 특정 tags 만 스트리밍 출력하는 경우

- `ONLY_STREAM_TAGS` 를 통해 스트리밍 출력하고 싶은 tags 만 설정할 수 있습니다.
- 여기서는 "WANT_TO_STREAM" 는 출력에서 배제하고 "WANT_TO_STREAM2" 만 출력하는 경우를 확인합니다.