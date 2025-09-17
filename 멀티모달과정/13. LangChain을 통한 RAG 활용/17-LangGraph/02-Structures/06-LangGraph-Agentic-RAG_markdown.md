# Agentic RAG

에이전트(Agent) 는 검색 도구를 사용할지 여부를 결정해야 할 때 유용합니다. 에이전트와 관련된 내용은 [Agent](https://wikidocs.net/233782) 페이지를 참고하세요.

검색 에이전트를 구현하기 위해서는 `LLM`에 검색 도구에 대한 접근 권한을 부여하기만 하면 됩니다.

이를 [LangGraph](https://langchain-ai.github.io/langgraph/)에 통합할 수 있습니다.

![langgraph-agentic-rag](assets/langgraph-agentic-rag.png)

---

## 환경 설정

---

## 기본 PDF 기반 Retrieval Chain 생성

여기서는 PDF 문서를 기반으로 Retrieval Chain 을 생성합니다. 가장 단순한 구조의 Retrieval Chain 입니다.

단, LangGraph 에서는 Retirever 와 Chain 을 따로 생성합니다. 그래야 각 노드별로 세부 처리를 할 수 있습니다.

**참고**

- 이전 튜토리얼에서 다룬 내용이므로, 자세한 설명은 생략합니다.

---

그 다음 `retriever_tool` 도구를 생성합니다.

**참고**

`document_prompt` 는 검색된 문서를 표현하는 프롬프트입니다.

**사용가능한 키** 

- `page_content`
- `metadata` 의 키: (예시) `source`, `page`

**사용예시**

`"<document><context>{page_content}</context><metadata><source>{source}</source><page>{page}</page></metadata></document>"`

---

## Agent 상태

그래프를 정의하겠습니다.

각 노드에 전달되는 `state` 객체입니다.

상태는 `messages` 목록으로 구성됩니다.

그래프의 각 노드는 이 목록에 내용을 추가합니다.

---

## 노드와 엣지

에이전트 기반 RAG 그래프는 다음과 같이 구성될 수 있습니다.

* **상태**는 메시지들의 집합입니다
* 각 **노드**는 상태를 업데이트(추가)합니다
* **조건부 엣지**는 다음에 방문할 노드를 결정합니다

---

간단한 채점기(Grader)를 만들어 보겠습니다.

---

## 그래프

* `call_model` 에이전트로 시작합니다
* 에이전트가 함수를 호출할지 결정합니다
* 함수 호출을 결정한 경우, 도구(retriever)를 호출하기 위한 `action`을 실행합니다
* 도구의 출력값을 메시지(`state`)에 추가하여 에이전트를 호출합니다

---

그래프를 시각화합니다.

---

## 그래프 실행

---

아래는 문서 검색이 **불필요한** 질문의 예시입니다.

---

아래는 임의로 **문서 검색이 불가능한** 질문 예시입니다.

따라서, 문서를 지속적으로 검색하는 과정에서 `GraphRecursionError` 가 발생하였습니다.

---

다음 튜토리얼에서는 이를 해결하는 방법을 다룹니다.