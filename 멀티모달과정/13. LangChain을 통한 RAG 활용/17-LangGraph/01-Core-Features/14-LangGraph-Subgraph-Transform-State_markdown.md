# `subgraph`의 입력과 출력을 변환하는 방법

`subgraph` **상태**가 `parent graph` 상태와 완전히 독립적일 수 있습니다. 

즉, 두 그래프 간에 중복되는 상태 키(state keys) 가 없을 수 있습니다. 

이러한 경우에는 `subgraph`를 호출하기 전에 입력을 변환하고, 반환하기 전에 출력을 변환해야 합니다. 

---

## 환경설정

---

## `graph`와 `subgraph` 정의

다음과 같이 3개의 `graph`를 정의하겠습니다.

- `parent graph`
  
- `parent graph` 에 의해 호출될 `child subgraph`

- `child graph` 에 의해 호출될 `grandchild subgraph`

---

## `grandchild` 정의

---

그래프를 시각화 합니다.

---

## `child` 정의

---

`grandchild_graph`의 호출을 별도의 함수(`call_grandchild_graph`)로 감싸고 있습니다. 

이 함수는 grandchild 그래프를 호출하기 전에 입력 상태를 변환하고, grandchild 그래프의 출력을 다시 child 그래프 상태로 변환합니다. 

만약 이러한 변환 없이 `grandchild_graph`를 직접 `.add_node`에 전달하면, child와 grandchild 상태 간에 공유된 상태 키(State Key) 이 없기 때문에 LangGraph에서 오류가 발생하게 됩니다.

**중요**

`child subgraph` 와 `grandchild subgraph`는 `parent graph`와 공유되지 않는 자신만의 **독립적인** `state`를 가지고 있다는 점에 유의하시기 바랍니다.

---

## `parent` 정의

---

그래프를 시각화 합니다.

---

`child_graph` 호출을 별도의 함수 `call_child_graph` 로 감싸고 있는데, 이 함수는 자식 그래프를 호출하기 전에 입력 상태를 변환하고 자식 그래프의 출력을 다시 부모 그래프 상태로 변환합니다. 

변환 없이 `child_graph`를 직접 `.add_node`에 전달하면 부모와 자식 상태 간에 공유된 상태 키(State Key) 이 없기 때문에 LangGraph에서 오류가 발생합니다.

---

그럼, 부모 그래프를 실행하여 자식 및 손자 하위 그래프가 올바르게 호출되는지 확인해보겠습니다.