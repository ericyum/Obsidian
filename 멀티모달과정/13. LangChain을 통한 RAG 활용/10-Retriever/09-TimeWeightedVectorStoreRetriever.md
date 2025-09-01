# 시간 가중 벡터저장소 리트리버(TimeWeightedVectorStoreRetriever)

`TimeWeightedVectorStoreRetriever` 는 의미론적 유사성과 시간에 따른 감쇠를 결합해 사용하는 검색 도구입니다. 이를 통해 문서 또는 데이터의 **"신선함"** 과 **"관련성"** 을 모두 고려하여 결과를 제공합니다.

스코어링 알고리즘은 다음과 같이 구성됩니다

$\text{semantic\_similarity} + (1.0 - \text{decay\_rate})^{hours\_passed}$

여기서 `semantic_similarity` 는 문서 또는 데이터 간의 의미적 유사도를 나타내고, `decay_rate` 는 시간이 지남에 따라 점수가 얼마나 감소하는지를 나타내는 비율입니다. `hours_passed` 는 객체가 마지막으로 접근된 후부터 현재까지 경과한 시간(시간 단위)을 의미합니다.

이 방식의 주요 특징은, 객체가 마지막으로 접근된 시간을 기준으로 하여 **"정보의 신선함"** 을 평가한다는 점입니다. 즉, **자주 접근되는 객체는 시간이 지나도 높은 점수**를 유지하며, 이를 통해 **자주 사용되거나 중요하게 여겨지는 정보가 검색 결과 상위에 위치할 가능성이 높아집니다.** 이런 방식은 최신성과 관련성을 모두 고려하는 동적인 검색 결과를 제공합니다.

특히, `decay_rate` 는 리트리버의 객체가 생성된 이후가 아니라 **마지막으로 액세스된 이후 경과된 시간** 을 의미합니다. 즉, 자주 액세스하는 객체는 '최신'으로 유지됩니다.

---

```python
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()
```

**결과:**
```
True
```

---

```python
# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH10-Retriever")
```

**결과:**
```
LangSmith 추적을 시작합니다.
[프로젝트명]
CH10-Retriever
```

---

## 낮은 감쇠율(low decay_rate)

- `decay rate` 가 낮다는 것은 (여기서는 극단적으로 0에 가깝게 설정할 것입니다) **기억이 더 오래 "기억될"** 것임을 의미합니다.

- `decay rate` 가 **0 이라는 것은 기억이 절대 잊혀지지 않는다**는 것을 의미하며, 이는 이 retriever를 vector lookup과 동등하게 만듭니다.

`TimeWeightedVectorStoreRetriever`를 초기화하며, 벡터 저장소, 감쇠율(`decay_rate`)을 매우 작은 값으로 설정하고, 검색할 벡터의 개수(k)를 1로 지정합니다.

---

```python
from datetime import datetime, timedelta

import faiss
from langchain.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# 임베딩 모델을 정의합니다.
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# 벡터 저장소를 빈 상태로 초기화합니다.
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {})

# 시간 가중치가 적용된 벡터 저장소 검색기를 초기화합니다. (여기서는, 낮은 감쇠율을 적용합니다)
retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore, decay_rate=0.0000000000000000000000001, k=1
)
```

---

간단한 예제 데이터를 추가합니다.

---

```python
# 어제 날짜를 계산합니다.
yesterday = datetime.now() - timedelta(days=1)

retriever.add_documents(
    # 문서를 추가하고, metadata에 어제 날짜를 설정합니다.
    [
        Document(
            page_content="테디노트 구독해 주세요.",
            metadata={"last_accessed_at": yesterday},
        )
    ]
)

# 다른 문서를 추가합니다. metadata는 별도로 설정하지 않았습니다.
retriever.add_documents([Document(page_content="테디노트 구독 해주실꺼죠? Please!")])
```

**결과:**
```
['ea70f0f7-6e59-46bf-a261-0bd1935b8c66']
```

---

`retriever.invoke()` 를 호출하여 검색을 수행합니다.

- 이는 가장 두드러진(salient) 문서이기 때문입니다.
- `decay_rate` 가 **0에 가깝기 때문** 에 문서는 여전히 최신(recent)으로 간주됩니다.

---

```python
# "테디노트 구독해 주세요." 가 가장 먼저 반환되는 이유는 가장 두드러지기 때문이며
# 감쇠율이 0에 가깝기 때문에 여전히 최신 상태를 유지하고 있음을 의미합니다.
retriever.invoke("테디노트")
```

**결과:**
```
[Document(metadata={'last_accessed_at': datetime.datetime(2025, 9, 1, 15, 30, 16, 431613), 'created_at': datetime.datetime(2025, 9, 1, 15, 30, 12, 981855), 'buffer_idx': 0}, page_content='테디노트 구독해 주세요.')]
```

---

## 높음 감쇠율(high decay_rate)

높은 `decay_rate`(예: 0.9999...)를 사용하면 `recency score`가 빠르게 0으로 수렴합니다.

(만약 이 값을 1로 설정하면 모든 객체의 `recency` 값이 0이 되어, Vector Lookup 과 동일한 결과를 얻게 됩니다.)

`TimeWeightedVectorStoreRetriever`를 사용하여 검색기를 초기화합니다. `decay_rate`를 0.999로 설정하여 시간에 따른 가중치 감소율을 조정합니다.

---

```python
# 임베딩 모델을 정의합니다.
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# 벡터 저장소를 빈 상태로 초기화합니다.
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {})

# 시간 가중치가 적용된 벡터 저장소 검색기를 초기화합니다.
retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore, decay_rate=0.999, k=1
)
```

---

다시 문서를 새롭게 추가합니다.

---

```python
# 어제 날짜를 계산합니다.
yesterday = datetime.now() - timedelta(days=1)

retriever.add_documents(
    # 문서를 추가하고, metadata에 어제 날짜를 설정합니다.
    [
        Document(
            page_content="테디노트 구독해 주세요.",
            metadata={"last_accessed_at": yesterday},
        )
    ]
)

# 다른 문서를 추가합니다. metadata는 별도로 설정하지 않았습니다.
retriever.add_documents([Document(page_content="테디노트 구독 해주실꺼죠? Please!")])
```

**결과:**
```
['6433e39c-73b6-474c-8da4-3b6f9ebebe75']
```

---

`retriever.invoke("테디노트")` 를 호출하면 `""테디노트 구독 해주실꺼죠? Please!""` 가 먼저 반환됩니다.
- 이는 retriever가 "테디노트 구독해 주세요." 와 관련된 문서를 대부분 잊어버렸기 때문입니다.

---

```python
# 검색 후 결과확인
retriever.invoke("테디노트")
```

**결과:**
```
[Document(metadata={'last_accessed_at': datetime.datetime(2025, 9, 1, 15, 30, 41, 786557), 'created_at': datetime.datetime(2025, 9, 1, 15, 30, 40, 506741), 'buffer_idx': 1}, page_content='테디노트 구독 해주실꺼죠? Please!')]
```

---

## 감쇠율(decay_rate) 정리

- `decay_rate` 를 0.000001 로 매우 작게 설정한 경우
  - 감쇠율(즉, 정보를 망각하는 비율)이 매우 낮기 때문에 정보를 거의 잊지 않습니다. 
  - 따라서, **최신 정보이든 오래된 정보든 시간 가중치 차이가 거의 없습니다.** 이럴때는 유사도에 더 높은 점수를 주게 됩니다.

- `decay_rate` 를 0.999 로 1에 가깝게 설정한 경우 
  - 감쇠율(즉, 정보를 망각하는 비율)이 매우 높습니다. 따라서, 과거의 정보는 거의다 잊어버립니다. 
  - 따라서, 이러한 경우는 최신 정보에 더 높은 점수를 주게 됩니다.

---

## 가상의 시간으로 `decay_rate` 조정

LangChain의 일부 유틸리티를 사용하면 시간 구성 요소를 모의(mock) 테스트 할 수 있습니다.

- `mock_now` 함수는 LangChain에서 제공하는 유틸리티 함수로, 현재 시간을 모의(mock)하는 데 사용됩니다.

`mock_now` 함수를 사용하여 현재 시간을 변경하면서 검색 결과를 테스트할 수 있습니다. 

- 해당 기능을 활용하여 적절한 `decay_rate` 를 찾는데 도움을 받을 수 있습니다.

[주의] 만약 너무 오래전의 시간으로 설정하면, decay_rate 계산시 오류가 발생할 수 있습니다.

# 필독!

**mock_now는 오직 langchain안에서만 작동하기 때문에 datetime같은 시스템 시간을 변경할 수 없다.**
  
**따라서 아래의 freezegun을 사용하던데 그냥 FakeListLLm을 통해 하드 코딩하는 식으로 가는 것이 좋다.** 

---

```python
import datetime
from langchain.prompts import PromptTemplate
from langchain_community.llms import FakeListLLM
from langchain.utils import mock_now

# Set the mock time
mock_now(datetime.datetime(2024, 8, 30, 0, 0))

# Create a prompt that uses a time variable
prompt = PromptTemplate.from_template("The current date is: {date}.")

# Use a placeholder LLM for demonstration
llm = FakeListLLM(responses=["The current date is: 2024-08-30."])

# Create a simple chain
chain = prompt | llm

# Invoke the chain, passing in the datetime.now() variable
# LangChain internally uses the mocked time for this variable.
print(chain.invoke({"date": datetime.datetime.now().strftime("%Y-%m-%d")}))
```

**결과:**
```
The current date is: 2024-08-30.
```

---

freezegun 라이브러리를 활용할 수도 있다.

---

```python
# !pip install freezegun
```

---

```python
from freezegun import freeze_time
import datetime

# Use a context manager to freeze the time
with freeze_time("2024-08-30"):
    # Inside this block, datetime.datetime.now() will return the frozen time
    print(datetime.datetime.now())

# Outside the block, the time returns to normal
print(datetime.datetime.now())
```

**결과:**
```
2024-08-30 00:00:00
2025-09-01 17:55:04.145803
```
```