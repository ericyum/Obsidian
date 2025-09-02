# Chain of Density (CoD) 프롬프트의 기능

`prompt = hub.pull("lawwu/chain_of_density")`로 불러온 프롬프트는 논문 **Chain of Density (CoD)**에서 제안된 방법을 구현한 것입니다. 이 프롬프트는 일반적인 요약과 달리, **점진적으로 더 조밀하고 정보 밀도가 높은 요약**을 생성하는 기능을 합니다.

---

#### 핵심 기능

**1. 점진적 요약 (Iterative Summarization)**
* 단 한 번에 최종 요약을 생성하는 것이 아니라, 여러 단계(기본적으로 5단계)를 거쳐 요약을 발전시킵니다.
* 첫 번째 요약은 4-5 문장으로 길고 모호하며, 비어있는 정보가 많은 상태로 시작합니다.

**2. 엔티티 기반 정보 추가 (Entity-based Information)**
* 각 단계마다 이전 요약에 없었지만 원본 문서에 존재하는 **중요한 엔티티(명사, 고유명사 등)**를 1-3개 식별합니다.
* 엔티티는 "주요 스토리와 관련 있고, 구체적이지만 간결하며, 이전 요약에 없고, 원문에 존재하는 것"이어야 합니다.

**3. 정보 밀도 향상 (Increased Density)**
* 새로운 요약을 쓸 때, 이전 요약의 모든 정보와 새로 식별한 엔티티를 모두 포함해야 합니다.
* 이 과정에서 불필요한 문구("이 문서는 ...에 대해 논한다" 등)를 제거하고, 문장을 압축하며 재구성하여 공간을 확보합니다.
* 결과적으로, 각 단계의 요약은 동일한 단어 수를 유지하면서 더 많은 정보를 담게 되어 정보 밀도가 높아집니다.

**4. JSON 형식 출력**
* 프롬프트는 모델에게 최종 결과를 JSON 형식의 리스트로 반환하도록 지시합니다.
* 각 JSON 객체는 `"Missing_Entities"` (누락된 엔티티)와 `"Denser_Summary"` (더 조밀해진 요약) 두 개의 키를 가집니다. 이를 통해 각 단계의 변화를 명확하게 추적할 수 있습니다.

---

### 요약

**"lawwu/chain_of_density" 프롬프트**는 주어진 글을 바탕으로, 5단계에 걸쳐 누락된 중요한 정보를 점진적으로 추가하고 문장을 압축함으로써, 최종적으로 원문의 핵심 내용이 응축된 **고밀도 요약**을 생성하는 기능을 합니다.

```python
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()
```

**Output:**

```
True
```

Chain of Density: https://arxiv.org/pdf/2309.04269.pdf

```python
import time
import textwrap

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.document_loaders import WebBaseLoader
from langchain.schema.runnable import RunnablePassthrough

# Load some data to summarize
loader = WebBaseLoader("https://teddylee777.github.io/data-science/optuna/")
docs = loader.load()
content = docs[0].page_content

# Get this prompt template
prompt = hub.pull("lawwu/chain_of_density")

# The chat model output is a JSON list of dicts, with SimpleJsonOutputParser
# we can convert it o a dict, and it suppors streaming.
json_parser = SimpleJsonOutputParser()
```

```python
prompt
```

**Output:**

```
ChatPromptTemplate(input_variables=['ARTICLE'], ...)
```

```python
chain = (
    {"ARTICLE": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.1)
    | json_parser
)
```

```python
chain.invoke(content)
```

**Output:**

```json
[{'Missing_Entities': 'Optuna', 'Denser_Summary': '...'}, {'Missing_Entities': 'objective function', 'Denser_Summary': '...'}, ...]
```

```python
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
import json


# Load some data to summarize
loader = WebBaseLoader("https://www.aitimes.com/news/articleView.html?idxno=131777")
docs = loader.load()
content = docs[0].page_content
# Load the prompt
# prompt = hub.pull("langchain-ai/chain-of-density:4f55305e")


class StreamCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        print(token, end="", flush=True)


prompt = ChatPromptTemplate.from_template(
    """Article: {ARTICLE}\n..."""
)


# Create the chain, including
chain = (
    prompt
    | ChatOpenAI(
        temperature=0,
        model="gpt-4-turbo-preview",
        streaming=True,
        callbacks=[StreamCallback()],
    )
    | JsonOutputParser()
    | (lambda x: x[-1]["Denser_Summary"])
)

# Invoke the chain
result = chain.invoke({"ARTICLE": content})
print(result)
```

**Output:**

```json
[
    {
        "Missing_Entities": "",
        "Denser_Summary": "이 기사는 데이터사이언스, 머신러닝, 인공지능에 대한 개념을 설명하고 있습니다. ..."
    },
    ...
]
```