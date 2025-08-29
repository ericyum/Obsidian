# 긴 문맥 재정렬(LongContextReorder)

모델의 아키텍처와 상관없이, 10개 이상의 검색된 문서를 포함할 경우 성능이 상당히 저하됩니다.

간단히 말해, 모델이 긴 컨텍스트 중간에 있는 관련 정보에 접근해야 할 때, 제공된 문서를 무시하는 경향이 있습니다.

자세한 내용은 다음 논문을 참조하세요

- https://arxiv.org/abs/2307.03172

이 문제를 피하기 위해, 검색 후 문서의 순서를 재배열하여 성능 저하를 방지할 수 있습니다.

- `Chroma` 벡터 저장소를 사용하여 텍스트 데이터를 저장하고 검색할 수 있는 `retriever`를 생성합니다.
- `retriever`의 `invoke` 메서드를 사용하여 주어진 쿼리에 대해 관련성이 높은 문서를 검색합니다.

```python
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()
```

```text
True
```

```python
# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH10-Retriever")
```

```text
LangSmith 추적을 시작합니다.
[프로젝트명]
CH10-Retriever
```

```python
from langchain_core.prompts import PromptTemplate
from langchain_community.document_transformers import LongContextReorder
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


# 임베딩을 가져옵니다.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

texts = [
    "이건 그냥 내가 아무렇게나 적어본 글입니다.",
    "사용자와 대화하는 것처럼 설계된 AI인 ChatGPT는 다양한 질문에 답할 수 있습니다.",
    "아이폰, 아이패드, 맥북 등은 애플이 출시한 대표적인 제품들입니다.",
    "챗GPT는 OpenAI에 의해 개발되었으며, 지속적으로 개선되고 있습니다.",
    "챗지피티는 사용자의 질문을 이해하고 적절한 답변을 생성하기 위해 대량의 데이터를 학습했습니다.",
    "애플 워치와 에어팟 같은 웨어러블 기기도 애플의 인기 제품군에 속합니다.",
    "ChatGPT는 복잡한 문제를 해결하거나 창의적인 아이디어를 제안하는 데에도 사용될 수 있습니다.",
    "비트코인은 디지털 금이라고도 불리며, 가치 저장 수단으로서 인기를 얻고 있습니다.",
    "ChatGPT의 기능은 지속적인 학습과 업데이트를 통해 더욱 발전하고 있습니다.",
    "FIFA 월드컵은 네 번째 해마다 열리며, 국제 축구에서 가장 큰 행사입니다."
]


# 검색기를 생성합니다. (K는 10으로 설정합니다)
retriever = Chroma.from_texts(texts, embedding=embeddings).as_retriever(
    search_kwargs={"k": 10}
)
```

검색기에 쿼리를 입력하여 검색을 수행합니다.

```python
query = "ChatGPT에 대해 무엇을 말해줄 수 있나요?"

# 관련성 점수에 따라 정렬된 관련 문서를 가져옵니다.
docs = retriever.invoke(query)
docs
```

```text
[Document(metadata={}, page_content='ChatGPT는 복잡한 문제를 해결하거나 창의적인 아이디어를 제안하는 데에도 사용될 수 있습니다.'),
 Document(metadata={}, page_content='ChatGPT의 기능은 지속적인 학습과 업데이트를 통해 더욱 발전하고 있습니다.'),
 Document(metadata={}, page_content='사용자와 대화하는 것처럼 설계된 AI인 ChatGPT는 다양한 질문에 답할 수 있습니다.'),
 Document(metadata={}, page_content='챗GPT는 OpenAI에 의해 개발되었으며, 지속적으로 개선되고 있습니다.'),
 Document(metadata={}, page_content='이건 그냥 내가 아무렇게나 적어본 글입니다.'),
 Document(metadata={}, page_content='챗지피티는 사용자의 질문을 이해하고 적절한 답변을 생성하기 위해 대량의 데이터를 학습했습니다.'),
 Document(metadata={}, page_content='비트코인은 디지털 금이라고도 불리며, 가치 저장 수단으로서 인기를 얻고 있습니다.'),
 Document(metadata={}, page_content='아이폰, 아이패드, 맥북 등은 애플이 출시한 대표적인 제품들입니다.'),
 Document(metadata={}, page_content='애플 워치와 에어팟 같은 웨어러블 기기도 애플의 인기 제품군에 속합니다.'),
 Document(metadata={}, page_content='FIFA 월드컵은 네 번째 해마다 열리며, 국제 축구에서 가장 큰 행사입니다.')]
```

`LongContextReorder` 클래스의 인스턴스인 `reordering`을 생성합니다.

- `reordering.transform_documents(docs)`를 호출하여 문서 목록 `docs`를 재정렬합니다.
  - 덜 관련된 문서는 목록의 중간에 위치하고, 더 관련된 문서는 시작과 끝에 위치하도록 재정렬됩니다.

```python
# 문서를 재정렬합니다
# 덜 관련된 문서는 목록의 중간에 위치하고 더 관련된 요소는 시작/끝에 위치합니다.
reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(docs)

# 4개의 관련 문서가 시작과 끝에 위치하는지 확인합니다.
reordered_docs
```

```text
[Document(metadata={}, page_content='ChatGPT의 기능은 지속적인 학습과 업데이트를 통해 더욱 발전하고 있습니다.'),
 Document(metadata={}, page_content='챗GPT는 OpenAI에 의해 개발되었으며, 지속적으로 개선되고 있습니다.'),
 Document(metadata={}, page_content='챗지피티는 사용자의 질문을 이해하고 적절한 답변을 생성하기 위해 대량의 데이터를 학습했습니다.'),
 Document(metadata={}, page_content='아이폰, 아이패드, 맥북 등은 애플이 출시한 대표적인 제품들입니다.'),
 Document(metadata={}, page_content='FIFA 월드컵은 네 번째 해마다 열리며, 국제 축구에서 가장 큰 행사입니다.'),
 Document(metadata={}, page_content='애플 워치와 에어팟 같은 웨어러블 기기도 애플의 인기 제품군에 속합니다.'),
 Document(metadata={}, page_content='비트코인은 디지털 금이라고도 불리며, 가치 저장 수단으로서 인기를 얻고 있습니다.'),
 Document(metadata={}, page_content='이건 그냥 내가 아무렇게나 적어본 글입니다.'),
 Document(metadata={}, page_content='사용자와 대화하는 것처럼 설계된 AI인 ChatGPT는 다양한 질문에 답할 수 있습니다.'),
 Document(metadata={}, page_content='ChatGPT는 복잡한 문제를 해결하거나 창의적인 아이디어를 제안하는 데에도 사용될 수 있습니다.')]
```

## Context Reordering을 사용하여 질의-응답 체인 생성

```python
def format_docs(docs):
    return "\n".join([doc.page_content for i, doc in enumerate(docs)])
```

```python
print(format_docs(docs))
```

```text
ChatGPT는 복잡한 문제를 해결하거나 창의적인 아이디어를 제안하는 데에도 사용될 수 있습니다.
ChatGPT의 기능은 지속적인 학습과 업데이트를 통해 더욱 발전하고 있습니다.
사용자와 대화하는 것처럼 설계된 AI인 ChatGPT는 다양한 질문에 답할 수 있습니다.
챗GPT는 OpenAI에 의해 개발되었으며, 지속적으로 개선되고 있습니다.
이건 그냥 내가 아무렇게나 적어본 글입니다.
챗지피티는 사용자의 질문을 이해하고 적절한 답변을 생성하기 위해 대량의 데이터를 학습했습니다.
비트코인은 디지털 금이라고도 불리며, 가치 저장 수단으로서 인기를 얻고 있습니다.
아이폰, 아이패드, 맥북 등은 애플이 출시한 대표적인 제품들입니다.
애플 워치와 에어팟 같은 웨어러블 기기도 애플의 인기 제품군에 속합니다.
FIFA 월드컵은 네 번째 해마다 열리며, 국제 축구에서 가장 큰 행사입니다.
```

```python
def format_docs(docs):
    return "\n".join(
        [
            f"[{i}] {doc.page_content} [source: teddylee777@gmail.com]"
            for i, doc in enumerate(docs)
        ]
    )

def reorder_documents(docs):
    # 재정렬
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)
    combined = format_docs(reordered_docs)
    print(combined)
    return combined
```

재정렬된 문서를 출력합니다.

```python
# 재정렬된 문서를 출력
_ = reorder_documents(docs)
```

```text
[0] ChatGPT의 기능은 지속적인 학습과 업데이트를 통해 더욱 발전하고 있습니다. [source: teddylee777@gmail.com]
[1] 챗GPT는 OpenAI에 의해 개발되었으며, 지속적으로 개선되고 있습니다. [source: teddylee777@gmail.com]
[2] 챗지피티는 사용자의 질문을 이해하고 적절한 답변을 생성하기 위해 대량의 데이터를 학습했습니다. [source: teddylee777@gmail.com]
[3] 아이폰, 아이패드, 맥북 등은 애플이 출시한 대표적인 제품들입니다. [source: teddylee777@gmail.com]
[4] FIFA 월드컵은 네 번째 해마다 열리며, 국제 축구에서 가장 큰 행사입니다. [source: teddylee777@gmail.com]
[5] 애플 워치와 에어팟 같은 웨어러블 기기도 애플의 인기 제품군에 속합니다. [source: teddylee777@gmail.com]
[6] 비트코인은 디지털 금이라고도 불리며, 가치 저장 수단으로서 인기를 얻고 있습니다. [source: teddylee777@gmail.com]
[7] 이건 그냥 내가 아무렇게나 적어본 글입니다. [source: teddylee777@gmail.com]
[8] 사용자와 대화하는 것처럼 설계된 AI인 ChatGPT는 다양한 질문에 답할 수 있습니다. [source: teddylee777@gmail.com]
[9] ChatGPT는 복잡한 문제를 해결하거나 창의적인 아이디어를 제안하는 데에도 사용될 수 있습니다. [source: teddylee777@gmail.com]
```

```python
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# 프롬프트 템플릿
template = """Given this text extracts:
{context}

-----

Please answer the following question:
{question}

Answer in the following languages: {language}
"""

# 프롬프트 정의
prompt = ChatPromptTemplate.from_template(template)

# Chain 정의
chain = (
    {
        "context": itemgetter("question")
        | retriever
        | RunnableLambda(reorder_documents),  # 질문을 기반으로 문맥을 검색합니다.
        "question": itemgetter("question"),  # 질문을 추출합니다.
        "language": itemgetter("language"),  # 답변 언어를 추출합니다.
    }
    | prompt  # 프롬프트 템플릿에 값을 전달합니다.
    | ChatOpenAI(model="gpt-4o-mini")  # 언어 모델에 프롬프트를 전달합니다.
    | StrOutputParser()  # 모델의 출력을 문자열로 파싱합니다.
)
```

`question` 에 쿼리를 입력하고 `language` 에 언어를 입력합니다.

- 재정렬된 문서의 검색 결과도 확인합니다.

```python
answer = chain.invoke(
    {"question": "ChatGPT에 대해 무엇을 말해줄 수 있나요?", "language": "KOREAN"}
)
```

```text
[0] ChatGPT의 기능은 지속적인 학습과 업데이트를 통해 더욱 발전하고 있습니다. [source: teddylee777@gmail.com]
[1] 챗GPT는 OpenAI에 의해 개발되었으며, 지속적으로 개선되고 있습니다. [source: teddylee777@gmail.com]
[2] 챗지피티는 사용자의 질문을 이해하고 적절한 답변을 생성하기 위해 대량의 데이터를 학습했습니다. [source: teddylee777@gmail.com]
[3] 아이폰, 아이패드, 맥북 등은 애플이 출시한 대표적인 제품들입니다. [source: teddylee777@gmail.com]
[4] FIFA 월드컵은 네 번째 해마다 열리며, 국제 축구에서 가장 큰 행사입니다. [source: teddylee777@gmail.com]
[5] 애플 워치와 에어팟 같은 웨어러블 기기도 애플의 인기 제품군에 속합니다. [source: teddylee777@gmail.com]
[6] 비트코인은 디지털 금이라고도 불리며, 가치 저장 수단으로서 인기를 얻고 있습니다. [source: teddylee777@gmail.com]
[7] 이건 그냥 내가 아무렇게나 적어본 글입니다. [source: teddylee777@gmail.com]
[8] 사용자와 대화하는 것처럼 설계된 AI인 ChatGPT는 다양한 질문에 답할 수 있습니다. [source: teddylee777@gmail.com]
[9] ChatGPT는 복잡한 문제를 해결하거나 창의적인 아이디어를 제안하는 데에도 사용될 수 있습니다. [source: teddylee777@gmail.com]
```

답변을 출력합니다.

```python
print(answer)
```

```text
ChatGPT는 OpenAI에 의해 개발된 AI로, 사용자의 질문을 이해하고 적절한 답변을 생성하기 위해 대량의 데이터를 학습하였습니다. 지속적인 학습과 업데이트를 통해 더욱 발전하고 있으며, 복잡한 문제를 해결하거나 창의적인 아이디어를 제안하는 데에도 활용될 수 있습니다. ChatGPT는 사용자와 대화하는 것처럼 설계되어 다양한 질문에 답할 수 있는 능력을 가지고 있습니다.
```