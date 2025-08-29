# MultiQueryRetriever

거리 기반 벡터 데이터베이스 검색은 고차원 공간에서의 쿼리 임베딩(표현)과 '거리'를 기준으로 유사한 임베딩을 가진 문서를 찾는 방식입니다. 하지만 쿼리의 **세부적인 차이나 임베딩이 데이터의 의미를 제대로 포착하지 못할 경우, 검색 결과가 달라질 수** 있습니다. 또한, 이를 수동으로 조정하는 프롬프트 엔지니어링이나 튜닝 작업은 번거로울 수 있습니다.

이런 문제를 해결하기 위해, `MultiQueryRetriever` 는 주어진 사용자 입력 쿼리에 대해 다양한 관점에서 여러 쿼리를 자동으로 생성하는 LLM(Language Learning Model)을 활용해 프롬프트 튜닝 과정을 자동화합니다.

이 방식은 각각의 쿼리에 대해 관련 문서 집합을 검색하고, 모든 쿼리를 아우르는 고유한 문서들의 합집합을 추출해, 잠재적으로 관련된 더 큰 문서 집합을 얻을 수 있게 해줍니다. 

여러 관점에서 동일한 질문을 생성함으로써, `MultiQueryRetriever` 는 거리 기반 검색의 제한을 일정 부분 극복하고, 더욱 풍부한 검색 결과를 제공할 수 있습니다.

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
# 샘플 벡터DB 구축
from langchain_community.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 블로그 포스트 로드
loader = WebBaseLoader(
    "https://teddylee777.github.io/openai/openai-assistant-tutorial/", encoding="utf-8"
)

# 문서 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = loader.load_and_split(text_splitter)

# 임베딩 정의
openai_embedding = OpenAIEmbeddings()

# 벡터DB 생성
db = FAISS.from_documents(docs, openai_embedding)

# retriever 생성
retriever = db.as_retriever()

# 문서 검색
query = "OpenAI Assistant API의 Functions 사용법에 대해 알려주세요."
relevant_docs = retriever.invoke(query)

# 검색된 문서의 개수 출력
len(relevant_docs)
```

```text
4
```

검색된 결과 중 1개 문서의 내용을 출력합니다.

```python
# 1번 문서를 출력합니다.
print(relevant_docs[1].page_content)
```

```text
가장 강력한 도구로서, Assistant에게 사용자 정의 함수를 지정할 수 있습니다. 이는 Chat Completions API에서의 함수 호출과 매우 유사합니다.


Function calling(함수 호출) 도구를 사용하면 Assistant 에게 사용자 정의 함수 를 설명하여 호출해야 하는 함수를 인자와 함께 지능적으로 반환하도록 할 수 있습니다.


Assistant API는 실행 중에 함수를 호출할 때 실행을 일시 중지하며, 함수 호출 결과를 다시 제공하여 Run 실행을 계속할 수 있습니다. (이는 사용자 피드백을 받아 재게할 수 있는 의미이기도 합니다. 아래 튜토리얼에서 상세히 다룹니다).
```

## 사용방법

`MultiQueryRetriever` 에 사용할 LLM을 지정하고 질의 생성에 사용하면, retriever가 나머지 작업을 처리합니다.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI


# ChatOpenAI 언어 모델을 초기화합니다. temperature는 0으로 설정합니다.
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

multiquery_retriever = MultiQueryRetriever.from_llm(  # MultiQueryRetriever를 언어 모델을 사용하여 초기화합니다.
    # 벡터 데이터베이스의 retriever와 언어 모델을 전달합니다.
    retriever=db.as_retriever(),
    llm=llm,
)
```

아래는 다중 쿼리를 생성하는 중간 과정을 디버깅하기 위하여 실행하는 코드입니다.

먼저 `"langchain.retrievers.multi_query"` 로거를 가져옵니다. 

이는 `logging.getLogger()` 함수를 사용하여 수행됩니다. 그 다음, 이 로거의 로그 레벨을 `INFO`로 설정하여, `INFO` 레벨 이상의 로그 메시지만 출력되도록 할 수 있습니다. 

```python
# 쿼리에 대한 로깅 설정
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
```

이 코드는 `retriever_from_llm` 객체의 `invoke` 메서드를 사용하여 주어진 `question`과 관련된 문서를 검색합니다. 

검색된 문서들은 `unique_docs`라는 변수에 저장되며, 이 변수의 길이를 확인함으로써 검색된 관련 문서의 총 개수를 알 수 있습니다. 이 과정을 통해 사용자의 질문에 대한 관련 정보를 효과적으로 찾아내고 그 양을 파악할 수 있습니다.

```python
# 질문을 정의합니다.
question = "OpenAI Assistant API의 Functions 사용법에 대해 알려주세요."
# 문서 검색
relevant_docs = multiquery_retriever.invoke(question)

# 검색된 고유한 문서의 개수를 반환합니다.
print(
    f"===============\n검색된 문서 개수: {len(relevant_docs)}",
    end="\n===============\n",
)

# 검색된 문서의 내용을 출력합니다.
print(relevant_docs[0].page_content)
```

```text
===============
검색된 문서 개수: 4
===============
OpenAI의 새로운 Assistants API는 대화와 더불어 강력한 도구 접근성을 제공합니다. 본 튜토리얼은 OpenAI Assistants API를 활용하는 내용을 다룹니다. 특히, Assistant API 가 제공하는 도구인 Code Interpreter, Retrieval, Functions 를 활용하는 방법에 대해 다룹니다. 이와 더불어 파일을 업로드 하는 내용과 사용자의 피드백을 제출하는 내용도 튜토리얼 말미에 포함하고 있습니다.



주요내용
```

## LCEL Chain 활용하는 방법

- 사용자 정의 프롬프트 정의하고, 정의한 프롬프트와 함께 Chain 을 생성합니다.
- Chain 은 사용자의 질문을 입력 받으면 (아래의 예제에서는) 5개의 질문을 생성한 뒤 `"\n"` 구분자로 구분하여 생성된 5개 질문을 반환합니다.

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 프롬프트 템플릿을 정의합니다.(5개의 질문을 생성하도록 프롬프트를 작성하였습니다)
prompt = PromptTemplate.from_template(
    """You are an AI language model assistant. 
Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. 
By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. 
Your response should be a list of values separated by new lines, eg: `foo\nbar\nbaz\n`

#ORIGINAL QUESTION: 
{question}

#Answer in Korean:
"""
)

# 언어 모델 인스턴스를 생성합니다.
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# LLMChain을 생성합니다.
custom_multiquery_chain = (
    {"question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
)

# 질문을 정의합니다.
question = "OpenAI Assistant API의 Functions 사용법에 대해 알려주세요."

# 체인을 실행하여 생성된 다중 쿼리를 확인합니다.
multi_queries = custom_multiquery_chain.invoke(question)
# 결과를 확인합니다.(5개 질문 생성)
multi_queries
```

```text
'OpenAI Assistant API의 Functions 기능을 사용하는 방법에 대해 설명해 주세요.  \nOpenAI Assistant API에서 Functions를 활용하는 방법이 궁금합니다.  \nOpenAI Assistant API의 Functions를 어떻게 사용할 수 있는지 알려주세요.  \nOpenAI Assistant API의 Functions 사용법에 대한 정보를 제공해 주세요.  \nOpenAI Assistant API에서 Functions를 사용하는 절차에 대해 설명해 주세요. '
```

이전에 생성한 Chain을 `MultiQueryRetriever` 에 전달하여 retrieve 할 수 있습니다.

```python
multiquery_retriever = MultiQueryRetriever.from_llm(
    llm=custom_multiquery_chain, retriever=db.as_retriever()
)
```

`MultiQueryRetriever`를 사용하여 문서를 검색하고 결과를 확인합니다.

```python
# 결과
relevant_docs = multiquery_retriever.invoke(question)

# 검색된 고유한 문서의 개수를 반환합니다.
print(
    f"===============\n검색된 문서 개수: {len(relevant_docs)}",
    end="\n===============\n",
)

# 검색된 문서의 내용을 출력합니다.
print(relevant_docs[0].page_content)
```

```text
===============
검색된 문서 개수: 4
===============
OpenAI의 새로운 Assistants API는 대화와 더불어 강력한 도구 접근성을 제공합니다. 본 튜토리얼은 OpenAI Assistants API를 활용하는 내용을 다룹니다. 특히, Assistant API 가 제공하는 도구인 Code Interpreter, Retrieval, Functions 를 활용하는 방법에 대해 다룹니다. 이와 더불어 파일을 업로드 하는 내용과 사용자의 피드백을 제출하는 내용도 튜토리얼 말미에 포함하고 있습니다.



주요내용
```

# MultiQueryRetriever의 질문 생성 방식 커스터마이징

## 기존 MultiQueryRetriever와의 차이점

* **기본 `MultiQueryRetriever`**:
    내부적으로 미리 정의된 프롬프트 템플릿을 사용합니다. 이 프롬프트는 일반적으로 LLM에게 "주어진 질문을 기반으로 여러 관점의 질문을 생성하라"고 지시합니다. 사용자가 이 프롬프트의 내용이나 질문 생성 방식을 직접 제어하기는 어렵습니다.

* **커스텀 체인 (`custom_multiquery_chain`) 사용**:
    1.  `PromptTemplate.from_template(...)`를 통해 질문 증강에 사용할 **프롬프트를 직접 정의**합니다.
    2.  예시 코드에서는 "5개의 다른 버전의 질문을 생성하고, 답변은 줄바꿈으로 구분하라"는 구체적인 지시사항을 포함하고 있습니다.
    3.  또한 "vector database"와 "distance-based similarity search" 같은 전문 용어를 사용하여 LLM이 보다 맥락에 맞는 질문을 생성하도록 유도할 수 있습니다.
    4.  이렇게 직접 만든 `custom_multiquery_chain`을 `MultiQueryRetriever.from_llm(llm=custom_multiquery_chain, ...)`에 전달하면, `MultiQueryRetriever`는 **기본 프롬프트 대신 사용자가 정의한 프롬프트**를 사용해 질문을 증강하게 됩니다.

## 결론

이 방법은 `MultiQueryRetriever`의 질문 생성 과정을 **사용자 정의 프롬프트**로 교체하는 방법입니다. 이를 통해 질문 생성의 방식, 개수, 그리고 원하는 답변의 형식까지 세밀하게 제어하여, 검색 효율을 높이는 데 최적화된 질문들을 직접 만들 수。
