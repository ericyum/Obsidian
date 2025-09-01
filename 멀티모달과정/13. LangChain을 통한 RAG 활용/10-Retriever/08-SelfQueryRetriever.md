# 셀프 쿼리(Self-querying)

`SelfQueryRetriever` 는 자체적으로 질문을 생성하고 해결할 수 있는 기능을 갖춘 검색 도구입니다. 

이는 사용자가 제공한 자연어 질의를 바탕으로, `query-constructing` LLM chain을 사용해 구조화된 질의를 만듭니다. 그 후, 이 구조화된 질의를 기본 벡터 데이터 저장소(VectorStore)에 적용하여 검색을 수행합니다.

이 과정을 통해, `SelfQueryRetriever` 는 단순히 사용자의 입력 질의를 저장된 문서의 내용과 의미적으로 비교하는 것을 넘어서, 사용자의 질의에서 문서의 메타데이터에 대한 **필터를 추출** 하고, 이 필터를 실행하여 관련된 문서를 찾을 수 있습니다. 

[참고]

- LangChain 이 지원하는 셀프 쿼리 검색기(Self-query Retriever) 목록은 [여기](https://python.langchain.com/docs/integrations/retrievers/self_query) 에서 확인해 주시기 바랍니다.

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

## 샘플 데이터 생성

화장품 상품의 설명과 메타데이터를 기반으로 유사도 검색이 가능한 벡터 저장소를 구축합니다.

---

```python
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# 화장품 상품의 설명과 메타데이터 생성
docs = [
    Document(
        page_content="수분 가득한 히알루론산 세럼으로 피부 속 깊은 곳까지 수분을 공급합니다.",
        metadata={"year": 2024, "category": "스킨케어", "user_rating": 4.7},
    ),
    Document(
        page_content="24시간 지속되는 매트한 피니시의 파운데이션, 모공을 커버하고 자연스러운 피부 표현이 가능합니다.",
        metadata={"year": 2023, "category": "메이크업", "user_rating": 4.5},
    ),
    Document(
        page_content="식물성 성분으로 만든 저자극 클렌징 오일, 메이크업과 노폐물을 부드럽게 제거합니다.",
        metadata={"year": 2023, "category": "클렌징", "user_rating": 4.8},
    ),
    Document(
        page_content="비타민 C 함유 브라이트닝 크림, 칙칙한 피부톤을 환하게 밝혀줍니다.",
        metadata={"year": 2023, "category": "스킨케어", "user_rating": 4.6},
    ),
    Document(
        page_content="롱래스팅 립스틱, 선명한 발색과 촉촉한 사용감으로 하루종일 편안하게 사용 가능합니다.",
        metadata={"year": 2024, "category": "메이크업", "user_rating": 4.4},
    ),
    Document(
        page_content="자외선 차단 기능이 있는 톤업 선크림, SPF50+/PA++++ 높은 자외선 차단 지수로 피부를 보호합니다.",
        metadata={"year": 2024, "category": "선케어", "user_rating": 4.9},
    ),
]

# 벡터 저장소 생성
vectorstore = Chroma.from_documents(
    docs, OpenAIEmbeddings(model="text-embedding-3-small")
)
```

**에러:**
```
Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
Failed to send telemetry event ClientCreateCollectionEvent: capture() takes 1 positional argument but 3 were given
```

---

## SelfQueryRetriever

이제 retriever를 인스턴스화할 수 있습니다. 이를 위해서는 문서가 지원하는 **메타데이터 필드** 와 문서 내용에 대한 **간단한 설명을 미리 제공** 해야 합니다.

`AttributeInfo` 클래스를 사용하여 화장품 메타데이터 필드에 대한 정보를 정의합니다.

- 카테고리(`category`): 문자열 타입, 화장품의 카테고리를 나타내며 ['스킨케어', '메이크업', '클렌징', '선케어'] 중 하나의 값을 가집니다.
- 연도(`year`): 정수 타입, 화장품이 출시된 연도를 나타냅니다.
- 사용자 평점(`user_rating`): 실수 타입, 1-5 범위의 사용자 평점을 나타냅니다.

---

```python
from langchain.chains.query_constructor.base import AttributeInfo


# 메타데이터 필드 정보 생성
metadata_field_info = [
    AttributeInfo(
        name="category",
        description="The category of the cosmetic product. One of ['스킨케어', '메이크업', '클렌징', '선케어']",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the cosmetic product was released",
        type="integer",
    ),
    AttributeInfo(
        name="user_rating",
        description="A user rating for the cosmetic product, ranging from 1 to 5",
        type="float",
    ),
]
```

---

`SelfQueryRetriever.from_llm()` 메서드를 사용하여 `retriever` 객체를 생성합니다.

- `llm`: 언어 모델
- `vectorstore`: 벡터 저장소
- `document_contents`: 문서들의 내용 설명
- `metadata_field_info`: 메타데이터 필드 정보

---

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI

# LLM 정의
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# SelfQueryRetriever 생성
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Brief summary of a cosmetic product",
    metadata_field_info=metadata_field_info,
)
```

---

### SelfQueryRetriever의 동작 방식 정리

SelfQueryRetriever는 사용자의 **자연어 질문**을 이해하여 **벡터 저장소**가 처리할 수 있는 **필터 조건**으로 변환하는 과정을 자동화합니다.

1.  **질문 분석 (LLM의 역할)**
    `retriever.invoke()`가 호출되면, `SelfQueryRetriever`는 **사용자의 질문**, **문서 내용에 대한 설명(`document_contents`)**, 그리고 **메타데이터 필드 정보(`metadata_field_info`)**를 LLM에게 전달합니다. LLM은 이 세 가지 정보를 바탕으로 질문이 문서 내용에 대한 것인지, 메타데이터에 대한 것인지를 파악하고 필터링 조건을 분석합니다.

2.  **구조화된 쿼리 생성 (LLM의 역할)**
    분석을 마친 LLM은 자연어 질문을 **'필터 객체'**라는 형태로 변환합니다. 이 필터 객체는 벡터 저장소가 이해할 수 있는 일종의 명령어 뭉치입니다. 예를 들어, "평점이 4.8 이상인 제품"이라는 질문은 `{"user_rating": {"gte": 4.8}}`와 같은 객체로 생성됩니다.

3.  **필터링 및 검색 (벡터 저장소의 역할)**
    LLM이 생성한 필터 객체는 **벡터 저장소에 직접 전달**됩니다. 벡터 저장소는 이 객체를 이용해 내부 데이터를 효율적으로 검색하여 필터 조건에 맞는 문서들만 찾아냅니다.

4.  **결과 반환**
    벡터 저장소가 찾아낸 문서들이 최종 결과로 사용자에게 반환됩니다.

결론적으로, **LLM은 자연어 질문을 벡터 저장소가 이해할 수 있는 '구조화된 쿼리' (즉, 필터 객체)로 변환해주는 역할을 수행합니다.** 실제 필터링과 검색 작업은 전적으로 벡터 저장소가 담당합니다.

---

## Query 테스트

필터를 걸 수 있는 질의를 입력하여 검색을 수행합니다.

---

```python
# Self-query 검색
retriever.invoke("평점이 4.8 이상인 제품을 추천해주세요")
```

**에러:**
```
Failed to send telemetry event CollectionQueryEvent: capture() takes 1 positional argument but 3 were given
```

**결과:**
```
[Document(id='ce6c9eee-8c21-456e-80fb-0d2940dfffa7', metadata={'category': '선케어', 'user_rating': 4.9, 'year': 2024}, page_content='자외선 차단 기능이 있는 톤업 선크림, SPF50+/PA++++ 높은 자외선 차단 지수로 피부를 보호합니다.'),
 Document(id='5d3101a6-af4a-4a72-b92a-d4f52f28a802', metadata={'category': '클렌징', 'user_rating': 4.8, 'year': 2023}, page_content='식물성 성분으로 만든 저자극 클렌징 오일, 메이크업과 노폐물을 부드럽게 제거합니다.')]
```

---

```python
# Self-query 검색
retriever.invoke("2023년에 출시된 상품을 추천해주세요")
```

**결과:**
```
[Document(id='690d84ad-d1b4-4329-8402-9b15ba321298', metadata={'category': '메이크업', 'user_rating': 4.5, 'year': 2023}, page_content='24시간 지속되는 매트한 피니시의 파운데이션, 모공을 커버하고 자연스러운 피부 표현이 가능합니다.'),
 Document(id='eafe1a8d-9cf7-4664-afcd-ab1e28629c19', metadata={'category': '스킨케어', 'user_rating': 4.6, 'year': 2023}, page_content='비타민 C 함유 브라이트닝 크림, 칙칙한 피부톤을 환하게 밝혀줍니다.'),
 Document(id='5d3101a6-af4a-4a72-b92a-d4f52f28a802', metadata={'category': '클렌징', 'user_rating': 4.8, 'year': 2023}, page_content='식물성 성분으로 만든 저자극 클렌징 오일, 메이크업과 노폐물을 부드럽게 제거합니다.')]
```

---

```python
# Self-query 검색
retriever.invoke("카테고리가 선케어인 상품을 추천해주세요")
```

**결과:**
```
[Document(id='ce6c9eee-8c21-456e-80fb-0d2940dfffa7', metadata={'category': '선케어', 'user_rating': 4.9, 'year': 2024}, page_content='자외선 차단 기능이 있는 톤업 선크림, SPF50+/PA++++ 높은 자외선 차단 지수로 피부를 보호합니다.')]
```

---

복합 필터를 사용하여 검색을 수행할 수 있습니다.

---

```python
# Self-query 검색
retriever.invoke(
    "카테고리가 메이크업인 상품 중에서 평점이 4.5 이상인 상품을 추천해주세요"
)
```

**결과:**
```
[Document(id='690d84ad-d1b4-4329-8402-9b15ba321298', metadata={'category': '메이크업', 'user_rating': 4.5, 'year': 2023}, page_content='24시간 지속되는 매트한 피니시의 파운데이션, 모공을 커버하고 자연스러운 피부 표현이 가능합니다.')]
```

---

`k`는 가져올 문서의 수를 의미합니다.

`SelfQueryRetriever`를 사용하여 `k`를 지정할 수도 있습니다. 이는 생성자에 `enable_limit=True`를 전달하여 수행할 수 있습니다.

---

```python
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Brief summary of a cosmetic product",
    metadata_field_info=metadata_field_info,
    enable_limit=True,  # 검색 결과 제한 기능을 활성화합니다.
    search_kwargs={"k": 2},  # k 의 값을 2로 지정하여 검색 결과를 2개로 제한합니다.
)
```

---

2023년도 출시된 상품은 3개가 있지만 "k" 값을 2로 지정하여 2개만 반환하도록 합니다.

---

```python
# Self-query 검색
retriever.invoke("2023년에 출시된 상품을 추천해주세요")
```

**결과:**
```
[Document(id='690d84ad-d1b4-4329-8402-9b15ba321298', metadata={'category': '메이크업', 'user_rating': 4.5, 'year': 2023}, page_content='24시간 지속되는 매트한 피니시의 파운데이션, 모공을 커버하고 자연스러운 피부 표현이 가능합니다.'),
 Document(id='eafe1a8d-9cf7-4664-afcd-ab1e28629c19', metadata={'category': '스킨케어', 'user_rating': 4.6, 'year': 2023}, page_content='비타민 C 함유 브라이트닝 크림, 칙칙한 피부톤을 환하게 밝혀줍니다.')]
```

---

하지만 코드로 명시적으로 `search_kwargs`를 지정하지 않고 query 에서 `1개, 2개` 등의 숫자를 사용하여 검색 결과를 제한할 수 있습니다.

---

```python
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Brief summary of a cosmetic product",
    metadata_field_info=metadata_field_info,
    enable_limit=True,  # 검색 결과 제한 기능을 활성화합니다.
)

# Self-query 검색
retriever.invoke("2023년에 출시된 상품 1개를 추천해주세요")
```

**결과:**
```
[Document(id='690d84ad-d1b4-4329-8402-9b15ba321298', metadata={'category': '메이크업', 'user_rating': 4.5, 'year': 2023}, page_content='24시간 지속되는 매트한 피니시의 파운데이션, 모공을 커버하고 자연스러운 피부 표현이 가능합니다.')]
```

---

```python
# Self-query 검색
retriever.invoke("2023년에 출시된 상품 2개를 추천해주세요")
```

**결과:**
```
[Document(id='690d84ad-d1b4-4329-8402-9b15ba321298', metadata={'category': '메이크업', 'user_rating': 4.5, 'year': 2023}, page_content='24시간 지속되는 매트한 피니시의 파운데이션, 모공을 커버하고 자연스러운 피부 표현이 가능합니다.'),
 Document(id='eafe1a8d-9cf7-4664-afcd-ab1e28629c19', metadata={'category': '스킨케어', 'user_rating': 4.6, 'year': 2023}, page_content='비타민 C 함유 브라이트닝 크림, 칙칙한 피부톤을 환하게 밝혀줍니다.')]
```

---

## 더 깊게 들어가기

내부에서 어떤 일이 일어나는지 확인하고 더 많은 사용자 정의 제어를 하기 위해, 우리는 retriever를 처음부터 재구성할 수 있습니다.

이 과정은 `query-construction chain` 을 생성하는 것부터 시작합니다.

- [참고 튜토리얼](https://github.com/langchain-ai/langchain/blob/master/cookbook/self_query_hotel_search.ipynb) 

---

### `query_constructor` chain 생성

구조화된 쿼리를 생성하는 `query_constructor` chain 을 생성합니다.

`get_query_constructor_prompt` 함수를 사용하여 쿼리 생성기 프롬프트를 가져옵니다.

---

```python
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)

# 문서 내용 설명과 메타데이터 필드 정보를 사용하여 쿼리 생성기 프롬프트를 가져옵니다.
prompt = get_query_constructor_prompt(
    "Brief summary of a cosmetic product",  # 문서 내용 설명
    metadata_field_info,  # 메타데이터 필드 정보
)

# StructuredQueryOutputParser 를 생성
output_parser = StructuredQueryOutputParser.from_components()

# query_constructor chain 을 생성
query_constructor = prompt | llm | output_parser
```

---

`prompt.format()` 메서드를 사용하여 `query` 매개변수에 "dummy question" 문자열을 전달하고, 그 결과를 출력하여 Prompt 내용을 확인해 보겠습니다.

---

```python
# prompt 출력
print(prompt.format(query="dummy question"))
```

**결과:**
```
Your goal is to structure the user's query to match the request schema provided below.

<< Structured Request Schema >>
When responding use a markdown code snippet with a JSON object formatted in the following schema:

```json
{
    "query": string \ text string to compare to document contents
    "filter": string \ logical condition statement for filtering documents
}
```

The query string should contain only text that is expected to match the contents of documents. Any conditions in the filter should not be mentioned in the query as well.

A logical condition statement is composed of one or more comparison and logical operation statements.

A comparison statement takes the form: `comp(attr, val)`:
- `comp` (eq | ne | gt | gte | lt | lte | contain | like | in | nin): comparator
- `attr` (string):  name of attribute to apply the comparison to
- `val` (string): is the comparison value

A logical operation statement takes the form `op(statement1, statement2, ...)`:
- `op` (and | or | not): logical operator
- `statement1`, `statement2`, ... (comparison statements or logical operation statements): one or more statements to apply the operation to

Make sure that you only use the comparators and logical operators listed above and no others.
Make sure that filters only refer to attributes that exist in the data source.
Make sure that filters only use the attributed names with its function names if there are functions applied on them.
Make sure that filters only use format `YYYY-MM-DD` when handling date data typed values.
Make sure that filters take into account the descriptions of attributes and only make comparisons that are feasible given the type of data being stored.
Make sure that filters are only used as needed. If there are no filters that should be applied return "NO_FILTER" for the filter value.

<< Example 1. >>
Data Source:
```json
{
    "content": "Lyrics of a song",
    "attributes": {
        "artist": {
            "type": "string",
            "description": "Name of the song artist"
        },
        "length": {
            "type": "integer",
            "description": "Length of the song in seconds"
        },
        "genre": {
            "type": "string",
            "description": "The song genre, one of \"pop\", \"rock\" or \"rap\""
        }
    }
}
```

User Query:
What are songs by Taylor Swift or Katy Perry about teenage romance under 3 minutes long in the dance pop genre

Structured Request:
```json
{
    "query": "teenager love",
    "filter": "and(or(eq(\"artist\", \"Taylor Swift\"), eq(\"artist\", \"Katy Perry\")), lt(\"length\", 180), eq(\"genre\", \"pop\"))"
}
```


<< Example 2. >>
Data Source:
```json
{
    "content": "Lyrics of a song",
    "attributes": {
        "artist": {
            "type": "string",
            "description": "Name of the song artist"
        },
        "length": {
            "type": "integer",
            "description": "Length of the song in seconds"
        },
        "genre": {
            "type": "string",
            "description": "The song genre, one of \"pop\", \"rock\" or \"rap\""
        }
    }
}
```

User Query:
What are songs that were not published on Spotify

Structured Request:
```json
{
    "query": "",
    "filter": "NO_FILTER"
}
```


<< Example 3. >>
Data Source:
```json
{
    "content": "Brief summary of a cosmetic product",
    "attributes": {
    "category": {
        "description": "The category of the cosmetic product. One of ['\uc2a4\ud0a8\ucf00\uc5b4', '\uba54\uc774\ud06c\uc5c5', '\ud074\ub80c\uc9d5', '\uc120\ucf00\uc5b4']",
        "type": "string"
    },
    "year": {
        "description": "The year the cosmetic product was released",
        "type": "integer"
    },
    "user_rating": {
        "description": "A user rating for the cosmetic product, ranging from 1 to 5",
        "type": "float"
    }
}
}
```

User Query:
dummy question

Structured Request:

```

---

`query_constructor.invoke()` 메서드를 호출하여 주어진 쿼리에 대한 처리를 수행합니다.

---

```python
query_output = query_constructor.invoke(
    {
        # 쿼리 생성기를 호출하여 주어진 질문에 대한 쿼리를 생성합니다.
        "query": "2023년도에 출시한 상품 중 평점이 4.5 이상인 상품중에서 스킨케어 제품을 추천해주세요"
    }
)
```

---

생성된 쿼리를 확인해 보겠습니다.

---

```python
# 쿼리 출력
query_output.filter.arguments
```

**결과:**
```
[Comparison(comparator=<Comparator.GTE: 'gte'>, attribute='year', value=2023),
 Comparison(comparator=<Comparator.GTE: 'gte'>, attribute='user_rating', value=4.5),
 Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='category', value='스킨케어')]
```

---

**위는 LLM이 사용자의 자연어 질문을 분석하여 스스로 생성한 필터의 내용이다.**

Self-query retriever의 핵심 요소는 query constructor입니다. 훌륭한 검색 시스템을 만들기 위해서는 query constructor가 잘 작동하도록 해야 합니다.

이를 위해서는 **프롬프트(Prompt), 프롬프트 내의 예시, 속성 설명 등을 조정** 해야 합니다.

---

### 구조화된 쿼리 변환기(Structured Query Translator)를 사용하여 구조화된 쿼리로 변환

다음으로 중요한 요소는 structured query translator입니다. 

이는 일반적인 `StructuredQuery` 객체를 사용 중인 vector store의 구문에 맞는 메타데이터 필터로 변환하는 역할을 담당합니다.

---

```python
from langchain.retrievers.self_query.chroma import ChromaTranslator

retriever = SelfQueryRetriever(
    query_constructor=query_constructor,  # 이전에 생성한 query_constructor chain 을 지정
    vectorstore=vectorstore,  # 벡터 저장소를 지정
    structured_query_translator=ChromaTranslator(),  # 쿼리 변환기
)
```

---

`retriever.invoke()` 메서드를 사용하여 주어진 질문에 대한 답변을 생성합니다.

1. query_constructor (LLM이 담당):
 - 사용자의 자연어 질문을 받아서 Comparison, Operation 등과 같은 LangChain의 일반적인 구조화된 쿼리 객체로 변환합니다. (예: Comparison(attribute='year', value=2023))

2. structured_query_translator (변환기가 담당):
 - ChromaTranslator는 query_constructor가 생성한 일반적인 쿼리 객체를 받아, 이를 Chroma 벡터 저장소가 실제로 이해할 수 있는 필터 구문으로 변환합니다. (예: { 'year': { '$gte': 2023 } })

3. vectorstore (벡터 저장소가 담당):
 - 최종적으로 ChromaTranslator에 의해 변환된 필터 구문을 받아 실제 검색을 실행하고, 해당 문서들을 반환합니다.

---

```python
retriever.invoke(
    # 질문
    "2023년도에 출시한 상품 중 평점이 4.5 이상인 상품중에서 스킨케어 제품을 추천해주세요"
)
```

**결과:**
```
[Document(id='eafe1a8d-9cf7-4664-afcd-ab1e28629c19', metadata={'category': '스킨케어', 'user_rating': 4.6, 'year': 2023}, page_content='비타민 C 함유 브라이트닝 크림, 칙칙한 피부톤을 환하게 밝혀줍니다.'),
 Document(id='85571ead-7cd9-43ec-82de-e0faa0ecf62f', metadata={'category': '스킨케어', 'user_rating': 4.7, 'year': 2024}, page_content='수분 가득한 히알루론산 세럼으로 피부 속 깊은 곳까지 수분을 공급합니다.')]
```