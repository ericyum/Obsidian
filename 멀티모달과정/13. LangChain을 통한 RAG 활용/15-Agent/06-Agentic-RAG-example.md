```python
# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()
```

```python
# !pip install -U duckduckgo-search
```

```python
import os
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor, tool
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun # 인터넷 검색 툴
```

```python
# 환경 변수 설정 (필요에 따라 주석 해제 후 API 키 입력)
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# 1. 영화 정보가 담긴 문서 리스트 생성
docs = [
    Document(
        page_content="크리스토퍼 놀란 감독의 SF 영화. 꿈속으로 들어가 현실을 조작한다.",
        metadata={
            "title": "인셉션",
            "director": "크리스토퍼 놀란",
            "year": 2010,
            "genre": "SF",
            "rating": 8.8,
        },
    ),
    Document(
        page_content="거대한 우주 모험을 다룬 영화. 블랙홀과 시간 여행이 주된 소재이다.",
        metadata={
            "title": "인터스텔라",
            "director": "크리스토퍼 놀란",
            "year": 2014,
            "genre": "SF",
            "rating": 8.6,
        },
    ),
    Document(
        page_content="미국 대공황 시대를 배경으로 한 마술 대결 영화.",
        metadata={
            "title": "프레스티지",
            "director": "크리스토퍼 놀란",
            "year": 2006,
            "genre": "미스터리",
            "rating": 8.5,
        },
    ),
    Document(
        page_content="스파이더맨의 능력과 책임에 대한 이야기. 고등학생 영웅의 성장기.",
        metadata={
            "title": "스파이더맨: 홈커밍",
            "director": "존 왓츠",
            "year": 2017,
            "genre": "액션",
            "rating": 7.4,
        },
    ),
    Document(
        page_content="마블 히어로들이 팀을 이루어 지구를 구하는 이야기. 화려한 액션이 특징.",
        metadata={
            "title": "어벤져스",
            "director": "조스 웨던",
            "year": 2012,
            "genre": "액션",
            "rating": 8.1,
        },
    ),
]

# 2. 벡터 스토어 및 임베딩 모델 설정
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings)
llm = ChatOpenAI(temperature=0)

# 3. 셀프 쿼리 리트리버를 위한 메타데이터 스키마 정의
metadata_field_info = [
    AttributeInfo(name="title", description="영화의 제목", type="string"),
    AttributeInfo(name="director", description="영화 감독의 이름", type="string"),
    AttributeInfo(name="year", description="영화가 개봉된 연도", type="integer"),
    AttributeInfo(name="genre", description="영화의 장르", type="string"),
    AttributeInfo(name="rating", description="IMDb 평점 (10점 만점)", type="float"),
]

# 4. 셀프 쿼리 리트리버 생성
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="영화의 줄거리와 특징에 대한 설명입니다.",
    metadata_field_info=metadata_field_info,
    verbose=True,
)


# 5. 셀프 쿼리 리트리버를 툴로 정의
@tool
def movie_retriever_tool(query: str) -> str:
    """
    사용자의 자연어 쿼리를 기반으로 영화를 검색합니다.
    감독, 개봉 연도, 장르, 평점과 같은 조건을 함께 사용하여 검색할 수 있습니다.
    예시: "크리스토퍼 놀란 감독이 2010년 이후에 만든 SF 영화"
    """
    return self_query_retriever.invoke(query)


# 6. 두 번째 툴: 인터넷 검색
internet_search = DuckDuckGoSearchRun()


@tool
def internet_search_tool(query: str) -> str:
    """
    웹에서 최신 정보를 검색하는 도구입니다. 영화 정보를 찾을 때 사용하세요.
    """
    try:
        return internet_search.invoke(query)
    except Exception as e:
        return f"인터넷 검색 중 오류 발생: {str(e)}"


# 7. 에이전트 및 에이전트 실행자 생성
tools = [movie_retriever_tool, internet_search_tool]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 영화 검색 전문가입니다. 사용자 질문에 답변하기 위해 주어진 툴을 활용하세요. 처음에는 VectorStore에 저장되어있는 내용을 서치하고, 만약 원하는 내용을 찾지 못했을 때에는 Search Tool을 통해 검색을 하세요.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

```python
# 8. 에이전트 실행
print("--- case 1: 복잡한 조건 검색 ---")
# "2010년 이후에 개봉한 크리스토퍼 놀란 감독의 SF 영화를 찾아줘"
result1 = agent_executor.invoke(
    {"input": "2010년 이후에 개봉한 크리스토퍼 놀란 감독의 SF 영화 알려줘"}
)
print(result1["output"])
```

```python
print("\n--- case 2: 단순한 쿼리 검색 ---")
# "마블 영화 중에서 평점 8.0점 이상인 영화를 찾아줘"
result2 = agent_executor.invoke(
    {"input": "마블 영화 중에서 평점 8.0점 이상인 영화를 찾아줘"}
)
print(result2["output"])
```

```python
print("\n--- case 2: 단순한 쿼리 검색 ---")
# "마블 영화 중에서 평점 8.0점 이상인 영화를 찾아줘"
result2 = agent_executor.invoke({"input": "아이언맨1편에 대해 알려줘"})
print(result2["output"])
```

```