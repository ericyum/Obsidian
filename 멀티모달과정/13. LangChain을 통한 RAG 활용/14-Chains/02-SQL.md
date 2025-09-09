# SQL

이번 튜토리얼은 `create_sql_query_chain` 을 활용하여 생성한 체인으로 SQL 쿼리를 생성 후 실행, 그리고 답변 도출하는 방법에 대해서 다룹니다. 그리고, SQL Agent 와의 동작 방식의 차이도 알아보겠습니다.

```python
# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()
```

```python
# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("SQL")
```

SQL Database 정보를 불러옵니다.

```python
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase

# SQLite 데이터베이스에 연결합니다.
db = SQLDatabase.from_uri("sqlite:///data/finance.db")

# 데이터베이스를 출력합니다.
print(db.dialect)

# 사용 가능한 테이블 이름들을 출력합니다.
print(db.get_usable_table_names())
```

LLM 객체를 생성하고 LLM 과 DB 를 매개변수로 입력하여 chain 을 생성합니다.

여기서 model 변경시 원활하게 동작하지 않을 수 있어, 이번 튜토리얼은 `gpt-3.5-turbo` 로 진행합니다.

```python
# model 은 gpt-3.5-turbo 를 지정
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# LLM 과 DB 를 매개변수로 입력하여 chain 을 생성합니다.
chain = create_sql_query_chain(llm, db)
```

(옵션) 아래의 방식으로 Prompt 를 직접 지정할 수 있습니다.

직접 작성시 table_info 와 더불어 설명가능한 column description 을 추가할 수 있습니다.

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:
{table_info}

Here is the description of the columns in the tables:
`cust`: customer name
`prod`: product name
`trans`: transaction date

Question: {input}"""
).partial(dialect=db.dialect)

# model 은 gpt-3.5-turbo 를 지정
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# LLM 과 DB 를 매개변수로 입력하여 chain 을 생성합니다.
chain = create_sql_query_chain(llm, db, prompt)
```

chain 을 실행하면 DB 기반으로 쿼리를 생성합니다.

```python
# chain 을 실행하고 결과를 출력합니다.
generated_sql_query = chain.invoke({"question": "고객의 이름을 나열하세요"})

# 생성된 쿼리를 출력합니다.
print(generated_sql_query.__repr__())
```

다음은 생성한 쿼리가 맞게 동작하는지 확인해 볼 차례입니다.

```python
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

# 생성한 쿼리를 실행하기 위한 도구를 생성합니다.
execute_query = QuerySQLDataBaseTool(db=db)
```

```python
execute_query.invoke({"query": generated_sql_query})
```

```python
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

# 도구
execute_query = QuerySQLDataBaseTool(db=db)

# SQL 쿼리 생성 체인
write_query = create_sql_query_chain(llm, db)

# 생성한 쿼리를 실행하기 위한 체인을 생성합니다.
chain = write_query | execute_query
```

```python
# 실행 결과 확인
chain.invoke({"question": "테디의 이메일을 조회하세요"})
```

## 답변을 LLM 으로 증강-생성

이전 단계에서 생성한 chain 을 사용하면 답변이 단답형 형식으로 출력됩니다. 이를 LCEL 문법의 체인으로 좀 더 자연스러운 답변을 받을 수 있도록 조정할 수 있습니다.

```python
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

answer = answer_prompt | llm | StrOutputParser()

# 생성한 쿼리를 실행하고 결과를 출력하기 위한 체인을 생성합니다.
chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
)
```

```python
runnable = RunnablePassthrough.assign(query=write_query).assign(result=itemgetter("query") | execute_query)

runnable.invoke({"question": "테디의 transaction 의 합계를 구하세요"})
```

```python
# 실행 결과 확인
chain.invoke({"question": "테디의 transaction 의 합계를 구하세요"})
```

## Agent

Agent를 활용하여 Sql 쿼리를 생성하고 실행 결과를 답변으로 출력이 가능합니다.

```python
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# model 은 gpt-3.5-turbo 를 지정
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# SQLite 데이터베이스에 연결합니다.
db = SQLDatabase.from_uri("sqlite:///data/finance.db")

# Agent 생성
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
```

```python
# 실행 결과 확인
agent_executor.invoke(
    {"input": "테디와 셜리의 transaction 의 합계를 구하고 비교하세요"}
)
```
