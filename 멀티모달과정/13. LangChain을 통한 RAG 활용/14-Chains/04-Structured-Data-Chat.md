```python
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()
```

# Python REPL(Read-Eval-Print Loop)

```python
print(eval("100+200"))
```

```python
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonAstREPLTool

df = pd.read_csv("./data/titanic.csv")
# titanic.csv 파일에서 데이터를 읽어와 DataFrame으로 저장합니다.
tool = PythonAstREPLTool(locals={"df": df})
# PythonAstREPLTool을 사용하여 로컬 변수 'df'를 포함하는 환경을 생성합니다.
tool.invoke("df")
# 'df' DataFrame에서 'Fare' 열의 평균 값을 계산합니다.
```

```python
hello = """
print("Hello, world!")

def add(a, b):
    return a + b

print(add(30, 40))

import pandas as pd

df = pd.read_csv("./data/titanic.csv")
df.head()
"""
```

```python
tool = PythonAstREPLTool(locals={"df": df})
# PythonAstREPLTool을 사용하여 로컬 변수 'df'를 포함하는 환경을 생성합니다.
tool.invoke(hello)
```

## 데이터 로드


pandas를 활용하여 csv 파일을 DataFrame 으로 로드합니다.


```python
import pandas as pd

df = pd.read_csv("data/titanic.csv")
df.head()
```

## Pandas DataFrame Agent

```python
# !pip install tabulate
```

```python
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.callbacks.base import BaseCallbackHandler


class StreamCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)


# # 에이전트 생성
agent = create_pandas_dataframe_agent(
    ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
        streaming=True,
        callbacks=[StreamCallback()],
    ),  # 모델 정의
    df,  # 데이터프레임
    verbose=True,  # 추론과정 출력
    # AgentType.ZERO_SHOT_REACT_DESCRIPTION # AgentType.ZERO_SHOT_REACT_DESCRIPTION: OpenAI의 함수 호출 기능이 아닌, LLM의 텍스트 기반 추론 능력을 활용해 도구를 사용하도록 하는 범용적인 에이전트 접근 방식
    agent_type=AgentType.OPENAI_FUNCTIONS, # AgentType.OPENAI_FUNCTIONS: LangChain 에이전트가 OpenAI 모델의 강력한 함수 호출 기능을 활용하여 데이터 분석과 같은 작업을 더 정확하고 효율적으로 수행하도록 설정하는 중요한 파라미터
    allow_dangerous_code=True, # Python 코드 중 위험성이 있는 코드도 허용
)
```

```python
# 질의
agent.invoke({"input": "데이터의 행과 열의 갯수는 어떻게 돼?"})
```

```python
# 질의
agent.run("남자 승객의 생존율을 어떻게 돼? %로 알려줘")
```

```python
# 질의
agent.run(
    "나이가 15세 이하인 승객중 1,2등급에 탑승한 남자 승객의 생존율은 어떻게 돼? %로 알려줘"
)
```

```python
# 질의
agent.run(
    "Pclass 가 1등급인 승객 중에서 나이가 20세~30세 사이이고, 여성 승객의 생존율은 어떻게 돼? %로 알려줘"
)
```

## 2개 이상의 DataFrame


2개 이상의 데이터프레임에 기반한 LLM 기반 질의를 할 수 있습니다. 2개 이상의 데이터프레임 입력시 `[]` 로 묶어주면 됩니다.


```python
# 샘플 데이터프레임 생성
df1 = df.copy()
df1 = df1.fillna(0) # fillna() : 결측치를 0으로 채운다.
df1.head()
```

```python
# 에이전트 생성
agent = create_pandas_dataframe_agent(
    ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
        streaming=True,
        callbacks=[StreamCallback()],
    ),
    [df, df1],
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True # Python 코드 중 위험성이 있는 코드도 허용
)

# 질의
agent.invoke({"input": "나이 컬럼의 나이의 평균차이는 어떻게 돼? %로 구해줘."})
```
