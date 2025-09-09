# 구조화된 출력을 사용하는 체인(with_structured_output)

```python
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()
```

```python
# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("Structured-Output-Chain")
```

특정 주제에 대한 4지선다형 퀴즈를 생성하는 과정을 구현합니다.

`Quiz` 클래스는 퀴즈의 질문, 난이도, 그리고 네 개의 선택지를 정의합니다.

`ChatOpenAI` 인스턴스는 GPT-4o 모델을 사용하여 자연어 처리를 수행하고, `ChatPromptTemplate`는 퀴즈 생성을 위한 대화형 프롬프트를 정의합니다.

```python
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List


class Quiz(BaseModel):
    """4지선다형 퀴즈의 정보를 추출합니다"""

    question: str = Field(..., description="퀴즈의 질문")
    level: str = Field(
        ..., description="퀴즈의 난이도를 나타냅니다. (쉬움, 보통, 어려움)"
    )
    options: List[str] = Field(..., description="퀴즈의 4개의 선택지 입니다.")


llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a world-famous quizzer and generates quizzes in structured formats.",
        ),
        (
            "human",
            "TOPIC 에 제시된 내용과 관련한 4지선다형 퀴즈를 출제해 주세요. 만약, 실제 출제된 기출문제가 있다면 비슷한 문제를 만들어 출제하세요."
            "단, 문제에 TOPIC 에 대한 내용이나 정보는 포함하지 마세요. \nTOPIC:\n{topic}",
        ),
        ("human", "Tip: Make sure to answer in the correct format"),
    ]
)

# 구조화된 출력을 위한 모델 생성
llm_with_structured_output = llm.with_structured_output(Quiz)

# 퀴즈 생성 체인 생성
chain = prompt | llm_with_structured_output
```

```python
# 퀴즈 생성을 요청합니다.
generated_quiz = chain.invoke({"topic": "ADSP(데이터 분석 준전문가) 자격 시험"})
```

생성된 퀴즈를 출력합니다.

```python
# 생성된 퀴즈 출력
print(f"{generated_quiz.question} (난이도: {generated_quiz.level})\n")
for i, opt in enumerate(generated_quiz.options):
    print(f"{i+1}) {opt}")
```