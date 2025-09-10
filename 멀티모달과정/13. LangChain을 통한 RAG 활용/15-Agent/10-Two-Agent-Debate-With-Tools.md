# 도구를 활용한 토론 에이전트(Agent Debates with Tools)

이 예제는 에이전트가 도구에 접근할 수 있는 다중 에이전트 대화를 시뮬레이션하는 방법을 보여줍니다.

LangSmith 추적을 위하여 초기화 합니다.

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
logging.langsmith("CH15-Debate-Agent")
```

## `DialogueAgent` 및 `DialogueSimulator`

이 노트북에서는 권한이 있는 에이전트가 발언할 사람을 결정하는 다중 에이전트 시뮬레이션을 구현하는 방법을 보여드립니다. 이는 다중 에이전트 분산형 화자 선택과 정반대의 선택 방식을 따릅니다.

[Multi-Player Authoritarian Speaker Selection](https://python.langchain.com/en/latest/use_cases/agent_simulations/multiagent_authoritarian.html)에서 정의된 것과 동일한 `DialogueAgent`와 `DialogueSimulator` 클래스를 사용할 것입니다.

## `DialogueAgent`

- `send` 메서드는 현재까지의 대화 기록과 에이전트의 접두사를 사용하여 채팅 모델에 메시지를 전달하고 응답을 반환합니다.
- `receive` 메서드는 다른 에이전트가 보낸 메시지를 대화 기록에 추가합니다.

```python
from typing import Callable, List


from langchain.schema (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI


class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        # 에이전트의 이름을 설정합니다.
        self.name = name
        # 시스템 메시지를 설정합니다.
        self.system_message = system_message
        # LLM 모델을 설정합니다.
        self.model = model
        # 에이전트 이름을 지정합니다.
        self.prefix = f"{self.name}: "
        # 에이전트를 초기화합니다.
        self.reset()

    def reset(self):
        """
        대화 내역을 초기화합니다.
        """
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        """
        메시지에 시스템 메시지 + 대화내용과 마지막으로 에이전트의 이름을 추가합니다.
        """
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join([self.prefix] + self.message_history)),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        name 이 말한 message 를 메시지 내역에 추가합니다.
        """
        self.message_history.append(f"{name}: {message}")
```

## `DialogueSimulator`

- `inject` 메서드는 주어진 이름(`name`)과 메시지(`message`)로 대화를 시작하고, 모든 에이전트가 해당 메시지를 받도록 합니다.
- `step` 메서드는 다음 발언자를 선택하고, 해당 발언자가 메시지를 보내면 모든 에이전트가 메시지를 받도록 합니다. 그리고 현재 단계를 증가시킵니다.

여러 에이전트 간의 대화를 시뮬레이션하는 기능을 제공합니다.

`DialogueAgent`는 개별 에이전트를 나타내며, `DialogueSimulator`는 에이전트들 간의 대화를 조정하고 관리합니다.

```python
class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        # 에이전트 목록을 설정합니다.
        self.agents = agents
        # 시뮬레이션 단계를 초기화합니다.
        self._step = 0
        # 다음 발언자를 선택하는 함수를 설정합니다.
        self.select_next_speaker = selection_function

    def reset(self):
        # 모든 에이전트를 초기화합니다.
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """
        name 의 message 로 대화를 시작합니다.
        """
        # 모든 에이전트가 메시지를 받습니다.
        for agent in self.agents:
            agent.receive(name, message)

        # 시뮬레이션 단계를 증가시킵니다.
        self._step += 1

    def step(self) -> tuple[str, str]:
        # 1. 다음 발언자를 선택합니다.
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. 다음 발언자에게 메시지를 전송합니다.
        message = speaker.send()

        # 3. 모든 에이전트가 메시지를 받습니다.
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. 시뮬레이션 단계를 증가시킵니다.
        self._step += 1

        # 발언자의 이름과 메시지를 반환합니다.
        return speaker.name, message
```

## `DialogueAgentWithTools`

`DialogueAgent`를 확장하여 도구를 사용할 수 있도록 `DialogueAgentWithTools` 클래스를 정의합니다.

- `DialogueAgentWithTools` 클래스는 `DialogueAgent` 클래스를 상속받아 구현되었습니다.
- `send` 메서드는 에이전트가 메시지를 생성하고 반환하는 역할을 합니다.
- `create_openai_tools_agent` 함수를 사용하여 에이전트 체인을 초기화합니다.
  - 초기화시 에이전트가 사용할 도구(tools) 를 정의합니다.

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub


class DialogueAgentWithTools:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
        tools,
    ) -> None:
        # 부모 클래스의 생성자를 호출합니다.
        super().__init__(name, system_message, model)
        # 주어진 도구 이름과 인자를 사용하여 도구를 로드합니다.
        self.tools = tools

    def send(self) -> str:
        """
        메시지 기록에 챗 모델을 적용하고 메시지 문자열을 반환합니다.
        """
        prompt = hub.pull("hwchase17/openai-functions-agent")
        agent = create_openai_tools_agent(self.model, self.tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=False)
        # AI 메시지를 생성합니다.
        message = AIMessage(
            content=agent_executor.invoke(
                {
                    "input": "\n".join(
                        [self.system_message.content]
                        + [self.prefix]
                        + self.message_history
                    )
                }
            )["output"]
        )

        # 생성된 메시지의 내용을 반환합니다.
        return message.content
```

## 도구 설정

### 문서 검색 도구(Retrieval Tool)를 정의합니다.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader

# PDF 파일 로드. 파일의 경로 입력
loader1 = TextLoader("data/의대증원반대.txt")
loader2 = TextLoader("data/의대증원찬성.txt")

# 텍스트 분할기를 사용하여 문서를 분할합니다.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# 문서를 로드하고 분할합니다.
docs1 = loader1.load_and_split(text_splitter)
docs2 = loader2.load_and_split(text_splitter)

# VectorStore를 생성합니다.
vector1 = FAISS.from_documents(docs1, OpenAIEmbeddings())
vector2 = FAISS.from_documents(docs2, OpenAIEmbeddings())

# Retriever를 생성합니다.
doctor_retriever = vector1.as_retriever(search_kwargs={"k": 5})
gov_retriever = vector2.as_retriever(search_kwargs={"k": 5})
```

```python
# langchain 패키지의 tools 모듈에서 retriever 도구를 생성하는 함수를 가져옵니다.
from langchain.tools.retriever import create_retriever_tool

doctor_retriever_tool = create_retriever_tool(
    doctor_retriever,
    name="document_search",
    description="This is a document about the Korean Medical Association's opposition to the expansion of university medical schools. "
    "Refer to this document when you want to present a rebuttal to the proponents of medical school expansion.",
)

gov_retriever_tool = create_retriever_tool(
    gov_retriever,
    name="document_search",
    description="This is a document about the Korean government's support for the expansion of university medical schools. "
    "Refer to this document when you want to provide a rebuttal to the opposition to medical school expansion.",
)
```

### 인터넷 검색 도구

인터넷에서 검색할 수 있는 도구를 생성합니다.

```python
# TavilySearchResults 클래스를 langchain_community.tools.tavily_search 모듈에서 가져옵니다.
from langchain_community.tools.tavily_search import TavilySearchResults

# TavilySearchResults 클래스의 인스턴스를 생성합니다
# k=6은 검색 결과를 6개까지 가져오겠다는 의미입니다
search = TavilySearchResults(k=6)
```

## 각 에이전트가 활용할 수 있는 도구를 설정합니다.

- `names` 딕셔너리는 토론자의 이름(prefix name) 과 각각의 토론 에이전트가 활용할 수 있는 도구를 정의합니다.
- `topic` 토론의 주제를 선정합니다.

### ① 문서에 기반한 도구

```python
names = {
    "Doctor Union(의사협회)": [doctor_retriever_tool],  # 의사협회 에이전트 도구 목록
    "Government(대한민국 정부)": [gov_retriever_tool],  # 정부 에이전트 도구 목록
}

# 토론 주제 선정
topic = "2024 현재, 대한민국 대학교 의대 정원 확대 충원은 필요한가?"

# 토론자를 설명하는 문구의 단어 제한
word_limit = 50
```

### ② 검색(Search) 기반 도구

```python
names_search = {
    "Doctor Union(의사협회)": [search],  # 의사협회 에이전트 도구 목록
    "Government(대한민국 정부)": [search],  # 정부 에이전트 도구 목록
}
# 토론 주제 선정
topic = "2024년 현재, 대한민국 대학교 의대 정원 확대 충원은 필요한가?"
word_limit = 50  # 작업 브레인스토밍을 위한 단어 제한
```

## LLM을 활용하여 주제 설명에 세부 내용 추가하기

LLM(Large Language Model)을 사용하여 주어진 주제에 대한 설명을 보다 상세하게 만들 수 있습니다.

이를 위해서는 먼저 주제에 대한 간단한 설명이나 개요를 LLM에 입력으로 제공합니다. 그런 다음 LLM에게 해당 주제에 대해 더 자세히 설명해줄 것을 요청합니다.

LLM은 방대한 양의 텍스트 데이터를 학습했기 때문에, 주어진 주제와 관련된 추가적인 정보와 세부 사항을 생성해낼 수 있습니다. 이를 통해 초기의 간단한 설명을 보다 풍부하고 상세한 내용으로 확장할 수 있습니다.

- 주어진 대화 주제(topic)와 참가자(names)를 기반으로 대화에 대한 설명(`conversation_description`)을 생성합니다.
- `agent_descriptor_system_message` 는 대화 참가자에 대한 설명을 추가할 수 있다는 내용의 SystemMessage입니다.
- `generate_agent_description` 함수는 각 참가자(name)에 대하여 LLM 이 생성한 설명을 생성합니다.
  - `agent_specifier_prompt` 는 대화 설명과 참가자 이름, 단어 제한(`word_limit`)을 포함하는 HumanMessage로 구성됩니다.
  - ChatOpenAI 모델을 사용하여 `agent_specifier_prompt` 를 기반으로 참가자에 대한 설명(agent_description)을 생성합니다.

```python
conversation_description = f"Here is the topic of conversation: {topic}
The participants are: {', '.join(names.keys())}"

agent_descriptor_system_message = SystemMessage(
    content="You can add detail to the description of the conversation participant."
)


def generate_agent_description(name):
    agent_specifier_prompt = [
        agent_descriptor_system_message,
        HumanMessage(
            content=f"""{conversation_description}
            Please reply with a description of {name}, in {word_limit} words or less in expert tone. 
            Speak directly to {name}.
            Give them a point of view.
            Do not add anything else. Answer in KOREAN."""
        ),
    ]
    # ChatOpenAI를 사용하여 에이전트 설명을 생성합니다.
    agent_description = ChatOpenAI(temperature=0)(agent_specifier_prompt).content
    return agent_description


# 각 참가자의 이름에 대한 에이전트 설명을 생성합니다.
agent_descriptions = {name: generate_agent_description(name) for name in names}

# 생성한 에이전트 설명을 출력합니다.
agent_descriptions
```

직접 각 토론자의 간략한 입장에 대하여 설명하는 문구를 작성할 수 있습니다.

```python
agent_descriptions = {
    "Doctor Union(의사협회)": "의사협회는 의료계의 권익을 보호하고 의사들의 이해관계를 대변하는 기관입니다. 의사들의 업무 환경과 안전을 중시하며, 환자 안전과 질 높은 의료 서비스를 제공하기 위해 노력합니다. "
    "지금도 의사의 수는 충분하다는 입장이며, 의대 증원은 필수 의료나 지방 의료 활성화에 대한 실효성이 떨어집니다. 의대 증원을 감행할 경우, 의료 교육 현장의 인프라가 갑작스러운 증원을 감당하지 못할 것이란 우려를 표합니다.",
    "Government(대한민국 정부)": "대한민국 정부는 국가의 행정을 책임지는 주체로서, 국민의 복지와 발전을 책임져야 합니다. "
    "우리나라는 의사수가 절대 부족한 상황이며, 노인인구가 늘어나면서 의료 수요가 급증하고 있습니다. OECD 국가들도 최근 의사수를 늘렸습니다. 또한, 증원된 의사 인력이 필수의료와 지역 의료로 갈 수있도록 튼튼한 의료사고 안정망 구축 및 보상 체계의 공정성을 높이고자 합니다.",
}
```

## 전역 System Message 설정

System message는 대화형 AI 시스템에서 사용자의 입력에 앞서 시스템이 생성하는 메시지입니다.

이러한 메시지는 대화의 맥락을 설정하고, 사용자에게 각 에이전트의 **입장과 목적** 을 알려주는 역할을 합니다.

효과적인 system message를 작성하면 사용자와의 상호작용을 원활하게 하고, 대화의 질을 높일 수 있습니다.

**프롬프트 설명**

- 에이전트의 이름과 설명을 알립니다.
- 에이전트는 도구를 사용하여 정보를 찾고 대화 상대방의 주장을 반박해야 합니다.
- 에이전트는 출처를 인용해야 하며, 가짜 인용을 하거나 찾아보지 않은 출처를 인용해서는 안 됩니다.
- 에이전트는 자신의 관점에서 말을 마치는 즉시 대화를 중단해야 합니다.

```python
def generate_system_message(name, description, tools):
    return f"""{conversation_description}
    
Your name is {name}.

Your description is as follows: {description}

Your goal is to persuade your conversation partner of your point of view.

DO look up information with your tool to refute your partner's claims.
DO cite your sources.

DO NOT fabricate fake citations.
DO NOT cite any source that you did not look up.

DO NOT restate something that has already been said in the past.
DO NOT add anything else.

Stop speaking the moment you finish speaking from your perspective.

Answer in KOREAN.
"""


agent_system_messages = {
    name: generate_system_message(name, description, tools)
    for (name, tools), description in zip(names.items(), agent_descriptions.values())
}
```

```python
# 에이전트 시스템 메시지를 순회합니다.
for name, system_message in agent_system_messages.items():
    # 에이전트의 이름을 출력합니다.
    print(name)
    # 에이전트의 시스템 메시지를 출력합니다.
    print(system_message)
```

`topic_specifier_prompt`를 정의하여 주어진 주제를 더 구체화하는 프롬프트를 생성합니다.

- `temperature` 를 조절하여 더 다양한 주제를 생성할 수 있습니다.

```python
topic_specifier_prompt = [
    # 주제를 더 구체적으로 만들 수 있습니다.
    SystemMessage(content="You can make a topic more specific."),
    HumanMessage(
        content=f"""{topic}
        
        You are the moderator. 
        Please make the topic more specific.
        Please reply with the specified quest in 100 words or less.
        Speak directly to the participants: {*names,}.  
        Do not add anything else.
        Answer in Korean."""  # 다른 것은 추가하지 마세요.
    ),
]
# 구체화된 주제를 생성합니다.
specified_topic = ChatOpenAI(temperature=1.0)(topic_specifier_prompt).content

print(f"Original topic:\n{topic}\n")  # 원래 주제를 출력합니다.
print(f"Detailed topic:\n{specified_topic}\n")  # 구체화된 주제를 출력합니다.
```

혹은 아래와 같이 직접 지정할 수 있습니다.

```python
# 직접 세부 주제 설정
specified_topic = "정부는 2025년 입시부터 의대 입학정원을 2000명 늘린다고 발표했습니다. 이에 의사단체는 전국에서 규탄집회를 열어 반발하고 있습니다. 의대 정원 확대를 둘러싼 논란 쟁점을 짚어보고, 필수 의료와 지역 의료 해법에 대해서 토론해주세요."
```

## 토론 Loop

토론 루프는 프로그램의 핵심 실행 부분으로, 주요 작업이 반복적으로 수행되는 곳입니다.

- 여기서 주요 작업은 각 에이전트의 메시지 청취 -> 도구를 활용하여 근거 탐색 -> 반박 의견 제시 등을 포함합니다.

```python
# 이는 결과가 컨텍스트 제한을 초과하는 것을 방지하기 위함입니다.
agents = [
    DialogueAgentWithTools(
        name=name,
        system_message=SystemMessage(content=system_message),
        model=ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0.2),
        tools=tools,
    )
    for (name, tools), system_message in zip(
        names.items(), agent_system_messages.values()
    )
]

agents_with_search = [
    DialogueAgentWithTools(
        name=name,
        system_message=SystemMessage(content=system_message),
        model=ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0.2),
        tools=tools,
    )
    for (name, tools), system_message in zip(
        names_search.items(), agent_system_messages.values()
    )
]

agents.extend(agents_with_search)
agents
```

`select_next_speaker` 함수는 다음 발언자를 선택하는 역할을 합니다.

```python
def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    # 다음 발언자를 선택합니다.
    # step을 에이전트 수로 나눈 나머지를 인덱스로 사용하여 다음 발언자를 순환적으로 선택합니다.
    idx = (step) % len(agents)
    return idx
```

여기서는 최대 6번의 토론을 실행합니다.(`max_iters=6`)

- `DialogueSimulator` 클래스의 인스턴스인 `simulator`를 생성하며, `agents`와 `select_next_speaker` 함수를 매개변수로 전달합니다.
- `simulator.reset()` 메서드를 호출하여 시뮬레이터를 초기화합니다.
- `simulator.inject()` 메서드를 사용하여 "Moderator" 에이전트에게 `specified_topic`을 주입합니다.
- "Moderator"가 말한 `specified_topic`을 출력합니다.
- `n`이 `max_iters`보다 작은 동안 반복합니다:
  - `simulator.step()` 메서드를 호출하여 다음 에이전트의 이름(`name`)과 메시지(`message`)를 가져옵니다.
  - 에이전트의 이름과 메시지를 출력합니다.
  - `n`을 1 증가시킵니다.

```python
max_iters = 30  # 최대 반복 횟수를 6으로 설정합니다.
n = 0  # 반복 횟수를 추적하는 변수를 0으로 초기화합니다.

# DialogueSimulator 객체를 생성하고, agents와 select_next_speaker 함수를 전달합니다.
simulator = DialogueSimulator(
    agents=agents_with_search, selection_function=select_next_speaker
)

# 시뮬레이터를 초기 상태로 리셋합니다.
simulator.reset()

# Moderator가 지정된 주제를 제시합니다.
simulator.inject("Moderator", specified_topic)

# Moderator가 제시한 주제를 출력합니다.
print(f"(Moderator): {specified_topic}")
print("\n")

while n < max_iters:  # 최대 반복 횟수까지 반복합니다.
    name, message = (
        simulator.step()
    )  # 시뮬레이터의 다음 단계를 실행하고 발언자와 메시지를 받아옵니다.
    print(f"({name}): {message}")  # 발언자와 메시지를 출력합니다.
    print("\n")
    n += 1  # 반복 횟수를 1 증가시킵니다.
```

```python

```

```