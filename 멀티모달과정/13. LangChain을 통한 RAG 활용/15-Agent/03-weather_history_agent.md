```python
# !pip install -q wikipedia-api==0.7.1
```

```python
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()
```

```python
import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain imports
from langchain.tools import tool, BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_teddynote.messages import AgentCallbacks, AgentStreamParser
```

```python
# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install -qU langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH15-Agents")
```

```python
# Import custom tools
from tools.weather_tool import WeatherTool
from tools.wikipedia_tool import WikipediaTool
```

```python
class WeatherHistoryAgent:
    """Weather and History Agent that combines weather and Wikipedia information."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        """Initialize the Weather History Agent.

        Args:
            model_name: OpenAI model to use
            temperature: Temperature for the model (0-1)
        """

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        # 1. Define custom tools
        self.tools = self._initialize_custom_tools()

        # 2. Create agent and executor
        self.agent = self._create_agent()
        self.agent_executor = self._create_agent_executor()

        # 3. Initialize callbacks for streaming
        self.stream_parser = self._setup_stream_callbacks()

        # 4. Setup chat history
        self.chat_history = self._setup_chat_history()

    def _initialize_custom_tools(self) -> List[BaseTool]:
        """Initialize and return the custom tools for the agent."""

        # Custom tool definitions using @tool decorator
        @tool
        def get_current_weather(city: str) -> str:
            """Get current weather information for a specific city using SerpAPI.
            Input should be a city name (e.g., 'Seoul', 'New York', 'Paris').
            Returns temperature, weather conditions, humidity, wind speed, and more.
            """
            weather_tool = WeatherTool()
            return weather_tool._run(city)

        @tool
        def search_wikipedia_info(query: str) -> str:
            """Search Wikipedia for information about a topic.
            Input should be a search query (e.g., 'Eiffel Tower', 'Korean cuisine', 'Tokyo history').
            Returns a summary of the Wikipedia article if found.
            """
            wiki_tool = WikipediaTool()
            return wiki_tool._run(query)

        return [get_current_weather, search_wikipedia_info]

    def _create_agent_prompt(self) -> ChatPromptTemplate:
        """Create the agent prompt template."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful weather and history assistant with access to real-time weather data and Wikipedia information.

You have access to the following tools:
- get_current_weather: Get current weather information for any city
- search_wikipedia_info: Search Wikipedia for historical, cultural, or general information

IMPORTANT LANGUAGE RULES:
- If the user asks in Korean, respond in Korean
- If the user asks in English, respond in English  
- Always match the language of the user's question

TOOL USAGE GUIDELINES:
- For weather queries: Use get_current_weather with the city name
- For historical/cultural/general information: Use search_wikipedia_info with relevant search terms
- You can use multiple tools if the question requires both weather and historical information
- Always provide comprehensive answers by combining information from multiple sources when relevant

RESPONSE GUIDELINES:
- Provide detailed, helpful responses
- Include specific data when available (temperatures, dates, statistics)
- Organize information clearly with bullet points or sections when appropriate
- Always respond in the same language as the user's question
""",
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        return prompt

    def _create_agent(self):
        """Create the tool calling agent."""
        prompt = self._create_agent_prompt()

        agent = create_tool_calling_agent(llm=self.llm, tools=self.tools, prompt=prompt)

        return agent

    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor with tools and configuration."""
        return AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=False,  # Set to False for clean output with custom callbacks
            max_iterations=10,
            max_execution_time=30,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

    def _setup_stream_callbacks(self):
        """Setup custom callbacks for stream output visualization."""

        def tool_callback(tool_info: Dict[str, Any]) -> None:
            """Callback function executed when a tool is called."""
            print("\n" + "🔧" + "=" * 50)
            print(f"🛠️  TOOL CALL: {tool_info.get('tool', 'Unknown')}")
            print(f"📝 INPUT: {tool_info.get('tool_input', 'N/A')}")
            print("🔧" + "=" * 50)

        def observation_callback(observation_info: Dict[str, Any]) -> None:
            """Callback function executed when an observation is received."""
            print("\n" + "👁️" + "=" * 50)
            print("👀 OBSERVATION:")
            observation_text = (
                observation_info.get("observation", [""])[0]
                if observation_info.get("observation")
                else "No observation"
            )
            print(f"{observation_text}")
            print("👁️" + "=" * 50)

        def result_callback(result: str) -> None:
            """Callback function executed when the final result is ready."""
            print("\n" + "✅" + "=" * 50)
            print("🎯 FINAL ANSWER:")
            print(result)
            print("✅" + "=" * 50)

        # Use teddynote callbacks if available
        agent_callbacks = AgentCallbacks(
            tool_callback=tool_callback,
            observation_callback=observation_callback,
            result_callback=result_callback,
        )
        return AgentStreamParser(agent_callbacks)

    def _setup_chat_history(self):
        """Setup chat history for conversation memory."""
        # Session storage
        self.store = {}

        def get_session_history(session_id: str):
            if session_id not in self.store:
                self.store[session_id] = ChatMessageHistory()
            return self.store[session_id]

        # Create agent with chat history
        agent_with_history = RunnableWithMessageHistory(
            self.agent_executor,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        return agent_with_history

    def run_with_stream(self, query: str, session_id: str = "default") -> None:
        """Run the agent with streaming output and step visualization."""
        print(f"\n🚀 질문을 처리하고 있습니다: '{query}'")
        print("=" * 80)

        try:
            # Stream the agent's response
            response_stream = self.chat_history.stream(
                {"input": query}, config={"configurable": {"session_id": session_id}}
            )

            # Process each step in the stream
            for step in response_stream:
                self.stream_parser.process_agent_steps(step)

        except Exception as e:
            print(f"\n❌ 에이전트 실행 중 오류가 발생했습니다: {str(e)}")

    def run_simple(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """Run the agent without streaming (simple invoke)."""
        try:
            result = self.chat_history.invoke(
                {"input": query}, config={"configurable": {"session_id": session_id}}
            )
            return {
                "success": True,
                "answer": result.get("output", ""),
                "intermediate_steps": result.get("intermediate_steps", []),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "answer": f"Error occurred: {str(e)}",
            }

    def _process_stream_step(self, step: Dict[str, Any]) -> None:
        """Fallback method to process stream steps when teddynote is not available."""
        if "actions" in step:
            # Tool call step
            for action in step["actions"]:
                print(f"\n🔧 TOOL CALL: {action.tool}")
                print(f"📝 INPUT: {action.tool_input}")

        elif "steps" in step:
            # Observation step
            for step_item in step["steps"]:
                print(f"\n👀 OBSERVATION: {step_item.observation}")

        elif "output" in step:
            # Final answer step
            print(f"\n✅ FINAL ANSWER: {step['output']}")
```

```python
class SimpleStreamParser:
    """Simple fallback stream parser when langchain_teddynote is not available."""

    def __init__(self, tool_callback, observation_callback, result_callback):
        self.tool_callback = tool_callback
        self.observation_callback = observation_callback
        self.result_callback = result_callback

    def process_agent_steps(self, step: Dict[str, Any]) -> None:
        """Process agent steps and call appropriate callbacks."""
        if "actions" in step:
            # Tool call
            for action in step["actions"]:
                self.tool_callback(
                    {"tool": action.tool, "tool_input": action.tool_input}
                )

        elif "steps" in step:
            # Observation
            observations = []
            for step_item in step["steps"]:
                observations.append(step_item.observation)
            self.observation_callback({"observation": observations})

        elif "output" in step:
            # Final result
            self.result_callback(step["output"])
```

```python
def main():
    """Main function to run the Weather History Agent."""
    load_dotenv()
    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file and add your OpenAI API key")
        return

    if not os.getenv("SERPAPI_API_KEY"):
        print("⚠️  Warning: SERPAPI_API_KEY not found")
        print("Weather functionality will be limited")

    print(
        """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     🌤️  날씨 & 역사 정보 에이전트 🏛️                         ║
║                                                                              ║
║  이 에이전트는 날씨 정보와 위키피디아 검색 기능을 결합합니다                     ║
║  - 실시간 도시별 날씨 정보                                                    ║
║  - 위키피디아를 통한 역사 및 문화 정보                                         ║
║  - 다국어 지원 (한국어/영어)                                                  ║
║  - 단계별 시각화된 스트림 출력                                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    )

    # Initialize the agent
    try:
        print("🚀 날씨 & 역사 정보 에이전트를 초기화하고 있습니다...")
        agent = WeatherHistoryAgent(model_name="gpt-4o-mini", temperature=0.3)
        print("✅ 에이전트가 성공적으로 초기화되었습니다!")
    except Exception as e:
        print(f"❌ 에이전트 초기화에 실패했습니다: {str(e)}")
        return

    # Example queries
    print("\n💡 예시 질문:")
    print("  - '서울 날씨와 서울의 역사에 대해 알려줘'")
    print("  - '도쿄 날씨와 유명한 랜드마크들을 알려줘'")
    print("  - '파리 날씨와 에펠탑에 대해 알려줘'")
    print("  - 'What's the weather in Seoul and tell me about Korean history?'")
    print("\n종료하려면 'exit' 또는 'quit'을 입력하세요")
    print("=" * 80)

    # Interactive mode
    session_id = "user_session"

    while True:
        try:
            # Get user input
            user_query = input("\n🤔 질문을 입력하세요: ").strip()

            # Check for exit commands
            if user_query.lower() in ["exit", "quit", "q"]:
                print(
                    "\n👋 날씨 & 역사 정보 에이전트를 사용해 주셔서 감사합니다! 안녕히 가세요!"
                )
                break

            # Skip empty inputs
            if not user_query:
                print("⚠️  질문을 입력해 주세요.")
                continue

            # Process query with streaming
            agent.run_with_stream(user_query, session_id)

            print("\n" + "=" * 80)

        except KeyboardInterrupt:
            print(
                "\n\n👋 중단되었습니다! 날씨 & 역사 정보 에이전트를 사용해 주셔서 감사합니다!"
            )
            break
        except EOFError:
            print("\n\n👋 날씨 & 역사 정보 에이전트를 사용해 주셔서 감사합니다!")
            break
        except Exception as e:
            print(f"\n❌ 오류가 발생했습니다: {str(e)}")
            print("다시 시도하거나 'exit'을 입력하여 종료하세요.")
```

```python
main()
```