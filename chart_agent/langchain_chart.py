from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

from langchain_chart_tool import chart_tool

import json

# LLM과 Prompt 설정
# llm = OpenAI(api_key="your_openai_api_key")
llm = OpenAI()
prompt = PromptTemplate(input_variables=["data"], template="다음 데이터를 기반으로 차트를 생성하세요: {data}")

# Tool 등록
tools = [
    Tool(name="Chart Generator", func=chart_tool, description="데이터를 입력받아 차트를 생성합니다."),
]

# Agent 생성
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# 데이터 입력 및 실행
input_data = {'months': ['January', 'February', 'March', 'April'], 'costs': [100.0, 120.0, 150.0, 130.0]}
json_input_data = json.dumps(input_data)
response = agent.run(f"다음 데이터를 이용해 차트를 생성하세요: {json_input_data}")
print(response)
