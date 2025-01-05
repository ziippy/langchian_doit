from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")

import streamlit as st
st.set_page_config(page_title="DBCopilot", page_icon="📊")
st.header('📊 정형 데이터베이스를 위한 DBCopilot입니다.')

from langchain_community.utilities import SQLDatabase
db = SQLDatabase.from_uri('sqlite:///Chinook_Sqlite.sqlite')

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")

from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

#################
prompt_prefix = """ 
##Instructions:
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
As part of your final answer, ALWAYS include an explanation of how to got to the final answer, including the SQL query you run. Include the explanation and the SQL query in the section that starts with "Explanation:".

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don\'t know" as the answer.

##Tools:

"""

prompt_format_instructions = """ 
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question.

Explanation:

<===Beging of an Example of Explanation:

I joined the invoices and customers tables on the customer_id column, which is the common key between them. This will allowed me to access the Total and Country columns from both tables. Then I grouped the records by the country column and calculate the sum of the Total column for each country, ordered them in descending order and limited the SELECT to the top 5.

```sql
SELECT c.country AS Country, SUM(i.total) AS Sales
FROM customer c
JOIN invoice i ON c.customer_id = i.customer_id
GROUP BY Country
ORDER BY Sales DESC
LIMIT 5;
```

===>End of an Example of Explanation
"""
##################

# 에이전트를 초기화
from langchain_community.agent_toolkits import create_sql_agent
agent_executor = create_sql_agent(
    # prefix=prompt_prefix,
    # format_instructions = prompt_format_instructions,
    llm=llm,
    toolkit=toolkit,
    # verbose=True,
    top_k=10,
)

# streamlit 의 세션 상태를 정의하여 대화형 및 메모리 인식 기능을 구현
if "messages" not in st.session_state or st.sidebar.button("대화 내역 지우기"):
    st.session_state["messages"] = [{"role": "assistant", "content": "무엇을 도와드릴까요?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="무엇이든 물어보세요!")

# 사용자가 쿼리를 수행할 때마다 애플리케이션의 로직을 정의
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent_executor.run(user_query, callbacks = [st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)