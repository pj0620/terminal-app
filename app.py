import os

from langchain.agents import AgentExecutor, create_structured_chat_agent, create_react_agent, create_json_chat_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.utilities.python import PythonREPL
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate, PromptTemplate
from langchain_core.tools import Tool, tool
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_openai import ChatOpenAI

from utils import load_prompt

def main():
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125")
    # repo_id = "cognitivecomputations/dolphincoder-starcoder2-15b"
    # llm = HuggingFaceEndpoint(
    #     repo_id=repo_id, max_length=128, temperature=0.5, token="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"
    # )

    @tool
    def bash_executor(command: str) -> str:
        """run a bash command on the computer"""
        try:
            os.system(command)
            return "success"
        except Exception as e:
            return "got error: " + str(e)
    
    python_repl = PythonREPL()
    repl_tool = Tool(
        name="python_repl",
        description="A Python shell with root privalges. Use this to execute python commands that can modify the computer since it has root access. Input should be a valid python program. If you want to see the output of a value, you should print it out with `print(...)`.",
        func=python_repl.run,
    )
    
    tools = [repl_tool, bash_executor]
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=load_prompt('system.txt'))),
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['tools', 'input', 'agent_scratchpad'],
                                                         template=load_prompt('human.txt')))
    ])
    
    memory = ConversationBufferWindowMemory(k=10)
    agent = create_json_chat_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, memory=memory
    )
    agent_executor.invoke({"input": "create a python clone of the game pong and directly save it to my Desktop directory. Do not run the code but only use the tools to save the code to my computer return the path to the python program you created after to me"})

with tracing_v2_enabled(project_name="gpt-pet"):
    main()
