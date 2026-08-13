from langchain_ollama import ChatOllama
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, RetryPolicy
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOllama(model="qwen2.5:3b",temperature=0)

def test_agent():

    system_prompt = SystemMessage(content="hello tell me about goku")
    human_prompt = HumanMessage(content="hello tell me about goku")
    response = llm.invoke([system_prompt, human_prompt])
    print(response.content)

    return response.content

test_agent()