from langchain_ollama import ChatOllama
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, RetryPolicy
from langchain_core.messages import HumanMessage, SystemMessage
from speech_to_text import transcribe_audio
from text_to_speech import text_to_speech
from record_audio import record_audio




llm = ChatOllama(model="qwen2.5:3b",temperature=0)

def test_agent():

    while True:

        print("SPEAK:")

        record_audio()

        transcribed_user_audio = transcribe_audio()

        system_prompt = SystemMessage(content="you are a helpful assistant")
        human_prompt = HumanMessage(content=transcribed_user_audio)
        response = llm.invoke([system_prompt, human_prompt])

        text_to_speech(response.content)

        if transcribed_user_audio == "exit":
            
            break

    return response.content


test_agent()