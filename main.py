from langchain_ollama import ChatOllama
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, RetryPolicy
from langchain_core.messages import HumanMessage, SystemMessage
from my_agent.utils.speech_to_text import transcribe_audio
from my_agent.utils.text_to_speech import text_to_speech
from my_agent.utils.record_audio import record_audio
import webrtcvad

vad = webrtcvad.Vad()
vad.set_mode(1)

llm = ChatOllama(model="qwen2.5:3b",temperature=0)

def test_agent():

    while True:

        print("SPEAK:")

        record_audio()

        transcribed_user_audio = transcribe_audio()

        if transcribed_user_audio.strip().lower().rstrip(".") == "exit":
            break

        system_prompt = SystemMessage(content=" 3 word or less answer only")
        human_prompt = HumanMessage(content=transcribed_user_audio)

        response = llm.invoke([system_prompt, human_prompt])

        text_to_speech(response.content)




test_agent()