from langchain_ollama import ChatOllama
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, RetryPolicy
from langchain_core.messages import HumanMessage, SystemMessage
from transcribe import transcribe_audio
import wave
from piper import PiperVoice


llm = ChatOllama(model="qwen2.5:3b",temperature=0)

def test_agent():

    transcribed_user_audio = transcribe_audio()

    system_prompt = SystemMessage(content="you are an anime expert")
    human_prompt = HumanMessage(content=transcribed_user_audio)
    response = llm.invoke([system_prompt, human_prompt])
    print(response.content)

    agent_voice_output(response.content)

    return response.content

def agent_voice_output(agent_response):
    voice = PiperVoice.load("en_US-lessac-medium.onnx")
    with wave.open("Agent.wav", "wb") as wav_file:
        voice.synthesize_wav(agent_response, wav_file)

test_agent()