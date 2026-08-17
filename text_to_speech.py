from piper import PiperVoice
import wave
from playsound import playsound


def text_to_speech (agent_response):
    voice = PiperVoice.load("en_US-lessac-medium.onnx")
    with wave.open("Agent.wav", "wb") as wav_file:
        voice.synthesize_wav(agent_response, wav_file)
        playsound("Agent.wav")
