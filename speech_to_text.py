from faster_whisper import WhisperModel
import wave
from pydub import AudioSegment

model = WhisperModel("small.en", device="cpu", compute_type="int8")

def transcribe_audio():
    print(" Transcribing...")
    segments, _ = model.transcribe("output.wav", language="en")

    for segment in segments:
        # print("SEGMETN",segment.text)
        human_speech_text = segment.text

        print("THE HUMAN SPEECH : ",human_speech_text )
        # print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text.strip()}")
        # print(" Done.\n")

    return human_speech_text


if __name__ == "__main__":
    # transcribe_audio("audio")
    transcribe_audio()
