from faster_whisper import WhisperModel
import wave
from pydub import AudioSegment


def transcribe_audio():
    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    print(" Transcribing...")
    segments, _ = model.transcribe("output.wav", language="en")

    for segment in segments:
        # print("SEGMETN",segment.text)
        print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text.strip()}")
    print(" Done.\n")


if __name__ == "__main__":
    # transcribe_audio("audio")
    transcribe_audio()
