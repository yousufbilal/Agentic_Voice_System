import sounddevice as sd
from scipy.io.wavfile import write

def record_audio():
    fs = 44100  # Sample rate
    seconds = 5  # Duration of recording

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  
    write('assets/output.wav', fs, myrecording)  # Save as WAV file

    return myrecording

record_audio()
