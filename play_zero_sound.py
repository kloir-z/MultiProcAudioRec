import numpy as np
import pyaudio

def play_zero_sound(sampling_rate, ready_event):
    print("Playing zero sound...")
    zero_sound = np.zeros(int(sampling_rate), dtype=np.float32)
    pya = pyaudio.PyAudio()
    stream = pya.open(format=pyaudio.paFloat32, channels=1, rate=sampling_rate, output=True)
    stream.write(zero_sound.tobytes())
    ready_event.set()
    try:
        while True:
            stream.write(zero_sound.tobytes())
    except KeyboardInterrupt:
        pass
