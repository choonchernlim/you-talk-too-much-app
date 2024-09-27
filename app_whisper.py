import numpy as np
import pyaudio
import whisper

# Load Whisper model
model = whisper.load_model("large-v3")  # base

# PyAudio configuration
RATE = 16000  # Sampling rate (16 kHz)
CHUNK = 8196

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open audio stream
stream = audio.open(
    format=pyaudio.paInt16,  # Audio format (16-bit int)
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
)

print("Listening...")

try:
    while True:
        if stream.get_read_available():
            frame = stream.read(CHUNK, exception_on_overflow=False)

            audio_data = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0

            # Process audio with Whisper model
            result = model.transcribe(audio_data, language="en", fp16=False)

            # Print the transcribed text
            print(result["text"])

except KeyboardInterrupt:
    # Stop the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    print("Stopped listening.")
