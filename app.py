import json
import os

import pyaudio
import vosk
from dotenv import load_dotenv

from conversation import conversation_parser

# https://medium.com/@nimritakoul01/offline-speech-to-text-in-python-f5d6454ecd02

# Install portaudio before running pip install
# https://stackoverflow.com/questions/33513522/when-installing-pyaudio-pip-cannot-find-portaudio-h-in-usr-local-include

load_dotenv()

# Here I have downloaded this model to my PC, extracted the files
# and saved it in local directory
# Set the model path
model_path = os.getenv("VOSK_MODEL_PATH")

# # Initialize the model with model-path
model = vosk.Model(model_path=model_path)

# if you don't want to download the model, just mention "lang" argument
# in vosk.Model() and it will download the right  model, here the language is
# US-English
# model = vosk.Model(lang="en-us")

SAMPLE_RATE = 16000
FRAMES_PER_BUFFER = 8192

# Create a recognizer
rec = vosk.KaldiRecognizer(model, SAMPLE_RATE)
rec.SetWords(True)

p = pyaudio.PyAudio()

# Open the microphone stream
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=FRAMES_PER_BUFFER,
)


def run():
    print("=====================================================")
    print("Listening for speech. Press CTRL+C to stop.")
    print("=====================================================")

    try:
        while True:
            if stream.get_read_available():
                data = stream.read(FRAMES_PER_BUFFER)
                # accept waveform of input voice
                if rec.AcceptWaveform(data):
                    # Parse the JSON result and get the recognized text
                    words = json.loads(rec.Result()).get("result", [])

                    conversation_parser(words)
            else:
                conversation_parser()
                # check_conversation_buffer()



    # capture CTRL+C to terminate
    except KeyboardInterrupt:
        conversation_parser()
        # check_conversation_buffer()

        print("=====================================================")
        print("Termination keyword detected. Stopping...")
        print("=====================================================")

        # Stop and close the stream
        stream.stop_stream()
        stream.close()

        # Terminate the PyAudio object
        p.terminate()


if __name__ == "__main__":
    run()
