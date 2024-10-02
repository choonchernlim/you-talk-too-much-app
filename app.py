import os
import time
from threading import Thread, Event

from dotenv import load_dotenv

from app.audio_capturer import AudioCapturer
from app.llm import LLM
from app.log_config import setup_logger
from app.onenote_client import OneNoteClient
from app.transcriber import WhisperTranscriber

load_dotenv()

logger = setup_logger(__name__)

# DONE try-catch both daemon threads and code base so that CTRL+C terminates the program gracefully?
# CAN'T BE DONE logging with line number and file name padded
# DONE parse out txt and replace with md or html in summarizer
# DONE remove unused files
# DONE set up src/ and tests/ directories
# DONE color logging?
# DONE test requirements.txt with new virtual environment

GCP_VERTEX_PROJECT = os.getenv('GCP_VERTEX_PROJECT')
GCP_VERTEX_LOCATION = os.getenv('GCP_VERTEX_LOCATION')
GCP_VERTEX_SA_KEY = os.getenv('GCP_VERTEX_SA_KEY')
GCP_VERTEX_MODEL = os.getenv('GCP_VERTEX_MODEL')
ONENOTE_SECTION_NAME = os.getenv('ONENOTE_SECTION_NAME')
AZURE_CLIENT_ID = os.getenv('AZURE_CLIENT_ID')
AZURE_TENANT_ID = os.getenv('AZURE_TENANT_ID')

transcriber = WhisperTranscriber()
audio_capturer = AudioCapturer(transcriber)
llm = LLM(GCP_VERTEX_PROJECT, GCP_VERTEX_LOCATION, GCP_VERTEX_SA_KEY, GCP_VERTEX_MODEL)
onenote_client = OneNoteClient(ONENOTE_SECTION_NAME, AZURE_CLIENT_ID, AZURE_TENANT_ID)

stop_event = Event()
audio_capture_thread = Thread(target=audio_capturer.capture_audio, args=(stop_event,), daemon=True)
audio_process_thread = Thread(target=audio_capturer.batch_process_buffer, args=(stop_event,), daemon=True)

# Start audio capture in a background thread
audio_capture_thread.start()
audio_process_thread.start()

# Block the main thread until CTRL+C is pressed
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    # Print a newline
    print()

    logger.info('KeyboardInterrupt detected. Broadcasting stop event to all threads...')
    stop_event.set()

logger.info('Waiting for audio capture thread to end...')
audio_capture_thread.join()

logger.info('Waiting for audio process thread to end...')
audio_process_thread.join()

# Only create OneNote page if conversation file exists
if os.path.exists(transcriber.get_conversation_file_path()):
    onenote_client.create_page(
        title=transcriber.get_formatted_datetime(),
        html_summary=llm.summarize(transcriber.get_conversation_file_path()),
    )

logger.info('Done!')
