import os
import sys
import time
from threading import Thread, Event
from typing import Optional

from dotenv import load_dotenv

from .audio_capturer import AudioCapturer
from .llm import LLM
from .log_config import setup_logger
from .onenote_client import OneNoteClient
from .transcriber import WhisperTranscriber
from .utils import get_key

load_dotenv()

logger = setup_logger(__name__)

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

stop_event: Optional[Event] = None
audio_capture_thread: Optional[Thread] = None
audio_process_thread: Optional[Thread] = None


def start_capture():
    global stop_event, audio_capture_thread, audio_process_thread

    if audio_capture_thread is not None or audio_process_thread is not None:
        return

    logger.info('Starting new capture...')

    stop_event = Event()
    audio_capture_thread = Thread(target=audio_capturer.capture_audio, args=(stop_event,), daemon=True)
    audio_process_thread = Thread(target=audio_capturer.batch_process_buffer, args=(stop_event,), daemon=True)

    # Start audio capture in a background thread
    audio_capture_thread.start()
    audio_process_thread.start()

    time.sleep(2)
    logger.info('Listening...')


def stop_capture():
    global stop_event, audio_capture_thread, audio_process_thread

    if audio_capture_thread is None or audio_process_thread is None:
        return

    # Print a newline
    print()

    logger.info('Stopping existing capture... Broadcasting stop event to all threads...')
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

    stop_event = None
    audio_capture_thread = None
    audio_process_thread = None
    logger.info('Stopped.')


def display_menu():
    logger.info('Press the following key:')
    logger.info("1) Start new capture")
    logger.info("2) Stop existing capture")
    logger.info("3) Quit program")


def run():
    display_menu()

    is_capture_started = False

    while True:
        key = get_key()
        if key == '1' and not is_capture_started:
            start_capture()
            is_capture_started = True
        elif key == '2' and is_capture_started:
            stop_capture()
            display_menu()
            is_capture_started = False
        elif key == '3':
            stop_capture()
            logger.info('Quitting the program...')
            break

    logger.info('Done!')


if __name__ == '__main__':
    run()
