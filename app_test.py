from log_config import setup_logging
from onenote import OneNote
from summarizer import Summarizer

setup_logging()

conversation_file = 'out/2024-09-27_12-33-24-conversation.txt'

summarizer = Summarizer()
html_summary = summarizer.summarize(conversation_file)

onenote = OneNote()

title = conversation_file.split('/')[-1].split('.')[0]

onenote.create_page(title, html_summary)
