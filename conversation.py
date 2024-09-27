import json
import logging
import re
from datetime import datetime
from typing import Optional, List

import nltk

from utils import append_file

# Create a logger for this module
logger = logging.getLogger(__name__)

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


class VoskConversationParser:
    COMMA_THRESHOLD_IN_SECONDS = 0.4
    PERIOD_THRESHOLD_IN_SECONDS = 1.0

    def __init__(self, conversation_file_path, raw_file_path):
        self.conversation_file_path = conversation_file_path
        self.raw_file_path = raw_file_path
        self.last_word_end_time = 1.0  # something greater than 0
        self.sentence_buffer = ""
        self.last_timestamp = datetime.now().timestamp()

    def truecase(self, text):
        if not text:
            return

        tagged_sent = nltk.pos_tag(nltk.tokenize.TweetTokenizer().tokenize(text))

        normalized_sent = [
            word.capitalize() if tag in {"NNP", "NNPS"} else word
            for (word, tag) in tagged_sent
        ]

        # Capitalize first word in sentence.
        normalized_sent[0] = normalized_sent[0].capitalize()

        # Use regular expression to get punctuation right.
        pretty_string = re.sub(
            " (?=[.,'!?:;])",
            "",
            " ".join(normalized_sent)
        )

        # return pretty_string
        print(pretty_string)

        append_file(self.conversation_file_path, pretty_string + "\n")

    def conversation_parser(self, words: Optional[List[dict]] = None):

        # print(f"{datetime.now()} [PARSER] ========= {words}")

        # if not words:
        #     return
        # no more words to process
        if words is None or not words:
            # if the buffer is not empty and the duration has exceeded the threshold, empty the buffer
            if self.sentence_buffer and datetime.now().timestamp() - self.last_timestamp > self.PERIOD_THRESHOLD_IN_SECONDS:
                # print("[EMPTY BUFFER] ======================================================")
                self.truecase(self.sentence_buffer + ".")
                self.sentence_buffer = ""

            return

        # print("[NEW STREAM] ======================================================")

        append_file(self.raw_file_path, json.dumps(words) + "\n")

        for idx, word in enumerate(words):
            pause_duration = word['start'] - self.last_word_end_time

            if self.sentence_buffer:
                if pause_duration > self.PERIOD_THRESHOLD_IN_SECONDS:
                    self.truecase(self.sentence_buffer + ".")
                    self.sentence_buffer = ""
                elif pause_duration > self.COMMA_THRESHOLD_IN_SECONDS:
                    self.sentence_buffer += ", "

            self.sentence_buffer += word['word'] + " "
            self.last_word_end_time = word['end']

            # print(f"{pause_duration:10.2f}", f"{word['start']:10.2f}", f"{word['end']:10.2f}", word['word'])

        self.last_timestamp = datetime.now().timestamp()

    def check_conversation_buffer(self):
        if not self.sentence_buffer:
            return

        if datetime.now().timestamp() - self.last_timestamp > self.PERIOD_THRESHOLD_IN_SECONDS:
            # print("[EMPTY BUFFER] ======================================================")
            self.truecase(self.sentence_buffer + ".")
            self.sentence_buffer = ""

# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
#
# COMMA_THRESHOLD_IN_SECONDS = 0.4
# PERIOD_THRESHOLD_IN_SECONDS = 1.0
# last_word_end_time = 1.0
# sentence_buffer = ""
# last_timestamp = datetime.now().timestamp()
# datetime_suffix = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# conversation_file_path = f"out/conversation-{datetime_suffix}.txt"
# raw_file_path = f"out/raw-{datetime_suffix}.txt"
#
# def truecase(text):
#     if not text:
#         return
#
#     tagged_sent = nltk.pos_tag(nltk.tokenize.TweetTokenizer().tokenize(text))
#
#     normalized_sent = [
#         word.capitalize() if tag in {"NNP", "NNPS"} else word
#         for (word, tag) in tagged_sent
#     ]
#
#     # Capitalize first word in sentence.
#     normalized_sent[0] = normalized_sent[0].capitalize()
#
#     # Use regular expression to get punctuation right.
#     pretty_string = re.sub(
#         " (?=[.,'!?:;])",
#         "",
#         " ".join(normalized_sent)
#     )
#
#     # return pretty_string
#     print(pretty_string)
#
#     append_file(conversation_file_path, pretty_string + "\n")
#
#
# def conversation_parser(words: Optional[List[dict]] = None):
#     global last_word_end_time, sentence_buffer, last_timestamp
#
#     # print(f"{datetime.now()} [PARSER] ========= {words}")
#
#     # if not words:
#     #     return
#     # no more words to process
#     if words is None or not words:
#         # if the buffer is not empty and the duration has exceeded the threshold, empty the buffer
#         if sentence_buffer and datetime.now().timestamp() - last_timestamp > PERIOD_THRESHOLD_IN_SECONDS:
#             # print("[EMPTY BUFFER] ======================================================")
#             truecase(sentence_buffer + ".")
#             sentence_buffer = ""
#
#         return
#
#     # print("[NEW STREAM] ======================================================")
#
#     append_file(raw_file_path, json.dumps(words) + "\n")
#
#     for idx, word in enumerate(words):
#         pause_duration = word['start'] - last_word_end_time
#
#         if sentence_buffer:
#             if pause_duration > PERIOD_THRESHOLD_IN_SECONDS:
#                 truecase(sentence_buffer + ".")
#                 sentence_buffer = ""
#             elif pause_duration > COMMA_THRESHOLD_IN_SECONDS:
#                 sentence_buffer += ", "
#
#         sentence_buffer += word['word'] + " "
#         last_word_end_time = word['end']
#
#         # print(f"{pause_duration:10.2f}", f"{word['start']:10.2f}", f"{word['end']:10.2f}", word['word'])
#
#     last_timestamp = datetime.now().timestamp()
#
#
# def check_conversation_buffer():
#     global sentence_buffer
#
#     if not sentence_buffer:
#         return
#
#     if datetime.now().timestamp() - last_timestamp > PERIOD_THRESHOLD_IN_SECONDS:
#         # print("[EMPTY BUFFER] ======================================================")
#         truecase(sentence_buffer + ".")
#         sentence_buffer = ""
