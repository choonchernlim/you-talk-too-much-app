import json
import re
import time
from datetime import datetime
from typing import Optional, List
import nltk

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

comma_threshold_in_seconds = 0.4
period_threshold_in_seconds = 1.0
last_word_end_time = 1.0
sentence_buffer = ""
last_timestamp = datetime.now().timestamp()
datetime_suffix = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
conversation_file_path = f"out/conversation-{datetime_suffix}.txt"
raw_file_path = f"out/raw-{datetime_suffix}.txt"


def truecase(text):
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

    with open(conversation_file_path, "a") as file:
        file.write(pretty_string + "\n")


def conversation_parser(words:Optional[List[dict]]=None):
    global last_word_end_time, sentence_buffer, last_timestamp

    # print(f"{datetime.now()} [PARSER] ========= {words}")

    # if not words:
    #     return
    # no more words to process
    if words is None or not words:
        # if the buffer is not empty and the duration has exceeded the threshold, empty the buffer
        if sentence_buffer and datetime.now().timestamp() - last_timestamp > period_threshold_in_seconds:
            # print("[EMPTY BUFFER] ======================================================")
            truecase(sentence_buffer + ".")
            sentence_buffer = ""

        return

    # print("[NEW STREAM] ======================================================")

    with open(raw_file_path, "a") as file:
        file.write(json.dumps(words) + "\n")

    for idx, word in enumerate(words):
        pause_duration = word['start'] - last_word_end_time

        if sentence_buffer:
            if pause_duration > period_threshold_in_seconds:
                truecase(sentence_buffer + ".")
                sentence_buffer = ""
            elif pause_duration > comma_threshold_in_seconds:
                sentence_buffer += ", "

        sentence_buffer += word['word'] + " "
        last_word_end_time = word['end']

        # print(f"{pause_duration:10.2f}", f"{word['start']:10.2f}", f"{word['end']:10.2f}", word['word'])

    last_timestamp = datetime.now().timestamp()


def check_conversation_buffer():
    global sentence_buffer

    if not sentence_buffer:
        return

    if datetime.now().timestamp() - last_timestamp > period_threshold_in_seconds:
        # print("[EMPTY BUFFER] ======================================================")
        truecase(sentence_buffer + ".")
        sentence_buffer = ""


def run():
    with open("voice.txt", "r") as file:
        lines = file.readlines()
        for stream_data in lines:
            conversation_parser(json.loads(stream_data))

    # ensure it's past the period threshold to empty the conversation buffer
    time.sleep(period_threshold_in_seconds + 1)

    conversation_parser()


if __name__ == "__main__":
    run()
