import sys
import tty

import termios


def read_file(file_path) -> str:
    with open(file_path, 'r') as file:
        text = file.read()

    return text


def write_file(file_path, text):
    with open(file_path, 'w') as file:
        file.write(text)


def append_file(file_path, text):
    with open(file_path, 'a') as file:
        file.write(text)


def get_key():
    """
    Captures and return a single key from the user.
    This approach is used instead of input() because input() requires
    the user to press Enter.

    :return: A single key from the user
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch
