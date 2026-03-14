def read_file(file_path: str) -> str:
    """Read and return the content of the specified file."""
    with open(file_path) as file:
        text = file.read()
    return text


def write_file(file_path: str, text: str) -> None:
    """Write the given text to the specified file."""
    with open(file_path, "w") as file:
        file.write(text)


def append_file(file_path: str, text: str) -> None:
    """Append the given text to the specified file."""
    with open(file_path, "a") as file:
        file.write(text)


def get_key() -> str:
    """Captures and return a single key from the user."""
    import sys
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch
