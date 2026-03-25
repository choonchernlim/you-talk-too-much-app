import select
import sys
import termios
import tty
from pathlib import Path


def read_file(file_path: str) -> str:
    """Read and return the content of the specified file."""
    return Path(file_path).read_text()


def write_file(file_path: str, text: str) -> None:
    """Write the given text to the specified file."""
    Path(file_path).write_text(text)


def append_file(file_path: str, text: str) -> None:
    """Append the given text to the specified file."""
    with Path(file_path).open("a") as file:
        file.write(text)


def get_key() -> str:
    """Capture and return a single key from the user (blocking)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def poll_key(timeout: float = 0.1) -> str | None:
    """Poll for a single keypress, returning None if no key within timeout."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.read(1)
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
