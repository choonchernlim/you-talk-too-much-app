import os
from collections.abc import Generator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path


@contextmanager
def suppress_output() -> Generator[None, None, None]:
    """Suppress stdout and stderr, e.g. during noisy model loading."""
    with (
        Path(os.devnull).open("w") as devnull,
        redirect_stdout(devnull),
        redirect_stderr(devnull),
    ):
        yield
