"""Terminal-only tqdm bars, including when stdout/stderr are piped through tee."""
import os
import sys
from contextlib import contextmanager

from tqdm import tqdm


@contextmanager
def progress_bar(iterable, **kwargs):
    stream = sys.stderr
    owned_stream = None
    # The launcher pipes both outputs into tee; use its controlling terminal
    # for transient bars so running.log only contains ordinary log records.
    if not stream.isatty() and os.environ.get('P2R_PROGRESS', '1') != '0':
        try:
            owned_stream = open('/dev/tty', 'w', buffering=1)
            stream = owned_stream
        except OSError:
            pass
    try:
        with tqdm(iterable, file=stream, dynamic_ncols=True, mininterval=0.5,
                  leave=False, disable=os.environ.get('P2R_PROGRESS', '1') == '0'
                  or not stream.isatty(), **kwargs) as bar:
            yield bar
    finally:
        if owned_stream is not None:
            owned_stream.close()
