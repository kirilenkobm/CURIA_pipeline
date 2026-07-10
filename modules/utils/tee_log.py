"""File-descriptor-level tee of stdout/stderr to a log file.

The CURIA pipeline prints from the main process, from a *spawned* GPU executor
process (fresh interpreter, so a Python-level ``sys.stdout`` swap would miss it),
and from subprocesses such as TOGA. To capture all of them we redirect at the
file-descriptor level: fds 1 and 2 are routed through ``os.pipe()``s and a reader
thread tees the bytes to both the real terminal and the log file. Every child that
inherits fds 1/2 is captured automatically.

Each line is flushed to disk as it arrives, so the log stays current even if the
process is killed (the "fflush" behaviour).
"""

import os
import sys
import threading
from pathlib import Path


class TeeLogger:
    """Tee stdout+stderr (including spawned children/subprocesses) to a log file.

    stdout and stderr stay visually separate on the terminal but are merged into a
    single log file (like ``2>&1``). Usable directly (``start()``/``stop()``) or as
    a context manager.
    """

    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self._active = False
        self._log = None
        self._saved_out = None
        self._saved_err = None
        self._r_out = None
        self._r_err = None
        self._threads = []
        self._lock = threading.Lock()

    def start(self):
        if self._active:
            return self
        # Line-buffered so each newline hits disk; flushed per read below regardless.
        self._log = open(
            self.log_path, "w", buffering=1, encoding="utf-8", errors="replace"
        )

        # Save the real terminal fds so we can echo to them and restore later.
        self._saved_out = os.dup(1)
        self._saved_err = os.dup(2)

        # One pipe per stream: preserves stdout/stderr separation on the terminal.
        self._r_out, w_out = os.pipe()
        self._r_err, w_err = os.pipe()
        os.dup2(w_out, 1)
        os.close(w_out)
        os.dup2(w_err, 2)
        os.close(w_err)

        # fd 1 is now a pipe (not a tty); Python would otherwise switch to block
        # buffering and output would lag. Force line buffering to keep it prompt.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(line_buffering=True)
            except Exception:
                pass

        # Make the spawned GPU executor child's output prompt too.
        os.environ.setdefault("PYTHONUNBUFFERED", "1")

        for read_fd, term_fd in (
            (self._r_out, self._saved_out),
            (self._r_err, self._saved_err),
        ):
            t = threading.Thread(target=self._pump, args=(read_fd, term_fd), daemon=True)
            t.start()
            self._threads.append(t)

        self._active = True
        return self

    def _pump(self, read_fd, term_fd):
        while True:
            try:
                data = os.read(read_fd, 65536)
            except OSError:
                break
            if not data:
                break
            # Echo to the real terminal.
            try:
                os.write(term_fd, data)
            except OSError:
                pass
            # Mirror into the log file, flushed immediately.
            with self._lock:
                if self._log is not None and not self._log.closed:
                    self._log.write(data.decode("utf-8", "replace"))
                    self._log.flush()

    def flush(self):
        """Best-effort flush; safe to call from a signal handler."""
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        with self._lock:
            if self._log is not None and not self._log.closed:
                try:
                    self._log.flush()
                except Exception:
                    pass

    def stop(self):
        if not self._active:
            return
        self.flush()

        # Restore the real terminal fds. This drops the parent's write-ends of the
        # pipes; the reader threads get EOF once all children have also exited.
        os.dup2(self._saved_out, 1)
        os.dup2(self._saved_err, 2)

        for t in self._threads:
            t.join(timeout=2)

        for fd in (self._saved_out, self._saved_err, self._r_out, self._r_err):
            try:
                os.close(fd)
            except OSError:
                pass

        with self._lock:
            if self._log is not None and not self._log.closed:
                self._log.close()

        self._active = False

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False
