"""Shared terminal output helpers for every AURORA command-line program."""

from __future__ import annotations

import atexit
import builtins
import getpass
import os
import sys
import threading
from collections.abc import Iterable
from typing import Any, TextIO, TypeVar


_T = TypeVar("_T")
PEAK_MEMORY_SAMPLE_INTERVAL_SECONDS = 0.5
CONSOLE_RULE_WIDTH = 60
PROGRESS_MIN_INTERVAL_SECONDS = 0.5


class _PeakMemoryTracker:
    """Track peak process RSS in a background thread.

    The operating system is sampled every half second. ``resource.ru_maxrss``
    is already cumulative from process start; the thread provides a portable
    fallback and leaves room for additional metrics.
    """

    def __init__(self) -> None:
        self._peak_kb: int = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="aurora-mem-tracker"
        )
        self._thread.start()

    def _sample(self) -> None:
        try:
            import resource

            kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                kb //= 1024  # macOS reports bytes; Linux reports KiB
            with self._lock:
                if kb > self._peak_kb:
                    self._peak_kb = kb
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.wait(PEAK_MEMORY_SAMPLE_INTERVAL_SECONDS):
            self._sample()

    def stop(self) -> None:
        """Stop the tracker and collect one final sample."""
        self._stop.set()
        self._sample()

    def peak_mib(self) -> float:
        with self._lock:
            return self._peak_kb / 1024


class _PassthroughProgress(Iterable[_T]):
    """Iterable-compatible fallback with the small tqdm API AURORA uses."""

    def __init__(self, iterable: Iterable[_T]) -> None:
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable)

    def close(self) -> None:
        """Match ``tqdm.close`` when progress rendering is unavailable."""


class AuroraConsole:
    """Render AURORA messages and progress bars with one terminal style."""

    rule_width = CONSOLE_RULE_WIDTH

    def __init__(self) -> None:
        self._mem = _PeakMemoryTracker()
        atexit.register(self._report_peak)

    def _report_peak(self) -> None:
        """Report peak RSS when the program exits."""
        self._mem.stop()
        mib = self._mem.peak_mib()
        if mib <= 0:
            return
        builtins.print(file=sys.stdout)
        self.detail(f"Peak RSS: {mib:.1f} MiB  ({mib / 1024:.2f} GiB)")
    _rule_characters = frozenset("-=─━")
    _legacy_prefixes = (
        ("✓", "✓"),
        ("→", "→"),
        ("!", "!"),
        ("✗", "✗"),
        ("•", "•"),
    )

    @staticmethod
    def _stream_supports_progress(stream: TextIO) -> bool:
        """Return whether progress animation is appropriate for a stream."""
        try:
            return stream.isatty()
        except (AttributeError, OSError):
            return False

    @classmethod
    def _is_rule(cls, text: str) -> bool:
        stripped = text.strip()
        return bool(stripped) and set(stripped) <= cls._rule_characters

    @staticmethod
    def _completion_title(text: str) -> str | None:
        stripped = text.strip()
        if not (stripped.startswith("===") and stripped.endswith("===")):
            return None
        title = stripped.strip("= ").strip()
        if not title.upper().startswith("AURORA "):
            return None
        return title[7:].strip()

    @staticmethod
    def _section_title(text: str) -> str | None:
        stripped = text.strip()
        if not stripped.upper().startswith("[AURORA]"):
            return None
        return stripped[len("[AURORA]") :].strip()

    @classmethod
    def _status_line(cls, text: str, *, error_stream: bool) -> str | None:
        stripped = text.strip()
        for old_prefix, marker in cls._legacy_prefixes:
            if stripped.startswith(old_prefix):
                message = stripped[len(old_prefix) :].lstrip()
                message = message.replace("\n", "\n    │ ")
                if error_stream:
                    marker = "✗"
                return f"  {marker} {message}"
        if "\n" not in text and text.startswith("    ") and stripped:
            return f"    • {stripped}"
        if error_stream:
            return f"  ✗ {stripped}"
        return None

    def rule(self, *, file: TextIO | None = None) -> None:
        """Write the canonical AURORA section rule."""
        builtins.print("─" * self.rule_width, file=file or sys.stdout)

    def section(self, title: str, *, file: TextIO | None = None) -> None:
        """Start a visibly separated AURORA output section."""
        stream = file or sys.stdout
        builtins.print(f"[AURORA] {title}", file=stream)
        self.rule(file=stream)

    def info(self, message: Any, *, file: TextIO | None = None) -> None:
        builtins.print(f"  → {message}", file=file or sys.stdout)

    def success(self, message: Any, *, file: TextIO | None = None) -> None:
        builtins.print(f"  ✓ {message}", file=file or sys.stdout)

    def warning(self, message: Any, *, file: TextIO | None = None) -> None:
        builtins.print(f"  ! {message}", file=file or sys.stdout)

    def error(self, message: Any, *, file: TextIO | None = None) -> None:
        builtins.print(f"  ✗ {message}", file=file or sys.stderr)

    def detail(self, message: Any, *, file: TextIO | None = None) -> None:
        builtins.print(f"    • {message}", file=file or sys.stdout)

    def prompt(self, message: str, *, secret: bool = False) -> str:
        """Read one value using the canonical AURORA prompt marker."""
        message = message.strip()
        suffix = "" if message.endswith((":", "?", "]")) else ":"
        prompt_text = f"  ? {message}{suffix} "
        if secret:
            return getpass.getpass(prompt_text)
        return builtins.input(prompt_text)

    def confirm(self, message: str, *, default: bool = False) -> bool:
        """Ask one yes/no question using a documented, consistent prompt."""
        choice_hint = "[Y/n]" if default else "[y/N]"
        response = self.prompt(f"{message.rstrip('?')}? {choice_hint}").strip()
        if not response:
            return default
        return response.lower() in {"y", "yes"}

    def complete(self, title: str, *, file: TextIO | None = None) -> None:
        """Write a canonical completion section."""
        stream = file or sys.stdout
        builtins.print(file=stream)
        self.section(f"{title} — complete", file=stream)

    def render_start(
        self,
        render_type: str,
        title: str,
        *,
        details: Iterable[Any] = (),
        file: TextIO | None = None,
    ) -> None:
        """Start a map or video renderer with the shared diagnostic layout."""
        normalized = render_type.strip().lower()
        if normalized not in {"map", "video"}:
            raise ValueError("render_type must be 'map' or 'video'")
        stream = file or sys.stdout
        self.section(f"{normalized.upper()} RENDER — {title}", file=stream)
        for detail in details:
            self.detail(detail, file=stream)

    def render_complete(
        self,
        render_type: str,
        title: str,
        *,
        output: Any | None = None,
        elapsed_seconds: float | None = None,
        file: TextIO | None = None,
    ) -> None:
        """Finish a map or video renderer with matching output diagnostics."""
        normalized = render_type.strip().lower()
        if normalized not in {"map", "video"}:
            raise ValueError("render_type must be 'map' or 'video'")
        stream = file or sys.stdout
        builtins.print(file=stream)
        self.section(
            f"{normalized.upper()} RENDER — {title} — complete",
            file=stream,
        )
        if output is not None:
            self.success(f"Output: {output}", file=stream)
        if elapsed_seconds is not None:
            elapsed = max(0.0, float(elapsed_seconds))
            self.detail(
                f"Runtime: {elapsed / 3600.0:.2f} h "
                f"({elapsed / 60.0:.1f} min, {elapsed:.1f} s)",
                file=stream,
            )

    def print(
        self,
        *values: Any,
        sep: str = " ",
        end: str = "\n",
        file: TextIO | None = None,
        flush: bool = False,
    ) -> None:
        """Normalize legacy AURORA ``print`` calls through the shared style.

        Existing scripts historically embedded ``[AURORA]``, arrows and
        checkmarks in their messages.  Accepting that notation here keeps the
        call sites concise while enforcing one rule width and error marker.
        Unmarked text, such as an ADQL query, is passed through unchanged.
        """
        stream = file or sys.stdout
        text = sep.join(str(value) for value in values)
        leading_break = text.startswith("\n")
        text = text.lstrip("\n") if leading_break else text

        if leading_break:
            builtins.print(file=stream)
        if not text and end == "\n":
            if not leading_break:
                builtins.print(file=stream, flush=flush)
            return
        if self._is_rule(text):
            return

        completion_title = self._completion_title(text)
        if completion_title is not None:
            self.complete(completion_title, file=stream)
            return

        section_title = self._section_title(text)
        if section_title is not None:
            self.section(section_title, file=stream)
            return

        status_line = self._status_line(
            text,
            error_stream=stream is sys.stderr,
        )
        if status_line is not None:
            builtins.print(status_line, end=end, file=stream, flush=flush)
            return

        builtins.print(text, end=end, file=stream, flush=flush)

    def progress(
        self,
        iterable: Iterable[_T],
        *,
        description: str | None = None,
        desc: str | None = None,
        total: int | None = None,
        unit: str = "item",
        file: TextIO | None = None,
        **kwargs: Any,
    ) -> Iterable[_T]:
        """Return a consistently formatted tqdm iterator when available."""
        try:
            from tqdm import tqdm
        except ImportError:
            return _PassthroughProgress(iterable)

        stream = file or sys.stdout
        label = description if description is not None else desc
        if label is None:
            label = "Working"

        progress_mode = os.environ.get("AURORA_PROGRESS", "auto").lower()
        if progress_mode not in {"auto", "always", "never"}:
            progress_mode = "auto"
        disable = kwargs.pop("disable", None)
        if disable is None:
            disable = (
                progress_mode == "never"
                or (
                    progress_mode == "auto"
                    and not self._stream_supports_progress(stream)
                )
            )

        options = {
            "total": total,
            "desc": label,
            "unit": unit,
            "file": stream,
            "ascii": True,
            "dynamic_ncols": True,
            "leave": False,
            "mininterval": PROGRESS_MIN_INTERVAL_SECONDS,
            "bar_format": (
                "  ↻ {desc}: {percentage:3.0f}%|{bar:24}| "
                "{n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            ),
            "disable": disable,
        }
        options.update(kwargs)
        return tqdm(iterable, **options)


console = AuroraConsole()
