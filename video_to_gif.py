#!/usr/bin/env python3
"""Convert a video file to an optimized animated GIF using FFmpeg."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


# User-facing conversion defaults and FFmpeg rendering limits.
DEFAULT_GIF_FPS = 15.0
DEFAULT_GIF_WIDTH = 720
DEFAULT_START_SECONDS = 0.0
DEFAULT_GIF_COLORS = 256
MIN_GIF_COLORS = 2
MAX_GIF_COLORS = 256
DEFAULT_GIF_LOOP = True
DEFAULT_OVERWRITE = False
GIF_SCALE_FILTER = "lanczos"
GIF_DITHER_MODE = "sierra2_4a"


def positive_float(value: str) -> float:
    """Return a positive float accepted by argparse."""
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("wartosc musi byc wieksza od zera")
    return number


def non_negative_float(value: str) -> float:
    """Return a non-negative float accepted by argparse."""
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("wartosc nie moze byc ujemna")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Konwertuje wideo do zoptymalizowanego pliku GIF.",
    )
    parser.add_argument(
        "video",
        type=Path,
        nargs="?",
        help="sciezka do pliku wideo; bez argumentu uruchamia tryb interaktywny",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="plik wynikowy (domyslnie: nazwa_filmu.gif)",
    )
    parser.add_argument(
        "--fps",
        type=positive_float,
        default=DEFAULT_GIF_FPS,
        help="liczba klatek na sekunde (domyslnie: 15)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_GIF_WIDTH,
        help="szerokosc GIF-a; wysokosc zostanie dobrana automatycznie (domyslnie: 720)",
    )
    parser.add_argument(
        "--start",
        type=non_negative_float,
        default=DEFAULT_START_SECONDS,
        help="czas rozpoczecia w sekundach (domyslnie: 0)",
    )
    parser.add_argument(
        "--duration",
        type=positive_float,
        help="dlugosc fragmentu w sekundach (domyslnie: do konca filmu)",
    )
    parser.add_argument(
        "--colors",
        type=int,
        default=DEFAULT_GIF_COLORS,
        help="liczba kolorow od 2 do 256 (domyslnie: 256)",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="nie zapetlaj animacji",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="nadpisz istniejacy plik wynikowy",
    )
    return parser


def ask_text(question: str, default: str | None = None) -> str:
    """Ask for text in the terminal and optionally supply a default value."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        answer = input(f"{question}{suffix}: ").strip()
        if answer:
            return answer
        if default is not None:
            return default
        print("Wpisz wartość.")


def ask_number(
    question: str,
    converter,
    validator,
    default=None,
    optional: bool = False,
):
    """Ask for a numeric value until it passes validation."""
    suffix = f" [{default}]" if default is not None else ""
    if optional:
        suffix += " (Enter = bez limitu)"

    while True:
        answer = input(f"{question}{suffix}: ").strip()
        if not answer:
            if optional:
                return None
            if default is not None:
                return default
        try:
            value = converter(answer)
        except ValueError:
            print("Wpisz poprawną liczbę.")
            continue
        if validator(value):
            return value
        print("Podana wartość jest poza dozwolonym zakresem.")


def ask_yes_no(question: str, default: bool = True) -> bool:
    """Ask a yes/no question in Polish."""
    hint = "T/n" if default else "t/N"
    while True:
        answer = input(f"{question} [{hint}]: ").strip().lower()
        if not answer:
            return default
        if answer in {"t", "tak", "y", "yes"}:
            return True
        if answer in {"n", "nie", "no"}:
            return False
        print("Odpowiedz: t (tak) albo n (nie).")


def terminal_path(value: str) -> Path:
    """Normalize a path pasted or dragged into the terminal."""
    return Path(value.strip().strip("\"'")).expanduser().resolve()


def collect_interactive_args() -> argparse.Namespace:
    """Collect all conversion settings through terminal input()."""
    print("\nKonwersja wideo do GIF-a")
    print("────────────────────────")

    while True:
        video = terminal_path(ask_text("Ścieżka do pliku wideo"))
        if video.is_file():
            break
        print(f"Nie znaleziono pliku: {video}")

    default_output = str(video.with_suffix(".gif"))
    while True:
        output = terminal_path(ask_text("Ścieżka pliku wynikowego", default_output))
        if output == video:
            print("Plik wynikowy musi być inny niż plik wejściowy.")
            continue
        if not output.exists():
            overwrite = False
            break
        if ask_yes_no(
            f"Plik {output} już istnieje. Nadpisać go?",
            default=DEFAULT_OVERWRITE,
        ):
            overwrite = True
            break
        default_output = str(output.with_name(f"{output.stem}_nowy.gif"))

    fps = ask_number(
        "Liczba klatek na sekundę",
        float,
        lambda value: value > 0,
        DEFAULT_GIF_FPS,
    )
    width = ask_number(
        "Szerokość GIF-a w pikselach",
        int,
        lambda value: value > 0,
        DEFAULT_GIF_WIDTH,
    )
    start = ask_number(
        "Początek fragmentu w sekundach",
        float,
        lambda value: value >= 0,
        DEFAULT_START_SECONDS,
    )
    duration = ask_number(
        "Długość fragmentu w sekundach",
        float,
        lambda value: value > 0,
        optional=True,
    )
    colors = ask_number(
        "Liczba kolorów",
        int,
        lambda value: MIN_GIF_COLORS <= value <= MAX_GIF_COLORS,
        DEFAULT_GIF_COLORS,
    )
    loop = ask_yes_no("Zapętlać GIF?", default=DEFAULT_GIF_LOOP)

    return argparse.Namespace(
        video=video,
        output=output,
        fps=fps,
        width=width,
        start=start,
        duration=duration,
        colors=colors,
        no_loop=not loop,
        overwrite=overwrite,
    )


def build_ffmpeg_command(args: argparse.Namespace, output: Path) -> list[str]:
    scale = f"scale={args.width}:-1:flags={GIF_SCALE_FILTER}"
    filters = (
        f"[0:v]fps={args.fps:g},{scale},split[frames][palette_source];"
        f"[palette_source]palettegen=max_colors={args.colors}:stats_mode=diff[palette];"
        f"[frames][palette]paletteuse=dither={GIF_DITHER_MODE}:"
        "diff_mode=rectangle"
    )

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if args.overwrite else "-n",
    ]
    if args.start:
        command.extend(("-ss", f"{args.start:g}"))
    command.extend(("-i", str(args.video)))
    if args.duration is not None:
        command.extend(("-t", f"{args.duration:g}"))
    command.extend(
        (
            "-filter_complex",
            filters,
            "-loop",
            "-1" if args.no_loop else "0",
            str(output),
        )
    )
    return command


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.video is None:
        try:
            args = collect_interactive_args()
        except (EOFError, KeyboardInterrupt):
            print("\nPrzerwano wprowadzanie danych.", file=sys.stderr)
            return 130

    if args.width <= 0:
        parser.error("--width musi byc wieksze od zera")
    if not MIN_GIF_COLORS <= args.colors <= MAX_GIF_COLORS:
        parser.error(
            f"--colors musi miescic sie w zakresie od {MIN_GIF_COLORS} "
            f"do {MAX_GIF_COLORS}"
        )
    if shutil.which("ffmpeg") is None:
        parser.error("nie znaleziono FFmpeg; zainstaluj go i dodaj do PATH")

    video = args.video.expanduser().resolve()
    output = (args.output or video.with_suffix(".gif")).expanduser().resolve()
    args.video = video

    if not video.is_file():
        parser.error(f"plik wideo nie istnieje: {video}")
    if output == video:
        parser.error("plik wynikowy musi byc inny niz plik wejsciowy")
    if output.exists() and not args.overwrite:
        parser.error(
            f"plik wynikowy juz istnieje: {output} (uzyj --overwrite)"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_ffmpeg_command(args, output)

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        print(f"Blad: FFmpeg zakonczyl prace kodem {error.returncode}.", file=sys.stderr)
        return error.returncode or 1
    except KeyboardInterrupt:
        print("\nPrzerwano konwersje.", file=sys.stderr)
        return 130

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Gotowe: {output} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
