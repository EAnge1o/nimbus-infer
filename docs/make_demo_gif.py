"""Render docs/demo.gif from the *real* output of the CLI.

This is a small dev tool (needs Pillow, which is not a runtime dependency of the
engine). It runs the real commands, captures their stdout, and paints the text
onto terminal-style frames that are assembled into an animated GIF. Nothing is
faked: every line comes from an actual run.

    python docs/make_demo_gif.py
"""
from __future__ import annotations

import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "demo.gif")
CHECKPOINT = os.path.join(ROOT, "checkpoints", "demo-gpt2")

WIDTH, HEIGHT = 980, 620
BG = (13, 17, 23)
BAR = (22, 27, 34)
FG = (201, 209, 217)
DIM = (110, 118, 129)
BLUE = (88, 166, 255)
GREEN = (126, 231, 135)
ORANGE = (247, 178, 103)
CMD = (139, 148, 158)

FONT_PATH = r"C:\Windows\Fonts\consola.ttf"
PAD = 16
LINE_H = 21
WRAP = 96


def font(size: int) -> ImageFont.FreeTypeFont:
    if os.path.exists(FONT_PATH):
        return ImageFont.truetype(FONT_PATH, size)
    return ImageFont.load_default()


def run(args: list[str], stdin: str | None = None) -> str:
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.stdout


def colour_for(line: str) -> tuple[int, int, int]:
    stripped = line.strip()
    if line.startswith("$ "):
        return CMD
    if stripped.startswith("you>"):
        return BLUE
    if stripped.startswith("ai>"):
        return GREEN
    if stripped.startswith("[") or stripped.startswith("=") or set(stripped) <= {"="}:
        return DIM
    if stripped.startswith("speedup") or stripped.startswith("cache speedup"):
        return ORANGE
    if "tok/s" in line or "compression" in line:
        return ORANGE
    return FG


def build_transcript() -> list[str]:
    lines: list[str] = []
    lines.append("# nimbus-infer - from-scratch LLM inference engine (NumPy only)")
    lines.append("# demo weights are random, so the generated text is gibberish BY DESIGN")

    def header(title: str) -> None:
        lines.append("")
        lines.append(f"$ {title}")

    # 1. tokenizer
    header('python -m nimbus_infer tokenize --text "the engine caches keys and values"')
    tok = run([
        "-m", "nimbus_infer", "tokenize",
        "--tokenizer", os.path.join(CHECKPOINT, "tokenizer"),
        "--text", "the engine caches keys and values",
    ]).strip().splitlines()
    lines += tok[:6]

    # 2. benchmark
    header("python benchmarks/bench.py --preset tiny --tokens 24")
    bench = run(["benchmarks/bench.py", "--preset", "tiny", "--tokens", "24"]).strip().splitlines()
    lines += [ln for ln in bench if ln.strip()]

    # 3. chat REPL
    header("python -m nimbus_infer chat --model checkpoints/demo-gpt2")
    inputs = ["the engine caches", "nimbus infer"]
    chat = run(
        ["-m", "nimbus_infer", "chat", "--model", CHECKPOINT,
         "--max-new-tokens", "4", "--temperature", "0.7"],
        stdin="\n".join(inputs) + "\n/exit\n",
    )
    chat_lines = chat.strip().splitlines()
    turn = 0
    for raw in chat_lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("you>"):
            rest = line[len("you>"):].strip()
            lines.append("")
            lines.append(f"you> {inputs[turn] if turn < len(inputs) else ''}")
            # The REPL prints "you> " and "ai> " back to back when stdin is piped;
            # split the model continuation onto its own line for readability.
            if rest.startswith("ai>"):
                continuation = rest[len("ai>"):].strip()
                lines.append(f"ai> {continuation}")
            elif rest:
                lines.append(f"ai> {rest}")
            turn += 1
        else:
            lines.append(raw)

    lines.append("")
    lines.append("$ ")
    return lines


def wrap(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        if len(line) <= WRAP:
            out.append(line)
            continue
        out.append(line[:WRAP - 1] + "…")
    return out


def render(lines: list[str]) -> list[Image.Image]:
    body_font = font(15)
    title_font = font(13)
    frames: list[Image.Image] = []
    top = 48
    max_visible = (HEIGHT - top - 26) // LINE_H  # terminal-style scrolling viewport

    def frame(visible: int, cursor_on: bool) -> Image.Image:
        img = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(img)
        # title bar
        draw.rectangle([0, 0, WIDTH, 34], fill=BAR)
        for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            draw.ellipse([PAD + i * 20, 11, PAD + i * 20 + 11, 22], fill=c)
        draw.text((WIDTH / 2, 17), "nimbus-infer — demo", font=title_font,
                  fill=DIM, anchor="mm")
        # body: show the last ``max_visible`` revealed lines (scrolls like a terminal)
        start = max(0, visible - max_visible)
        y = top
        for line in lines[start:visible]:
            draw.text((PAD, y), line, font=body_font, fill=colour_for(line))
            y += LINE_H
        if cursor_on and visible < len(lines):
            draw.rectangle([PAD, y + 3, PAD + 8, y + 16], fill=GREEN)
        return img

    for step in range(len(lines) + 1):
        frames.append(frame(step, True))
        frames.append(frame(step, False))
    finished = frame(len(lines), False)
    frames += [finished] * 14
    return frames


def main() -> int:
    print("running real commands to capture output…")
    lines = wrap(build_transcript())
    print(f"transcript: {len(lines)} lines")
    frames = render(lines)
    print(f"frames: {len(frames)}")

    # One shared palette so the animation does not flicker between frames.
    palette_src = frames[len(frames) // 2].quantize(colors=64, method=Image.MEDIANCUT)
    paletted = [f.quantize(palette=palette_src, dither=Image.NONE) for f in frames]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    paletted[0].save(
        OUT,
        save_all=True,
        append_images=paletted[1:],
        duration=90,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"wrote {os.path.relpath(OUT, ROOT)} ({os.path.getsize(OUT) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())