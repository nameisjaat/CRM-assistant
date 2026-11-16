"""Chat with the Olist assistant in the terminal, with the answer streamed as it's written.

    .venv\\Scripts\\python.exe run_chatbot.py

Type a question after "> " and press Enter. Type exit (or quit, or press Ctrl+C) to stop.
The conversation has short-term memory (InMemorySaver) for as long as the program runs.

Only the final answer is printed: no routes, SQL, logs or warnings. The answer is
streamed token by token from the shared synthesizer (the `synthesize` node of the chatbot
graph). If the synthesizer has to rewrite its answer (the grounding check found numbers
not in the results), the first draft is erased from the screen and the final version is
streamed in its place, so only the final output remains. Out-of-scope messages get their
fixed reply printed directly (no LLM writes it).

Reads OPENAI_API_KEY and DATABASE_URL from the project's .env.
"""
import logging
import os
import re
import shutil
import sys
import uuid
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)          # e.g. sqlglot's "unsupported syntax" warnings

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver

from chatbot.graph import build_chatbot
from sql_lane.execute import ConnectionFailure

ANSWER_NODE = "synthesize"                # the only node whose tokens are shown
EXIT_WORDS = {"exit", "quit", "bye"}


class AnswerStream:
    """Pulls the text of the "answer" field out of the synthesizer's streamed JSON.

    The synthesizer returns structured output, so its tokens arrive as pieces of
    {"answer": "..."}. This decodes the string value as it streams (including escapes
    like \\n and \\") and returns only the new readable text for each piece.
    """

    _START = re.compile(r'"answer"\s*:\s*"$')
    _ESCAPES = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "/": "/", "r": "", "b": "", "f": ""}

    def __init__(self):
        self.prefix, self.started, self.done = "", False, False
        self.escape, self.unicode = False, None

    def feed(self, piece: str) -> str:
        out = []
        for ch in piece:
            if self.done:
                break
            if not self.started:
                self.prefix += ch
                self.started = bool(self._START.search(self.prefix))
            elif self.unicode is not None:                 # inside \\uXXXX
                self.unicode += ch
                if len(self.unicode) == 4:
                    out.append(chr(int(self.unicode, 16)))
                    self.unicode = None
            elif self.escape:
                self.escape = False
                if ch == "u":
                    self.unicode = ""
                else:
                    out.append(self._ESCAPES.get(ch, ch))
            elif ch == "\\":
                self.escape = True
            elif ch == '"':
                self.done = True                           # end of the answer string
            else:
                out.append(ch)
        return "".join(out)


def erase(text: str) -> None:
    """Remove `text` (already printed from the current line on) from the terminal."""
    if not text:
        return
    width = max(1, shutil.get_terminal_size().columns)
    lines = sum(max(1, -(-len(line) // width)) for line in text.split("\n"))
    up = lines - 1
    sys.stdout.write("\r" + (f"\x1b[{up}A" if up else "") + "\x1b[J")   # cursor up, clear below
    sys.stdout.flush()


def answer_turn(bot, question: str, config: dict) -> None:
    """Run one turn, streaming the synthesizer's answer; end with only the final answer shown."""
    printed, run_id, decoder = "", None, AnswerStream()

    for chunk, meta in bot.stream({"question": question}, config, stream_mode="messages"):
        if meta.get("langgraph_node") != ANSWER_NODE or not isinstance(chunk, AIMessageChunk):
            continue
        if chunk.id != run_id:                 # a new synthesizer call (e.g. the grounding retry)
            erase(printed)
            printed, run_id, decoder = "", chunk.id, AnswerStream()
        text = decoder.feed(chunk.content if isinstance(chunk.content, str) else "")
        if text:
            sys.stdout.write(text)
            sys.stdout.flush()
            printed += text

    # Whatever was streamed, the screen must end with the final answer (out-of-scope reply,
    # a fallback answer, or a model that didn't stream tokens).
    final = bot.get_state(config).values.get("answer", "") or ""
    if printed.strip() != final.strip():
        erase(printed)
        sys.stdout.write(final)
    sys.stdout.write("\n\n")
    sys.stdout.flush()


def main() -> None:
    load_dotenv(ROOT / ".env")
    if os.name == "nt":
        os.system("")                          # enable ANSI cursor codes in Windows terminals
    bot = build_chatbot(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": f"terminal-{uuid.uuid4().hex[:8]}"}}

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in EXIT_WORDS:
            break
        try:
            answer_turn(bot, question, config)
        except ConnectionFailure:
            print("Sorry, the database can't be reached right now. Please try again later.\n")
        except KeyboardInterrupt:
            print("\n")


if __name__ == "__main__":
    main()
