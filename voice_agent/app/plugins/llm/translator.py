import time
from dataclasses import dataclass
from typing import AsyncIterator

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.output import ToolOutput

from app.frame import (
    InterruptFrame,
    StopFrame,
    TextFrame,
    TextStreamFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from app.metrics.latency import mark
from app.pipeline.component import PipelineComponent
from app.plugins.turn.alarm import Alarm

from .naive_agent import split_sentences
from .openrouter import create_pydantic_ai_model as open_router_model

SYSTEM_PROMPT = (
    "Translate the user's Chinese text into natural, concise English. "
    "Output only the translation, nothing else."
)


class Translator:
    def __init__(self, model: Model | None = None, system_prompt: str = SYSTEM_PROMPT):
        model = model if model else open_router_model()
        self.agent = Agent(model=model, system_prompt=system_prompt)

    async def translate(self, text: str) -> str:
        result = await self.agent.run(text)
        return result.output.strip()


class TranslatorComponent(PipelineComponent):
    """Translates each incoming Chinese TextFrame into an English TextFrame.

    Pass `model` / `system_prompt` to configure the agent; `translator` replaces it entirely (e.g. a fake in tests).
    """

    def __init__(self,
            model: Model | None = None,
            system_prompt: str = SYSTEM_PROMPT,
            *,
            translator: Translator | None = None,
            controller=None,
            downstream=None,
        ):
        super().__init__(controller, downstream)
        self.translator = translator or Translator(model=model, system_prompt=system_prompt)

    async def handle(self, frame):
        if not isinstance(frame, TextFrame) or not frame.text.strip():
            return

        tag = f"[#{frame.utterance_id}]"
        self.logger.debug(f"{tag} LLM start: {frame.text!r}")

        started = time.perf_counter()
        text = await self.translator.translate(frame.text)
        elapsed = time.perf_counter() - started

        self.logger.info(f"{tag} LLM {elapsed:.2f}s: {frame.text!r} -> {text!r}")
        if text:
            mark(self.controller, frame.utterance_id, "llm_first")
            self.push(TextFrame(text=text, utterance_id=frame.utterance_id, origin_ts=frame.origin_ts))


# ---- simultaneous interpretation: translate fragments as they come, or wait for more ----

SIMUL_PROMPT = """\
You are a simultaneous interpreter from Chinese into English, like a human interpreter in a booth.

The speaker's Chinese arrives in fragments, cut wherever the speaker briefly pauses, so a fragment can be
half a sentence. Speech recognition may add a full stop to a fragment: that does not mean the sentence is
over. For every new fragment, decide one of two things:

1. Translate: output the English for ALL the Chinese you have not translated yet (fragments you waited on,
   plus this one). Output only the translation, no notes. Do not repeat what you already translated:
   continue naturally from your previous English.
2. Call the `wait` tool: only when the untranslated Chinese cannot be rendered meaningfully yet, e.g. it
   ends inside a phrase ("我想问一下关于"), it is a lone topic or subject still waiting for its verb, or a
   verb still waiting for its object.

Be fast. Like a real interpreter, start as soon as there is a meaningful chunk, even half a sentence;
do not wait for whole sentences. Every wait delays the listener.

A message "（停顿）" means the speaker has stopped for now: translate everything still untranslated.
"""
PAUSE = "（停顿）"  # sent when the speaker stops after a fragment the interpreter waited on


def _fragment_text(text: str) -> str:
    """A fragment without the full stop speech recognition put at its end: it may be half a sentence."""
    return text.strip().rstrip("。.")


@dataclass
class Wait:
    """The interpreter's decision to hear the next fragment before translating."""


def wait() -> Wait:
    """The untranslated Chinese cannot be translated meaningfully yet: wait for the next fragment."""
    return Wait()


def _trim(messages: list[ModelMessage], max_turns: int) -> list[ModelMessage]:
    """Keep the last `max_turns` fragments, cutting only before a plain user prompt (never between a
    `wait` tool call and its return)."""
    starts = [i for i, m in enumerate(messages)
              if isinstance(m, ModelRequest) and all(isinstance(p, UserPromptPart) for p in m.parts)]
    return messages[starts[-max_turns]:] if len(starts) > max_turns else messages


class SimulTranslator:
    """The interpreter agent: keeps the conversation (fragments, translations, waits) and decides per fragment."""

    def __init__(self, model: Model | None = None, system_prompt: str = SIMUL_PROMPT, max_turns: int = 30):
        self.agent = Agent(
            model=model or open_router_model(),
            instructions=system_prompt,  # not system_prompt=: pydantic-ai drops that with an external history
            output_type=[str, ToolOutput(wait, name="wait")],
        )
        self.max_turns = max_turns
        self.history: list[ModelMessage] = []

    async def stream(self, text: str, allow_wait: bool = True) -> AsyncIterator[str | Wait]:
        """Yields the translation as text deltas, or a single Wait. allow_wait=False must translate."""
        kwargs = {} if allow_wait else {"output_type": str}
        async with self.agent.run_stream(text, message_history=self.history or None, **kwargs) as result:
            sent = ""
            async for output in result.stream_output(debounce_by=None):
                if isinstance(output, str) and output.startswith(sent) and len(output) > len(sent):
                    yield output[len(sent):]
                    sent = output
            output = await result.get_output()
            if isinstance(output, str) and output.startswith(sent) and len(output) > len(sent):
                yield output[len(sent):]
            # recorded before a Wait is handed out: the caller stops reading at it
            self.history = _trim(result.all_messages(), self.max_turns)
        if isinstance(output, Wait):
            yield output


class _Pause:
    """Put in the component's own queue when the speaker stopped after a fragment it waited on."""


class SimulTranslatorComponent(PipelineComponent):
    """Simultaneous interpretation: Chinese fragments in (one per VAD utterance), English speech-ready text out.

    The VAD cuts at short pauses, so fragments can be half a sentence. For each fragment the agent, which keeps
    the conversation history, either translates all that is untranslated so far, streamed sentence by sentence
    to the TTS (TextStreamFrame, is_first / is_end per translation), or calls `wait`: nothing goes to the TTS,
    the fragment stays untranslated and the decision is only kept in the history.

    It never waits for long: once the speaker stops (UserStoppedSpeakingFrame) with something untranslated,
    `pause_secs` later it must translate; and after `max_waits` waits in a row, the next fragment must be
    translated too.
    """

    control_frames = (StopFrame, InterruptFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame)

    def __init__(self,
            model: Model | None = None,
            system_prompt: str = SIMUL_PROMPT,
            *,
            pause_secs: float = 1.0,
            max_waits: int = 2,
            translator: SimulTranslator | None = None,
            controller=None,
            downstream=None,
        ):
        super().__init__(controller, downstream)
        self.translator = translator or SimulTranslator(model=model, system_prompt=system_prompt)
        self.pause_secs = pause_secs
        self.max_waits = max_waits
        self._pending: list[TextFrame] = []  # fragments waited on, not translated yet
        self._waits = 0  # waits in a row on the pending fragments
        self._user_speaking = False
        self._pause = Alarm(lambda: self.put(_Pause()))

    async def on_control(self, frame):
        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
            self._pause.cancel()  # more is coming: its text decides
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            if self._pending and not self._pause.armed:
                self._pause.start(self.pause_secs)
        else:
            await super().on_control(frame)

    async def handle(self, frame):
        if isinstance(frame, _Pause):
            if self._pending:
                self.logger.info(f"[#{self._pending[-1].utterance_id}] speaker paused: translate what is pending")
                await self._run(self._pending[-1], PAUSE, allow_wait=False)
            return
        if not isinstance(frame, TextFrame) or not frame.text.strip():
            return
        self._pause.cancel()
        # fragments that arrived while the previous call ran go in together, not one call each
        fragments = [frame] + self._take_queued_fragments()
        self._pending.extend(fragments)
        prompt = "".join(_fragment_text(f.text) for f in fragments) or frame.text
        # waited max_waits times in a row: this one must be translated
        await self._run(fragments[-1], prompt, allow_wait=self._waits < self.max_waits)

    def _take_queued_fragments(self) -> list[TextFrame]:
        """Take the fragments waiting in the queue (other items stay, in order)."""
        fragments, others = [], []
        while not self.queue.empty():
            item = self.queue.get_nowait()
            (fragments if isinstance(item, TextFrame) and item.text.strip() else others).append(item)
        for item in others:
            self.queue.put_nowait(item)
        if fragments:
            self.logger.debug(f"[#{fragments[-1].utterance_id}] {len(fragments)} queued fragment(s) merged")
        return fragments

    async def _run(self, frame: TextFrame, prompt: str, allow_wait: bool):
        tag = f"[#{frame.utterance_id}]"
        started = time.perf_counter()
        buffer, translated, is_first = "", "", True
        try:
            async for delta in self.translator.stream(prompt, allow_wait=allow_wait):
                if isinstance(delta, Wait):
                    self._waits += 1
                    waited = "".join(f.text for f in self._pending)
                    self.logger.info(f"{tag} ⏸ wait {time.perf_counter() - started:.2f}s: {waited!r}")
                    if not self._user_speaking:
                        self._pause.start(self.pause_secs)
                    return
                buffer += delta
                translated += delta
                sentence, buffer = split_sentences(buffer)
                if sentence:
                    self._push_text(sentence, frame, started, is_first=is_first)
                    is_first = False
            if buffer.strip():
                self._push_text(buffer.strip(), frame, started, is_first=is_first)
                is_first = False
        except Exception as e:
            self.logger.error(f"{tag} LLM error: {e}")
        source = "".join(f.text for f in self._pending)
        self._pending.clear()
        self._waits = 0
        if is_first:
            return  # nothing was said: no reply to close
        self._push_text("", frame, started, is_end=True)
        self.logger.info(f"{tag} LLM {time.perf_counter() - started:.2f}s: {source!r} -> {translated.strip()!r}")

    def _push_text(self, text: str, frame: TextFrame, started: float, is_first: bool = False, is_end: bool = False):
        if is_first:
            mark(self.controller, frame.utterance_id, "llm_first")
            self.logger.info(f"[#{frame.utterance_id}] LLM +{time.perf_counter() - started:.2f}s: {text!r}")
        self.push(TextStreamFrame(
            text=text, utterance_id=frame.utterance_id, origin_ts=frame.origin_ts, is_first=is_first, is_end=is_end,
        ))

    async def cleanup(self):
        self._pause.cancel()
