"""The conversation as the user experienced it: what they said, and what of each bot reply they actually heard.

An LLM agent left to itself remembers every reply it generated, including the parts that were never played
because the user cut the bot off. Here the history is kept apart from the LLM and filled from two sides:

    user turn     added when the turn reaches the LLM (NaiveAgentComponent), in pipeline order
    bot reply     the generated text is kept as provisional; the transport then reports what was played
                  (BotSpokeFrame, through the controller) and that replaces it. Without such a report the
                  provisional text is used, i.e. the old behaviour.

The LLM only reads it, as pydantic-ai messages (`prompt_and_history`).

Rules:
  - an interrupted reply keeps only the sentences heard to the end (`partial="cut"` also keeps the start of
    the sentence that was cut, by the fraction played), followed by `interrupted_marker`; a TTS chunk holding
    several sentences is split, assuming audio proportional to characters
  - a TTS that does not mark sentences (VolTTS2): the transport only knows what fraction of the reply's
    audio was played, and the generated text is cut by that fraction the same way
  - a reply nothing of which was heard leaves no assistant message; the user turns around it are merged into
    one user message (chat templates such as Gemma's expect user and assistant to alternate)
  - a user turn that starts with the previous, unanswered user turn replaces it: turn-taking re-opened that
    turn and merged the new speech onto it (BargeInAggregator, cut off while thinking)
  - at most `max_turns` turns are kept
"""
from dataclasses import dataclass
import re

from loguru import logger as _logger
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart

from app.frame import BotSpokeFrame, ConversationFrame
from app.frame.base import _join_text

logger = _logger.bind(component="Context")

FULLY_HEARD = 0.999  # float slack: a sentence played to its last chunk
# a sentence: up to an end mark, with the closing quotes / brackets that follow it
_SENTENCE = re.compile(r"[^。！？!?.\n]*[。！？!?.\n]+[”’」』）)】\]》\"']*|[^。！？!?.\n]+$")


def split_sentences(text: str) -> list[str]:
    """'从前有座山。山里有座庙' -> ['从前有座山。', '山里有座庙']; joining the pieces gives the text back."""
    return [s for s in _SENTENCE.findall(text) if s] or [text]


@dataclass
class ContextTurn:
    utterance_id: int | None
    user: str
    assistant: str | None = None  # what the bot said: provisional (generated) until the transport reports
    reported: bool = False        # assistant is what the transport says was heard
    interrupted: bool = False
    generated: str | None = None  # the whole reply the LLM generated, heard or not (for display)
    played: float | None = None   # TTS without sentence marks: fraction of the reply's audio played

    @property
    def answered(self) -> bool:
        return bool(self.assistant and self.assistant.strip())


class ConversationContext:
    """One per session (each NaiveAgentComponent creates its own). Subscribes to BotSpokeFrame and, once bound
    to a controller, broadcasts a ConversationFrame with the whole history whenever it changes (for the page)."""

    control_frames = (BotSpokeFrame,)

    def __init__(self, 
            max_turns: int = 50, 
            partial: str = "drop", 
            interrupted_marker: str = "（被打断）"
        ):
        if partial not in ("drop", "cut"):
            raise ValueError(f"partial must be 'drop' or 'cut', got {partial!r}")
        self.max_turns = max_turns
        self.partial = partial
        self.interrupted_marker = interrupted_marker
        self.turns: list[ContextTurn] = []
        self.controller = None

    def bind(self, controller):
        """Receive BotSpokeFrame from the controller, and publish changes through it."""
        self.controller = controller
        controller.attach(self)

    def _publish(self):
        if self.controller is not None:
            self.controller.receive(ConversationFrame(turns=self.snapshot()))

    def _find(self, utterance_id: int | None) -> ContextTurn | None:
        for turn in reversed(self.turns):
            if turn.utterance_id == utterance_id:
                return turn
        return None

    # ---- writing ----

    def add_user(self, utterance_id: int | None, text: str):
        
        last = self.turns[-1] if self.turns else None
        # aggregator有一个机制，bot在思考的时候被用户打断了，这时候，aggregator会合并上一轮被打断的消息和新的用户输入，形成一个新的turn
        # 此处做一个检查，如果上一轮的的消息没有被回答，且新的用户输入是上一轮消息的延续，则复用上一轮的turn，而不是新建一个turn
        # 此时，替换对话历史中的最后一轮。
        if last is not None and not last.answered and text.startswith(last.user):
            logger.debug(f"[#{utterance_id}] re-opened turn #{last.utterance_id} replaced: {text!r}")
            self.turns[-1] = ContextTurn(utterance_id, text)
            
        # 如果不是思考时被打断，直接新建一个Turn
        else:
            self.turns.append(ContextTurn(utterance_id, text))
            
        # 删除超过最大轮次的对话历史
        del self.turns[:-self.max_turns]
        self._publish()

    def set_generated(self, utterance_id: int | None, text: str):
        """The LLM finished the reply: provisional until the transport reports what was heard."""
        turn = self._find(utterance_id)
        if turn is None:
            return
        turn.generated = text
        if not turn.reported:
            turn.assistant = text
        elif turn.played is not None:
            turn.assistant = self._cut_by_time(text, turn.played)  # the report came first
        self._publish()

    def set_spoken(self, frame: BotSpokeFrame):
        turn = self._find(frame.utterance_id)
        if turn is None or (turn.reported and not turn.interrupted):
            return  # not a turn of this conversation, or already complete
        turn.reported = True
        turn.interrupted = frame.interrupted
        if frame.sentences is not None:
            turn.assistant = self._heard_text(frame.sentences)
        elif frame.played is not None:
            # which text was played is unknown, only how much of the audio: cut the generated text by that
            turn.played = frame.played
            if turn.generated is not None:
                turn.assistant = self._cut_by_time(turn.generated, frame.played)
        # else: nothing known about what was played: keep the generated text
        if frame.interrupted:
            if frame.sentences is not None:
                done = sum(1 for _, f in frame.sentences if f >= FULLY_HEARD)
                heard = f"{done}/{len(frame.sentences)} sentence(s)"
            else:
                heard = f"~{(frame.played or 0) * 100:.0f}% by time"
            logger.info(f"[#{frame.utterance_id}] reply cut off, heard {heard}: {turn.assistant!r}")
        self._publish()

    def _cut_by_time(self, text: str, played: float) -> str:
        """The generated text cut at the fraction of its audio played, assuming audio proportional to
        characters (whole sentences only, unless partial="cut")."""
        return text if played >= FULLY_HEARD else self._heard_text([(text, played)])

    def on_control(self, frame):
        if isinstance(frame, BotSpokeFrame):
            self.set_spoken(frame)

    def _heard_text(self, sentences: list[tuple[str, float]]) -> str:
        """Join what was heard. A TTS chunk may hold several sentences (the LLM streamed faster than one
        sentence at a time): a partly heard chunk keeps the sentences that fit in the fraction played,
        assuming audio proportional to characters."""
        text = ""
        for chunk, fraction in sentences:
            if fraction >= FULLY_HEARD:
                text = _join_text(text, chunk)
                continue
            budget = fraction * len(chunk)
            for sentence in split_sentences(chunk):
                if len(sentence) <= budget:
                    text = _join_text(text, sentence)
                    budget -= len(sentence)
                    continue
                if self.partial == "cut" and budget > 0:
                    cut = sentence[:int(budget)].rstrip()
                    if cut:
                        text = _join_text(text, cut + "…")
                break
            break  # nothing after a chunk cut off was heard
        return text

    def clear(self):
        self.turns.clear()
        self._publish()

    def snapshot(self) -> list[dict]:
        """The turns as plain dicts: `said` is exactly what the LLM gets as the bot's message next turn."""
        return [
            {
                "utterance_id": t.utterance_id,
                "user": t.user,
                "said": self._said(t),
                "heard": t.assistant if t.reported else None,
                "generated": t.generated,
                "reported": t.reported,
                "interrupted": t.interrupted,
            }
            for t in self.turns
        ]

    # ---- reading ----

    def _said(self, turn: ContextTurn) -> str | None:
        if not turn.answered:
            return None
        text = turn.assistant.strip()
        if turn.interrupted and self.interrupted_marker:
            text += self.interrupted_marker
        return text

    def prompt_and_history(self, utterance_id: int | None) -> tuple[str, list[ModelMessage]]:
        """For the turn `utterance_id`: the user prompt and the pydantic-ai history before it.

        User turns left unanswered just before it are part of the prompt, so the history always ends with an
        assistant message and user / assistant alternate.
        """
        index = next((i for i in range(len(self.turns) - 1, -1, -1)
                      if self.turns[i].utterance_id == utterance_id), None)
        if index is None:
            raise KeyError(f"turn #{utterance_id} is not in the context: add_user() first")
        history: list[ModelMessage] = []
        pending: list[str] = []  # user turns since the last thing the user heard from the bot
        for turn in self.turns[:index]:
            pending.append(turn.user)
            said = self._said(turn)
            if said:
                history.append(ModelRequest(parts=[UserPromptPart("\n".join(pending))]))
                history.append(ModelResponse(parts=[TextPart(said)]))
                pending = []
        prompt = "\n".join(pending + [self.turns[index].user])
        return prompt, history

    def transcript(self) -> str:
        """Readable dump, for logs and debugging."""
        lines = []
        for turn in self.turns:
            lines.append(f"[#{turn.utterance_id}] user: {turn.user}")
            said = self._said(turn)
            if said:
                state = "heard" if turn.reported else "generated"
                lines.append(f"[#{turn.utterance_id}] bot ({state}): {said}")
        return "\n".join(lines)
