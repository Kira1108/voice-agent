import time
from typing import AsyncGenerator
from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.context import ConversationContext
from app.frame import InterruptFrame, StopFrame, TextFrame, TextStreamFrame
from app.metrics.latency import mark
from app.pipeline.component import PipelineComponent

from .openrouter import create_pydantic_ai_model as open_router_model
from .local_llm import create_pydantic_ai_model as local_llm_model

SYSTEM_PROMPT = (
    "You are Vivian, a smart AI assistant. "
    "You interact with users in a helpful and intelligent manner."
    "You always answer in relative short sentences, in the same language as the user."
)

SENTENCE_ENDS = ".。!！?？\n"
# closing marks that belong to the sentence before them, e.g. 他说：“好的。”
CLOSING_MARKS = "\"'”’」』）)】]》>"
MIN_SENTENCE_LEN = 3


def split_sentences(buffer: str) -> tuple[str, str]:
    """Returns (text to send, rest): everything up to the last sentence end, unless it is too short."""
    last = max(buffer.rfind(c) for c in SENTENCE_ENDS)
    if last == -1:
        return "", buffer
    while last + 1 < len(buffer) and buffer[last + 1] in CLOSING_MARKS:
        last += 1
    head, rest = buffer[:last + 1], buffer[last + 1:]
    if len(head.strip()) <= MIN_SENTENCE_LEN:
        return "", buffer
    return head, rest


class NaiveAgentRunner:
    def __init__(self,
            model:Model = None, 
            system_prompt: str = SYSTEM_PROMPT):
        
        model = model if model else open_router_model()
        # instructions, not system_prompt: pydantic-ai drops system_prompt when a message_history is passed in
        self.agent = Agent(model=model, instructions=system_prompt)
        self.message_history = None

    async def stream(self, text: str, message_history: list | None = None) -> AsyncGenerator[str, None]:
        """Stream the reply to `text`.

        message_history: the conversation so far, kept by the caller (app/context); nothing is remembered here.
        None: the runner keeps the history itself, every generated reply in full, heard or not.
        """
        own_history = message_history is None
        history = self.message_history if own_history else message_history
        async with self.agent.run_stream(text, message_history=history or None) as result:
            async for message in result.stream_text(delta=True):
                yield message
            # only a fully streamed turn is remembered; an aborted one leaves the history as it was
            if own_history:
                self.message_history = result.all_messages()

    def reset(self):
        self.message_history = None

class NaiveAgentComponent(PipelineComponent):
    """Handles each incoming TextFrame using the NaiveAgent.

    Pass `model` / `system_prompt` to configure the agent; `runner` replaces it entirely (e.g. a fake in tests).

    The conversation history is a ConversationContext (app/context): it remembers what the user actually heard
    of each reply, as reported by the transport. Pass your own to configure it, or `context=False` for the
    runner to remember every generated reply in full.
    """

    control_frames = (StopFrame, InterruptFrame)

    def __init__(self,
            model: Model | None = None,
            system_prompt: str = SYSTEM_PROMPT,
            *,
            context: ConversationContext | bool | None = None,
            runner: NaiveAgentRunner | None = None,
            controller=None,
            downstream=None,
        ):
        # set before super().__init__, which binds to the controller when one is given
        if context is None or context is True:
            context = ConversationContext()
        self.context: ConversationContext | None = context or None
        super().__init__(controller, downstream)
        self.runner = runner or NaiveAgentRunner(model=model, system_prompt=system_prompt)

    def bind(self, controller):
        super().bind(controller)
        if self.context is not None:
            self.context.bind(controller)  # BotSpokeFrame in, ConversationFrame out

    def _push_text(self, 
            text: str, 
            frame, tag: str, 
            started: float, 
            is_first: bool = False, 
            is_end: bool = False
        ):
        if is_first:
            mark(self.controller, frame.utterance_id, "llm_first")
            self.logger.info(f"{tag} LLM +{time.perf_counter() - started:.2f}s: {text!r}")
        
        self.push(TextStreamFrame(
            text=text, 
            utterance_id=frame.utterance_id, 
            origin_ts=frame.origin_ts,
            is_first=is_first,
            is_end=is_end
        ))

    async def handle(self, frame):
        
        if not isinstance(frame, TextFrame) or not frame.text.strip():
            return

        tag = f"[#{frame.utterance_id}]"
        self.logger.debug(f"{tag} LLM start: {frame.text!r}")

        started = time.perf_counter()
        buffer = ""  # local, so an aborted turn leaves nothing behind
        generated = ""

        if self.context is None:
            stream = self.runner.stream(frame.text)
        else:
            # the user's turn enters the conversation now; the reply only once it has been heard
            self.context.add_user(frame.utterance_id, frame.text)
            prompt, history = self.context.prompt_and_history(frame.utterance_id)
            stream = self.runner.stream(prompt, message_history=history)

        # 句首空格和 strip():现在不处理没关系,但如果以后要在 TTS 前做文本匹配或者统计字数,这个空格会让人困惑。
        # 一次推送包含多句:这个会影响首句延迟,但要等你实际用麦克风体验过,觉得回复开始得慢,再回来调。现在的日志已经能看到每次推送的耗时,到时候有数据可看。
        try:
            is_first = True
            async for delta in stream:
                buffer += delta
                generated += delta
                sentence, buffer = split_sentences(buffer)
                if sentence:
                    self._push_text(sentence, frame, tag, started, is_first=is_first)
                    is_first = False
                    

            if buffer.strip():
                self._push_text(buffer.strip(), frame, tag, started, is_first=is_first)
                is_first = False
                
            self._push_text("", frame, tag, started, is_first=is_first, is_end=True)
                
        except Exception as e:
            generated = "不好意思，系统出了点毛病,再见。"
            self._push_text(generated, frame, tag, started, is_first=is_first, is_end=True)
            self.logger.error(f"{tag} LLM error: {e}")

        # provisional: replaced by what the transport reports as heard (an interrupt never gets here)
        if self.context is not None:
            self.context.set_generated(frame.utterance_id, generated.strip())
            
        elapsed = time.perf_counter() - started

        self.logger.info(f"{tag} LLM {elapsed:.2f}s: {frame.text!r}")
