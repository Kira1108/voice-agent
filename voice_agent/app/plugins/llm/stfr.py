"""STFR: slow thinker, fast responder.

    user ⇄ Responder ── delegate / cancel_task / show_task / list_tasks ──► Orchestrator ──► Thinker × N
              ▲                                                               │
              └──────────────────── task finished (into the component queue) ─┘

Responder  the front agent: talks to the user right away, never does slow work itself. It hands slow work to
           the orchestrator with a self-contained text instruction and gets a task id back immediately.
Orchestrator  not an LLM: the table of Tasks, starts / cancels thinker jobs, calls back when one finishes.
Thinker    the back agent: one instruction + the conversation so far in, one result out. Has its own tools
           (THINKER_TOOLS: the time, web search and fetch).
Task       one delegated job: instruction, status, result, and whether the user has been told about it.

How a result reaches the user:
  - proactively: a task finished while nobody is speaking -> the responder gets a notice turn and says it.
    Such a turn has no user utterance: it gets a negative utterance id, and NOTICE stands in for the user.
  - with the next turn: every responder prompt starts with the task board (what is being looked up, results
    not told yet, the last results told for follow-up questions), so a result that came in while someone
    was speaking goes out with the next reply.
The responder must sound like it looked things up itself: the board is labelled 【资料】, the notice turn's
user text is just （用户没说话）, and the prompt forbids talking about tasks or a background.
A result counts as reported once a reply that saw it has been generated to the end (an interrupted one
leaves it on the board).

Lifetimes: an interrupt cancels the reply being generated, never a task; only cancel_task or the end of the
session does. The turn-taking aggregator only knows the user's turns, so it cannot interrupt a notice turn:
when the user's next turn comes in while one is still playing, this component interrupts it itself.
"""
import asyncio
import re
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import Model

from app.context import ConversationContext
from app.frame import (
    BotStoppedSpeakingFrame,
    InterruptFrame,
    StopFrame,
    TasksFrame,
    TextFrame,
    TextStreamFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from app.metrics.latency import mark
from app.pipeline.component import PipelineComponent

from .naive_agent import split_sentences
from .openrouter import create_pydantic_ai_model as open_router_model
from .web_tools import web_fetch, web_search

RESPONDER_PROMPT = """你叫 Vivian，是一个语音助手。你说的每个字都会被直接念给用户听。
- 用口语短句说话，不要列表、markdown、表情符号，用和用户相同的语言。平时一两句就够。
- 你自己不查资料、不做计算。需要事实、日期时间、查询、计算或多步推理时，调用 delegate 交出去，
  指令要写清楚、能单独看懂，照用户的原意写，不要自己补日期、数字这类事实；
  然后用一句话告诉用户你去查了，不要编答案。
- 每个新的需求都要单独调用一次 delegate，正在查的事不会顺带处理新需求。
  没有调用 delegate，就不能说“我帮你查”“我去算”。
- 用户问进度时用 list_tasks / show_task 查；用户说不要了，用 cancel_task 取消。
- 用户的话前面可能有【资料】，是系统给你的，用户看不到：
  - “刚查到”的，合适的时候告诉用户，就像你自己查到的一样，可以说“查到了，……”。
    挑用户最关心的具体信息说：人名、地名、时间、数字、事件，不要只给笼统的结论，可以说两三句。
  - “已经告诉过用户”的，用户追问细节时用它回答；资料里没有的，再 delegate。
  - 永远不要提“后台”“任务”“资料”“系统”，也不要说“有结果了”这种话。
- 只有【资料】里写着的才能当作查到的结果说出来，绝对不要凭自己的记忆说“查到了”。
  用户问“查到了吗”，而【资料】里既没有结果、也没有“正在查”的事，就说还没查，然后马上 delegate。
- 对话记录里括号里的“（用户没说话）”“（被打断）”是系统标记，不是谁说的话，你的回复里绝对不要写。
- 如果用户这一栏是“（用户没说话）”，说明用户没开口，是你刚查到了东西：直接自然地说出来。"""

THINKER_PROMPT = """你是语音助手的研究员。助手把用户想知道的事交给你查，你写一份资料给助手，助手会挑重点讲给用户听。
- 涉及“今天”“今年”“最近”的，先用 get_current_time 确认当前时间。
- 新闻、价格、天气、赛事、活动这类会变的信息，用 web_search 搜；摘要不够再用 web_fetch 读原文。
- 资料要具体、信息量足：先一句话回答问题，再列出关键细节——人名、地名、机构、日期时间、数字、
  事件经过、注意事项，有几条写几条，每条一行，用“- ”开头。不要笼统概括，不要放网址和 markdown 标题。
- 查不到或信息互相矛盾，就照实写明，不要编。"""

NOTICE = "（用户没说话）"
# the responder promised to look something up: if it did not delegate, the component does it for it
PROMISE = re.compile(r"(帮你|给你|我来|我去|我再|这就|马上)\S{0,2}(查|看|搜|找|算|问)|(查|搜|看|找)(一下|一查|查|看)|稍等|等我")
AUTO_INSTRUCTION = "用户刚才说：“{text}”。结合对话记录弄清楚用户想知道什么，然后去查。"
KEEP_REPORTED = 3  # results already told, kept on the board for follow-up questions

TaskStatus = Literal["running", "done", "failed", "cancelled"]


def get_current_time() -> str:
    """The current local date, weekday and time."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S ") + "星期" + "一二三四五六日"[now.weekday()]


THINKER_TOOLS = (get_current_time, web_search, web_fetch)


# ---------------------------------------------------------------- task + orchestrator

class TaskLimitError(RuntimeError):
    """Too many tasks running: the responder is told, and can wait or cancel one."""


@dataclass
class Task:
    id: int
    instruction: str
    status: TaskStatus = "running"
    result: str | None = None
    error: str | None = None
    reported: bool = False  # the user has been told how it ended (or it was cancelled on purpose)
    started: float = field(default_factory=time.time)
    finished: float | None = None
    steps: list[dict] = field(default_factory=list)  # the thinker's tool calls, for display
    job: asyncio.Task | None = field(default=None, repr=False)

    def snapshot(self) -> dict:
        return {
            "id": self.id, "instruction": self.instruction, "status": self.status,
            "result": self.result, "error": self.error, "reported": self.reported,
            "started": self.started, "finished": self.finished, "steps": [dict(s) for s in self.steps],
        }

    def describe(self) -> str:
        line = f"#{self.id}「{self.instruction}」"
        if self.status == "running":
            return f"{line} 正在查，已 {time.time() - self.started:.0f} 秒"
        if self.status == "done":
            return f"{line} 查到：{self.result}"
        if self.status == "failed":
            return f"{line} 失败：{self.error}"
        return f"{line} 已取消"


class Orchestrator:
    """The tasks of one session. Starts a thinker job per task; `on_finished` is called when one is done or
    failed (not when cancelled: whoever cancelled it knows); `on_change` on any change, steps included."""

    def __init__(self, thinker: "Thinker", max_tasks: int = 5,
                 on_finished: Callable[[Task], None] | None = None,
                 on_change: Callable[[], None] | None = None):
        self.thinker = thinker
        self.max_tasks = max_tasks
        self.on_finished = on_finished
        self.on_change = on_change
        self.tasks: dict[int, Task] = {}
        self._next_id = 1

    def changed(self):
        if self.on_change:
            self.on_change()

    def snapshot(self) -> list[dict]:
        return [t.snapshot() for t in self.tasks.values()]

    def running(self) -> list[Task]:
        return [t for t in self.tasks.values() if t.status == "running"]

    def unreported(self) -> list[Task]:
        return [t for t in self.tasks.values() if t.status in ("done", "failed") and not t.reported]

    def delegate(self, instruction: str, transcript: str) -> Task:
        if len(self.running()) >= self.max_tasks:
            raise TaskLimitError(f"已有 {self.max_tasks} 个任务在进行，先等一等或取消一个")
        task = Task(id=self._next_id, instruction=instruction)
        self._next_id += 1
        self.tasks[task.id] = task
        task.job = asyncio.create_task(self._run(task, transcript))
        self.changed()
        return task

    async def _run(self, task: Task, transcript: str):
        def on_step(call_id: str, tool: str, args: dict | None):
            if args is not None:
                task.steps.append({"id": call_id, "tool": tool, "args": args, "done": False, "ts": time.time()})
            else:
                for step in task.steps:
                    if step["id"] == call_id:
                        step["done"] = True
            self.changed()

        try:
            task.result = await self.thinker.run(task.instruction, transcript, on_step=on_step)
            task.status = "done"
        except asyncio.CancelledError:
            return  # cancel() already set the status
        except Exception as e:
            task.status, task.error = "failed", str(e) or type(e).__name__
        task.finished = time.time()
        self.changed()
        if self.on_finished:
            self.on_finished(task)

    def cancel(self, task_id: int) -> Task | None:
        task = self.tasks.get(task_id)
        if task is None or task.status != "running":
            return task
        task.status, task.reported, task.finished = "cancelled", True, time.time()
        if task.job:
            task.job.cancel()
        self.changed()
        return task

    async def shutdown(self):
        jobs = [t.job for t in self.running() if t.job]
        for task in self.running():
            self.cancel(task.id)
        await asyncio.gather(*jobs, return_exceptions=True)

    def board(self) -> str:
        """What the responder is shown before the user's words (never by these names, see the prompt): what is
        being looked up, results not told yet, and the last results told, for follow-up questions."""
        running, unreported = self.running(), self.unreported()
        told = [t for t in self.tasks.values() if t.status == "done" and t.reported][-KEEP_REPORTED:]
        if not running and not unreported and not told:
            return ""
        lines = ["【资料】（用户看不到）"]
        if running:
            lines += ["正在查："] + [f"- {t.instruction}" for t in running]
        if unreported:
            lines.append("刚查到、还没告诉用户：")
            for t in unreported:
                found = t.result if t.status == "done" else f"没查到（{t.error}）"
                lines.append(f"◆ 用户想知道：{t.instruction}\n{found}")
        if told:
            lines.append("已经告诉过用户（追问时用）：")
            lines += [f"◆ {t.instruction}\n{t.result}" for t in told]
        return "\n".join(lines) + "\n【用户】\n"


# ---------------------------------------------------------------- the two agents

class Thinker:
    """The back agent. `tools`: plain functions it may call (default: THINKER_TOOLS)."""

    def __init__(self, model: Model | None = None, system_prompt: str = THINKER_PROMPT,
                 tools: Sequence[Callable] = THINKER_TOOLS):
        self.agent = Agent(model=model or open_router_model(), instructions=system_prompt, tools=list(tools))

    async def run(self, instruction: str, transcript: str,
                  on_step: Callable[[str, str, dict | None], None] | None = None) -> str:
        """on_step(call_id, tool, args) when a tool is called, on_step(call_id, tool, None) when it returns."""
        prompt = f"对话记录：\n{transcript or '（还没有对话）'}\n\n要做的事：{instruction}"
        async with self.agent.iter(prompt) as run:
            async for node in run:
                if not on_step:
                    continue
                if Agent.is_call_tools_node(node):
                    for part in node.model_response.parts:
                        if isinstance(part, ToolCallPart):
                            on_step(part.tool_call_id, part.tool_name, part.args_as_dict())
                elif Agent.is_model_request_node(node):
                    for part in node.request.parts:
                        if isinstance(part, ToolReturnPart):
                            on_step(part.tool_call_id, part.tool_name, None)
        return run.result.output.strip()


class Responder:
    """The front agent: streams what to say; its tools act on the orchestrator. The deps of a run are the
    transcript, handed to the thinker of any task delegated during it."""

    def __init__(self, model: Model | None = None, system_prompt: str = RESPONDER_PROMPT):
        self.orchestrator: Orchestrator | None = None  # set by the component
        self.agent = Agent(model=model or open_router_model(), instructions=system_prompt, deps_type=str)

        # async tools: pydantic-ai runs sync ones in a worker thread, where no task can be started
        @self.agent.tool
        async def delegate(ctx: RunContext[str], instruction: str) -> str:
            """Hand slow work to the background: `instruction` must be understandable on its own."""
            try:
                task = self.orchestrator.delegate(instruction, ctx.deps)
            except TaskLimitError as e:
                return str(e)
            return f"已交给后台，任务 #{task.id}"

        @self.agent.tool_plain
        async def cancel_task(task_id: int) -> str:
            """Cancel a background task."""
            task = self.orchestrator.cancel(task_id)
            return task.describe() if task else f"没有任务 #{task_id}"

        @self.agent.tool_plain
        async def show_task(task_id: int) -> str:
            """Status and result of a background task."""
            task = self.orchestrator.tasks.get(task_id)
            return task.describe() if task else f"没有任务 #{task_id}"

        @self.agent.tool_plain
        async def list_tasks() -> str:
            """All background tasks of this conversation."""
            return "\n".join(t.describe() for t in self.orchestrator.tasks.values()) or "没有任务"

    async def stream(self, prompt: str, history: list[ModelMessage], transcript: str) -> AsyncIterator[str]:
        """Text deltas of the reply, across tool calls (text before a tool call, and after it).

        After a tool call some models start over with what they already said: that echo is not said again."""
        said = ""      # everything yielded so far
        echo = None    # the text of this response held back while it still repeats `said`
        async with self.agent.run_stream_events(prompt, message_history=history or None, deps=transcript) as events:
            async for event in events:
                if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                    delta, echo = event.part.content, ("" if said else None)
                elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                    delta = event.delta.content_delta
                else:
                    continue
                if echo is not None:
                    echo += delta
                    if said.strip().startswith(echo.strip()) or not echo.strip():
                        continue  # still repeating (or only whitespace so far)
                    repeated = echo.strip().startswith(said.strip())
                    delta, echo = (echo.strip()[len(said.strip()):].lstrip() if repeated else echo), None
                    if not delta.strip():
                        continue
                said += delta
                yield delta


# ---------------------------------------------------------------- the component

@dataclass
class _Report:
    """Queued for the component itself: see whether a result can be told now."""


class STFRComponent(PipelineComponent):
    """TextFrame in, TextStreamFrame out, like NaiveAgentComponent; the slow work happens in thinker tasks."""

    control_frames = (StopFrame, InterruptFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
                      BotStoppedSpeakingFrame)

    def __init__(self,
            model: Model | None = None,
            thinker_model: Model | None = None,
            *,
            max_tasks: int = 5,
            thinker_tools: Sequence[Callable] = THINKER_TOOLS,
            responder: Responder | None = None,
            thinker: Thinker | None = None,
            context: ConversationContext | None = None,
            controller=None,
            downstream=None,
        ):
        self.context = context or ConversationContext()  # before super().__init__, which may bind
        super().__init__(controller, downstream)
        self.responder = responder or Responder(model=model)
        self.orchestrator = Orchestrator(
            thinker or Thinker(model=thinker_model, tools=thinker_tools),
            max_tasks=max_tasks,
            on_finished=self._on_task_finished,
            on_change=self._publish_tasks,
        )
        self.responder.orchestrator = self.orchestrator
        self._user_speaking = False
        self._playing: int | None = None  # utterance id of a reply of ours not played to the end yet
        self._turn_id: int | None = None  # utterance id of the turn being handled
        self._notice_id = 0  # notice turns count down: -1, -2, ...
        self._notice_cut = asyncio.Event()

    def bind(self, controller):
        super().bind(controller)
        self.context.bind(controller)

    # ---- events

    def _publish_tasks(self):
        if self.controller is not None:
            self.controller.receive(TasksFrame(tasks=self.orchestrator.snapshot()))

    def _on_task_finished(self, task: Task):
        self.logger.info(f"task {task.describe()}")
        self.put(_Report())

    async def on_control(self, frame):
        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            if frame.utterance_id is None:
                self.put(_Report())  # a false start: nobody will speak after all
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if frame.utterance_id == self._playing:
                self._playing = None
                self.put(_Report())
        else:
            await super().on_control(frame)

    def interrupt(self, frame: InterruptFrame):
        uid = frame.utterance_id
        if uid is not None and uid < 0 and uid != self._turn_id:
            # we cut our own notice turn for the user's next one: only its playback goes, not this turn
            if self._playing == uid:
                self._playing = None
            self._notice_cut.set()
            return
        self._playing = None
        super().interrupt(frame)

    async def cleanup(self):
        await self.orchestrator.shutdown()

    # ---- turns

    async def handle(self, frame):
        if isinstance(frame, _Report):
            if self.orchestrator.unreported() and not self._user_speaking and self._playing is None:
                self._notice_id -= 1
                await self._turn(self._notice_id, NOTICE)
            return
        if not isinstance(frame, TextFrame) or not frame.text.strip():
            return
        if self._playing is not None and self._playing < 0:
            await self._cut_notice(self._playing)
        await self._turn(frame.utterance_id, frame.text, frame)

    async def _cut_notice(self, uid: int):
        self.logger.info(f"[#{uid}] user spoke over the notice: interrupting it")
        self._notice_cut.clear()
        self.controller.receive(InterruptFrame(utterance_id=uid))
        try:
            await asyncio.wait_for(self._notice_cut.wait(), timeout=0.5)  # flushed before the new reply
        except asyncio.TimeoutError:
            pass

    def _transcript(self) -> str:
        lines = []
        for turn in self.context.snapshot():
            lines.append(f"用户：{turn['user']}")
            if turn["said"]:
                lines.append(f"助手：{turn['said']}")
        return "\n".join(lines)

    async def _turn(self, uid: int, text: str, frame: TextFrame | None = None):
        tag = f"[#{uid}]"
        self._turn_id = uid
        started = time.perf_counter()
        self.context.add_user(uid, text)
        prompt, history = self.context.prompt_and_history(uid)
        seen = self.orchestrator.unreported()  # told with this reply, if it gets to the end
        tasks_before = len(self.orchestrator.tasks)
        board = self.orchestrator.board()
        transcript = self._transcript()
        buffer, generated, is_first = "", "", True
        try:
            async for delta in self.responder.stream(board + prompt, history, transcript):
                buffer += delta
                generated += delta
                sentence, buffer = split_sentences(buffer)
                sentence = self._unmark(sentence)
                if sentence.strip():
                    self._push(sentence, uid, frame, started, is_first=is_first)
                    is_first = False
            buffer = self._unmark(buffer)
            if buffer.strip():
                self._push(buffer.strip(), uid, frame, started, is_first=is_first)
                is_first = False
        except Exception as e:
            self.logger.error(f"{tag} responder error: {e}")
            if frame is None:
                return  # a notice nobody asked for: say nothing, the result stays on the board
            generated = "不好意思，我这边出了点问题。"
            self._push(generated, uid, frame, started, is_first=is_first)
            is_first = False
        finally:
            self._turn_id = None
        if not is_first:
            self._push("", uid, frame, started, is_end=True)
            self._playing = uid
        self._keep_promise(tag, text, generated, frame, delegated=len(self.orchestrator.tasks) > tasks_before)
        for task in seen:
            task.reported = True
        if seen:
            self.orchestrator.changed()
        generated = self._unmark(generated)
        self.context.set_generated(uid, generated.strip())
        self.logger.info(f"{tag} responder {time.perf_counter() - started:.2f}s: {text!r} -> {generated.strip()!r}")

    def _keep_promise(self, tag: str, text: str, generated: str, frame: TextFrame | None, delegated: bool):
        """The responder said it would look something up but called no tool, and nothing is being looked up:
        delegate the user's words, so what it promised does happen (models skip the tool call now and then)."""
        if frame is None or delegated or self.orchestrator.running() or not PROMISE.search(generated):
            return
        try:
            task = self.orchestrator.delegate(AUTO_INSTRUCTION.format(text=text), self._transcript())
        except TaskLimitError:
            return
        self.logger.warning(f"{tag} responder promised without delegating: task #{task.id} started for it")

    def _unmark(self, text: str) -> str:
        """Remove the markers of the history (（用户没说话）, （被打断）): models imitate them in their replies.
        A marker never holds a sentence end, so it is never split across two pieces of the stream."""
        for marker in (NOTICE, self.context.interrupted_marker):
            if marker:
                text = text.replace(marker, "")
        return text

    def _push(self, text: str, uid: int, frame: TextFrame | None, started: float,
              is_first: bool = False, is_end: bool = False):
        if is_first and frame is not None:
            mark(self.controller, uid, "llm_first")
        if is_first:
            self.logger.info(f"[#{uid}] responder +{time.perf_counter() - started:.2f}s: {text!r}")
        self.push(TextStreamFrame(
            text=text,
            utterance_id=uid,
            origin_ts=frame.origin_ts if frame else time.time(),
            is_first=is_first,
            is_end=is_end,
        ))
