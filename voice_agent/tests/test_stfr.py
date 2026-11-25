"""STFR: the fast responder answers right away and hands slow work to background thinker tasks."""
import asyncio

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from app.frame import (
    BotStoppedSpeakingFrame,
    TasksFrame,
    InterruptFrame,
    TextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from app.plugins.llm.stfr import NOTICE, Orchestrator, Responder, STFRComponent, TaskLimitError, Thinker


class Sink:
    def __init__(self):
        self.frames = []

    def put(self, frame):
        self.frames.append(frame)


def last_prompt(messages) -> str:
    request = next(m for m in reversed(messages) if isinstance(m, ModelRequest) and
                   any(isinstance(p, UserPromptPart) for p in m.parts))
    return next(p.content for p in reversed(request.parts) if isinstance(p, UserPromptPart))


def tool_returns(messages) -> list[ToolReturnPart]:
    return [p for m in messages if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, ToolReturnPart)]


class FakeResponder:
    """Calls a tool when the user's words are in `tools` (then says `after`), otherwise says `replies[words]`."""

    def __init__(self, tools=None, replies=None, after="我帮你查一下。"):
        self.tools = tools or {}       # user words -> (tool name, args)
        self.replies = replies or {}   # user words -> what to say
        self.after = after
        self.prompts = []              # every prompt the responder got, task board included

    async def stream(self, messages, info: AgentInfo):
        prompt = last_prompt(messages)
        words = prompt.rsplit("【用户】\n", 1)[-1]
        if not tool_returns(messages[-1:]):
            self.prompts.append(prompt)
            if words in self.tools:
                name, args = self.tools[words]
                yield {0: DeltaToolCall(name=name, json_args=args, tool_call_id=f"c{len(self.prompts)}")}
                return
        else:
            words = "__after__"
        yield self.replies.get(words, self.after if words == "__after__" else f"好的，{words}")


class FakeThinker:
    """Asks for the time with its own tool, then answers once `release` is set."""

    def __init__(self):
        self.release = asyncio.Event()
        self.prompts = []

    async def respond(self, messages, info: AgentInfo):
        returns = tool_returns(messages)
        if not returns:
            self.prompts.append(last_prompt(messages))
            return ModelResponse(parts=[ToolCallPart("get_current_time", {}, "t1")])
        await self.release.wait()
        return ModelResponse(parts=[TextPart(f"现在是 {returns[-1].content}")])


def component(responder: FakeResponder, thinker: FakeThinker, **kwargs) -> STFRComponent:
    comp = STFRComponent(
        responder=Responder(model=FunctionModel(stream_function=responder.stream)),
        thinker=Thinker(model=FunctionModel(function=thinker.respond)),
        **kwargs,
    )
    comp.downstream = Sink()
    return comp


def said(comp, uid=None):
    return [f.text for f in comp.downstream.frames if uid is None or f.utterance_id == uid]


DELEGATE_TIME = ("delegate", '{"instruction": "查一下现在几点"}')


async def wait_for(condition, secs=1.0):
    for _ in range(int(secs / 0.01)):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met in time")


async def test_delegate_answers_at_once_and_the_thinker_runs_in_the_background_with_its_own_tools():
    responder, thinker = FakeResponder(tools={"几点了": DELEGATE_TIME}), FakeThinker()
    comp = component(responder, thinker)
    await comp.handle(TextFrame(text="几点了", utterance_id=1))
    assert said(comp, 1) == ["我帮你查一下。", ""]                  # spoken without waiting for the thinker
    task = comp.orchestrator.tasks[1]
    assert task.status == "running" and task.instruction == "查一下现在几点"
    thinker.release.set()
    await wait_for(lambda: task.status == "done")
    assert task.result.startswith("现在是 20")                     # the thinker called get_current_time
    assert "用户：几点了" in thinker.prompts[0] and "查一下现在几点" in thinker.prompts[0]


async def test_a_result_is_told_proactively_once_nobody_is_speaking(controller):
    responder, thinker = FakeResponder(tools={"几点了": DELEGATE_TIME}, replies={NOTICE: "查到了，现在是下午。"}), FakeThinker()
    comp = component(responder, thinker)
    comp.bind(controller)
    run = asyncio.create_task(comp.run())
    comp.put(TextFrame(text="几点了", utterance_id=1))
    await wait_for(lambda: comp._playing == 1)
    thinker.release.set()                                           # done while the bridge is still playing
    await wait_for(lambda: comp.orchestrator.tasks[1].status == "done")
    await asyncio.sleep(0.05)
    assert said(comp, -1) == []                                     # not over the bot's own voice
    controller.receive(BotStoppedSpeakingFrame(utterance_id=1))
    await wait_for(lambda: said(comp, -1))
    assert said(comp, -1) == ["查到了，现在是下午。", ""]
    assert "刚查到、还没告诉用户" in responder.prompts[-1] and "现在是" in responder.prompts[-1]
    assert comp.orchestrator.tasks[1].reported
    assert comp.context.snapshot()[-1]["user"] == NOTICE
    run.cancel()
    await comp.orchestrator.shutdown()


async def test_a_result_that_lands_while_the_user_speaks_goes_out_with_the_next_reply(controller):
    responder, thinker = FakeResponder(tools={"几点了": DELEGATE_TIME}), FakeThinker()
    comp = component(responder, thinker)
    comp.bind(controller)
    run = asyncio.create_task(comp.run())
    await comp.handle(TextFrame(text="几点了", utterance_id=1))
    controller.receive(BotStoppedSpeakingFrame(utterance_id=1))
    controller.receive(UserStartedSpeakingFrame(utterance_id=2))
    await asyncio.sleep(0.02)
    thinker.release.set()
    await wait_for(lambda: comp.orchestrator.tasks[1].status == "done")
    await asyncio.sleep(0.05)
    assert said(comp, -1) == []                                     # the user has the floor
    controller.receive(UserStoppedSpeakingFrame(utterance_id=2))
    comp.put(TextFrame(text="谢谢", utterance_id=2))
    await wait_for(lambda: said(comp, 2))
    assert "刚查到、还没告诉用户" in responder.prompts[-1]           # on the board of the user's turn
    assert comp.orchestrator.tasks[1].reported
    controller.receive(BotStoppedSpeakingFrame(utterance_id=2))
    await asyncio.sleep(0.05)
    assert said(comp, -1) == []                                     # told already: no notice afterwards
    run.cancel()
    await comp.orchestrator.shutdown()


async def test_an_interrupt_cancels_the_reply_never_the_task(controller):
    responder, thinker = FakeResponder(tools={"几点了": DELEGATE_TIME}), FakeThinker()
    comp = component(responder, thinker)
    comp.bind(controller)
    await comp.handle(TextFrame(text="几点了", utterance_id=1))
    controller.receive(InterruptFrame(utterance_id=1))
    await asyncio.sleep(0.02)
    assert comp._playing is None and comp.orchestrator.tasks[1].status == "running"
    await comp.orchestrator.shutdown()


async def test_cancel_task_stops_the_thinker_and_nothing_is_reported():
    responder = FakeResponder(tools={"几点了": DELEGATE_TIME, "算了": ("cancel_task", '{"task_id": 1}')},
                              after="好的，不查了。")
    thinker = FakeThinker()
    comp = component(responder, thinker)
    await comp.handle(TextFrame(text="几点了", utterance_id=1))
    job = comp.orchestrator.tasks[1].job
    await comp.handle(TextFrame(text="算了", utterance_id=2))
    await asyncio.sleep(0)
    assert comp.orchestrator.tasks[1].status == "cancelled" and job.done()
    assert comp.orchestrator.unreported() == [] and comp.orchestrator.board() == ""


async def test_delegate_beyond_max_tasks_is_refused():
    thinker = FakeThinker()
    orchestrator = Orchestrator(Thinker(model=FunctionModel(function=thinker.respond)), max_tasks=2)
    orchestrator.delegate("a", "")
    orchestrator.delegate("b", "")
    try:
        orchestrator.delegate("c", "")
        raise AssertionError("expected a refusal")
    except TaskLimitError as e:
        assert "2 个任务" in str(e)
    assert len(orchestrator.running()) == 2
    await orchestrator.shutdown()
    assert {t.status for t in orchestrator.tasks.values()} == {"cancelled"}


async def test_the_user_speaking_over_a_notice_cuts_it_and_still_gets_an_answer(controller, broadcasts):
    responder, thinker = FakeResponder(replies={NOTICE: "查到了。", "嗯": "你说。"}), FakeThinker()
    comp = component(responder, thinker)
    comp.bind(controller)
    run = asyncio.create_task(comp.run())
    comp._playing = -1                                              # a notice turn is playing
    comp.put(TextFrame(text="嗯", utterance_id=3))
    await wait_for(lambda: said(comp, 3))
    assert any(isinstance(f, InterruptFrame) and f.utterance_id == -1 for f in broadcasts)
    assert said(comp, 3) == ["你说。", ""]                           # our own interrupt did not cancel the turn
    run.cancel()


async def test_what_was_said_before_a_tool_call_is_not_said_again_after_it():
    async def echoing(messages, info):
        if not tool_returns(messages[-1:]):
            yield "好的，我帮你查一下。"
            yield {0: DeltaToolCall(name="list_tasks", json_args="{}", tool_call_id="x")}
        else:
            for piece in ("好的，我帮", "你查一下。", "\n\n现在没有任务。"):
                yield piece
    comp = STFRComponent(responder=Responder(model=FunctionModel(stream_function=echoing)),
                         thinker=Thinker(model=FunctionModel(function=FakeThinker().respond)))
    comp.downstream = Sink()
    await comp.handle(TextFrame(text="有什么任务", utterance_id=1))
    assert said(comp) == ["好的，我帮你查一下。", "现在没有任务。", ""]
    await comp.orchestrator.shutdown()   # "我帮你查" with nothing running: a task was started for it


async def test_results_told_stay_on_the_board_for_follow_up_questions():
    responder, thinker = FakeResponder(tools={"几点了": DELEGATE_TIME}), FakeThinker()
    comp = component(responder, thinker)
    await comp.handle(TextFrame(text="几点了", utterance_id=1))
    thinker.release.set()
    await wait_for(lambda: comp.orchestrator.tasks[1].status == "done")
    await comp.handle(TextFrame(text="好", utterance_id=2))           # told with this reply
    await comp.handle(TextFrame(text="具体几分", utterance_id=3))
    board = responder.prompts[-1]
    assert "刚查到" not in board and "已经告诉过用户" in board and "现在是 20" in board
    assert "任务" not in board.split("【用户】")[0]                   # nothing that reads as a background job


async def test_every_change_of_a_task_is_broadcast_for_the_page(controller, broadcasts):
    responder, thinker = FakeResponder(tools={"几点了": DELEGATE_TIME}), FakeThinker()
    comp = component(responder, thinker)
    comp.bind(controller)
    await comp.handle(TextFrame(text="几点了", utterance_id=1))
    frames = lambda: [f for f in broadcasts if isinstance(f, TasksFrame)]
    await wait_for(lambda: any(f.tasks[0]["steps"] and f.tasks[0]["steps"][0]["done"] for f in frames()))
    steps = [f.tasks[0]["steps"][0] for f in frames() if f.tasks[0]["steps"]]
    assert steps[0]["tool"] == "get_current_time" and not steps[0]["done"]   # shown while it runs, then done
    thinker.release.set()
    await wait_for(lambda: frames()[-1].tasks[0]["status"] == "done")
    task = frames()[-1].tasks[0]
    assert task["steps"][0]["done"] and task["result"].startswith("现在是") and task["finished"] >= task["started"]
    await comp.handle(TextFrame(text="好", utterance_id=2))
    await asyncio.sleep(0.02)
    assert frames()[-1].tasks[0]["reported"]


async def test_history_markers_the_model_imitates_are_never_spoken_nor_remembered():
    responder = FakeResponder(replies={"天津有什么好玩的": "我去帮你看看。（用户没说话）", NOTICE: "查到了（被打断）。"})
    comp = component(responder, FakeThinker())
    await comp.handle(TextFrame(text="天津有什么好玩的", utterance_id=1))
    await comp.handle(TextFrame(text=NOTICE, utterance_id=-1))
    assert said(comp) == ["我去帮你看看。", "", "查到了。", ""]
    assert [t["said"] for t in comp.context.snapshot()] == ["我去帮你看看。", "查到了。"]


async def test_a_promise_to_look_something_up_without_a_tool_call_is_kept():
    responder = FakeResponder(replies={"黄仁勋最近在干嘛": "没问题，我帮你查查黄仁勋最近在做什么。", "你好": "你好呀。",
                                       "还没好吗": "在查呢，稍等。"})
    thinker = FakeThinker()
    comp = component(responder, thinker)
    await comp.handle(TextFrame(text="你好", utterance_id=1))
    assert comp.orchestrator.tasks == {}                             # nothing promised, nothing started
    await comp.handle(TextFrame(text="黄仁勋最近在干嘛", utterance_id=2))
    task = comp.orchestrator.tasks[1]
    assert "黄仁勋最近在干嘛" in task.instruction and task.status == "running"
    await wait_for(lambda: thinker.prompts)
    assert "用户：黄仁勋最近在干嘛" in thinker.prompts[0]              # the thinker sees the conversation too
    await comp.handle(TextFrame(text="还没好吗", utterance_id=3))       # "我帮你查…" again: already being looked up
    assert len(comp.orchestrator.tasks) == 1
    await comp.orchestrator.shutdown()
