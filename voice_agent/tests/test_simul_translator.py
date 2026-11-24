"""Simultaneous interpretation: translate fragments as they come, or wait for more (only kept in the history)."""
import asyncio

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from app.frame import TextFrame, TextStreamFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame
from app.plugins.llm.translator import PAUSE, SimulTranslator, SimulTranslatorComponent, _trim


class Sink:
    def __init__(self):
        self.frames = []

    def put(self, frame):
        self.frames.append(frame)


class Script:
    """Fake interpreter: waits while the last prompt is in `wait_on`, otherwise translates."""

    def __init__(self, wait_on=(), translations=None):
        self.wait_on = set(wait_on)
        self.translations = translations or {}
        self.calls = []  # (prompt, wait tool offered)

    async def stream(self, messages, info: AgentInfo):
        prompt = next(p.content for p in reversed(messages[-1].parts) if isinstance(p, UserPromptPart))
        offered = [t.name for t in info.output_tools]
        self.calls.append((prompt, "wait" in offered))
        if prompt in self.wait_on and "wait" in offered:
            yield {0: DeltaToolCall(name="wait", json_args="{}", tool_call_id=f"c{len(self.calls)}")}
            return
        for word in self.translations.get(prompt, f"<{prompt}>").split(" "):
            yield word + " "


def component(script, **kwargs):
    comp = SimulTranslatorComponent(model=FunctionModel(stream_function=script.stream), **kwargs)
    comp.downstream = Sink()
    return comp


def said(comp):
    return [(f.text, f.is_first, f.is_end) for f in comp.downstream.frames]


async def test_waits_then_translates_everything_pending_in_one_reply():
    script = Script(wait_on={"我想问一下"}, translations={"关于明天的会议": "I'd like to ask about tomorrow's meeting."})
    comp = component(script)
    await comp.handle(TextFrame(text="我想问一下", utterance_id=1))
    assert comp.downstream.frames == []              # a wait says nothing
    assert [f.text for f in comp._pending] == ["我想问一下"]
    await comp.handle(TextFrame(text="关于明天的会议", utterance_id=2))
    assert said(comp) == [("I'd like to ask about tomorrow's meeting.", True, False), ("", False, True)]
    assert {f.utterance_id for f in comp.downstream.frames} == {2}   # the reply answers the latest fragment
    assert comp._pending == []
    # the wait is only in the history: a `wait` tool call between the two fragments
    parts = [p for m in comp.translator.history for p in m.parts]
    order = [p.content if isinstance(p, UserPromptPart) else p.tool_name
             for p in parts if isinstance(p, (UserPromptPart, ToolCallPart))]
    assert order == ["我想问一下", "wait", "关于明天的会议"]


async def test_after_max_waits_the_next_fragment_must_be_translated():
    script = Script(wait_on={"a", "b", "c"})
    comp = component(script, max_waits=2)
    for i, text in enumerate("abc", 1):
        await comp.handle(TextFrame(text=text, utterance_id=i))
    assert [offered for _, offered in script.calls] == [True, True, False]
    assert said(comp)[-1] == ("", False, True) and comp._pending == []


async def test_speaker_pause_forces_the_pending_translation(controller):
    script = Script(wait_on={"帮我"}, translations={PAUSE: "help me."})
    comp = component(script, pause_secs=0.05)
    comp.bind(controller)
    run = asyncio.create_task(comp.run())
    comp.put(TextFrame(text="帮我", utterance_id=1))
    await asyncio.sleep(0.1)
    assert comp._pause.armed or said(comp)           # waited, the speaker is silent: the pause timer runs
    await asyncio.sleep(0.2)
    assert script.calls[-1] == (PAUSE, False)        # forced: no wait offered
    assert said(comp) == [("help me.", True, False), ("", False, True)]
    run.cancel()


async def test_the_speaker_starting_again_holds_the_pause(controller):
    script = Script(wait_on={"我想"})
    comp = component(script, pause_secs=0.05)
    comp.bind(controller)
    await comp.handle(TextFrame(text="我想", utterance_id=1))
    assert comp._pause.armed
    controller.receive(UserStartedSpeakingFrame(utterance_id=2))
    await asyncio.sleep(0.02)
    assert not comp._pause.armed                     # more is coming: its text decides
    controller.receive(UserStoppedSpeakingFrame(utterance_id=2))
    await asyncio.sleep(0.02)
    assert comp._pause.armed                         # stopped again with something pending
    comp._pause.cancel()


async def test_long_translation_streams_sentence_by_sentence():
    script = Script(translations={"讲个故事": "Once upon a time. There was a mountain. The end"})
    comp = component(script)
    await comp.handle(TextFrame(text="讲个故事", utterance_id=1))
    assert [f.text for f in comp.downstream.frames] == ["Once upon a time.", " There was a mountain.", "The end", ""]
    assert all(isinstance(f, TextStreamFrame) for f in comp.downstream.frames)


def test_history_is_trimmed_only_before_a_plain_user_prompt():
    user = lambda t: ModelRequest(parts=[UserPromptPart(t)])
    history = [
        user("1"), ModelResponse(parts=[TextPart("one")]),
        user("2"), ModelResponse(parts=[ToolCallPart("wait", {}, "c")]),
        ModelRequest(parts=[UserPromptPart("3")]), ModelResponse(parts=[TextPart("two three")]),
        user("4"), ModelResponse(parts=[TextPart("four")]),
    ]
    assert _trim(history, 2)[0].parts[0].content == "3"
    assert _trim(history, 10) == history


async def test_a_failed_call_drops_the_pending_fragments_and_says_nothing():
    async def broken(messages, info):
        raise RuntimeError("down")
        yield ""
    comp = SimulTranslatorComponent(translator=SimulTranslator(model=FunctionModel(stream_function=broken)))
    comp.downstream = Sink()
    await comp.handle(TextFrame(text="你好", utterance_id=1))
    assert comp.downstream.frames == [] and comp._pending == []


async def test_fragments_queued_during_a_call_go_in_together_without_the_asr_full_stop():
    script = Script()
    comp = component(script)
    comp.put(TextFrame(text="能不能改到。", utterance_id=3))
    comp.put(TextFrame(text="周四上午。", utterance_id=4))
    comp.put(UserStoppedSpeakingFrame(utterance_id=4))   # not a fragment: stays in the queue
    await comp.handle(TextFrame(text="我想问一下。", utterance_id=2))
    assert script.calls == [("我想问一下能不能改到周四上午", True)]
    assert {f.utterance_id for f in comp.downstream.frames} == {4}
    assert comp.queue.qsize() == 1 and isinstance(comp.queue.get_nowait(), UserStoppedSpeakingFrame)
