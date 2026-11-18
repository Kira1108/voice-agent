"""The conversation context: what the user heard of each reply goes into the history, nothing more."""
import asyncio

import numpy as np
import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.context import ConversationContext
from app.frame import AudioFrame, BotSpokeFrame, ConversationFrame, InterruptFrame, TextFrame
from app.plugins.llm.naive_agent import NaiveAgentComponent
from app.plugins.tts.component import Qwen3TTSComponent
from app.transport.spoken import SpokenLedger
from app.transport.webrtc import WebRTCOutput

MARKER = ConversationContext().interrupted_marker  # appended to a cut-off reply


def spoken(uid, sentences, interrupted=False):
    return BotSpokeFrame(utterance_id=uid, sentences=sentences, interrupted=interrupted)


def roles(history):
    """[("user", text), ("bot", text), ...] for easy comparison."""
    out = []
    for m in history:
        for p in m.parts:
            out.append(("user" if isinstance(m, ModelRequest) else "bot", p.content))
    return out


# ---- ConversationContext ----

def test_a_reply_played_to_the_end_goes_into_history_as_heard():
    ctx = ConversationContext()
    ctx.add_user(1, "你好")
    ctx.set_generated(1, "你好呀。今天天气不错。")
    ctx.set_spoken(spoken(1, [("你好呀。", 1.0), ("今天天气不错。", 1.0)]))
    ctx.add_user(2, "是啊")
    prompt, history = ctx.prompt_and_history(2)
    assert prompt == "是啊"
    assert roles(history) == [("user", "你好"), ("bot", "你好呀。今天天气不错。")]


def test_an_interrupted_reply_keeps_only_whole_sentences_heard_and_a_marker():
    ctx = ConversationContext()
    ctx.add_user(1, "讲个故事")
    ctx.set_generated(1, "从前有座山。山里有座庙。庙里有个老和尚。")
    ctx.set_spoken(spoken(1, [("从前有座山。", 1.0), ("山里有座庙。", 0.5), ("庙里有个老和尚。", 0.0)], True))
    ctx.add_user(2, "停")
    prompt, history = ctx.prompt_and_history(2)
    assert roles(history) == [("user", "讲个故事"), ("bot", "从前有座山。" + MARKER)]
    assert prompt == "停"


def test_partial_cut_keeps_the_start_of_the_sentence_that_was_cut():
    ctx = ConversationContext(partial="cut", interrupted_marker="")
    ctx.add_user(1, "q")
    ctx.set_spoken(spoken(1, [("One.", 1.0), ("Two three four five.", 0.5)], True))
    ctx.add_user(2, "next")
    assert roles(ctx.prompt_and_history(2)[1])[1] == ("bot", "One. Two three…")


def test_a_partly_heard_chunk_of_several_sentences_keeps_the_ones_that_fit():
    ctx = ConversationContext(interrupted_marker="")
    ctx.add_user(1, "q")
    # one TTS chunk with three 6-character sentences, 60% played: 10.8 characters
    ctx.set_spoken(spoken(1, [("从前有座山。山里有座庙。庙里有个和尚", 0.6)], True))
    ctx.add_user(2, "x")
    assert roles(ctx.prompt_and_history(2)[1])[1] == ("bot", "从前有座山。")


def test_split_sentences_gives_the_text_back():
    from app.context.conversation import split_sentences
    text = "他说：“好的。”然后走了！真的吗?Yes. 最后一句没有句号"
    pieces = split_sentences(text)
    assert "".join(pieces) == text
    assert pieces[:2] == ["他说：“好的。”", "然后走了！"] and pieces[-1] == " 最后一句没有句号"


def test_nothing_heard_leaves_no_bot_message_and_merges_the_user_turns():
    ctx = ConversationContext()
    ctx.add_user(1, "帮我订机票")
    ctx.set_generated(1, "好的，请问去哪里？")       # generated, but cut before any audio played
    ctx.set_spoken(spoken(1, [], True))
    ctx.add_user(2, "算了，订火车票")
    prompt, history = ctx.prompt_and_history(2)
    assert history == []
    assert prompt == "帮我订机票\n算了，订火车票"     # user / assistant still alternate


def test_without_a_report_the_generated_reply_is_used():
    ctx = ConversationContext()
    ctx.add_user(1, "hi")
    ctx.set_generated(1, "hello there")              # e.g. a transport that does not report
    ctx.add_user(2, "how are you")
    assert roles(ctx.prompt_and_history(2)[1]) == [("user", "hi"), ("bot", "hello there")]


def test_unknown_text_keeps_the_generated_reply_and_marks_the_interrupt():
    ctx = ConversationContext()
    ctx.add_user(1, "hi")
    ctx.set_generated(1, "hello there")
    ctx.set_spoken(spoken(1, None, True))            # TTS without sentence marks
    ctx.add_user(2, "x")
    assert roles(ctx.prompt_and_history(2)[1])[1] == ("bot", "hello there" + MARKER)


def test_report_order_does_not_matter_and_a_late_interrupt_does_not_undo_a_complete_reply():
    ctx = ConversationContext()
    ctx.add_user(1, "hi")
    ctx.set_spoken(spoken(1, [("Hello.", 1.0)]))
    ctx.set_generated(1, "Hello. And more.")         # late: the report wins
    ctx.set_spoken(spoken(1, [], True))              # late interrupt of a finished reply
    ctx.add_user(2, "x")
    assert roles(ctx.prompt_and_history(2)[1])[1] == ("bot", "Hello.")


def test_a_reopened_turn_replaces_the_unanswered_one():
    ctx = ConversationContext()
    ctx.add_user(1, "我想问一下")
    ctx.set_spoken(spoken(1, [], True))              # cut while thinking, turn re-opened
    ctx.add_user(2, "我想问一下明天天气")             # the merged turn
    prompt, history = ctx.prompt_and_history(2)
    assert prompt == "我想问一下明天天气" and history == []
    assert [t.utterance_id for t in ctx.turns] == [2]


def test_at_most_max_turns_are_kept():
    ctx = ConversationContext(max_turns=2)
    for uid in (1, 2, 3):
        ctx.add_user(uid, f"q{uid}")
        ctx.set_spoken(spoken(uid, [(f"a{uid}", 1.0)]))
    assert [t.utterance_id for t in ctx.turns] == [2, 3]
    with pytest.raises(KeyError):
        ctx.prompt_and_history(1)


# ---- SpokenLedger (transport side) ----

def test_ledger_reports_heard_fractions_per_sentence():
    ledger = SpokenLedger()
    s1 = ledger.add(1, 100, text="A.")
    ledger.add(1, 100)                               # more audio of the same sentence
    s2 = ledger.add(1, 100, text="B.")
    s1.heard, s2.heard = 200, 25
    frame = ledger.finish(1, interrupted=True)
    assert frame.sentences == [("A.", 1.0), ("B.", 0.25)] and frame.interrupted
    assert ledger.finish(1, interrupted=True) is None  # reported once


def test_ledger_interrupt_reports_a_reply_cut_before_any_audio():
    ledger = SpokenLedger()
    assert [(f.utterance_id, f.sentences) for f in ledger.interrupt(5)] == [(5, [])]


def test_ledger_without_sentence_marks_reports_unknown_text():
    ledger = SpokenLedger()
    ledger.add(1, 100)
    assert ledger.finish(1, interrupted=False).sentences is None


def test_ledger_reply_played_to_the_end_is_fully_heard_without_counting():
    ledger = SpokenLedger()                          # the websocket transport never counts `heard`
    ledger.add(1, 1.0, text="A.", start=10.0, end=11.0)
    ledger.add(1, 1.0, text="B.", start=11.0, end=12.0)
    assert ledger.finish(1, interrupted=False).sentences == [("A.", 1.0), ("B.", 1.0)]


def test_ledger_estimates_heard_from_the_playback_clock():
    ledger = SpokenLedger()
    ledger.add(1, 1.0, text="A.", start=10.0, end=11.0)
    ledger.add(1, 1.0, text="B.", start=11.0, end=12.0)
    ledger.hear_until(11.5)
    assert ledger.finish(1, interrupted=True).sentences == [("A.", 1.0), ("B.", 0.5)]


# ---- TTS marks sentence starts ----

class FakeTTS:
    sample_rate = 24000

    async def synthesize_stream(self, text):
        for _ in range(3):
            yield np.zeros(240, dtype=np.float32)

    async def aclose(self):
        pass


class Collect:
    def __init__(self):
        self.frames = []

    def put(self, frame):
        self.frames.append(frame)


async def test_tts_marks_the_first_chunk_of_each_sentence_with_its_text():
    tts, out = Qwen3TTSComponent(FakeTTS()), Collect()
    tts.downstream = out
    await tts.handle(TextFrame(text="你好。", utterance_id=1))
    assert [f.text for f in out.frames if f.audio_data] == ["你好。", None, None]


# ---- end to end: LLM component + WebRTC transport + controller ----

REPLY = ["从前有座山。", "山里有座庙。", "庙里有个老和尚。"]


@pytest.fixture
def llm_calls():
    return []


@pytest.fixture
def model(llm_calls):
    async def stream(messages, info: AgentInfo):
        llm_calls.append((messages, info.instructions))
        for sentence in REPLY:
            yield sentence
    return FunctionModel(stream_function=stream)


async def play_reply(llm, out, uid, text):
    """LLM reply → fake TTS (one marked audio frame per pushed chunk, 50 ms per character) → transport queue."""
    chunks = Collect()
    llm.downstream = chunks
    await llm.handle(TextFrame(text=text, utterance_id=uid))
    for frame in chunks.frames:
        if frame.text:
            audio = bytes(int(len(frame.text) * 0.05 * 24000) * 2)
            await out.handle(AudioFrame(audio_data=audio, sample_rate=24000, utterance_id=uid, text=frame.text))
    await out.handle(AudioFrame(audio_data=b"", sample_rate=24000, utterance_id=uid, is_end=True))


async def test_the_next_turn_only_sees_what_was_heard(controller, model, llm_calls):
    llm = NaiveAgentComponent(model=model, system_prompt="SYS")
    llm.bind(controller)
    out = WebRTCOutput(controller)

    # turn 1: played to the end
    await play_reply(llm, out, 1, "讲个故事")
    for _ in range(200):
        out.track._next_chunk()
    await asyncio.sleep(0.05)

    # turn 2: 18 characters = 0.9 s of audio; cut after 0.4 s (20 chunks of 20 ms), in the second sentence
    await play_reply(llm, out, 2, "再讲一个")
    for _ in range(20):
        out.track._next_chunk()
    controller.receive(InterruptFrame(utterance_id=2))
    await asyncio.sleep(0.05)

    await llm.handle(TextFrame(text="你刚才说到哪了", utterance_id=3))
    messages, instructions = llm_calls[-1]
    assert instructions == "SYS"                     # applied even with an external history
    assert roles(messages) == [
        ("user", "讲个故事"), ("bot", "".join(REPLY)),
        ("user", "再讲一个"), ("bot", "从前有座山。" + MARKER),
        ("user", "你刚才说到哪了"),
    ]


async def test_context_false_keeps_the_old_behaviour(controller, model, llm_calls):
    llm = NaiveAgentComponent(model=model, context=False)
    llm.bind(controller)
    llm.downstream = Collect()
    await llm.handle(TextFrame(text="讲个故事", utterance_id=1))
    controller.receive(spoken(1, [], True))          # ignored: no context
    await asyncio.sleep(0.05)
    await llm.handle(TextFrame(text="然后呢", utterance_id=2))
    assert roles(llm_calls[-1][0]) == [("user", "讲个故事"), ("bot", "".join(REPLY)), ("user", "然后呢")]


# ---- shown on the page ----

async def test_every_change_is_published_with_what_was_generated_and_heard(controller):
    ctx = ConversationContext()
    ctx.bind(controller)
    seen = []
    controller.subscribe(seen.append, ConversationFrame)
    ctx.add_user(1, "讲个故事")
    ctx.set_generated(1, "从前有座山。山里有座庙。")
    controller.receive(spoken(1, [("从前有座山。山里有座庙。", 0.6)], True))
    await asyncio.sleep(0.05)
    assert len(seen) == 3                            # user turn, generated, heard
    turn = seen[-1].turns[-1]
    assert turn["said"] == "从前有座山。" + MARKER     # exactly what the LLM gets next turn
    assert turn["heard"] == "从前有座山。" and turn["generated"] == "从前有座山。山里有座庙。"
    assert turn["reported"] and turn["interrupted"]


async def test_webrtc_output_forwards_the_history_to_the_page(controller):
    out = WebRTCOutput(controller)
    sent = []
    out.events.send = sent.append
    controller.receive(ConversationFrame(turns=[{"user": "hi"}]))
    await asyncio.sleep(0.05)
    assert [f.turns for f in sent if isinstance(f, ConversationFrame)] == [[{"user": "hi"}]]
