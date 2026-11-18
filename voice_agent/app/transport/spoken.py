"""What the user actually heard of each bot reply, sentence by sentence.

The TTS marks the first audio chunk of each sentence with its text (AudioFrame.text). The transport adds every
chunk of reply audio here as it queues it, and records how much has been played: exactly when it plays the
audio itself (WebRTC), from its playback estimate otherwise (websocket). When a reply is over, played to the end
or cut off by an interrupt, it becomes a BotSpokeFrame for whoever keeps the conversation (app/context).
"""
from collections import deque
from dataclasses import dataclass

from app.frame import BotSpokeFrame


@dataclass
class Sentence:
    text: str | None            # None: audio that came without its text (a TTS that does not mark sentences)
    total: float = 0.0          # amount of audio, in the transport's unit (bytes, seconds)
    heard: float = 0.0          # how much of it has been played
    start: float | None = None  # wall-clock playback window, for transports that estimate playback
    end: float | None = None

    def fraction(self, complete: bool) -> float:
        # played to the end: all of it was heard, whether or not the transport counted `heard`
        # (the websocket one only estimates it when a reply is cut off)
        if complete:
            return 1.0
        if self.total <= 0:
            return 0.0
        return min(1.0, self.heard / self.total)


class SpokenLedger:
    def __init__(self):
        # key 是这一轮的 utterance_id；value 是这一轮回复的各个 TTS 段（每段可能含几句），按播放顺序
        self._replies: dict[int | None, list[Sentence]] = {}
        
        # 最近已经报告过（发出 BotSpokeFrame）的轮次，防止重复报告；
        # 比如播完后才到的打断，不能把"完整听到"改成"被打断"
        self._finished: deque = deque(maxlen=64)  # replies already reported: a late interrupt must not undo them

    def add(self, utterance_id: int | None, amount: float, text: str | None = None,
            start: float | None = None, end: float | None = None) -> Sentence:
        """Queue reply audio; `text` starts a new sentence, otherwise the audio extends the current one."""
        sentences = self._replies.setdefault(utterance_id, [])
        
        # 带文字：一个 TTS 段开始了，新开一段（文字只在每段的第一块音频上）。
        # 不带文字：属于正在播的最后一段。
        # 列表还是空的却不带文字：TTS 没标文字（如 VolTTS2），开一个 text=None 的段接住音频，最后报告为"不知道文字"。
        if text is not None or not sentences:
            sentences.append(Sentence(text))
        
        sentence = sentences[-1]
        # 这块音频算进最后一段的总量（WebRTC 用字节，WebSocket 用秒；最后只看 heard / total 的比例）
        sentence.total += amount
        
        # 记录开始时间和结束时间
        if start is not None and sentence.start is None:
            sentence.start = start
        if end is not None:
            sentence.end = end
        return sentence

    def hear_until(self, now: float):
        """For transports that estimate playback: everything scheduled before `now` counts as played."""
        # 这个方法是为了ws这种估算播放时长的场景用的
        for sentences in self._replies.values():
            for s in sentences:
                if s.start is None or s.end is None:
                    continue
                if s.end <= s.start:
                    ratio = 1.0 if now >= s.start else 0.0
                else:
                    ratio = max(0.0, min(1.0, (now - s.start) / (s.end - s.start)))
                s.heard = s.total * ratio

    def finish(self, utterance_id: int | None, interrupted: bool) -> BotSpokeFrame | None:
        """The reply is over: what was heard of it. None if it was already reported."""
        # 一个已结算的 utterance不会重复报告
        if utterance_id in self._finished:
            return None
        # 记录计算过的utterances
        self._finished.append(utterance_id)
        
        sentences = self._replies.pop(utterance_id, [])
        complete = not interrupted
        
        # 所有的sentence的文本都为空，说明什么也没有听到
        if sentences and all(s.text is None for s in sentences):
            heard = None  
        
        # 至少有一个 sentence 有文本，记录每个 sentence 的文本和播放比例
        else:
            heard = [(s.text, round(s.fraction(complete), 3)) for s in sentences if s.text is not None]
        return BotSpokeFrame(utterance_id=utterance_id, sentences=heard, interrupted=interrupted)

    def interrupt(self, utterance_id: int | None) -> list[BotSpokeFrame]:
        """Everything still playing or queued was cut off. The interrupted reply is reported even if none of
        its audio arrived yet (cut while thinking): then nothing of it was heard."""
        uids = list(self._replies)
        if utterance_id is not None and utterance_id not in uids:
            uids.append(utterance_id)
        return [frame for uid in uids if (frame := self.finish(uid, interrupted=True)) is not None]
