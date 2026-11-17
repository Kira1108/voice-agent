from pydantic import BaseModel, Field
import uuid
from typing import Optional
import time
import datetime
from typing import Literal


class EventFrame(BaseModel):
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the event frame")
    event_type: str = Field(..., description = "type of the event frame")
    timestamp: int = Field(
        default_factory=lambda: int(time.time() * 1000), 
        description="Timestamp of the event frame in milliseconds"
    )
    
class StopFrame(EventFrame):
    event_type: Literal["stop"] = Field(default="stop", description = "type of the event frame")


class ReplyEndFrame(EventFrame):
    """Sent to the client after the last audio chunk of a reply."""
    event_type: Literal["reply_end"] = Field(default="reply_end", description = "type of the event frame")
    utterance_id: Optional[int] = Field(default=None, description="utterance this reply answers")


class BotStartedSpeakingFrame(EventFrame):
    """Broadcast by the transport when the first audio chunk of a reply is sent."""
    event_type: Literal["bot_started_speaking"] = Field(default="bot_started_speaking", description = "type of the event frame")
    utterance_id: Optional[int] = Field(default=None, description="utterance this reply answers")


class BotStoppedSpeakingFrame(EventFrame):
    """Broadcast by the transport once the client is expected to have finished playing a reply."""
    event_type: Literal["bot_stopped_speaking"] = Field(default="bot_stopped_speaking", description = "type of the event frame")
    utterance_id: Optional[int] = Field(default=None, description="utterance this reply answers")


class InterruptFrame(EventFrame):
    """The user cut the bot off: drop the current reply everywhere. Also sent to the client to flush playback."""
    event_type: Literal["interrupt"] = Field(default="interrupt", description = "type of the event frame")
    utterance_id: Optional[int] = Field(default=None, description="utterance whose reply is being interrupted")


class BotSpokeFrame(EventFrame):
    """Broadcast by the transport once a reply is over, played to the end or cut off: what the user heard of it.

    `sentences` lists each sentence of the reply with how much of its audio was played (1.0 = all of it,
    0.0 = none), in order. None when the transport could not tell which text the audio was (a TTS that does
    not mark sentence starts); an empty list when nothing was played at all.
    """
    event_type: Literal["bot_spoke"] = Field(default="bot_spoke", description = "type of the event frame")
    utterance_id: Optional[int] = Field(default=None, description="utterance this reply answers")
    sentences: Optional[list[tuple[str, float]]] = Field(default=None, description="(sentence, fraction played)")
    interrupted: bool = Field(default=False, description="cut off before the end")


class UserStartedSpeakingFrame(EventFrame):
    """Broadcast by the VAD as soon as it hears possible speech (before it is confirmed).
    Its text arrives later, after STT; may be followed by a stop without any text (false start)."""
    event_type: Literal["user_started_speaking"] = Field(default="user_started_speaking", description = "type of the event frame")
    utterance_id: Optional[int] = Field(default=None, description="id the utterance will get if confirmed")


class UserStoppedSpeakingFrame(EventFrame):
    """Broadcast by the VAD when the user goes quiet again: an utterance ended, or a false start was dropped."""
    event_type: Literal["user_stopped_speaking"] = Field(default="user_stopped_speaking", description = "type of the event frame")
    utterance_id: Optional[int] = Field(default=None, description="utterance that ended, None for a false start")


class ConversationFrame(EventFrame):
    """Broadcast by the ConversationContext whenever it changes: the whole history, as the LLM will see it next
    turn, plus what each reply generated, so a page can show what was cut. Informational, for display."""
    event_type: Literal["conversation"] = Field(default="conversation", description = "type of the event frame")
    turns: list[dict] = Field(default_factory=list, description="ContextTurn snapshots, oldest first")
