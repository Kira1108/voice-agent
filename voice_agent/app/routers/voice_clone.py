"""Voice cloning endpoints (Volcengine / Doubao), on top of app/plugins/voice_clone/seed.py.

    GET    /voice_clone                     the page: your voices, record or upload a sample, clone
    POST   /voice_clone                     upload a voice sample (multipart form) and train a voice
    GET    /voice_clone/status              query a voice's status and demo audio
    GET    /voice_clone/voices              your voice slots (voices.json) with their live status and config
    PUT    /voice_clone/voices/{speaker_id} add a slot, or rename it: {"name": ..., "note": ...}
    DELETE /voice_clone/voices/{speaker_id} forget a slot locally (the voice on Volcengine is untouched)

Example:
    curl -F file=@me.wav -F speaker_id=S_xxxxxxxx -F name=我的声音 -F demo_text="你好，这是我复刻的声音。" \
         http://localhost:8000/voice_clone
    curl "http://localhost:8000/voice_clone/status?speaker_id=S_xxxxxxxx"
"""
import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.plugins.voice_clone.registry import VoiceRegistry
from app.plugins.voice_clone.seed import (
    AUDIO_FORMATS,
    LANGUAGES,
    MAX_AUDIO_BYTES,
    SeedVoiceClone,
    VoiceCloneError,
    VoiceStatus,
)

router = APIRouter(
    prefix="/voice_clone",
    tags=["voice_clone"],
)

PAGE = Path(__file__).resolve().parent.parent / "static" / "voice_clone.html"
registry = VoiceRegistry()


@router.get("", response_class=HTMLResponse)
async def page():
    return PAGE.read_text(encoding="utf-8")  # read per request, so edits show up on reload


class VoiceCloneResult(BaseModel):
    speaker_id: str
    status: str                             # NOT_FOUND / TRAINING / SUCCESS / FAILED / ACTIVE
    ready: bool                             # SUCCESS or ACTIVE: usable for synthesis
    available_training_times: int | None    # trainings left on this voice
    create_time: int | None                 # ms timestamp
    language: int | None
    demo_audio: str | None                  # demo clip URL, valid for an hour after training
    model_type: int | None                  # 5 = voice clone 2.0

    @classmethod
    def from_status(cls, status: VoiceStatus) -> "VoiceCloneResult":
        return cls(
            speaker_id=status.speaker_id,
            status=status.status.name,
            ready=status.ready,
            available_training_times=status.available_training_times,
            create_time=status.create_time,
            language=status.language,
            demo_audio=status.demo_audio,
            model_type=status.model_type,
        )


def _http_error(e: Exception) -> HTTPException:
    if isinstance(e, VoiceCloneError):
        # the service's own error code and logid, for a support ticket
        return HTTPException(502, {"code": e.code, "message": str(e), "logid": e.logid})
    if isinstance(e, TimeoutError):
        return HTTPException(504, str(e))
    return HTTPException(400, str(e))


@router.post("", response_model=VoiceCloneResult)
async def clone_voice(
    # the voice sample: wav / mp3 / ogg / m4a / aac / pcm (pcm: 24 kHz mono 16-bit), at most 10 MB.
    # 10-30 s of clean speech from one person works best.
    file: UploadFile = File(..., description="声音样本:wav/mp3/ogg/m4a/aac/pcm(pcm 须 24kHz 单声道),≤10MB"),

    # which voice to train, exactly one of the two:
    #   speaker_id         a prepaid slot bought in the console, e.g. "S_xxxxxxxx"
    #   custom_speaker_id  a postpaid name of your own (needs postpaid cloning enabled); 8-256 letters,
    #                      digits, '-' or '_', starting with a letter; free to train and preview,
    #                      the slot fee is charged on its first synthesis
    speaker_id: str | None = Form(None, description="预付费音色 ID,如 S_xxxxxxxx(与 custom_speaker_id 二选一)"),
    custom_speaker_id: str | None = Form(None, description="后付费自定义音色名,如 myVoice01(首次合成时收槽位费)"),

    # the audio format; taken from the file name when omitted, required for pcm and m4a
    audio_format: str | None = Form(None, description="音频格式,不填按文件后缀判断;pcm、m4a 必填"),

    # what the sample says; if the speech differs too much the training fails (45001109 WERError)
    text: str | None = Form(None, description="样本的朗读文本,和录音差异过大会训练失败"),

    # language of the sample: cn, en, ja, es, id, pt, de, fr, ko, it, th, vi, ru, fil, ms, ar, mx, pt-br, pl, tr, sv
    language: str = Form("cn", description="样本语种,如 cn、en、ja"),

    # 4-300 characters to synthesize as a demo clip, in `language`; longer text makes training slower
    demo_text: str | None = Form(None, description="试听文本,4~300 字,语种需与 language 一致"),

    # denoise a noisy sample; costs some detail on clean ones
    denoise: bool = Form(False, description="样本噪声大时开启降噪"),

    # keep the sample's loudness: closer to the sample, but less uniform volume
    disable_volume_normalization: bool = Form(False, description="关闭音量归一化,音量更贴近样本"),

    # wait for training to finish (polling the status) instead of returning right away
    wait: bool = Form(True, description="是否等训练完成再返回"),

    # how long to wait when `wait` is on, in seconds
    timeout: float = Form(180.0, description="等待训练完成的最长秒数"),

    # a name for the voice in your voice list (voices.json), e.g. "我的声音"; empty keeps the current one
    name: str | None = Form(None, description="给这个音色起的名字,记在音色列表里"),
):
    if bool(speaker_id) == bool(custom_speaker_id):
        raise HTTPException(400, "speaker_id 和 custom_speaker_id 必须且只能填一个")

    audio_format = audio_format or (Path(file.filename or "").suffix.lstrip(".").lower() or None)
    if audio_format and audio_format not in AUDIO_FORMATS:
        raise HTTPException(400, f"不支持的音频格式 {audio_format!r},支持:{', '.join(sorted(AUDIO_FORMATS))}")
    data = await file.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "音频超过 10MB")

    ids = dict(speaker_id=speaker_id or None, custom_speaker_id=custom_speaker_id or None)
    try:
        async with SeedVoiceClone() as vc:
            status = await vc.clone(
                data, **ids,
                audio_format=audio_format,
                text=text or None,
                language=language,
                demo_text=demo_text or None,
                denoise=denoise,
                disable_volume_normalization=disable_volume_normalization,
            )
            if wait and not status.ready:
                status = await vc.wait_until_ready(**ids, timeout=timeout)
    except (VoiceCloneError, TimeoutError, ValueError) as e:
        raise _http_error(e)
    # remember the slot (and its name) once it holds a voice
    registry.upsert(speaker_id or custom_speaker_id, name=name or None)
    return VoiceCloneResult.from_status(status)


@router.get("/status", response_model=VoiceCloneResult)
async def voice_status(
    # exactly one of the two, as for training
    speaker_id: str | None = Query(None, description="预付费音色 ID,如 S_xxxxxxxx"),
    custom_speaker_id: str | None = Query(None, description="后付费自定义音色名"),
):
    if bool(speaker_id) == bool(custom_speaker_id):
        raise HTTPException(400, "speaker_id 和 custom_speaker_id 必须且只能填一个")
    try:
        async with SeedVoiceClone() as vc:
            status = await vc.get_voice(speaker_id=speaker_id, custom_speaker_id=custom_speaker_id)
    except (VoiceCloneError, ValueError) as e:
        raise _http_error(e)
    return VoiceCloneResult.from_status(status)


# ---- your voices ----

MODEL_LABELS = {1: "声音复刻 1.0", 5: "声音复刻 2.0"}
LANGUAGE_NAMES = {code: name for name, code in LANGUAGES.items()}


class VoiceVersion(BaseModel):
    model_type: int | None
    model_version: int | None
    label: str                      # 声音复刻 1.0 / 2.0
    demo_audio: str | None          # demo clip of this model, signed URL valid for about an hour
    icl_speaker_id: str | None


class VoiceInfo(BaseModel):
    speaker_id: str
    name: str
    note: str
    status: str                     # NOT_FOUND (empty slot) / TRAINING / SUCCESS / FAILED / ACTIVE, or ERROR
    ready: bool                     # usable for synthesis
    error: str | None = None        # the query failed (e.g. a slot of another account)
    available_training_times: int | None = None
    create_time: int | None = None  # ms timestamp of the last training
    language: str | None = None     # of the sample: cn, en, ...
    demo_text: str | None = None    # the text the demo clips say
    versions: list[VoiceVersion] = []


class VoiceUpdate(BaseModel):
    name: str | None = None
    note: str | None = None


def _ids(speaker_id: str) -> dict:
    """Prepaid slots are "S_..."; anything else is a postpaid custom name."""
    return {"speaker_id": speaker_id} if speaker_id.startswith("S_") else {"custom_speaker_id": speaker_id}


def _voice_info(entry, status: VoiceStatus | None, error: str | None) -> VoiceInfo:
    info = VoiceInfo(speaker_id=entry.speaker_id, name=entry.name, note=entry.note,
                     status="ERROR" if error else status.status.name, ready=bool(status and status.ready), error=error)
    if status is None:
        return info
    raw = status.raw
    try:
        demo_text = json.loads(raw.get("extra_params") or "{}").get("demo_text")
    except (TypeError, ValueError):
        demo_text = None
    versions = sorted(
        (VoiceVersion(model_type=s.get("model_type"), model_version=s.get("model_version"),
                      label=MODEL_LABELS.get(s.get("model_type"), f"model_type {s.get('model_type')}"),
                      demo_audio=s.get("demo_audio"), icl_speaker_id=s.get("icl_speaker_id"))
         for s in raw.get("speaker_status") or []),
        key=lambda v: -(v.model_version or 0),  # 2.0 first: the one VolTTS / VolTTS2 use
    )
    return info.model_copy(update=dict(
        available_training_times=status.available_training_times,
        create_time=status.create_time or None,
        language=LANGUAGE_NAMES.get(status.language) if status.ready else None,
        demo_text=demo_text,
        versions=versions,
    ))


@router.get("/voices", response_model=list[VoiceInfo])
async def list_voices():
    entries = registry.all()
    async with SeedVoiceClone() as vc:
        async def one(entry):
            try:
                return _voice_info(entry, await vc.get_voice(**_ids(entry.speaker_id)), None)
            except (VoiceCloneError, ValueError) as e:
                return _voice_info(entry, None, str(e))
        return await asyncio.gather(*(one(e) for e in entries))


@router.put("/voices/{speaker_id}", response_model=VoiceInfo)
async def save_voice(speaker_id: str, body: VoiceUpdate):
    entry = registry.upsert(speaker_id, name=body.name, note=body.note)
    try:
        async with SeedVoiceClone() as vc:
            return _voice_info(entry, await vc.get_voice(**_ids(speaker_id)), None)
    except (VoiceCloneError, ValueError) as e:
        return _voice_info(entry, None, str(e))


@router.delete("/voices/{speaker_id}")
async def forget_voice(speaker_id: str):
    if not registry.remove(speaker_id):
        raise HTTPException(404, f"{speaker_id} 不在音色列表里")
    return {"removed": speaker_id}
