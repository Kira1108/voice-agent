"""The voices you own, by name: a small local list of voice slots ("S_..." ids) with a name and a note each.

Volcengine only knows the slot ids; which slot is whose voice lives here, in a JSON file (`voices.json` in
voice_agent/, kept out of git: the ids belong to one account). Status, demo audio and the rest are always
queried live (see app/routers/voice_clone.py).

    registry = VoiceRegistry()
    registry.upsert("S_xxxxxxxx", name="我的声音")
    registry.all()  # [VoiceEntry(speaker_id="S_xxxxxxxx", name="我的声音", note="")]
"""
import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import get_config


@dataclass
class VoiceEntry:
    speaker_id: str
    name: str = ""
    note: str = ""


class VoiceRegistry:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else get_config().voices_path
        self._lock = threading.Lock()

    def _load(self) -> list[VoiceEntry]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return [VoiceEntry(**{k: v for k, v in item.items() if k in VoiceEntry.__dataclass_fields__})
                for item in data.get("voices", [])]

    def _save(self, entries: list[VoiceEntry]):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"voices": [asdict(e) for e in entries]}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, self.path)  # atomic: a crash never leaves half a file

    def all(self) -> list[VoiceEntry]:
        with self._lock:
            return self._load()

    def get(self, speaker_id: str) -> VoiceEntry | None:
        return next((e for e in self.all() if e.speaker_id == speaker_id), None)

    def upsert(self, speaker_id: str, name: str | None = None, note: str | None = None) -> VoiceEntry:
        """Add the slot, or update its name / note (None leaves a field as it is)."""
        speaker_id = speaker_id.strip()
        if not speaker_id:
            raise ValueError("speaker_id 不能为空")
        with self._lock:
            entries = self._load()
            entry = next((e for e in entries if e.speaker_id == speaker_id), None)
            if entry is None:
                entry = VoiceEntry(speaker_id)
                entries.append(entry)
            if name is not None:
                entry.name = name.strip()
            if note is not None:
                entry.note = note.strip()
            self._save(entries)
            return entry

    def remove(self, speaker_id: str) -> bool:
        """Forget a slot locally (the voice on Volcengine is untouched). False if it was not listed."""
        with self._lock:
            entries = self._load()
            kept = [e for e in entries if e.speaker_id != speaker_id]
            if len(kept) == len(entries):
                return False
            self._save(kept)
            return True
