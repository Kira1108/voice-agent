"""Voice list: the local registry and the /voice_clone/voices endpoints (Volcengine replaced by a fake)."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.routers.voice_clone as vcr
from app.plugins.voice_clone.registry import VoiceRegistry
from app.plugins.voice_clone.seed import VoiceCloneError, VoiceStatus, VoiceStatusCode


def test_registry_add_rename_remove(tmp_path):
    reg = VoiceRegistry(tmp_path / "voices.json")
    assert reg.all() == []
    reg.upsert("S_a", name="我的声音")
    reg.upsert("S_b")
    reg.upsert("S_a", note="录于客厅")               # name kept, note added
    assert [(e.speaker_id, e.name, e.note) for e in reg.all()] == [("S_a", "我的声音", "录于客厅"), ("S_b", "", "")]
    assert reg.remove("S_b") and not reg.remove("S_b")
    assert VoiceRegistry(tmp_path / "voices.json").get("S_a").name == "我的声音"  # persisted


TRAINED = {
    "available_training_times": 14, "create_time": 1794721731000, "language": 0, "status": 2,
    "extra_params": '{"demo_text": "你好"}',
    "speaker_status": [
        {"demo_audio": "https://x/1", "icl_speaker_id": "ICL_1", "model_type": 1, "model_version": 1},
        {"demo_audio": "https://x/2", "icl_speaker_id": "ICL_2", "model_type": 5, "model_version": 2},
    ],
}


class FakeVoiceClone:
    cloned = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def get_voice(self, speaker_id=None, custom_speaker_id=None):
        if speaker_id == "S_other":
            raise VoiceCloneError(55000000, "resource ID is mismatched")
        if speaker_id == "S_trained":
            return VoiceStatus.from_response({"speaker_id": speaker_id, **TRAINED})
        return VoiceStatus(speaker_id, VoiceStatusCode.NOT_FOUND, available_training_times=15)

    async def clone(self, data, speaker_id=None, custom_speaker_id=None, **kwargs):
        self.cloned.append(speaker_id)
        return VoiceStatus(speaker_id, VoiceStatusCode.SUCCESS)


def client(tmp_path, monkeypatch):
    reg = VoiceRegistry(tmp_path / "voices.json")
    monkeypatch.setattr(vcr, "registry", reg)
    monkeypatch.setattr(vcr, "SeedVoiceClone", FakeVoiceClone)
    app = FastAPI()
    app.include_router(vcr.router)
    return TestClient(app), reg


def test_list_shows_live_status_names_and_versions(tmp_path, monkeypatch):
    c, reg = client(tmp_path, monkeypatch)
    reg.upsert("S_trained", name="女儿")
    reg.upsert("S_empty")
    reg.upsert("S_other")
    voices = {v["speaker_id"]: v for v in c.get("/voice_clone/voices").json()}
    t = voices["S_trained"]
    assert (t["name"], t["status"], t["ready"], t["language"], t["demo_text"]) == ("女儿", "SUCCESS", True, "cn", "你好")
    assert [v["label"] for v in t["versions"]] == ["声音复刻 2.0", "声音复刻 1.0"]  # 2.0 first
    assert (voices["S_empty"]["status"], voices["S_empty"]["available_training_times"]) == ("NOT_FOUND", 15)
    assert voices["S_other"]["status"] == "ERROR" and "55000000" in voices["S_other"]["error"]  # one failure, not all


def test_put_adds_and_renames_delete_forgets(tmp_path, monkeypatch):
    c, reg = client(tmp_path, monkeypatch)
    assert c.put("/voice_clone/voices/S_new", json={"name": "爷爷"}).json()["name"] == "爷爷"
    assert c.put("/voice_clone/voices/S_new", json={"note": "老家"}).json()["name"] == "爷爷"
    assert c.delete("/voice_clone/voices/S_new").status_code == 200
    assert c.delete("/voice_clone/voices/S_new").status_code == 404
    assert reg.all() == []


def test_cloning_remembers_the_slot_and_its_name(tmp_path, monkeypatch):
    c, reg = client(tmp_path, monkeypatch)
    r = c.post("/voice_clone", files={"file": ("me.wav", b"RIFFxxxx")},
               data={"speaker_id": "S_empty", "name": "我的声音", "wait": "false"})
    assert r.status_code == 200 and FakeVoiceClone.cloned[-1] == "S_empty"
    assert reg.get("S_empty").name == "我的声音"
