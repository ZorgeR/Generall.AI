import json

from bot.auth import AuthStore


def make_store(tmp_path, **kw):
    return AuthStore(path=str(tmp_path / "userlist.json"), **kw)


def test_base_chat_ids_are_authorized_and_persisted(tmp_path):
    store = make_store(tmp_path, base_chat_ids=["1", "2"])
    store.load()
    assert store.is_authorized("1") and store.is_authorized("2")
    assert not store.is_authorized("3")
    data = json.loads((tmp_path / "userlist.json").read_text())
    assert set(data["users"]) == {"1", "2"}


def test_admin_is_authorized_even_when_not_in_chat_ids(tmp_path):
    store = make_store(tmp_path, base_chat_ids=["1"], admin_id="99")
    assert store.is_admin("99")
    assert store.is_authorized("99")


def test_allow_all_still_respects_block_list(tmp_path):
    store = make_store(tmp_path, allow_all=True)
    assert store.is_authorized("555")
    store.block("555")
    assert not store.is_authorized("555")
    store.unblock("555")
    assert store.is_authorized("555")


def test_invite_flow(tmp_path):
    store = make_store(tmp_path, base_chat_ids=["1"])
    code = store.generate_invite("1")
    assert len(code) == 8
    assert store.unused_invite_count("1") == 1
    assert store.find_invite(code) == "1"
    assert store.use_invite(code, "2") == "1"
    assert store.is_authorized("2")
    assert store.find_invite(code) is None  # single use
    assert store.use_invite(code, "3") is None
    assert store.unused_invite_count("1") == 0

    reloaded = make_store(tmp_path, base_chat_ids=["1"])
    reloaded.load()
    assert reloaded.is_authorized("2")
    assert reloaded.invites["1"][code]["used_by"] == "2"
