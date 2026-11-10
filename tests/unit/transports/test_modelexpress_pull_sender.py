from types import SimpleNamespace

import torch
from modelexpress_rl import WeightVersionRef

from prime_rl.transports.weights import modelexpress


def build_sender(monkeypatch, events: list, held: dict[str, int]):
    def publish(model, step, step_dir):
        events.append(("publish", f"v{step}"))
        return WeightVersionRef(f"v{step}")

    def release_version(*, version, wait=True):
        # ModelExpress's own wait gives up at its RPC deadline, so the sender polls under its broadcast timeout.
        assert not wait
        if held.get(version.version_id, 0) > 0:
            held[version.version_id] -= 1
            events.append(("engine still reading", version.version_id))
            return False
        events.append(("release", version.version_id))
        return True

    sender = modelexpress.ModelExpressPullWeightSender.__new__(modelexpress.ModelExpressPullWeightSender)
    sender.timeout = 60
    sender.world = SimpleNamespace(is_master=True, world_size=2)
    sender.logger = SimpleNamespace(info=lambda message: None)
    sender._published = []
    sender._trainer = SimpleNamespace(release_version=release_version)
    sender._control = SimpleNamespace(delete_weight_version=lambda version_id: events.append(("delete", version_id)))
    sender._publish = publish
    monkeypatch.setattr(modelexpress.dist, "barrier", lambda: None)
    monkeypatch.setattr(modelexpress.time, "sleep", lambda seconds: None)
    return sender


def test_trainer_frees_its_copy_before_publishing_and_peers_keep_three_versions(monkeypatch):
    events = []
    sender = build_sender(monkeypatch, events, held={})

    for step in range(1, 6):
        sender._broadcast(None, step, None)

    assert events == [
        ("publish", "v1"),
        ("release", "v1"),
        ("publish", "v2"),
        ("release", "v2"),
        ("publish", "v3"),
        ("release", "v3"),
        ("publish", "v4"),
        ("delete", "v1"),
        ("release", "v4"),
        ("publish", "v5"),
        ("delete", "v2"),
    ]


def test_trainer_waits_for_an_engine_reading_its_copy_and_publishes_every_version(monkeypatch):
    events = []
    sender = build_sender(monkeypatch, events, held={"v1": 2})

    sender._broadcast(None, 1, None)
    sender._broadcast(None, 2, None)

    assert events == [
        ("publish", "v1"),
        ("engine still reading", "v1"),
        ("engine still reading", "v1"),
        ("release", "v1"),
        ("publish", "v2"),
    ]


def test_trainer_retires_every_version_once_training_ends(monkeypatch):
    events = []
    sender = build_sender(monkeypatch, events, held={})
    for step in range(1, 3):
        sender._broadcast(None, step, None)
    events.clear()

    sender.close()

    assert events == [("release", "v2"), ("delete", "v1"), ("delete", "v2")]


def test_publishing_refreshes_bound_tensors_that_copy_the_live_weights():
    packed = torch.zeros(4)
    model = SimpleNamespace(state_dict=lambda: {"packed": packed, "q_proj": packed[:2].clone()})
    sender = modelexpress.ModelExpressPullWeightSender.__new__(modelexpress.ModelExpressPullWeightSender)
    sender._bound = model.state_dict()
    # A sharded split yields copies, so a training step leaves the bound q_proj behind.
    packed += 1

    sender._refresh_bound(model)

    assert sender._bound["q_proj"].tolist() == [1.0, 1.0]
    assert sender._bound["packed"] is packed
