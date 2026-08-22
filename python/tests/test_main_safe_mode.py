import numpy as np

import main


class FakeClock:
    def __init__(self, value):
        self.value = float(value)

    def __call__(self):
        return self.value


class FakePort:
    def __init__(self, events):
        self.events = events

    def closePort(self):
        self.events.append("close")


class FakeClient:
    def __init__(self, current):
        self.current = np.asarray(current, dtype=float)
        self.events = []
        self.port_handler = FakePort(self.events)

    def connect(self):
        self.events.append("connect")

    def disconnect(self):
        self.events.append("disconnect")

    def read_pos(self):
        self.events.append("read")
        return self.current.copy()

    def sync_write(self, ids, values, address, size):
        self.events.append(("sync", address, tuple(np.asarray(values).astype(int))))

    def set_torque_enabled(self, ids, enabled, retries=-1):
        self.events.append(("torque", enabled))

    def write_desired_pos(self, ids, pose):
        self.events.append(("position", np.asarray(pose, dtype=float).copy()))


def test_safe_leap_node_prevents_startup_jump(monkeypatch):
    measured = np.full(16, 0.25)
    client = FakeClient(measured)
    monkeypatch.setattr(main, "DynamixelClient", lambda *args: client)

    node = main.LeapNode(port="/dev/fake", safe_mode=True, sleep=lambda _: None)

    position_index = next(
        i for i, event in enumerate(client.events) if event[0] == "position"
    )
    assert position_index < client.events.index(("torque", True))
    np.testing.assert_allclose(client.events[position_index][1], measured)
    node.disconnect()


def test_safe_leap_node_rate_limits_commands(monkeypatch):
    clock = FakeClock(10.0)
    client = FakeClient(main.OPEN_POSE)
    monkeypatch.setattr(main, "DynamixelClient", lambda *args: client)
    node = main.LeapNode(
        port="/dev/fake",
        safe_mode=True,
        clock=clock,
        sleep=lambda _: None,
    )

    clock.value += 0.1
    node.set_leap(main.OPEN_POSE + np.ones(16))

    np.testing.assert_allclose(node.curr_pos, main.OPEN_POSE + 0.1, atol=1e-6)
    node.disconnect()


def test_safe_disconnect_returns_open_then_disables_torque(monkeypatch):
    clock = FakeClock(20.0)
    client = FakeClient(main.OPEN_POSE)
    monkeypatch.setattr(main, "DynamixelClient", lambda *args: client)
    node = main.LeapNode(
        port="/dev/fake",
        safe_mode=True,
        clock=clock,
        sleep=lambda _: None,
    )
    clock.value += 0.1
    node.set_leap(main.OPEN_POSE + np.ones(16))

    node.disconnect()

    assert client.events[-2:] == [("torque", False), "close"]


def test_legacy_leap_node_keeps_normal_limits(monkeypatch):
    client = FakeClient(main.OPEN_POSE)
    monkeypatch.setattr(main, "DynamixelClient", lambda *args: client)

    node = main.LeapNode(port="/dev/fake")

    assert node.kP == 600
    assert node.kD == 200
    assert node.curr_lim == 350
    node.disconnect()
