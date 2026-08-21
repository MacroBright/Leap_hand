import numpy as np
import pytest

from gesture_mapping.safe_leap_controller import SafeLeapController


class FakePort:
    def __init__(self, events):
        self.events = events

    def closePort(self):
        self.events.append("close")


class FakeClient:
    def __init__(self, current=None):
        self.current = np.asarray(
            current if current is not None else np.ones(16), dtype=float
        )
        self.events = []
        self.writes = []
        self.fail_write = False
        self.port_handler = FakePort(self.events)

    def connect(self):
        self.events.append("connect")

    def read_pos(self):
        self.events.append("read")
        return self.current.copy()

    def sync_write(self, ids, values, address, size):
        self.events.append(
            ("sync", address, tuple(np.asarray(values).astype(int)))
        )

    def set_torque_enabled(self, ids, enabled, retries=-1):
        self.events.append(("torque", enabled))

    def write_desired_pos(self, ids, pose):
        if self.fail_write:
            raise RuntimeError("write failed")
        copied = np.asarray(pose, dtype=float).copy()
        self.writes.append(copied)
        self.events.append(("position", copied))


def make_controller(clock):
    client = FakeClient()
    controller = SafeLeapController(
        client=client,
        open_pose=np.zeros(16),
        clock=lambda: clock[0],
        sleep=lambda _: None,
        max_speed_rad_s=1.0,
        loss_timeout_s=0.5,
    )
    controller.commanded_pose = np.zeros(16)
    controller.last_update_time = clock[0]
    return controller


def test_target_step_is_limited_by_elapsed_time():
    clock = [10.0]
    controller = make_controller(clock)
    clock[0] = 10.1
    result = controller.track(np.ones(16))
    np.testing.assert_allclose(result, np.full(16, 0.1), atol=1e-6)


def test_short_tracking_loss_holds_last_target():
    clock = [20.0]
    controller = make_controller(clock)
    controller.last_seen_time = 20.0
    controller.commanded_pose[:] = 0.4
    clock[0] = 20.49
    result = controller.on_tracking_lost()
    np.testing.assert_allclose(result, np.full(16, 0.4))


def test_long_tracking_loss_moves_toward_open():
    clock = [30.0]
    controller = make_controller(clock)
    controller.last_seen_time = 30.0
    controller.last_update_time = 30.5
    controller.commanded_pose[:] = 0.4
    clock[0] = 30.6
    result = controller.on_tracking_lost()
    np.testing.assert_allclose(result, np.full(16, 0.3), atol=1e-6)


def test_startup_writes_current_pose_before_enabling_torque():
    client = FakeClient(current=np.full(16, 0.25))
    controller = SafeLeapController(client, np.zeros(16), sleep=lambda _: None)
    controller.start(interpolation_s=0.0)
    position_index = next(
        i for i, event in enumerate(client.events) if event[0] == "position"
    )
    torque_on_index = client.events.index(("torque", True))
    assert position_index < torque_on_index
    np.testing.assert_allclose(
        client.events[position_index][1], np.full(16, 0.25)
    )


def test_startup_applies_validated_low_force_configuration():
    client = FakeClient()
    controller = SafeLeapController(client, np.zeros(16), sleep=lambda _: None)
    controller.start(interpolation_s=0.0)
    assert any(
        event[0:2] == ("sync", 84) and set(event[2]) == {300}
        for event in client.events
    )
    assert any(
        event[0:2] == ("sync", 80) and set(event[2]) == {100}
        for event in client.events
    )
    assert any(
        event[0:2] == ("sync", 102) and set(event[2]) == {150}
        for event in client.events
    )


def test_shutdown_disables_torque_and_closes_port():
    client = FakeClient()
    controller = SafeLeapController(client, np.zeros(16), sleep=lambda _: None)
    controller.start(interpolation_s=0.0)
    controller.shutdown(return_open=False)
    assert client.events[-2:] == [("torque", False), "close"]


def test_startup_failure_attempts_torque_off_and_close():
    client = FakeClient()
    client.fail_write = True
    controller = SafeLeapController(client, np.zeros(16), sleep=lambda _: None)
    with pytest.raises(RuntimeError, match="write failed"):
        controller.start(interpolation_s=0.0)
    assert ("torque", False) in client.events
    assert "close" in client.events


def test_runtime_write_failure_stops_immediately():
    clock = [50.0]
    client = FakeClient()
    controller = SafeLeapController(
        client,
        np.zeros(16),
        clock=lambda: clock[0],
        sleep=lambda _: None,
    )
    controller.start(interpolation_s=0.0)
    client.fail_write = True
    clock[0] += 0.1
    with pytest.raises(RuntimeError, match="write failed"):
        controller.track(np.ones(16))
    assert client.events[-2:] == [("torque", False), "close"]


def test_shutdown_is_idempotent():
    client = FakeClient()
    controller = SafeLeapController(client, np.zeros(16), sleep=lambda _: None)
    controller.start(interpolation_s=0.0)
    controller.shutdown(return_open=False)
    event_count = len(client.events)
    controller.shutdown(return_open=False)
    assert len(client.events) == event_count
