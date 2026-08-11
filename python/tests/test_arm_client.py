"""arm_client 单测：用 pyserial loop:// 回环捕获写入的 remote_event 命令。

注: pyserial 的 loop:// 是"自回环"（每次 serial_for_url 各自持有独立队列）。
因此测试与 ArmClient 共享同一个 loop:// 连接（经 ser= 注入），
才能在同一回环上捕获/喂入数据。
"""
import threading
import time

import serial  # noqa: F401  (确保 pyserial 可用)


def test_remote_event_format():
    from gesture_mapping.arm_client import ArmClient
    import serial as _s
    s = _s.serial_for_url("loop://", baudrate=115200, timeout=0.1)
    c = ArmClient("loop://", ser=s)
    c.remote_event(vx=0.7, vy=-0.3, vz=0.5, j5=0.4, j6=-0.9, j4=0.6)
    time.sleep(0.05)
    line = s.readline().decode().strip()
    # 期望 p0=-0.700 p1=-0.300 p2=-0.900 p3=-0.400 p4=0.500 p5=-0.500 p6=0.600
    parts = line.split()
    assert parts[0] == "remote_event"
    vals = [float(v) for v in parts[1:8]]
    assert len(vals) == 7
    assert abs(vals[0] - (-0.7)) < 1e-3   # p0=-vx
    assert abs(vals[1] - (-0.3)) < 1e-3   # p1=vy
    assert abs(vals[2] - (-0.9)) < 1e-3   # p2=j6
    assert abs(vals[3] - (-0.4)) < 1e-3   # p3=-j5
    assert abs(vals[4] - 0.5) < 1e-3      # p4=vz
    assert abs(vals[5] - (-0.5)) < 1e-3   # p5=-vz
    assert abs(vals[6] - 0.6) < 1e-3      # p6=j4
    c.close()
    s.close()


def test_remote_event_j4_defaults_zero():
    from gesture_mapping.arm_client import ArmClient
    import serial as _s
    s = _s.serial_for_url("loop://", baudrate=115200, timeout=0.1)
    c = ArmClient("loop://", ser=s)
    c.remote_event(vx=0.0, vy=0.0, vz=0.0, j5=0.0)   # j6/j4 走默认 0
    time.sleep(0.05)
    line = s.readline().decode().strip()
    vals = [float(v) for v in line.split()[1:8]]
    assert len(vals) == 7
    assert vals[6] == 0.0   # p6=j4=0.0
    c.close()
    s.close()


def test_soft_reset():
    from gesture_mapping.arm_client import ArmClient
    import serial as _s
    s = _s.serial_for_url("loop://", baudrate=115200, timeout=0.1)
    c = ArmClient("loop://", ser=s)
    c.soft_reset()
    time.sleep(0.05)
    line = s.readline().decode().strip()
    assert line == "soft_reset"
    c.close()
    s.close()


def test_get_state_parse():
    from gesture_mapping.arm_client import ArmClient
    import serial as _s
    s = _s.serial_for_url("loop://", baudrate=115200, timeout=0.1)
    c = ArmClient("loop://", ser=s)

    def feed():
        time.sleep(0.05)
        s.write(b"STATE:90.00,45.00,67.00,-157.00,0.00,5.00,"
                b"0,0,0,0,0,0,0,0,0,0,0,0\n")

    t = threading.Thread(target=feed, daemon=True)
    t.start()
    angles, _, _ = c.get_state()
    assert len(angles) == 6
    assert abs(angles[4] - 0.0) < 1e-6
    assert abs(angles[0] - 90.0) < 1e-6
    c.close()
    s.close()


def test_get_ee_parse():
    from gesture_mapping.arm_client import ArmClient
    import serial as _s
    s = _s.serial_for_url("loop://", baudrate=115200, timeout=0.1)
    c = ArmClient("loop://", ser=s)

    def feed():
        time.sleep(0.05)
        s.write(b"EE:0.50,0.10,0.30,0.00,0.00,0.00\n")

    t = threading.Thread(target=feed, daemon=True)
    t.start()
    ee = c.get_ee()
    assert ee == [0.5, 0.1, 0.3]
    c.close()
    s.close()
