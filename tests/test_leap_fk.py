"""Tests for leap_fk.py (LEAP V1 FK) and measure_following.py (四误差评估)."""
import numpy as np
import pytest

from gesture_mapping.leap_fk import LeapFK, _FINGER_ORDER, _FINGER_CHAIN
from gesture_mapping.measure_following import FollowEvaluator

_FINGER = _FINGER_ORDER


# ─── LeapFK 几何正确性 ─────────────────────────────────────────────

def test_open_hand_shape():
    """q=0 (全开): 标准指指尖沿 +x 伸展且张开, 拇指外展; 全部有限."""
    fk = LeapFK()
    tips = fk.fingertip_positions(np.zeros(16))
    for f in ("index", "middle", "pinky"):
        p = tips[f]
        assert p[0] > 0.05, f"{f} 开位未沿 +x 伸展"
        assert np.all(np.isfinite(p))
    assert np.linalg.norm(tips["thumb"]) > 0.05


def test_flexion_curls_fingers_toward_palm():
    """屈曲 (mcp/pip/dip +1.5) → 标准指尖 x 收缩 (卷向掌心)."""
    fk = LeapFK()
    a = np.zeros(16)
    for i in [1, 2, 3, 5, 6, 7, 9, 10, 11]:
        a[i] = 1.5
    t0 = fk.fingertip_positions(np.zeros(16))
    t1 = fk.fingertip_positions(a)
    for f in ("index", "middle", "pinky"):
        assert t1[f][0] < t0[f][0] - 0.05, f"{f} 屈曲未卷向掌心"


def test_thumb_flexion_moves_tip():
    """拇指 mcp 屈曲 → 拇指 TIP 位置显著改变 (FK 响应拇指驱动).
    注意: 该 URDF 拇指对掌范围有限 (见 leap_fk docstring), 不断言"到达食指",
    只断言屈曲确实移动了 TIP."""
    fk = LeapFK()
    a = np.zeros(16)
    a[12] = 1.5   # thumb mcp flex
    t0 = fk.fingertip_positions(np.zeros(16))
    t1 = fk.fingertip_positions(a)
    assert np.linalg.norm(t1["thumb"] - t0["thumb"]) > 0.03


def test_finger_chain_positions():
    """finger_joint_positions 返回 4 个关节 (MCP/PIP/DIP/tip) 且严格递增."""
    fk = LeapFK()
    for f in _FINGER:
        ps = fk.finger_joint_positions(np.zeros(16), f)
        assert len(ps) == 4
        for a, b in zip(ps, ps[1:]):
            assert np.linalg.norm(b - a) > 1e-3


# ─── measure_following 评估稳定性 ──────────────────────────────────

def test_metrics_for_pose_deterministic_and_finite():
    """同一人手点云两次评估 → 数值相同且全有限 (无 NaN)."""
    rng = np.random.default_rng(7)
    pts = rng.uniform(0, 1, size=(21, 3))
    ev = FollowEvaluator(angle_fn=lambda p: np.zeros(16))
    m1 = ev.metrics_for_pose(pts)
    m2 = ev.metrics_for_pose(pts)
    assert m1.keys() == m2.keys()
    for k in m1:
        assert np.isfinite(m1[k])
        assert m1[k] == pytest.approx(m2[k])


def test_open_hand_eval_machine_reference():
    """机器全开 (angle=0) 的规范化指尖: 标准指向前, 拇指反向 — 参考描述稳定."""
    ev = FollowEvaluator(angle_fn=lambda p: np.zeros(16))
    _w, lmid, llat, lnorm, lw = ev._leap_palm()
    tips = ev.fk.fingertip_matrix(np.zeros(16))
    canon = ev._canon(tips, np.zeros(3), (lmid, llat, lnorm), lw)
    # 标准指: 沿 +x (mid 分量显著为正); 拇指: 分量与手指反号 (外展)
    assert canon[0, 0] > 0.3 and canon[1, 0] > 0.3 and canon[2, 0] > 0.3
    assert canon[3, 0] < -0.3
