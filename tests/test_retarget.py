"""Tests for retarget_mapper.py (reach-fraction retarget solver)."""
import numpy as np
import pytest

from gesture_mapping.retarget_mapper import RetargetMapper
from gesture_mapping.retarget_obs import human_reach_obs
from gesture_mapping.leap_fk import LeapFK, _FINGER_ORDER

_F = _FINGER_ORDER


def _mk_obs_from_fk(rm, q):
    """把 FK 位形 q 的 reach 作为目标观测."""
    st = rm._fk_state(q)
    obs = {}
    for f in _F:
        frac, d = rm._frac_dir(*st[f], rm._straight[f])
        obs[f] = (frac, d)
    thumb = st["thumb"][1]
    primary = st["index"][1]
    pin = float(np.clip(1.0 - np.linalg.norm(thumb - primary) / rm._palm_width, 0, 1))
    return obs, pin


def _reach_err(rm, q, obs):
    st = rm._fk_state(q)
    return max(abs(rm._frac_dir(*st[f], rm._straight[f])[0] - obs[f][0]) for f in _F)


def test_convergence_recover_known_pose():
    """求解器应还原已知 FK 位形的指尖 reach (误差 < 0.02)."""
    rm = RetargetMapper()
    for a in (np.zeros(16),
              np.array([0, 0, 0, 0, 0, 1.2, 1.0, 0.5, 0, 0, 0, 0, 0.3, 0.2, 0.3, 0.2]),
              np.array([0, 1.5, 1.6, 1.2, 0, 1.5, 1.6, 1.2, 0, 1.5, 1.6, 1.2,
                        1.0, 0.8, 0.9, 0.5])):
        obs, pin = _mk_obs_from_fk(rm, a)
        rm.reset()
        q = rm.solve(obs, pinch_tgt=pin)
        assert _reach_err(rm, q, obs) < 0.02


def test_solve_realtime_budget():
    """单帧求解 ≤ 50ms (20Hz 预算内; 30fps=33ms 目标)."""
    rm = RetargetMapper()
    obs, pin = _mk_obs_from_fk(rm, np.zeros(16))
    rm.solve(obs, pinch_tgt=pin)   # 预热
    rm.reset()
    q = rm.solve(obs, pinch_tgt=pin)
    assert rm.last_solve_ms < 50.0, f"solve {rm.last_solve_ms:.0f}ms > 50ms"


def test_output_within_angle_bounds():
    """输出在 [-0.5, 2.8] (与直映 _ANGLE_* 一致)."""
    rm = RetargetMapper()
    obs, pin = _mk_obs_from_fk(rm, np.array([0, 1.5, 1.6, 1.2, 0, 0, 0, 0, 0, 0, 0, 0,
                                             1.0, 0.8, 0.9, 0.5]))
    q = rm.solve(obs, pinch_tgt=pin)
    assert q.min() >= -0.5 and q.max() <= 2.8


def test_obs_scale_invariant():
    """human_reach_obs 对点云尺度不变 (分数 + 基底方向)."""
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 1, size=(21, 3))
    o1 = human_reach_obs(pts)
    o2 = human_reach_obs(pts * 0.37)
    for f in o1:
        assert o1[f][0] == pytest.approx(o2[f][0], abs=1e-6)
