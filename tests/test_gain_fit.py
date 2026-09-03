"""Tests for calibrate_gain.fit_all_gains (origin-passing least squares)."""
import numpy as np
from gesture_mapping.calibrate_gain import fit_all_gains, _GAIN_MIN, _GAIN_MAX


def _mk_collected(gain_true, n_poses=6, noise=0.05, motion=1.0):
    """合成 N 个姿势: a = 各关节角度, b = gain_true * a + noise."""
    rng = np.random.default_rng(7)
    collected = []
    for i in range(n_poses):
        a = rng.uniform(motion, 2.0, size=16) * (-1 if i % 2 else 1)
        b = gain_true * a + rng.normal(0, noise, size=16)
        collected.append((f"pose{i}", a, b))
    return collected


def test_fit_recovers_true_gain():
    """过原点最小二乘应恢复接近真实的增益."""
    gain_true = np.array([0.5, 1.5, 2.0, 0.8, 1.2, 0.4, 1.7, 1.0,
                          0.9, 2.2, 0.6, 1.3, 1.8, 0.7, 1.4, 2.5])
    collected = _mk_collected(gain_true)
    gains, errs = fit_all_gains(collected)
    assert np.allclose(gains, gain_true, atol=0.15), f"{gains} vs {gain_true}"
    assert np.all(errs < 0.15)


def test_fit_zero_gain_for_no_motion():
    """关节无运动 (|a|<min_motion) → gain 保持 1.0 而非噪声主导."""
    gain_true = np.ones(16) * 2.0
    gain_true[3] = 0.0            # ID3 不动
    collected = _mk_collected(gain_true)
    # 强制 ID3 的 a 在所有姿势都接近 0
    mask = np.ones(16)
    mask[3] = 0.01
    collected = [(n, a * mask, b) for n, a, b in collected]
    gains, _ = fit_all_gains(collected)
    assert gains[3] == 1.0        # 保持默认
    assert np.allclose(gains[[0, 1, 2, 4]], [2.0] * 4, atol=0.3)


def test_gain_clipped_to_bounds():
    """拟合增益 clip 到 [_GAIN_MIN, _GAIN_MAX]."""
    # 极端数据: a 很小 b 很大 → gain 应被 clip 到上限
    collected = []
    for i in range(5):
        a = np.ones(16) * 0.2
        b = np.ones(16) * 10.0
        collected.append((f"p{i}", a, b))
    gains, _ = fit_all_gains(collected, min_motion=0.1)
    assert np.all(gains <= _GAIN_MAX + 1e-9)
    assert np.all(gains >= _GAIN_MIN - 1e-9)
    assert np.allclose(gains, _GAIN_MAX, atol=0.2)


def test_fit_requires_two_points():
    """不足 2 个有效点 → gain 保持 1.0."""
    a = np.ones(16) * 1.0
    b = np.ones(16) * 2.0
    collected = [("p0", a, b)]   # 只有 1 组
    gains, _ = fit_all_gains(collected)
    assert np.all(gains == 1.0)
