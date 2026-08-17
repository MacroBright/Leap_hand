"""Retarget solver: optimize LEAP 16 angles so FK fingertip reach matches human.

参考 Mingrui-Yu retargeting 目标项公式 (借思路, 代码自写, 无 LICENSE 依赖)。
用 reach-fraction (见 retarget_obs.py) 规避 LEAP vs 人手指长比例差。

求解器: 手写 Levenberg-Marquardt, 数值 Jacobian (纯 numpy, 零新依赖)。
实时性: iters×16 列×1 FK-state ≈ 每帧 ≤ 160 次 FK-state (每次 = 4 指尖+4 MCP
forward), 每 state ≈ 数十 µs → 20-30ms/帧, 满足 20-30Hz 命令率。

目标残差 (越小越接近人手):
  per 指 f:  w_frac * (frac_FK - frac_tgt)           # 指尖弯曲度 (reach-fraction)
             w_dir  * (dir_FK - dir_tgt) (3 维)       # 指尖方向 (LEAP 掌心基底)
  捏合:      w_pinch * (pinch_FK - pinch_tgt)         # 拇指↔主指接近度
  关节正则:  w_reg  * (q - q_ref)                      # 时间正则 (向 prev/开位)

输出 16 相对角 (0=全开), 走共用下游 (JOINT_DIR/OPEN_POSE/motor_limits)。
"""

import time
from typing import Dict, Optional, Tuple

import numpy as np

from .leap_fk import LeapFK, _FINGER_ORDER

# 目标权重 (可调; 捏合权重高以突出重定向价值)
_W_FRAC = 1.0
_W_DIR = 0.5
_W_PINCH = 1.5
_W_REG = 0.05
_EPS = 1e-3           # 数值 Jacobian 步长
_MAX_ITERS = 6
_LAMBDA0 = 1e-2


class RetargetMapper:
    """16-DOF retargeting solver (reach-fraction objective + numpy LM)."""

    def __init__(self, fk: Optional[LeapFK] = None):
        self.fk = fk or LeapFK()
        self.limits = self.fk.limits()
        self._palm_width = None
        # 预热: 计算各指伸直长与 LEAP 掌宽
        a0 = np.zeros(16)
        self._straight = {f: self.fk.finger_straight_length(f) for f in _FINGER_ORDER}
        mcp0 = self.fk.mcp_positions(a0)
        self._palm_width = float(np.linalg.norm(
            mcp0["index"] - mcp0["pinky"]))
        self.prev_q: Optional[np.ndarray] = None
        self.last_solve_ms = 0.0

    # ── FK state: {finger: (mcp_pos, tip_pos)} (fast chain FK) ───
    def _fk_state(self, q: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        chain = {f: self.fk.finger_chain_positions_fast(q, f) for f in _FINGER_ORDER}
        return {f: (chain[f][0], chain[f][-1]) for f in _FINGER_ORDER}

    def _frac_dir(self, mcp: np.ndarray, tip: np.ndarray, straight: float):
        """(reach-fraction, 方向向量[LEAP +x/+y/+z]) from FK mcp/tip."""
        reach = tip - mcp
        nr = float(np.linalg.norm(reach))
        frac = nr / straight if straight > 1e-9 else 0.0
        return frac, (reach / nr if nr > 1e-9 else np.zeros(3))

    # ── 残差 ──────────────────────────────────────────────────────
    def _residual(self, q: np.ndarray, obs, pinch_tgt: float) -> np.ndarray:
        st = self._fk_state(q)
        r = []
        for f in _FINGER_ORDER:
            tfrac, tdir = obs[f]
            frac, d = self._frac_dir(*st[f], self._straight[f])
            r.append(_W_FRAC * (frac - tfrac))
            r.extend((_W_DIR * (d - tdir)).tolist())
        # 捏合: LEAP 拇指↔主指接近度 (掌宽归一) 匹配人手
        thumb = st["thumb"][1]
        primary = st["index"][1]
        pin = float(np.clip(1.0 - np.linalg.norm(thumb - primary) / self._palm_width,
                            0.0, 1.0))
        r.append(_W_PINCH * (pin - pinch_tgt))
        # 关节正则 (向 prev 或开位)
        ref = self.prev_q if self.prev_q is not None else np.zeros(16)
        r.extend((_W_REG * (q - ref)).tolist())
        return np.asarray(r, dtype=np.float64)

    # ── 求解 ──────────────────────────────────────────────────────
    def solve(self, obs, pinch_tgt: float = 0.0,
              prev: Optional[np.ndarray] = None) -> np.ndarray:
        """obs: retarget_obs.human_reach_obs 输出. 返回 16 相对角."""
        t0 = time.monotonic()
        q = np.clip(prev.copy() if prev is not None
                    else np.zeros(16), self.limits[:, 0], self.limits[:, 1])
        lam = _LAMBDA0
        r = self._residual(q, obs, pinch_tgt)
        cost = float(r @ r)
        for _ in range(_MAX_ITERS):
            # 数值 Jacobian (单侧差分, n×16)
            J = np.zeros((len(r), 16))
            for j in range(16):
                if j in (0, 4, 8, 12, 13):   # side 关节步长略大 (量程小)
                    eps = _EPS * 5
                else:
                    eps = _EPS
                qp = q.copy()
                qp[j] = min(q[j] + eps, self.limits[j, 1])
                J[:, j] = (self._residual(qp, obs, pinch_tgt) - r) / eps
            # LM 步
            H = J.T @ J
            g = J.T @ r
            for _inner in range(10):
                dq = np.linalg.solve(H + lam * np.eye(16), -g)
                qn = np.clip(q + dq, self.limits[:, 0], self.limits[:, 1])
                rn = self._residual(qn, obs, pinch_tgt)
                cn = float(rn @ rn)
                if cn < cost:
                    q, r, cost = qn, rn, cn
                    lam = max(lam * 0.5, 1e-4)
                    break
                lam *= 10.0
                if lam > 1e6:
                    break
        self.prev_q = q.copy()
        self.last_solve_ms = (time.monotonic() - t0) * 1000.0
        return np.clip(q, -0.5, 2.8)   # 输出限幅同直映 _ANGLE_* 边界

    def reset(self):
        self.prev_q = None
