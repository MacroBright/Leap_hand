"""MediaPipe 21kp → 重定向规范观测 (reach-fraction 表述).

人手指尖的"弯曲程度"用 reach-fraction 描述, 使其与 LEAP 可比 (规避手指长度
比例差, LEAP 手指短于人手指):
  curl_fraction = |tip − base| / 手指伸直段长      (1 = 伸直, <1 = 弯曲/对掌)
  direction     = (tip − base) 归一化, 投影到 掌心基底 (mid_dir/lateral/normal)

求解器 retarget_mapper.py 用这两项 + 捏合项, 优化 LEAP 16 角使 FK 指尖
reach-fraction 与方向匹配人手目标。

供 retarget_mapper.py 与 measure_following.py (重定向对比) 使用。
"""

import numpy as np
from typing import Dict, Tuple

from .joint_mapper import JointMapper, Landmark

_FINGER_ORDER = ["index", "middle", "pinky", "thumb"]

# 人手指链 (base, ..., tip) — MediaPipe 索引
_CHAIN_MP: Dict[str, list] = {
    "index": [Landmark["INDEX_MCP"], Landmark["INDEX_PIP"],
              Landmark["INDEX_DIP"], Landmark["INDEX_TIP"]],
    "middle": [Landmark["MIDDLE_MCP"], Landmark["MIDDLE_PIP"],
               Landmark["MIDDLE_DIP"], Landmark["MIDDLE_TIP"]],
    "pinky": [Landmark["PINKY_MCP"], Landmark["PINKY_PIP"],
              Landmark["PINKY_DIP"], Landmark["PINKY_TIP"]],
    "thumb": [Landmark["THUMB_CMC"], Landmark["THUMB_MCP"],
              Landmark["THUMB_IP"], Landmark["THUMB_TIP"]],
}


def human_reach_obs(pts: np.ndarray) -> Dict[str, Tuple[float, np.ndarray]]:
    """人手点云 (21,3) → {finger: (curl_fraction, direction_basis(3,))}.

    direction_basis 在 掌心基底 (mid_dir, lateral, normal) 内 — 与 LEAP 的
    (+x, +y, +z) 对齐 (手指方向→+x, 侧摆→+y, 法线→+z)。
    """
    pts = np.asarray(pts, dtype=np.float64)
    _wrist, normal, mid_dir, lateral = JointMapper._palm_frame(pts)

    out = {}
    for f in _FINGER_ORDER:
        chain = _CHAIN_MP[f]
        base = pts[chain[0]]
        tip = pts[chain[-1]]
        straight = sum(float(np.linalg.norm(pts[chain[i + 1]] - pts[chain[i]]))
                       for i in range(len(chain) - 1))
        reach = tip - base
        nr = float(np.linalg.norm(reach))
        frac = nr / straight if straight > 1e-9 else 0.0
        frac = float(np.clip(frac, 0.0, 1.2))
        if nr > 1e-9:
            d = reach / nr
            d_basis = np.array([d @ mid_dir, d @ lateral, d @ normal])
        else:
            d_basis = np.zeros(3)
        out[f] = (frac, d_basis)
    return out


def human_pinch_gap(pts: np.ndarray, primary: str = "index") -> float:
    """拇指↔主捏合指尖 归一化间隙 (掌宽归一), 供捏合目标."""
    tip_thumb = pts[Landmark["THUMB_TIP"]]
    tip_primary = pts[_CHAIN_MP[primary][-1]]
    palm_width = float(np.linalg.norm(
        pts[Landmark["INDEX_MCP"]] - pts[Landmark["PINKY_MCP"]]))
    if palm_width < 1e-9:
        return 0.0
    return float(np.clip(1.0 - np.linalg.norm(tip_thumb - tip_primary) / palm_width,
                         0.0, 1.0))
