"""LEAP Hand V1 16-DOF forward kinematics — numpy, zero dependencies.

Geometry reference: LEAP hand URDF (palm_lower-rooted hand tree, 16 revolute
joints joint_0..joint_15, all axis = local z). The constants below are the
joint origin (xyz/rpy) / limits — geometric facts from the URDF; the FK code
is self-written. No runtime URDF file needed.

Key convention (verified on the URDF): **q = 0 is the fully open hand** —
fingers straight, spread, thumb abducted. So our pipeline's relative angles
(0 = open, positive = flexed, same convention as JointMapper output) map
1:1 to FK joint values: q[i] = angle[i]. No zero-point calibration required.

Input convention: 16 relative angles in JointMapper order
(index 0-3, middle 4-7, pinky→LEAP ring 8-11, thumb 12-15;
 12=mcp, 13=side). Positive = flexion / abduction as defined by the URDF.

Used by:
  - measure_following.py  (四误差 "跟手" 评估: 直映 vs 重定向)
  - retarget_mapper.py   (Phase 3 重定向: 指尖位姿目标优化)

⚠ 拇指对掌限制 (2026-08-10 实测): 该 URDF 的拇指链 (j12-j15) 无法把拇指尖
跨到手指侧 (+x) — 全关节限位网格搜索, 拇指尖距食指尖最小仍 ~0.167m (≈开位
距离)。可能原因是此 panda 装配 URDF 的拇指为简化模型, 或对掌靠真实机上的
其他运动学。Phase 3 重定向若要表达捏合, 需换官方 LEAP_Hand URDF 或真机标定
拇指对掌运动学; Phase 1 四误差评估 (相对对比) 不受影响。

Run `python -m gesture_mapping.leap_fk` for a self-test.
"""

import numpy as np
from typing import Dict, List, Tuple

_BASE_LINK = "palm_lower"

# joint_id → (parent_link, child_link, origin_xyz, origin_rpy, (limit_lower, limit_upper))
# rpy = roll-pitch-yaw (rad); all joint axes = local (0,0,-1)
_JOINTS: Dict[int, Tuple[str, str, List[float], List[float], Tuple[float, float]]] = {
    0:  ("mcp_joint",  "pip",           [-0.0122, 0.0381, 0.0145], [-np.pi/2, 0.0, np.pi/2],  (-1.047, 1.047)),
    1:  ("palm_lower", "mcp_joint",     [-0.0070953, 0.0230578, -0.0187224], [np.pi/2, np.pi/2, 0.0], (-0.314, 2.230)),
    2:  ("pip",        "dip",           [0.015, 0.0143, -0.013], [np.pi/2, -np.pi/2, 0.0],   (-0.506, 1.885)),
    3:  ("dip",        "fingertip",     [0.0, -0.0361, 0.0002], [0.0, 0.0, 0.0],             (-0.366, 2.042)),
    4:  ("mcp_joint_2","pip_2",         [-0.0122, 0.0381, 0.0145], [-np.pi/2, 0.0, np.pi/2],  (-1.047, 1.047)),
    5:  ("palm_lower", "mcp_joint_2",   [-0.0070953, -0.0223922, -0.0187224], [np.pi/2, np.pi/2, 0.0], (-0.314, 2.230)),
    6:  ("pip_2",      "dip_2",         [0.015, 0.0143, -0.013], [np.pi/2, -np.pi/2, 0.0],   (-0.506, 1.885)),
    7:  ("dip_2",      "fingertip_2",   [0.0, -0.0361, 0.0002], [0.0, 0.0, 0.0],             (-0.366, 2.042)),
    8:  ("mcp_joint_3","pip_3",         [-0.0122, 0.0381, 0.0145], [-np.pi/2, 0.0, np.pi/2],  (-1.047, 1.047)),
    9:  ("palm_lower", "mcp_joint_3",   [-0.0070952, -0.0678422, -0.0187224], [np.pi/2, np.pi/2, 0.0], (-0.314, 2.230)),
    10: ("pip_3",      "dip_3",         [0.015, 0.0143, -0.013], [np.pi/2, -np.pi/2, 0.0],   (-0.506, 1.885)),
    11: ("dip_3",      "fingertip_3",   [0.0, -0.0361, 0.0002], [0.0, 0.0, 0.0],             (-0.366, 2.042)),
    12: ("palm_lower", "pip_4",         [-0.0693952, -0.0012422, -0.0216224], [0.0, np.pi/2, 0.0], (-0.349, 2.094)),
    13: ("pip_4",      "thumb_pip",     [0.0, 0.0143, -0.013], [np.pi/2, -np.pi/2, 0.0],     (-0.470, 2.443)),
    14: ("thumb_pip",  "thumb_dip",     [0.0, 0.0145, -0.017], [-np.pi/2, 0.0, 0.0],          (-1.200, 1.900)),
    15: ("thumb_dip",  "thumb_fingertip", [0.0, 0.0466, 0.0002], [0.0, 0.0, np.pi],          (-1.340, 1.880)),
}

# fingertip link per finger (order matches JointMapper output: index/middle/pinky/thumb)
_FINGER_TIPS: Dict[str, str] = {
    "index": "fingertip",
    "middle": "fingertip_2",
    "pinky": "fingertip_3",
    "thumb": "thumb_fingertip",
}
_FINGER_ORDER = ["index", "middle", "pinky", "thumb"]

# per-finger joint IDs in KINEMATIC order (palm_lower → tip), not numeric order
_FINGER_CHAIN: Dict[str, List[int]] = {
    "index": [1, 0, 2, 3],
    "middle": [5, 4, 6, 7],
    "pinky": [9, 8, 10, 11],
    "thumb": [12, 13, 14, 15],
}


def _rot_z(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot(rpy: np.ndarray) -> np.ndarray:
    """Rz@Ry@Rx euler → rotation matrix (URDF rpy convention)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class LeapFK:
    """Forward kinematics for LEAP Hand V1 (16 revolute joints, palm_lower root).

    angles: (16,) relative joint angles — 0 = fully open (URDF q=0), positive
    = flexed/abducted. Order = JointMapper order (0-3 index, 4-7 middle,
    8-11 pinky→LEAP ring, 12-15 thumb with 12=mcp, 13=side).
    """

    def __init__(self):
        # adjacency for BFS FK
        self._adj: Dict[str, List[Tuple[int, str]]] = {}
        for jid, (par, chi, *_rest) in _JOINTS.items():
            self._adj.setdefault(par, []).append((jid, chi))
            self._adj.setdefault(chi, []).append((jid, par))

    def _forward(self, angles: np.ndarray, target_link: str) -> np.ndarray:
        """Position of a link frame origin in the palm_lower frame (BFS)."""
        angles = np.asarray(angles, dtype=np.float64)
        from collections import deque
        dq = deque([(_BASE_LINK, np.eye(4))])
        seen = {_BASE_LINK}
        while dq:
            lnk, T = dq.popleft()
            if lnk == target_link:
                return T[:3, 3].copy()
            for jid, chi in self._adj.get(lnk, []):
                if chi in seen:
                    continue
                _par, _c, xyz, rpy, _lim = _JOINTS[jid]
                Tj = np.eye(4)
                Tj[:3, :3] = _rot(np.asarray(rpy, dtype=np.float64))
                Tj[:3, 3] = xyz
                Rq = np.eye(4)
                Rq[:3, :3] = _rot_z(angles[jid])
                seen.add(chi)
                dq.append((chi, T @ Tj @ Rq))
        raise KeyError(f"link {target_link} not reachable from {_BASE_LINK}")

    def fingertip_positions(self, angles: np.ndarray) -> Dict[str, np.ndarray]:
        """Return per-finger fingertip 3D positions (palm_lower frame)."""
        return {f: self._forward(angles, lnk) for f, lnk in _FINGER_TIPS.items()}

    def fingertip_matrix(self, angles: np.ndarray) -> np.ndarray:
        """Return (4, 3) fingertip positions, rows = index/middle/pinky/thumb."""
        return np.array([self.fingertip_positions(angles)[f] for f in _FINGER_ORDER])

    def joint_positions(self, angles: np.ndarray) -> Dict[str, np.ndarray]:
        """All link origins in the palm_lower frame (wrist/MCP/PIP/DIP/tips)."""
        links = set()
        for par, chi, *_ in _JOINTS.values():
            links.add(par)
            links.add(chi)
        return {lnk: self._forward(angles, lnk) for lnk in sorted(links)}

    def finger_joint_positions(self, angles: np.ndarray, finger: str) -> List[np.ndarray]:
        """Positions of the 4 joints of one finger, in order: [MCP, PIP, DIP, TIP].
        (child link origin of each joint in the kinematic chain)."""
        joints = _FINGER_CHAIN[finger]
        return [self._forward(angles, _JOINTS[jid][1]) for jid in joints]

    def mcp_positions(self, angles: np.ndarray) -> Dict[str, np.ndarray]:
        """Per-finger proximal (MCP/CMC) link positions — reach reference."""
        mcp_link = {"index": "mcp_joint", "middle": "mcp_joint_2",
                    "pinky": "mcp_joint_3", "thumb": "pip_4"}
        return {f: self._forward(angles, lnk) for f, lnk in mcp_link.items()}

    def finger_chain_positions_fast(self, angles: np.ndarray, finger: str) -> List[np.ndarray]:
        """[MCP, PIP, DIP, TIP] via direct chain composition (no BFS).

        ~16× faster than finger_joint_positions (retarget 求解器热路径).
        """
        angles = np.asarray(angles, dtype=np.float64)
        T = np.eye(4)
        out = []
        for jid in _FINGER_CHAIN[finger]:
            _par, _chi, xyz, rpy, _lim = _JOINTS[jid]
            Tj = np.eye(4)
            Tj[:3, :3] = _rot(np.asarray(rpy, dtype=np.float64))
            Tj[:3, 3] = xyz
            Rq = np.eye(4)
            Rq[:3, :3] = _rot_z(angles[jid])
            T = T @ Tj @ Rq
            out.append(T[:3, 3].copy())
        return out

    def finger_straight_length(self, finger: str) -> float:
        """Finger straight length = |MCP(q=0) - tip(q=0)| (reach 归一化基准)."""
        a0 = np.zeros(16)
        return float(np.linalg.norm(self.mcp_positions(a0)[finger]
                                    - self.fingertip_positions(a0)[finger]))

    @staticmethod
    def limits() -> np.ndarray:
        """(16, 2) URDF joint limits for reference."""
        return np.array([_JOINTS[i][4] for i in range(16)])


def _selftest():
    """Sanity checks — run with `python -m gesture_mapping.leap_fk`."""
    fk = LeapFK()
    a0 = np.zeros(16)
    tips0 = fk.fingertip_positions(a0)
    # open hand: standard fingers extended +x; thumb abducted (its own direction)
    for f in ("index", "middle", "pinky"):
        p = tips0[f]
        assert p[0] > 0.05, f"{f} open tip not extended: {p}"
    assert np.linalg.norm(tips0["thumb"]) > 0.05, "thumb open tip too close to palm"
    # flexion curls fingertips toward palm: x shrinks
    a1 = np.zeros(16)
    for i in [1, 2, 3, 5, 6, 7, 9, 10, 11]:
        a1[i] = 1.5
    a1[12] = 1.3
    a1[14] = 1.0
    a1[15] = 0.6
    t1 = fk.fingertip_positions(a1)
    for f in ("index", "middle", "pinky"):
        assert t1[f][0] < tips0[f][0] - 0.05, f"{f} did not curl in flexion"
    print("[leap_fk] self-test PASSED")
    print("  open tips:", {k: np.round(v, 3).tolist() for k, v in tips0.items()})
    print("  flex tips:", {k: np.round(v, 3).tolist() for k, v in t1.items()})


LEAPHandFK = LeapFK

if __name__ == "__main__":
    _selftest()

