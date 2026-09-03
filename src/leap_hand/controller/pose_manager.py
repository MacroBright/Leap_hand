"""Pose management, validation, persistence and encoder unwrapping for LEAP Hand."""
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# ─── 默认实测姿势 (LEAP 角度, 单位 rad) ──────────────────────────
DEFAULT_POSES = {
    "全开/平伸": np.array([3.1155, 4.6204, 3.2076, 1.5785,
                          3.2413, 3.0618, 3.117,  4.5927,
                          3.1186, 3.0756, 3.1799, 4.6004,
                          3.1186, 1.5555, 4.6234, 4.7247]),

    "半握": np.array([3.1094, 4.559,  1.8132, -0.0614,
                     1.7579, 3.0434, 3.0588,  3.114,
                     3.1155, 3.0726, 1.8362,  2.8685,
                     3.1462, 1.511,  4.5421,  2.7811]),

    "全握拳": np.array([3.091,  4.0574, 1.3775, -0.1319,
                       1.4711, 2.4636, 3.094,   2.8624,
                       3.1155, 2.3869, 1.4603,  2.8808,
                       3.5343, 1.4972, 3.2122,  2.7826]),

    "食指指": np.array([3.1201, 4.7968, 3.1845, 1.5463,
                       1.4711, 2.3761, 3.0756, 2.8655,
                       3.1462, 2.3853, 1.4619, 2.8808,
                       3.5159, 1.4174, 3.1983, 2.7826]),

    "比耶": np.array([2.8624, 4.8198, 3.1845, 1.5463,
                     2.9974, 3.2628, 3.3441, 4.8167,
                     3.1447, 2.008,  1.5125, 2.8532,
                     3.5205, 1.4281, 3.2137, 2.7826]),

    "OK手势": np.array([3.1416, 3.4975, 1.9282, -0.0383,
                       2.9836, 3.229,  3.0634,  4.964,
                       3.0971, 2.9943, 3.4284,  4.6111,
                       2.8103, 1.1704, 3.3119,  2.7811]),

    "竖拇指": np.array([3.1401, 3.5113, 1.7135, -0.112,
                       1.399,  2.0862, 3.025,   3.4775,
                       3.1431, 1.9957, 1.6674,  3.2306,
                       3.0542, -0.0138, 4.7277, 5.4487]),
}

OPEN_POSE_DEFAULT = DEFAULT_POSES["全开/平伸"]


def get_default_config_path(filename: str = "poses.json") -> Path:
    """按优先级寻找配置文件的绝对路径."""
    candidates = [
        Path(__file__).resolve().parent.parent / "configs" / filename,
        Path(__file__).resolve().parents[3] / "configs" / filename,
        Path(__file__).resolve().parents[3] / "python" / filename,
        Path(__file__).resolve().parents[3] / "python" / "gesture_mapping" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def is_valid_pose(pose: np.ndarray) -> bool:
    """判断一组 LEAP 角度是否为真实电机读数 (排除读取失败回退的全零/常数/越界)."""
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (16,):
        return False
    if not np.all(np.isfinite(pose)):
        return False
    # 读取失败回退特征: 全零 或 16 个值完全相同
    if np.all(np.abs(pose) < 1e-6) or np.ptp(pose) < 1e-6:
        return False
    # 真实手可动作范围: 依据实测 motor_limits.json 全局边界
    if pose.min() < -2.5 or pose.max() > 8.5:
        return False
    return True


def unwrap_to_limits(pos: np.ndarray, limits_path: Optional[Path] = None) -> np.ndarray:
    """把读取到的 16 关节值对齐到实测限位圈 (消除编码器跨 0 点回绕).

    锚点 = 每关节 motor_limits 中点: 读数若与锚点相差 > π, 平移 ±2π 归位.
    """
    if limits_path is None:
        limits_path = get_default_config_path("motor_limits.json")

    if limits_path.exists():
        try:
            with open(limits_path, "r", encoding="utf-8") as f:
                lim = json.load(f)
            mid = (np.array(lim["min"], dtype=float) + np.array(lim["max"], dtype=float)) / 2.0
        except Exception:
            mid = np.array(pos, dtype=float)
    else:
        mid = np.array(pos, dtype=float)

    out = np.array(pos, dtype=float).copy()
    for i in range(16):
        diff = out[i] - mid[i]
        while diff > np.pi:
            out[i] -= 2.0 * np.pi
            diff -= 2.0 * np.pi
        while diff < -np.pi:
            out[i] += 2.0 * np.pi
            diff += 2.0 * np.pi
    return out


def load_poses(config_file: Optional[Path] = None) -> Tuple[Dict[str, np.ndarray], bool]:
    """加载姿势字典 (优先文件, 回退默认值). 返回 (poses_dict, is_valid)."""
    poses = {k: v.copy() for k, v in DEFAULT_POSES.items()}
    if config_file is None:
        config_file = get_default_config_path("poses.json")

    valid = True
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            for k, v in loaded.items():
                poses[k] = np.array(v, dtype=float)
            open_pose = poses.get("全开/平伸", OPEN_POSE_DEFAULT)
            if not is_valid_pose(open_pose):
                valid = False
        except Exception:
            valid = False
    return poses, valid


def save_poses(poses: Dict[str, np.ndarray], config_file: Optional[Path] = None) -> Path:
    """保存姿势字典到 JSON 文件."""
    if config_file is None:
        config_file = get_default_config_path("poses.json")
    config_file.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in poses.items()}
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    return config_file
