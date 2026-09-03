# 🐍 LEAP Hand Python SDK API 参考与开发指南

`leap_hand` 提供简洁、面向对象且高度模块化的 Python API，支持从底层舵机总线通信到高层视觉手势重定向的全栈开发。

---

## 1. 快速导入与基本使用

```python
from leap_hand import LeapHand, OPEN_POSE, POSES
import time

# 1. 连接机械手 (自动扫描串口，默认限制最大电流 150mA 保护)
hand = LeapHand(port="/dev/ttyUSB0", curr_lim=150)

# 2. 读取当前 16 舵机实时弧度
current_pos = hand.read_pos()
print("当前关节弧度:", current_pos)

# 3. 运行预设姿态
hand.set_pose("全握拳")
time.sleep(1.0)

# 4. 复位回全开位并断开
hand.set_open()
hand.disconnect()
```

---

## 2. 核心类 API 参考

### 2.1 `LeapHand` (`LeapNode`) — 灵巧手高层控制器
位于 `leap_hand.controller` 或直接从 `leap_hand` 导出。

#### 初始化
```python
hand = LeapHand(
    port: Optional[str] = None,       # 串口路径，None 时自动探测 /dev/ttyUSB0 /dev/ttyUSB1
    calib_mode: bool = False,         # 为 True 时即使标定无效也允许连接 (供校准工具使用)
    kP: int = 300,                    # 位置环比例增益 (四指基座自动降至 75% 增加柔顺度)
    kI: int = 0,                      # 积分增益
    kD: int = 100,                    # 微分增益
    curr_lim: int = 150               # 最大电流限制 (mA)，建议 150~350mA
)
```

#### 控制方法
- **`set_leap(pose: Sequence[float])`**  
  直接向 16 个舵机发送绝对目标角度（弧度制）。
- **`set_pose(name: str)`**  
  从 `poses.json` 中查找并执行姿态名（如 `"全开/平伸"`, `"半握"`, `"全握拳"`, `"OK手势"` 等）。
- **`set_open()`**  
  复位到真机标定的 `OPEN_POSE`。
- **`set_joint(motor_id: int, relative_angle: float)`**  
  以全开位为零基准，单独转动指定舵机（自动通过 `angle_safety_clip` 进行安全防过卷裁剪）。
- **`set_finger(finger_start_id: int, relative_angles: Sequence[float])`**  
  控制整根手指的 4 个连续关节（食指起点 0，中指 4，无名指 8，拇指 12）。

#### 状态读取方法
- **`read_pos() -> np.ndarray`**：返回长度 16 的实时角度数组（rad）。
- **`read_vel() -> np.ndarray`**：返回长度 16 的实时角速度（rad/s）。
- **`read_cur() -> np.ndarray`**：返回长度 16 的实时驱动电流（mA）。
- **`pos_vel_eff_srv() -> Tuple[np.ndarray, np.ndarray, np.ndarray]`**：在单次 4Mbps USB 事务中同时读取位置、速度与电流。
- **`disconnect()`**：释放总线端口并退出。

---

### 2.2 `JointMapper` — 空间关键点映射引擎
位于 `leap_hand.vision`。

```python
from leap_hand.vision import JointMapper
import numpy as np

mapper = JointMapper()

# 输入 MediaPipe / HaMeR 提取的 (21, 3) 关键点坐标 (米或毫米均可)
dummy_hand_3d = np.zeros((21, 3))
angles_16 = mapper.map_points_to_leap(dummy_hand_3d)

print("解算出的 16 舵机目标弧度:", angles_16)
```

---

### 2.3 `LEAPHandFK` — 16-DOF 正向运动学引擎
位于 `leap_hand.kinematics`。

提供基于 LEAP Hand URDF 几何链的指尖空间坐标正解（纯 NumPy 实现，零外部硬件依赖）。

```python
from leap_hand.kinematics import LEAPHandFK
import numpy as np

fk = LEAPHandFK()

# 输入 16 关节相对角度 (0 为全开)
q = np.zeros(16)
tips = fk.fingertip_positions(q)

print("食指指尖笛卡尔坐标 (相对于手掌基座):", tips["index"])
print("拇指指尖笛卡尔坐标 (相对于手掌基座):", tips["thumb"])
```

---

### 2.4 `OneEuroFilter` — 自适应低延迟防抖滤波
位于 `leap_hand.kinematics`。

```python
from leap_hand.kinematics import OneEuroFilter
import numpy as np

filter_1e = OneEuroFilter(n_joints=16, min_cutoff=1.0, beta=0.007)

# 在高频控制循环中调用
raw_angles = np.random.randn(16)
smooth_angles = filter_1e(raw_angles)
```
