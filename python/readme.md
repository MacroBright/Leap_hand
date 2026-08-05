# LEAP Hand 右手 Python SDK

> 中医按摩灵巧手 — 右手 16DOF (Dynamixel XC330-M288)

---

## 环境安装

### Ubuntu (conda)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate leap_hand
# 装包优先 conda install -c conda-forge; 渠道没有的用 pip
```

> 依赖已装于 `leap_hand` 环境: dynamixel-sdk 4.0.5, mediapipe 1.0.0, opencv-contrib 5.0.0, pyrealsense2, pyserial, sounddevice。
> sounddevice 需要 portaudio（已装于环境内）；如需重装: `conda install -c conda-forge portaudio`。

### Windows

```powershell
python -m venv venv
.\venv\Scripts\activate.ps1
pip install dynamixel_sdk numpy
```

---

## 硬件连接

1. 灵巧手接 5V 电源 (电机供电)
2. USB 线接 PC (通信, `/dev/ttyUSB0`)
3. 设置串口权限:

```bash
sudo chmod 666 /dev/ttyUSB0
# 或持久化:
sudo chmod 666 /dev/serial/by-id/*
```

> ⚠️ 使用 Python API 时不能同时打开 Dynamixel Wizard

---

## 脚本说明

| 脚本 | 用途 | 运行 |
|------|------|------|
| `main.py` | 核心库 — LeapNode 类、POSES 字典、OPEN_POSE | — (被导入) |
| `calibrate.py` | **校准 & 录制** — 重录全开位或姿势, 存到 `poses.json` | `python calibrate.py` |
| `interactive_control.py` | **交互控制** — 逐指控制、手势执行、状态查看 | `python interactive_control.py` |

---

### calibrate.py — 校准 & 录制

```bash
python calibrate.py
```

| 选项 | 功能 |
|------|------|
| **1. 校准** | 关扭矩 → 手摆全开位 → Enter → 保存 OPEN_POSE |
| **2. 录制** | 选已有姿势重录 或 自定义命名新姿势 → 保存 |

录制完一个姿势后回到列表继续选。选 `0` 返回主菜单。

---

### interactive_control.py — 交互控制

```bash
python interactive_control.py
```

#### 手势命令

| 命令 | 姿势 |
|------|------|
| `open` | 全开/平伸 |
| `fist` | 全握拳 |
| `half` | 半握 |
| `point` | 食指指 |
| `peace` | 比耶 |
| `ok` | OK手势 |
| `thumbup` | 竖拇指 |

#### 逐指控制

```
LEAP> index mcp 0.5    # 食指 MCP前后 向手心弯 0.5 rad
LEAP> middle all 0.0   # 中指全开
LEAP> ring dip -0.3    # 无名指 DIP 向手背伸展
LEAP> thumb side 0.3   # 拇指侧摆
```

格式: `<手指> <关节> <角度>`
- 手指: `index` / `middle` / `ring` / `thumb`
- 关节: `side` / `mcp` / `pip` / `dip` / `all`
- 角度: 正值=向手心弯曲, 负值=向手背伸展 (rad)

#### 其他命令

| 命令 | 功能 |
|------|------|
| `state` | 显示 16 关节 LEAP 角度、全开位偏差、电流 |
| `raw <ID> <LEAP角度>` | 直接写电机 LEAP 角度 |
| `poses` | 列出所有已录姿势 |
| `help` / `quit` | 帮助 / 退出 |

---

## 关节映射速查

| ID | 手指 | 关节 | 全开位 (LEAP rad) |
|----|------|------|-------------------|
| 0 | 食指 | MCP侧摆 | 3.12 |
| 1 | 食指 | MCP前后 | 4.62 |
| 2 | 食指 | PIP | 3.21 |
| 3 | 食指 | DIP | 1.58 |
| 4 | 中指 | MCP侧摆 | 3.24 |
| 5 | 中指 | MCP前后 | 3.06 |
| 6 | 中指 | PIP | 3.12 |
| 7 | 中指 | DIP | 4.59 |
| 8 | 无名指 | MCP侧摆 | 3.12 |
| 9 | 无名指 | MCP前后 | 3.08 |
| 10 | 无名指 | PIP | 3.18 |
| 11 | 无名指 | DIP | 4.60 |
| 12 | 拇指 | MCP侧摆 | 3.12 |
| 13 | 拇指 | MCP前后 | 1.56 |
| 14 | 拇指 | PIP | 4.62 |
| 15 | 拇指 | DIP | 4.72 |

---

## 数据文件

| 文件 | 用途 |
|------|------|
| `poses.json` | 录制姿势数据, `main.py` 启动时自动加载 |
| `calibration_offset.json` | 旧版偏移量 (已废弃) |

---

## 项目总纲

详见 `../CLAUDE.md` 和 `.claude/workstreams/`
