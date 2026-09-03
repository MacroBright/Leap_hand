# 🔌 LEAP Hand 硬件配置与接线指南

本指南详细说明 LEAP Hand 16-DOF 灵巧手的硬件连接、电气规范、舵机分配表以及 Linux 串口权限配置。

---

## 1. 硬件规格参数

| 参数项 | 规格标准 | 说明 |
| :--- | :--- | :--- |
| **执行器** | 16 × Dynamixel XC330-M288-T | Robotis 智能总线舵机 |
| **供电电压** | 12.0 V DC (推荐 12V 5A 稳压电源) | 电压过低将导致舵机低压保护或掉线 |
| **通信接口** | TTL 半双工异步串行通信 (3-Pin) | 配套 USB 转 TTL 串口适配板 (如 U2D2) |
| **波特率** | **4,000,000 bps (4 Mbps)** | 极速总线通信，单帧 16 舵机 SyncRead < 3ms |
| **控制模式** | 位置-电流限制混合模式 (Mode 5) | 限制最大电流防堵转烧结，默认限制 150~350mA |
| **整机自由度** | 16 DOF (4 指 × 4 自由度) | 右手构型 (食指/中指/无名指/拇指) |

---

## 2. 16 舵机 ID 与关节映射全表

LEAP Hand 采用菊花链（Daisy Chain）串联 16 个舵机，出厂分配 ID 0 ~ 15：

| 舵机 ID | 所属手指 | 关节运动 | 正方向定义 (正值) | 默认全开角度 (rad) | 备注 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 食指 (Index) | MCP 侧摆 (Abduction) | 向小指侧外摆 | 3.1155 | 基座旋转 |
| **1** | 食指 (Index) | MCP 弯曲 (Flexion) | 伸展 (向手心弯曲为负) | 4.6204 | 掌指主关节 |
| **2** | 食指 (Index) | PIP 弯曲 (Flexion) | 伸展 (向手心弯曲为负) | 3.2076 | 近端指间关节 |
| **3** | 食指 (Index) | DIP 弯曲 (Flexion) | 伸展 (向手心弯曲为负) | 1.5785 | 远端指间关节 |
| **4** | 中指 (Middle) | MCP 侧摆 (Abduction) | 向小指侧外摆 | 3.2413 | 基座旋转 |
| **5** | 中指 (Middle) | MCP 弯曲 (Flexion) | 伸展 (向手心弯曲为负) | 3.0618 | 掌指主关节 |
| **6** | 中指 (Middle) | PIP 弯曲 (Flexion) | 伸展 (向手心弯曲为负) | 3.1170 | 近端指间关节 |
| **7** | 中指 (Middle) | DIP 弯曲 (Flexion) | 伸展 (向手心弯曲为负) | 4.5927 | 远端指间关节 |
| **8** | 无名指 (Ring)* | MCP 侧摆 (Abduction) | 向小指侧外摆 | 3.1186 | 映射人手**小指** |
| **9** | 无名指 (Ring)* | MCP 弯曲 (Flexion) | 伸展 (向手心弯曲为负) | 3.0756 | 掌指主关节 |
| **10** | 无名指 (Ring)* | PIP 弯曲 (Flexion) | 伸展 (向手心弯曲为负) | 3.1799 | 近端指间关节 |
| **11** | 无名指 (Ring)* | DIP 弯曲 (Flexion) | 伸展 (向手心弯曲为负) | 4.6004 | 远端指间关节 |
| **12** | 拇指 (Thumb) | **MCP 弯曲 (Flexion)** | **弯向手心 (正值弯曲)** | 3.1186 | ⚠ **线序反向** |
| **13** | 拇指 (Thumb) | **MCP 侧摆 (Abduction)** | 向外伸展 | 1.5555 | ⚠ **线序反向** |
| **14** | 拇指 (Thumb) | PIP 弯曲 (Flexion) | 伸展 (向手心弯曲为负) | 4.6234 | 近端指间关节 |
| **15** | 拇指 (Thumb) | DIP 弯曲 (Flexion) | 伸展 (向手心弯曲为负) | 4.7247 | 远端指间关节 |

> [!CAUTION]
> ### ⚠ 拇指线序特殊反向说明（硬件关键）
> 四指（食指/中指/无名指）线序均为：`ID x = MCP 侧摆`, `ID x+1 = MCP 弯曲`。
> 但**拇指物理线序与其他指相反**：`ID 12 = MCP 弯曲`, `ID 13 = MCP 侧摆`！
> 此外，拇指 ID 12 的弯曲符号与其余指相反（正值弯向手心）。
> 本项目的 `leap_hand.vision.JointMapper` 与 `leap_hand.cli.control` 内部已对该硬件特性做了解耦和符号统一（`JOINT_DIR` 矩阵），外部无需手动反转。

---

## 3. 串口权限与 udev 规则配置

在 Linux 下使用 USB-TTL 连接 Dynamixel 舵机时，默认可能无串口访问权限。

### 3.1 临时授权
```bash
sudo chmod 666 /dev/ttyUSB0
```

### 3.2 永久授权（将用户加入 dialout 组）
```bash
sudo usermod -aG dialout $USER
# 重启或注销后生效
```

### 3.3 绑定固定设备别名（udev 规则，推荐）
为防止拔插 USB 后串口号在 `/dev/ttyUSB0` 与 `/dev/ttyUSB1` 之间漂移，可创建固定别名规则：
```bash
sudo bash -c 'cat <<EOF > /etc/udev/rules.d/99-leap-hand.rules
# FTDI / Robotis U2D2 USB Serial Converter
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6014", MODE="0666", SYMLINK+="leap_hand"
EOF'
sudo udevadm control --reload-rules && sudo udevadm trigger
```
配置后即可直接使用设备路径 `/dev/leap_hand`。

---

## 4. 硬件自检与物理限位扫描

机械手上电后，建议在首次调试时使用 `leap-diagnose` 扫描全部舵机的物理限位边界：
```bash
# 运行硬件物理限位扫描体检
leap-diagnose
```
该工具会在安全的零力矩状态下，引导用户手动轻推关节到机械挡块，记录实测最小/最大弧度并保存为 `configs/motor_limits.json`，为全系统提供底层的硬限位防爆安全保障。
