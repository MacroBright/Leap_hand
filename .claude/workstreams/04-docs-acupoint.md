# Window 4: 文档 & 穴位优化 (📝)

## Scope
| 文件/模块 | 用途 |
|-----------|------|
| `CLAUDE.md` | 项目总纲 (多窗口工作流 + 架构决策) |
| `README.md` | 项目 README |
| `docs/ARCHITECTURE.md` | 全系统架构文档 |
| `python/acupoint/acupoint_detector.py` | 穴位检测模型接口 |
| `python/acupoint/__init__.py` | 包导出 |
| `experiments/` | 实验记录模板 |

## Current State
- [x] LEAP Hand Python SDK 工作
- [x] 7 个手势姿势录制
- [ ] 穴位检测模型待优化
- [ ] CLAUDE.md 待更新
- [ ] 全系统架构文档

## Next Tasks (prioritized)
1. 🔴 更新 CLAUDE.md:
   - 添加项目概述 (中医按摩灵巧手)
   - 添加多窗口工作流表 (4 窗口)
   - 记录 5 个架构决策
   - 添加技术栈 (LEAP Hand + Arm-robot_VLA + SmolVLA)
2. 🔴 穴位检测接口封装:
   - 实验室现有姿态估计模型的接口化
   - 输入: 背部图像 → 输出: 穴位名称+坐标列表
   - 常用穴位: 大椎/肩井/肺俞/心俞/肝俞/脾俞/肾俞/命门
3. 🔴 定义穴位→按摩手法映射规则:
   - 大椎 → 按揉 (拇指施压)
   - 肩井 → 捏拿 (四指+拇指)
   - 肺俞 → 推揉 (掌根推)
4. 🟡 创建 docs/ARCHITECTURE.md
5. 🟡 实验追踪模板 (experiments/)
6. 🟢 与 Arm-robot_VLA 文档同步

## Interfaces
### Output → 所有窗口
- CLAUDE.md (项目总纲)
- 穴位检测接口规范

### Input ← 实验室
- 现有姿态估计模型

## Session Start
```
加载 .claude/workstreams/04-docs-acupoint.md，当前任务: [描述]
```

## References
- Arm CLAUDE.md: `/home/bright/office/Arm-robot_VLA/CLAUDE.md`
- Arm 架构文档: `/home/bright/office/Arm-robot_VLA/docs/ARCHITECTURE.md`
- 常用背部穴位图 (待补充)
