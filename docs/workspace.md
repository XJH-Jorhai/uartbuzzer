# Git 与并行研究安排

2026-10-29 已与两个研究任务逐一沟通并获确认。目录和环境保持原路径，研究继续，未执行 checkout/reset/stash/clean。

| 所有者 | 当前写入范围 | 首发处理 |
|---|---|---|
| 主维护任务 | 根 Git、README、忽略规则、MIT、CI、稳定版发行 | 从已发布 1.3.1 源码存档建立初始提交 |
| 优化 UART 8N1 音频编码 | `research/beam_integration/`、Beam 核心及测试、协调后的 desktop 最小适配、`.venv-beam/`、`results/beam_integration/` | 保留工作文件，未完成研究不提交；预览构建用 `release/beam_preview/` |
| 研究负载感知 UART 编码器 | `research/generative_uart/`、独立 CPU 环境、`results/generative_uart/<run>/` | 原路径继续，文献/物理一致性/预注册先于 GPU 训练；首发排除 |

`research/differentiable_uart/` 和 `research/uart8n1_round2/` 是既有研究参考。其稳定原创实现可在所有者给出可公开白名单后分别提交；包含音源、原论文和第三方作者代码的目录/ZIP不能整体公开。

共享工作区可能比公开 `main` 多出正在编辑的 Beam 内容。初始 Git index 由稳定源码快照填入，**不会用存档覆盖实际工作文件**。Git diff 显示这些未提交改动是预期状态，不要为“清理”而丢弃。

后续提交按“研究实现和测试 → 桌面最小适配”分开，逐项指定文件。新 worktree 须先确认路径、环境、冻结输入和进程安排。发行 EXE/ZIP 放 GitHub Releases；用户 AppData 中的歌单和长期转换文件不移动、不提交。
