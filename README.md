# UART Audio Studio / CH343 UART 蜂鸣器播放器

使用 CH343 的 5 V TTL UART 8N1 字节流驱动电磁蜂鸣器的 Windows 播放器，以及实测复阻抗拟合、UART 编码和 SPICE 离线验证工具。

公开稳定版为 **1.3.1**。Beam/MPC 与可微分编码器仍在独立研究，尚未作为稳定功能发布。

## 下载和使用

在 [GitHub Releases](https://github.com/XJH-Jorhai/uartbuzzer/releases) 下载单文件 `UARTAudioStudio.exe`、可替换 Qt 动态库的 `UARTAudioStudio-portable.zip` 或对应代码 `UARTAudioStudio-source.zip`。

无需安装 Python、VS Code 或 ngspice。启动不自动连接串口或麦克风。

1. 导入 WAV 或文件夹，在文件库创建歌单。
2. 选择 CH343 串口并连接后播放。支持实时音量、静音、波形、FFT及麦克风录音。
3. 默认**单曲播放，播完停止**，可选顺序、列表循环、单曲循环或随机。

转换结果长期保存为文件并关联歌单，同一音频和配置再次载入不重新编码。数据默认位于 `%LOCALAPPDATA%\ZipLab\UART Audio Studio`，界面提供入口。原 WAV 保留原位置，移除列表引用不会删除它。

见 [应用说明](docs/desktop_app.md)、[歌单和缓存](docs/persistence.md)。

## 从源码运行

推荐 Python 3.12 x64，在项目根目录运行：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-gui.txt
.venv\Scripts\python desktop_app.py
```

内置原创多声部演示乐曲。支持整数或浮点 PCM WAV、中文路径，单首最长10分钟。

## 测试和构建

```powershell
.venv\Scripts\python scripts/test_desktop_only.py
.venv\Scripts\python scripts/run_tests.py
.venv\Scripts\python scripts/qa_library.py
.venv\Scripts\python scripts/build_desktop.py --onefile
.venv\Scripts\python scripts/build_desktop.py
```

完整测试包含 ngspice，需要安装 CLI 或设置 `NGSPICE_DLL` 指向兼容64位共享库。桌面运行、打包和第一条桌面/DSP测试不需要 ngspice，也不会打开串口或麦克风。稳定快照已通过74项软件检查及最终 EXE 自检和歌单重启验证。

离线研究入口 `python run.py --quick` 不打开串口。自有测量 CSV 在 `measurements/`，模型在 `desktop_assets/impedance_fit.json` 与 `spice/`。

## 项目布局

| 路径 | 内容 |
|---|---|
| `src/uart_buzzer/desktop/` | PySide6界面、歌单、转换文件、串口和录音 |
| `src/uart_buzzer/` | 音频处理、8N1编码、阻抗拟合、SPICE和评测 |
| `desktop_assets/` | 内置模型、配置、图标、第三方许可 |
| `measurements/` | 项目自有扫频和DCR CSV |
| `scripts/`, `tests/` | 稳定版构建、测试和验收 |
| `research/` | 本地并行研究，未完成内容未纳入首发 |
| `results/`, `release/`, `tmp/`, `.venv*/` | 本地产物，Git忽略 |

见 [协作规范](AGENTS.md) 和 [目录归属](docs/workspace.md)。公开内容不含用户原曲、录音、个人歌单、处理缓存、文献PDF和下载的作者源码包。

## 适用边界

稳定配置对应 CJ3401 高边、5V、55Ω限流、1Ω采样在续流回路内，Rg/Rgs=100Ω/10kΩ 的实验平台。M2 是实测小信号负载模型，预测线圈电流，不能当作声压预测。

音乐使用带限、宽带补偿和平滑响度控制，未以2kHz单音代替整曲验证。预失真候选跨曲验证未通过，默认关闭。连续载波增加休止期耗电，软件停止不替代失电保护。绝对SPL、全频高保真及任意硬件温升未认证。

## 许可

原创代码、自有测量数据采用 [MIT](LICENSE)。第三方按各自许可，见 [第三方声明](THIRD_PARTY_NOTICES.md)。MIT不重新授权第三方文献、下载代码或商业歌曲。
