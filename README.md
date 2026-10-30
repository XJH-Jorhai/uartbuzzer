# UART Audio Studio 1.4.1

用于 CH343 5 V TTL UART 8N1 电磁蜂鸣器的 Windows 播放器，包含 WAV 文件库、歌单、长期转换缓存、波形、FFT、录音与单曲校准。

从 [GitHub Releases](https://github.com/XJH-Jorhai/uartbuzzer/releases) 下载 `UARTAudioStudio.exe` 或 `UARTAudioStudio-portable.zip`。便携包解压后运行 EXE，保持 `_internal` 和 `LICENSES` 目录完整。启动不会自动打开串口或麦克风。内置演示乐曲为项目原创。

默认编码器为 B0。可选 Beam 快速档、高质量档通过相同设备电流模型优化 UART 字节。1.4.1 对 CPU 搜索做精确 TopW/SIMD 和计算复用：既有离线对照中快速档约 1.20–1.35 倍、高质量档约 1.50–1.62 倍；未缩小搜索或降低精度，首次 JIT 仍需时间。电流模型不是校准声压模型。公开包是删除私有来源记录、整理许可后的重新封装，具有新的二进制哈希；性能证据来自同一 DSP 源码的验收构建。

从源码运行（Windows x64、Python 3.12）：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-gui.txt
.venv\Scripts\python desktop_app.py
```

测试与构建：

```powershell
.venv\Scripts\python -m pip install -r requirements-test.txt
.venv\Scripts\python scripts/test_desktop_only.py
.venv\Scripts\python scripts/build_desktop.py --out dist/single-v141 --work build/single-v141
.venv\Scripts\python scripts/build_desktop.py --onedir --out dist/portable-v141 --work build/portable-v141
```

每次构建使用新的输出和工作目录。构建器复制源码后在工作目录准备资产，不改输入源码，不要求已有实验结果。`--snapshot` 可指定另一个完整源码目录。`requirements-build.txt` 提供单独构建依赖；测试命令不打开设备。Numba 及私有选择辅助函数有版本和源码 SHA 守卫，替换它们需重新验证；不支持时运行时保留原始搜索回退。

操作说明见 [应用说明](docs/desktop_app.md)。许可见 [第三方声明](THIRD_PARTY_NOTICES.md)。项目原创代码使用 MIT；Qt/PySide6、Numba、LLVM 等保持原许可。

可选离线阻抗/SPICE 工具：安装 `requirements-offline.txt` 后运行 `python run.py --quick`；完整 SPICE 测试另需 ngspice。`measurements/` 是项目自有测量数据，不包含音乐或录音。硬件播放与离线测试分开进行。
