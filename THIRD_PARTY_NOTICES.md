# Third-party notices

The root MIT license covers project-original code, the original synthetic demo
composition and the project's own measurement data. It does not replace the
licenses of dependencies, literature, music or downloaded code.

The application uses Python, NumPy, SciPy, PySide6/Qt, pyqtgraph, pyserial,
sounddevice/PortAudio and cffi. Optional tools use pycaw/comtypes; PyInstaller is
a build dependency. Collected notices are in `desktop_assets/licenses/`;
dependency distributions retain their complete notices and copyright attribution.

Qt/PySide6 uses its LGPLv3 option. Full LGPLv3/GPLv3 texts and upstream
source/rebuild links accompany the application. The portable build keeps its
dynamic libraries replaceable in `_internal`. See
`desktop_assets/licenses/Qt_SOURCE_AND_REBUILD.txt`. The root MIT does not
convert Qt/PySide6 to MIT or remove LGPL obligations. ngspice is optional and
is not bundled with the application.

Original papers, author archives, NC-licensed references, commercial songs,
microphone captures and research ZIPs containing these are excluded from the
initial repository and public release assets.

Public Windows builds collect the standard PortAudio library used for
MME/DirectSound/WDM/WASAPI, excluding the unused ASIO-enabled variant. Additional
Python, PortAudio, OpenSSL, libffi, expat and zlib notices and their upstream
source URLs are retained in `desktop_assets/licenses/`. Rebuild with the supplied
`scripts/hooks/hook-sounddevice.py`; retain those notices when redistributing.

Numba 0.62.1 and llvmlite 0.45.1 are bundled with their BSD and third-party
notices, including LLVM notices. The exact selector helper is imported from
Numba, not relicensed as project-original code. The build checks its source hash.
No downloaded research papers, transcription models or their weights are bundled.
