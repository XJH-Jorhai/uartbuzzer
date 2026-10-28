"""Reproducible Windows build; no external Python installation needed by users."""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for f in ('tmp/plot_deps','tmp/audio_deps','tmp/gui_deps','tmp/sim_deps','src'):
    sys.path.insert(0,str(ROOT/f))
import argparse
import subprocess
import shutil
import json
import importlib.metadata as metadata
import PyInstaller.__main__

# Do not resolve Windows system ICU against unrelated tools (e.g. Poppler) on PATH.
# Qt's Windows build imports the OS's unversioned ICU API, not ICU-78's renamed API.
os.environ['PATH']=os.pathsep.join([str(Path(sys.executable).parent),str(ROOT/'tmp/gui_deps/PySide6'),
        str(ROOT/'tmp/gui_deps/shiboken6'),str(Path(os.environ.get('SystemRoot',r'C:\Windows'))/'System32'),
        os.environ.get('SystemRoot',r'C:\Windows')])

p=argparse.ArgumentParser();p.add_argument('--onefile',action='store_true');args=p.parse_args()
kind='single' if args.onefile else 'portable'
work=ROOT/'tmp/desktop_build'/kind;dest=ROOT/'release'/kind
assert work.resolve().is_relative_to(ROOT.resolve()) and dest.resolve().is_relative_to(ROOT.resolve())
work.mkdir(parents=True,exist_ok=True);dest.mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,str(ROOT/'scripts/prepare_desktop_assets.py')],check=True,cwd=ROOT)
version=ROOT/'desktop_assets/version_info.txt'
version.write_text("""VSVersionInfo(ffi=FixedFileInfo(filevers=(1,3,1,0),prodvers=(1,3,1,0),mask=0x3f,flags=0x0,OS=0x40004,fileType=0x1,subtype=0x0,date=(0,0)),kids=[StringFileInfo([StringTable('040904B0',[StringStruct('CompanyName','ZipLab'),StringStruct('FileDescription','UART Audio Studio - WAV Serial Player'),StringStruct('FileVersion','1.3.1'),StringStruct('InternalName','UARTAudioStudio'),StringStruct('OriginalFilename','UARTAudioStudio.exe'),StringStruct('ProductName','UART Audio Studio'),StringStruct('ProductVersion','1.3.1')])]),VarFileInfo([VarStruct('Translation',[1033,1200])])])""",encoding='utf-8')
licenses=ROOT/'desktop_assets/licenses';licenses.mkdir(exist_ok=True)
for name in ('numpy','scipy','PySide6-Essentials','shiboken6','pyqtgraph','pyserial','sounddevice','cffi','pyinstaller'):
    try:
        d=metadata.distribution(name)
        for f in d.files or []:
            if any(word in f.name.lower() for word in ('license','copying','notice')) and f.suffix.lower() in ('','.txt','.md','.rst'):
                src=Path(d.locate_file(f))
                if src.is_file():
                    target=licenses/(name+'_'+f.name)
                    if not target.exists():shutil.copy2(src,target)
    except metadata.PackageNotFoundError:pass
cmd=[str(ROOT/'desktop_app.py'),'--name','UARTAudioStudio','--windowed','--noupx','--noconfirm',
     '--onefile' if args.onefile else '--onedir','--distpath',str(dest),'--workpath',str(work),'--specpath',str(work),
     '--icon',str(ROOT/'desktop_assets/app.ico'),'--version-file',str(version),
     '--add-data',str(ROOT/'desktop_assets')+os.pathsep+'desktop_assets',
     '--hidden-import','serial.tools.list_ports_windows','--hidden-import','serial.serialwin32',
     '--hidden-import','_cffi_backend']
if (ROOT/'scripts/hooks').is_dir():cmd+=['--additional-hooks-dir',str(ROOT/'scripts/hooks')]
for f in ('src','tmp/sim_deps','tmp/gui_deps','tmp/audio_deps'):
    cmd+=['--paths',str(ROOT/f)]
for name in ('matplotlib','pandas','IPython','jupyter','pytest','tkinter','PyQt5','PyQt6','PySide6.QtQml','PySide6.QtQuick','PySide6.QtWebEngineCore','OpenGL','numba'):
    cmd+=['--exclude-module',name]
PyInstaller.__main__.run(cmd)
print('BUILD COMPLETE:',dest)
