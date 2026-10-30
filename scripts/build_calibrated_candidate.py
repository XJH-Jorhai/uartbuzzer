"""Build 1.5.0 preview from source without modifying the input tree."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import re
from pathlib import Path
import shutil
import sys

VERSION='1.5.0-preview1'
SELECTOR_NUMBA_VERSION='0.62.1'
SELECTOR_SOURCE_SHA256='694eabf5ab6e67c4983fc5044494260f0f84714877af3c0090d44e8d0d133db5'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selector_attestation():
    dist=metadata.distribution('numba')
    source=Path(dist.locate_file('numba/np/arraymath.py'))
    actual=sha(source)
    if dist.version!=SELECTOR_NUMBA_VERSION or actual!=SELECTOR_SOURCE_SHA256:
        raise RuntimeError('Unverified private Numba selection dependency; re-review before building')
    return dict(numba_version=dist.version,arraymath_sha256=actual)


def collect_distribution_notices(name,dist,notices):
    """Keep wheel aggregate licenses and conflicting child notices separately."""
    for item in dist.files or []:
        if not any(s in item.name.lower() for s in ('license','copying','notice')):continue
        source=Path(dist.locate_file(item))
        if not source.is_file():continue
        parts=item.parts
        aggregate=(item.name.lower()=='license.txt' and
                   any(part.endswith('.dist-info') for part in parts) and
                   (len(parts)==2 or (len(parts)==3 and parts[-2]=='licenses')))
        if aggregate:
            destination=notices/(name+'_WHEEL_LICENSE.txt')
        else:
            destination=notices/(name+'_'+item.name)
            if destination.exists() and sha(destination)!=sha(source):
                path_id=hashlib.sha256(item.as_posix().encode()).hexdigest()[:16]
                destination=notices/(name+'_'+path_id+'_'+item.name)
        if destination.exists() and sha(destination)!=sha(source):
            raise RuntimeError('Conflicting license destination: '+destination.name)
        shutil.copy2(source,destination)


def prepare_assets(snapshot,root):
    assets=snapshot/'desktop_assets'
    attestation=selector_attestation()
    names=('music.py','nonlinear.py','uart_codec.py','impedance.py','desktop/engine.py','beam.py','beam_model.py','_beam_kernels.py')
    fingerprint=dict({Path(n).name:sha(snapshot/'src/uart_buzzer'/n) for n in names},numba_selector_dependency=attestation)
    (assets/'dsp_fingerprint.json').write_text(json.dumps(fingerprint,indent=2))
    (assets/'beam_backend.json').write_text(json.dumps(attestation,indent=2),encoding='utf-8')
    score_files=sorted((snapshot/'src/uart_buzzer/score').glob('*.py'))
    (assets/'score_fingerprint.json').write_text(json.dumps({p.name:sha(p) for p in score_files},indent=2),encoding='utf-8')
    version=(assets/'version_info.txt').read_text()
    version=re.sub(r'(filevers|prodvers)=\([^)]*\)',lambda m:m.group(1)+'=(1,5,0,1)',version)
    version=re.sub(r"StringStruct\('FileVersion',\s*'[^']*'\)","StringStruct('FileVersion', '1.5.0.1')",version)
    version=re.sub(r"StringStruct\('ProductVersion',\s*'[^']*'\)","StringStruct('ProductVersion', '"+VERSION+"')",version)
    if not re.search(r"StringStruct\('ProductVersion',\s*'"+re.escape(VERSION)+r"'\)",version):raise ValueError('Product version resource was not updated')
    (assets/'version_info.txt').write_text(version)
    (assets/'NOTICE.txt').write_text('UART Audio Studio '+VERSION+'\nIncludes Python, NumPy, SciPy, PySide6/Qt, pyqtgraph, pyserial, sounddevice, Numba and LLVM/llvmlite. See bundled licenses.\n',encoding='utf-8')
    notices=assets/'licenses';notices.mkdir(exist_ok=True)
    for name in ('numba','llvmlite','numpy','scipy','PySide6-Essentials','shiboken6','pyqtgraph','pyserial','sounddevice','cffi','pyinstaller'):
        dist=metadata.distribution(name)
        collect_distribution_notices(name,dist,notices)
    shutil.copy2(root/'LICENSE',notices/'PROJECT-MIT.txt')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,help='Source tree; defaults to this repository')
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--work',required=True,type=Path)
    p.add_argument('--onedir',action='store_true',help='Keep replaceable runtime DLLs in a portable directory')
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    source=(a.snapshot or root).resolve();out=a.out.resolve();work=a.work.resolve()
    if out.exists() or work.exists():raise FileExistsError('Use new output and work directories')
    if out==work or out.is_relative_to(work) or work.is_relative_to(out):raise ValueError('Separate output/work directories required')
    if any(source.is_relative_to(x) for x in (out,work)):raise ValueError('Output/work cannot contain source')
    attestation=selector_attestation()
    out.mkdir(parents=True);work.mkdir(parents=True)
    snapshot=work/'source';snapshot.mkdir()
    for name in ('src','desktop_assets','scripts/hooks'):
        shutil.copytree(source/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbc','*.nbi'))
    shutil.copy2(source/'desktop_app.py',snapshot/'desktop_app.py')
    prepare_assets(snapshot,source)
    assets=snapshot/'desktop_assets';notices=assets/'licenses'
    files={f.relative_to(snapshot).as_posix():sha(f) for f in sorted(snapshot.rglob('*')) if f.is_file()}
    source_manifest=json.dumps(files,sort_keys=True,indent=2)
    (snapshot/'SOURCE_MANIFEST.json').write_text(source_manifest,encoding='utf-8')
    os.environ['PATH']=os.pathsep.join([str(Path(sys.executable).parent),str(Path(sys.executable).parents[1]/'Lib/site-packages/PySide6'),str(Path(sys.executable).parents[1]/'Lib/site-packages/shiboken6'),str(Path(os.environ.get('SystemRoot',r'C:\Windows'))/'System32'),os.environ.get('SystemRoot',r'C:\Windows')])
    import PyInstaller.__main__
    cmd=[str(snapshot/'desktop_app.py'),'--name','UARTAudioStudio','--windowed','--onedir' if a.onedir else '--onefile','--noupx','--noconfirm',
         '--distpath',str(out),'--workpath',str(work/'pyinstaller'),'--specpath',str(work), '--paths',str(snapshot/'src'),
         '--icon',str(assets/'app.ico'),'--version-file',str(assets/'version_info.txt'),
         '--add-data',str(assets)+os.pathsep+'desktop_assets','--hidden-import','serial.tools.list_ports_windows',
         '--hidden-import','serial.serialwin32','--hidden-import','_cffi_backend','--collect-all','numba','--collect-all','llvmlite',
         '--additional-hooks-dir',str(snapshot/'scripts/hooks')]
    for name in ('pandas','IPython','jupyter','pytest','tkinter','PyQt5','PyQt6','matplotlib','numba.cuda','torch','tensorflow','OpenGL','PySide6.QtQml','PySide6.QtQuick','PySide6.QtWebEngineCore'):
        cmd+=['--exclude-module',name]
    PyInstaller.__main__.run(cmd)
    bundle=out/'UARTAudioStudio' if a.onedir else out
    exe=bundle/'UARTAudioStudio.exe'
    shutil.copytree(notices,bundle/'LICENSES')
    shutil.copy2(source/'LICENSE',bundle/'LICENSE.txt')
    shutil.copy2(source/'THIRD_PARTY_NOTICES.md',bundle/'THIRD_PARTY_NOTICES.md')
    deps={n:metadata.version(n) for n in ('numpy','scipy','PySide6-Essentials','shiboken6','pyqtgraph','pyserial','sounddevice','cffi','psutil','numba','llvmlite','pyinstaller')}
    manifest=dict(version=VERSION,packaging='onedir' if a.onedir else 'onefile',exe='UARTAudioStudio.exe',sha256=sha(exe),bytes=exe.stat().st_size,dependencies=deps,selector_attestation=attestation,source_manifest_sha256=sha(snapshot/'SOURCE_MANIFEST.json'),builder_sha256=sha(__file__),default_encoder='B0')
    (bundle/'build_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (bundle/'SOURCE_MANIFEST.json').write_text(source_manifest,encoding='utf-8')
    print('BUILD COMPLETE',exe,flush=True)


if __name__=='__main__':main()
