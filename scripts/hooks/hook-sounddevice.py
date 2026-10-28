"""Bundle only the standard PortAudio host APIs used by this application."""
from pathlib import Path
from PyInstaller.utils.hooks import get_module_file_attribute

folder=Path(get_module_file_attribute('sounddevice')).parent/'_sounddevice_data'/'portaudio-binaries'
destination='_sounddevice_data/portaudio-binaries'
binaries=[(str(folder/'libportaudio64bit.dll'),destination)]
datas=[(str(folder/'README.md'),destination)] if (folder/'README.md').exists() else []
