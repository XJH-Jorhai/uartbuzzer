"""Native Qt Windows frontend; no web server, browser or VS Code dependency."""
import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly,butter,sosfilt
from fractions import Fraction
from PySide6 import QtCore,QtGui,QtWidgets
import pyqtgraph as pg
from .theme import STYLE
from .engine import convert,validate_profile,fit_recording,export_document
from .devices import Microphone,SerialLink,Playback,serial_devices,microphones
from .cache import ConversionCache,LibraryAudioCache
from .library_ui import LibraryController
from ..music import FS,music_fixture

VERSION='1.3.1'
ASSETS=Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parents[3]))/'desktop_assets'

def data_root(data_dir=None):
    if data_dir:return Path(data_dir).resolve()
    return Path(QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.GenericDataLocation))/'ZipLab'/'UART Audio Studio'


class Job(QtCore.QThread):
    result=QtCore.Signal(object)
    error=QtCore.Signal(str)
    progress=QtCore.Signal(float)
    telemetry=QtCore.Signal(object)
    def __init__(self,function,parent=None):super().__init__(parent);self.function=function
    def run(self):
        try:self.result.emit(self.function(self))
        except InterruptedError:self.error.emit('操作已取消')
        except Exception as e:
            logging.exception('Background task failed');self.error.emit(str(e))


def duration_text(seconds):
    return f'{int(seconds)//60:02d}:{int(seconds)%60:02d}'


def label(text,name=None):
    w=QtWidgets.QLabel(text)
    if name:w.setObjectName(name)
    return w


def button(text,slot=None,primary=False):
    b=QtWidgets.QPushButton(text)
    if primary:b.setObjectName('Primary')
    if slot:b.clicked.connect(slot)
    return b


class MainWindow(LibraryController,QtWidgets.QMainWindow):
    def __init__(self,data_dir=None):
        super().__init__();self.setWindowTitle('UART Audio Studio · 串口音乐工作台')
        available=QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.resize(min(1160,available.width()-24),min(820,available.height()-48));self.setMinimumSize(760,520)
        self.setAcceptDrops(True)
        self.root=data_root(data_dir)
        self.root.mkdir(parents=True,exist_ok=True)
        self.cache=LibraryAudioCache(self.root/'prepared_audio',self.root/'cache/conversions')
        logging.basicConfig(filename=str(self.root/'application.log'),level=logging.INFO,encoding='utf-8')
        self.settings=QtCore.QSettings(str(self.root/'settings.ini'),QtCore.QSettings.IniFormat)
        self.base_profile=validate_profile(json.loads((ASSETS/'music_profile.json').read_text(encoding='utf-8')))
        self.profile=dict(self.base_profile);self.fit_path=ASSETS/'impedance_fit.json'
        saved_profile=self.settings.value('profile_path')
        if saved_profile:
            try:self.profile=validate_profile(json.loads(Path(saved_profile).read_text(encoding='utf-8-sig')))
            except Exception as e:logging.warning('Saved profile unavailable: %s',e)
        if not self.profile.get('predistortion') and self.profile.get('available_predistortion'):self.profile['predistortion']=self.profile['available_predistortion']
        self.doc=None;self.jobs=[];self.loading=False;self.conversion_job=None;self.player=None;self.play_job=None
        self.link=SerialLink();self.mic=Microphone();self.position=0.;self.last_recording=None;self.current_path=None;self.current_source=None
        self.disconnect_pending=False;self.exit_pending=False;self.live_decoded=np.zeros(2048);self.live_decode_rate=FS
        self.decode_filter=None;self.decode_state=None
        self.init_library()
        self._build();self._load_settings();self.refresh_ports();self.refresh_mics();self._update_controls()
        self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.update_graphs);self.timer.start(100)
        self.log('就绪。导入 WAV 自动转换；选择串口后即可播放。')
        self.statusBar().showMessage('就绪  ·  未连接串口')
        self.log(f'数据目录：{self.root} · 歌单恢复 {len(self.library.ids("all"))} 首')
        if self.library.warning:self.fail(self.library.warning)

    def _build(self):
        self.setStyleSheet(STYLE)
        icon=QtGui.QIcon(str(ASSETS/'app.ico'))
        if not icon.isNull():self.setWindowIcon(icon)
        root=QtWidgets.QWidget();self.setCentralWidget(root);layout=QtWidgets.QHBoxLayout(root);layout.setContentsMargins(0,0,0,0);layout.setSpacing(0)
        sidebar=QtWidgets.QFrame();sidebar.setObjectName('Sidebar')
        sidebar_area=QtWidgets.QScrollArea();sidebar_area.setWidgetResizable(True);sidebar_area.setFixedWidth(238)
        sidebar_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff);sidebar_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        sidebar_area.setWidget(sidebar);layout.addWidget(sidebar_area)
        side=QtWidgets.QVBoxLayout(sidebar);side.setContentsMargins(14,14,14,12);side.setSpacing(6)
        side.addWidget(label('UART AUDIO','Brand'));side.addWidget(label('串口音乐工作台  /  DESKTOP','Muted'))
        side.addSpacing(7);side.addWidget(label('串口连接','SectionTitle'))
        ports_row=QtWidgets.QHBoxLayout();self.ports=QtWidgets.QComboBox();self.ports.setObjectName('portCombo');self.ports.setMinimumWidth(120)
        self.ports.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.ports.setToolTip('仅枚举端口，启动时不会自动打开串口')
        refresh=button('刷新',self.refresh_ports);refresh.setFixedWidth(58);ports_row.addWidget(self.ports,1);ports_row.addWidget(refresh);side.addLayout(ports_row)
        self.baud=QtWidgets.QComboBox();self.baud.addItem('460800 baud',460800);self.baud.addItem('921600 baud',921600)
        self.baud.currentIndexChanged.connect(self.processing_changed);side.addWidget(self.baud)
        side.addWidget(label('8 数据位  ·  无校验  ·  1 停止位','Muted'))
        self.connect_button=button('连接串口',self.toggle_serial,True);self.connect_button.setObjectName('Primary');side.addWidget(self.connect_button)
        self.connection=label('●  未连接','Muted');side.addWidget(self.connection)
        side.addWidget(label('音频处理','SectionTitle'))
        self.mode=QtWidgets.QComboBox();self.mode.addItem('清晰响亮  ·  音乐优化','music');self.mode.addItem('M2 电气模型补偿','electrical');self.mode.addItem('仅带限，不做补偿','bypass')
        self.mode.currentIndexChanged.connect(self.processing_changed);side.addWidget(self.mode)
        self.model_badge=label('M2 负载先验  /  音乐优化','Badge');side.addWidget(self.model_badge)
        side.addWidget(label('16 kHz  ·  400–5000 Hz  ·  保持音高','Muted'))
        self.loudness=QtWidgets.QCheckBox('音乐响度增强 · 平滑限幅');self.loudness.setChecked(True)
        self.loudness.setToolTip('前瞻限幅：瞬态不会压低整首音乐的音量，不硬截波形。')
        self.loudness.toggled.connect(self.processing_changed);side.addWidget(self.loudness)
        self.loudness_level=QtWidgets.QComboBox()
        for title,target in [('响度：标准',.32),('响度：增强',.52),('响度：强劲',.7)]:self.loudness_level.addItem(title,target)
        self.loudness_level.setCurrentIndex(min(range(3),key=lambda i:abs(self.loudness_level.itemData(i)-self.profile.get('loudness_target_rms',.52))))
        self.loudness_level.setToolTip('更高响度会收紧音乐动态；所有档位均保留平滑峰值限幅。')
        self.loudness_level.currentIndexChanged.connect(self.processing_changed);side.addWidget(self.loudness_level)
        self.continuous=QtWidgets.QCheckBox('连续载波 · 减少休止爆音');self.continuous.setChecked(True)
        self.continuous.setToolTip('音乐休止期间维持载波以减少平均驱动突变；静音、0%和停止后仍停发。载波会增加休止期间的耗电。')
        self.continuous.toggled.connect(self.processing_changed);side.addWidget(self.continuous)
        self.predistortion_box=QtWidgets.QCheckBox('歌曲预失真候选 · 可比较')
        self.predistortion_box.setChecked(bool(self.profile.get('predistortion')) and self.profile.get('predistortion_default_enabled',True))
        self.predistortion_box.setToolTip('有界二阶记忆补偿，随音量平方缩放。当前实测歌曲两段改善，跨曲未通过，默认关闭；导入其他歌曲会关闭，可手动比较。')
        self.predistortion_box.toggled.connect(self.processing_changed);side.addWidget(self.predistortion_box)
        advanced_button=QtWidgets.QToolButton();advanced_button.setText('配置与录音拟合 ▸');advanced_button.setCheckable(True);side.addWidget(advanced_button)
        advanced=QtWidgets.QWidget();advanced_layout=QtWidgets.QVBoxLayout(advanced);advanced_layout.setContentsMargins(0,0,0,0);advanced_layout.setSpacing(6)
        advanced.hide();side.addWidget(advanced)
        advanced_button.toggled.connect(lambda on:(advanced.setVisible(on),advanced_button.setText('配置与录音拟合 ▾' if on else '配置与录音拟合 ▸')))
        profile_row=QtWidgets.QHBoxLayout();self.import_profile_button=button('导入配置',self.import_profile);self.save_profile_button=button('导出配置',self.save_profile)
        profile_row.addWidget(self.import_profile_button);profile_row.addWidget(self.save_profile_button);advanced_layout.addLayout(profile_row)
        self.fit_button=button('用最近录音拟合补偿',self.fit_latest);self.fit_button.setToolTip('基于整段音乐的宽带正则拟合；需要完整、未暂停的同步播放录音')
        advanced_layout.addWidget(self.fit_button)
        side.addWidget(label('麦克风','SectionTitle'))
        self.mics=QtWidgets.QComboBox();self.mics.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon);side.addWidget(self.mics)
        self.mic_button=button('启用麦克风',self.toggle_mic);side.addWidget(self.mic_button)
        self.auto_record=QtWidgets.QCheckBox('播放时同步录音');self.auto_record.setChecked(True);side.addWidget(self.auto_record)
        self.record_button=button('开始独立录音',self.toggle_recording);side.addWidget(self.record_button)
        self.record_state=label('●  麦克风未启用','Muted');side.addWidget(self.record_state)
        side.addStretch(1)
        side.addWidget(button('打开录音与会话目录',self.open_sessions))
        side.addWidget(label(f'v{VERSION}  ·  本地处理，无需网络','Muted'))
        main=QtWidgets.QWidget();layout.addWidget(main,1);content=QtWidgets.QVBoxLayout(main);content.setContentsMargins(14,12,14,10);content.setSpacing(6)
        header=QtWidgets.QHBoxLayout();header.addWidget(label('音频工作区','Title'));header.addStretch();header.addWidget(label('PCM → UART 8N1','Badge'));content.addLayout(header)
        file_card=QtWidgets.QFrame();file_card.setObjectName('Card');file_layout=QtWidgets.QHBoxLayout(file_card);file_layout.setContentsMargins(12,9,12,9)
        details=QtWidgets.QVBoxLayout();self.file_label=label('导入一份 WAV，开始播放');self.file_label.setStyleSheet('font-weight:600; font-size:14px;')
        self.file_info=label('支持拖放文件 · 自动转换为串口数据 · 最长 10 分钟','Muted');details.addWidget(self.file_label);details.addWidget(self.file_info);file_layout.addLayout(details,1)
        self.demo_button=button('演示乐曲',self.load_demo);self.open_button=button('导入 WAV',self.open_file,True);file_layout.addWidget(self.demo_button);file_layout.addWidget(self.open_button);content.addWidget(file_card)
        self.banner=label('');self.banner.setWordWrap(True);self.banner.hide();content.addWidget(self.banner)
        conversion_row=QtWidgets.QHBoxLayout();self.convert_label=label('等待音频','Muted');self.conversion=QtWidgets.QProgressBar();self.conversion.setRange(0,1000);self.conversion.setValue(0);self.conversion.setTextVisible(False);self.conversion.setFixedHeight(4)
        self.cancel_button=button('取消转换',self.cancel_conversion);self.cancel_button.hide();conversion_row.addWidget(self.convert_label);conversion_row.addWidget(self.conversion,1);conversion_row.addWidget(self.cancel_button);content.addLayout(conversion_row)
        analysis=QtWidgets.QHBoxLayout();analysis.addWidget(label('信号监视','SectionTitle'));analysis.addStretch()
        self.signal_source=QtWidgets.QComboBox();self.signal_source.addItem('处理后音频','processed');self.signal_source.addItem('UART 数字解算','decoded');self.signal_source.addItem('麦克风实测','mic');self.signal_source.currentIndexChanged.connect(self.reset_plots)
        self.signal_source.setMinimumWidth(150);analysis.addWidget(self.signal_source)
        self.overlay=QtWidgets.QCheckBox('叠加源音频');analysis.addWidget(self.overlay)
        reset=button('复位视图',self.reset_plots);analysis.addWidget(reset);content.addLayout(analysis)
        pg.setConfigOptions(antialias=False,background='#202022',foreground='#9da7b0')
        self.wave=pg.PlotWidget();self.wave.setObjectName('waveformPlot');self.wave.setMinimumHeight(85)
        self.wave.showGrid(x=True,y=True,alpha=.13);self.wave.setLabel('left','幅值','FS');self.wave.setLabel('bottom','时间','s');self.wave.setYRange(-1,1,padding=.02)
        self.wave.setXRange(-.2,0,padding=0);self.wave_curve=self.wave.plot(pen=pg.mkPen('#4bb1ea',width=1.5));self.wave_raw=self.wave.plot(pen=pg.mkPen('#797f88',width=1))
        self.spectrum=pg.PlotWidget();self.spectrum.setObjectName('spectrumPlot');self.spectrum.setMinimumHeight(110)
        self.spectrum.showGrid(x=True,y=True,alpha=.13);self.spectrum.setLabel('left','幅度','dBFS');self.spectrum.setLabel('bottom','频率','Hz');self.spectrum.setYRange(-120,0);self.spectrum.setXRange(0,8000,padding=0)
        self.spec_curve=self.spectrum.plot(pen=pg.mkPen('#62b5e5',width=1.5),fillLevel=-140,brush=pg.mkBrush(31,101,150,35));self.spec_raw=self.spectrum.plot(pen=pg.mkPen('#777d87',width=1))
        splitter=QtWidgets.QSplitter(QtCore.Qt.Vertical);splitter.addWidget(self.wave);splitter.addWidget(self.spectrum);splitter.setSizes([200,230]);content.addWidget(splitter,1)
        self.analysis_hint=label('频谱为所选信号的实时 FFT；UART 数字解算不代表实测线圈电流。','Muted');content.addWidget(self.analysis_hint)
        transport=QtWidgets.QFrame();transport.setObjectName('Card');tl=QtWidgets.QVBoxLayout(transport);tl.setContentsMargins(14,10,14,10)
        timeline=QtWidgets.QHBoxLayout();self.time_label=label('00:00 / 00:00','Muted');self.timeline=QtWidgets.QProgressBar();self.timeline.setRange(0,10000);self.timeline.setTextVisible(False);self.timeline.setFixedHeight(5)
        timeline.addWidget(self.time_label);timeline.addWidget(self.timeline,1);self.byte_label=label('0 B','Muted');timeline.addWidget(self.byte_label);tl.addLayout(timeline)
        controls=QtWidgets.QHBoxLayout();self.play_button=button('▶  播放',self.toggle_play,True);self.play_button.setMinimumWidth(110)
        self.stop_button=button('■  停止',self.stop_play);self.stop_button.setObjectName('Danger');controls.addWidget(self.play_button);controls.addWidget(self.stop_button)
        self.export_button=button('导出串口数据',self.export_data);controls.addWidget(self.export_button);controls.addStretch()
        self.mute=QtWidgets.QToolButton();self.mute.setText('静音');self.mute.setCheckable(True);self.mute.toggled.connect(self.volume_changed);controls.addWidget(self.mute)
        self.volume=QtWidgets.QSlider(QtCore.Qt.Horizontal);self.volume.setObjectName('volumeSlider');self.volume.setRange(0,100);self.volume.setValue(80);self.volume.setFixedWidth(140);self.volume.valueChanged.connect(self.volume_changed);self.volume.setToolTip('实时音量；0% 停发静音，变化带平滑渐变')
        controls.addWidget(self.volume);self.volume_label=label('80%');self.volume_label.setFixedWidth(37);controls.addWidget(self.volume_label);tl.addLayout(controls);content.addWidget(transport)
        log_head=QtWidgets.QHBoxLayout();log_toggle=QtWidgets.QToolButton();log_toggle.setText('运行日志 ▸');log_toggle.setCheckable(True);log_head.addWidget(log_toggle);log_head.addStretch();clear=QtWidgets.QToolButton();clear.setText('清空');log_head.addWidget(clear);content.addLayout(log_head)
        self.log_view=QtWidgets.QPlainTextEdit();self.log_view.setReadOnly(True);self.log_view.setMaximumBlockCount(300);self.log_view.setFixedHeight(64);self.log_view.hide()
        log_toggle.toggled.connect(lambda on:(self.log_view.setVisible(on),log_toggle.setText('运行日志 ▾' if on else '运行日志 ▸')))
        clear.clicked.connect(self.log_view.clear);content.addWidget(self.log_view)
        self.statusBar().setSizeGripEnabled(True)
        QtGui.QShortcut(QtGui.QKeySequence('Space'),self,activated=self.toggle_play)
        QtGui.QShortcut(QtGui.QKeySequence('Escape'),self,activated=self.stop_play)
        QtGui.QShortcut(QtGui.QKeySequence('Ctrl+O'),self,activated=self.open_file)
        queue_controls=QtWidgets.QHBoxLayout();tl.addLayout(queue_controls);self.build_library(header,queue_controls)

    def _load_settings(self):
        self.volume.setValue(int(self.settings.value('volume',80)))
        baud=int(self.settings.value('baud',460800));self.baud.setCurrentIndex(max(0,self.baud.findData(baud)))
        geometry=self.settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)
            available=QtGui.QGuiApplication.primaryScreen().availableGeometry()
            self.resize(min(self.width(),available.width()-24),min(self.height(),available.height()-48))

    def ensure_visible(self):
        available=self.screen().availableGeometry()
        self.resize(min(self.width(),available.width()-24),min(self.height(),available.height()-48))
        self.move(available.x()+max(0,(available.width()-self.width())//2),available.y()+max(0,(available.height()-self.height())//2))

    def log(self,text):
        self.log_view.appendPlainText(datetime.now().strftime('%H:%M:%S')+'  '+str(text));logging.info(text)

    def fail(self,text):
        self.banner.setText(str(text));self.banner.setStyleSheet('background:#412c2b; color:#efb0a9; padding:8px; border:1px solid #69413b; border-radius:3px;');self.banner.show();self.log('提示：'+str(text));self._update_controls()

    def clear_banner(self):self.banner.hide()

    def run_job(self,function,result=None,error=None):
        job=Job(function,self);self.jobs.append(job)
        if result:job.result.connect(result)
        job.error.connect(error or self.fail)
        job.finished.connect(lambda:self._job_finished(job));job.start();return job

    def _job_finished(self,job):
        if job in self.jobs:self.jobs.remove(job)
        job.deleteLater()
        if self.exit_pending and not self.jobs:self._finish_close()
        else:self.library_job_finished()

    def refresh_ports(self):
        current=self.ports.currentData() or self.settings.value('port','COM6');self.ports.clear()
        try:
            for p in serial_devices():self.ports.addItem(p['port']+'  ·  '+p['description'],p['port'])
        except Exception as e:self.log('端口枚举：'+str(e))
        i=self.ports.findData(current)
        if i>=0:self.ports.setCurrentIndex(i)

    def refresh_mics(self):
        self.mics.clear()
        try:
            for m in microphones():self.mics.addItem(m['name'],m)
        except Exception as e:self.log('麦克风枚举：'+str(e))
        for i in range(self.mics.count()):
            if 'Realtek' in self.mics.itemText(i):self.mics.setCurrentIndex(i);break

    def toggle_serial(self):
        self.clear_banner()
        if self.link.connected:
            if self.player:self.disconnect_pending=True;self.stop_play();return
            self.link.close();self.log('串口已断开，TX 保持非 BREAK 状态。')
        else:
            name=self.ports.currentData()
            if not name:self.fail('没有发现串口。连接USB串口后点击刷新。');return
            try:self.link.open(name,self.baud.currentData());self.log(f'已连接 {name} · {self.baud.currentData()} · 8N1')
            except Exception as e:self.fail('无法连接串口：'+str(e));return
            self.settings.setValue('port',name)
        self._update_controls()

    def toggle_mic(self):
        try:
            if self.mic.stream:
                if self.mic.recording:self._save_manual_recording()
                self.mic.close();self.log('麦克风已关闭。')
            else:
                d=self.mics.currentData()
                if not d:raise RuntimeError('没有可用的WASAPI输入设备')
                self.mic.open(d['index'],d['rate']);self.log('麦克风已启用：'+d['name'])
            self._update_controls()
        except Exception as e:self.fail('麦克风：'+str(e))

    def new_session(self,prefix):
        return self.root/'sessions'/(prefix+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))

    def toggle_recording(self):
        try:
            if self.mic.recording:self._save_manual_recording()
            else:self.mic.start();self.log('独立录音已开始；保存原始幅度。')
            self._update_controls()
        except Exception as e:self.fail(str(e))

    def _save_manual_recording(self):
        path=self.mic.stop(self.new_session('recording'),dict(kind='independent_microphone_recording'))
        if path:self.last_recording=Path(path);self.log('录音已保存：'+str(path))

    def open_sessions(self):
        folder=self.root/'sessions';folder.mkdir(exist_ok=True);QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))

    def open_file(self):
        paths,_=QtWidgets.QFileDialog.getOpenFileNames(self,'导入 WAV 到文件库',str(self.settings.value('last_dir','')),'WAV 音频 (*.wav)')
        if paths:self.import_tracks(paths)

    def load_file(self,path):
        self.doc=None
        try:self.current_track_id=self.library.ensure(path);self.library_panel.refresh_tracks()
        except Exception as e:self.log('文件库记录失败：'+str(e));self.current_track_id=None
        if self.current_path and self.profile.get('reset_predistortion_on_import',True):
            self.predistortion_box.blockSignals(True);self.predistortion_box.setChecked(False);self.predistortion_box.blockSignals(False)
        self.current_source=None;self.current_path=str(path);self.settings.setValue('last_dir',str(Path(path).parent));self.start_conversion()

    def load_demo(self):
        self.current_track_id=None;self.queue_active=False;self.queue_ids=[]
        self.predistortion_box.blockSignals(True);self.predistortion_box.setChecked(False);self.predistortion_box.blockSignals(False)
        a,_=music_fixture('train',register_semitones=12);b,_=music_fixture('holdout',register_semitones=12)
        self.current_source=np.r_[a,b,b,a];self.current_path='demo://多声部演示乐曲.wav';self.start_conversion()

    def processing_changed(self):
        if not hasattr(self,'current_path'):return
        if hasattr(self,'loudness_level'):self.profile['loudness_target_rms']=self.loudness_level.currentData()
        if self.link.connected and not self.player:
            try:self.link.port.baudrate=self.baud.currentData()
            except Exception as e:self.fail(str(e))
        self.settings.setValue('baud',self.baud.currentData())
        if self.current_path and not self.loading and not self.player:self.start_conversion()

    def start_conversion(self):
        if self.loading or self.player:return
        self.clear_banner();self.loading=True;self.position=0.;self.conversion.setValue(0);self.convert_label.setText('正在转换音频…');self.cancel_button.show();self._update_controls()
        path=self.current_path;source=self.current_source;profile=dict(self.profile);profile['baud']=self.baud.currentData();mode=self.mode.currentData()
        profile.update(conditioning='music_loudness_lookahead_v2' if self.loudness.isChecked() else 'legacy',
                       continuous_carrier=self.continuous.isChecked(),soft_carrier_edges=True,
                       loudness_target_rms=self.loudness_level.currentData())
        if not self.predistortion_box.isChecked():profile.pop('predistortion',None)
        name='多声部演示乐曲.wav' if source is not None else Path(path).name
        job=Job(lambda j:self.cache.convert(path,profile,mode,self.fit_path,progress=j.progress.emit,cancel_check=j.isInterruptionRequested,source=source,name=name),self)
        self.conversion_job=job;self.jobs.append(job);job.progress.connect(lambda v:self.conversion.setValue(int(v*1000)))
        job.result.connect(self.converted);job.error.connect(self.conversion_failed);job.finished.connect(lambda:self._job_finished(job));job.start()

    def cancel_conversion(self):
        if self.conversion_job:self.conversion_job.requestInterruption()

    def converted(self,doc):
        if self.pending_track:
            self.loading=False;self.conversion_job=None;self.cancel_button.hide();return
        self.doc=doc;self.loading=False;self.conversion_job=None;self.cancel_button.hide();self.conversion.setValue(1000);self.clear_banner()
        self.file_label.setText(doc.name);self.file_label.setToolTip(doc.path)
        size=doc.metadata['total_bytes']/1024
        self.file_info.setText(f'{duration_text(doc.duration)}  ·  16 kHz 单声道  ·  {size:,.1f} KiB 串口数据  ·  {len(doc.segments)} 段')
        cached=doc.metadata.get('cache',{})
        status=(f'已读取缓存 · {cached["seconds"]:.2f} 秒' if cached.get('hit') else '转换完成 · 已缓存' if cached.get('persisted') else '转换完成')
        self.convert_label.setText(status);self.time_label.setText('00:00 / '+duration_text(doc.duration));self.byte_label.setText(f'{doc.metadata["total_bytes"]:,} B')
        self.log(f'{status}：{doc.name} · {doc.duration:.2f}s · {doc.profile["baud"]} baud · {doc.metadata["total_bytes"]:,} 字节')
        if cached.get('error'):self.log('缓存写入不可用，本次播放不受影响：'+cached['error'])
        self._update_controls();self.reset_plots()
        self.library_conversion_done(doc)

    def conversion_failed(self,text):
        self.loading=False;self.conversion_job=None;self.cancel_button.hide();self.convert_label.setText('转换未完成')
        if not self.pending_track:self.autoplay_pending=False;self.queue_active=False;self.fail(text)

    def toggle_play(self):
        if self.player:
            self.player.paused=not self.player.paused;self.log('播放已暂停。' if self.player.paused else '继续播放。');self._update_controls();return
        if not self.doc or self.loading:return
        if not self.link.connected:self.fail('请先选择并连接串口。');return
        if self.current_track_id:
            if not self.queue_ids or self.current_track_id not in self.queue_ids:self.queue_ids=self.library.ids()
            self.queue_active=True
        self.clear_banner();self.position=0.;self.timeline.setValue(0)
        self.live_decoded=np.zeros(1);self.live_decode_rate=self.doc.profile['baud']/10
        self.decode_filter=butter(4,[400,5000],btype='bandpass',fs=self.live_decode_rate,output='sos');self.decode_state=np.zeros((len(self.decode_filter),2))
        self.player=Playback(self.link,self.doc,self.mic,self.new_session('playback'),self.volume.value()/100,
                             record=self.auto_record.isChecked())
        self.player.muted=self.mute.isChecked()
        job=Job(lambda j:self.player.run(),self);self.play_job=job;self.jobs.append(job)
        self.player.telemetry=job.telemetry.emit;job.telemetry.connect(self.play_telemetry);job.result.connect(self.play_finished);job.error.connect(self.play_failed)
        job.finished.connect(lambda:self._job_finished(job));job.start();self.log('开始串口播放：'+self.doc.name);self._update_controls()

    def stop_play(self):
        self.pending_track=None;self.autoplay_pending=False;self.queue_active=False
        if self.loading:self.cancel_conversion()
        if self.player:self.player.stop();self.log('正在停止并清空主机待发送队列…');self.stop_button.setEnabled(False)

    def play_telemetry(self,data):
        self.position=data['position_s'];self.live_decode_rate=data['decoded_rate']
        decoded,self.decode_state=sosfilt(self.decode_filter,data['decoded'],zi=self.decode_state)
        self.live_decoded=np.r_[self.live_decoded,decoded][-int(self.live_decode_rate*.3):]
        self.byte_label.setText(f'{data["bytes"]:,} B');self.time_label.setText(duration_text(self.position)+' / '+duration_text(self.doc.duration))
        self.timeline.setValue(int(min(1,self.position/self.doc.duration)*10000))

    def _play_cleanup(self):
        if self.player and self.player.last_recording:self.last_recording=Path(self.player.last_recording);self.log('同步录音：'+str(self.last_recording))
        self.player=None;self.play_job=None
        if self.disconnect_pending:self.link.close();self.disconnect_pending=False
        self._update_controls()

    def play_finished(self,result):
        advance=result['completed'] and self.queue_active and not self.pending_track and not self.exit_pending and not self.disconnect_pending
        self.position=result['position_s'];self._play_cleanup();self.log('播放完成。' if result['completed'] else '播放已停止。')
        if result['completed']:self.timeline.setValue(10000)
        if advance:self.navigate(1,True)

    def play_failed(self,text):
        self.queue_active=False;self.autoplay_pending=False
        self._play_cleanup();self.fail('播放未完成：'+text)

    def volume_changed(self,*args):
        if not hasattr(self,'volume_label'):return
        self.volume_label.setText(f'{self.volume.value()}%');self.settings.setValue('volume',self.volume.value())
        if self.player:self.player.volume=self.volume.value()/100;self.player.muted=self.mute.isChecked()

    def export_data(self):
        if not self.doc:return
        folder=QtWidgets.QFileDialog.getExistingDirectory(self,'选择导出位置',str(self.root))
        if not folder:return
        target=Path(folder)/('uart_audio_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.log('正在导出字节与休止时间表…')
        self.run_job(lambda j:export_document(self.doc,target,0 if self.mute.isChecked() else self.volume.value()/100),
                     lambda p:self.log('串口数据已导出：'+str(p)))

    def import_profile(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'导入拟合配置',str(self.root),'JSON 配置 (*.json)')
        if not path:return
        try:
            self.profile=validate_profile(json.loads(Path(path).read_text(encoding='utf-8-sig')))
            self.settings.setValue('profile_path',path)
            if not self.profile.get('predistortion') and self.profile.get('available_predistortion'):self.profile['predistortion']=self.profile['available_predistortion']
            self.predistortion_box.blockSignals(True);self.predistortion_box.setChecked(bool(self.profile.get('predistortion')) and self.profile.get('predistortion_default_enabled',True));self.predistortion_box.blockSignals(False)
            self.loudness_level.blockSignals(True)
            self.loudness_level.setCurrentIndex(min(range(3),key=lambda i:abs(self.loudness_level.itemData(i)-self.profile.get('loudness_target_rms',.32))))
            self.loudness_level.blockSignals(False)
            self.mode.setCurrentIndex(0);self.model_badge.setText('自定义音乐补偿配置');self.log('配置已导入：'+Path(path).name)
            if self.current_path:self.start_conversion()
        except Exception as e:self.fail('配置无效：'+str(e))

    def save_profile(self):
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,'导出拟合配置',str(self.root/'music_profile.json'),'JSON 配置 (*.json)')
        if path:
            selected=dict(self.doc.profile if self.doc else self.profile)
            if not self.predistortion_box.isChecked():selected.pop('predistortion',None)
            selected['predistortion_default_enabled']=self.predistortion_box.isChecked()
            Path(path).write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf-8');self.log('配置已导出：'+path)

    def fit_latest(self):
        if not self.last_recording:return
        folder=self.last_recording.parent
        try:
            meta=json.loads((folder/'playback.json').read_text(encoding='utf-8'))
            recording=json.loads((folder/'recording.json').read_text(encoding='utf-8'))
            if not meta['completed'] or meta['paused']:raise ValueError('拟合需要连续完整播放；请勿中途暂停或停止')
            if recording['clipped_fraction']>0 or recording['portaudio_status']:raise ValueError('这次录音存在削波或采集异常，请重新录制')
            ref_rate,reference=wavfile.read(folder/'reference_processed.wav')
            sr,capture=wavfile.read(self.last_recording)
            if len(reference)<2*FS or np.sqrt(np.mean(reference**2))<1e-6:raise ValueError('请使用至少2秒有声音乐完成一次同步录音')
            a=int(meta['record_source_start_s']*sr);capture=capture[a:a+int(len(reference)/FS*sr)]
            ratio=Fraction(FS,int(sr));capture=resample_poly(capture,ratio.numerator,ratio.denominator)
            if len(capture)<len(reference)*.95:raise ValueError('录音长度不足')
        except Exception as e:self.fail('无法拟合：'+str(e));return
        self.loading=True;self._update_controls();self.convert_label.setText('正在按整段音乐拟合补偿…');self.log('开始整段音乐宽带拟合；新配置需要换曲验证。')
        def done(result):
            self.profile,analysis=result
            dest=self.root/'profiles';dest.mkdir(exist_ok=True)
            path=dest/('music_fit_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
            path.write_text(json.dumps(self.profile,ensure_ascii=False,indent=2),encoding='utf-8')
            self.settings.setValue('profile_path',str(path))
            path.with_suffix('.analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2),encoding='utf-8')
            self.loading=False;self.model_badge.setText('本机音乐拟合 · 待跨曲验证');self.mode.setCurrentIndex(0);self.log('拟合完成，已保存：'+str(path));self.start_conversion();self._update_controls()
        def failed(text):self.loading=False;self.fail(text)
        self.run_job(lambda j:fit_recording(reference,capture,self.profile,self.fit_path),done,failed)

    def _update_controls(self):
        playing=self.player is not None;connected=self.link.connected;busy=self.loading
        self.loudness.setEnabled(not playing and not busy);self.continuous.setEnabled(not playing and not busy)
        self.loudness_level.setEnabled(not playing and not busy and self.loudness.isChecked())
        self.predistortion_box.setEnabled(not playing and not busy and bool(self.profile.get('predistortion')))
        self.ports.setEnabled(not connected and not playing);self.baud.setEnabled(not playing and not busy);self.mode.setEnabled(not playing and not busy)
        self.connect_button.setText('断开串口' if connected else '连接串口');self.connection.setText(('●  '+self.link.name+' 已连接') if connected else '●  未连接')
        self.connection.setStyleSheet('color:#83c99e;' if connected else 'color:#93999f;')
        self.play_button.setEnabled((self.doc is not None and not busy and connected) or playing)
        self.play_button.setText(('▶  继续' if self.player.paused else 'Ⅱ  暂停') if playing else '▶  播放')
        self.stop_button.setEnabled(playing and not self.player.stop_event.is_set() if playing else False)
        if hasattr(self,'next_button'):
            enabled=bool(self.library.playable(self.queue_ids or self.library.ids()))
            self.next_button.setEnabled(enabled);self.previous_button.setEnabled(enabled)
        for b in (self.open_button,self.demo_button,self.import_profile_button):b.setEnabled(not playing and not busy)
        self.export_button.setEnabled(self.doc is not None and not busy and not playing)
        self.save_profile_button.setEnabled(not busy)
        self.fit_button.setEnabled(self.last_recording is not None and not playing and not busy)
        self.mic_button.setEnabled(not playing);self.mics.setEnabled(not bool(self.mic.stream) and not playing)
        self.mic_button.setText('关闭麦克风' if self.mic.stream else '启用麦克风')
        self.record_button.setEnabled(bool(self.mic.stream) and not playing);self.record_button.setText('停止并保存录音' if self.mic.recording else '开始独立录音')
        self.auto_record.setEnabled(bool(self.mic.stream) and not playing)
        self.record_state.setText('●  正在录音' if self.mic.recording else ('●  麦克风已启用' if self.mic.stream else '●  麦克风未启用'))
        self.record_state.setStyleSheet('color:#ef817a;' if self.mic.recording else 'color:#93999f;')
        state='播放中' if playing else ('转换中' if busy else '就绪')
        self.statusBar().showMessage(state+'  ·  '+(self.link.name+' / 8N1' if connected else '未连接串口')+'  ·  所有音频在本地处理')

    def reset_plots(self,*args):
        if self.signal_source.currentData()=='mic':self.wave.enableAutoRange(axis='y',enable=True)
        else:self.wave.disableAutoRange(axis='y');self.wave.setYRange(-1,1,padding=.02)
        self.wave.setXRange(-.2,0,padding=0);self.spectrum.setXRange(0,8000,padding=0);self.spectrum.setYRange(-120,0,padding=0)

    def update_graphs(self):
        def display_points(x,y):
            # Peak-preserving display reduction only. FFT/recording use full samples.
            step=max(1,int(np.ceil(len(y)/600)))
            if step==1:return x,y
            n=len(y)//step
            xx=x[:n*step].reshape(n,step).mean(axis=1)
            yy=y[:n*step].reshape(n,step)
            return np.repeat(xx,2),np.column_stack((yy.min(axis=1),yy.max(axis=1))).ravel()
        source=self.signal_source.currentData();rate=FS
        if source=='mic':
            rate=self.mic.rate;y=self.mic.recent(8192) if self.mic.stream else np.zeros(4096)
            self.analysis_hint.setText('麦克风原始幅度与 FFT（dBFS），未做声压校准。')
        elif source=='decoded' and self.player:
            rate=self.live_decode_rate;y=self.live_decoded
            self.analysis_hint.setText('从实际已发送字节实时重建音频密度；并非实测线圈电流。')
        elif self.doc:
            a=self.doc.decoded if source=='decoded' else self.doc.processed
            end=min(len(a),max(32 if self.player else 4096,int(self.position*FS)))
            y=a[max(0,end-4096):end]*(0 if self.mute.isChecked() else self.volume.value()/100)
            if source!='decoded' and self.doc.quadratic_correction is not None:
                gain=0 if self.mute.isChecked() else self.volume.value()/100
                y=self.doc.base_processed[max(0,end-4096):end]*gain+self.doc.quadratic_correction[max(0,end-4096):end]*gain**2
            self.analysis_hint.setText('UART 数字解算：帧密度经音频带宽重建；不是实测线圈电流。' if source=='decoded' else '处理后 PCM：与当前音量关联；补偿及编解码均在本地完成。')
        else:y=np.zeros(4096)
        if len(y)<32:return
        n=min(len(y),int(rate*.2));w=y[-n:];t=(np.arange(n)-n+1)/rate
        self.wave_curve.setData(*display_points(t,w))
        data=y[-min(len(y),4096):];win=np.hanning(len(data));amp=abs(np.fft.rfft((data-data.mean())*win))*2/max(win.sum(),1)
        f=np.fft.rfftfreq(len(data),1/rate);self.spec_curve.setData(*display_points(f,20*np.log10(np.maximum(amp,1e-8))))
        if self.overlay.isChecked() and self.doc and source!='mic':
            end=min(len(self.doc.source),max(4096,int(self.position*FS)));raw=self.doc.source[max(0,end-4096):end]
            nr=min(len(raw),3200);self.wave_raw.setData((np.arange(nr)-nr+1)/FS,raw[-nr:])
            rw=np.hanning(len(raw));ra=abs(np.fft.rfft((raw-raw.mean())*rw))*2/max(rw.sum(),1)
            self.spec_raw.setData(np.fft.rfftfreq(len(raw),1/FS),20*np.log10(np.maximum(ra,1e-8)))
        else:self.wave_raw.clear();self.spec_raw.clear()
        self.record_state.setText('●  正在录音' if self.mic.recording else ('●  麦克风已启用' if self.mic.stream else '●  麦克风未启用'))
        self.record_state.setStyleSheet('color:#ef817a;' if self.mic.recording else 'color:#93999f;')

    def dragEnterEvent(self,event):
        if event.mimeData().hasUrls() and any(u.toLocalFile().lower().endswith('.wav') or Path(u.toLocalFile()).is_dir() for u in event.mimeData().urls()):event.acceptProposedAction()

    def dropEvent(self,event):
        self.import_tracks([u.toLocalFile() for u in event.mimeData().urls()]);event.acceptProposedAction()

    def closeEvent(self,event):
        if self.player or self.jobs:
            self.exit_pending=True;self.cancel_conversion();self.stop_play();event.ignore();self.hide()
            return
        self._finish_close();event.accept()

    def _finish_close(self):
        try:
            self.settings.setValue('library_visible',not self.library_dock.isHidden())
            if self.mic.recording:self._save_manual_recording()
            self.mic.close();self.link.close();self.settings.setValue('geometry',self.saveGeometry());self.settings.sync()
        finally:QtWidgets.QApplication.instance().quit()


def self_test(path):
    profile=validate_profile(json.loads((ASSETS/'music_profile.json').read_text(encoding='utf-8')))
    x,_=music_fixture('train');doc=convert('embedded-demo.wav',profile,source=x[:FS*2],fit_path=ASSETS/'impedance_fit.json')
    fit_profile,fit=fit_recording(doc.processed,doc.processed*.01,profile,ASSETS/'impedance_fit.json')
    pd_ok=True
    if profile.get('available_predistortion'):
        candidate=dict(profile,predistortion=profile['available_predistortion'])
        pd=convert('embedded-demo.wav',candidate,source=x[:FS*2],fit_path=ASSETS/'impedance_fit.json')
        pd_ok=bool(pd.metadata['predistortion']['enabled'] and pd.quadratic_correction is not None)
    cache_folder=Path(path).parent/'selftest_conversion_cache'
    first=ConversionCache(cache_folder).convert('embedded-demo.wav',profile,source=x[:FS*2])
    cached=ConversionCache(cache_folder).convert('embedded-demo.wav',profile,source=x[:FS*2])
    cache_ok=cached.metadata['cache'].get('tier')=='disk' and [s['data'] for s in first.segments]==[s['data'] for s in cached.segments]
    report=dict(version=VERSION,frozen=bool(getattr(sys,'frozen',False)),assets_present=True,
                decoded_samples=len(doc.decoded),bytes=doc.metadata['total_bytes'],fit_bins=fit['excited_frequency_bins'],
                predistortion_self_test=pd_ok,predistortion_default_enabled=bool(profile.get('predistortion_default_enabled',False)),
                persistent_cache_self_test=cache_ok,success=doc.duration==2 and fit['excited_frequency_bins']>50 and pd_ok and cache_ok)
    Path(path).write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not report['success']:raise RuntimeError('Self test failed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path);parser.add_argument('--self-test',type=Path)
    parser.add_argument('--wav',type=Path,help='Load a WAV on startup, without starting playback')
    parser.add_argument('--qa',type=Path,help=argparse.SUPPRESS);parser.add_argument('--qa-hardware',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--qa-source',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--qa-predistortion',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--state-report',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.self_test:self_test(args.self_test);return
    app=QtWidgets.QApplication(sys.argv[:1]);app.setApplicationName('UART Audio Studio');app.setOrganizationName('ZipLab')
    root=data_root(args.data_dir);root.mkdir(parents=True,exist_ok=True)
    if args.state_report:
        from .library import LibraryStore
        store=LibraryStore(root/'library.json');settings=QtCore.QSettings(str(root/'settings.ini'),QtCore.QSettings.IniFormat)
        args.state_report.write_text(json.dumps(dict(version=VERSION,root=str(root),library=str(store.path),warning=store.warning,
            tracks=len(store.ids('all')),playlists=store.data['playlists'],play_mode=settings.value('play_mode','single'),
            prepared_files=len(list((root/'prepared_audio').glob('*.npz')))),ensure_ascii=False,indent=2),encoding='utf-8');return
    instance_lock=QtCore.QLockFile(str(root/'application.lock'))
    if not instance_lock.tryLock(0):
        QtWidgets.QMessageBox.information(None,'UART Audio Studio','同一数据目录的播放器已经打开。请切回现有窗口，避免多个窗口覆盖歌单。\n'+str(root));return
    app.setStyle('Fusion');app.setFont(QtGui.QFont('Microsoft YaHei UI',9))
    window=MainWindow(args.data_dir);window.show();QtCore.QTimer.singleShot(100,window.ensure_visible)
    if args.wav:QtCore.QTimer.singleShot(200,lambda:window.load_file(str(args.wav)))
    elif not args.qa:
        current=window.library.data.get('current')
        if current in window.library.playable([current]):QtCore.QTimer.singleShot(200,lambda:window.load_file(window.library.data['tracks'][current]['path']))
    if args.qa:
        from .qa import start_qa
        start_qa(window,args.qa,args.qa_hardware,args.qa_source,args.qa_predistortion)
    sys.exit(app.exec())
