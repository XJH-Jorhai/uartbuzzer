"""Score inspection, named capability profiles and explicit characterization UI."""
import copy,json,uuid
from pathlib import Path
from PySide6 import QtCore,QtWidgets
from ..score.contracts import ProfileStore,flat_profile,device_profile,load_json
from ..score.model import digest
from ..score.pipeline import settings as score_settings
from ..score.export import export_json,export_midi


def spin(low,high,value):
    w=QtWidgets.QSpinBox();w.setRange(low,high);w.setValue(value);return w


def table(headers,rows):
    w=QtWidgets.QTableWidget(len(rows),len(headers));w.setHorizontalHeaderLabels(headers)
    w.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers);w.setAlternatingRowColors(True)
    for i,row in enumerate(rows):
        for j,value in enumerate(row):w.setItem(i,j,QtWidgets.QTableWidgetItem(str(value)))
    w.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
    return w


class ScoreController:
    def init_score(self):
        self.device_profiles=ProfileStore(self.root/'device_profiles');self.score_config=score_settings({});self.score_dialog=None
        if self.device_profiles.errors:self.profile_migration_warning='部分器件档案无法验证，原文件保留：'+'；'.join(self.device_profiles.errors[:3])
        try:
            saved=json.loads(self.settings.value('score_config','{}'));self.score_config=score_settings(saved)
        except Exception as e:self.profile_migration_warning='曲谱设置恢复失败，使用未测范围：'+str(e)

    def show_score(self):
        dialog=ScoreDialog(self);self.score_dialog=dialog;dialog.show()

    def score_options(self):return copy.deepcopy(self.score_config)


class ScoreDialog(QtWidgets.QDialog):
    def __init__(self,window):
        super().__init__(window);self.window=window;self.setWindowTitle('离散曲谱 · 主旋律与器件档案');self.resize(900,700)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        layout=QtWidgets.QVBoxLayout(self);tabs=QtWidgets.QTabWidget();layout.addWidget(tabs)
        setup=QtWidgets.QWidget();form=QtWidgets.QFormLayout(setup);tabs.addTab(setup,'拟合与器件档案')
        self.profiles=QtWidgets.QComboBox();self.reload_profiles();form.addRow('明确选择能力档案',self.profiles)
        cfg=window.score_config;fit=cfg['fit'];self.palette=spin(1,32,fit['palette_size']);self.auto=QtWidgets.QCheckBox('自动覆盖（最多上述音高数）');self.auto.setChecked(fit['automatic_palette'])
        form.addRow('有限音高预算',self.palette);form.addRow(self.auto)
        self.quantum=QtWidgets.QComboBox()
        for ms in (10,20,40,80):self.quantum.addItem(f'{ms} ms',ms/1000)
        self.quantum.setCurrentIndex(max(0,self.quantum.findData(fit['time_quantum_s'])));form.addRow('时间量化',self.quantum)
        self.octave=QtWidgets.QCheckBox('允许全局八度适配（记录±12半音）');self.octave.setChecked(fit['allow_octave_adaptation']);form.addRow(self.octave)
        self.response=QtWidgets.QCheckBox('使用可靠相对响应 / 手动估计（线性近似）');self.response.setChecked(cfg['response_enabled']);form.addRow(self.response)
        warning=QtWidgets.QLabel('默认未测平坦。M2电流不是SPL。声学响应只在实测音量、baud、密度和渲染器配置匹配时使用；变化后停用响应权重。\n分析音区固定MIDI48–96；扩大器件范围不会扩展扒谱音区。前端10ms、曲谱量化、器件最小时值共同限制起止精度。');warning.setWordWrap(True);form.addRow(warning)
        row=QtWidgets.QHBoxLayout()
        for title,slot in [('导入档案JSON',self.import_profile),('导出所选档案',self.export_profile),('播放器特性测试…',self.characterize)]:
            b=QtWidgets.QPushButton(title);b.clicked.connect(slot);row.addWidget(b)
        form.addRow(row)
        self.name=QtWidgets.QLineEdit('我的未测播放器');self.low=spin(24,108,48);self.high=spin(24,108,96)
        self.transport=QtWidgets.QComboBox();self.transport.addItem('UART 8N1', 'uart8n1');self.transport.addItem('通用曲谱导出（无运动命令）','generic')
        form.addRow('新档案名称',self.name);ranges=QtWidgets.QHBoxLayout();ranges.addWidget(self.low);ranges.addWidget(QtWidgets.QLabel('至 MIDI'));ranges.addWidget(self.high);form.addRow('手动可用音高范围',ranges);form.addRow('后端',self.transport)
        create=QtWidgets.QPushButton('保存新的未测平坦档案（不覆盖旧档案）');create.clicked.connect(self.create_profile);form.addRow(create)
        apply=QtWidgets.QPushButton('应用选择并重新转换当前歌曲');apply.clicked.connect(self.apply);form.addRow(apply)
        self.summary=QtWidgets.QLabel();self.summary.setWordWrap(True);layout.addWidget(self.summary)
        score=window.doc.metadata.get('score') if window.doc else None
        if score:
            d=score['diagnostics'];p=score['fitting']['palette']
            self.summary.setText(f'实际{d["unique_event_pitches"]}音高 / {d["events"]}事件；筛选候选权重覆盖 {p.get("weighted_exact_pitch_coverage",0):.1%}；'
                f'原主线时长覆盖 {d["melody_temporal_coverage"]:.1%}，未表示 {d["unrepresented_reference_s"]:.3f}s，休止 {d["rest_s"]:.3f}s；'
                f'全局移调 {score["global_semitone_shift"]:+}半音；相对原主线平均绝对偏移 {d["duration_weighted_absolute_pitch_error_semitones"]:.2f}半音。\n'
                +'；'.join(d['limitations']))
            tabs.addTab(table(['ID','Hz','MIDI'],[(n['id'],f'{n["frequency_hz"]:.3f}',f'{n["midi_pitch"]:.2f}') for n in score['palette']]),'有限音高表')
            rows=[(f'{e["start_s"]:.3f}',f'{e["duration_s"]:.3f}',e['note_id'],f'{e["frequency_hz"]:.2f}',e['velocity'],f'{e["confidence_score"]:.2f}') for e in score['events']]
            tabs.addTab(table(['开始s','时长s','音符','Hz','有限力度','算法分（非概率）'],rows),'音符事件')
            tabs.addTab(table(['开始s','时长s','原主线MIDI','说明'],[(f'{e["start_s"]:.3f}',f'{e["duration_s"]:.3f}',e['reference_midi'],e['reason']) for e in score.get('unresolved',[])]),'未表示 / 待确认')
        else:self.summary.setText('选择“离散曲谱（主旋律）”并转换歌曲后，可查看字典、事件、覆盖和未表示片段。')
        buttons=QtWidgets.QHBoxLayout();layout.addLayout(buttons)
        for title,slot in [('导出曲谱JSON',lambda:self.export_score(False)),('导出MIDI（整数音高近似）',lambda:self.export_score(True))]:
            b=QtWidgets.QPushButton(title);b.setEnabled(bool(score));b.clicked.connect(slot);buttons.addWidget(b)
        close=QtWidgets.QPushButton('关闭');close.clicked.connect(self.close);buttons.addWidget(close)

    def selected(self):
        key=self.profiles.currentData()
        return flat_profile() if key=='default' else self.window.device_profiles.profiles[key]
    def reload_profiles(self,select=None):
        selected=select or digest(self.window.score_config['device_profile'])
        self.profiles.clear();self.profiles.addItem('未测平坦 · 默认 MIDI48–96','default')
        for key,p in self.window.device_profiles.profiles.items():self.profiles.addItem(p['device_id']+' · '+p['response_kind']+' · '+key[:8],key)
        index=self.profiles.findData(selected);self.profiles.setCurrentIndex(max(0,index))
    def guarded(self,fn):
        try:fn()
        except Exception as e:QtWidgets.QMessageBox.warning(self,'操作未完成',str(e))
    def import_profile(self):
        name,_=QtWidgets.QFileDialog.getOpenFileName(self,'导入命名器件档案','','JSON (*.json)')
        if name:self.guarded(lambda:(self.window.device_profiles.add(device_profile(load_json(name))),self.reload_profiles()))
    def export_profile(self):
        name,_=QtWidgets.QFileDialog.getSaveFileName(self,'导出所选器件档案','device-profile.json','JSON (*.json)')
        if name:self.guarded(lambda:Path(name).write_text(json.dumps(self.selected(),ensure_ascii=False,indent=2),encoding='utf-8'))
    def create_profile(self):
        def create():
            p=flat_profile(self.name.text().strip(),self.low.value(),self.high.value(),self.transport.currentData());key=self.window.device_profiles.add(p);self.reload_profiles(key)
        self.guarded(create)
    def apply(self):
        def apply():
            w=self.window
            if w.loading or w.player:raise ValueError('请先停止播放或等待转换完成')
            fit=dict(w.score_config['fit'],palette_size=self.palette.value(),automatic_palette=self.auto.isChecked(),time_quantum_s=self.quantum.currentData(),allow_octave_adaptation=self.octave.isChecked())
            w.score_config=score_settings(dict(device_profile=self.selected(),fit=fit,response_enabled=self.response.isChecked()))
            w.settings.setValue('score_config',json.dumps(w.score_config,ensure_ascii=False));w.settings.sync()
            w.encoder_backend.setCurrentIndex(w.encoder_backend.findData('discrete_score'))
            if w.current_path and not w.loading:w.start_conversion()
            self.close()
        self.guarded(apply)
    def export_score(self,midi):
        score=copy.deepcopy(self.window.doc.metadata['score']);name,_=QtWidgets.QFileDialog.getSaveFileName(self,'MIDI将Hz舍入到整数半音；JSON保留原Hz' if midi else '导出有限曲谱','score.mid' if midi else 'score.json','MIDI (*.mid)' if midi else 'JSON (*.json)')
        if name:self.guarded(lambda:(export_midi if midi else export_json)(score,name))
    def characterize(self):
        w=self.window
        if self.selected().get('transport','uart8n1')=='generic':
            QtWidgets.QMessageBox.information(self,'通用曲谱档案','本版通用播放器只导出曲谱。特性测试仅支持UART 8N1，不生成电机控制命令。');return
        if w.player or w.loading or w.link.connected or w.mic.stream:
            QtWidgets.QMessageBox.information(self,'需要独占设备','请停止播放/转换，并断开主窗口串口、关闭主窗口麦克风后开始特性测试。');return
        dialog=CharacterizationDialog(w,self);self.characterization_dialog=dialog;dialog.open()


class CharacterizationDialog(QtWidgets.QDialog):
    def __init__(self,window,parent):
        super().__init__(parent);self.window=window;self.job=None;self.pending=None;self.setWindowTitle('播放器特性测试 · 主动短音扫测');self.resize(660,620);self.setWindowModality(QtCore.Qt.ApplicationModal)
        layout=QtWidgets.QVBoxLayout(self);form=QtWidgets.QFormLayout();layout.addLayout(form)
        self.name=QtWidgets.QLineEdit('新的相对声学档案');self.port=QtWidgets.QComboBox();self.mic=QtWidgets.QComboBox()
        for i in range(window.ports.count()):self.port.addItem(window.ports.itemText(i),window.ports.itemData(i))
        for i in range(window.mics.count()):self.mic.addItem(window.mics.itemText(i),window.mics.itemData(i))
        self.low=spin(36,96,60);self.high=spin(36,96,72);self.volume=spin(10,100,80);self.duration=spin(250,1000,350);self.position=QtWidgets.QLineEdit('')
        for label,widget in [('新档案名称',self.name),('串口（点击开始后打开）',self.port),('WASAPI麦克风（点击开始后打开）',self.mic),('最低MIDI',self.low),('最高MIDI（最多24点）',self.high),('参考音量 %（另测半档）',self.volume),('每音时长 ms',self.duration),('麦克风位置/增益说明',self.position)]:form.addRow(label,widget)
        self.notice=QtWidgets.QLabel('将录背景，再逐音播放两档短音并留静音间隔。固定麦克风位置和系统增益。\n结果仅相对响应、启发式可靠度；非校准SPL。失败点保留证据，不进入有效档案。\n完成后仍需确认保存新档案，再到曲谱设置明确选择；不会覆盖本曲录音校准。');self.notice.setWordWrap(True);layout.addWidget(self.notice)
        self.progress=QtWidgets.QProgressBar();layout.addWidget(self.progress);self.details=table(['MIDI','音量','实际基音Hz','最大谱峰Hz','有效','原因'],[]);layout.addWidget(self.details)
        row=QtWidgets.QHBoxLayout();layout.addLayout(row)
        self.start=QtWidgets.QPushButton('开始播放并采集');self.start.clicked.connect(self.begin);row.addWidget(self.start)
        self.cancel=QtWidgets.QPushButton('取消');self.cancel.clicked.connect(self.stop);row.addWidget(self.cancel)
        self.save=QtWidgets.QPushButton('确认保存为新档案');self.save.setEnabled(False);self.save.clicked.connect(self.accept_profile);row.addWidget(self.save)
    def begin(self):
        try:
            from .characterization import run
            from ..score.uart_renderer import VERSION as renderer_version
            w=self.window;mic=self.mic.currentData()
            if self.job or w.player or w.loading or w.link.connected or w.mic.stream:raise ValueError('设备未独占或已有任务运行')
            if not mic or not self.port.currentData():raise ValueError('请选择串口和麦克风')
            cfg=dict(name=self.name.text().strip(),port=self.port.currentData(),microphone=mic['index'],rate=mic['rate'],midi_min=self.low.value(),midi_max=self.high.value(),volume=self.volume.value()/100,duration_s=self.duration.value()/1000,gap_s=.2,baud=w.baud.currentData(),position=self.position.text())
            cfg.update(bias=w.profile['bias'],amplitude=w.profile['amplitude'],renderer=renderer_version)
            self.folder=w.root/'characterization'/uuid.uuid4().hex
            self.job=w.run_job(lambda j:run(cfg,self.folder,j.isInterruptionRequested,j.progress.emit),self.ready,self.failed)
            self.job.progress.connect(lambda v:self.progress.setValue(round(v*100)));self.job.finished.connect(self.finished)
            self.start.setEnabled(False);self.save.setEnabled(False);self.notice.setText('测量中；取消会停止发送、关闭专用设备并保留当前证据。')
        except Exception as e:self.failed(str(e))
    def ready(self,p):self.pending=p;self.save.setEnabled(True);self.notice.setText('测量完成。请审阅有效/失效点后，确认保存新的命名档案；当前编码器尚未切换档案。');self.show_evidence()
    def failed(self,text):self.notice.setText('未生成可用档案：'+text);self.show_evidence()
    def show_evidence(self):
        if not hasattr(self,'folder') or not (self.folder/'evidence.json').exists():return
        data=load_json(self.folder/'evidence.json');self.details.setRowCount(len(data['records']))
        for i,r in enumerate(data['records']):
            for j,value in enumerate([r['midi'],r['command_volume'],r['actual_fundamental_hz'],r['largest_spectral_peak_hz'],r['valid'],'；'.join(r['reasons'])]):self.details.setItem(i,j,QtWidgets.QTableWidgetItem(str(value)))
    def finished(self):self.job=None;self.start.setEnabled(True)
    def stop(self):
        if self.job:self.job.requestInterruption();self.notice.setText('正在取消并回收设备…')
        else:self.close()
    def accept_profile(self):
        if self.pending:
            self.window.device_profiles.add(self.pending);self.parent().reload_profiles();self.pending=None;self.accept()
    def closeEvent(self,event):
        if self.job:self.job.requestInterruption();event.ignore()
        else:event.accept()
