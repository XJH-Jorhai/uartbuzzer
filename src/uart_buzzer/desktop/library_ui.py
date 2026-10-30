"""Local library panel and playback queue integration."""
from pathlib import Path
from PySide6 import QtCore,QtWidgets,QtGui
from .library import LibraryStore,inspect_paths,move_items,next_track

class LibraryPanel(QtWidgets.QWidget):
    def __init__(self,owner):
        super().__init__();self.owner=owner
        layout=QtWidgets.QVBoxLayout(self);layout.setContentsMargins(8,8,8,8)
        row=QtWidgets.QHBoxLayout();self.lists=QtWidgets.QComboBox();row.addWidget(self.lists,1)
        for text,slot in [('+',self.create),('…',self.manage)]:
            b=QtWidgets.QPushButton(text);b.setFixedWidth(30);b.clicked.connect(slot);row.addWidget(b)
        layout.addLayout(row);self.lists.currentIndexChanged.connect(self.activate)
        self.search=QtWidgets.QLineEdit();self.search.setPlaceholderText('搜索歌曲 / 文件路径');self.search.textChanged.connect(self.refresh_tracks);layout.addWidget(self.search)
        self.tree=QtWidgets.QTreeWidget();self.tree.setHeaderLabels(['歌曲','时长 / 状态']);self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection);self.tree.setAlternatingRowColors(True)
        self.tree.header().setStretchLastSection(False);self.tree.header().setSectionResizeMode(0,QtWidgets.QHeaderView.Stretch);self.tree.header().setSectionResizeMode(1,QtWidgets.QHeaderView.ResizeToContents)
        self.tree.itemDoubleClicked.connect(lambda item,col:owner.request_track(item.data(0,QtCore.Qt.UserRole),True,True))
        self.tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu);self.tree.customContextMenuRequested.connect(self.menu);layout.addWidget(self.tree,1)
        self.count=QtWidgets.QLabel();self.count.setWordWrap(True);layout.addWidget(self.count)
        row=QtWidgets.QHBoxLayout()
        for text,slot in [('导入 WAV',owner.open_file),('导入文件夹',self.folder)]:
            b=QtWidgets.QPushButton(text);b.clicked.connect(slot);row.addWidget(b)
        layout.addLayout(row);row=QtWidgets.QHBoxLayout()
        for text,slot in [('↑',lambda:self.move(-1)),('↓',lambda:self.move(1)),('移除',self.remove)]:
            b=QtWidgets.QPushButton(text);b.clicked.connect(slot);row.addWidget(b)
        layout.addLayout(row);self.refresh()

    def selected(self):return [i.data(0,QtCore.Qt.UserRole) for i in self.tree.selectedItems()]
    def refresh(self):
        store=self.owner.library;self.lists.blockSignals(True);self.lists.clear();self.lists.addItem('全部音乐','all')
        for p in store.data['playlists']:self.lists.addItem(p['name'],p['id'])
        self.lists.setCurrentIndex(max(0,self.lists.findData(store.data['active'])));self.lists.blockSignals(False);self.refresh_tracks()
    def refresh_tracks(self,*args):
        selected=set(self.selected());self.tree.clear();store=self.owner.library;q=self.search.text().casefold();shown=0
        for i in store.ids():
            t=store.data['tracks'][i]
            if q and q not in (t.get('title','')+' '+t['path']).casefold():continue
            exists=Path(t['path']).is_file();d=t.get('duration_s');status=f'{int(d)//60:02}:{int(d)%60:02}' if d is not None else '待分析'
            prepared=[p for p in t.get('prepared_files',[]) if Path(p.get('file','')).is_file()]
            cached=any((self.owner.cache.folder/(k+'.npz')).is_file() for k in t.get('cache_keys',[]))
            if not exists:status='文件缺失'
            elif t.get('error'):status='格式异常'
            elif prepared:status+=' · 已转换'
            elif cached:status+=' · 缓存'
            item=QtWidgets.QTreeWidgetItem([('▶ ' if i==self.owner.current_track_id else '')+t.get('title',Path(t['path']).stem),status]);item.setData(0,QtCore.Qt.UserRole,i)
            item.setToolTip(0,t['path']);item.setToolTip(1,t.get('error') or '\n'.join(p['file'] for p in prepared) or '播放时自动核对当前配置的转换文件。')
            if not exists or t.get('error'):item.setForeground(0,QtGui.QColor('#c68b83'))
            self.tree.addTopLevelItem(item);item.setSelected(i in selected);shown+=1
        self.count.setText(f'{shown} / {len(store.ids())} 首 · 双击播放 · 支持拖入 WAV / 文件夹')
    def activate(self,*args):
        try:self.owner.library.activate(self.lists.currentData());self.refresh_tracks()
        except Exception as e:self.owner.fail(str(e))
    def create(self):
        name,ok=QtWidgets.QInputDialog.getText(self,'新建歌单','歌单名称')
        if ok:
            try:self.owner.library.create_playlist(name);self.refresh()
            except Exception as e:self.owner.fail(str(e))
    def manage(self):
        menu=QtWidgets.QMenu(self);rename=menu.addAction('重命名歌单');delete=menu.addAction('删除歌单');i=self.lists.currentData()
        rename.setEnabled(i!='all');delete.setEnabled(i not in ('all','default'));choice=menu.exec(QtGui.QCursor.pos())
        try:
            if choice==rename:
                name,ok=QtWidgets.QInputDialog.getText(self,'重命名歌单','歌单名称',text=self.lists.currentText())
                if ok:self.owner.library.rename_playlist(i,name)
            elif choice==delete:self.owner.library.delete_playlist(i)
            self.refresh()
        except Exception as e:self.owner.fail(str(e))
    def folder(self):
        folder=QtWidgets.QFileDialog.getExistingDirectory(self,'导入文件夹中的 WAV',str(self.owner.settings.value('last_dir','')))
        if folder:self.owner.import_tracks([folder])
    def move(self,direction):
        try:self.owner.library.reorder(move_items(self.owner.library.ids(),self.selected(),direction));self.refresh_tracks()
        except Exception as e:self.owner.fail(str(e))
    def remove(self):
        try:self.owner.library.remove(self.selected());self.refresh_tracks()
        except Exception as e:self.owner.fail(str(e))
    def menu(self,pos):
        ids=self.selected()
        if not ids:return
        menu=QtWidgets.QMenu(self);sub=menu.addMenu('添加到歌单');actions={}
        for p in self.owner.library.data['playlists']:actions[sub.addAction(p['name'])]=p['id']
        prepared=menu.addAction('打开已转换文件目录');prepared.setEnabled(len(ids)==1 and bool(self.owner.library.data['tracks'][ids[0]].get('prepared_files')))
        relink=menu.addAction('重新定位文件…');relink.setEnabled(len(ids)==1);remove=menu.addAction('从当前列表移除');action=menu.exec(self.tree.viewport().mapToGlobal(pos))
        try:
            if action in actions:self.owner.library.add_to_playlist(ids,actions[action])
            elif action==prepared:
                files=self.owner.library.data['tracks'][ids[0]].get('prepared_files',[])
                if files:QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(Path(files[0]['file']).parent)))
            elif action==remove:self.owner.library.remove(ids)
            elif action==relink:
                path,_=QtWidgets.QFileDialog.getOpenFileName(self,'重新定位 WAV','','WAV (*.wav)')
                if path:self.owner.relink_track(ids[0],path)
            self.refresh_tracks()
        except Exception as e:self.owner.fail(str(e))

class LibraryController:
    def relink_track(self,old,path):
        new=self.library.relink(old,path)
        self.queue_ids=list(dict.fromkeys(new if i==old else i for i in self.queue_ids))
        if self.pending_track and self.pending_track[0]==old:
            self.pending_track=(new,self.pending_track[1])
        if self.current_track_id==old:
            self.current_track_id=new
            self.request_track(new,False)
        self.library_panel.refresh_tracks()
        return new

    def init_library(self):
        self.library=LibraryStore(self.root/'library.json');self.current_track_id=None;self.queue_ids=[]
        self.pending_track=None;self.autoplay_pending=False;self.queue_active=False;self.importing_library=False
    def build_library(self,header,controls):
        self.library_panel=LibraryPanel(self);self.library_dock=QtWidgets.QDockWidget('文件库 / 歌单',self);self.library_dock.setObjectName('musicLibraryDock')
        self.library_dock.setWidget(self.library_panel);self.library_dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea|QtCore.Qt.RightDockWidgetArea)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea,self.library_dock);self.resizeDocks([self.library_dock],[310],QtCore.Qt.Horizontal)
        toggle=QtWidgets.QToolButton();toggle.setDefaultAction(self.library_dock.toggleViewAction());header.addWidget(toggle)
        self.previous_button=QtWidgets.QPushButton('⏮');self.previous_button.setToolTip('上一首');self.previous_button.clicked.connect(lambda:self.navigate(-1));controls.insertWidget(0,self.previous_button)
        self.next_button=QtWidgets.QPushButton('⏭');self.next_button.setToolTip('下一首');self.next_button.clicked.connect(lambda:self.navigate(1));controls.addWidget(self.next_button);controls.addStretch()
        self.play_mode=QtWidgets.QComboBox()
        for name,key in [('单曲播放 · 播完停止','single'),('顺序播放','sequential'),('列表循环','loop'),('单曲循环','repeat_one'),('随机播放','shuffle')]:self.play_mode.addItem(name,key)
        if not self.settings.value('single_play_default_v131',False,type=bool):
            self.settings.setValue('play_mode','single');self.settings.setValue('single_play_default_v131',True)
        self.play_mode.setCurrentIndex(max(0,self.play_mode.findData(self.settings.value('play_mode','single'))))
        self.play_mode.currentIndexChanged.connect(lambda:self.settings.setValue('play_mode',self.play_mode.currentData()));controls.addWidget(self.play_mode)
        default='true' if self.screen().availableGeometry().width()>=1100 else 'false'
        self.library_dock.setVisible(str(self.settings.value('library_visible',default)).lower()=='true')
        open_data=QtWidgets.QPushButton('打开歌单 / 转换文件目录');open_data.setToolTip(str(self.root));open_data.clicked.connect(lambda:QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.root))))
        self.library_panel.layout().addWidget(open_data)
        location=QtWidgets.QLabel('歌单：'+str(self.library.path));location.setWordWrap(True);location.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse);location.setStyleSheet('color:#93999f;font-size:10px;')
        location.setSizePolicy(QtWidgets.QSizePolicy.Ignored,QtWidgets.QSizePolicy.Preferred)
        self.library_panel.layout().addWidget(location)
    def import_tracks(self,paths):
        if self.importing_library:return
        self.importing_library=True;target=self.library.data['active'];self.log('正在读取文件库元数据…')
        def done(tracks):
            self.importing_library=False
            try:
                ids=self.library.add(tracks,target);self.library_panel.refresh();self.log(f'已导入 {len(ids)} 首 WAV，重复路径自动合并。')
                if ids and not self.player and not self.loading:self.request_track(ids[0],False,True)
            except Exception as e:self.fail(str(e))
        def failed(text):self.importing_library=False;self.fail(text)
        self.run_job(lambda j:inspect_paths(paths),done,failed)
    def request_track(self,i,autoplay=False,new_queue=False):
        if i not in self.library.playable([i]):self.fail('歌曲文件缺失或不支持，请右键重新定位。');return
        if new_queue:self.queue_ids=self.library.ids()
        self.pending_track=(i,autoplay);self.autoplay_pending=False
        if self.player:self.player.stop();return
        if self.loading:self.cancel_conversion();return
        self.drain_track()
    def drain_track(self):
        if self.exit_pending or self.player or self.loading or self.jobs or not self.pending_track:return
        i,autoplay=self.pending_track;self.pending_track=None
        if i not in self.library.data['tracks']:return
        self.autoplay_pending=autoplay and self.link.connected;self.queue_active=self.autoplay_pending
        self.load_file(self.library.data['tracks'][i]['path'])
        if autoplay and not self.link.connected:self.log('歌曲已载入；连接串口后点击播放。')
    def navigate(self,direction=1,automatic=False):
        ids=self.library.playable(self.queue_ids if self.queue_ids else self.library.ids())
        i=next_track(ids,self.current_track_id,self.play_mode.currentData(),direction,automatic)
        if i:self.request_track(i,automatic or self.player is not None or self.queue_active)
        else:self.queue_active=False
    def library_conversion_done(self,doc):
        if self.current_track_id:
            try:
                key=doc.metadata.get('cache',{}).get('key');prepared=self.cache.folder/(key+'.npz') if key else None
                self.library.converted(self.current_track_id,doc,prepared if prepared and prepared.is_file() else None);self.library_panel.refresh_tracks()
            except Exception as e:self.log('文件库保存失败：'+str(e))
    def library_job_finished(self):
        if self.exit_pending:return
        if self.pending_track:self.drain_track()
        elif self.autoplay_pending and not self.jobs and not self.loading:
            self.autoplay_pending=False
            if self.link.connected:self.toggle_play()
