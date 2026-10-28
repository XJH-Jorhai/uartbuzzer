"""ngspice shared C API or CLI subprocess. No hardware I/O; every deck is retained."""
import ctypes as ct
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
from .impedance import network,impedance
from .io_data import dump


class VecInfo(ct.Structure):
    _fields_=[('name',ct.c_char_p),('type',ct.c_int),('flags',ct.c_short),
              ('real',ct.POINTER(ct.c_double)),('complex',ct.c_void_p),('length',ct.c_int)]


class Engine:
    def __init__(self,dll=None):
        self.logs=[];self.exits=[]
        self.cli=shutil.which('ngspice') if not dll else None
        if self.cli:
            self.lib=None;return
        paths=[dll,os.environ.get('NGSPICE_DLL'),r'C:\Program Files\KiCad\10.0\bin\ngspice.dll']
        path=next((Path(p) for p in paths if p and Path(p).is_file()),None)
        if path is None:
            raise FileNotFoundError('ngspice missing: install CLI or pass --dll/NGSPICE_DLL')
        self.path=path
        self.dll_dir=os.add_dll_directory(str(path.parent)) if os.name=='nt' else None
        self.lib=ct.CDLL(str(path))
        sendtype=ct.CFUNCTYPE(ct.c_int,ct.c_char_p,ct.c_int,ct.c_void_p)
        exittype=ct.CFUNCTYPE(ct.c_int,ct.c_int,ct.c_bool,ct.c_bool,ct.c_int,ct.c_void_p)
        self.send=sendtype(lambda s,i,u:self.logs.append(s.decode(errors='replace')) or 0)
        self.exit=exittype(lambda s,a,b,i,u:self.exits.append(s) or 0)
        self.lib.ngSpice_Init.argtypes=[ct.c_void_p]*7
        self.lib.ngSpice_Command.argtypes=[ct.c_char_p]
        self.lib.ngSpice_Circ.argtypes=[ct.POINTER(ct.c_char_p)]
        self.lib.ngGet_Vec_Info.argtypes=[ct.c_char_p]
        self.lib.ngGet_Vec_Info.restype=ct.POINTER(VecInfo)
        if self.lib.ngSpice_Init(self.send,None,self.exit,None,None,None,None):
            raise RuntimeError('ngSpice_Init failed')

    def cmd(self,s):
        if self.lib.ngSpice_Command(s.encode()):
            raise RuntimeError('ngspice command failed: '+s)

    def run(self,deck,path,vectors,ac=False):
        path=Path(path);path.write_text(deck,encoding='ascii')
        self.logs=[];self.exits=[]
        if self.cli:
            # Portable CLI fallback exports all requested real-valued expressions.
            raw=path.with_suffix('.txt')
            commands=['run','set wr_singlescale','set wr_vecnames','set numdgt=15']
            expressions=[]
            for j,(label,expr) in enumerate(vectors.items()):
                commands.append(f'let export{j} = {expr}');expressions.append(f'export{j}')
            commands += [f'wrdata "{raw.resolve().as_posix()}" '+ ' '.join(expressions),'quit']
            path.write_text(deck.replace('.end',' .control\n'+'\n'.join(commands)+'\n.endc\n.end'),encoding='ascii')
            proc=subprocess.run([self.cli,'-b',str(path)],capture_output=True,text=True)
            self.logs=[proc.stdout,proc.stderr]
            path.with_suffix('.log').write_text('\n'.join(self.logs),encoding='utf-8')
            if proc.returncode or not raw.exists():
                raise RuntimeError(f'SPICE CLI failed: {path}')
            a=np.loadtxt(raw,skiprows=1);raw.unlink()
            return dict(zip(vectors,a[:,1:].T)) | {'time' if not ac else 'frequency':a[:,0]}
        self.cmd('destroy all')
        lines=deck.encode().splitlines();arr=(ct.c_char_p*(len(lines)+1))(*lines,None)
        if self.lib.ngSpice_Circ(arr):
            raise RuntimeError('ngSpice_Circ failed')
        try:
            self.cmd('run')
            result={}
            for label,expr in vectors.items():
                self.cmd('let exported = '+expr)
                ptr=self.lib.ngGet_Vec_Info(b'exported')
                if not ptr or not ptr.contents.real:
                    raise RuntimeError(f'Missing SPICE vector {expr}')
                vec=ptr.contents
                result[label]=np.ctypeslib.as_array(vec.real,shape=(vec.length,)).copy()
            scale='frequency' if ac else 'time'
            self.cmd('let exported_scale = real('+scale+')')
            ptr=self.lib.ngGet_Vec_Info(b'exported_scale')
            if not ptr:
                raise RuntimeError('Missing scale')
            result[scale]=np.ctypeslib.as_array(ptr.contents.real,shape=(ptr.contents.length,)).copy()
            if self.exits or any(('error' in s.lower() or 'timestep too small' in s.lower()) for s in self.logs):
                raise RuntimeError('ngspice error; inspect '+str(path.with_suffix('.log')))
            if not ac and result[scale][-1] < float(deck.split('.tran ')[1].split()[1])*.999:
                raise RuntimeError('Transient did not reach requested end')
            return result
        finally:
            path.with_suffix('.log').write_text('\n'.join(self.logs),encoding='utf-8')


def ac_check(engine,p,model,out):
    deck='Fitted positive passive load AC verification\nItest 0 load AC 1\n'+ '\n'.join(network(p,model))+ '\nVreturn sh 0 0\n.ac lin 1981 100 10000\n.end\n'
    a=engine.run(deck,out/'fitted_ac.cir',dict(R='real(v(load))',X='imag(v(load))'),True)
    expect=impedance(a['frequency'],p,model)
    error=abs(a['R']+1j*a['X']-expect)
    result=dict(points=len(error),max_error_ohm=float(error.max()),positive_real=bool(np.all(a['R']>0)))
    dump(out/'ac_check.json',result);np.savez_compressed(out/'ac_check.npz',**a)
    if error.max()>1e-5:
        raise AssertionError(result)
    return result


DEFAULT=dict(topology='pmos',vio=5.,supply=5.,rtx=33.,rg=100.,rgs=10000.,rb=680.,rpull=4700.,
             rlimit=0.,clamp_r=0.,cscale=1.,vth=.9,kp=3.8,beta=150.,tf=80e-9,tr=300e-9,
             rcu=None,tx_mode='normal',q1pull=1000.,txpull=22000.,nm_vth=1.5,diode_is=4e-9,
             lscale=1.,supply_mode='normal',schottky=False,open_load=False)


def deck(p,model,wave,duration,cfg=None,dt=2e-7):
    c=DEFAULT | (cfg or {})
    p=np.array(p).copy()
    if c['rcu'] is not None:p[0]=c['rcu']
    p[[1,3,5]]*=c['lscale']
    if c['supply_mode']=='normal':
        supply=f'DC {c["supply"]}'
    else:
        supply=f'PWL(0 0 20u {c["supply"]} {duration*.65} {c["supply"]} {duration*.65+20e-6} 0)'
    lines=['UART buzzer provisional device sensitivity model; not manufacturer SPICE',f'V5 src 0 {supply}',
           'Rsupply src v5 0.25','Cbulk v5 0 10u','Cdec v5 0 100n']
    if isinstance(wave,str):
        stim=wave
    else:
        stim='PWL('+ ' '.join(f'{t:.12g} {v:.8g}' for t,v in wave)+')'
    lines += [f'Vtx txideal 0 {stim}',f'Rtx txideal tx {c["rtx"] if c["tx_mode"]=="normal" else 1e12}', 'Vback tx txclamp 0']
    if c['tx_mode']=='unpowered':
        lines += ['Dtx txclamp vio DCLAMP','Vvio vio 0 0']
    if c['tx_mode']=='power_loss':
        lines += ['Dtx txclamp vio DCLAMP',f'Vvio vio 0 PWL(0 {c["vio"]} {duration*.45} {c["vio"]} {duration*.45+1e-6} 0)']
    lines += [f'Vsw v5 emit 0']
    if c['topology']=='pmos':
        lines += [f'Rg tx drv {max(.01,c["rg"])}','Vg drv gate 0',f'Rgs gate v5 {c["rgs"]}',
                  'Mhigh out gate emit emit PMT W=1u L=1u',
                  f'Cgs gate emit {877e-12*c["cscale"]}',f'Cgd gate out {77e-12*c["cscale"]}',
                  f'Cds out emit {38e-12*c["cscale"]}','Dbody out emit DBODY']
    else:
        drive='tx'
        if c['topology']=='dual_nmos':
            lines += [f'Vvio_on vi 0 {c["vio"]}',f'Rtxpull vi tx {c["txpull"]}',f'Rq1 vi nx {c["q1pull"]}',
                      'Mone nx tx 0 0 NMT W=1u L=1u','Mtwo drv_n nx 0 0 NMT W=1u L=1u',
                      'Cq1 tx 0 30p','Cq2 nx 0 30p']
            drive='drv_n'
        if c['topology']=='single_nmos':
            lines += [f'Vvio_on vi 0 {c["vio"]}','Mone drv_n vi tx tx NMT W=1u L=1u','Cq1 vi tx 30p']
            drive='drv_n'
        lines += [f'Rb {drive} drv {c["rb"]}','Vg drv gate 0',f'Rpull gate v5 {c["rpull"]}',
                  'Qhigh out gate emit PNP_TYP']
        if c['schottky']:
            lines += ['Danti out gate DSCH']
    lines += [f'Rlimit out probe {max(c["rlimit"],.001) if not c["open_load"] else 1e12}',
              'Vcoil probe load 0',*network(p,model),'Rsh sh 0 1',
              'Vfly 0 an 0','Dfly an cl DF',f'Rclamp cl out {max(.001,c["clamp_r"])}',
              f'.model PMT PMOS (LEVEL=1 VTO=-{c["vth"]} KP={c["kp"]} LAMBDA=0.01)',
              f'.model NMT NMOS (LEVEL=1 VTO={c["nm_vth"]} KP=0.1 LAMBDA=0.01)',
              f'.model PNP_TYP PNP (IS=1e-13 BF={c["beta"]} VAF=50 IKF=0.5 RB=5 RC=0.3 RE=0.15 CJE=25p CJC=15p TF={c["tf"]} TR={c["tr"]})',
              f'.model DF D (IS={c["diode_is"]} N=1.8 RS=0.6 CJO=4p TT=4n BV=100 IBV=100u)',
              '.model DBODY D (IS=1n N=1.2 RS=0.1 CJO=20p TT=20n BV=30)',
              '.model DCLAMP D (IS=1n N=1 RS=5 CJO=5p)',
              '.model DSCH D (IS=1u N=1.1 RS=1 CJO=10p)',
              '.options reltol=0.001 abstol=1e-10 vntol=1e-6 method=trap',
              f'.tran {dt:.12g} {duration:.12g} 0 {dt:.12g}', '.end']
    return '\n'.join(lines)+'\n'


VECTORS=dict(coil='i(vcoil)',diode='i(vfly)',switch_i='i(vsw)',control_i='i(vg)',backfeed='i(vback)',
             tx_i='-i(vtx)',supply_i='-i(v5)',tx='v(tx)',txideal='v(txideal)',v5='v(v5)',
             out='v(out)',gate='v(gate)',sh='v(sh)',load='v(load)',n1='v(n1)',n2='v(n2)',
             n3='v(n3)',cl='v(cl)',an='v(an)',emit='v(emit)',
             ilinf='i(linf)',ileddy='i(leddy)',ilmot='i(lmot)',src='v(src)')


def simulate(engine,p,model,wave,duration,path,cfg=None,dt=2e-7,save=True):
    c=DEFAULT | (cfg or {})
    vectors=VECTORS | (dict(n4='v(n4)',ilmot2='i(lmot2)') if model=='M2' else {})
    a=engine.run(deck(p,model,wave,duration,c,dt),path,vectors)
    if model!='M2':a['n4']=a['sh'].copy()
    a['vcontrol']=a['gate']-a['emit']
    a['vce_ds']=a['out']-a['emit']
    pull=c['rgs'] if c['topology']=='pmos' else c['rpull']
    a['device_control_i']=a['control_i']-(a['gate']-a['v5'])/pull
    a['device_output_i']=a['switch_i']+a['device_control_i']
    if save:
        np.savez_compressed(Path(path).with_suffix('.npz'),**a)
    return a
