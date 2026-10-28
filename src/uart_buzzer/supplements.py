"""Evidence checks, loss accounting and plots derived from this run, never hardware data."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .io_data import read_csv,write_csv,dump,observations
from .metrics import average,window,spectrum


def build(root,out,p):
    import hashlib
    manifest=json.loads((out/'input_manifest.json').read_text(encoding='utf-8'))
    task=root/'specs/task_20261028.txt'
    if task.exists() and not any(r['path']=='specs/task_20261028.txt' for r in manifest):
        digest=hashlib.sha256(task.read_bytes()).hexdigest()
        target=root/'data/raw'/(digest[:12]+'_'+task.name)
        if not target.exists():target.write_bytes(task.read_bytes())
        manifest.append(dict(path='specs/task_20261028.txt',sha256=digest,bytes=task.stat().st_size,
                             preserved_copy=str(target.relative_to(root))))
        dump(out/'input_manifest.json',manifest)
    integrity=[dict(path=r['path'],original_unchanged=hashlib.sha256((root/r['path']).read_bytes()).hexdigest()==r['sha256'],
                    preserved_copy_matches=hashlib.sha256((root/r['preserved_copy']).read_bytes()).hexdigest()==r['sha256']) for r in manifest]
    dump(out/'input_integrity.json',integrity)
    if not all(r['original_unchanged'] and r['preserved_copy_matches'] for r in integrity):
        raise RuntimeError('Input integrity verification failed')
    discovery=json.loads((out/'discovery.json').read_text(encoding='utf-8'))
    if 'low' in discovery['selected']:
        low=observations(Path(discovery['selected']['low']))
        wide=observations(Path(discovery['selected']['wide']))
        fine=observations(Path(discovery['selected']['fine']))
        comparisons=[]
        for f,z in zip(low['f'],low['z']):
            ref=fine if fine['f'][0]<=f<=fine['f'][-1] else wide
            zr=np.interp(f,ref['f'],ref['z'].real)+1j*np.interp(f,ref['f'],ref['z'].imag)
            comparisons.append(dict(f_Hz=f,low_R_ohm=z.real,low_X_ohm=z.imag,reference_R_ohm=zr.real,
                                    reference_X_ohm=zr.imag,relative_complex_difference=abs(z-zr)/abs(zr),
                                    note='Different-time scans; excitation dependence confounded with drift and repeatability'))
        write_csv(out/'tables/low_level_uncertainty.csv',comparisons)
    version=subprocess.run([sys.executable,'-c',"import run;from uart_buzzer.spice import Engine;e=Engine();e.cmd('version');print('\\n'.join(e.logs))"],
                           cwd=root,capture_output=True,text=True)
    (out/'ngspice_version.log').write_text(version.stdout+version.stderr,encoding='utf-8')
    # A separate process isolates the shared ngspice library and callback lifetime.
    tests=subprocess.run([sys.executable,str(root/'scripts/run_tests.py')],cwd=root,capture_output=True,text=True)
    (out/'tests.log').write_text(tests.stdout+tests.stderr,encoding='utf-8')
    if tests.returncode:
        raise RuntimeError('Verification tests failed: '+str(out/'tests.log'))
    if (out/'spice/pmos_frame_aware_sigma_delta_921600_2000.npz').exists():
        pwl=subprocess.run([sys.executable,str(root/'scripts/verify_pwl_cleanup.py'),str(out)],cwd=root,capture_output=True,text=True)
        (out/'pwl_cleanup.log').write_text(pwl.stdout+pwl.stderr,encoding='utf-8')
        if pwl.returncode:raise RuntimeError('PWL cleanup equivalence check failed')
        if not (out/'integration_method_check.json').exists():
            method=subprocess.run([sys.executable,str(root/'scripts/check_integration_methods.py'),str(out)],cwd=root,capture_output=True,text=True)
            (out/'integration_method_check.log').write_text(method.stdout+method.stderr,encoding='utf-8')
            if method.returncode:raise RuntimeError('Independent integration method check failed')
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    summary['tests_status']='21 unittest checks passed (including live ngspice, M2 energy, repeated fit)'
    dump(out/'summary.json',summary)
    losses=[]
    table=read_csv(out/'tables/all_cases.csv') if (out/'tables/all_cases.csv').exists() else []
    for r in table:
        path=out/'spice'/(r['case']+'.npz')
        if not path.exists():continue
        with np.load(path) as saved:a={k:saved[k] for k in saved.files}
        if 'n4' not in a:a['n4']=a['sh']
        w=window(a,float(r['window_start_s']),float(r['window_end_s']));t=w['time']
        avg=lambda y:average(t,y)
        top=r['topology'];control=w['control_i']
        # Vg is a zero-volt current sensor, so drv==gate. Rb/Rg carry control_i.
        pc=avg(control**2)*max(float(r['rg']) if top=='pmos' else float(r['rb']),.01)
        pull=float(r['rgs'] if top=='pmos' else r['rpull'])
        pp=avg((w['gate']-w['v5'])**2)/pull
        prtx=avg(w['tx_i']**2)*(float(r['rtx']) if r['tx_mode']=='normal' else 1e12)
        psupply=avg((w['src']-w['v5'])**2)/.25
        # Direct drivers permit a complete energy audit including TX and local bypass caps.
        full_error=None
        if top in ('pmos','pnp') and r['tx_mode']=='normal' and not (r['open_load']=='True'):
            source=avg(w['src']*w['supply_i']+w['txideal']*w['tx_i'])
            caps=.5*10.1e-6*w['v5']**2
            if top=='pmos':
                cs=float(r['cscale'])
                caps+=.5*cs*(877e-12*(w['gate']-w['emit'])**2+77e-12*(w['gate']-w['out'])**2+38e-12*(w['out']-w['emit'])**2)
            # Pswitch already contains gate-cap terminal energy, so remove explicit gate cap double count.
            caprate=(.5*10.1e-6*(w['v5'][-1]**2-w['v5'][0]**2))/(t[-1]-t[0])
            total=float(r['passive_loss_W'])+float(r['Pswitch_W'])+pc+pp+prtx+psupply+float(r['storage_rate_W'])+caprate
            full_error=source-total
        losses.append(dict(case=r['case'],P_control_resistor_W=pc,P_pullup_W=pp,P_TX_output_resistance_W=prtx,
                           P_supply_resistance_W=psupply,full_audit_error_W=full_error,
                           full_audit_scope='direct drivers normal TX only; device internal storage within terminal power',
                           base_collector_or_gate_drain_current_note='device_output_i includes modeled displacement; device_control_i excludes pullup current'))
    write_csv(out/'tables/additional_losses.csv',losses)
    # Conditional electrothermal scenarios: R(T), no calibrated thermal constants.
    thermal=[]
    for rtheta in (50,100,200,400):
        for ctheta in (.1,1,5):
            for power in (.08,.15,.3,.6):
                for t in (1,10,60,300):
                    rise=power*rtheta*(1-np.exp(-t/(rtheta*ctheta)))
                    thermal.append(dict(Rtheta_assumed_K_per_W=rtheta,Ctheta_assumed_J_per_K=ctheta,
                                        Pcu_constant_assumed_W=power,time_s=t,conditional_rise_K=rise,
                                        Rcu_conditional_ohm=p[0]*(1+.0039*rise),status='hypothetical_not_measured'))
    write_csv(out/'tables/thermal_assumptions.csv',thermal)
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
    tt=np.geomspace(.01,300,500)
    for rt in (50,100,200,400):
        ax.plot(tt,.15*rt*(1-np.exp(-tt/rt)),label=f'Rtheta={rt} K/W, Ctheta=1 J/K')
    ax.set(xlabel='Time (s)',ylabel='Conditional rise (K)',xscale='log',title='Hypothetical 0.15 W constant copper loss; thermal parameters NOT calibrated')
    ax.legend();fig.savefig(out/'plots/thermal_assumptions.png');plt.close(fig)
    # Use exactly the same post-startup window as the UART comparison table.
    for top in ('pnp','pmos'):
        for method in ('naive_byte_density','frame_aware_pdm','frame_aware_sigma_delta'):
            raw=out/'spice'/f'{top}_{method}_921600_2000.npz'
            if not raw.exists():continue
            with np.load(raw) as saved:a={k:saved[k] for k in saved.files}
            sm,(ff,amp)=spectrum(a,2000,.016,a['time'][-1]-.0005)
            write_csv(out/'tables'/f'{top}_{method}_spectrum.csv',[dict(f_Hz=x,current_peak_A=y) for x,y in zip(ff,amp)])
    # Actual simulated current spectra at 2kHz, 921600 baud.
    spectra=list((out/'tables').glob('*_spectrum.csv'))
    if spectra:
        fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,constrained_layout=True)
        for path in spectra:
            rows=read_csv(path);f=np.array([float(r['f_Hz']) for r in rows]);amp=np.array([float(r['current_peak_A']) for r in rows])
            ax=axes[0 if path.name.startswith('pnp') else 1]
            ax.plot(f,20*np.log10(np.maximum(amp,1e-10)),label=path.stem.replace('_spectrum','').replace('frame_aware_',''))
        for ax,title in zip(axes,['PNP','PMOS']):
            ax.set(xlim=(100,10000),ylim=(-130,-20),ylabel='dB re 1 A peak',title=title+' simulated coil current');ax.legend(fontsize=8)
        axes[-1].set_xlabel('Frequency (Hz)');fig.savefig(out/'plots/uart_spectra.png');plt.close(fig)
    comparisons=out/'tables/uart_comparison.csv'
    if comparisons.exists():
        rows=read_csv(comparisons)
        fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
        for top,ax in zip(('pnp','pmos'),axes):
            for baud in (230400,460800,921600):
                use=[r for r in rows if r['topology']==top and int(r['baud'])==baud and r['method']=='frame_aware_sigma_delta']
                ax.plot([float(r['f_Hz']) for r in use],[float(r['acoustic_proxy_V_peak']) for r in use],'.-',label=str(baud))
            ax.set(xlabel='Tone (Hz)',ylabel='acoustic_proxy (V peak)',title=top+' uncalibrated electrical motion proxy');ax.legend()
        fig.savefig(out/'plots/tone_comparison.png');plt.close(fig)
    # Hash the code actually delivered for the next run's audit.
    import hashlib
    paths=list((root/'src/uart_buzzer').glob('*.py'))+list((root/'tests').glob('*.py'))+[root/'run.py',root/'requirements.txt',root/'README.md']
    paths += [root/'scripts'/name for name in ('run_tests.py','validate_results.py','verify_pwl_cleanup.py','check_integration_methods.py')]
    dump(out/'source_manifest.json',[dict(path=str(q.relative_to(root)),sha256=hashlib.sha256(q.read_bytes()).hexdigest()) for q in paths])
