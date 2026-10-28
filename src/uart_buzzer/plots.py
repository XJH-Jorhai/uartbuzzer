import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .impedance import impedance

plt.rcParams.update({'figure.dpi':140,'axes.grid':True,'grid.alpha':.2,'font.size':10})


def impedance_plots(groups,p,model,out):
    for zoom in (False,True):
        fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
        f=np.linspace(1900,2400,2500) if zoom else np.geomspace(100,10000,4000)
        z=impedance(f,p,model)
        funcs=[lambda z:z.real,lambda z:z.imag,abs,lambda z:np.rad2deg(np.angle(z))]
        for ax,func,label in zip(axes.flat,funcs,['R (ohm)','X (ohm)','|Z| (ohm)','Phase (deg)']):
            ax.plot(f,func(z),label=model)
            for name,g in groups.items():
                use=(g['f']>=f[0])&(g['f']<=f[-1])
                ax.plot(g['f'][use],func(g['z'][use]),'.',ms=3,label=name)
            ax.set(xlabel='Frequency (Hz)',ylabel=label,xlim=(f[0],f[-1]))
            if not zoom:ax.set_xscale('log')
        axes[0,0].legend();fig.suptitle('100 mV setpoint: measured vs passive fitted network')
        fig.savefig(out/('impedance_zoom.png' if zoom else 'impedance_wide.png'));plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,6),constrained_layout=True)
    for name,g in groups.items():
        z=impedance(g['f'],p,model)-g['z']
        axes[0].plot(g['f'],z.real,'.-',label=name+' R');axes[0].plot(g['f'],z.imag,'.-',label=name+' X')
        axes[1].plot(g['f'],100*abs(z)/abs(g['z']),'.-',label=name)
    axes[0].set(ylabel='Residual (ohm)',xscale='log');axes[1].set(ylabel='Complex error (%)',xlabel='Frequency (Hz)',xscale='log')
    for ax in axes:ax.legend()
    fig.savefig(out/'fit_residuals.png');plt.close(fig)


def wave_plot(a,path,title,limits=None):
    fig,axes=plt.subplots(4,1,figsize=(11,8),sharex=True,constrained_layout=True)
    t=a['time']*1000
    axes[0].plot(t,a['tx'],label='TX');axes[0].plot(t,a['out'],label='collector/drain');axes[0].plot(t,a['gate'],label='base/gate')
    axes[1].plot(t,a['coil']*1000,label='coil / shunt');axes[1].plot(t,a['diode']*1000,label='1N4148');axes[1].plot(t,a['supply_i']*1000,label='5V port',alpha=.6)
    axes[2].plot(t,a['tx_i']*1000,label='TX source (+) / sink (-)');axes[2].plot(t,a['control_i']*1000,label='base/gate')
    axes[3].plot(t,a['vce_ds'],label='VCE / VDS');axes[3].plot(t,a['vcontrol'],label='VBE / VGS')
    for ax,label in zip(axes,['Voltage (V)','Current (mA)','Current (mA)','Voltage (V)']):
        ax.set(ylabel=label);ax.legend(loc='upper right',ncol=3,fontsize=8)
        if limits:ax.set_xlim(np.array(limits)*1000)
    axes[-1].set_xlabel('Time (ms)');fig.suptitle(title);fig.savefig(path);plt.close(fig)


def pareto_plot(rows,out):
    fig,ax=plt.subplots(figsize=(9,6),constrained_layout=True)
    for top in sorted(set(r['topology'] for r in rows)):
        use=[r for r in rows if r['topology']==top]
        ax.scatter([r['Pcu_W']*1000 for r in use],[r.get('acoustic_proxy_V_peak',0) for r in use],label=top,alpha=.7,s=20)
    ax.set(xlabel='Coil copper loss (mW)',ylabel='acoustic_proxy: motional voltage peak (V)',title='Exploratory tradeoffs; no candidate is hardware-certified')
    ax.legend();fig.savefig(out/'pareto.png');plt.close(fig)
