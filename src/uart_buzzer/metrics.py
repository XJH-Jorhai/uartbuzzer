import numpy as np
from scipy.integrate import trapezoid


def average(t,y):
    return float(trapezoid(y,t)/(t[-1]-t[0]))


def window(a,start,end=None):
    t=a['time'];end=t[-1] if end is None else end
    use=(t>start)&(t<end)
    # Interpolate endpoints so every reported steady window has its declared duration.
    tt=np.r_[start,t[use],end]
    return {k:np.interp(tt,t,v) for k,v in a.items() if len(v)==len(t)} | {'time':tt}


def electrical(a,p,c,start=.006,end=None):
    w=window(a,start,end);t=w['time'];i=w['coil']
    mean=lambda y:average(t,y)
    r=p[0] if c.get('rcu') is None else c['rcu']
    loss_cu=mean(i*i)*r
    loss_eddy=mean((w['n2']-w['n3'])**2)/p[2]
    loss_mot=mean((w['n3']-w['n4'])**2)/p[4]
    if len(p)>7:
        loss_mot+=mean((w['n4']-w['sh'])**2)/p[7]
    vgs=w['gate']-w['emit'];vds=w['out']-w['emit']
    # Vs supplies positive E/S current; base/gate current positive into control terminal.
    p_switch=mean(-vds*(w['switch_i']+w['device_control_i'])+vgs*w['device_control_i'])
    # Gate capacitance energy included in this terminal balance; mean is meaningful in periodic steady state.
    p_diode=mean((w['an']-w['cl'])*w['diode'])
    p_lim=mean(i*i)*max(c.get('rlimit',0),.001)
    p_shunt=mean(i*i)
    # Full passive network energy audit (load + limiter + shunt + diode + clamp).
    input_power=mean(w['out']*(i-w['diode']))
    p_clamp=mean(w['diode']**2)*max(c.get('clamp_r',0),.001)
    energy=.5*(p[1]*c.get('lscale',1)*w['ilinf']**2+p[3]*c.get('lscale',1)*w['ileddy']**2+
               p[5]*c.get('lscale',1)*w['ilmot']**2+p[6]*(w['n3']-w['n4'])**2)
    if len(p)>7:
        energy+=.5*(p[8]*w['ilmot2']**2+p[9]*(w['n4']-w['sh'])**2)
    storage_rate=(energy[-1]-energy[0])/(t[-1]-t[0])
    loss_sum=loss_cu+loss_eddy+loss_mot+p_lim+p_shunt+p_diode+p_clamp
    off=w['txideal']>c.get('vio',5)*.9
    startup=window(a,0,min(.001,a['time'][-1]))
    return dict(Iavg_A=mean(i),Irms_A=np.sqrt(mean(i*i)),Ipeak_A=float(np.max(a['coil'])),
                startup_Irms_A=np.sqrt(average(startup['time'],startup['coil']**2)),
                startup_Ipeak_A=float(np.max(a['coil'][a['time']<min(start,a['time'][-1])])),
                Pcu_W=loss_cu,Peddy_W=loss_eddy,Pmot_W=loss_mot,Pswitch_W=p_switch,
                Pdiode_W=p_diode,Plimit_W=p_lim,Pshunt_W=p_shunt,
                diode_peak_A=float(np.max(a['diode'])),diode_avg_A=mean(w['diode']),
                diode_reverse_peak_A=float(-np.min(a['diode'])),
                diode_reverse_V=float(np.max(a['cl']-a['an'])),
                TX_sink_peak_A=float(max(0,np.max(-a['tx_i']+a['backfeed']))),TX_source_peak_A=float(max(0,np.max(a['tx_i']))),
                TX_backfeed_peak_A=float(max(0,np.max(a['backfeed']))),
                TX_abs_avg_A=mean(abs(w['tx_i'])),control_peak_A=float(np.max(abs(a['control_i']))),
                device_control_peak_A=float(np.max(abs(a['device_control_i']))),
                device_output_peak_A=float(np.max(abs(a['device_output_i']))),
                Vcontrol_abs_peak_V=float(np.max(abs(a['vcontrol']))),Vswitch_abs_peak_V=float(np.max(abs(a['vce_ds']))),
                shunt_error_A=float(np.max(abs(a['sh']-a['coil']))),
                OFF_Irms_A=np.sqrt(mean(np.where(off,i*i,0))),
                port_avg_A=mean(w['supply_i']),passive_input_W=input_power,passive_loss_W=loss_sum,
                storage_rate_W=storage_rate,energy_error_W=input_power-loss_sum-storage_rate,
                energy_relative_error=abs(input_power-loss_sum-storage_rate)/max(abs(input_power),1e-9),
                diode_25C_selected_datasheet_limit_pass=bool(np.max(a['diode'])<.45 and mean(w['diode'])<.2 and p_diode<.5 and np.max(a['cl']-a['an'])<100),
                switch_voltage_limit_pass=(bool(np.max(abs(a['vce_ds']))<30 and np.max(abs(a['vcontrol']))<12) if c['topology']=='pmos' else 'unverified_identity'),
                TX_assumed_5mA_peak_budget_pass=bool(np.max(abs(a['tx_i']))<.005),
                window_start_s=start,window_end_s=float(t[-1]),absolute_SPL_dB='unknown',thermal_safety='unverified')


def spectrum(a,f,start,end,sample_rate=200000,acoustic_proxy='motional_voltage'):
    t=np.arange(start,end,1/sample_rate)
    y=np.interp(t,a['time'],a['coil'])
    # Simultaneous sine/cos LS tolerates noncoherent tones, unlike a single FFT bin.
    X=np.column_stack([np.ones(len(t))]+[g(2*np.pi*k*f*t) for k in range(1,6) for g in (np.sin,np.cos)])
    coef=np.linalg.lstsq(X,y,rcond=None)[0];amps=np.hypot(coef[1::2],coef[2::2])
    residual=y-X@coef
    win=np.hanning(len(t));ff=np.fft.rfftfreq(len(t),1/sample_rate)
    fft=abs(np.fft.rfft((y-y.mean())*win))*2/win.sum()
    rf=abs(np.fft.rfft(residual*win))*2/win.sum()
    # Window ENBW normalization; proxy uses total motional-branch voltage, not SPL.
    band=(ff>=100)&(ff<=10000)
    noise=np.sqrt(np.sum(rf[band]**2)/(2*1.5))
    proxies={'motional_voltage':a['n3']-a['n4'],'motional_inductor_current':a['ilmot']}
    if acoustic_proxy not in proxies:
        raise ValueError('Unknown acoustic_proxy; choose '+str(list(proxies)))
    motion=np.interp(t,a['time'],proxies[acoustic_proxy])
    mcoef=np.linalg.lstsq(X,motion,rcond=None)[0]
    return dict(fundamental_peak_A=amps[0],THD_2to5=float(np.linalg.norm(amps[1:])/max(amps[0],1e-12)),
                residual_100Hz_10kHz_rms_A=noise,
                inband_residual_to_fundamental=float(noise/(max(amps[0],1e-12)/np.sqrt(2))),
                acoustic_proxy_V_peak=float(np.hypot(mcoef[1],mcoef[2])),
                acoustic_proxy_definition=acoustic_proxy+'; uncalibrated candidate movement proxy'),(ff,fft)


def edge_metrics(a,baud,vio=5.):
    t=np.arange(a['time'][0],a['time'][-1],min(1/baud/80,2e-8))
    tx=np.interp(t,a['time'],a['txideal'])<vio/2
    # Collector/drain threshold follows physical delivered output, not only gate threshold.
    out=np.interp(t,a['time'],a['out'])>2.5
    edges=np.flatnonzero(tx[1:]!=tx[:-1])+1
    delays=[];miss=0
    for j,e in enumerate(edges):
        stop=edges[j+1] if j+1<len(edges) else len(t)
        q=np.flatnonzero(out[e:stop]==tx[e])
        if len(q):delays.append(t[e+q[0]]-t[e])
        else:miss+=1
    return dict(edge_count=len(edges),missing_edges=miss,
                delay_max_s=max(delays,default=0.),delay_median_s=float(np.median(delays)) if delays else 0.,
                delivered_ON_density=float(out.mean()),command_ON_density=float(tx.mean()),
                bit_width_s=1/baud)
