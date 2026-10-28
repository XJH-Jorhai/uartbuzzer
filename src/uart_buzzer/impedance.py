import numpy as np
from scipy.optimize import least_squares
from .io_data import dump, write_csv

NAMES = ['Rcu_ohm','Linf_H','Reddy_ohm','Leddy_H','Rmot_ohm','Lmot_H','Cmot_F']
SEED = np.array([40.80377,2.262e-3,92.06,1.958e-3,40.69,142.3e-6,38.26e-6])


def impedance(f, p, model='M1'):
    s = 2j*np.pi*np.asarray(f)
    if model == 'M0':
        return p[0]+s*p[1]
    r,l,re,le,rm,lm,cm = p[:7]
    z = r+s*l+1/(1/re+1/(s*le))+1/(1/rm+1/(s*lm)+s*cm)
    if model == 'M2':
        r2,l2,c2 = p[7:]
        z += 1/(1/r2+1/(s*l2)+s*c2)
    return z


def fit(groups, dcr, out, seed=20261028, starts=8, bootstraps=32):
    rng = np.random.default_rng(seed)
    # Fixed DCR is an exact constraint for optimization, not a claim of exact measurement.
    sigma_dcr = max(.05, np.std(dcr, ddof=1))
    allfits, summaries = {}, {}
    for model in ('M0','M1','M2'):
        initial = SEED[1:2] if model=='M0' else SEED[1:]
        if model=='M2':
            initial = np.r_[initial, 4., .3e-3, 5e-6]
        lo = np.array([1e-5] if model=='M0' else [1e-5,.1,1e-6,.1,1e-7,1e-9])
        hi = np.array([.1] if model=='M0' else [.1,1e5,.1,1e4,.1,.01])
        if model=='M2':
            lo=np.r_[lo,.01,1e-7,1e-9]; hi=np.r_[hi,1e4,.1,.01]
        masks = {k: np.arange(len(g['f']))%5 != 2 for k,g in groups.items()}
        def residual(x, selected):
            p=np.r_[np.mean(dcr),np.exp(x)]
            parts=[]
            for k,g in groups.items():
                idx=selected[k]
                f,z=g['f'][idx],g['z'][idx]
                # Equal total weight per dataset and log-frequency quadrature within each.
                base=np.gradient(np.log(g['f']))
                w=np.sqrt(base[idx]/base[idx].sum())
                diff=(impedance(f,p,model)-z)/np.maximum(abs(z),40)
                parts.extend((diff.real*w*10,diff.imag*w*10))
            return np.concatenate(parts)
        solutions=[]
        for n in range(starts):
            x=np.log(initial)+(rng.normal(0,.35,len(initial)) if n else 0)
            sol=least_squares(residual,np.clip(x,np.log(lo)+1e-6,np.log(hi)-1e-6),
                              args=(masks,),bounds=(np.log(lo),np.log(hi)),loss='soft_l1',f_scale=.03,max_nfev=3500)
            solutions.append(sol)
        best=min(solutions,key=lambda s: s.cost)
        p=np.r_[np.mean(dcr),np.exp(best.x)]
        allfits[model]=p
        sv=np.linalg.svd(best.jac,compute_uv=False)
        cov=np.linalg.pinv(best.jac.T@best.jac)*np.sum(best.fun**2)/max(1,len(best.fun)-len(best.x))
        diag=np.sqrt(np.maximum(np.diag(cov),1e-30))
        corr=cov/diag[:,None]/diag[None,:]
        info=dict(parameters=dict(zip(NAMES+(['Rmot2_ohm','Lmot2_H','Cmot2_F'] if model=='M2' else []),p)),
                  dcr_constraint='fixed measured mean', dcr_assumed_uncertainty_ohm=sigma_dcr,
                  jacobian_condition=float(sv[0]/sv[-1]),log_parameter_correlation=corr,
                  log_covariance_approx=cov, multistart_costs=[s.cost for s in solutions], datasets={})
        if model=='M0':
            info['parameters']=dict(Rcu_ohm=p[0],L_H=p[1])
        for k,g in groups.items():
            pred=impedance(g['f'],p,model)
            e=abs(pred-g['z'])/abs(g['z'])
            info['datasets'][k]=dict(train_mean=float(e[masks[k]].mean()),
                                     holdout_mean=float(e[~masks[k]].mean()),
                                     quantiles=dict(zip(['p50','p90','p95','max'],np.quantile(e,[.5,.9,.95,1]))),
                                     bands={f'{a}-{b}':float(e[(g['f']>=a)&(g['f']<=b)].mean()) for a,b in
                                            [(100,1900),(1900,2400),(2400,10000)] if np.any((g['f']>=a)&(g['f']<=b))})
            write_csv(out/f'{model}_{k}_residuals.csv', [dict(f_Hz=f, measured_R=z.real,measured_X=z.imag,
                      model_R=v.real,model_X=v.imag,relative_complex_error=er,split='train' if tr else 'holdout')
                      for f,z,v,er,tr in zip(g['f'],g['z'],pred,e,masks[k])])
        summaries[model]=info
    # M2 must improve BOTH holdouts by at least 15%, and retain reasonable conditioning.
    use_m2=all(summaries['M2']['datasets'][k]['holdout_mean']<.85*summaries['M1']['datasets'][k]['holdout_mean'] for k in groups)
    use_m2 &= summaries['M2']['jacobian_condition'] < 1e6
    selected='M2' if use_m2 else 'M1'
    p=allfits[selected]
    # Pairs bootstrap within each training set; captures sample scatter, not model discrepancy.
    samples=[]
    model=selected
    # Closure now uses selected model and its bounds after the M2 loop.
    lo=np.array([1e-5,.1,1e-6,.1,1e-7,1e-9]+([.01,1e-7,1e-9] if selected=='M2' else []))
    hi=np.array([.1,1e5,.1,1e4,.1,.01]+([1e4,.1,.01] if selected=='M2' else []))
    for _ in range(bootstraps):
        indices={k:rng.choice(np.flatnonzero(m),np.count_nonzero(m),replace=True) for k,m in masks.items()}
        s=least_squares(residual,np.log(p[1:]),args=(indices,),bounds=(np.log(lo),np.log(hi)),loss='soft_l1',f_scale=.03,max_nfev=700)
        samples.append(np.r_[rng.normal(np.mean(dcr),sigma_dcr),np.exp(s.x)])
    write_csv(out/'bootstrap_parameters.csv',[dict(zip(NAMES+(['Rmot2_ohm','Lmot2_H','Cmot2_F'] if selected=='M2' else []),v)) for v in samples])
    fine=groups['fine']; f=fine['f']; z=fine['z']; v=impedance(f,p,selected)
    peaks={}
    for label,func in [('R_peak',lambda x:np.argmax(x.real)),('Z_peak',lambda x:np.argmax(abs(x))),
                       ('Z_valley_after_peak',lambda x:np.argmax(abs(x))+np.argmin(abs(x)[np.argmax(abs(x)):]))]:
        a,b=int(func(z)),int(func(v)); peaks[label]=dict(measured_Hz=f[a],model_Hz=f[b],error_Hz=f[b]-f[a])
    result=dict(selected=selected, selection_rule='both holdouts improve >=15%, condition <1e6',
                seed=seed,models=summaries,selected_parameters=p,peaks=peaks,
                bootstrap_95pct=np.quantile(samples,[.025,.975],axis=0),
                scope='100mV setpoint,100Hz-10kHz,unbiased; bootstrap excludes systematic/model uncertainty')
    dump(out/'fit.json',result)
    return p,selected,result


def network(p, model='M1'):
    r,l,re,le,rm,lm,cm=p[:7]
    lines=[f'Rcu load n1 {r:.12g}',f'Linf n1 n2 {l:.12g}',f'Reddy n2 n3 {re:.12g}',
           f'Leddy n2 n3 {le:.12g}',f'Rmot n3 sh {rm:.12g}',f'Lmot n3 sh {lm:.12g}',f'Cmot n3 sh {cm:.12g}']
    if model=='M2':
        lines=[s.replace('n3 sh','n3 n4') for s in lines]
        r2,l2,c2=p[7:];lines += [f'Rmot2 n4 sh {r2}',f'Lmot2 n4 sh {l2}',f'Cmot2 n4 sh {c2}']
    return lines
