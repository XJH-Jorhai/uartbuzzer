import numpy as np


def nondominated(rows):
    # Maximize mechanical proxy, minimize copper/TX/device losses and switch stress.
    costs=np.array([[-r.get('acoustic_proxy_V_peak',0),r['Pcu_W'],r['TX_abs_avg_A'],
                     r['Pswitch_W']+r['Pdiode_W'],r['Vswitch_abs_peak_V'],r.get('delay_max_s',0)] for r in rows])
    return [r | {'pareto':True,'hard_gate':'unverified: TX/module, power-loss isolation, coil thermal rating'}
            for i,r in enumerate(rows) if not np.any(np.all(costs<=costs[i],axis=1)&np.any(costs<costs[i],axis=1))]


def sensitivity_configs(seed=20261028,n=24):
    rng=np.random.default_rng(seed)
    result=[]
    for top in ('pnp','pmos','dual_nmos'):
        for _ in range(n):
            result.append(dict(topology=top,rcu=float(rng.choice([37,40.80377,47])),
                supply=float(rng.uniform(4.75,5.25)),rtx=float(rng.uniform(20,150)),
                cscale=float(rng.uniform(.7,1.5)),vth=float(rng.uniform(.7,1.3)),kp=float(rng.uniform(2.7,5)),
                beta=float(rng.uniform(60,300)),tf=float(10**rng.uniform(-8,-6.3)),
                tr=float(10**rng.uniform(-7,-5.5)),lscale=float(rng.uniform(.8,1.2)),
                nm_vth=float(rng.uniform(1,2.5)),diode_is=float(10**rng.uniform(-9,-8.1)),
                rg=220.,rlimit=float(rng.choice([0,33,68]))))
    return result
