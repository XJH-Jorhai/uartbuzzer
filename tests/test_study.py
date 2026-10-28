import json
from pathlib import Path
import tempfile
import uuid
import unittest
import numpy as np
from uart_buzzer.uart_codec import frame_bits,CANDIDATES,encode,events,tone_target
from uart_buzzer.impedance import impedance,network,SEED,fit
from uart_buzzer.io_data import observations,read_csv,discover
from uart_buzzer.metrics import average,electrical
from uart_buzzer.spice import deck,Engine,simulate,ac_check

ROOT=Path(__file__).resolve().parents[1]


class UARTTests(unittest.TestCase):
    def test_every_byte_lsb_order_and_stop(self):
        for value in range(256):
            actual=frame_bits(bytes([value])).tolist()
            self.assertEqual(actual,[0]+[(value>>bit)&1 for bit in range(8)]+[1])

    def test_forced_density(self):
        self.assertAlmostEqual((1-frame_bits(bytes([255])*100)).mean(),.1)
        self.assertAlmostEqual((1-frame_bits(bytes([0])*100)).mean(),.9)

    def test_physical_durations(self):
        for baud in (230400,460800,921600):
            w,end=events(bytes([0x55,0xAA]),baud,start=0)
            self.assertTrue(np.all(np.diff(w[:,0])>0))
            centers=(np.arange(20)+.5)/baud
            self.assertTrue(np.array_equal((np.interp(centers,w[:,0],w[:,1])>2.5).astype(int),frame_bits(bytes([0x55,0xAA]))))
            self.assertAlmostEqual(end,20/baud+.0005)

    def test_stop_idle_is_true_off(self):
        w,d=events(b'',921600)
        self.assertTrue(np.all(w[:,1]==5))
        self.assertEqual(len(frame_bits(b'')),0)

    def test_rest_and_jitter_gaps_are_idle(self):
        w,end=events(bytes([0,0]),460800,start=0,gaps={1:.001})
        self.assertEqual(np.interp(.0005,w[:,0],w[:,1]),5)
        self.assertAlmostEqual(end,20/460800+.001+.0005)

    def test_encoders_real_bytes_and_integral_state(self):
        target=tone_target(2000,921600,.002)
        for method in ('naive_byte_density','frame_aware_pdm','frame_aware_sigma_delta'):
            b,error,cost=encode(target,method)
            self.assertIsInstance(b,bytes)
            self.assertEqual(len(b)*10,len(target))
            self.assertAlmostEqual(error,np.sum(target-(1-frame_bits(b))),places=8)

    def test_sigma_delta_state_can_be_carried(self):
        target=tone_target(2150,460800,.004)
        n=(len(target)//20)*10
        whole,err,_=encode(target,'frame_aware_sigma_delta')
        a,state,_=encode(target[:n],'frame_aware_sigma_delta')
        b,end,_=encode(target[n:],'frame_aware_sigma_delta',state)
        self.assertEqual(whole,a+b);self.assertAlmostEqual(err,end)

    def test_sigma_delta_finite_under_extreme_density(self):
        for val in (.1,.9,.5):
            b,e,_=encode(np.full(10000,val),'frame_aware_sigma_delta')
            self.assertLess(abs(e),1.01)


class NetworkTests(unittest.TestCase):
    def test_orientation_and_flyback_sampling_loop(self):
        p=deck(SEED,'M1','DC 5',.001)
        self.assertIn('Mhigh out gate emit emit',p)
        self.assertIn('Dbody out emit',p)
        self.assertIn('Vfly 0 an',p)
        self.assertIn('Dfly an cl',p)
        self.assertIn('Rsh sh 0 1',p)
        self.assertNotIn('Dfly sh',p)
        pnp=deck(SEED,'M1','DC 5',.001,{'topology':'pnp'})
        self.assertIn('Qhigh out gate emit',pnp)

    def test_clamp_present_in_unpowered_case(self):
        p=deck(SEED,'M1','DC 0',.001,{'tx_mode':'unpowered'})
        self.assertIn('Dtx txclamp vio DCLAMP',p)
        self.assertIn('Vvio vio 0 0',p)
        self.assertIn('1000000000000.0',p)

    def test_passive_positive_real_over_extended_band(self):
        f=np.geomspace(.01,1e8,10000)
        self.assertTrue(np.all(impedance(f,SEED).real>0))
        self.assertAlmostEqual(impedance([1e-8],SEED)[0].real,SEED[0],places=8)

    def test_rms_is_time_weighted_and_includes_off(self):
        t=np.array([0,.25,.5,1.])
        i=np.full(4,.05)
        self.assertAlmostEqual(np.sqrt(average(t,i*i)),.05)
        self.assertAlmostEqual(average(t,i*i)*40,.1)

    def test_negative_power_not_created_by_passive_rlc(self):
        rng=np.random.default_rng(1)
        for _ in range(20):
            p=SEED*np.exp(rng.normal(0,.5,len(SEED)))
            self.assertTrue(np.all(impedance(np.geomspace(1,1e6,500),p).real>0))


class SpiceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine=Engine()
        cls.folder=ROOT/'results/test_runs'/uuid.uuid4().hex[:12]
        cls.folder.mkdir(parents=True)

    @classmethod
    def tearDownClass(cls):pass  # Keep decks and logs as verification evidence.

    def run_case(self,name,wave,cfg,dt=1e-7,duration=.004):
        return simulate(self.engine,SEED,'M1',wave,duration,self.folder/(name+'.cir'),cfg,dt,False)

    def test_ac_independent_spice_agreement(self):
        report=ac_check(self.engine,SEED,'M1',self.folder)
        self.assertLess(report['max_error_ohm'],1e-5)

    def test_5V_polarity_for_both_direct_drivers(self):
        for top in ('pnp','pmos'):
            high=self.run_case(top+'high','DC 5',{'topology':top})
            low=self.run_case(top+'low','DC 0',{'topology':top})
            self.assertLess(abs(high['coil'][-1]),1e-6)
            self.assertGreater(low['coil'][-1],.1)
            if top=='pmos':
                self.assertLess(low['vcontrol'][-1],-4)
                self.assertLess(abs(high['vcontrol'][-1]),.01)

    def test_flyback_off_current_shunt_and_energy(self):
        a=self.run_case('flyback','PULSE(5 0 500u 10n 10n 250u 500u)',{'topology':'pmos'})
        self.assertLess(np.max(abs(a['sh']-a['coil'])),1e-8)
        off=(a['txideal']>4.9)&(a['time']>.001)
        self.assertGreater(a['diode'][off].max(),.02)
        self.assertGreater(a['coil'][off].max(),.02)
        r=electrical(a,SEED,{'topology':'pmos'},.002,.004)
        self.assertLess(r['energy_relative_error'],.003)
        self.assertGreater(r['Pcu_W'],r['Iavg_A']**2*SEED[0])

    def test_external_supply_dead_USB_is_dangerous(self):
        for top in ('pnp','pmos'):
            a=self.run_case(top+'dead','DC 0',{'topology':top,'vio':0,'tx_mode':'unpowered'})
            self.assertGreater(a['coil'][-1],.1)
            self.assertGreater(a['backfeed'][-1],1e-4)

    def test_highZ_live_supply_off(self):
        for top in ('pnp','pmos','dual_nmos'):
            a=self.run_case(top+'highz','DC 0',{'topology':top,'tx_mode':'highZ'})
            self.assertLess(abs(a['coil'][-1]),1e-5)

    def test_timestep_convergence_and_gate_load(self):
        w,d=events(bytes([0x55,0xAA])*5,921600)
        c={'topology':'pmos','rg':220}
        a=self.run_case('edge50',w,c,5e-8,d);b=self.run_case('edge25',w,c,2.5e-8,d)
        uniform=np.arange(.00051,d-.0005,2e-8)
        ia=np.interp(uniform,a['time'],a['coil']);ib=np.interp(uniform,b['time'],b['coil'])
        self.assertLess(np.sqrt(np.mean((ia-ib)**2)),.0003)
        self.assertGreater(abs(a['control_i']).max(),.005)
        self.assertLess(abs(a['vcontrol']).max(),5.3)

    def test_second_branch_spice_energy(self):
        p=np.r_[SEED,6.,1e-4,66e-6]
        a=simulate(self.engine,p,'M2','PULSE(5 0 500u 10n 10n 250u 500u)',.004,
                   self.folder/'second_branch.cir',{'topology':'pmos'},1e-7,False)
        result=electrical(a,p,{'topology':'pmos'},.002,.004)
        self.assertLess(result['energy_relative_error'],.003)


class FittingTests(unittest.TestCase):
    def test_fit_positive_and_seed_reproducible(self):
        import uuid
        folder=ROOT/'results/test_runs'/uuid.uuid4().hex[:12];folder.mkdir(parents=True)
        chosen,_=discover(ROOT,folder)
        groups={k:observations(chosen[k]) for k in ('wide','fine')}
        dcr=[float(r['DCR_ohm']) for r in read_csv(chosen['dcr'])]
        a,am,ra=fit(groups,dcr,folder,starts=1,bootstraps=2)
        other=folder/'repeat';other.mkdir()
        b,bm,rb=fit(groups,dcr,other,starts=1,bootstraps=2)
        self.assertEqual(am,bm)
        np.testing.assert_allclose(a,b,rtol=1e-10)
        self.assertTrue(np.all(a>0))
        self.assertLess(ra['models'][am]['datasets']['fine']['holdout_mean'],.025)


if __name__=='__main__':unittest.main()
