from __future__ import annotations
from collections import Counter
from pathlib import Path
import tempfile,unittest

from . import h2d_lifecycle_v10 as c
from llm_infer_model.performance.parameters import UnknownParameter

SCOPE=dict(context_sha256='a'*64,preparation_sha256='b'*64)
EVIDENCE=('profile-only-completed-copy-labels',)

class LifecycleContracts(unittest.TestCase):
    def test_independent_three_axis_grid_and_fresh_processes(self):
        programs={s:dict(target_action_index=2) for s in ('profile','heldout')}
        jobs=c.plans(programs,'c'*64)
        self.assertEqual(len(jobs),12)
        for axis in (c.SIZES,c.IDLES,c.WINDOWS):self.assertFalse(set(axis['profile'])&set(axis['heldout']))
        for split in ('profile','heldout'):
            selected=[j for j in jobs if j['split']==split]
            self.assertEqual(Counter(j['memory_mode'] for j in selected),Counter(registered=3,pageable=3))
            for job in selected:
                r=job['request'];self.assertEqual(len(r['condition_orders']),12)
                wanted=set(range(len(r['conditions'])))
                self.assertTrue(all(set(order)==wanted and len(order)==len(wanted) for order in r['condition_orders']))
                self.assertFalse(r['telemetry_between_bursts'])

    def test_first_and_sustained_windows_use_same_model_not_n_lookup(self):
        theta=[20.,50.,10.,20.,40.,10.,2.,100.]
        a=c.predict('prepared_initial_recovery',theta,SCOPE,'registered',EVIDENCE,dict(bytes=1048576,idle_ms=2,copies=1))
        b=c.predict('prepared_initial_recovery',theta,SCOPE,'registered',EVIDENCE,dict(bytes=1048576,idle_ms=2,copies=128))
        self.assertEqual(a[0],a[1]);self.assertEqual(a[0],b[0]);self.assertGreater(b[1],a[1]);self.assertLess(b[1]/128,b[0])

    def test_integer_freeze_and_smooth_estimation_remain_same_equation(self):
        theta=[20.1,50.3,10.4,20.2,40.5,10.6,2.,100.]
        cell=dict(bytes=524288,idle_ms=10,copies=64)
        a=c.predict('prepared_initial_recovery',theta,SCOPE,'pageable',EVIDENCE,cell)
        b=c.predict('prepared_initial_recovery',theta,SCOPE,'pageable',EVIDENCE,cell,integer_clock=True)
        self.assertLess(abs(a[0]-b[0]),1.);self.assertLess(abs(a[1]-b[1]),64.)

    def test_context_and_unknown_payload_do_not_silently_fallback(self):
        m=c.model('zero_initial_recovery',[20.,50.,50.,30.,2.,100.],SCOPE,'registered',EVIDENCE)
        with self.assertRaises(UnknownParameter):m.predict_burst(1048576,2000000,8,context_sha256='d'*64,preparation_sha256='b'*64,require_qualified=False)
        with self.assertRaises(UnknownParameter):m.predict_burst(16777216,2000000,8,context_sha256='a'*64,preparation_sha256='b'*64,require_qualified=False)

    def test_heldout_marker_blocks_refit_before_reading_any_data(self):
        with tempfile.TemporaryDirectory(dir=c.ROOT/'build') as tmp:
            path=Path(tmp);(path/'heldout-started.json').write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'single target-blind fit'):c.fit_and_freeze(path)
            self.assertFalse((path/'fit.json').exists())

    def test_cv_is_reported_not_used_as_primary_prediction_gate(self):
        p=c.protocol();self.assertFalse(p['acceptance']['raw_CV_is_prediction_gate'])
        self.assertTrue(p['old_v9_failures_preserved']);self.assertEqual(p['acceptance']['primary_complete_window_condition_mean_APE_max'],.10)
        self.assertEqual(p['acceptance']['primary_first_COPY_condition_mean_APE_max'],.10)

if __name__=='__main__':unittest.main()
