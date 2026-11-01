from __future__ import annotations

from collections import Counter
from pathlib import Path
import tempfile
import unittest

from . import h2d_bare_v9 as campaign
from . import h2d_bare_v9_fit as fit


SCOPE={'context_sha256':'a'*64,'preparation_sha256':'b'*64}
EVIDENCE=('owned-profile-labels-only',)


class BareProtocolTests(unittest.TestCase):
    def test_every_declared_order_is_position_balanced(self):
        for width in (2,3,4):
            for round_id in range(3):
                rows=campaign.orders(width,round_id)
                self.assertTrue(all(set(row)==set(range(width)) for row in rows))
                for position in range(width):
                    self.assertEqual(Counter(row[position] for row in rows),Counter({i:12//width for i in range(width)}))

    def test_unseen_grid_and_one_acquisition_per_fresh_context(self):
        programs={s:{'target_action_index':42} for s in ('profile','heldout')}
        jobs=campaign.plans(programs,'c'*64)
        self.assertEqual(len(jobs),12)
        for split in ('profile','heldout'):
            rows=[j for j in jobs if j['split']==split]
            self.assertEqual(len(rows),6)
            self.assertEqual(Counter(j['memory_mode'] for j in rows),Counter(registered=3,pageable=3))
            self.assertTrue(all(not j['request']['telemetry_between_bursts'] for j in rows))
        self.assertFalse(set(campaign.SIZES['profile']) & set(campaign.SIZES['heldout']))
        self.assertFalse(set(campaign.IDLES['profile']) & set(campaign.IDLES['heldout']))

    def test_existing_heldout_marker_refuses_fit_before_any_data_read(self):
        with tempfile.TemporaryDirectory(dir=campaign.ROOT/'build') as d:
            out=Path(d);(out/'heldout-started.json').write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'heldout already acquired'):
                fit.fit_and_freeze(out)
            self.assertFalse((out/'fit.json').exists())

    def test_no_implicit_acquisition_authorization(self):
        with tempfile.TemporaryDirectory(dir=campaign.ROOT/'build') as d:
            with self.assertRaises(FileNotFoundError):campaign.collect(d,'profile')
            self.assertFalse((Path(d)/'profile-started.json').exists())


class NamedResponseTests(unittest.TestCase):
    def test_nonzero_preparation_changes_zero_idle_first_copy(self):
        # All model computations, including the fitting curve, go through modeling.
        p=fit.curve('prepared_initial_recovery',[20.,50.,10.,20.,40.,10.,2.,100.],SCOPE,'registered',EVIDENCE,[1048576],[0])
        self.assertAlmostEqual(p[0,0,0],100000.) # 70 us base + 30 us prepared impact
        self.assertGreater(p[0,0,0],p[0,0,-1])

    def test_zero_initial_is_exact_nested_case_and_saturates_capacity(self):
        zero=[20.,50.,50.,30.,2.,100.]
        prepared=[20.,50.,0.,0.,50.,30.,2.,100.]
        a=fit.curve('zero_initial_recovery',zero,SCOPE,'pageable',EVIDENCE,[131072,2097152],[0,1,5,20])
        b=fit.curve('prepared_initial_recovery',prepared,SCOPE,'pageable',EVIDENCE,[131072,2097152],[0,1,5,20])
        self.assertTrue((a==b).all())
        self.assertAlmostEqual(a[0,0,0],26250.)
        self.assertLessEqual(a[-1,-1,0],20e3+50e3*2+50e3+30e3*2)

    def test_fitting_curve_and_integer_forecast_agree_to_one_ns(self):
        theta=[20.,50.,10.,20.,40.,10.,2.,100.]
        smooth=fit.curve('prepared_initial_recovery',theta,SCOPE,'registered',EVIDENCE,[262144,1048576],[2,10])
        integer=fit.curve('prepared_initial_recovery',theta,SCOPE,'registered',EVIDENCE,[262144,1048576],[2,10],continuous=False)
        self.assertLess(float(abs(smooth-integer).max()),2.)

    def test_fitter_cannot_silently_call_a_mismatched_runtime(self):
        from llm_infer_model.performance.parameters import UnknownParameter
        model=fit.model_for('zero_initial_recovery',[20.,50.,50.,30.,2.,100.],SCOPE,'registered',EVIDENCE)
        with self.assertRaises(UnknownParameter):
            model.continuous_burst(262144,2000000,16,context_sha256='d'*64,preparation_sha256='b'*64)


class SupportTests(unittest.TestCase):
    def records(self,drift=False):
        records=[]
        for memory in ('registered','pageable'):
            for process in range(3):
                bursts=[]
                for size in campaign.SIZES['profile']:
                    for trial in range(12):
                        value=100.+size/1048576
                        if drift and memory=='pageable' and process==1 and size==131072 and trial>=6:value*=1.5
                        bursts.append(dict(bytes=size,requested_idle_ms=0,trial=trial,copies=[dict(wall_ns=value) for _ in range(16)]))
                records.append(dict(job=dict(memory_mode=memory,id=f'{memory}-{process}'),value=dict(bursts=bursts)))
        return records

    def test_complete_baseline_support_reports_every_declared_cell(self):
        gate=fit.baseline_support(self.records())
        self.assertTrue(gate['passed']);self.assertEqual(len(gate['early_late']),18);self.assertEqual(len(gate['fresh_processes']),6)

    def test_one_late_shift_refuses_pooling_without_removing_it(self):
        gate=fit.baseline_support(self.records(drift=True))
        self.assertFalse(gate['passed']);self.assertEqual(gate['raw_samples_removed'],0)
        failed=[r for r in gate['early_late'] if not r['passed']]
        self.assertEqual(len(failed),1);self.assertAlmostEqual(failed[0]['relative_difference'],.5)
        self.assertEqual(len(gate['fresh_processes']),6)


if __name__=='__main__':unittest.main()
