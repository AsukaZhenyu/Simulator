from __future__ import annotations
from collections import Counter
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
from . import h2d_impulse_v11 as c

class ImpulseProtocolTests(unittest.TestCase):
    def test_finite_same_coordinate_holdout_is_declared_honestly(self):
        p=c.protocol()
        self.assertTrue(p['same_coordinates_across_profile_and_heldout']);self.assertTrue(p['no_unseen_coordinate_claim'])
        self.assertFalse(p['idle_interpolation']);self.assertFalse(p['raw_CV_is_prediction_gate']);self.assertEqual(p['first_samples_per_coordinate_per_split'],216)
        self.assertIn('no additional COPY primer',p['preparation'])

    def test_every_window_order_is_complete_and_n_position_balanced(self):
        jobs=c.plans({s:dict(target_action_index=2) for s in ('profile','heldout')},'c'*64)
        self.assertEqual(len(jobs),12)
        for split in ('profile','heldout'):
            self.assertEqual(Counter(j['memory_mode'] for j in jobs if j['split']==split),Counter(registered=3,pageable=3))
        for job in jobs:
            r=job['request'];want=set(range(len(r['conditions'])))
            self.assertEqual(len(r['condition_orders']),24)
            self.assertTrue(all(set(row)==want for row in r['condition_orders']))
            for coordinate in range(len(c.SIZES)*len(c.IDLES)):
                positions=[]
                for row in r['condition_orders']:
                    actual=[r['conditions'][i]['window_index'] for i in row if (r['conditions'][i]['size_index'],r['conditions'][i]['idle_index'])==divmod(coordinate,len(c.IDLES))]
                    positions.append(actual)
                for pos in range(3):self.assertEqual(Counter(row[pos] for row in positions),Counter({0:8,1:8,2:8}))

    def test_heldout_marker_blocks_any_parameter_read_or_refit(self):
        with tempfile.TemporaryDirectory(dir=c.ROOT/'build') as tmp:
            p=Path(tmp);(p/'heldout-started.json').write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'single target-blind'):c.fit_and_freeze(p)
            self.assertFalse((p/'fit.json').exists())

    def test_parameter_estimation_separates_first_from_sustained_and_future_n(self):
        def records(_out,_split,memory):
            rs=[]
            for process in range(3):
                bursts=[]
                for size,idle,n in __import__('itertools').product(c.SIZES,c.IDLES,c.WINDOWS):
                    base=20000.+size/16.;first=base+30000.+idle*1000.
                    for _ in range(24):bursts.append(dict(bytes=size,requested_idle_ms=idle,copies_count=n,copies=[dict(wall_ns=first)]+[dict(wall_ns=base)]*(n-1)))
                rs.append(dict(native=dict(path='profile-'+memory+str(process),sha256='c'*64),bursts=bursts))
            return dict(scope=dict(context_sha256='a'*64,preparation_sha256='b'*64)),rs
        with tempfile.TemporaryDirectory(dir=c.ROOT/'build') as tmp:
            out=Path(tmp);(out/'manifest.json').write_text('{}',encoding='utf-8')
            with patch.object(c,'observations',side_effect=records):c.fit_and_freeze(out)
            pred=c.read(out/'heldout-predictions.json')['rows']
            a=next(r for r in pred if (r['memory_mode'],r['bytes'],r['idle_ms'],r['copies'])==('registered',1048576,2,16))
            b=next(r for r in pred if (r['memory_mode'],r['bytes'],r['idle_ms'],r['copies'])==('registered',1048576,2,64))
            self.assertEqual(a['first_prediction_ns'],b['first_prediction_ns'])
            self.assertEqual(b['window_prediction_ns']-a['window_prediction_ns'],48*(20000+1048576/16))
            self.assertEqual(a['first_prediction_ns'],20000+1048576/16+32000)

if __name__=='__main__':unittest.main()
