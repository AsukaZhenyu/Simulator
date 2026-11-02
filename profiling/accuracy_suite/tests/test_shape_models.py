from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
module=load("new_shape_models_test",HERE.parents[1]/"shape_models.py")
fixtures=load("shape_model_fixtures",HERE.parents[2]/"composition_calibration/tests/test_model.py")


def records():
    rows=fixtures.known_fit_rows()
    another=deepcopy(rows[0]);another['id']='profile-other';another['case']['id']='other'
    another['case']['hidden']=16
    another['raw_sha256']='c'*64
    return rows+[another]


class ShapeModelsTests(unittest.TestCase):
    def test_models_keep_signed_sources_and_freeze_every_geometry_fold(self):
        rows=records();before=deepcopy(rows)
        bundle=module.freeze(rows)
        self.assertEqual(rows,before)
        self.assertEqual(bundle['full']['batch_repetitions'],1)
        self.assertEqual(len(bundle['folds']),3)
        for fold in bundle['folds']:
            excluded={r['id'] for r in rows if r['case']['id']==fold['excluded_case_id']}
            self.assertFalse(excluded & set(fold['training_row_ids']))
        self.assertAlmostEqual(module.predict(bundle,'ffn','hybrid/resident')['predicted_ns'],17000)

    def test_held_geometry_targets_cannot_change_its_fold_prediction(self):
        rows=records();first=module.freeze(rows)
        for row in rows:
            if row['case']['id']=='ffn':
                for s in fixtures.model.primary_strategies(row['case']):row['metrics'][s]=[9999999]*3
        second=module.freeze(rows)
        self.assertNotEqual(first['full']['coefficients'],second['full']['coefficients'])
        a=module.predict(first,'ffn','hybrid/preload',leave_case_out=True)
        b=module.predict(second,'ffn','hybrid/preload',leave_case_out=True)
        self.assertEqual(a,b)

    def test_nonprofile_rejected_and_failed_sources_are_recorded_without_fallback(self):
        rows=records();rows[0]['split']='heldout'
        with self.assertRaises(ValueError):module.freeze(rows)
        rows=records();rows[0]['status']='failed';bundle=module.freeze(rows)
        self.assertEqual(bundle['failed_profile_rows'],['profile-ffn'])
        with self.assertRaises(ValueError):module.predict(bundle,'ffn','hybrid/resident')


if __name__=='__main__':unittest.main()
