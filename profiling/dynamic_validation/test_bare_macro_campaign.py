from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import bare_macro_campaign as campaign


class MacroCampaignTests(unittest.TestCase):
    def fixture(self,parent):
        jobs=[]
        for state in campaign.STATES:
            for round_id in range(3):
                for mode in ('stamped','runtime_callbacks'):
                    job=dict(id=f'profile-{state[0]}{state[1]}-r{round_id}-{mode}',split='profile',stage='oldstage',offered_state=list(state),round=round_id,
                        request=dict(split='profile',instrumentation=mode,offered_state=list(state),round_test_tag=round_id))
                    jobs.append(job)
                    if state==(0,0) and mode=='stamped':
                        target=parent/'jobs'/job['id']/'execution.json';target.parent.mkdir(parents=True);target.write_text('{}\n',encoding='utf-8',newline='\n')
        for plan in ('a','b'):
            for round_id in range(3):jobs.append(dict(id=f'held-{plan}-{round_id}',split='heldout',stage='heldout',request=dict(split='heldout',instrumentation='stamped',plan=plan)))
        return dict(jobs=jobs)

    def test_all_three_off_and_nine_on_preserved_without_callback_selection(self):
        with tempfile.TemporaryDirectory(dir=campaign.ROOT/'build') as d:
            parent=Path(d);manifest=self.fixture(parent);before=copy.deepcopy(manifest);jobs=campaign.plans(manifest,parent)
            self.assertEqual(manifest,before)
            self.assertEqual(len(jobs),18);self.assertEqual(sum(j['imported'] for j in jobs),3)
            self.assertEqual({j['round'] for j in jobs if j['imported']},{0,1,2})
            source={j['id']:j for j in before['jobs']}
            self.assertTrue(all(j['request']==source[j['id']]['request'] for j in jobs))
            self.assertTrue(all(j['request']['instrumentation']=='stamped' for j in jobs))
            for state in campaign.STATES:self.assertEqual(sum(j['split']=='profile' and tuple(j['offered_state'])==state for j in jobs),3)

    def test_missing_an_off_case_refuses_reuse(self):
        with tempfile.TemporaryDirectory(dir=campaign.ROOT/'build') as d:
            parent=Path(d);manifest=self.fixture(parent);manifest['jobs']=[j for j in manifest['jobs'] if j['id']!='profile-00-r1-stamped']
            with self.assertRaisesRegex(ValueError,'all three OFF'):campaign.plans(manifest,parent)

    def test_macro_collection_does_not_make_fine_provider_qualified(self):
        p=campaign.protocol()
        self.assertFalse(p['fine_API_provider_qualified'])
        self.assertFalse(p['callback_timing_labels_used'])
        self.assertFalse(p['calibration_uses_whole_LLM_target'])
        self.assertFalse(p['actual_heldout_boundaries_used_to_reanchor_prediction'])
        self.assertEqual(p['acceptance']['process_mean_APE_max'],.10)
        self.assertEqual(p['acceptance']['within_process_block_APE_P90_max'],.20)

    def test_heldout_requires_supported_component_forecast_freeze(self):
        auth=dict(collection_authorized=True,split='heldout',manifest_sha256='manifest',frozen_barrier_sha256='barrier')
        frozen=dict(manifest_sha256='manifest',profile_supported=False,parameters_frozen=True,predictions_frozen=True,
                    calibration_target_is_component=True,whole_LLM_target_time_used_for_fit=False,files={})
        def read(path):return auth if path.name=='heldout-authorization.json' else frozen
        with patch.object(campaign,'check',return_value={}),patch.object(campaign,'read',side_effect=read),patch.object(campaign,'sha',side_effect=lambda p:'manifest' if p.name=='manifest.json' else 'barrier'):
            with self.assertRaisesRegex(ValueError,'complete component cells'):campaign.authorization(Path('unused'),'heldout')


if __name__=='__main__':unittest.main()
