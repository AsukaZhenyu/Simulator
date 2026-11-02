"""Owned-event ordering and fresh protocol tests; no native workload is run."""
from argparse import Namespace
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from profiling import event_stage_campaign as collector


class Clock:
    def now(self):return 123


class EventCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='event-protocol-',dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.events=[];self.children=[];self.log=[]
        owner=self

        class FakeEvents:
            def __init__(self):
                self.number=len(owner.events)
                self.names={r:f'Local\\fake-{self.number}-{r}' for r in ('go','stop')}
                self.handles={r:self.number*10+i+1 for i,r in enumerate(('go','stop'))}
                self.child=None;owner.events.append(self)
            def signal(self,role):
                path=self.child.directory/(role+'.json')
                owner.assertTrue(path.exists(),'signal before committed receipt')
                json.loads(path.read_text(encoding='utf-8'))
                owner.log.append(('signal',self.child.name,role))
            def close(self):
                owner.assertTrue(self.child is None or self.child.process.poll() is not None,'close while child alive')
                owner.log.append(('close',None if self.child is None else self.child.name))
                self.handles.clear()

        class FakeChild:
            def __init__(self,name,directory,command,env,clock,**kwargs):
                self.name,self.directory,self.clock,self.env=name,directory,clock,dict(env)
                self.exited=False;self.process=self;self.stop_path=directory/'stop.json'
                directory.mkdir(parents=True)
                ready={}
                if 'SIMULATOR_PHYSICAL_GO_EVENT' in env:
                    pair=next(e for e in owner.events if e.names['go']==env['SIMULATOR_PHYSICAL_GO_EVENT'])
                    pair.child=self
                    ready['event_parking']=dict(go_event=pair.names['go'],stop_event=pair.names['stop'],post_sampling_periodic_wakeup=False)
                    if kwargs.get('bad_ready'):ready['event_parking']['go_event']='wrong'
                self.record=dict(name=name,ready=ready)
                owner.children.append(self)
            def poll(self):return 0 if self.exited else None
            def stop(self,*,cleanup=False):
                owner.log.append(('base_stop',self.name,cleanup));self.exited=True
                self.record.update(returncode=0,process_exited_verified=True,forced_termination=False)
                return self.record

        self.FakeEvents,self.FakeChild=FakeEvents,FakeChild

    def test_distinct_actor_pairs_receipts_before_signals_and_exit_before_close(self):
        oldsave=collector.campaign.run.save
        with patch.object(collector,'OwnedEvents',self.FakeEvents),patch.object(collector.campaign,'OwnedChild',self.FakeChild):
            with collector.event_actors(dict(parking='event',cycles=True)):
                main=collector.campaign.OwnedChild('main',self.root/'main',[],{},Clock())
                before=collector.campaign.OwnedChild('before-cpu',self.root/'before-cpu',[],{},Clock())
                worker=collector.campaign.OwnedChild('cpu-worker',self.root/'worker',[],{},Clock())
                self.assertEqual(main.env['SIMULATOR_PHYSICAL_CYCLES'],'1')
                self.assertEqual(before.env['SIMULATOR_PHYSICAL_CYCLES'],'1')
                self.assertNotIn('SIMULATOR_PHYSICAL_GO_EVENT',worker.env)
                self.assertEqual(len({n for e in self.events for n in e.names.values()}),4)
                collector.campaign.run.save(main.directory/'go.json',dict(schema='fake-go'))
                collector.campaign.run.save(before.directory/'go.json',dict(schema='fake-go'))
                main.stop();before.stop();worker.stop()
            self.assertIs(collector.campaign.OwnedChild,self.FakeChild)
            self.assertIs(collector.campaign.run.save,oldsave)
        for name in ('main','before-cpu'):
            stop=self.log.index(('signal',name,'stop'))
            exited=self.log.index(('base_stop',name,False))
            closed=self.log.index(('close',name))
            self.assertLess(stop,exited);self.assertLess(exited,closed)

    def test_ready_validation_failure_stops_child_before_handle_close(self):
        oldsave=collector.campaign.run.save
        with patch.object(collector,'OwnedEvents',self.FakeEvents),patch.object(collector.campaign,'OwnedChild',self.FakeChild):
            with self.assertRaisesRegex(ValueError,'READY event identities'):
                with collector.event_actors(dict(parking='event',cycles=False)):
                    collector.campaign.OwnedChild('main',self.root/'main',[],{},Clock(),bad_ready=True)
            self.assertIs(collector.campaign.OwnedChild,self.FakeChild)
            self.assertIs(collector.campaign.run.save,oldsave)
        self.assertTrue(self.children[0].exited)
        self.assertLess(self.log.index(('base_stop','main',True)),self.log.index(('close','main')))

    def test_poll_control_creates_no_events_and_restores_hooks_on_failure(self):
        oldsave=collector.campaign.run.save
        with patch.object(collector,'OwnedEvents',self.FakeEvents),patch.object(collector.campaign,'OwnedChild',self.FakeChild):
            with self.assertRaisesRegex(RuntimeError,'synthetic'):
                with collector.event_actors(dict(parking='poll',cycles=False)):
                    child=collector.campaign.OwnedChild('main',self.root/'poll',[],{},Clock())
                    child.stop();raise RuntimeError('synthetic')
            self.assertIs(collector.campaign.OwnedChild,self.FakeChild)
            self.assertIs(collector.campaign.run.save,oldsave)
        self.assertEqual(self.events,[])

    def fixture(self):
        event=self.root/'event.exe';event.write_bytes(b'fake event')
        poll=self.root/'poll.exe';poll.write_bytes(b'fake poll')
        args=Namespace(out=self.root/'campaign',binary=event,poll_binary=poll)
        parent=dict(binary=str(event),protocol=dict(gpu_reference_mask=48),source_inventory_sha256='fake',programs={})
        def prepare(_args):
            args.out.mkdir();collector.campaign.run.save(args.out/'manifest.json',parent)
        with patch.object(collector.campaign,'prepare',side_effect=prepare):collector.prepare(args)
        return args,parent,collector.campaign.run.read(args.out/'event-protocol.json')

    def test_37_jobs_predeclare_matched_observers_disjoint_phases_and_cycles(self):
        _,_,p=self.fixture();jobs=p['jobs']
        self.assertEqual(len(jobs),37)
        self.assertEqual(len({j['id'] for j in jobs}),37)
        self.assertEqual({k:sum(j['split']==k for j in jobs) for k in ('smoke','observer','diagnostic','profile','heldout')},
                         dict(smoke=1,observer=9,diagnostic=3,profile=12,heldout=12))
        for split in ('profile','heldout'):
            group=[j for j in jobs if j['split']==split]
            self.assertTrue(all(j['parking']=='event' and j['cycles']==(split=='profile') and j['samples']==96 for j in group))
            self.assertEqual({j['condition'] for j in group},{'idle','cpu','gpu','both'})
        for repeat in range(3):
            group=[j for j in jobs if j['id'].startswith(f'control-{repeat}-')]
            self.assertEqual({(j['parking'],j['cycles']) for j in group},{('poll',False),('event',False),('event',True)})
        self.assertFalse(p['CPU_cycles_used_as_heldout_input'])

    def test_collect_selects_native_by_parking_and_keeps_parent_unmodified(self):
        args,parent,p=self.fixture();args.split='observer';args.barrier=None
        selected=[]
        def execute(root,manifest,job):
            selected.append((job['parking'],manifest['binary'],manifest['protocol']['gpu_reference_mask']))
            return dict(status='completed',cleanup_verified=True)
        with patch.object(collector,'verify',return_value=(parent,p)),patch.object(collector.campaign.run,'measurement_lock',return_value=nullcontext()), \
             patch.object(collector.campaign,'execute',side_effect=execute):
            outcome=collector.collect(args)
        self.assertEqual(outcome['jobs'],9)
        self.assertTrue(all(binary==(p['event_binary'] if parking=='event' else p['poll_binary']) for parking,binary,_ in selected))
        self.assertTrue(all(mask==4 for _,_,mask in selected))
        self.assertEqual(parent['protocol']['gpu_reference_mask'],48)


if __name__=='__main__':unittest.main()
