"""Event-parked measurements with matched polling controls and fresh splits.

Only orchestration changes. The original Executor actions are not optimized.
GO/STOP files remain immutable receipts and each physical actor has its own
parent-owned manual-reset Windows event pair. All backgrounds stop after the
foreground completes, before unloaded after-reference construction.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import uuid

from profiling.available_service import campaign
from profiling.physical_posttarget_collection import _posttarget_hooks


class OwnedEvents:
    def __init__(self):
        if os.name != 'nt':raise RuntimeError('Windows event collector requires Windows')
        self.api=ctypes.WinDLL('kernel32',use_last_error=True)
        self.api.CreateEventW.argtypes=(ctypes.c_void_p,wintypes.BOOL,wintypes.BOOL,wintypes.LPCWSTR)
        self.api.CreateEventW.restype=wintypes.HANDLE
        self.api.SetEvent.argtypes=(wintypes.HANDLE,);self.api.SetEvent.restype=wintypes.BOOL
        self.api.CloseHandle.argtypes=(wintypes.HANDLE,);self.api.CloseHandle.restype=wintypes.BOOL
        self.names={role:'Local\\LLMSimulator-'+uuid.uuid4().hex+'-'+role for role in ('go','stop')}
        self.handles={}
        try:
            for role,name in self.names.items():
                ctypes.set_last_error(0)
                handle=self.api.CreateEventW(None,True,False,name)
                code=ctypes.get_last_error()
                if not handle or code==183:
                    if handle:self.api.CloseHandle(handle)
                    raise OSError(code,'event create failed or collided')
                self.handles[role]=handle
        except BaseException:
            self.close();raise

    def signal(self,role):
        if not self.api.SetEvent(self.handles[role]):raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        errors=[]
        for role,handle in list(self.handles.items()):
            if not self.api.CloseHandle(handle):errors.append(ctypes.get_last_error())
            del self.handles[role]
        if errors:raise OSError(errors,'owned event handle close failed')


@contextmanager
def event_actors(job):
    original_child,original_save=campaign.OwnedChild,campaign.run.save
    registry={};actors=[]
    class EventChild(original_child):
        def __init__(self,name,directory,command,env,clock,**kwargs):
            self.events=None
            if job['parking']=='event' and not name.endswith('-worker'):
                self.events=OwnedEvents();actors.append(self)
                env=dict(env,SIMULATOR_PHYSICAL_GO_EVENT=self.events.names['go'],
                         SIMULATOR_PHYSICAL_STOP_EVENT=self.events.names['stop'],
                         SIMULATOR_PHYSICAL_CYCLES='1' if name!='main' or job['cycles'] else '0')
                registry[str((directory/'go.json').resolve())]=self
            try:
                super().__init__(name,directory,command,env,clock,**kwargs)
                if self.events:
                    ready=self.record['ready']['event_parking']
                    campaign.require(ready['go_event']==self.events.names['go'] and
                                     ready['stop_event']==self.events.names['stop'] and
                                     ready['post_sampling_periodic_wakeup'] is False,'READY event identities differ')
                    self.record['event_controller']=dict(names=dict(self.events.names),manual_reset=True)
            except BaseException:
                if getattr(self,'process',None) is not None and self.process.poll() is None:
                    self.stop(cleanup=True)
                if self.events and self.events.handles:self.events.close()
                raise

        def stop(self,*,cleanup=False):
            if self.events and getattr(self,'process',None) is not None and 'returncode' not in self.record:
                if not self.stop_path.exists():
                    original_save(self.stop_path,dict(requested_tick=self.clock.now(),cleanup=cleanup,
                                                       signal_contract='file committed before parent STOP event'))
                self.events.signal('stop')
            try:
                return super().stop(cleanup=cleanup)
            finally:
                if self.events and self.events.handles and (getattr(self,'process',None) is None or self.process.poll() is not None):
                    self.events.close()

    def save(path,value):
        original_save(path,value)
        actor=registry.get(str(Path(path).resolve()))
        if actor is not None:actor.events.signal('go')

    campaign.OwnedChild,campaign.run.save=EventChild,save
    try:
        yield
    finally:
        campaign.OwnedChild,campaign.run.save=original_child,original_save
        for actor in actors:
            if actor.events.handles:
                campaign.require(getattr(actor,'process',None) is None or actor.process.poll() is not None,
                                 'cannot close event handles while actor is alive')
                actor.events.close()


def prepare(args):
    campaign.prepare(args)
    root=args.out.resolve();parent=campaign.run.read(root/'manifest.json')
    jobs=[dict(id='event-smoke',split='smoke',program_key='profile-0',condition='idle',cpu_background=0,
               gpu_background=0,instrumentation='bare',references=False,samples=2,parking='event',cycles=False)]
    for repeat in range(3):
        for parking,cycles in (('poll',False),('event',False),('event',True)):
            jobs.append(dict(id=f'control-{repeat}-{parking}-{int(cycles)}',split='observer',program_key=f'profile-{repeat}',
                condition='idle',cpu_background=0,gpu_background=0,instrumentation='bare',references=True,
                samples=32,parking=parking,cycles=cycles))
        state,cpu,gpu=[('cpu',1,0),('gpu',0,1),('both',1,1)][repeat]
        jobs.append(dict(id=f'cycles-{repeat}-{state}',split='diagnostic',program_key=f'profile-{repeat}',condition=state,
            cpu_background=cpu,gpu_background=gpu,instrumentation='cupti',references=True,samples=64,parking='event',cycles=True))
    states=[('idle',0,0),('cpu',1,0),('gpu',0,1),('both',1,1)]
    for split in ('profile','heldout'):
        for repeat in range(3):
            order=states[repeat:]+states[:repeat]
            if split=='heldout':order=list(reversed(order))
            for condition,cpu,gpu in order:
                jobs.append(dict(id=f'{split}-{repeat}-{condition}',split=split,program_key=f'{split}-{repeat}',
                    condition=condition,cpu_background=cpu,gpu_background=gpu,instrumentation='bare',references=True,
                    samples=96,parking='event',cycles=split=='profile'))
    poll=args.poll_binary.resolve()
    campaign.require(poll.is_file(),'explicit prior polling binary required for matched control')
    sources={str(Path(__file__).resolve()):campaign.run.sha(__file__),str(poll):campaign.run.sha(poll),
             str(Path(_posttarget_hooks.__code__.co_filename).resolve()):campaign.run.sha(_posttarget_hooks.__code__.co_filename)}
    protocol=dict(schema='event-stage-campaign-1',declared_at_utc=campaign.run.utc(),
        parent_manifest_sha256=campaign.run.sha(root/'manifest.json'),event_binary=parent['binary'],poll_binary=str(poll),
        jobs=jobs,source_receipts=sources,protocol_overrides=dict(gpu_reference_mask=4),
        before_reference_cpu_matches_main_affinity=True,posttarget_after_reference_scope='unloaded_drift_only',
        event_reference_cycles_enabled=True,profile_target_cycles_enabled=True,heldout_target_cycles_enabled=False,
        Executor_actions_altered=False,CPU_cycles_used_as_heldout_input=False,physical_CPU_time_inferred_from_cycles=False,
        parameter_freeze_required=True,all_slow_calls_retained=True,observer_error_max=.05,
        point_error_max=.20,point_p90_error_max=.20,uncertainty_must_not_replace_point_gates=True)
    campaign.run.save(root/'event-protocol.json',protocol)
    return dict(status='prepared',jobs=len(jobs),native_timed_work_started=False)


def verify(root):
    parent=campaign.run.read(root/'manifest.json');campaign.verify(parent)
    protocol=campaign.run.read(root/'event-protocol.json')
    campaign.require(protocol['parent_manifest_sha256']==campaign.run.sha(root/'manifest.json'),'parent changed')
    for path,expected in protocol['source_receipts'].items():
        campaign.require(campaign.run.sha(path)==expected,'event protocol source changed: '+path)
    return parent,protocol


def collect(args):
    root=args.out.resolve();parent,protocol=verify(root)
    barrier=None
    if args.split=='heldout':
        campaign.require(args.barrier is not None,'frozen barrier is required')
        barrier=campaign.run.read(args.barrier)
        campaign.require(barrier['event_protocol_sha256']==campaign.run.sha(root/'event-protocol.json'),'barrier protocol differs')
        campaign.require(campaign.run.sha(barrier['frozen_file'])==barrier['frozen_file_sha256'],'frozen parameters changed')
    campaign.run.save(root/(args.split+'-started.json'),dict(started_at_utc=campaign.run.utc(),split=args.split,
        event_protocol_sha256=campaign.run.sha(root/'event-protocol.json'),barrier=barrier))
    rows=[]
    with campaign.run.measurement_lock():
        for job in protocol['jobs']:
            if job['split']!=args.split:continue
            verify(root)
            if barrier:
                campaign.require(campaign.run.sha(barrier['frozen_file'])==barrier['frozen_file_sha256'],'frozen parameters changed during heldout')
            manifest=deepcopy(parent)
            manifest['binary']=protocol['event_binary'] if job['parking']=='event' else protocol['poll_binary']
            manifest['protocol'].update(protocol['protocol_overrides'])
            with event_actors(job):
                with _posttarget_hooks(root,job,root/'event-protocol.json'):
                    row=campaign.execute(root,manifest,job)
            rows.append(row)
            print(json.dumps(dict(job=job['id'],status=row['status'],cleanup_verified=row['cleanup_verified'])),flush=True)
    campaign.run.save(root/(args.split+'-executions.json'),rows)
    return dict(status='collected',split=args.split,jobs=len(rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--poll-binary',type=Path,required=True);p.add_argument('--background-binary',type=Path,required=True)
    p.add_argument('--runtime-dir',type=Path,action='append',required=True);p.add_argument('--seed',type=int,required=True)
    p=sub.add_parser('collect');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--split',choices=('smoke','observer','diagnostic','profile','heldout'),required=True)
    p.add_argument('--barrier',type=Path)
    args=parser.parse_args();print(json.dumps(prepare(args) if args.command=='prepare' else collect(args)))


if __name__=='__main__':main()
