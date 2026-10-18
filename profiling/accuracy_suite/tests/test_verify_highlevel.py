"""Synthetic, offline corruption tests. No native program or hardware is run."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
for directory in (ROOT/"modeling",ROOT/"mapping",ROOT/"profiling"):
    if str(directory) not in sys.path:sys.path.insert(0,str(directory))
from accuracy_suite import verify_highlevel as verifier
from highlevel_accuracy.bridge import BOUNDARY,RUNTIME_FLAGS,export_plan,reference_ports
from highlevel_accuracy.profiles import GATES,compare_heldout,freeze_profile,run_plan_with_profile
from highlevel_accuracy.run import load_case,prepare,write_new


def native(program,mode,prediction=None):
    tensors={t["id"]:t for t in program["graph"]["workload"]["tensors"]}
    return dict(format_version="highlevel-accuracy-native-v1",program_sha256=program["program_sha256"],boundary_contract=BOUNDARY,
        actions=program["actions"],ports=reference_ports(program),copy_values_equal=True,
        observed_runtime_flags=RUNTIME_FLAGS,final_execution_ports_captured_outside_timing=True,
        copies=[dict(index=i,tensor_id=a["target_id"],src=a["src"],dst=a["dst"],storage_bytes=tensors[a["target_id"]]["storage_bytes"])
                for i,a in enumerate(program["actions"]) if a["kind"]=="COPY"],
        mode=mode,samples=3,warmup=0,timing_scope="synthetic_unit_fixture_not_hardware",
        workspace_scope="synthetic_unit_fixture_not_native_memory",
        components=[dict(action_index=i,action=a,samples_ns=[100,100,100],workspace_bytes=64)
                    for i,a in enumerate(program["actions"]) if a["kind"] in {"COPY","COMPUTE"}] if mode=="components" else [],
        total_samples_ns=[] if mode=="components" else [prediction]*3)


def reseal(document,key):
    document.pop(key,None);document[key]=verifier.digest(document);return document


class OfflineHighLevelVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        cls.root=Path(cls.temp.name);cls.experiment=cls.root/"experiment"
        prepare(cls.experiment,"smoke")
        campaign=verifier.read(cls.experiment/"campaign.json")
        base=datetime(2030,1,1,tzinfo=timezone.utc)
        campaign["created_at"]=base.isoformat();reseal(campaign,"campaign_sha256")
        (cls.experiment/"campaign.json").write_text(json.dumps(campaign))
        cls.executable=cls.root/"synthetic-never-executed.exe";cls.executable.write_bytes(b"test fixture only")
        # Full experiment tests substitute ONLY software byte verification.
        # Separate tests below cover current-byte/runtime snapshot rejection.
        cls.software=dict(binary_sha256=verifier.file_sha(cls.executable),successful_build_record=dict(linked_at=(base-timedelta(days=1)).isoformat()))
        counter=0;frozen_rows=[];phase_rows=[];saved=[]
        def raw(directory,program,mode,program_file,prediction=None):
            nonlocal counter
            directory.mkdir();n=native(program,mode,prediction)
            output=directory/"native.json";write_new(output,n)
            started=base+timedelta(seconds=counter+1);counter+=2
            record=dict(command=[str(cls.executable),str(program_file),mode,"0","3",str(output)],
                started_at=started.isoformat(),completed_at=(started+timedelta(seconds=1)).isoformat(),
                pid_scope="one_fresh_native_process",returncode=0,stdout=json.dumps(dict(status="ok",mode=mode,ports=len(n["ports"]),program_sha256=n["program_sha256"])),stderr="",timed_out=False,requested_runtime_flags=RUNTIME_FLAGS)
            write_new(directory/"execution.json",record)
            return n,verifier.raw_evidence(directory)[2],record
        for item in campaign["cases"]:
            case,plan,program=load_case(cls.experiment,item);merged=None;processes=[];evidence=[]
            for i in range(3):
                n,e,_=raw(case/f"profile-process-{i}",program,"components",case/"program.json")
                evidence.append(e);processes.append(dict(process_index=i,components=deepcopy(n["components"]),raw_evidence=e))
                if merged is None:merged=deepcopy(n)
                else:
                    for left,right in zip(merged["components"],n["components"]):left["samples_ns"].extend(right["samples_ns"])
                    merged["samples"]+=3
            merged["independent_processes"]=processes
            ledger=freeze_profile(program,merged,executable_sha256=cls.software["binary_sha256"],
                evidence=dict(processes=evidence,software=cls.software,independent_process_count=3),frozen_at=(base+timedelta(seconds=counter+1)).isoformat())
            write_new(case/"frozen_profile.json",ledger)
            prediction=run_plan_with_profile(plan,program,ledger);write_new(case/"modelplan_measured_prediction.json",prediction)
            frozen_rows.append(dict(case_id=item["id"],ledger_sha256=ledger["ledger_sha256"]))
            phase_rows.append(dict(case_id=item["id"],status="frozen",source_quality_passed=True,ledger_sha256=ledger["ledger_sha256"]))
            saved.append((item,case,plan,ledger))
        barrier=dict(format_version="highlevel-all-profiles-frozen-v1",campaign_sha256=campaign["campaign_sha256"],frozen_at=(base+timedelta(seconds=counter+2)).isoformat(),
            ledgers=frozen_rows,software=cls.software,profile_processes_per_case=3,samples_per_process=3,source_quality_passed=True,fixed_quality_gates=GATES)
        reseal(barrier,"barrier_sha256");write_new(cls.experiment/"all_profiles_frozen.json",barrier);counter+=4
        write_new(cls.experiment/"profile_phase_status.json",dict(format_version="highlevel-profile-phase-status-v1",campaign_sha256=campaign["campaign_sha256"],cases=phase_rows,
            failures=[],all_cases_frozen=True,source_quality_passed=True,fixed_quality_gates=GATES))
        results=[];groups=[]
        for item,case,plan,ledger in saved:
            local=[]
            for i in range(3):
                p=export_plan(plan,seed=100_000_000+item["profile_seed"]*10+i);source=case/f"heldout-program-{i}.json";write_new(source,p)
                prediction=run_plan_with_profile(plan,p,ledger)
                n,e,execution=raw(case/f"heldout-process-{i}",p,"replay",source,prediction["makespan_ns"])
                compared=compare_heldout(p,ledger,n,heldout_started_at=execution["started_at"],executable_sha256=cls.software["binary_sha256"])
                compared.update(case_id=item["id"],process_index=i,raw_evidence=e,actual_modelplan_prediction=prediction)
                write_new(case/f"comparison-{i}.json",compared);results.append(compared);local.append(compared)
            groups.append(verifier.group_result(item["id"],local,ledger,3,GATES))
        write_new(cls.experiment/"report.json",dict(format_version="highlevel-accuracy-campaign-report-v1",campaign_sha256=campaign["campaign_sha256"],barrier_sha256=barrier["barrier_sha256"],
            all_profiles_frozen_before_any_heldout=True,independent_heldout_processes_per_case=3,results=results,case_groups=groups,failures=[],fixed_quality_gates=GATES,
            all_cases_accepted=True,max_absolute_relative_error=0.0,within_explicit_tolerance=None))
        cls.first=cls.experiment/campaign["cases"][0]["id"]

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    @contextmanager
    def changed(self,path,mutator):
        original=path.read_bytes()
        try:
            value=json.loads(original);mutator(value);path.write_text(json.dumps(value));yield
        finally:path.write_bytes(original)

    def verify(self):
        with patch.object(verifier,"verify_software",return_value=self.executable):
            return verifier.verify_experiment(self.experiment)

    def test_complete_offline_chain_reconstruction(self):
        result=self.verify()
        self.assertEqual(result["status"],"verified");self.assertFalse(result["hardware_executed"])
        self.assertEqual((result["source_processes"],result["heldout_processes"]),(9,9))
        self.assertTrue(result["all_cases_accepted"])

    def test_tampered_raw_component_is_rejected(self):
        path=self.first/"profile-process-1/native.json"
        with self.changed(path,lambda n:n["components"][0]["samples_ns"].__setitem__(0,999)):
            with self.assertRaisesRegex(verifier.VerificationError,"frozen ledger"):self.verify()

    def test_tampered_operand_is_rejected(self):
        path=self.first/"heldout-program-0.json"
        with self.changed(path,lambda p:p["bindings"][0]["operands"].__setitem__("hidden","wrong-operand")):
            with self.assertRaisesRegex(verifier.VerificationError,"heldout actual compiled"):self.verify()

    def test_tampered_ledger_and_full_report_are_rejected(self):
        with self.changed(self.first/"frozen_profile.json",lambda l:l["components"][0].__setitem__("duration_ns",12345)):
            with self.assertRaisesRegex(ValueError,"frozen profile changed"):self.verify()
        with self.changed(self.experiment/"report.json",lambda r:r["case_groups"][0].__setitem__("accepted",False)):
            with self.assertRaisesRegex(verifier.VerificationError,"complete campaign report"):self.verify()

    def test_profile_heldout_chronology_and_seed_coverage_are_rejected(self):
        path=self.first/"heldout-process-0/execution.json"
        with self.changed(path,lambda e:e.update(started_at="2029-01-01T00:00:00+00:00",completed_at="2029-01-01T00:00:01+00:00")):
            with self.assertRaisesRegex(verifier.VerificationError,"before ALL profiles"):self.verify()
        with self.changed(self.first/"heldout-program-0.json",lambda p:p.__setitem__("seed",1)):
            with self.assertRaisesRegex(verifier.VerificationError,"seed/process"):self.verify()

    def test_missing_process_or_barrier_coverage_is_rejected(self):
        path=self.first/"comparison-2.json";original=path.read_bytes();path.unlink()
        try:
            with self.assertRaisesRegex(verifier.VerificationError,"missing declared heldout"):self.verify()
        finally:path.write_bytes(original)
        with self.changed(self.experiment/"all_profiles_frozen.json",lambda b:(b["ledgers"].pop(),reseal(b,"barrier_sha256"))):
            with self.assertRaisesRegex(verifier.VerificationError,"ordered all-profile"):self.verify()

    def test_current_runtime_or_source_hash_tampering_is_rejected(self):
        file=self.root/"runtime-snapshot.dll";file.write_bytes(b"original")
        record=dict(source_sha256={str(file):verifier.file_sha(file)})
        file.write_bytes(b"tampered")
        with self.assertRaisesRegex(verifier.VerificationError,"current bytes changed"):
            verifier.verify_software(record)

    def test_native_runtime_flag_tampering_is_rejected(self):
        path=self.first/"heldout-process-0/native.json"
        with self.changed(path,lambda n:n["observed_runtime_flags"].__setitem__("NVIDIA_TF32_OVERRIDE","1")):
            with self.assertRaisesRegex(verifier.VerificationError,"observed runtime flags"):self.verify()


if __name__=="__main__":unittest.main()
