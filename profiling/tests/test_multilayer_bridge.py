"""Multilayer numerical/dataflow contracts, with no native timing campaign."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from llm_infer_model.tensor.spec import InvalidInput

from profiling.highlevel_accuracy.bridge import BridgeError, export_plan, reference_ports, sha, validate_program
from profiling.highlevel_accuracy.multilayer import build_multilayer_case


def reseal(p):
    p["identity"] = {key: sha(p[key]) for key in p["identity"]}
    p.pop("program_sha256", None)
    p["program_sha256"] = sha(p)
    return p


class MultilayerBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "highlevel_accuracy")
        root = Path(cls.temp.name)
        cls.programs = {}
        for count in (2, 4):
            plan = build_multilayer_case(root / str(count), layers=count, tokens=3, history=0,
                hidden=8, intermediate=16, heads=2, kv_heads=1, head_dim=4,
                devices=("cpu",)*3, threads=1, attention_gate="none", qk_norm="none")
            cls.programs[count] = export_plan(plan.compile(), seed=817)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_real_layer_connections_and_distinct_weights(self):
        for count, p in self.programs.items():
            self.assertEqual(len(p["bindings"]), 3*count)
            self.assertEqual(len(p["invocation_scopes"]), count)
            for left, right in zip(p["invocation_scopes"], p["invocation_scopes"][1:]):
                self.assertEqual(set(left["hidden_outputs"]), set(right["incoming_hidden"]))
                self.assertFalse(any(tid in p["initial_values"] for tid in right["incoming_hidden"]))
            catalogues = [set(v.values()) for v in p["layer_weight_bindings"].values()]
            self.assertEqual(sum(map(len, catalogues)), len(set.union(*catalogues)))

    def test_final_output_matches_independent_sequential_block_math(self):
        # Deliberately independent from reference_ports' helpers and bindings.
        for count, p in self.programs.items():
            c = p["physical"]["config"]
            h, d, nh, nk, n = c["hidden"], c["head_dim"], c["heads"], c["kv_heads"], 3
            first = p["invocation_scopes"][0]["incoming_hidden"][0]
            x = np.asarray(p["initial_values"][first], dtype=np.float64).reshape(n, h)
            def materialize(v): return np.asarray(v, dtype=np.float32).astype(np.float64)
            def rms(v, gamma): return v / np.sqrt(np.mean(v*v, axis=-1, keepdims=True) + c["norm_epsilon"]) * gamma
            for scope in p["invocation_scopes"]:
                weights = p["layer_weight_bindings"][scope["id"]]
                def w(role, rows=1):
                    a = np.asarray(p["initial_values"][weights[role]], dtype=np.float64)
                    return a if rows == 1 else a.reshape(rows, -1)
                z = rms(x, w("input_norm"))
                q, k, v = [z @ w(role, (nh if role == "q" else nk)*d).T for role in ("q", "k", "v")]
                q, k, v = q.reshape(n,nh,d), k.reshape(n,nk,d), v.reshape(n,nk,d)
                for pos in range(n):
                    for pair in range(d//2):
                        angle = pos / c["rope_base"] ** (2*pair/d)
                        for value in (q,k):
                            u, vv = value[pos,:,2*pair].copy(), value[pos,:,2*pair+1].copy()
                            value[pos,:,2*pair] = u*np.cos(angle)-vv*np.sin(angle)
                            value[pos,:,2*pair+1] = u*np.sin(angle)+vv*np.cos(angle)
                q,k,v = map(materialize,(q,k,v))
                context = np.empty((n,nh,d))
                for head in range(nh):
                    scores = q[:,head] @ k[:,head//(nh//nk)].T / np.sqrt(d)
                    scores[np.triu_indices(n,1)] = -np.inf
                    a = np.exp(scores - np.max(scores,axis=-1,keepdims=True))
                    a /= a.sum(axis=-1,keepdims=True)
                    context[:,head] = a @ v[:,head//(nh//nk)]
                x = materialize(context.reshape(n,nh*d) @ w("output",h).T + x)
                z = rms(x,w("post_attention_norm"))
                gate,up = z@w("gate",c["intermediate"]).T,z@w("up",c["intermediate"]).T
                x = materialize((gate/(1+np.exp(-gate))*up) @ w("down",h).T + x)
            actual = reference_ports(p)[p["invocation_scopes"][-1]["hidden_outputs"][0]]
            np.testing.assert_allclose(np.asarray(actual).reshape(n,h),x,atol=2e-6,rtol=2e-6)

    def test_independently_initialized_later_hidden_rejected(self):
        p = deepcopy(self.programs[2])
        tid = p["invocation_scopes"][1]["incoming_hidden"][0]
        p["initial_values"][tid] = [0.] * 24
        with self.assertRaisesRegex(BridgeError,"initial tensor inventory"):
            validate_program(reseal(p))

    def test_scope_hidden_seam_change_rejected(self):
        p=deepcopy(self.programs[2]);p["invocation_scopes"][1]["incoming_hidden"]=[p["invocation_scopes"][0]["incoming_hidden"][0]]
        with self.assertRaisesRegex(BridgeError,"scope hidden inputs"):
            validate_program(reseal(p))

    def test_cross_layer_weight_catalogue_rejected(self):
        p=deepcopy(self.programs[2]);scopes=p["invocation_scopes"]
        p["layer_weight_bindings"][scopes[1]["id"]]["q"] = p["layer_weight_bindings"][scopes[0]["id"]]["q"]
        with self.assertRaisesRegex(BridgeError,"weight ownership"):
            validate_program(reseal(p))

    def test_source_leaf_change_rejected_before_export(self):
        root=Path(self.temp.name)/"tampered"
        plan=build_multilayer_case(root,layers=2,tokens=3,history=0,hidden=8,intermediate=16,
                                  heads=2,kv_heads=1,head_dim=4,devices=("cpu",)*3,threads=1)
        compiled=plan.compile()
        path=root/"layer1/layer.json"
        path.write_text(path.read_text()+" ",encoding="utf-8")
        with self.assertRaisesRegex(InvalidInput,"source changed"):
            export_plan(compiled)

    def test_multiround_token_gate_remains_rejected(self):
        from llm_simulator import AnalyticalCosts, Device, Hardware, Model, Scheduling, Simulator, Workload
        simulator = Simulator(model=Model.dense(layers=2,hidden_size=8,intermediate_size=16,heads=2,kv_heads=1),
            hardware=Hardware(cpu=Device(1e9,1e9),gpu=Device(1e9,1e9),dram_capacity_bytes=1<<30,
                              vram_capacity_bytes=1<<30,transfer_bandwidth_bytes_per_second=1e9),
            workload=Workload(prefill_tokens=3,decode_steps=1),cost_model=AnalyticalCosts(workspace_bytes=0))
        with simulator.compile(scheduling=Scheduling(default_device="cpu",release="last_use"),
                               directory=Path(self.temp.name)/"multiround") as compiled:
            with self.assertRaisesRegex(BridgeError,"multi-round trace token gates"):
                export_plan(compiled.plan)

    def test_distinct_cuda_devices_are_rejected_without_native_launch(self):
        from llm_infer_model.m2.dense_cuda_contract import signature
        from llm_infer_model.m2.costs import canonical_contract_hash
        root=Path(self.temp.name)/"cuda-device-mismatch"
        plan=build_multilayer_case(root,layers=2,tokens=3,history=0,hidden=8,intermediate=16,
                                  heads=2,kv_heads=1,head_dim=4,devices=("cpu",)*3,threads=1)
        layer_path,cost_path=root/"layer1/layer.json",root/"layer1/costs.json"
        layer=json.loads(layer_path.read_text(encoding="utf-8"));costs=json.loads(cost_path.read_text(encoding="utf-8"))
        layer["physical"]["cuda_device"]=1
        for row in costs["entries"]:
            if row["device"]=="gpu":
                contract=signature(layer["physical"]["config"],row["kind"],row["phase"],row["tokens"],row["visible_history"],1)
                row["physical_signature"]=contract;row["contract_hash"]=canonical_contract_hash(contract)
        for path,value in ((layer_path,layer),(cost_path,costs)):
            with path.open("w",encoding="utf-8",newline="\n") as stream:json.dump(value,stream);stream.write("\n")
        # Valid per-layer costs/configurations compile, but cannot share one
        # native executor's GPU. The unit test never initializes any GPU.
        compiled=plan.compile()
        with self.assertRaisesRegex(BridgeError,"CUDA device"):
            export_plan(compiled)


if __name__ == "__main__":
    unittest.main()
