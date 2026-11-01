"""Tiny native LLMCompass graph/component checks; pinned source is cwd.

No GPU execution, kernel search, or matched end-to-end timing comparison.
The returned latency is the external package's own roofline value, not a
measurement and not used as an accuracy ground truth for this paper.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from software_model.matmul import Matmul
from software_model.utils import Tensor, data_type_dict
from hardware_model.device import device_dict


device = device_dict["A100_80GB_fp16"]
dtype = data_type_dict["fp16"]
results = []
for h in (8,16):
    for q in (1,3):
        for role,k,n in (("projection",h,h), ("ffn_up",h,2*h), ("ffn_down",2*h,h)):
            op = Matmul(dtype)
            out = op(Tensor([q,k], dtype), Tensor([k,n], dtype))
            assert out.shape == [q,n]
            assert op.flop_count == 2*q*k*n
            assert op.io_count == q*k+k*n+q*n
            latency = op.roofline_model(device)
            assert math.isfinite(latency) and latency > 0
            results.append(dict(role=role, input_shapes=[[q,k],[k,n]], output_shape=out.shape,
                                dtype="fp16", flop_count=op.flop_count, io_count=op.io_count,
                                native_roofline_seconds=latency))
try:
    Matmul(dtype)(Tensor([1,8],dtype), Tensor([9,8],dtype))
except AssertionError:
    incompatible_k_rejected = True
else:
    raise AssertionError("Native matmul K validation did not reject")
print("E1_NATIVE_JSON=" + json.dumps({
    "scope":"native matmul logical shapes, FLOPs, IO counts and finite external roofline call",
    "cases":results, "shape_checks_passed":len(results),
    "incompatible_k_rejected":incompatible_k_rejected,
    "device":"A100_80GB_fp16 (native specification, not local hardware)",
    "whole_transformer_mathematics_matched":False,
    "whole_system_executed":False, "gpu_execution":False,
    "core_modifications":0,
}))
