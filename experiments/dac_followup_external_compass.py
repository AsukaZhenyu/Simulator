"""Execute pinned LLMCompass whole blocks without modifying its source tree.

Run from the external checkout. The only in-memory compatibility adapter
initializes the zero TP=1 collective diagnostic variable. Its original failure
and exact source/adapter hashes are retained. No CUDA kernels are executed.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import sys
import textwrap
import traceback


def native_block_rows(block, device):
    groups = {
        "projection": ["Q_proj", "K_proj", "V_proj"],
        "attention": ["Q_mul_K", "A_softmax", "A_mul_V", "H_matmul0", "layer_norm0"],
        "dense_ffn": ["H_matmul1", "H_gelu", "H_matmul2", "layer_norm1"],
    }
    overheads = device.compute_module.overhead
    rows = []
    for macro, names in groups.items():
        for name in names:
            op = getattr(block, name)
            kind = ("softmax" if name == "A_softmax" else "layernorm" if name.startswith("layer_norm")
                    else "gelu" if name == "H_gelu" else "matmul")
            cost = op.roofline_model(device)
            overhead = float(getattr(overheads, kind))
            if hasattr(op, "input1_shape"):
                shapes = [op.input1_shape, op.input2_shape]
                output = op.output_shape
                flops = 2 * op.M * op.K * op.N * getattr(op, "bs", 1)
            else:
                shapes = [op.shape]
                output = op.shape
                flops = (op.M * (10 + device.compute_module.core.vector_unit.flops_per_exp)
                         if kind == 'gelu' else int(op.flop_count))
            rows.append(dict(id=name, macro=macro, operator=type(op).__name__,
                             input_shapes=shapes, output_shape=output,
                             roofline_seconds=cost, overhead_seconds=overhead,
                             inclusive_seconds=cost + overhead,
                             logical_input_bytes=[math.prod(s) * 2 for s in shapes],
                             logical_output_bytes=math.prod(output) * 2,
                             logical_port_dtype='fp16',
                             native_internal_cost_dtype=(op.computational_graph.data_type.name
                                 if hasattr(op,'computational_graph') else 'fp16'),
                             native_flop_count=flops))
    return rows


def main(out):
    sys.path.insert(0,str(Path.cwd()))
    import_failure=None
    unused_gpu_decorators=[]
    try:
        import software_model.transformer as module
    except RuntimeError as error:
        if not any(message in str(error) for message in (
                'not yet supported for torch.compile',
                'Dynamo is not supported on Python 3.12+')):
            raise
        import_failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        import torch
        original_compile=torch.compile
        def unused_helper_decorator(function=None,**kwargs):
            identity=(getattr(function,'__module__',None),getattr(function,'__name__',None))
            if identity in {('software_model.layernorm','layernorm_gpu'),('software_model.gelu','gelu_gpu')}:
                unused_gpu_decorators.append('.'.join(identity))
                return function
            return original_compile(function,**kwargs)
        torch.compile=unused_helper_decorator
        try:
            import software_model.transformer as module
        finally:
            torch.compile=original_compile
    from software_model.utils import Tensor, data_type_dict
    from hardware_model.device import device_dict
    from hardware_model.system import System

    cls = module.TransformerBlockInitComputationTP
    source = inspect.getsource(cls.roofline_model)
    before = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    original_failure = None
    first = cls(128, 2, 1, data_type_dict["fp16"])
    first(Tensor([1, 16, 128], data_type_dict["fp16"]))
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            first.roofline_model(System(device_dict["A100_80GB_fp16"], None))
    except Exception as error:
        original_failure = dict(type=type(error).__name__, message=str(error),
                                traceback=traceback.format_exc())
    if original_failure is None or original_failure['type']!='UnboundLocalError' or 'allreduce_latency' not in original_failure['message']:
        raise RuntimeError('Unexpected native TP1 outcome; do not apply a guessed compatibility fix: '+str(original_failure))
    # This changes no cost term: TP=1 has no all-reduce and the original
    # allreduce_total_latency is already zero. Only a print/log local was unset.
    tree = ast.parse(textwrap.dedent(source))
    function = tree.body[0]
    function.body.insert(0, ast.Assign(targets=[ast.Name(id="allreduce_latency", ctx=ast.Store())],
                                       value=ast.Constant(value=0.0)))
    ast.fix_missing_locations(tree)
    adapter_source = ast.unparse(tree)
    namespace = {}
    exec(compile(tree, "<declared-compass-tp1-null-collective-adapter>", "exec"),
         module.__dict__, namespace)
    cls.roofline_model = namespace["roofline_model"]
    cases = []
    for h in (128, 256):
        for q in (16, 64):
            for hardware in ("A100_80GB_fp16", "MI210"):
                block = cls(h, h // 64, 1, data_type_dict["fp16"])
                output = block(Tensor([1, q, h], data_type_dict["fp16"]))
                log = io.StringIO()
                with contextlib.redirect_stdout(log):
                    total = block.roofline_model(System(device_dict[hardware], None))
                rows = native_block_rows(block, device_dict[hardware])
                metadata=[]
                for name in ('Q_reshape','K_reshape','V_reshape','Q_transpose',
                             'K_transpose','V_transpose','H_transpose','H_reshape'):
                    op=getattr(block,name)
                    metadata.append(dict(id=name,operator=type(op).__name__,
                                         input_shape=op.input_shape,output_shape=op.output_shape,
                                         permutation=getattr(op,'permute',None),
                                         priced_by_native_whole_block=False,
                                         physical_zero_cost_claimed=False))
                independent_sum = sum(r["inclusive_seconds"] for r in rows)
                assert len(rows) == 12
                assert math.isclose(total, independent_sum, rel_tol=1e-13, abs_tol=1e-18)
                assert output.shape == [1, q, h]
                cases.append(dict(id=f"h{h}-q{q}-{hardware}", hidden=h, tokens=q,
                                  heads=h//64, intermediate=4*h, dtype="fp16",
                                  hardware=hardware, output_shape=output.shape,
                                  native_whole_block_seconds=total,
                                  independent_native_operator_sum_seconds=independent_sum,
                                  operations=rows,native_metadata_operations=metadata,
                                  native_stdout=log.getvalue()))
    after = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    assert before == after
    payload = dict(schema="dac-followup-compass-whole-block-1", cases=cases,
                   original_import_failure=import_failure,
                   unused_gpu_decorator_import_adapter=dict(
                       bypassed_helpers=unused_gpu_decorators,
                       reason='Installed torch.compile rejects Python3.12; helpers are not called by any roofline method',
                       gpu_helpers_executed=False,torch_compile_restored=True,
                       cost_formula_changed=False),
                   original_native_tp1_failure=original_failure,
                   original_native_stdout=stdout.getvalue(),
                   in_memory_adapter=dict(reason="Initialize diagnostic null collective to zero at TP=1",
                                          original_function_sha256=hashlib.sha256(source.encode()).hexdigest(),
                                          adapted_function_sha256=hashlib.sha256(adapter_source.encode()).hexdigest(),
                                          source=adapter_source, cost_formula_changed=False),
                   external_transformer_source_unchanged=before == after,
                   source_sha256=before,
                   native_whole_blocks_executed=len(cases),
                   real_device_execution=False,
                   model_scope="Exact native Compass init block: MHA, two LayerNorms, GeLU FFN; no residual, RoPE or explicit causal-mask cost",
                   kernel_compiler_simulation=False,
                   timing_scope="Native whole-block roofline_model, not compile_and_simulate or measured latency")
    Path(out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("FOLLOWUP_JSON=" + json.dumps({"blocks":len(cases), "source_unchanged":True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    main(parser.parse_args().out)
