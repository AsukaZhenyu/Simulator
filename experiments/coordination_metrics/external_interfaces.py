"""Execute exact retained upstream definitions for coordination interface audits.

No project mapping/modeling imports, numerical kernels, full serving engine,
or hardware simulator. These are versioned interface calls, not tool timings.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple, Union

ROOT = Path(__file__).resolve().parents[3]
COMPASS = ROOT/'Simulator/build/dac-validation-20261025/e1-external-v2/LLMCompass'
SERVING = ROOT/'Simulator/build/dac-validation-20261025/e1-external-v2/LLMServingSim'
CHAKRA = ROOT/'Simulator/build/dac-followup-20261025/external-wsl-build-v1/astra-sim/extern/graph_frontend/chakra'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class ExactDefinitions:
    def __init__(self):
        self.evidence = []

    def get(self, path, name, environment, parent=None):
        path = Path(path)
        tree = ast.parse(path.read_text(encoding='utf-8'))
        scope = tree.body
        if parent:
            scope = next(n for n in scope if isinstance(n, ast.ClassDef) and n.name == parent).body
        node = next(n for n in scope if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
        exec(compile(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]), str(path), 'exec'), environment)
        self.evidence.append(dict(path=str(path.relative_to(ROOT)), symbol=(parent+'.' if parent else '')+name,
            first_line=node.lineno, last_line=node.end_lineno, source_sha256=sha(path),
            body_modified=False))
        return environment[name]

class InterfaceBundle:
    def __init__(self):
        self.extractor = ExactDefinitions()
        get = self.extractor.get
        env = dict(List=List, Tuple=Tuple, Union=Union)
        get(COMPASS/'utils.py', 'size_of_list', env)
        get(COMPASS/'utils.py', 'size', env)
        self.DataType = get(COMPASS/'software_model/utils.py', 'DataType', env)
        env['data_type_dict'] = {'fp16': self.DataType('fp16', 2)}
        self.Tensor = get(COMPASS/'software_model/utils.py', 'Tensor', env)
        self.MatmulGraph = get(COMPASS/'software_model/matmul.py', 'ComputationalGraph', env, 'Matmul')
        self.matmul_call = get(COMPASS/'software_model/matmul.py', '__call__', env, 'Matmul')
        self.SoftmaxGraph = get(COMPASS/'software_model/softmax.py', 'ComputationalGraph', env, 'Softmax')
        self.softmax_call = get(COMPASS/'software_model/softmax.py', '__call__', env, 'Softmax')
        self.reshape_call = get(COMPASS/'software_model/operators.py', '__call__', env, 'Reshape')
        self.concat_call = get(COMPASS/'software_model/operators.py', '__call__', env, 'Concat')
        self.transpose_call = get(COMPASS/'software_model/operators.py', '__call__', env, 'Transpose')
        self.dtype = self.DataType('fp32', 4)
        self.config = None
        senv = {'get_config': lambda model: self.config}
        self.calculate_sizes = get(SERVING/'serving/core/memory_model.py', 'calculate_sizes', senv)
        self.get_device = get(SERVING/'serving/core/config_builder.py', 'get_device', {})
        self.indexed_cols = get(SERVING/'serving/core/trace_generator.py', 'indexed_cols', {'_TRACE_ROW_FIELDS': 11})
        self.Layer = get(CHAKRA/'src/converter/llm_converter.py', 'Layer', {'List': List})

    def matmul(self, lhs, rhs):
        op = SimpleNamespace(data_type=self.dtype, ComputationalGraph=self.MatmulGraph)
        out = self.matmul_call(op, self.Tensor(list(lhs), self.dtype), self.Tensor(list(rhs), self.dtype))
        return dict(kind='matmul', lhs=list(lhs), rhs=list(rhs), output_shape=out.shape,
                    flops=op.flop_count, logical_io_elements=op.io_count)

    def softmax(self, shape):
        op = SimpleNamespace(data_type=self.dtype, ComputationalGraph=self.SoftmaxGraph)
        out = self.softmax_call(op, self.Tensor(list(shape), self.dtype))
        return dict(kind='softmax', input_shape=list(shape), output_shape=out.shape, M=op.M, N=op.N)

    def reshape(self, before, after):
        op = SimpleNamespace(data_type=self.dtype)
        out = self.reshape_call(op, self.Tensor(list(before), self.dtype), list(after))
        return dict(kind='reshape', input_shape=list(before), output_shape=out.shape)

    def transpose(self, before, permutation):
        op = SimpleNamespace(data_type=self.dtype)
        out = self.transpose_call(op, self.Tensor(list(before), self.dtype), list(permutation))
        return dict(kind='transpose', input_shape=list(before), permutation=list(permutation), output_shape=out.shape)

    def concat(self, lhs, rhs, axis):
        op = SimpleNamespace(data_type=self.dtype)
        out = self.concat_call(op, self.Tensor(list(lhs), self.dtype), self.Tensor(list(rhs), self.dtype), axis)
        return dict(kind='concat', lhs=list(lhs), rhs=list(rhs), axis=axis, output_shape=out.shape)

    def serving_sizes(self, config, layer, tokens, history=0, parallel=1):
        self.config = dict(config)
        inp, weights, out = self.calculate_sizes('coordination-fixture', layer, tokens, kv_len=history,
                                                parallel=parallel, fp=4)
        return dict(layer=layer, input_bytes=inp, weight_bytes=weights, output_bytes=out)

    def serving_placement(self, weight_location, kv_location='LOCAL', layer_override=None):
        placement = dict(default=dict(weights=weight_location, kv_loc=kv_location, kv_evict_loc=kv_location),
                         block=[], layer=layer_override or {})
        return dict(source=placement, weights=self.get_device(placement, 0, 'audit_stage', 'weights'),
                    kv_loc=self.get_device(placement, 0, 'audit_stage', 'kv_loc'))

    def parse_rows(self, rows):
        parsed = [self.Layer(cols=cols) for cols in self.indexed_cols(rows)]
        return [dict(name=p.name, compute_ns=p.comp_time, input_bytes=p.input_memory_size,
                     weight_bytes=p.weight_memory_size, output_bytes=p.output_memory_size,
                     input_location=p.input_memory_loc, weight_location=p.weight_memory_loc,
                     output_location=p.output_memory_loc) for p in parsed]

    def sources_unchanged(self):
        return all(sha(ROOT/item['path']) == item['source_sha256'] for item in self.extractor.evidence)
