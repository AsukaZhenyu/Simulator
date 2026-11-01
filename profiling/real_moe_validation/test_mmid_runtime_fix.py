"""Pure source/command contracts. No compiler, CUDA runtime, or model launched."""
from pathlib import Path
import unittest

from .mmid_primitive import ANCHOR, FENCE
from .mmid_runtime_fix import compile_command, fence_source, link_response


class MMIDRuntimeFix(unittest.TestCase):
    def test_exact_one_fence_and_all_lanes_reach_it(self):
        source = 'start\n' + ANCHOR + '    if (threadIdx.x != 0) { return; }\nend\n'
        fixed = fence_source(source)
        self.assertEqual(fixed.count(FENCE), 1)
        self.assertEqual(fixed.replace(FENCE, ''), source)

    def test_already_fenced_duplicate_and_missing_site_refused(self):
        source = ANCHOR + '    if (threadIdx.x != 0) { return; }\n'
        for changed in (source + ANCHOR, source.replace(ANCHOR, ''), source + FENCE):
            with self.assertRaises(ValueError):
                fence_source(changed)

    def test_compile_preserves_math_arch_graph_and_host_flags(self):
        old_source, old_object, old_include = map(Path, ('old/main.cu', 'old/main.obj', 'old/source'))
        command = ['nvcc', '-c', '-O2', '--use_fast_math', '-cudart', 'static', '-std=c++17',
            '--generate-code=arch=compute_89,code=[compute_89,sm_89]', '-DGGML_CUDA_USE_GRAPHS',
            '-Xcompiler=/EHsc,/Ob2,/MD,/Zc:preprocessor', '-DSIMULATOR_GPU_TAP_BUILD',
            '-I' + str(old_include), '-o', str(old_object), str(old_source)]
        result = compile_command(command, old_source=old_source, old_object=old_object,
            old_include=old_include, source=Path('new/mmid.cu'), object_file=Path('new/mmid.obj'), include=Path('new/source'))
        restore = [str(old_source) if x == str(Path('new/mmid.cu')) else str(old_object) if x == str(Path('new/mmid.obj'))
                   else '-I' + str(old_include) if x == '-I' + str(Path('new/source')) else x for x in result]
        self.assertEqual(restore, [x for x in command if x != '-DSIMULATOR_GPU_TAP_BUILD'])
        self.assertNotIn('-DSIMULATOR_GPU_TAP_BUILD', result)
        for absent in ('-O2', '-DGGML_CUDA_USE_GRAPHS', '-DSIMULATOR_GPU_TAP_BUILD'):
            with self.assertRaises(ValueError):
                compile_command([x for x in command if x != absent], old_source=old_source,
                    old_object=old_object, old_include=old_include, source=Path('new/mmid.cu'),
                    object_file=Path('new/mmid.obj'), include=Path('new/source'))

    def test_link_only_replaces_one_object_and_destination_artifacts(self):
        text = '"old/mmid.obj"\n"old/main.obj"\n"old/another.obj"\n"cudart.lib"\n/DLL /OUT:"old/cuda.dll" /IMPLIB:"old/cuda.lib"\n'
        kwargs = dict(old_object=Path('old/mmid.obj'), new_object=Path('new/mmid.obj'),
            old_dll=Path('old/cuda.dll'), new_dll=Path('new/cuda.dll'),
            old_implib=Path('old/cuda.lib'), new_implib=Path('new/cuda.lib'))
        # Match native separators on Windows too.
        for key in ('old_object', 'old_dll', 'old_implib'):
            text = text.replace(str(kwargs[key]).replace('\\', '/'), str(kwargs[key]))
        result = link_response(text, **kwargs)
        self.assertIn('"old/main.obj"', result)
        self.assertIn('"old/another.obj"', result)
        self.assertIn('"cudart.lib"', result)
        with self.assertRaises(ValueError):
            link_response(text + '"' + str(kwargs['old_object']) + '"\n', **kwargs)


if __name__ == '__main__':
    unittest.main()
