"""Synthetic byte-binding checks; no native or hardware execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("test_strict_runtime",HERE/"runtime_manifest.py")
r=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(r)


class RuntimeManifestTests(unittest.TestCase):
    def test_manifest_maps_all_bound_bytes_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory(prefix="strict-runtime-synthetic-",dir=HERE/"tests")as temporary:
            root=Path(temporary).resolve();self.assertTrue(root.is_relative_to((HERE/"tests").resolve()))
            files=[]
            for name in("original.cpp","input.obj","ggml-cuda.dll","patch.txt"):
                file=root/name;file.write_bytes(("SYNTHETIC ONLY "+name).encode());files.append(file)
            document=dict(format_version="ggml-private-strict-f32-runtime-v1",patch_hunks=1,
                **{group:{str(file):r.sha(file)}for group,file in zip(r.GROUPS,files)})
            path=root/"strict-f32-runtime.json"
            path.write_text(json.dumps(document)+"\n",encoding="utf-8",newline="\n")
            records,assets=r.discover([root])
            self.assertEqual(1,len(records));self.assertEqual(5,len(assets));self.assertTrue(r.verify_recorded(records,assets))
            self.assertFalse(records[0]["loader_paths_observed"])
            with self.assertRaises(ValueError):r.verify_recorded(records,{str(path):r.sha(path)})
            files[1].write_bytes(b"corrupt input")
            with self.assertRaises(ValueError):r.discover([root])

    def test_absent_manifest_is_not_invented_as_strict_backend(self):
        with tempfile.TemporaryDirectory(prefix="no-manifest-",dir=HERE/"tests")as temporary:
            root=Path(temporary).resolve();self.assertTrue(root.is_relative_to((HERE/"tests").resolve()))
            self.assertEqual(([],{}),r.discover([root]))


if __name__=="__main__":unittest.main()
