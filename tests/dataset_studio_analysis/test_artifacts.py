"""Synthetic contract tests; never import or execute ML."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('studio_artifacts',Path(__file__).resolve().parents[2]/'extensions_built_in/dataset_studio_analysis/artifacts.py')
a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)

class Artifacts(unittest.TestCase):
    def test_pinned_configuration_stable_and_complete(self):
        value,digest=a.configuration()
        self.assertEqual(digest,a.configuration()[1])
        self.assertEqual(set(value['models']),{'face','faceDetector','depth','person','pose'})
        self.assertEqual(value['models']['face']['files'].keys(),{'glintr100.onnx'})
        for model in value['models'].values():
            self.assertEqual(len(model['revision']),40)
            for file in model['files'].values(): self.assertEqual(len(file['sha256']),64)

    def test_escape_symlink_and_missing_refused(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); (root/'link').symlink_to('/tmp')
            for path in ['../escape','/tmp/file','link/foo','missing']:
                with self.subTest(path=path),self.assertRaises((ValueError,FileNotFoundError)):
                    a.safe_file(root,path)

    def test_real_sha_refuses_changed_model_and_processor(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); model=root/'face'; model.mkdir(); file=model/'sample.onnx'; file.write_bytes(b'synthetic model contract')
            expected={'size':file.stat().st_size,'sha256':a.digest(file)}
            manifest={'models':{'face':{'files':{'sample.onnx':expected}}}}
            with patch.object(a,'configuration',return_value=(manifest,'config')):
                self.assertEqual(a.verified_models(root)['face'],model)
                file.write_bytes(b'changed model')
                with self.assertRaisesRegex(ValueError,'checksum'): a.verified_models(root)
            # Processor JSON is independently hashed, not trusted merely because
            # the binary weights match their pin.
            file.write_bytes(b'synthetic model contract')
            processor=model/'preprocessor_config.json';processor.write_bytes(b'{"size":128}')
            manifest['models']['face']['files']['preprocessor_config.json']={'size':processor.stat().st_size,'sha256':a.digest(processor)}
            with patch.object(a,'configuration',return_value=(manifest,'config')):
                self.assertEqual(a.verified_models(root)['face'],model)
                processor.write_bytes(b'{"size":256}')
                with self.assertRaisesRegex(ValueError,'checksum'): a.verified_models(root)

    def test_atomic_receipt_round_trip_no_nan(self):
        with tempfile.TemporaryDirectory() as t:
            file=Path(t)/'result.json'; a.atomic_json(file,{'sha':'abc','progress':1})
            self.assertEqual(json.loads(file.read_text()),{'sha':'abc','progress':1})
            with self.assertRaises(ValueError): a.atomic_json(file,{'value':float('nan')})
            self.assertEqual(json.loads(file.read_text()),{'sha':'abc','progress':1})

if __name__=='__main__': unittest.main()
