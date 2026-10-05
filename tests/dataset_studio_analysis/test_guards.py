"""No-ML tests of actual entry-point gates using only a native-base interface stub.
The final image separately imports the real pinned native base; these are not model proofs.
"""
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]/'extensions_built_in/dataset_studio_analysis'

class MinimalBase:
    def __init__(self,process_id,job,config): self.config=config; self.job=job
    def get_conf(self,key,required=False): return self.config[key]
    def run(self): pass

def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/file)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value

class Guards(unittest.TestCase):
    def test_mac_process_refuses_before_ml_or_storage_access(self):
        package=types.ModuleType('studio_test');package.__path__=[str(ROOT)]
        jobs=types.ModuleType('jobs.process');jobs.BaseExtensionProcess=MinimalBase
        with patch.dict(sys.modules,{'studio_test':package,'jobs.process':jobs}):
            module('studio_test.artifacts','artifacts.py');process=module('studio_test.process','process.py')
            value=process.DatasetStudioAnalysisProcess(0,types.SimpleNamespace(name='fixture'),{'request':'/must-not-read','sqlite_db_path':'/must-not-open'})
            before=set(sys.modules)
            with patch.object(process.platform,'system',return_value='Darwin'),patch.dict(os.environ,{'DATASET_STUDIO_ANALYSIS_ENABLED':'1'}):
                with self.assertRaisesRegex(RuntimeError,'dedicated Linux'): value.run()
            self.assertNotIn('studio_test.inference',sys.modules)
            self.assertFalse(any(n in set(sys.modules)-before for n in ('torch','onnxruntime','transformers')))

    def test_provision_refuses_disabled_linux_before_network_or_storage(self):
        package=types.ModuleType('studio_provision');package.__path__=[str(ROOT)]
        with patch.dict(sys.modules,{'studio_provision':package}):
            module('studio_provision.artifacts','artifacts.py');provision=module('studio_provision.provision','provision.py')
            with patch.object(provision.platform,'system',return_value='Linux'),patch.dict(os.environ,{'DATASET_STUDIO_ANALYSIS_ENABLED':'0'}),patch.object(provision.urllib.request,'urlopen',side_effect=AssertionError('must not connect')):
                with self.assertRaisesRegex(SystemExit,'authorized dedicated Linux'): provision.main()

if __name__=='__main__': unittest.main()

class ProvisionRetry(unittest.TestCase):
    def test_interrupted_synthetic_download_retry_cleans_only_owned_partial(self):
        import hashlib
        import io
        import tempfile
        package=types.ModuleType('studio_retry');package.__path__=[str(ROOT)]
        with patch.dict(sys.modules,{'studio_retry':package}):
            artifacts=module('studio_retry.artifacts','artifacts.py');provision=module('studio_retry.provision','provision.py')
            payload=b'pure synthetic contract bytes; not model weights'
            manifest={'models':{'face':{'repo':'synthetic/fixture','revision':'a'*40,'files':{'fixture.onnx':{'size':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}}}}}
            class Interrupted(io.BytesIO):
                def read(self,n=-1):
                    if self.tell()>0:raise OSError('simulated connection interruption')
                    return super().read(4)
            with tempfile.TemporaryDirectory() as t,patch.object(provision.platform,'system',return_value='Linux'),patch.dict(os.environ,{'DATASET_STUDIO_ANALYSIS_ENABLED':'1','DATASET_STUDIO_ROOT':t,'DATASET_STUDIO_MIN_FREE_BYTES':'0'}),patch.object(provision,'configuration',return_value=(manifest,'config')),patch.object(artifacts,'configuration',return_value=(manifest,'config')):
                root=Path(t)/'cache/analysis-models';root.mkdir(parents=True)
                foreign=root/'foreign.download';foreign.write_bytes(b'foreign untouched')
                with patch.object(provision.urllib.request,'urlopen',return_value=Interrupted(payload)):
                    with self.assertRaisesRegex(OSError,'interruption'):provision.main()
                self.assertEqual(list(root.rglob('*.download-*')),[])
                with patch.object(provision.urllib.request,'urlopen',return_value=io.BytesIO(payload)):
                    provision.main()
                self.assertEqual((root/'face/fixture.onnx').read_bytes(),payload)
                self.assertEqual(foreign.read_bytes(),b'foreign untouched')
                self.assertEqual(list(root.rglob('*.download-*')),[])
