"""Exercise real process progress receipts with synthetic bytes and a no-ML analyzer."""
import json
from contextlib import closing
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from test_guards import MinimalBase, module, ROOT

class Progress(unittest.TestCase):
    def test_actual_entry_reports_measured_samples_and_cached_completion_without_ml(self):
        package=types.ModuleType('progress_test');package.__path__=[str(ROOT)]
        jobs=types.ModuleType('jobs.process');jobs.BaseExtensionProcess=MinimalBase
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules,{'progress_test':package,'jobs.process':jobs}):
            artifacts=module('progress_test.artifacts','artifacts.py');process=module('progress_test.process','process.py')
            root=Path(tmp);folder=root/'analysis';folder.mkdir();(root/'cache/analysis-models').mkdir(parents=True)
            db=root/'native.db';config='c'*64;items=[]
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.execute('CREATE TABLE Job(id TEXT,name TEXT,job_type TEXT,job_ref TEXT,status TEXT,info TEXT,step INTEGER,total_steps INTEGER,stop INTEGER)')
                conn.execute('INSERT INTO Job VALUES(?,?,?,?,?,?,?,?,?)',('own','own-job','analysis',str(root/'dataset'),'queued','',0,3,0))
            for i in range(3):
                source=folder/f'input-{i:06d}.png';source.write_bytes(f'synthetic-{i}'.encode());items.append({'id':str(i),'sha':artifacts.digest(source),'input':source.name,'output':f'result-{i:06d}'})
            request=folder/'request.json';request.write_text(json.dumps({'name':'own-job','dataset':str(root/'dataset'),'config':config,'items':items}))
            calls=[]
            class Analyzer:
                def __init__(self,models):pass
                def analyze(self,source,out,sha):
                    calls.append(source.name);(out/'depth.png').write_bytes(b'synthetic-depth')
                    return {'sha':sha,'config':config,'depth':{'sha':artifacts.digest(out/'depth.png')}}
            inference=types.ModuleType('progress_test.inference');inference.Analyzer=Analyzer
            out=folder/items[2]['output'];out.mkdir();(out/'depth.png').write_bytes(b'cached-depth')
            (out/'result.json').write_text(json.dumps({'sha':items[2]['sha'],'config':config,'depth':{'sha':artifacts.digest(out/'depth.png')}}))
            read_text=Path.read_text;readlink=os.readlink
            def proc_link(p,*args,**kwargs):
                return 'pid:[fixture]' if str(p)=='/proc/self/ns/pid' else readlink(p,*args,**kwargs)
            def proc_read(p,*args,**kwargs):
                if str(p)=='/proc/self/stat':return '1 (fixture) '+' '.join(['R']+['0']*18+['123'])
                if str(p)=='/proc/sys/kernel/random/boot_id':return 'synthetic-boot'
                return read_text(p,*args,**kwargs)
            with patch.dict(sys.modules,{'progress_test.inference':inference}),patch.dict(os.environ,{'DATASET_STUDIO_ROOT':str(root),'DATASET_STUDIO_ANALYSIS_ENABLED':'1','AITK_JOB_ID':'own'}),patch.object(process.platform,'system',return_value='Linux'),patch.object(process,'configuration',return_value=({},config)),patch.object(process,'verified_models',return_value={}),patch.object(process.os,'readlink',side_effect=proc_link),patch.object(Path,'read_text',proc_read),patch.object(process.time,'monotonic',side_effect=[1.,3.,4.,8.]):
                value=process.DatasetStudioAnalysisProcess(0,types.SimpleNamespace(name='own-job'),{'request':str(request),'sqlite_db_path':str(db)});value.run()
            receipt=json.loads((folder/'progress.json').read_text())
            self.assertEqual(receipt['done'],3);self.assertEqual(receipt['total'],3);self.assertEqual(receipt['samples'],[2.,4.]);self.assertGreaterEqual(receipt['finishedAt'],receipt['startedAt'])
            self.assertEqual(calls,['input-000000.png','input-000001.png'])
            with closing(sqlite3.connect(db)) as conn, conn:self.assertEqual(conn.execute('SELECT status,step,total_steps FROM Job').fetchone(),('completed',3,3))
            self.assertNotIn('torch',sys.modules)
