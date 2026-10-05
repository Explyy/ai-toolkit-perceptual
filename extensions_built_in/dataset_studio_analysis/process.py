from __future__ import annotations
import json
from contextlib import closing
import os
import platform
import socket
import time
import sqlite3
from pathlib import Path
from jobs.process import BaseExtensionProcess
from .artifacts import configuration, verified_models, safe_file, digest, atomic_json

class DatasetStudioAnalysisProcess(BaseExtensionProcess):
    def __init__(self, process_id, job, config):
        super().__init__(process_id, job, config)
        self.request = Path(self.get_conf('request', required=True))
        self.db = Path(self.get_conf('sqlite_db_path', required=True))
        self.job_id = os.environ.get('AITK_JOB_ID')

    def run(self):
        super().run()
        # This gate precedes every ML import, even when invoked manually via run.py.
        if platform.system() != 'Linux' or os.environ.get('DATASET_STUDIO_ANALYSIS_ENABLED') != '1':
            raise RuntimeError('Analysis requires explicitly enabled dedicated Linux GPU host')
        root=Path(os.environ['DATASET_STUDIO_ROOT']).resolve(strict=True)
        if self.request.is_symlink(): raise ValueError('Symlink request refused')
        self.request.resolve(strict=True).relative_to(root)
        request=json.loads(self.request.read_text())
        manifest,config=configuration()
        if request.get('config') != config or request.get('name') != self.job.name or not self.job_id:
            raise ValueError('Analysis request/config/job identity differs')
        folder=self.request.parent
        with closing(sqlite3.connect(self.db,timeout=30)) as db, db:
            row=db.execute('SELECT name,job_type,job_ref FROM Job WHERE id=?',(self.job_id,)).fetchone()
            if row != (self.job.name,'analysis',request['dataset']): raise ValueError('Native analysis identity differs')
        start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
        atomic_json(folder/'runtime.json',{'pid':os.getpid(),'start':start,'host':socket.gethostname(),'boot':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'namespace':os.readlink('/proc/self/ns/pid'),'name':self.job.name,'config':config})
        def update(status,info,step=0):
            with closing(sqlite3.connect(self.db,timeout=30)) as db, db:
                db.execute('UPDATE Job SET status=?,info=?,step=?,total_steps=? WHERE id=? AND name=? AND job_type=?',
                           (status,info,step,len(request['items']),self.job_id,self.job.name,'analysis'))
        started=time.time()*1000
        samples=[]
        def progress(done,current=None,finished=False):
            value={'name':self.job.name,'config':config,'total':len(request['items']),'done':done,'startedAt':started,'samples':samples[-16:]}
            if current is not None:value['currentIndex']=current
            if finished:value['finishedAt']=time.time()*1000
            atomic_json(folder/'progress.json',value)
        progress(0)
        update('running','Loading pinned analysis models')
        try:
            models=verified_models(safe_file(root,'cache/analysis-models'))
            from .inference import Analyzer
            analyzer=Analyzer(models)
            for i,item in enumerate(request['items']):
                with closing(sqlite3.connect(self.db,timeout=30)) as db, db:
                    stop=db.execute('SELECT stop FROM Job WHERE id=?',(self.job_id,)).fetchone()
                if stop and stop[0]:
                    update('stopped','Analysis stopped; completed cache retained',i)
                    return
                progress(i,i)
                source=safe_file(folder,item['input'])
                if digest(source)!=item['sha']: raise ValueError('Staged input SHA differs')
                out=safe_file(folder,item['output'],missing=True)
                out.mkdir(parents=True,exist_ok=True)
                result=out/'result.json'
                cached=json.loads(result.read_text()) if result.is_file() and not result.is_symlink() else None
                if not cached or cached.get('sha')!=item['sha'] or cached.get('config')!=config or digest(safe_file(out,'depth.png'))!=cached.get('depth',{}).get('sha'):
                    sample=time.monotonic()
                    atomic_json(result,analyzer.analyze(source,out,item['sha']))
                    samples.append(time.monotonic()-sample)
                progress(i+1,i)
                update('running','Analyzing images',i+1)
            progress(len(request['items']),finished=True)
            update('completed','Analysis completed',len(request['items']))
        except BaseException as e:
            update('error','Analysis unavailable or failed; originals preserved')
            raise
