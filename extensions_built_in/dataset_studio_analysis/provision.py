"""Explicit provisioning only; never called by inference or the web API."""
import json
import os
import platform
import shutil
import urllib.request
import uuid
from pathlib import Path
from .artifacts import configuration, digest, atomic_json, verified_models, safe_file

def main():
    if platform.system()!='Linux' or os.environ.get('DATASET_STUDIO_ANALYSIS_ENABLED')!='1':
        raise SystemExit('Provision only on explicitly authorized dedicated Linux cloud host')
    owner=Path(os.environ['DATASET_STUDIO_ROOT']).resolve(strict=True)
    root=safe_file(owner,'cache/analysis-models',missing=True)
    root.mkdir(parents=True,exist_ok=True)
    if root.is_symlink(): raise ValueError('Symlink model cache refused')
    manifest,_=configuration()
    reserve=int(os.environ.get('DATASET_STUDIO_MIN_FREE_BYTES','24000000000'))
    projected=sum(e['size'] for m in manifest['models'].values() for e in m['files'].values())
    if shutil.disk_usage(root).free < reserve + projected:
        raise ValueError('Model provisioning would breach persistent storage reserve')
    for key,model in manifest['models'].items():
        folder=root/key
        if folder.is_symlink(): raise ValueError('Symlink model folder refused')
        folder.mkdir(exist_ok=True)
        entries={**model['files'],**model.get('notices',{})}
        for name,expected in entries.items():
            file=folder/name
            if file.is_symlink(): raise ValueError('Symlink model refused')
            if file.exists() and file.stat().st_size==expected['size'] and digest(file)==expected['sha256']: continue
            if file.exists(): raise ValueError('Existing model differs; no overwrite: '+key+'/'+name)
            url=expected.get('url') or ('https://media.githubusercontent.com/media/'+model['repo']+'/'+model['revision']+'/models/face_detection_yunet/'+name if key=='faceDetector'
                 else 'https://huggingface.co/'+model['repo']+'/resolve/'+model['revision']+'/'+name)
            temporary=file.with_name(file.name+'.download-'+uuid.uuid4().hex)
            try:
                with urllib.request.urlopen(url,timeout=60) as response, temporary.open('xb') as out:
                    total=0
                    while chunk:=response.read(1024*1024):
                        total+=len(chunk)
                        if total>expected['size']: raise ValueError('Model exceeds pinned size')
                        out.write(chunk)
                    out.flush(); os.fsync(out.fileno())
                if total!=expected['size'] or digest(temporary)!=expected['sha256']:
                    raise ValueError('Downloaded model checksum differs')
                # Exclusive publication: never replace a concurrently created foreign file.
                try:
                    os.link(temporary,file)
                except FileExistsError:
                    if file.is_symlink() or file.stat().st_size!=expected['size'] or digest(file)!=expected['sha256']:
                        raise ValueError('Concurrent model file differs; no overwrite')
            finally:
                temporary.unlink(missing_ok=True)
        if key in ('depth','person','pose'):
            atomic_json(folder/'provision.json',{'revision':model['revision'],'files':{n:e['sha256'] for n,e in model['files'].items()}})
    verified_models(root)
    print('Pinned analysis models verified; no inference performed')

if __name__=='__main__': main()
