#!/usr/bin/env python3
"""Guarded startup for a dedicated Dataset Studio image; no provider API calls."""
from __future__ import annotations
import argparse
from contextlib import closing
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import tempfile

class StartupError(RuntimeError):
    pass

def require(condition, message):
    if not condition:
        raise StartupError(message)

def absolute(value, label):
    p = Path(value)
    require(p.is_absolute() and '..' not in p.parts, f'{label} must be an absolute path without traversal')
    for parent in [p, *p.parents]:
        require(not parent.is_symlink(), f'{label} must not contain symlinks')
    return p

def sqlite_schema(connection):
    # Keep quoted identifiers/literals intact; ignore formatting and SQL keyword
    # case only. Full DDL includes PK, nullability, defaults and unique indexes.
    pattern = r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^\]]*\]|[A-Za-z_][A-Za-z_0-9]*|\d+|[^\s]"
    result = []
    for kind, name, table, sql in connection.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
    ):
        tokens = re.findall(pattern, sql or '')
        normalized = [token if token[0] in "'\"`[" else token.upper() for token in tokens]
        result.append((kind, name, table, normalized))
    return result

def verify_schema(db, schema, initialize):
    # Build a pristine reference with the shipped Prisma engine on ephemeral
    # storage. The persistent DB is opened read-only and is never db-pushed.
    with tempfile.TemporaryDirectory(prefix='studio-prisma-reference-') as directory:
        root = Path(directory).resolve()
        fixture = root / 'reference.db'
        fixture.touch(mode=0o600)
        copied_schema = root / 'schema.prisma'
        source, substitutions = re.subn(r'(?m)^(\s*url\s*=\s*)"[^"]*"',
                                        lambda match: match[1] + json.dumps('file:' + str(fixture)), schema.read_text())
        require(substitutions == 1, 'Unexpected native datasource configuration')
        copied_schema.write_text(source)
        try:
            initialize(fixture, copied_schema)
        except (OSError, subprocess.CalledProcessError) as error:
            raise StartupError('Pristine native schema reference failed; existing database left unchanged') from error
        with closing(sqlite3.connect(f'file:{fixture}?mode=ro', uri=True)) as reference:
            expected = sqlite_schema(reference)
    with closing(sqlite3.connect(f'file:{db}?mode=ro', uri=True)) as connection:
        require(sqlite_schema(connection) == expected,
                'Persistent native schema/constraints differ; no automatic migration')
        require(connection.execute('PRAGMA quick_check').fetchone()[0] == 'ok', 'Persistent DB integrity check failed')

def write_json_once(file, value):
    require(not file.is_symlink(), 'Persistent identity files must not be symlinks')
    if file.exists():
        require(json.loads(file.read_text()) == value, 'Persistent Studio volume identity differs')
        return
    with file.open('x') as output:
        json.dump(value, output, sort_keys=True)
        output.flush()
        os.fsync(output.fileno())

def configuration(env):
    require(re.fullmatch(r'[A-Za-z0-9_-]{32,256}', env.get('AI_TOOLKIT_AUTH', '')), 'AI_TOOLKIT_AUTH must be a random 32–256 character URL-safe token')
    require(env.get('HF_TOKEN'), 'HF_TOKEN must be configured on the server')
    require(env.get('DATASET_STUDIO_VOLUME_ID'), 'DATASET_STUDIO_VOLUME_ID must match the provider binding')
    require(env.get('DATASET_STUDIO_PREVIEW', '') != '1', 'Cloud startup cannot run in preview mode')
    mount = absolute(env.get('DATASET_STUDIO_MOUNT', '/workspace'), 'Mount')
    root = absolute(env.get('DATASET_STUDIO_ROOT', str(mount / 'dataset-studio')), 'Studio root')
    require(root == mount / 'dataset-studio', 'Use the dedicated dataset-studio namespace directly under the mount')
    for key, value in {'DATASET_STUDIO_DATA_ROOT': root / 'data', 'DATASET_STUDIO_DATASETS_ROOT': root / 'datasets',
                       'DATASET_STUDIO_TRAINING_ROOT': root / 'output', 'DATASET_STUDIO_DB_URL': 'file:' + str(root / 'aitk_db.db')}.items():
        require(key not in env or env[key] == str(value), f'{key} conflicts with persistent namespace')
    require(env.get('AI_TOOLKIT_DB_JOURNAL_MODE', 'DELETE') == 'DELETE', 'Network volume requires DELETE journal mode')
    return mount, root

def prepare(env, toolkit, mount_check=os.path.ismount, initialize=None):
    mount, root = configuration(env)
    require(mount_check(mount), 'Persistent mount is missing; refusing ephemeral fallback')
    require(mount.is_dir(), 'Persistent mount is not a directory')
    require(os.statvfs(mount).f_bavail * os.statvfs(mount).f_frsize >= int(env.get('DATASET_STUDIO_MIN_FREE_BYTES', '24000000000')), 'Persistent volume reserve is insufficient')
    root.mkdir(exist_ok=True, mode=0o700)
    marker = root / 'volume.json'
    require(not marker.is_symlink(), 'Volume identity must not be a symlink')
    write_json_once(marker, {'schema': 1, 'volumeId': env['DATASET_STUDIO_VOLUME_ID'], 'namespace': 'dataset-studio'})
    for name in ['datasets', 'data', 'output', 'cache']:
        absolute(str(root / name), name).mkdir(exist_ok=True)
    db = absolute(str(root / 'aitk_db.db'), 'Database')
    alias = toolkit / 'aitk_db.db'
    require(not alias.exists() or alias.is_symlink(), 'Image DB must be absent; never replace a runtime DB')
    if alias.is_symlink():
        require(alias.resolve() == db, 'Native DB binding differs; refusing replacement')
    else:
        alias.symlink_to(db)
    schema = toolkit / 'ui/prisma/schema.prisma'
    def initialize_native(db_file, schema_file):
        subprocess.run(['node', str(toolkit / 'ui/node_modules/prisma/build/index.js'), 'db', 'push',
                        '--schema', str(schema_file), '--skip-generate'], cwd=toolkit / 'ui', check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    initializer = initialize or initialize_native
    fresh = not db.exists()
    if fresh:
        db.touch(mode=0o600, exist_ok=False)
        try:
            initializer(db, schema)
        except Exception as error:
            raise StartupError('Fresh native database initialization failed; inspect private runtime logs') from error
    verify_schema(db, schema, initializer)
    write_json_once(root / 'schema.json', {'schemaSha256': hashlib.sha256(schema.read_bytes()).hexdigest()})
    # Seed authoritative roots for cron/file-server, which use native settings.
    with closing(sqlite3.connect(db)) as connection, connection:
        connection.execute('PRAGMA journal_mode=DELETE')
        for key, value in {'DATASETS_FOLDER': str(root / 'datasets'), 'DATA_ROOT': str(root / 'data'),
                           'TRAINING_FOLDER': str(root / 'output'), 'MODELS_PATH': str(root / 'cache/models'),
                           'HF_TOKEN': env['HF_TOKEN']}.items():
            connection.execute('INSERT INTO Settings (key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value', (key, value))
    env.update(DATASET_STUDIO_DATA_ROOT=str(root / 'data'), DATASET_STUDIO_DATASETS_ROOT=str(root / 'datasets'),
               DATASET_STUDIO_TRAINING_ROOT=str(root / 'output'), DATASET_STUDIO_DB_URL='file:' + str(db),
               HF_HOME=str(root / 'cache/huggingface'), AI_TOOLKIT_DB_JOURNAL_MODE='DELETE')
    # Verify server and native worker see the very same persistent inode.
    require(alias.samefile(db), 'Native and UI database identities differ')
    return root, db

def receipt(root, db, toolkit):
    release = json.loads((toolkit / 'studio-release.json').read_text())
    return {'release': release, 'root': str(root), 'database': str(db),
            'databaseDevice': db.stat().st_dev, 'databaseInode': db.stat().st_ino,
            'mountAvailableBytes': os.statvfs(root).f_bavail * os.statvfs(root).f_frsize}

def source_manifest(toolkit):
    files = []
    for directory in ['ui/src', 'ui/cron', 'ui/public', 'ui/tests', 'docker/dataset-studio']:
        files.extend(p for p in (toolkit / directory).rglob('*') if p.is_file())
    files.extend(toolkit / name for name in ['ui/package.json', 'ui/package-lock.json', 'ui/prisma/schema.prisma',
                 'ui/tsconfig.json', 'ui/tsconfig.worker.json', 'ui/next.config.ts', 'ui/next-env.d.ts',
                 'ui/postcss.config.mjs', 'ui/tailwind.config.ts', 'scripts/dataset_studio_cloud.py', '.github/workflows/dataset-studio.yml'])
    rows = []
    for file in sorted(files):
        relative = file.relative_to(toolkit).as_posix()
        require(not file.is_symlink(), 'Release source cannot contain symlinks')
        if '__pycache__' in file.parts or file.name.startswith('.env') or file.suffix in ['.db', '.pem']:
            continue
        rows.append({'path': relative, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()})
    return {'schema': 1, 'files': rows}

def acquire_lock(root):
    lock = absolute(str(root / '.instance.lock'), 'Instance lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise StartupError('Another Studio instance owns this persistent database') from error
    return lock

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['serve', 'check', 'manifest'])
    parser.add_argument('--toolkit', type=Path, default=Path('/app/ai-toolkit'))
    args = parser.parse_args()
    if args.action == 'manifest':
        print(json.dumps(source_manifest(args.toolkit), sort_keys=True, separators=(',', ':')))
        return
    env = dict(os.environ)
    mount, root = configuration(env)
    require(os.path.ismount(mount), 'Persistent mount is missing; refusing ephemeral fallback')
    root.mkdir(exist_ok=True, mode=0o700)
    lock = acquire_lock(root)
    root, db = prepare(env, args.toolkit)
    print(json.dumps(receipt(root, db, args.toolkit)), flush=True)
    if args.action == 'check':
        return
    auth_file = Path('/run/dataset-studio-auth')
    hashed = subprocess.run(['openssl', 'passwd', '-6', '-stdin'], input=env['AI_TOOLKIT_AUTH'] + '\n', text=True,
                            capture_output=True, check=True).stdout.strip()
    auth_file.write_text('studio:' + hashed + '\n')
    auth_file.chmod(0o600)
    gateway = Path('/run/dataset-studio-nginx.conf')
    gateway.write_text(Path('/etc/nginx/dataset-studio.conf').read_text())
    gateway.chmod(0o600)
    processes = []
    def stop(signum, frame):
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        next_env = dict(env)
        # Outer gateway owns authentication; native bearer login would conflict
        # with the browser's Basic Authorization header. This child is loopback.
        next_env.pop('AI_TOOLKIT_AUTH', None)
        commands = [(['node', 'dist/cron/worker.js'], env),
                    (['node', 'node_modules/next/dist/bin/next', 'start', '--hostname', '127.0.0.1', '--port', '8676'], next_env),
                    (['nginx', '-c', str(gateway), '-g', 'daemon off;'], next_env)]
        for command, child_env in commands:
            processes.append(subprocess.Popen(command, cwd=args.toolkit / 'ui', env=child_env, start_new_session=True))
        import time
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
        code = next(process.returncode for process in processes if process.poll() is not None)
    finally:
        stop(signal.SIGTERM, None)
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        auth_file.unlink(missing_ok=True)
        gateway.unlink(missing_ok=True)
    sys.exit(code)

if __name__ == '__main__':
    try:
        main()
    except StartupError as error:
        print('Dataset Studio startup refused: ' + str(error), file=sys.stderr)
        sys.exit(1)
    except (OSError, ValueError, sqlite3.Error):
        # Environment/secrets and third-party exception messages are never logged.
        print('Dataset Studio startup refused: configuration, mount, identity or schema guard failed.', file=sys.stderr)
        sys.exit(1)
