"""Deterministic startup/persistence guards using a disposable real Prisma DB."""
import importlib.util
from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('cloud', REPO / 'scripts/dataset_studio_cloud.py')
cloud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cloud)

class StartupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.mount = self.base / 'workspace'
        self.mount.mkdir()
        self.toolkit = self.base / 'toolkit'
        schema = self.toolkit / 'ui/prisma/schema.prisma'
        schema.parent.mkdir(parents=True)
        shutil.copy(REPO / 'ui/prisma/schema.prisma', schema)
        self.env = {'AI_TOOLKIT_AUTH': 'a' * 48, 'HF_TOKEN': 'private-test-token',
                    'DATASET_STUDIO_VOLUME_ID': '3489', 'DATASET_STUDIO_MOUNT': str(self.mount),
                    'DATASET_STUDIO_ROOT': str(self.mount / 'dataset-studio')}

    def initialize(self, db, schema):
        subprocess.run(['node', str(REPO / 'ui/node_modules/prisma/build/index.js'), 'db', 'push', '--schema', str(schema), '--skip-generate'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def prepare(self, env=None, mounted=True):
        space = type('Space', (), {'f_bavail': 40_000_000_000, 'f_frsize': 1})()
        with patch.object(cloud.os, 'statvfs', return_value=space):
            return cloud.prepare(env or self.env, self.toolkit, lambda _: mounted, self.initialize)

    def test_auth_child_is_loopback_and_does_not_receive_hf_secret(self):
        commands = cloud.child_commands(self.toolkit, self.env, Path('/run/test-gateway.conf'))
        auth_command, auth_env = commands[0]
        self.assertEqual(auth_command, [sys.executable, str(self.toolkit / 'scripts/dataset_studio_access.py'), 'serve'])
        self.assertEqual(auth_env, {'AI_TOOLKIT_AUTH': self.env['AI_TOOLKIT_AUTH'], 'PYTHONDONTWRITEBYTECODE': '1'})
        self.assertEqual(commands[1][1]['HF_TOKEN'], self.env['HF_TOKEN'])
        for command, env in commands[2:]:
            self.assertNotIn('AI_TOOLKIT_AUTH', env)
        self.assertIn('127.0.0.1', commands[2][0])
        self.assertIn('8676', commands[2][0])

    def test_release_manifest_binds_access_module_bytes(self):
        before = cloud.source_manifest(REPO)
        rows = {row['path']: row['sha256'] for row in before['files']}
        import hashlib
        self.assertEqual(rows['scripts/dataset_studio_access.py'], hashlib.sha256((REPO / 'scripts/dataset_studio_access.py').read_bytes()).hexdigest())
        self.assertIn('docker/dataset-studio/test_access.py', rows)
        self.assertFalse(any('.private' in path or path.endswith('.db') for path in rows))

    def test_restart_preserves_db_dataset_and_metadata(self):
        root, db = self.prepare()
        inode = db.stat().st_ino
        (root / 'datasets/tiny.png').write_bytes(b'unique synthetic original')
        (root / 'data/selection.json').write_text('{"caption":"saved","selected":true}')
        with closing(sqlite3.connect(db)) as c, c:
            c.execute("INSERT INTO Queue(gpu_ids,is_running) VALUES ('0',false)")
        root2, db2 = self.prepare()
        self.assertEqual(db2.stat().st_ino, inode)
        self.assertTrue((self.toolkit / 'aitk_db.db').samefile(db))
        self.assertEqual((root2 / 'datasets/tiny.png').read_bytes(), b'unique synthetic original')
        self.assertTrue(json.loads((root2 / 'data/selection.json').read_text())['selected'])
        with closing(sqlite3.connect(db2)) as c:
            self.assertEqual(c.execute('SELECT gpu_ids,is_running FROM Queue').fetchall(), [('0', 0)])
            self.assertEqual(c.execute("SELECT value FROM Settings WHERE key='DATASETS_FOLDER'").fetchone()[0], str(root / 'datasets'))
            self.assertEqual(c.execute('PRAGMA journal_mode').fetchone()[0], 'delete')

    def test_auth_hf_and_mount_required_before_writes(self):
        for key in ['AI_TOOLKIT_AUTH', 'HF_TOKEN', 'DATASET_STUDIO_VOLUME_ID']:
            env = dict(self.env); env.pop(key)
            with self.assertRaises(cloud.StartupError): self.prepare(env)
        with self.assertRaises(cloud.StartupError): self.prepare(mounted=False)
        self.assertFalse((self.mount / 'dataset-studio').exists())

    def test_single_instance_lock_refuses_second_writer_and_recovers(self):
        root = self.mount / 'dataset-studio'; root.mkdir()
        first = cloud.acquire_lock(root)
        try:
            with self.assertRaises(cloud.StartupError): cloud.acquire_lock(root)
        finally: first.close()
        cloud.acquire_lock(root).close()

    def test_preview_and_nonpersistent_override_refused(self):
        for key, value in [('DATASET_STUDIO_PREVIEW', '1'), ('DATASET_STUDIO_DATA_ROOT', '/tmp/data'),
                           ('DATASET_STUDIO_ROOT', str(self.mount / 'existing-datasets')), ('AI_TOOLKIT_DB_JOURNAL_MODE', 'WAL'),
                           ('AI_TOOLKIT_AUTH', 'a' * 40 + '\ninclude evil;')]:
            env = dict(self.env); env[key] = value
            with self.assertRaises(cloud.StartupError): self.prepare(env)

    def test_volume_identity_rejects_other_binding(self):
        root, db = self.prepare()
        env = dict(self.env); env['DATASET_STUDIO_VOLUME_ID'] = 'other'
        before = db.read_bytes()
        with self.assertRaises(cloud.StartupError): self.prepare(env)
        self.assertEqual(db.read_bytes(), before)

    def test_existing_runtime_db_is_never_replaced(self):
        self.toolkit.mkdir(exist_ok=True)
        db = self.toolkit / 'aitk_db.db'; db.write_bytes(b'original runtime')
        with self.assertRaises(cloud.StartupError): self.prepare()
        self.assertEqual(db.read_bytes(), b'original runtime')

    def test_schema_drift_and_corruption_refused(self):
        root, db = self.prepare()
        with closing(sqlite3.connect(db)) as c, c: c.execute('ALTER TABLE Job ADD COLUMN unrelated TEXT')
        before = db.read_bytes()
        with self.assertRaises(cloud.StartupError): self.prepare()
        self.assertEqual(db.read_bytes(), before)

    def assert_schema_refused_without_writes(self, db):
        before = db.read_bytes()
        with self.assertRaisesRegex(cloud.StartupError, 'schema/constraints differ'):
            self.prepare()
        self.assertEqual(db.read_bytes(), before, 'Guard must not modify or repair existing DB')

    def test_missing_unique_constraints_refused(self):
        root, db = self.prepare()
        for index in ['Job_name_key', 'Queue_gpu_ids_key', 'Settings_key_key']:
            with self.subTest(index=index):
                with closing(sqlite3.connect(db)) as c, c:
                    sql = c.execute('SELECT sql FROM sqlite_master WHERE name=?', (index,)).fetchone()[0]
                    c.execute('DROP INDEX "' + index + '"')
                self.assert_schema_refused_without_writes(db)
                with closing(sqlite3.connect(db)) as c, c: c.execute(sql)

    def test_missing_native_nonunique_index_refused(self):
        root, db = self.prepare()
        with closing(sqlite3.connect(db)) as c, c: c.execute('DROP INDEX "Job_status_idx"')
        self.assert_schema_refused_without_writes(db)

    def rebuild_job(self, db, old, new):
        with closing(sqlite3.connect(db)) as c, c:
            ddl = c.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='Job'").fetchone()[0]
            self.assertIn(old, ddl)
            indexes = [row[0] for row in c.execute("SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='Job' AND sql IS NOT NULL")]
            c.execute('DROP TABLE Job')
            c.execute(ddl.replace(old, new, 1))
            for sql in indexes: c.execute(sql)

    def test_changed_primary_key_refused(self):
        root, db = self.prepare()
        self.rebuild_job(db, 'PRIMARY KEY', '')
        self.assert_schema_refused_without_writes(db)

    def test_changed_not_null_refused(self):
        root, db = self.prepare()
        self.rebuild_job(db, '"name" TEXT NOT NULL', '"name" TEXT')
        self.assert_schema_refused_without_writes(db)

    def test_changed_default_refused(self):
        root, db = self.prepare()
        self.rebuild_job(db, '"status" TEXT NOT NULL DEFAULT \'stopped\'', '"status" TEXT NOT NULL DEFAULT \'queued\'')
        self.assert_schema_refused_without_writes(db)

    def test_reference_fixture_does_not_modify_source_schema(self):
        schema = self.toolkit / 'ui/prisma/schema.prisma'
        before = schema.read_bytes()
        self.prepare()
        self.prepare()
        self.assertEqual(schema.read_bytes(), before)
        self.assertFalse((schema.parent / 'reference.db').exists())

    def test_symlink_escape_refused(self):
        root = self.mount / 'dataset-studio'; root.symlink_to(self.toolkit, target_is_directory=True)
        with self.assertRaises(cloud.StartupError): self.prepare()

    def test_volume_reserve_refused(self):
        space = type('Space', (), {'f_bavail': 1, 'f_frsize': 1})()
        with patch.object(cloud.os, 'statvfs', return_value=space):
            with self.assertRaises(cloud.StartupError): cloud.prepare(self.env, self.toolkit, lambda _: True, self.initialize)
        self.assertFalse((self.mount / 'dataset-studio').exists())

if __name__ == '__main__': unittest.main()
