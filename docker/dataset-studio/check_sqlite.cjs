// Real native-addon ABI gate. Uses only a disposable in-memory database.
const assert = require('node:assert/strict');
const path = require('node:path');
const { createRequire } = require('node:module');
const toolkitUI = process.argv[2] || '/app/ai-toolkit/ui';
const load = createRequire(path.join(toolkitUI, 'package.json'));

async function check() {
  assert.equal(process.versions.node.split('.')[0], '22', 'Studio requires the pinned Node 22 runtime');
  const sqlite = load('sqlite3'); // dlopen must succeed on this stage's libc.
  const db = await new Promise((resolve, reject) => {
    new sqlite.Database(':memory:', function (error) {
      if (error) reject(error);
      else resolve(this);
    });
  });
  let version;
  try {
    const run = (sql, parameters = []) => new Promise((resolve, reject) => {
      db.run(sql, parameters, error => error ? reject(error) : resolve());
    });
    await run('CREATE TABLE studio_probe (id INTEGER PRIMARY KEY, value TEXT NOT NULL)');
    await run('INSERT INTO studio_probe (value) VALUES (?)', ['synthetic ABI probe']);
    const row = await new Promise((resolve, reject) => {
      db.get('SELECT id, value, sqlite_version() AS version FROM studio_probe WHERE id = ?', [1],
        (error, result) => error ? reject(error) : resolve(result));
    });
    assert.equal(row.id, 1);
    assert.equal(row.value, 'synthetic ABI probe');
    assert.match(row.version, /^\d+\.\d+\.\d+$/);
    version = row.version;
  } finally {
    await new Promise((resolve, reject) => db.close(error => error ? reject(error) : resolve()));
  }
  console.log(JSON.stringify({ node: process.versions.node, platform: process.platform, architecture: process.arch,
    sqlite: version, openQueryClose: 'passed' }));
}

check().catch(error => {
  console.error('Studio native SQLite ABI check failed:', error.message);
  process.exitCode = 1;
});
