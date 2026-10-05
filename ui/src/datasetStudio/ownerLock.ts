import fs from 'node:fs/promises';
import path from 'node:path';
import sqlite3 from 'sqlite3';
import { ensure, Problem } from './domain';
export class OperationBusy extends Problem {
  constructor() {
    super(409, 'Dataset operation active; refresh');
  }
}
export async function ownedLock<T>(folder: string, action: () => Promise<T>) {
  // Never guess whether an old mkdir-lock owner is dead, including foreign PID namespaces.
  const legacy = await fs.lstat(path.join(folder, 'lock')).catch((e: any) => {
    if (e.code !== 'ENOENT') throw e;
    return null;
  });
  ensure(!legacy, 'Legacy lock requires explicit reconciliation; owner death unproven', 409);
  const file = path.join(folder, '.operation-lock.db');
  const prior = await fs.lstat(file).catch((e: any) => {
    if (e.code !== 'ENOENT') throw e;
    return null;
  });
  ensure(!prior || (prior.isFile() && !prior.isSymbolicLink()), 'Operation lock path is unsafe', 403);
  for (const suffix of ['-journal', '-wal', '-shm']) {
    const side = await fs.lstat(file + suffix).catch((e: any) => {
      if (e.code !== 'ENOENT') throw e;
      return null;
    });
    ensure(!side || (side.isFile() && !side.isSymbolicLink()), 'Operation lock journal path is unsafe', 403);
  }
  // Exclusive creation prevents following a concurrently inserted symlink for a fresh lock DB.
  if (!prior) {
    const f = await fs.open(file, 'wx', 0o600).catch((e: any) => {
      if (e.code !== 'EEXIST') throw e;
      return null;
    });
    await f?.close();
  }
  const checked = await fs.lstat(file);
  ensure(checked.isFile() && !checked.isSymbolicLink(), 'Operation lock path changed', 403);
  const db = await new Promise<sqlite3.Database>((resolve, reject) => {
    const value = new sqlite3.Database(file, sqlite3.OPEN_READWRITE, e => (e ? reject(e) : resolve(value)));
  });
  const exec = (sql: string) => new Promise<void>((resolve, reject) => db.exec(sql, e => (e ? reject(e) : resolve())));
  let transaction = false;
  try {
    await exec('PRAGMA busy_timeout=100; PRAGMA journal_mode=DELETE; BEGIN IMMEDIATE;');
    transaction = true;
    const value = await action();
    await exec('COMMIT');
    transaction = false;
    return value;
  } catch (e: any) {
    if (e.code === 'SQLITE_BUSY' || e.code === 'SQLITE_LOCKED') throw new OperationBusy();
    throw e;
  } finally {
    if (transaction) await exec('ROLLBACK').catch(() => {});
    await new Promise<void>((resolve, reject) => db.close(e => (e ? reject(e) : resolve())));
  }
}
