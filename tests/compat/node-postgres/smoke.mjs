import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import net from 'node:net';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import pg from 'pg';

const { Client, Pool } = pg;

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, '../../..');

function freeLocalPort() {
  return new Promise((resolvePort, reject) => {
    const server = net.createServer();
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      server.close(() => resolvePort(address.port));
    });
  });
}

function waitForEndpoint(port, timeoutMs = 10_000) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolveReady, reject) => {
    const attempt = () => {
      const socket = net.createConnection({ host: '127.0.0.1', port });
      socket.once('connect', () => {
        socket.destroy();
        resolveReady();
      });
      socket.once('error', (error) => {
        socket.destroy();
        if (Date.now() >= deadline) {
          reject(new Error(`thunderbolt-db-server did not start on 127.0.0.1:${port}: ${error.message}`));
          return;
        }
        setTimeout(attempt, 25);
      });
    };
    attempt();
  });
}

async function startServer() {
  const build = spawnSync(
    'cargo',
    ['build', '-p', 'gpu_db_server', '--bin', 'thunderbolt-db-server'],
    { cwd: repoRoot, stdio: 'inherit' },
  );
  assert.equal(build.status, 0, 'cargo build for thunderbolt-db-server failed');

  const port = await freeLocalPort();
  const bin = resolve(repoRoot, 'target/debug/thunderbolt-db-server');
  const child = spawn(bin, ['--listen', `127.0.0.1:${port}`], {
    cwd: repoRoot,
    stdio: ['ignore', 'ignore', 'pipe'],
  });

  let stderr = '';
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', (chunk) => {
    stderr += chunk;
  });

  try {
    await waitForEndpoint(port);
  } catch (error) {
    child.kill();
    throw new Error(`${error.message}\nserver stderr:\n${stderr}`);
  }

  return {
    port,
    async stop() {
      if (child.exitCode === null) {
        child.kill();
        await new Promise((resolveStopped) => child.once('exit', resolveStopped));
      }
    },
  };
}

function connectionConfig(port, applicationName) {
  return {
    host: '127.0.0.1',
    port,
    user: 'postgres',
    database: 'postgres',
    application_name: applicationName,
    ssl: false,
  };
}

function localIsoTimestamp(value) {
  const pad = (part, width = 2) => String(part).padStart(width, '0');
  return `${value.getFullYear()}-${pad(value.getMonth() + 1)}-${pad(value.getDate())}`
    + `T${pad(value.getHours())}:${pad(value.getMinutes())}:${pad(value.getSeconds())}.${pad(value.getMilliseconds(), 3)}Z`;
}

async function main() {
  const server = await startServer();
  try {
    const client = new Client(connectionConfig(server.port, 'node_postgres_smoke'));
    await client.connect();
    assert.ok(Number.isInteger(client.processID) && client.processID > 0);
    assert.ok(Number.isInteger(client.secretKey));

    const simple = await client.query('SELECT 1 AS one');
    assert.equal(simple.rows.length, 1);
    assert.equal(simple.rows[0].one, 1);

    await client.query(`
      CREATE TABLE node_people (id INT, name TEXT);
      INSERT INTO node_people (id, name)
      VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
    `);

    const prepared = await client.query({
      name: 'node_people_by_id',
      text: 'SELECT id, name FROM node_people WHERE id = $1',
      values: [2],
    });
    assert.deepEqual(prepared.rows, [{ id: 2, name: 'Linus' }]);

    const empty = await client.query({
      name: 'node_people_by_id',
      text: 'SELECT id, name FROM node_people WHERE id = $1',
      values: [99],
    });
    assert.deepEqual(empty.rows, []);

    await client.query(`
      CREATE TABLE node_all_types (
        row_id INT PRIMARY KEY, i2 SMALLINT, i4 INT, i8 BIGINT, amount NUMERIC(12,4),
        flag BOOL, note TEXT, day DATE, created_at TIMESTAMP, ident UUID
      )
    `);
    const day = '1999-12-31';
    // node-postgres has no date-only/timestamp-without-time-zone native value; its normal
    // PostgreSQL representation for those columns is their ISO text form.
    const createdAt = '2000-01-01 00:00:01.234567';
    const ident = '550e8400-e29b-41d4-a716-446655440000';
    await client.query({
      name: 'node_all_types_insert',
      text: 'INSERT INTO node_all_types VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)',
      values: [1, -7, 42, '-9', '12345.6700', true, 'Grüße', day, createdAt, ident],
    });
    await client.query({
      name: 'node_all_types_insert',
      text: 'INSERT INTO node_all_types VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)',
      values: [2, null, null, null, null, null, null, null, null, null],
    });
    const allTypes = await client.query(
      'SELECT i2, i4, i8, amount, flag, note, day, created_at, ident FROM node_all_types WHERE row_id = $1',
      [1],
    );
    assert.deepEqual(
      {
        i2: allTypes.rows[0].i2,
        i4: allTypes.rows[0].i4,
        i8: allTypes.rows[0].i8,
        amount: allTypes.rows[0].amount,
        flag: allTypes.rows[0].flag,
        note: allTypes.rows[0].note,
        day: localIsoTimestamp(allTypes.rows[0].day).slice(0, 10),
        createdAt: localIsoTimestamp(allTypes.rows[0].created_at),
        ident: allTypes.rows[0].ident,
      },
      {
        i2: -7,
        i4: 42,
        i8: '-9',
        amount: '12345.6700',
        flag: true,
        note: 'Grüße',
        day,
        createdAt: '2000-01-01T00:00:01.234Z',
        ident,
      },
    );
    const typedNulls = await client.query(
      'SELECT i2, i4, i8, amount, flag, note, day, created_at, ident FROM node_all_types WHERE row_id = $1',
      [2],
    );
    assert.ok(Object.values(typedNulls.rows[0]).every((value) => value === null));

    await client.query(`
      CREATE TABLE node_check_contract (
        id INT PRIMARY KEY, amount NUMERIC(4,2),
        CONSTRAINT node_check_positive CHECK (amount > 0.00)
      )
    `);
    await client.query('BEGIN');
    await assert.rejects(
      () => client.query('INSERT INTO node_check_contract VALUES ($1, $2)', [1, '-1.00']),
      (error) => error.code === '23514',
      'CHECK violation must preserve SQLSTATE 23514',
    );
    await assert.rejects(
      () => client.query('SELECT 1'),
      (error) => error.code === '25P02',
      'CHECK failure must abort the explicit transaction',
    );
    await client.query('ROLLBACK');
    await client.query('INSERT INTO node_check_contract VALUES ($1, $2)', [1, '1.00']);

    await assert.rejects(
      () => client.query("COPY node_people FROM STDIN WITH CSV HEADER DELIMITER ','"),
      (error) => error.code === '0A000',
      'broader COPY CSV options should stay explicitly unsupported',
    );

    const recovered = await client.query('SELECT name FROM node_people WHERE id = 3');
    assert.deepEqual(recovered.rows, [{ name: 'Grace' }]);

    await client.end();

    const pool = new Pool({
      ...connectionConfig(server.port, 'node_postgres_pool_smoke'),
      max: 1,
    });
    const pooled = await pool.query('SELECT 1 AS one');
    assert.equal(pooled.rows[0].one, 1);
    await pool.end();

    const reconnected = new Client(connectionConfig(server.port, 'node_postgres_reconnect_smoke'));
    await reconnected.connect();
    const reconnectSimple = await reconnected.query('SELECT 1 AS one');
    assert.equal(reconnectSimple.rows[0].one, 1);
    await reconnected.end();
  } finally {
    await server.stop();
  }
}

await main();
