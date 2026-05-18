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
          reject(new Error(`gpu-db-server did not start on 127.0.0.1:${port}: ${error.message}`));
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
    ['build', '-p', 'gpu_db_protocol', '--bin', 'gpu-db-server'],
    { cwd: repoRoot, stdio: 'inherit' },
  );
  assert.equal(build.status, 0, 'cargo build for gpu-db-server failed');

  const port = await freeLocalPort();
  const bin = resolve(repoRoot, 'target/debug/gpu-db-server');
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

async function main() {
  const server = await startServer();
  try {
    const client = new Client(connectionConfig(server.port, 'node_postgres_smoke'));
    await client.connect();

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

    await assert.rejects(
      () => client.query('COPY node_people FROM STDIN WITH CSV HEADER'),
      (error) => error.code === '0A000',
      'COPY FROM STDIN WITH CSV HEADER should stay explicitly unsupported',
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
