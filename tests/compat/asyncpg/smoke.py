import asyncio
import socket
import subprocess
import sys
import time
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import asyncpg


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_endpoint(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.025)
    raise RuntimeError(f"gpu-db-engine-server did not start on 127.0.0.1:{port}")


class Server:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.port = free_local_port()
        self.process = None

    def start(self) -> None:
        subprocess.run(
            ["cargo", "build", "-p", "gpu_db_server", "--bin", "gpu-db-engine-server"],
            cwd=self.repo_root,
            check=True,
        )
        binary = self.repo_root / "target" / "debug" / "gpu-db-engine-server"
        self.process = subprocess.Popen(
            [str(binary), "--listen", f"127.0.0.1:{self.port}"],
            cwd=self.repo_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            wait_for_endpoint(self.port)
        except Exception:
            stderr = self.stop()
            raise RuntimeError(f"server failed to start\n{stderr}")

    def stop(self) -> str:
        if self.process is None:
            return ""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        return self.process.stderr.read() if self.process.stderr is not None else ""


async def connect(port: int, application_name: str):
    return await asyncpg.connect(
        host="127.0.0.1",
        port=port,
        user="postgres",
        database="postgres",
        ssl=False,
        server_settings={"application_name": application_name},
    )


async def main(repo_root: Path) -> None:
    server = Server(repo_root)
    server.start()
    try:
        conn = await connect(server.port, "asyncpg_smoke")

        setup_status = await conn.execute(
            """
            CREATE TABLE asyncpg_people (id INT, name TEXT);
            INSERT INTO asyncpg_people (id, name)
            VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
            """
        )
        assert setup_status == "INSERT 0 3"

        statement = await conn.prepare(
            "SELECT id, name FROM asyncpg_people WHERE id = $1"
        )
        rows = await statement.fetch(2)
        assert len(rows) == 1
        assert dict(rows[0]) == {"id": 2, "name": "Linus"}

        empty_rows = await statement.fetch(99)
        assert empty_rows == []

        await conn.execute(
            """
            CREATE TABLE asyncpg_all_types (
                row_id INT PRIMARY KEY, i2 SMALLINT, i4 INT, i8 BIGINT, amount NUMERIC(12,4),
                flag BOOL, note TEXT, day DATE, created_at TIMESTAMP, ident UUID
            )
            """
        )
        native_values = (
            1, -7, 42, -9, Decimal("12345.6700"), True, "Grüße",
            date(1999, 12, 31), datetime(2000, 1, 1, 0, 0, 1, 234567),
            UUID("550e8400-e29b-41d4-a716-446655440000"),
        )
        await conn.execute(
            "INSERT INTO asyncpg_all_types VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
            *native_values,
        )
        await conn.execute(
            "INSERT INTO asyncpg_all_types VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
            2, None, None, None, None, None, None, None, None, None,
        )
        all_types = await conn.fetchrow(
            "SELECT i2, i4, i8, amount, flag, note, day, created_at, ident "
            "FROM asyncpg_all_types WHERE row_id = 1"
        )
        assert tuple(all_types.values()) == native_values[1:]
        typed_nulls = await conn.fetchrow(
            "SELECT i2, i4, i8, amount, flag, note, day, created_at, ident "
            "FROM asyncpg_all_types WHERE row_id = 2"
        )
        assert tuple(typed_nulls.values()) == (None,) * 9

        await conn.execute("CREATE TABLE asyncpg_not_null_contract (id INT PRIMARY KEY)")
        await conn.execute("BEGIN")
        try:
            await conn.execute("INSERT INTO asyncpg_not_null_contract VALUES ($1)", None)
            raise AssertionError("PRIMARY KEY NULL unexpectedly succeeded")
        except asyncpg.PostgresError as error:
            assert error.sqlstate == "23502"
        try:
            await conn.fetchval("SELECT 1")
            raise AssertionError("constraint failure did not abort the explicit transaction")
        except asyncpg.PostgresError as error:
            assert error.sqlstate == "25P02"
        await conn.execute("ROLLBACK")
        await conn.execute("INSERT INTO asyncpg_not_null_contract VALUES ($1)", 1)

        try:
            await conn.execute("COPY asyncpg_people FROM STDIN WITH CSV HEADER DELIMITER ','")
            raise AssertionError("broader COPY CSV options unexpectedly succeeded")
        except asyncpg.PostgresError as error:
            assert error.sqlstate == "0A000"

        recovered = await conn.fetchrow(
            "SELECT name FROM asyncpg_people WHERE id = 3"
        )
        assert recovered["name"] == "Grace"

        await conn.close()

        pool = await asyncpg.create_pool(
            host="127.0.0.1",
            port=server.port,
            user="postgres",
            database="postgres",
            ssl=False,
            min_size=1,
            max_size=1,
            server_settings={"application_name": "asyncpg_pool_smoke"},
        )
        try:
            async with pool.acquire() as pooled_conn:
                pooled_setup = await pooled_conn.execute(
                    """
                    CREATE TABLE asyncpg_pool_check (id INT, name TEXT);
                    INSERT INTO asyncpg_pool_check (id, name) VALUES (1, 'pooled');
                    """
                )
                assert pooled_setup == "INSERT 0 1"
                pooled = await pooled_conn.fetchrow(
                    "SELECT id, name FROM asyncpg_pool_check WHERE id = $1", 1
                )
                assert dict(pooled) == {"id": 1, "name": "pooled"}

            reused = await pool.fetchrow(
                "SELECT name FROM asyncpg_pool_check WHERE id = $1", 1
            )
            assert reused["name"] == "pooled"
        finally:
            await pool.close()

        reconnected = await connect(server.port, "asyncpg_reconnect_smoke")
        try:
            reconnect_setup = await reconnected.execute(
                """
                CREATE TABLE asyncpg_reconnect_check (id INT, name TEXT);
                INSERT INTO asyncpg_reconnect_check (id, name) VALUES (1, 'reconnected');
                """
            )
            assert reconnect_setup == "INSERT 0 1"
            reconnect_simple = await reconnected.fetchrow(
                "SELECT id, name FROM asyncpg_reconnect_check WHERE id = $1", 1
            )
            assert dict(reconnect_simple) == {"id": 1, "name": "reconnected"}
        finally:
            await reconnected.close()
    finally:
        server.stop()


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))
