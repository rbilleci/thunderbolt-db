import asyncio
import socket
import subprocess
import sys
import time
from pathlib import Path

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
    raise RuntimeError(f"gpu-db-server did not start on 127.0.0.1:{port}")


class Server:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.port = free_local_port()
        self.process = None

    def start(self) -> None:
        subprocess.run(
            ["cargo", "build", "-p", "gpu_db_protocol", "--bin", "gpu-db-server"],
            cwd=self.repo_root,
            check=True,
        )
        binary = self.repo_root / "target" / "debug" / "gpu-db-server"
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

        try:
            await conn.execute("COPY asyncpg_people FROM STDIN WITH CSV")
            raise AssertionError("COPY FROM STDIN WITH CSV unexpectedly succeeded")
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
