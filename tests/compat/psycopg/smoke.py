import socket
import subprocess
import sys
import time
from pathlib import Path

import psycopg
from psycopg import errors
from psycopg.types.numeric import Int4
from psycopg_pool import ConnectionPool


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


def conninfo(port: int, application_name: str) -> str:
    return (
        f"host=127.0.0.1 port={port} user=postgres dbname=postgres "
        f"sslmode=disable application_name={application_name}"
    )


def connect(port: int, application_name: str):
    return psycopg.connect(conninfo(port, application_name), autocommit=True)


def main(repo_root: Path) -> None:
    server = Server(repo_root)
    server.start()
    try:
        with connect(server.port, "psycopg_smoke") as conn:
            simple = conn.execute("SELECT 1 AS one").fetchone()
            assert simple == (1,)

            conn.execute(
                """
                CREATE TABLE psycopg_people (id INT, name TEXT);
                INSERT INTO psycopg_people (id, name)
                VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
                """
            )

            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, name FROM psycopg_people WHERE id = %s::int4",
                    (Int4(2),),
                    prepare=True,
                )
                assert cur.fetchall() == [(2, "Linus")]

                cur.execute(
                    "SELECT id, name FROM psycopg_people WHERE id = %s::int4",
                    (Int4(99),),
                    prepare=True,
                )
                assert cur.fetchall() == []

            try:
                conn.execute(
                    "COPY psycopg_people FROM STDIN WITH CSV HEADER DELIMITER ','"
                )
                raise AssertionError("broader COPY CSV options unexpectedly succeeded")
            except errors.FeatureNotSupported as error:
                assert error.sqlstate == "0A000"

            recovered = conn.execute(
                "SELECT name FROM psycopg_people WHERE id = 3"
            ).fetchone()
            assert recovered == ("Grace",)

        with ConnectionPool(
            conninfo(server.port, "psycopg_pool_smoke"),
            min_size=1,
            max_size=1,
            kwargs={"autocommit": True},
        ) as pool:
            with pool.connection() as pooled_conn:
                pooled_conn.execute(
                    """
                    CREATE TABLE psycopg_pool_check (id INT, name TEXT);
                    INSERT INTO psycopg_pool_check (id, name) VALUES (1, 'pooled');
                    """
                )
                pooled = pooled_conn.execute(
                    "SELECT id, name FROM psycopg_pool_check WHERE id = %s::int4",
                    (Int4(1),),
                    prepare=True,
                ).fetchone()
                assert pooled == (1, "pooled")

            with pool.connection() as reused_conn:
                reused = reused_conn.execute(
                    "SELECT name FROM psycopg_pool_check WHERE id = %s::int4",
                    (Int4(1),),
                    prepare=True,
                ).fetchone()
                assert reused == ("pooled",)

        with connect(server.port, "psycopg_reconnect_smoke") as reconnected:
            reconnected.execute(
                """
                CREATE TABLE psycopg_reconnect_check (id INT, name TEXT);
                INSERT INTO psycopg_reconnect_check (id, name) VALUES (1, 'reconnected');
                """
            )
            reconnect_simple = reconnected.execute(
                "SELECT id, name FROM psycopg_reconnect_check WHERE id = %s::int4",
                (Int4(1),),
                prepare=True,
            ).fetchone()
            assert reconnect_simple == (1, "reconnected")
    finally:
        server.stop()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
