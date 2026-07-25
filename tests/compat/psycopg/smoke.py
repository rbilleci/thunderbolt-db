import socket
import subprocess
import sys
import time
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import psycopg
from psycopg import errors
from psycopg.types.numeric import Int2, Int4, Int8
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

            conn.execute(
                """
                CREATE TABLE psycopg_all_types (
                    row_id INT PRIMARY KEY, i2 SMALLINT, i4 INT, i8 BIGINT, amount NUMERIC(12,4),
                    flag BOOL, note TEXT, day DATE, created_at TIMESTAMP, ident UUID
                )
                """
            )
            native_values = (
                Int4(1), Int2(-7), Int4(42), Int8(-9), Decimal("12345.6700"), True, "Grüße",
                date(1999, 12, 31), datetime(2000, 1, 1, 0, 0, 1, 234567),
                UUID("550e8400-e29b-41d4-a716-446655440000"),
            )
            conn.execute(
                "INSERT INTO psycopg_all_types VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                native_values,
                prepare=True,
            )
            conn.execute(
                "INSERT INTO psycopg_all_types VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (Int4(2), None, None, None, None, None, None, None, None, None),
                prepare=True,
            )
            all_types = conn.execute(
                "SELECT i2, i4, i8, amount, flag, note, day, created_at, ident "
                "FROM psycopg_all_types WHERE row_id = %s::int4",
                (Int4(1),),
                prepare=True,
            ).fetchone()
            assert all_types == native_values[1:]
            typed_nulls = conn.execute(
                "SELECT i2, i4, i8, amount, flag, note, day, created_at, ident "
                "FROM psycopg_all_types WHERE row_id = %s::int4",
                (Int4(2),),
                prepare=True,
            ).fetchone()
            assert typed_nulls == (None,) * 9

            conn.execute("CREATE TABLE psycopg_fk_parent (id INT PRIMARY KEY)")
            conn.execute("CREATE TABLE psycopg_fk_child (id INT PRIMARY KEY, parent_id INT)")
            conn.execute(
                "ALTER TABLE ONLY psycopg_fk_child ADD CONSTRAINT psycopg_fk_child_parent_fk "
                "FOREIGN KEY (parent_id) REFERENCES psycopg_fk_parent(id)"
            )
            conn.execute("INSERT INTO psycopg_fk_parent VALUES (1)")
            conn.execute("BEGIN")
            try:
                conn.execute("INSERT INTO psycopg_fk_child VALUES (1, 999)")
                raise AssertionError("foreign-key violation unexpectedly succeeded")
            except errors.ForeignKeyViolation as error:
                assert error.sqlstate == "23503"
            try:
                conn.execute("SELECT 1")
                raise AssertionError("foreign-key failure did not abort the explicit transaction")
            except errors.InFailedSqlTransaction as error:
                assert error.sqlstate == "25P02"
            conn.execute("ROLLBACK")
            conn.execute("INSERT INTO psycopg_fk_child VALUES (1, 1)")

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
