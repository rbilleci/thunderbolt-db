package dev.thunderboltdb.compat;

import io.r2dbc.postgresql.PostgresqlConnectionConfiguration;
import io.r2dbc.postgresql.PostgresqlConnectionFactory;
import io.r2dbc.postgresql.client.SSLMode;
import io.r2dbc.spi.Connection;
import io.r2dbc.spi.ConnectionFactory;
import io.r2dbc.spi.R2dbcException;
import io.r2dbc.spi.Result;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.file.Path;
import java.math.BigDecimal;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.UUID;
import org.reactivestreams.Publisher;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

public final class R2dbcSmoke {
    private static final Duration AWAIT_TIMEOUT = Duration.ofSeconds(10);

    private R2dbcSmoke() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("usage: R2dbcSmoke <repo-root>");
        }
        Path repoRoot = Path.of(args[0]).toAbsolutePath().normalize();
        try (Server server = Server.start(repoRoot)) {
            ConnectionFactory factory = connectionFactory(server.port(), "r2dbc_smoke");
            Connection conn = await(factory.create());
            try {
                requireSimpleQuery(conn);
                setupTable(conn);
                requirePreparedQuery(conn);
                requireEmptyResult(conn);
                requireAllTypesAndTypedNulls(conn);
                requireCheckRollback(conn);
                requireUnsupportedCopyRecovery(conn);
            } finally {
                await(conn.close());
            }

            Connection reusedFactoryConn = await(factory.create());
            try {
                requireSimpleQuery(reusedFactoryConn);
            } finally {
                await(reusedFactoryConn.close());
            }

            Connection reconnect = await(connectionFactory(server.port(), "r2dbc_reconnect_smoke").create());
            try {
                requireSimpleQuery(reconnect);
            } finally {
                await(reconnect.close());
            }
        }
    }

    private static ConnectionFactory connectionFactory(int port, String applicationName) {
        PostgresqlConnectionConfiguration config = PostgresqlConnectionConfiguration.builder()
                .host("127.0.0.1")
                .port(port)
                .username("postgres")
                .database("postgres")
                .applicationName(applicationName)
                .sslMode(SSLMode.DISABLE)
                .build();
        return new PostgresqlConnectionFactory(config);
    }

    private static void requireSimpleQuery(Connection conn) {
        Integer one = scalar(conn, "SELECT 1 AS one", "one", Integer.class);
        require(one != null && one == 1, "simple query returned wrong value: " + one);
    }

    private static void setupTable(Connection conn) {
        execute(conn, "CREATE TABLE r2dbc_people (id INT, name TEXT)");
        execute(conn, """
                INSERT INTO r2dbc_people (id, name)
                VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace')
                """);
    }

    private static void requirePreparedQuery(Connection conn) {
        Person person = Flux.from(conn.createStatement("SELECT id, name FROM r2dbc_people WHERE id = $1")
                        .bind("$1", 2)
                        .execute())
                .flatMap(result -> result.map((row, metadata) ->
                        new Person(row.get("id", Integer.class), row.get("name", String.class))))
                .single()
                .block(AWAIT_TIMEOUT);
        require(person != null && person.id() == 2, "prepared query returned wrong id");
        require(person != null && "Linus".equals(person.name()), "prepared query returned wrong text");
    }

    private static void requireEmptyResult(Connection conn) {
        long count = Flux.from(conn.createStatement("SELECT id, name FROM r2dbc_people WHERE id = $1")
                        .bind("$1", 99)
                        .execute())
                .flatMap(result -> result.map((row, metadata) -> row.get("id", Integer.class)))
                .count()
                .block(AWAIT_TIMEOUT);
        require(count == 0, "empty result query unexpectedly returned " + count + " rows");
    }

    private static void requireAllTypesAndTypedNulls(Connection conn) {
        execute(conn, """
                CREATE TABLE r2dbc_all_types (
                    row_id INT PRIMARY KEY, i2 SMALLINT, i4 INT, i8 BIGINT, amount NUMERIC(12,4),
                    flag BOOL, note TEXT, day DATE, created_at TIMESTAMP, ident UUID
                )
                """);
        LocalDate day = LocalDate.of(1999, 12, 31);
        LocalDateTime createdAt = LocalDateTime.of(2000, 1, 1, 0, 0, 1, 234_567_000);
        UUID ident = UUID.fromString("550e8400-e29b-41d4-a716-446655440000");
        BigDecimal amount = new BigDecimal("12345.6700");
        execute(conn.createStatement(
                "INSERT INTO r2dbc_all_types VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)")
                .bind("$1", 1)
                .bind("$2", (short) -7)
                .bind("$3", 42)
                .bind("$4", -9L)
                .bind("$5", amount)
                .bind("$6", true)
                .bind("$7", "Grüße")
                .bind("$8", day)
                .bind("$9", createdAt)
                .bind("$10", ident));
        execute(conn.createStatement(
                "INSERT INTO r2dbc_all_types VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)")
                .bind("$1", 2)
                .bindNull("$2", Short.class)
                .bindNull("$3", Integer.class)
                .bindNull("$4", Long.class)
                .bindNull("$5", BigDecimal.class)
                .bindNull("$6", Boolean.class)
                .bindNull("$7", String.class)
                .bindNull("$8", LocalDate.class)
                .bindNull("$9", LocalDateTime.class)
                .bindNull("$10", UUID.class));
        AllTypes allTypes = Flux.from(conn.createStatement(
                        "SELECT i2, i4, i8, amount, flag, note, day, created_at, ident "
                                + "FROM r2dbc_all_types WHERE row_id = $1")
                        .bind("$1", 1)
                        .execute())
                .flatMap(result -> result.map((row, metadata) -> new AllTypes(
                        row.get("i2", Short.class), row.get("i4", Integer.class), row.get("i8", Long.class),
                        row.get("amount", BigDecimal.class), row.get("flag", Boolean.class),
                        row.get("note", String.class), row.get("day", LocalDate.class),
                        row.get("created_at", LocalDateTime.class), row.get("ident", UUID.class))))
                .single()
                .block(AWAIT_TIMEOUT);
        require(new AllTypes((short) -7, 42, -9L, amount, true, "Grüße", day, createdAt, ident)
                        .equals(allTypes),
                "native all-types result mismatch: " + allTypes);
        Object[] typedNulls = Flux.from(conn.createStatement(
                        "SELECT i2, i4, i8, amount, flag, note, day, created_at, ident "
                                + "FROM r2dbc_all_types WHERE row_id = $1")
                        .bind("$1", 2)
                        .execute())
                .flatMap(result -> result.map((row, metadata) -> new Object[] {
                        row.get("i2"), row.get("i4"), row.get("i8"), row.get("amount"), row.get("flag"),
                        row.get("note"), row.get("day"), row.get("created_at"), row.get("ident")}))
                .single()
                .block(AWAIT_TIMEOUT);
        for (Object value : typedNulls) {
            require(value == null, "typed-NULL result decoded as a non-null value");
        }
    }

    private static void requireCheckRollback(Connection conn) {
        execute(conn, """
                CREATE TABLE r2dbc_check_contract (
                    id INT PRIMARY KEY, amount NUMERIC(4,2),
                    CONSTRAINT r2dbc_check_positive CHECK (amount > 0.00)
                )
                """);
        execute(conn, "BEGIN");
        try {
            execute(conn.createStatement("INSERT INTO r2dbc_check_contract VALUES ($1, $2)")
                    .bind("$1", 1)
                    .bind("$2", new BigDecimal("-1.00")));
            throw new AssertionError("CHECK violation unexpectedly succeeded");
        } catch (R2dbcException error) {
            require("23514".equals(error.getSqlState()),
                    "CHECK violation returned " + error.getSqlState() + ", want 23514");
        }
        try {
            scalar(conn, "SELECT 1 AS one", "one", Integer.class);
            throw new AssertionError("CHECK failure did not abort the explicit transaction");
        } catch (R2dbcException error) {
            require("25P02".equals(error.getSqlState()),
                    "failed transaction returned " + error.getSqlState() + ", want 25P02");
        }
        execute(conn, "ROLLBACK");
        execute(conn.createStatement("INSERT INTO r2dbc_check_contract VALUES ($1, $2)")
                .bind("$1", 1)
                .bind("$2", new BigDecimal("1.00")));
    }

    private static void requireUnsupportedCopyRecovery(Connection conn) {
        try {
            execute(conn, "COPY r2dbc_people FROM STDIN WITH CSV HEADER DELIMITER ','");
            throw new AssertionError("broader COPY CSV options unexpectedly succeeded");
        } catch (R2dbcException error) {
            require("0A000".equals(error.getSqlState()),
                    "unsupported COPY returned SQLSTATE " + error.getSqlState() + ", want 0A000");
        }

        String recovered = scalar(conn, "SELECT name FROM r2dbc_people WHERE id = 3", "name", String.class);
        require("Grace".equals(recovered), "recovery query returned wrong text: " + recovered);
    }

    private static void execute(Connection conn, String sql) {
        execute(conn.createStatement(sql));
    }

    private static void execute(io.r2dbc.spi.Statement statement) {
        await(Flux.from(statement.execute())
                .flatMap(Result::getRowsUpdated)
                .then());
    }

    private static <T> T scalar(Connection conn, String sql, String column, Class<T> type) {
        return Flux.from(conn.createStatement(sql).execute())
                .flatMap(result -> result.map((row, metadata) -> row.get(column, type)))
                .single()
                .block(AWAIT_TIMEOUT);
    }

    private static <T> T await(Publisher<T> publisher) {
        return Mono.from(publisher).block(AWAIT_TIMEOUT);
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    private record Person(Integer id, String name) {
    }

    private record AllTypes(Short i2, Integer i4, Long i8, BigDecimal amount, Boolean flag, String note,
                            LocalDate day, LocalDateTime createdAt, UUID ident) {
    }

    private record Server(Process process, int port) implements AutoCloseable {
        static Server start(Path repoRoot) throws IOException, InterruptedException {
            Process build = new ProcessBuilder(
                    "cargo", "build", "-p", "gpu_db_server", "--bin", "thunderbolt-db-server")
                    .directory(repoRoot.toFile())
                    .inheritIO()
                    .start();
            int buildExit = build.waitFor();
            if (buildExit != 0) {
                throw new IllegalStateException("cargo build for thunderbolt-db-server failed with exit " + buildExit);
            }

            int port = freeLocalPort();
            Process server = new ProcessBuilder(
                    repoRoot.resolve("target/debug/thunderbolt-db-server").toString(),
                    "--listen",
                    "127.0.0.1:" + port)
                    .directory(repoRoot.toFile())
                    .redirectOutput(ProcessBuilder.Redirect.DISCARD)
                    .redirectError(ProcessBuilder.Redirect.INHERIT)
                    .start();

            try {
                waitForEndpoint(port);
            } catch (RuntimeException error) {
                server.destroyForcibly();
                server.waitFor();
                throw error;
            }

            return new Server(server, port);
        }

        private static int freeLocalPort() throws IOException {
            try (ServerSocket socket = new ServerSocket()) {
                socket.bind(new InetSocketAddress("127.0.0.1", 0));
                return socket.getLocalPort();
            }
        }

        private static void waitForEndpoint(int port) {
            Instant deadline = Instant.now().plus(Duration.ofSeconds(10));
            while (Instant.now().isBefore(deadline)) {
                try (Socket socket = new Socket()) {
                    socket.connect(new InetSocketAddress("127.0.0.1", port), 250);
                    return;
                } catch (IOException error) {
                    try {
                        Thread.sleep(25);
                    } catch (InterruptedException interrupted) {
                        Thread.currentThread().interrupt();
                        throw new IllegalStateException(
                                "interrupted while waiting for thunderbolt-db-server", interrupted);
                    }
                }
            }
            throw new IllegalStateException("thunderbolt-db-server did not start on 127.0.0.1:" + port);
        }

        @Override
        public void close() throws InterruptedException {
            if (process.isAlive()) {
                process.destroyForcibly();
                process.waitFor();
            }
        }
    }
}
