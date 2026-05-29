package dev.gpudb.compat;

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
import java.time.Duration;
import java.time.Instant;
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
        await(Flux.from(conn.createStatement(sql).execute())
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

    private record Server(Process process, int port) implements AutoCloseable {
        static Server start(Path repoRoot) throws IOException, InterruptedException {
            Process build = new ProcessBuilder(
                    "cargo", "build", "-p", "gpu_db_protocol", "--bin", "gpu-db-server")
                    .directory(repoRoot.toFile())
                    .inheritIO()
                    .start();
            int buildExit = build.waitFor();
            if (buildExit != 0) {
                throw new IllegalStateException("cargo build for gpu-db-server failed with exit " + buildExit);
            }

            int port = freeLocalPort();
            Process server = new ProcessBuilder(
                    repoRoot.resolve("target/debug/gpu-db-server").toString(),
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
                        throw new IllegalStateException("interrupted while waiting for gpu-db-server", interrupted);
                    }
                }
            }
            throw new IllegalStateException("gpu-db-server did not start on 127.0.0.1:" + port);
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
