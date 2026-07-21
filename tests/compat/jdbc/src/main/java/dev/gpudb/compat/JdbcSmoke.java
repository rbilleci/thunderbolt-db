package dev.gpudb.compat;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.time.Duration;
import java.time.Instant;
import java.util.Properties;

public final class JdbcSmoke {
    private JdbcSmoke() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("usage: JdbcSmoke <repo-root>");
        }
        Path repoRoot = Path.of(args[0]).toAbsolutePath().normalize();
        try (Server server = Server.start(repoRoot)) {
            try (Connection conn = connect(server.port(), "jdbc_smoke")) {
                requireSimpleQuery(conn);
                setupTable(conn);
                requirePreparedQuery(conn);
                requireEmptyResult(conn);
                requireUnsupportedCopyRecovery(conn);
            }

            try (Connection conn = connect(server.port(), "jdbc_reconnect_smoke")) {
                requireSimpleQuery(conn);
            }
        }
    }

    private static Connection connect(int port, String applicationName) throws SQLException {
        String url = "jdbc:postgresql://127.0.0.1:" + port + "/postgres";
        Properties props = new Properties();
        props.setProperty("user", "postgres");
        props.setProperty("sslmode", "disable");
        props.setProperty("ApplicationName", applicationName);
        props.setProperty("assumeMinServerVersion", "12");
        props.setProperty("preferQueryMode", "extendedForPrepared");
        return DriverManager.getConnection(url, props);
    }

    private static void requireSimpleQuery(Connection conn) throws SQLException {
        try (Statement stmt = conn.createStatement();
             ResultSet rows = stmt.executeQuery("SELECT 1 AS one")) {
            require(rows.next(), "simple query returned no row");
            require(rows.getInt("one") == 1, "simple query returned wrong value");
            require(!rows.next(), "simple query returned extra rows");
        }
    }

    private static void setupTable(Connection conn) throws SQLException {
        try (Statement stmt = conn.createStatement()) {
            stmt.execute("""
                CREATE TABLE jdbc_people (id INT, name TEXT);
                INSERT INTO jdbc_people (id, name)
                VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
                """);
        }
    }

    private static void requirePreparedQuery(Connection conn) throws SQLException {
        try (PreparedStatement stmt = conn.prepareStatement(
                "SELECT id, name FROM jdbc_people WHERE id = ?")) {
            stmt.setInt(1, 2);
            try (ResultSet rows = stmt.executeQuery()) {
                require(rows.next(), "prepared query returned no row");
                require(rows.getInt("id") == 2, "prepared query returned wrong id");
                require("Linus".equals(rows.getString("name")), "prepared query returned wrong text");
                require(!rows.next(), "prepared query returned extra rows");
            }
        }
    }

    private static void requireEmptyResult(Connection conn) throws SQLException {
        try (PreparedStatement stmt = conn.prepareStatement(
                "SELECT id, name FROM jdbc_people WHERE id = ?")) {
            stmt.setInt(1, 99);
            try (ResultSet rows = stmt.executeQuery()) {
                require(!rows.next(), "empty result query unexpectedly returned a row");
            }
        }
    }

    private static void requireUnsupportedCopyRecovery(Connection conn) throws SQLException {
        try (Statement stmt = conn.createStatement()) {
            stmt.execute("COPY jdbc_people FROM STDIN WITH CSV HEADER DELIMITER ','");
            throw new AssertionError("broader COPY CSV options unexpectedly succeeded");
        } catch (SQLException error) {
            require("0A000".equals(error.getSQLState()),
                    "unsupported COPY returned SQLSTATE " + error.getSQLState() + ", want 0A000");
        }

        try (Statement stmt = conn.createStatement();
             ResultSet rows = stmt.executeQuery("SELECT name FROM jdbc_people WHERE id = 3")) {
            require(rows.next(), "recovery query returned no row");
            require("Grace".equals(rows.getString("name")), "recovery query returned wrong text");
            require(!rows.next(), "recovery query returned extra rows");
        }
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    private record Server(Process process, int port) implements AutoCloseable {
        static Server start(Path repoRoot) throws IOException, InterruptedException {
            Process build = new ProcessBuilder(
                    "cargo", "build", "-p", "gpu_db_server", "--bin", "gpu-db-engine-server")
                    .directory(repoRoot.toFile())
                    .inheritIO()
                    .start();
            int buildExit = build.waitFor();
            if (buildExit != 0) {
                throw new IllegalStateException("cargo build for gpu-db-engine-server failed with exit " + buildExit);
            }

            int port = freeLocalPort();
            Process server = new ProcessBuilder(
                    repoRoot.resolve("target/debug/gpu-db-engine-server").toString(),
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
                                "interrupted while waiting for gpu-db-engine-server", interrupted);
                    }
                }
            }
            throw new IllegalStateException("gpu-db-engine-server did not start on 127.0.0.1:" + port);
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
