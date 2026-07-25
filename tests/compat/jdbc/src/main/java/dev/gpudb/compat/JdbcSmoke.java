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
import java.sql.Types;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.math.BigDecimal;
import java.util.Properties;
import java.util.UUID;

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
                requireAllTypesAndTypedNulls(conn);
                requireNotNullRollback(conn);
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

    private static void requireAllTypesAndTypedNulls(Connection conn) throws SQLException {
        try (Statement stmt = conn.createStatement()) {
            stmt.execute("""
                CREATE TABLE jdbc_all_types (
                    row_id INT PRIMARY KEY, i2 SMALLINT, i4 INT, i8 BIGINT, amount NUMERIC(12,4),
                    flag BOOL, note TEXT, day DATE, created_at TIMESTAMP, ident UUID
                )
                """);
        }
        LocalDate day = LocalDate.of(1999, 12, 31);
        LocalDateTime createdAt = LocalDateTime.of(2000, 1, 1, 0, 0, 1, 234_567_000);
        UUID ident = UUID.fromString("550e8400-e29b-41d4-a716-446655440000");
        BigDecimal amount = new BigDecimal("12345.6700");
        try (PreparedStatement stmt = conn.prepareStatement(
                "INSERT INTO jdbc_all_types VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")) {
            stmt.setInt(1, 1);
            stmt.setShort(2, (short) -7);
            stmt.setInt(3, 42);
            stmt.setLong(4, -9L);
            stmt.setBigDecimal(5, amount);
            stmt.setBoolean(6, true);
            stmt.setString(7, "Grüße");
            stmt.setObject(8, day);
            stmt.setObject(9, createdAt);
            stmt.setObject(10, ident);
            require(stmt.executeUpdate() == 1, "native all-types insert affected wrong row count");

            stmt.setInt(1, 2);
            stmt.setNull(2, Types.SMALLINT);
            stmt.setNull(3, Types.INTEGER);
            stmt.setNull(4, Types.BIGINT);
            stmt.setNull(5, Types.NUMERIC);
            stmt.setNull(6, Types.BOOLEAN);
            stmt.setNull(7, Types.VARCHAR);
            stmt.setNull(8, Types.DATE);
            stmt.setNull(9, Types.TIMESTAMP);
            stmt.setNull(10, Types.OTHER);
            require(stmt.executeUpdate() == 1, "typed-NULL insert affected wrong row count");
        }
        try (PreparedStatement stmt = conn.prepareStatement(
                "SELECT i2, i4, i8, amount, flag, note, day, created_at, ident "
                        + "FROM jdbc_all_types WHERE row_id = ?")) {
            stmt.setInt(1, 1);
            try (ResultSet rows = stmt.executeQuery()) {
                require(rows.next(), "all-types result returned no row");
                require(rows.getShort("i2") == -7, "all-types int2 mismatch");
                require(rows.getInt("i4") == 42, "all-types int4 mismatch");
                require(rows.getLong("i8") == -9L, "all-types int8 mismatch");
                require(amount.equals(rows.getBigDecimal("amount")), "all-types numeric mismatch");
                require(rows.getBoolean("flag"), "all-types bool mismatch");
                require("Grüße".equals(rows.getString("note")), "all-types text mismatch");
                require(day.equals(rows.getObject("day", LocalDate.class)), "all-types date mismatch");
                require(createdAt.equals(rows.getObject("created_at", LocalDateTime.class)),
                        "all-types timestamp mismatch");
                require(ident.equals(rows.getObject("ident", UUID.class)), "all-types UUID mismatch");
            }
            stmt.setInt(1, 2);
            try (ResultSet rows = stmt.executeQuery()) {
                require(rows.next(), "typed-NULL result returned no row");
                for (String column : new String[] {
                        "i2", "i4", "i8", "amount", "flag", "note", "day", "created_at", "ident"}) {
                    require(rows.getObject(column) == null, "typed-NULL column " + column + " was non-null");
                }
            }
        }
    }

    private static void requireNotNullRollback(Connection conn) throws SQLException {
        try (Statement stmt = conn.createStatement()) {
            stmt.execute("CREATE TABLE jdbc_not_null_contract (id INT PRIMARY KEY)");
        }
        conn.setAutoCommit(false);
        try (PreparedStatement stmt = conn.prepareStatement("INSERT INTO jdbc_not_null_contract VALUES (?)")) {
            stmt.setNull(1, Types.INTEGER);
            try {
                stmt.executeUpdate();
                throw new AssertionError("PRIMARY KEY NULL unexpectedly succeeded");
            } catch (SQLException error) {
                require("23502".equals(error.getSQLState()),
                        "PRIMARY KEY NULL returned " + error.getSQLState() + ", want 23502");
            }
        }
        try (Statement stmt = conn.createStatement()) {
            try {
                stmt.executeQuery("SELECT 1");
                throw new AssertionError("constraint failure did not abort the explicit transaction");
            } catch (SQLException error) {
                require("25P02".equals(error.getSQLState()),
                        "failed transaction returned " + error.getSQLState() + ", want 25P02");
            }
        }
        conn.rollback();
        conn.setAutoCommit(true);
        try (PreparedStatement stmt = conn.prepareStatement("INSERT INTO jdbc_not_null_contract VALUES (?)")) {
            stmt.setInt(1, 1);
            require(stmt.executeUpdate() == 1, "connection was not reusable after ROLLBACK");
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
