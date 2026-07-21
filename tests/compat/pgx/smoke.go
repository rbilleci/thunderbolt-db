package main

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

type server struct {
	port int
	cmd  *exec.Cmd
}

func repoRoot() (string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	return filepath.Clean(filepath.Join(cwd, "../../..")), nil
}

func freeLocalPort() (int, error) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer listener.Close()
	return listener.Addr().(*net.TCPAddr).Port, nil
}

func waitForEndpoint(port int, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", port), 250*time.Millisecond)
		if err == nil {
			_ = conn.Close()
			return nil
		}
		time.Sleep(25 * time.Millisecond)
	}
	return fmt.Errorf("gpu-db-engine-server did not start on 127.0.0.1:%d", port)
}

func startServer(ctx context.Context, root string) (*server, error) {
	build := exec.CommandContext(ctx, "cargo", "build", "-p", "gpu_db_server", "--bin", "gpu-db-engine-server")
	build.Dir = root
	build.Stdout = os.Stdout
	build.Stderr = os.Stderr
	if err := build.Run(); err != nil {
		return nil, fmt.Errorf("cargo build for gpu-db-engine-server failed: %w", err)
	}

	port, err := freeLocalPort()
	if err != nil {
		return nil, err
	}

	binary := filepath.Join(root, "target", "debug", "gpu-db-engine-server")
	cmd := exec.CommandContext(ctx, binary, "--listen", fmt.Sprintf("127.0.0.1:%d", port))
	cmd.Dir = root
	cmd.Stdin = nil
	cmd.Stdout = nil
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, err
	}

	if err := waitForEndpoint(port, 10*time.Second); err != nil {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return nil, err
	}

	return &server{port: port, cmd: cmd}, nil
}

func (s *server) stop() {
	if s == nil || s.cmd == nil || s.cmd.Process == nil {
		return
	}
	_ = s.cmd.Process.Kill()
	_ = s.cmd.Wait()
}

func connectConfig(port int, applicationName string) string {
	return fmt.Sprintf("postgres://postgres@127.0.0.1:%d/postgres?sslmode=disable&application_name=%s", port, applicationName)
}

func requireEqual[T comparable](got, want T, label string) {
	if got != want {
		panic(fmt.Sprintf("%s: got %v, want %v", label, got, want))
	}
}

func main() {
	ctx := context.Background()

	root, err := repoRoot()
	if err != nil {
		panic(err)
	}

	srv, err := startServer(ctx, root)
	if err != nil {
		panic(err)
	}
	defer srv.stop()

	conn, err := pgx.Connect(ctx, connectConfig(srv.port, "pgx_smoke"))
	if err != nil {
		panic(err)
	}

	var one int32
	if err := conn.QueryRow(ctx, "SELECT 1 AS one", pgx.QueryExecModeSimpleProtocol).Scan(&one); err != nil {
		panic(err)
	}
	requireEqual(one, int32(1), "simple query")

	_, err = conn.Exec(ctx, `
		CREATE TABLE pgx_people (id INT, name TEXT);
		INSERT INTO pgx_people (id, name)
		VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
	`)
	if err != nil {
		panic(err)
	}

	var id int32
	var name string
	if err := conn.QueryRow(ctx, "SELECT id, name FROM pgx_people WHERE id = $1", int32(2)).Scan(&id, &name); err != nil {
		panic(err)
	}
	requireEqual(id, int32(2), "prepared id")
	requireEqual(name, "Linus", "prepared text")

	rows, err := conn.Query(ctx, "SELECT id, name FROM pgx_people WHERE id = $1", int32(99))
	if err != nil {
		panic(err)
	}
	defer rows.Close()
	if rows.Next() {
		panic("empty result query unexpectedly returned a row")
	}
	if err := rows.Err(); err != nil {
		panic(err)
	}

	_, err = conn.Exec(ctx, "COPY pgx_people FROM STDIN WITH CSV HEADER DELIMITER ','")
	if err == nil {
		panic("broader COPY CSV options unexpectedly succeeded")
	}
	var pgErr *pgconn.PgError
	if !errors.As(err, &pgErr) || pgErr.Code != "0A000" {
		panic(fmt.Sprintf("unsupported COPY returned %T %[1]v, want SQLSTATE 0A000", err))
	}

	var recovered string
	if err := conn.QueryRow(ctx, "SELECT name FROM pgx_people WHERE id = 3").Scan(&recovered); err != nil {
		panic(err)
	}
	requireEqual(recovered, "Grace", "recovery query")
	_ = conn.Close(ctx)

	poolCfg, err := pgxpool.ParseConfig(connectConfig(srv.port, "pgx_pool_smoke"))
	if err != nil {
		panic(err)
	}
	poolCfg.MaxConns = 1
	pool, err := pgxpool.NewWithConfig(ctx, poolCfg)
	if err != nil {
		panic(err)
	}
	defer pool.Close()

	var pooled int32
	if err := pool.QueryRow(ctx, "SELECT 1 AS one", pgx.QueryExecModeSimpleProtocol).Scan(&pooled); err != nil {
		panic(err)
	}
	requireEqual(pooled, int32(1), "pooled query")

	reconnected, err := pgx.Connect(ctx, connectConfig(srv.port, "pgx_reconnect_smoke"))
	if err != nil {
		panic(err)
	}
	defer reconnected.Close(ctx)
	var reconnectOne int32
	if err := reconnected.QueryRow(ctx, "SELECT 1 AS one", pgx.QueryExecModeSimpleProtocol).Scan(&reconnectOne); err != nil {
		panic(err)
	}
	requireEqual(reconnectOne, int32(1), "reconnect query")
}
