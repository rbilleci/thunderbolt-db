# BENCH-001 canonical OLTP workload v1

This versioned document is the immutable workload manifest behind the charter's aggregate throughput targets. It is
a benchmark contract, not a work ledger or an implementation claim. `PLAN.md` owns BENCH-001 execution. Any change
to this manifest requires an explicit ADR-008 target revision and a new independently reviewed manifest version;
benchmark operators may not tune the workload after seeing either engine's result.

## Database and initial state

Both engines start from the same logical rows and execute PostgreSQL `READ COMMITTED` transactions with synchronous
commit. Every statement is prepared before warm-up. `BEGIN`/`COMMIT` are control statements and are not counted as
relational operations.

```sql
CREATE TABLE accounts (
    tenant_id      int4   NOT NULL,
    account_id     int8   NOT NULL,
    balance_cents  int8   NOT NULL,
    version        int8   NOT NULL,
    status         int2   NOT NULL,
    PRIMARY KEY (tenant_id, account_id)
);
CREATE INDEX accounts_by_status
    ON accounts (tenant_id, status, account_id);

CREATE TABLE account_limits (
    tenant_id          int4 NOT NULL,
    account_id         int8 NOT NULL,
    debit_limit_cents  int8 NOT NULL,
    PRIMARY KEY (tenant_id, account_id)
);

CREATE TABLE ledger_entries (
    entry_id      int8  NOT NULL PRIMARY KEY,
    tenant_id     int4  NOT NULL,
    account_id    int8  NOT NULL,
    transfer_id   uuid  NOT NULL,
    amount_cents  int8  NOT NULL,
    direction     int2  NOT NULL,
    created_seq   int8  NOT NULL
);
CREATE INDEX ledger_by_account
    ON ledger_entries (tenant_id, account_id, entry_id);

CREATE TABLE pending_entries (
    tenant_id   int4 NOT NULL,
    pending_id  int8 NOT NULL,
    account_id  int8 NOT NULL,
    payload     int8 NOT NULL,
    PRIMARY KEY (tenant_id, pending_id)
);
CREATE INDEX pending_by_account
    ON pending_entries (tenant_id, account_id, pending_id);
```

The seed contains exactly 1,000 tenants; 10,000 accounts and matching limit rows per tenant (10,000,000 each);
10,000,000 ledger rows; and 4,000,000 pending rows. Account IDs are `tenant_id * 10,000 + local_id`, with
`tenant_id` in `[0, 999]` and `local_id` in `[0, 9,999]`. Every balance starts at 1,000,000,000,000 cents, every
version at zero, every status at one, and every debit limit at 100,000,000 cents. For seed ledger ordinal `j` in
`[0, 9,999,999]`, `entry_id=j`, `tenant_id=j%1,000`, `local_id=(j/1,000)%10,000`, `amount_cents=(j%100,000)+1`,
`direction=j%2`, and `created_seq=j`; its UUID is the 16 big-endian bytes formed by
`splitmix64_once(j) || splitmix64_once(j XOR 0x9e3779b97f4a7c15)`. `splitmix64_once(x)` means the output from one
SplitMix64 step whose initial 64-bit state is exactly `x`. For seed pending ordinal `j` in `[0, 3,999,999]`,
`tenant_id=j%1,000`, `pending_id=j`, `local_id=(j/1,000)%10,000`, and `payload=j`. Run-generated ledger IDs are
`10,000,000 + transaction_sequence * 8 + ledger_insert_ordinal`, where `ledger_insert_ordinal` is zero-based among
that transaction's ledger INSERTs: W1 INSERT uses zero; T8 debit/credit use zero/one; and T32 account order uses
zero through seven. Thus every transaction owns a disjoint eight-ID range and seed/run IDs are disjoint. The five
W1 DELETE template entries in each unshuffled 200-transaction cycle are labeled `d=0,1,2,3,4`; the label moves with
the entry through permutation and fixes `pending_id=cycle_number*5+d`. The published account, limit, ledger, and
pending generations plus all named indexes must be resident before warm-up; the canonical R1/W1/T8/T32 routes
permit zero cold accesses.

## Deterministic selection

The workload generator uses SplitMix64 with seed `0x6a09e667f3bcc909`. Its step adds
`0x9e3779b97f4a7c15`, applies XOR-shift/multiply pairs `(>>30, 0xbf58476d1ce4e5b9)` and
`(>>27, 0x94d049bb133111eb)`, then XOR-shifts by 31, all modulo 2^64. Transaction sequence is continuous from zero
through warm-up, sustained measurement, and `B01`–`B10`, and restarts from the identical snapshot/zero for the
other engine. Each 200-transaction cycle begins as 120 R1, 35 W1 INSERT, 10 W1 UPDATE, 5 W1 DELETE, 20 T8, and
10 T32 entries. Fisher–Yates walks indices 199 down to 1 and swaps with `next_u64() % (index+1)` using a generator
initialized to `seed XOR cycle_number`. For global `transaction_sequence=s`, `cycle_number=floor(s/200)` and the
transaction is the permuted entry at zero-based index `s%200`; neither engine may receive a different order.
Transaction parameters use a separate generator initialized to
`seed XOR transaction_sequence XOR 0xd1b54a32d192ed03`.

Except for W1 DELETE, the first parameter-generator output selects `tenant_id=next_u64()%1,000`. Each requested
account then consumes exactly two outputs in generated account order. The first selects hot when
`next_u64()%10 < 8` and cold otherwise; the second selects local ID `next_u64()%100` for hot or
`100 + next_u64()%9,900` for cold. A collision with an earlier account in the same transaction is resolved without
another generator output by incrementing modulo that selection's `[0,99]` or `[100,9,999]` range until distinct.
R1, W1 INSERT, and W1 UPDATE generate one account; T8 generates `a0,a1`; T32 generates `a0..a7`. W1 DELETE consumes
no parameter-generator output and instead derives `tenant_id=pending_id%1,000`, so its labeled seeded target exists;
this is uniform across tenants over each 1,000-delete span.

After all accounts have been generated, parameters consume outputs in this exact order:

- R1 consumes none.
- W1 INSERT consumes one amount output, then one direction output. Its amount is
  `(next_u64()%100,000)+1`, its direction is `next_u64()%2`, and its ledger-insert ordinal is zero.
- W1 UPDATE consumes one amount output, then one sign output. The magnitude is
  `(next_u64()%100,000)+1`; the delta is positive when `next_u64()%2=0` and negative otherwise.
- W1 DELETE consumes none; its zero-based `d` label and pending ID were fixed before permutation.
- T8 consumes one amount output. Account `a0` is debit and `a1` is credit; the magnitude is
  `(next_u64()%100,000)+1`, deltas are respectively negative/positive, and their ledger-insert ordinals are zero/one.
- T32 consumes exactly four amount outputs. Pair `p` for `p=0,1,2,3` is `(a[2p],a[2p+1])`; the first account is
  debit, the second credit, and magnitude `p` is `(next_u64()%100,000)+1` from the corresponding output. The eight
  deltas in account order are `[-m0,+m0,-m1,+m1,-m2,+m2,-m3,+m3]`, and their ledger-insert ordinals are zero through
  seven in that same order.

Transfer UUID generation is independent of the parameter stream. Let `x = seed XOR transaction_sequence`; the UUID
is the 16 big-endian bytes `splitmix64_once(x) || splitmix64_once(bitwise_not_64(x))`, where `bitwise_not_64` flips
all 64 bits. Each generated ledger row uses
`created_seq=transaction_sequence`, the signed delta's magnitude as `amount_cents`, and direction zero for debit or
one for credit; W1 INSERT uses its separately generated positive amount and direction. DELETE consumes its labeled
pending ID exactly once. A missing target, constraint failure, transaction abort, dropped request, or retry inside
the measured cohort fails the workload; it is not silently replaced.

## Exact route manifests

R1 is one operation:

```sql
SELECT balance_cents, version, status
FROM accounts
WHERE tenant_id = $1 AND account_id = $2;
```

W1 INSERT, UPDATE, and DELETE are one operation each:

```sql
INSERT INTO ledger_entries
    (entry_id, tenant_id, account_id, transfer_id, amount_cents, direction, created_seq)
VALUES ($1, $2, $3, $4, $5, $6, $7)
RETURNING entry_id;

UPDATE accounts
SET balance_cents = balance_cents + $3, version = version + 1
WHERE tenant_id = $1 AND account_id = $2
RETURNING balance_cents, version;

DELETE FROM pending_entries
WHERE tenant_id = $1 AND pending_id = $2
RETURNING pending_id;
```

The limit read used by T8/T32 is:

```sql
SELECT debit_limit_cents
FROM account_limits
WHERE tenant_id = $1 AND account_id = $2;
```

T8 is one atomic two-account transfer with exactly eight relational operations and four mutations, in this order:

1. read the debit `accounts` row with the R1 predicate;
2. read the credit `accounts` row;
3. read the debit `account_limits` row;
4. read the credit `account_limits` row;
5. update the debit account with the W1 UPDATE statement and negative amount;
6. update the credit account with the W1 UPDATE statement and positive amount;
7. insert the debit `ledger_entries` row with the W1 INSERT statement; and
8. insert the credit `ledger_entries` row.

T32 is one atomic eight-account/four-pair posting with exactly 32 relational operations and 16 mutations, in this
order:

1. operations 1–8 read the eight `accounts` rows in generated account order;
2. operations 9–16 read the matching eight `account_limits` rows;
3. operations 17–24 update the eight accounts in the same order; and
4. operations 25–32 insert one matching `ledger_entries` row per account in the same order.

Every T8/T32 read is an equality lookup on the declared primary key. Every mutation uses the exact prepared W1 SQL
above. T8 returns the two UPDATE results and two entry IDs; T32 returns eight UPDATE results and eight entry IDs.
There is no client think time between operations.

## Numeric route envelopes

The byte cap is the sum of encoded post-images plus canonical logical WAL intent/outcome payload before physical
frame padding. Index fanout is the total number of maintained latest-head/index entries affected by the mutations.
Result bytes exclude fixed wire framing. Admission fails before sequence/WAL claim if any actual request exceeds its
row; the benchmark may not raise a cap. Physical WAL/frame bytes are reported separately for both engines.

| Route | Operations | Mutations | Post-image + logical WAL bytes | Maintained-index fanout | Touched tables | Cold accesses | Result bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 | 1 | 0 | 0 | 0 | 1 | 0 | 64 |
| W1 INSERT | 1 | 1 | 512 | 2 | 1 | 0 | 64 |
| W1 UPDATE | 1 | 1 | 512 | 2 | 1 | 0 | 64 |
| W1 DELETE | 1 | 1 | 512 | 2 | 1 | 0 | 64 |
| T8 | 8 | 4 | 4,096 | 8 | 3 | 0 | 2,048 |
| T32 | 32 | 16 | 16,384 | 32 | 3 | 0 | 8,192 |

## Sustained and peak protocol

After loading and residency proof, let `warmup_start` be a monotonic-clock timestamp. Warm-up schedules exactly
3,300,000 transactions, with zero-based warm-up transaction `i` arriving at
`warmup_start + floor(i * 1,000,000,000 / 110,000)` nanoseconds for `i` in `[0,3,299,999]`. Measurement begins at
`measurement_start = warmup_start + 30 seconds` and follows immediately: it schedules exactly 66,000,000
transactions, with zero-based measurement transaction `i` arriving at
`measurement_start + floor(i * 1,000,000,000 / 110,000)` nanoseconds for `i` in `[0,65,999,999]`. This fixed,
evenly paced open-loop sequence—not Poisson arrivals, per-second clumps, or a closed-loop client—defines exactly
110,000 scheduled transactions/s in each phase. Sustained achieved TPS is the number of measurement-scheduled
transactions whose terminal committed completions are timestamped inside
`[measurement_start, measurement_start + 600 seconds)` divided by 600; warm-up completions are excluded. It must be
strictly greater than 100,000. Latency uses scheduled arrival through terminal client-visible completion for
transactions scheduled inside the measurement window. At the end, every admitted-stage population must be at or
below its measurement-start value and must reach the pre-warm-up idle bound within one second; otherwise backlog
invalidates the result.

Exactly 30 seconds after the sustained queue returns to the idle bound, peak bursts run in fixed order `B01` through
`B10`. Burst `Bk` schedules exactly 400,000 transactions during `[start_k, start_k + 1 second)`, with transaction `i`
scheduled at `start_k + floor(i * 1,000,000,000 / 400,000)` nanoseconds. A burst's **cohort TPS** is its terminal
committed cohort count divided by the fixed 1.000-second arrival interval; it is not completions observed inside that
same wall-clock second. Passing requires all 400,000 scheduled transactions to commit, cohort TPS = 400,000, every
class latency envelope to pass from scheduled arrival, and all admitted-stage populations to return to or below the
pre-burst values no later than one second after the last scheduled arrival. The next burst starts exactly five
seconds after that drain deadline. All ten named bursts are reported and all ten must pass; failed, late, or missing
bursts cannot be omitted or replaced. Wall-clock completion throughput and last-completion time are reported as
additional diagnostics.

The deterministic mix contains 650 logical operations per 200 transactions. The sustained pass therefore implies
more than 325,000 logical operations/s and each peak cohort reports exactly 1,300,000 logical operations/s. R1, each
W1 operation, the W1 mix, T8, and T32 retain separate p50/p99/p99.9/p99.99 distributions. A pooled latency,
standalone saturation run, different seed/order/data shape, or best subwindow cannot satisfy this contract.
