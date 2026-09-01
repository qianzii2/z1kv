# Contributing to Z1KV

Thank you for considering a contribution!

## Development setup

You need a stable Rust toolchain (see `rust-version` in `Cargo.toml`).

```sh
git clone <your-fork-url>
cd z1kv
cargo build
cargo test
```

## Before opening a PR

Please make sure all of the following pass locally:

```sh
cargo fmt --check                              # formatting
cargo clippy --all-targets -- -D warnings     # zero clippy warnings
cargo test --all-targets                      # unit + integration tests
cargo test --doc                              # doc tests
```

CI runs the same checks on Linux, Windows and macOS. Because the engine has
platform-specific durability code paths (`cfg(windows)` /
`cfg(unix)`), a PR that only builds on your local platform may still fail
CI — that is exactly what CI is for.

## Code conventions

- **Invariants are sacred.** D4 (WAL-first), D12 (absent commit_ts_map ⇒
  invisible), D5 (TTL clocks on `inserted_at`), D7 (replay watermark is a
  TTL lower bound) and D8 (recent-flush cache bridges the flush race window)
  are the correctness core of the engine. If your change touches any code
  path involved in these invariants, explain in the PR how the invariant is
  preserved.
- Every new bug fix should come with a regression test that fails on the
  unfixed code.
- Comments and documentation are in English.

## Commit messages

Use short imperative subjects, e.g. `fix: abort checkpoint when flush fails`.
Body text is optional but welcome for non-obvious changes.

## Reporting bugs

Open a GitHub issue with: the version, the platform, a minimal reproduction
(a test is ideal), and the observed vs expected behavior. For suspected data
corruption, include the recovery logs if you can.
