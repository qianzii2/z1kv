# Fuzz Targets

Four fuzz targets: WAL record parsing / `.zpatch` parsing / checkpoint
envelope / visibility decisions.

## Prerequisites

The libFuzzer backend needs the MSVC `clang_rt.asan` x64 runtime library
(missing in this project's original environment). After installing the VS
component "C++ AddressSanitizer":

```sh
rustup component add llvm-tools-preview --toolchain nightly
cargo +nightly fuzz build
cargo +nightly fuzz run parse_wal_record -- -max_total_time=600
```

## Running without an ASan environment

`tests/fuzz_smoke.rs` drives the same input generation and parse loops
(pseudo-random with fixed seeds) as part of `cargo test` on the stable
toolchain — losing coverage guidance and continuous mutation, but keeping
the contract regression capability.
