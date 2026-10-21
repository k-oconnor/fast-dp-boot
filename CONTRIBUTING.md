# Contributing

Install and run the tests from the repository root:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

Core library changes must work without R or the upstream checkout. The checked-in
synthetic fixtures cover agreement with deconvolveR. Keep global sensitivity,
replace-one adjacency, randomness and calibration assumptions explicit when
changing sampling or privacy accounting.

For inference changes, compare the model, fitted distribution, uncertainty
statistics and interval endpoints against the reference before measuring speed.
Benchmark results must state workload, included work, reference version,
numerical differences and statistical limitations. Preserve previous result
files when an experiment changes meaning.

The optional native backend is a separate package, leaving the core install
pure Python. With Rust and a native linker available:

```powershell
cargo test --manifest-path rust/Cargo.toml --locked
cargo fmt --manifest-path rust/Cargo.toml --check
cargo clippy --manifest-path rust/Cargo.toml --all-targets --features python -- -D warnings
python -m pip install ./rust
python -m pytest -q
python -m benchmarks.rust_sampling --repeats 20
```

Native distribution tests skip when the extension is absent. Install it before
validating changes to Rust or backend dispatch. Preserve NumPy as the default
and compare distributions rather than expecting identical random streams.

With the pinned reference checkout and R dependencies installed, run the
interleaved three-way comparison without replacing earlier result files:

```powershell
python -m benchmarks.matched_rust_pipeline --trials 20
```

Use `--rscript` to specify an Rscript executable outside PATH. This comparison
times the original R pipeline and both Python sampling backends on shared inputs.

The main evidence and reproduction commands are in
[benchmarks/MATCHED_RESULTS.md](benchmarks/MATCHED_RESULTS.md). R benchmark
drivers currently use a local Windows R path. Downloaded dependencies and
reference code stay in the ignored `reference/` directory.

The project is research software; passing tests does not certify numerical
privacy bounds or finite confidence-interval coverage. Avoid introducing claims
stronger than the available mathematical and experimental evidence.
