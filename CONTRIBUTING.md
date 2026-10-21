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

The main evidence and reproduction commands are in
[benchmarks/MATCHED_RESULTS.md](benchmarks/MATCHED_RESULTS.md). R benchmark
drivers currently use a local Windows R path. Downloaded dependencies and
reference code stay in the ignored `reference/` directory.

The project is research software; passing tests does not certify numerical
privacy bounds or finite confidence-interval coverage. Avoid introducing claims
stronger than the available mathematical and experimental evidence.
