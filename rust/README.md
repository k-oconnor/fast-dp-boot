# Native mean-bootstrap kernels

This crate exposes `mean_draws` and `mean_counts` for confidential, already
clipped data. Both return **noiseless bootstrap means**. They do not perform
privacy calibration, add noise, or provide a safe-to-publish result. Use the
Python `bootstrap_mean(..., backend="rust")` entry point for the existing
calibrated release pipeline.

The Rust library works without Python:

```rust
use dp_boot_native::mean_counts;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

// Fixed seed for a synthetic experiment only.
let mut rng = ChaCha20Rng::from_seed([7; 32]);
let confidential_means = mean_counts(&[0.0, 1.0], &[70, 30], 256, &mut rng).unwrap();
assert_eq!(confidential_means.len(), 256);
```

Direct draws sample row indices uniformly with replacement and accumulate each
mean without allocating a resample. Counts implement a multinomial using
conditional binomials with integer suffix totals. There is no support binning,
Poissonization, or normal approximation. Counts may be zero; total count must
be in `[1, 2**53]`. Replicates must be at least two and input values finite.
These are ordinary floating-point samplers, not certified exact arithmetic.

From the repository root:

```powershell
cargo test --manifest-path rust/Cargo.toml --locked
python -m pip install ./rust
python -m pytest -q
```

The `python` feature builds the PyO3/NumPy bindings. Maturin creates the separate
`fast-dp-boot-native` Python distribution, imported internally as
`dp_boot_native`. Its functions are internal noiseless kernels, not DP releases.
Bindings copy NumPy input before releasing the GIL and transfer output buffers
back into NumPy without copying. They accept strided one-dimensional arrays.
The Python wrapper uses NumPy for Gaussian perturbation and calibration checks;
Rust consumes a ChaCha20 stream seeded by 32 bytes from that NumPy generator.

Keep `Cargo.lock` under version control to record tested dependencies. Rebuild
the extension after changing Rust sources; the installed binary is not editable.
On Windows the MSVC toolchain needs Visual Studio C++ Build Tools and a Windows
SDK. No Rust compiler is needed to install or use the core Python package.
