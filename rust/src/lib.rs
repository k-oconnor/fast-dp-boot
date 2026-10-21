//! Exact Efron bootstrap means, with no privacy noise or data-dependent release.
//!
//! These kernels consume already clipped, confidential inputs. Callers must
//! calibrate and add noise before releasing results. Floating-point arithmetic
//! and pseudorandom sampling are not a hardened differential privacy system.

use rand::distr::{Distribution, Uniform};
use rand::Rng;
use rand_distr::Binomial;

fn validate(values: &[f64], replicates: usize) -> Result<(), &'static str> {
    if values.is_empty() || values.iter().any(|x| !x.is_finite()) {
        return Err("values must be nonempty and finite");
    }
    if replicates < 2 {
        return Err("replicates must be >= 2");
    }
    Ok(())
}

/// Direct independent uniform row draws, O(Bn) time and O(B) output memory.
/// Sampling scratch space is constant; no index/resample matrices are built.
pub fn mean_draws<R: Rng + ?Sized>(
    values: &[f64],
    replicates: usize,
    rng: &mut R,
) -> Result<Vec<f64>, &'static str> {
    validate(values, replicates)?;
    let rows = Uniform::new(0, values.len()).map_err(|_| "invalid row count")?;
    let n = values.len() as f64;
    let mut out = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut sum = 0.0;
        for _ in 0..values.len() {
            sum += values[rows.sample(rng)];
        }
        out.push(sum / n);
    }
    Ok(out)
}

/// Multinomial draws via conditional binomials, O(BK) time and O(K+B) memory.
/// Integer suffix counts avoid subtracting nearly equal floating probabilities.
/// Zero-count categories and duplicate support values are permitted.
pub fn mean_counts<R: Rng + ?Sized>(
    values: &[f64],
    counts: &[u64],
    replicates: usize,
    rng: &mut R,
) -> Result<Vec<f64>, &'static str> {
    validate(values, replicates)?;
    if values.len() != counts.len() {
        return Err("counts must match values");
    }
    let n = counts
        .iter()
        .try_fold(0_u64, |total, &count| total.checked_add(count))
        .ok_or("total count overflow")?;
    // All integers through 2^53 are exactly representable in binary64.
    if n == 0 || n > (1_u64 << 53) {
        return Err("Rust frequency-table total must lie in [1, 2**53]");
    }
    let mut remaining = n;
    let probabilities: Vec<f64> = counts
        .iter()
        .map(|&count| {
            let p = if remaining == 0 {
                0.0
            } else {
                count as f64 / remaining as f64
            };
            remaining -= count;
            p
        })
        .collect();
    let mut out = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut draws_left = n;
        let mut sum = 0.0;
        for (&value, &p) in values.iter().zip(&probabilities) {
            if draws_left == 0 {
                break;
            }
            let weight = if p == 1.0 {
                draws_left
            } else if p == 0.0 {
                0
            } else {
                Binomial::new(draws_left, p)
                    .map_err(|_| "invalid conditional binomial")?
                    .sample(rng)
            };
            sum += weight as f64 * value;
            draws_left -= weight;
        }
        out.push(sum / n as f64);
    }
    Ok(out)
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn seeded(seed: &[u8]) -> PyResult<ChaCha20Rng> {
        let seed: [u8; 32] = seed
            .try_into()
            .map_err(|_| PyValueError::new_err("seed must contain 32 bytes"))?;
        Ok(ChaCha20Rng::from_seed(seed))
    }

    #[pyfunction(name = "mean_draws")]
    fn py_mean_draws<'py>(
        py: Python<'py>,
        values: PyReadonlyArray1<'py, f64>,
        replicates: usize,
        seed: &[u8],
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut rng = seeded(seed)?;
        // Own the input before releasing the GIL: another Python thread may
        // otherwise mutate its NumPy storage during sampling.
        let values: Vec<f64> = values.as_array().iter().copied().collect();
        let out = py
            .detach(move || mean_draws(&values, replicates, &mut rng))
            .map_err(PyValueError::new_err)?;
        Ok(out.into_pyarray(py))
    }

    #[pyfunction(name = "mean_counts")]
    fn py_mean_counts<'py>(
        py: Python<'py>,
        values: PyReadonlyArray1<'py, f64>,
        counts: PyReadonlyArray1<'py, u64>,
        replicates: usize,
        seed: &[u8],
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut rng = seeded(seed)?;
        let values: Vec<f64> = values.as_array().iter().copied().collect();
        let counts: Vec<u64> = counts.as_array().iter().copied().collect();
        let out = py
            .detach(move || mean_counts(&values, &counts, replicates, &mut rng))
            .map_err(PyValueError::new_err)?;
        Ok(out.into_pyarray(py))
    }

    #[pymodule]
    fn dp_boot_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(py_mean_draws, m)?)?;
        m.add_function(wrap_pyfunction!(py_mean_counts, m)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn sampling_and_repeatability() {
        let x = [0.0, 1.0, 2.0];
        let a = mean_draws(&x, 100, &mut ChaCha20Rng::from_seed([7; 32])).unwrap();
        let b = mean_draws(&x, 100, &mut ChaCha20Rng::from_seed([7; 32])).unwrap();
        assert_eq!(a, b);
        assert!(a.iter().all(|&v| (0.0..=2.0).contains(&v)));
    }

    #[test]
    fn degenerate_counts_and_validation() {
        let mut rng = ChaCha20Rng::from_seed([1; 32]);
        assert_eq!(
            mean_counts(&[9.0, 0.25, -4.0], &[0, 10, 0], 20, &mut rng).unwrap(),
            vec![0.25; 20]
        );
        assert!(mean_draws(&[], 2, &mut rng).is_err());
        assert!(mean_draws(&[f64::NAN], 2, &mut rng).is_err());
        assert!(mean_draws(&[1.0], 1, &mut rng).is_err());
        assert!(mean_counts(&[1.0], &[0], 2, &mut rng).is_err());
        assert!(mean_counts(&[1.0], &[1, 2], 2, &mut rng).is_err());
        assert!(mean_counts(&[1.0], &[(1 << 53) + 1], 2, &mut rng).is_err());
        assert!(mean_counts(&[1.0, 2.0], &[u64::MAX, 1], 2, &mut rng).is_err());
    }
}
