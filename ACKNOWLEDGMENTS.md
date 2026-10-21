# Research and reference implementations

Fast DP Boot implements and accelerates the Gaussian differential-privacy
bootstrap methods described by **Zhanyu Wang, Guang Cheng, and Jordan Awan** in
[Differentially Private Bootstrap: New Privacy Analysis and Inference Strategies](https://arxiv.org/abs/2210.06140).

The authors' [reference repository](https://github.com/Zhanyu-Wang/Differentially_Private_Bootstrap)
is pinned in our comparisons to commit
`fd001561ed338d47143c94b995f4d47478ab9eb0`. That repository is Apache-2.0 licensed.
Benchmark drivers extract and execute selected functions from a separate local
checkout; the downloaded checkout is not included in this repository.

The matched deconvolution model is validated against
[deconvolveR](https://cran.r-project.org/package=deconvolveR), by Bradley Efron
and Balasubramanian Narasimhan. Its R package is GPL-2-or-later licensed and
installed separately for reference comparisons. Core Python inference uses
NumPy and SciPy and does not invoke R. Checked-in fixtures are synthetic
numerical outputs from R 4.6.0 and deconvolveR 1.2.2; their provenance accompanies
the fixtures in `tests/fixtures/deconvolver/PROVENANCE.txt`.

Performance changes here preserve the reference statistical model where stated;
they do not establish a new privacy theorem or universal speedup. Credit for
the underlying research belongs to the paper's authors. See the comparison
reports for the scope of fidelity checks and numerical limitations.
