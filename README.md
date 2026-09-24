# Numerical linear algebra, written out

![CI](https://github.com/ducquanng/numerical-linear-algebra/actions/workflows/ci.yml/badge.svg)

QR factorisation four ways, the power method, SVD, PageRank and eigenfaces — implemented
from scratch in NumPy. LAPACK appears only as the reference the tests check against.

```bash
pip install -e ".[dev]"
pytest -q                # 60 tests, seconds
python -m numla.study    # every experiment, writes reports/
```

## Why nobody uses the QR algorithm textbooks teach first

Four factorisations on Hilbert matrices of rising condition number, measured by how far
Q drifts from orthogonal, ‖I − QᵀQ‖₂:

| n | Condition number | Classical GS | Modified GS | Householder | Givens |
| --- | --- | --- | --- | --- | --- |
| 4 | 1.6e+04 | 3.5e−11 | 3.0e−13 | 1.2e−15 | 3.0e−16 |
| 8 | 1.5e+10 | **1.0e+00** | 2.6e−07 | 9.7e−16 | 6.1e−16 |
| 12 | 1.6e+16 | **5.0e+00** | 2.3e−01 | 1.2e−15 | 1.1e−15 |
| 14 | 3.2e+17 | **6.6e+00** | 9.6e−01 | 1.7e−15 | 1.3e−15 |

![Loss of orthogonality](reports/figures/orthogonality.png)

By n = 7 classical Gram-Schmidt produces a Q whose columns are no longer independent in
any useful sense. Modified Gram-Schmidt **differs by one line** — each projection is
subtracted from the running vector rather than from the original column — and survives
three orders of magnitude further. Householder and Givens stay at machine precision
throughout, because they apply exactly orthogonal transformations instead of
accumulating subtractions.

The measured curves track the ε·κ and ε·κ² guides, which is the theory. A test asserts
the ordering so the result cannot silently rot.

## The power method, and what a small spectral gap costs

| \|λ₂/λ₁\| | Iterations to 1e−12 | Observed decay rate |
| --- | --- | --- |
| 0.20 | 19 | 0.200 |
| 0.50 | 42 | 0.500 |
| 0.80 | 124 | 0.800 |
| 0.95 | 513 | 0.950 |
| 0.99 | 2,456 | 0.990 |

The observed decay matches the predicted ratio to three decimals, and the cost of a
0.99 gap is a hundred times the cost of a 0.20 gap — same algorithm, same code.

## PageRank is the power method

A page's rank is the chance a random surfer is on it, which makes the rank vector the
dominant eigenvector of the link matrix. On a six-page graph with a dangling node, power
iteration converges in 41 steps and agrees with a full eigendecomposition to 2e−13.

Damping is not a tuning knob: without it a disconnected component traps the surfer and
the stationary distribution isn't unique. The eigendecomposition route is exact and
unusable at web scale — dense O(n³) on a matrix with billions of rows — which is exactly
why the power method is what Google described.

## Eigenfaces on the hand-written SVD

Olivetti faces, 40 identities, 280 training and 120 test images at 64×64. The basis
comes from `power_svd`, not from a library.

![Eigenfaces](reports/figures/eigenfaces.png)

| Components | Accuracy (my SVD) | Accuracy (scikit-learn PCA) |
| --- | --- | --- |
| 5 | 0.692 | 0.692 |
| 10 | 0.833 | 0.833 |
| 20 | 0.892 | 0.892 |
| 80 | 0.892 | 0.892 |

Matching scikit-learn at every size is the point: it shows the decomposition is correct,
not merely producing plausible-looking pictures. Twenty numbers per face replace 4,096
pixels — 200× compression — with accuracy holding at 89%.

## What's here

```
src/numla/
  qr.py           classical and modified Gram-Schmidt, Householder, Givens
  eigen.py        power method with Rayleigh quotient, deflation, convergence tracking
  svd.py          truncated SVD via the Gram matrix, with its precision cost documented
  pagerank.py     Google matrix, power iteration, eigendecomposition cross-check
  eigenfaces.py   centred basis, projection, nearest-neighbour classification
  stability.py    the orthogonality experiment
  study.py        runs everything, writes the report
```

60 tests, all against known answers: factorisations reconstruct their matrix and agree
with LAPACK up to column signs, the truncated SVD hits the Eckart-Young bound exactly,
PageRank sums to one and is uniform on a symmetric ring, Givens rotations survive
entries of 1e200 (the reason `np.hypot` is in there), and the stability ordering above
is asserted rather than assumed.

## Honest limitations

- `power_svd` forms AᵀA, which squares the condition number and loses singular values
  below about 1e−8 of the largest. Fine when the small directions are noise, as in
  eigenfaces; wrong when they are the answer. LAPACK never forms the Gram matrix.
- Deflation accumulates error across eigenpairs, so a full spectrum built this way
  degrades as it goes.
- These are for understanding and for testing, not for speed: unblocked, uncached,
  single-threaded. Use LAPACK in production — and now you know what it is protecting you
  from.

## Provenance

Rebuilt from my Math 3 (numerical linear algebra) coursework in the BSc Business
Analytics at the University of Amsterdam, which implemented QR by Gram-Schmidt and
Givens rotations, PageRank three ways, and eigenfaces on a power-method SVD. Those were
graded group submissions on the course's scaffolding; this is my own rewrite, with the
stability study, the convergence measurements, the LAPACK comparisons and the tests
added.
