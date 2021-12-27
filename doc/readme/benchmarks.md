# Benchmarks

The benchmarks for this project may be found in the `benchmarks` folder.
Using the `BUILD_BENCHMARKS` option, `CMake` can be configured to build the
benchmarks instead of the unit tests. They can then be executed by hand or,
for non-MPI tests, sequentially using CTest. Timings are performed using
deal.II's `Timer` class.

These results were produced on a 2012 Macbook Pro Retina (MacBookPro10,1), 
which has a 2.6 GHz quad-core Intel Core i7 that supports (only) AVX1
instructions, and has 256 KB L2 Cache per core, 6 MB L3 cache and 16 GB of
1600 MHz DDR3 RAM. 
Given the age of this machine, it's not going to produce any record-setting
performance results. But hopefully it'll provide some coarse indication as to
what the performance deficit (or gains) with this implementation are and
provide some insights as to whether its worth considering using this
project within your own code.

## Step-44 (matrix-based)

Firstly, the parameters of the assessed benchmark test. Modifications of
`step-44` were limited to the assembly loop, which was highly optimised in
its original implementation. Static condensation was not used (meaning that
linear operators were used as part of the linear solve process).

### Test descriptions

A summary of the various implementations are as follows:
- `step-44-variant_01a`: 
  Replication of "stock `step-44`".
- `step-44-variant_01d`: 
  Assembly using composite weak forms, with the Piola stress being used for the
  problem parameterisation (i.e. this is an application of the two-point
  formulation).
- `step-44-variant_01h`:
  Assembly using composite weak forms, with the Piola-Kirchhoff stress being
  used for the problem parameterisation (i.e. this is an application of the fully
  referential formulation).
- `step-44-variant_01i`:
  Assembly using composite weak forms, with the Kirchhoff stress is used for the
  problem parameterisation (i.e. the same parameterisation as the original
  `step-44` implementation).
- `step-44-variant_02-energy_functional_sd_02`:
  Assembly using a self-linearizing energy functional weak form in conjunction
  with symbolic differentiation. The internal energy is calculated by hand, and
  the external energy is also supplied. From this, all first and second
  derivatives are calculated automatically. For the symbolic batch optimiser, the
  `lambda` optimiser type is selected and `all` available optimisation methods
  are utilised.
- `step-44-variant_02-energy_functional_sd_02a`: 
  Identical to `step-44-variant_02-energy_functional_sd_02`, except that all
  SD calculations are cached between timesteps and Newton iterations.
- `step-44-variant_02-energy_functional_sd_03`:
  Identical to `step-44-variant_02-energy_functional_sd_02`, except that the
  high performance `LLVM` optimiser type is selected.
- `step-44-variant_02-energy_functional_sd_03a`:
  Identical to `step-44-variant_02-energy_functional_sd_03`, except that it also
  includes SD calculation caching.
- `step-44-variant_02-residual_view_sd_02`:
  Assembly using a self-linearizing residual weak form in conjunction with
  symbolic differentiation. The contributions to the residual are provided to
  the weak form, and from this its linearisation is calculated automatically.
  For the symbolic batch optimiser, the `lambda` optimiser type is selected and
  `all` available optimisation methods are utilised. 
- `step-44-variant_02-residual_view_sd_02a`: 
  Identical to `step-44-variant_02-residual_view_sd_02`, except that all
  SD calculations are cached between timesteps and Newton iterations.
- `step-44-variant_02-residual_view_sd_03`: 
  Identical to `step-44-variant_02-residual_view_sd_02`, except that the
  high performance `LLVM` optimiser type is selected.
- `step-44-variant_02-residual_view_sd_03a`: 
  Identical to `step-44-variant_02-residual_view_sd_03`, except that it also
  includes SD calculation caching.

### Cases

The four cases that were studied had the following discretisations:

Benchmark              | 1     | 2     | 3      | 4
-----------------------|-------|-------|--------|-------
Polynomial order       | 1     | 2     | 1      | 2
Quadrature order       | 2     | 3     | 2      | 3
Global refinements     | 4     | 3     | 5      | 4
Number of active cells | 4096  | 512   | 32768  | 49096
Number of DoFs         | 22931 | 18835 | 173347 | 140579

### Results

The performance ratio is defined as the ratio of the assembly time of the
nominated test versus that of the "manual" or "hand" implementation performed
in `step-44-variant_01a`.

#### Benchmark 1

Benchmark                                   | Assembly time (s) | Solver time (s) | Total time (s) | Assembly time (% of total) | Performance ratio
--------------------------------------------|-------------------|-----------------|----------------|----------------------------|------------------
step-44-variant_01a                         | 4.072             | 250             | 255.5          | 1.59%                      | 1.00
step-44-variant_01d                         | 8.876             | 251.2           | 261.5          | 3.39%                      | 2.18
step-44-variant_01h                         | 15.26             | 248.7           | 265.5          | 5.75%                      | 3.75
step-44-variant_01i                         | 9.224             | 240.4           | 251            | 3.67%                      | 2.27
step-44-variant_02-energy_functional_sd_02  | 44.04             | 251.2           | 296.8          | 14.84%                     | 10.82
step-44-variant_02-energy_functional_sd_02a | 47.42             | 259.1           | 308.1          | 15.39%                     | 11.65
step-44-variant_02-energy_functional_sd_03  | 38.1              | 255.2           | 294.8          | 12.92%                     | 9.36
step-44-variant_02-energy_functional_sd_03a | 30.61             | 255.4           | 287.5          | 10.65%                     | 7.52
step-44-variant_02-residual_view_sd_02      | 113.7             | 247.4           | 362.9          | 31.33%                     | 27.92
step-44-variant_02-residual_view_sd_02a     | 109.6             | 256.4           | 367.8          | 29.80%                     | 26.92
step-44-variant_02-residual_view_sd_03      | 76.27             | 255.2           | 333.3          | 22.88%                     | 18.73
step-44-variant_02-residual_view_sd_03a     | 34.07             | 258.3           | 294            | 11.59%                     | 8.37

#### Benchmark 2

Benchmark                                   | Assembly time (s) | Solver time (s) | Total time (s) | Assembly time (% of total) | Performance ratio
--------------------------------------------|-------------------|-----------------|----------------|----------------------------|------------------
step-44-variant_01a                         | 15.5              | 296.3           | 313            | 4.95%                      | 1.00
step-44-variant_01d                         | 16.39             | 287.5           | 305.1          | 5.37%                      | 1.06
step-44-variant_01h                         | 26.11             | 288.2           | 315.5          | 8.28%                      | 1.68
step-44-variant_01i                         | 13.34             | 295.3           | 309.9          | 4.30%                      | 0.86
step-44-variant_02-energy_functional_sd_02  | 37.39             | 286.6           | 325.3          | 11.49%                     | 2.41
step-44-variant_02-energy_functional_sd_02a | 37.4              | 287.8           | 326.5          | 11.45%                     | 2.41
step-44-variant_02-energy_functional_sd_03  | 35.52             | 288.9           | 325.7          | 10.91%                     | 2.29
step-44-variant_02-energy_functional_sd_03a | 30.09             | 288.7           | 320            | 9.40%                      | 1.94
step-44-variant_02-residual_view_sd_02      | 65.52             | 278.9           | 345.9          | 18.94%                     | 4.23
step-44-variant_02-residual_view_sd_02a     | 58.39             | 280.2           | 340.1          | 17.17%                     | 3.77
step-44-variant_02-residual_view_sd_03      | 58.73             | 298.3           | 358.6          | 16.38%                     | 3.79
step-44-variant_02-residual_view_sd_03a     | 29.48             | 312.9           | 343.7          | 8.58%                      | 1.90

#### Benchmark 3

Benchmark                                   | Assembly time (s) | Solver time (s) | Total time (s) | Assembly time (% of total) | Performance ratio
--------------------------------------------|-------------------|-----------------|----------------|----------------------------|------------------
step-44-variant_01a                         | 35.11             | 5920            | 5964           | 0.59%                      | 1.00
step-44-variant_01i                         | 83.72             | 5918            | 6010           | 1.39%                      | 2.38
step-44-variant_02-energy_functional_sd_03a | 262.8             | 5852            | 6123           | 4.29%                      | 7.49
step-44-variant_02-residual_view_sd_03a     | 287.9             | 5897            | 6194           | 4.65%                      | 8.20

#### Benchmark 4

Benchmark                                   | Assembly time (s) | Solver time (s) | Total time (s) | Assembly time (% of total) | Performance ratio
--------------------------------------------|-------------------|-----------------|----------------|----------------------------|------------------
step-44-variant_01a                         | 131.4             | 6998            | 7135           | 1.84%                      | 1.00
step-44-variant_01i                         | 106.9             | 6781            | 6894           | 1.55%                      | 0.81
step-44-variant_02-energy_functional_sd_03a | 254.6             | 7092            | 7352           | 3.46%                      | 1.94
step-44-variant_02-residual_view_sd_03a     | 250.7             | 7170            | 7427           | 3.38%                      | 1.91

### Observations

Considering the tabulated data, here are some preliminary observations and
comments on the comparative performance of the current implementation:

- The assembly time of the four "hand implementation" variants, `01a`, `01d`,
  `01h` and `01i` are relatively similar. The `01d` and `01h` variants have some
  performance deficit due to the additional pull-back operations (some could)
  be removed with a re-statement of the constitutive laws.
  The test `01h` is quite a bit more expensive due to the push-forward of shape
  functions and additional split of the material and geometric elastic tangents.
- Symbolic computations using an energy functional or a residual view are always
  more costly than hand calculations. Using the `LLVM` JIT compiler in conjunction
  with symbolic expression caching (i.e. the `03a` variant) always increases the
  overall performance.
- For polynomial order `2`, the weak form implementation `01i` is
  actually 15%-20% quicker than the original implementation. This is likely due to
  the additional parallelisation that vectorisation (across the quadrature loop)
  provides. One might speculate than the benefit might be greater at even higher
  FE degrees.
- For polynomial order `2`, the performance deficit of the automatic linearisation
  strategies is minimised at around `1.9x` the manual implementation.
  The increase in performance of the `03a` benchmarks when compared to the lowest
  order FE discretisation is quite substantial. This suggests that increasing
  the polynomial order offsets the costs of symbolic computations. This aligns
  with the expectations from the implementation, as the symbolic calculations
  are done on a quadrature-point level and is then more heavily reused as the
  number of quadrature points per cell increases.

In the end, there is certainly some overhead to using this (symbolic) weak forms
implementation. It is the judgement of the user as to whether the trade off of
raw performance versus any convenience that it provides is worthwhile.
