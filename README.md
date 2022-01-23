# Weak forms for deal.II
------------------------
An implementation of a symbolic weak form interface for `deal.II`, using 
expression templates as well as automatic and symbolic differentiation. 

Author: Jean-Paul Pelteret, 2020 - 2022

# Table of Contents

- [Concept](#concept)
- [Features](#features)
   - [Highlights](#highlights)
   - [Wishlist and work in progress](#wishlist-and-work-in-progress)
- [Class documentation](#class-documentation)
- [Examples](#examples)
- [Benchmarks](#benchmarks)
- [Building the library](#building-the-library)
- [Citing the library](#citing-the-library)
- [Similar projects that inspired this work](#similar-projects-that-inspired-this-work)
- [Acknowledgements](#acknowledgements)
- [Contributing](#contributing)
- [License](#license)

# Concept
---------
The idea for this library is to offer an abstraction for the discretisation
of finite element weak forms using the `deal.II` open source finite element
library. It, effectively, allows one to express the assembly of discretised
linear system by the weak form alone, the components of which can be reused
as they are lazily evaluated.

To summarise what this library offers the user: by providing an intuitive
method to compose and re-use operations, the user is able to focus more on
*what* they are doing (i.e. the algorithm or discretisation that they are
wanting to implement) rather than *how* they are doing it (i.e. the
programmatic implementation itself). This way you need not be distracted
the minutia of the implementation -- all of the calculation loops are now
hidden details, as are the data storage and value extraction processes.
Permitting the storing and reusing intermediate values (or composite
operations) means that if you want to change that definition, then you only
need to do it in one place -- in the case of a composite operation, all of
the updates to the underlying value composition follow seemlessly, and without
any further intervention. On top of that, by leveraging automatic or symbolic
differentiation, the number of hand-calculations required to implement
complex constitutive laws *correctly* diminishes significantly.

Details as to exactly what the abstraction level is, and how it differs to the
"native" functionality that is already provided in `deal.II` can be found
[here](doc/readme/concept.md).


# Features
----------

## Highlights
- Easy to read and interpret expression of weak forms
- Output of forms in ASCII and LaTeX formats
- Any and all quantities can be retained as intermediate calculations
- Wrappers for many of the commonly used `deal.II` functions and classes
- Operator and function overloading for many `deal.II` dense linear algebra
  classes
- Support for scalar, vector and tensor-valued finite elements
- Support for multi-field forms, thereby supporting the implementation of 
  (coupled) multi-physics problems
- The use of `std::function`s as input to user-definable class value definitions
- Self-linearising forms with specified parameterisations that leverage
  automatic and symbolic differentiation frameworks. This allows for problems
  to be implemented as a (scalar) energy functional or the expression of
  residuals alone. The AD/SD frameworks permit efficient derivative computations
  derived from provided quantities.
- Volume, boundary and interface integration
  - Assembly loops, assembling cell or face matrix and/or vector contributions
    into a global linear system
  - Summation of quantities (like the integral of a field value)
- Supports MPI and serial computing concepts
- Automatically implements multi-threading along with `SIMD` vectorisation
  (when available)

## Wishlist and work in progress
- Currently only supports non-hp finite element methods, but
  [hp-FEM support is imminent](https://github.com/dealii/dealii/pull/13181)
- The datatype for calculations (`float`, `double`, `std::complex<...>`) is,
  chosen only at assembly time and is in principle, generic. 
  This feature, however, needs to be tested more thoroughly.
- Performance improvements for the self-linearising forms is an ongoing area of
  interest and research. 
- A matrix-free implementation will be investigated in the future.
- Python bindings, to be usable in Jupyter Notebooks, will be investigated.

# Class documentation
---------------------
Some documentation for the user-interactive classes and some details
of how they can be used are found [here](doc/readme/classes.md).


# Examples
----------
Some examples and output can be found [here](doc/readme/examples.md).


# Benchmarks
------------
The results of some preliminary benchmarks can be found [here](doc/readme/benchmarks.md).

To summarise, for matrix-based methods the convenience that might be found in
using such a library does come at some overhead. The overheads may be mitigated
when higher order finite element methods are used (i.e. when using higher order 
FEs, a "typical" hand-written assembly loop (meaning, the canonical approach used
in the `deal.II` tutorials) *may* be evaluated slower than the assembly loop 
generated by this library and when all the appropriate settings permitting
optimisations have been chosen). However, there are many factors that might
influence the performance of a code so this comment, guided by the observations
made in the (limited) benchmarking study, should not be considered general truth.
It might be prudent to conduct some examinations of your own before accepting
the analysis done here and following any guidance given by the author.


# Building the library
----------------------
This library requires `deal.II` version `10.0.0` (at the time of writing, this
means the developer version), and at the moment requires that `deal.II` is built
with the following dependencies:
-  ADOL-C
-  Trilinos (with Sacado)
-  SymEngine

Since interaction with these libraries is actually optional, at some point in 
the future these requirements will be removed.

This project uses `CMake` as a build generator. The code block below encapsulates
the various options that can be passed on `CMake` to configure the project before
compilation.

```bash
cmake \
-DCMAKE_BUILD_TYPE=[Debug/Release] \
-DCMAKE_INSTALL_PREFIX=<path_to_installation_location> \
-BUILD_BENCHMARKS=[ON/OFF] \
-DBUILD_DOCUMENTATION=[ON/OFF] \
-DBUILD_TESTS=[ON/OFF] \
-DDEAL_II_DIR=<path> \
-DDEAL_II_SOURCE_DIR=<path> \ # Only required when tests or benchmarks are enabled
-DDOXYGEN_EXECUTABLE=<path_to_doxygen> \ # Only required when documentation is built
-DCLANGFORMAT=[ON/OFF] \
-DCLANGFORMAT_EXECUTABLE=<path_to_clang-format> \ # Only required when code formatting is quired
<path_to_weak_forms_source>
```

To date, this library has only been built and tested on `MacOS` with the `Clang`
compiler. In the future, the `GCC` compiler and the `Linux` operating system will
be tested as well.

## Dependency recommendations
-----------------------------
- If you plan on using multi-threading and the `SD` framework provided by one of
  the self-linearising functors, then SymEngine (a `deal.II` dependency) should
  be built with the option `-DWITH_SYMENGINE_THREAD_SAFE:BOOL=ON`.


# Citing the library
--------------------
This library has been created for the author's enjoyment, so no direct citation
is necessary, thanks. Since this library acts as a convenience wrapper around
data structures and algorithms implemented in the `deal.II` library, a citation
of the latest release paper, as well as the design paper, would be appreciated.
```bibtex
@article{dealII93,
  title     = {The \texttt{deal.II} Library, Version 9.3},
  author    = {Daniel Arndt and Wolfgang Bangerth and Bruno Blais and
               Marc Fehling and Rene Gassm{\"o}ller and Timo Heister
               and Luca Heltai and Uwe K{\"o}cher and Martin
               Kronbichler and Matthias Maier and Peter Munch and
               Jean-Paul Pelteret and Sebastian Proell and Konrad
               Simon and Bruno Turcksin and David Wells and Jiaqi
               Zhang},
  journal   = {Journal of Numerical Mathematics},
  year      = {2021, accepted for publication},
  url       = {https://dealii.org/deal93-preprint.pdf},
  doi       = {10.1515/jnma-2021-0081},
  volume    = {29},
  number    = {3},
  pages     = {171--186}
}

@article{dealII2019design,
  title   = {The {deal.II} finite element library: Design, features,
             and insights},
  author  = {Daniel Arndt and Wolfgang Bangerth and Denis Davydov and
             Timo Heister and Luca Heltai and Martin Kronbichler and
             Matthias Maier and Jean-Paul Pelteret and Bruno Turcksin and
             David Wells},
  journal = {Computers \& Mathematics with Applications},
  year    = {2021},
  DOI     = {10.1016/j.camwa.2020.02.022},
  pages   = {407-422},
  volume  = {81},
  issn    = {0898-1221},
  url     = {https://arxiv.org/abs/1910.13247}
}
```
As the matrix-based implementation leverages some awesome concepts and data
structures in the `deal.II` library, it would be appropriate to cite the
authors of those specific areas of work.
- `MeshWorker::mesh_loop()`, built on top of the `WorkStream` pattern, for
  efficient multithreading
  ```bibtex
  @article{Turcksin2016,
    author    = {B. Turcksin and M. Kronbichler and W. Bangerth},
    title     = {WorkStream -- a design pattern for multicore-enabled finite
                 element computations},
    journal   = {ACM Transactions on Mathematical Software, pp. 2/1-29},
    year      = {2016},
    volume    = {43}
  }
  ```
- `VectorizedArray` and `AlignedVector` for `SIMD` vectorisation
  ```bibtex
  @article{Kronbichler2012,
    author = {Martin Kronbichler and Katharina Kormann},
    title = {A generic interface for parallel cell-based finite element
             operator application},
    journal = {Computers {\&} Fluids},
    doi = {10.1016/j.compfluid.2012.04.012},
    url = {https://doi.org/10.1016/j.compfluid.2012.04.012},
    year = {2012},
    month = jun,
    publisher = {Elsevier {BV}},
    volume = {63},
    pages = {135--147}
  }
  ```
- `MeshWorker::ScratchData` and `GeneralDataStorage` for generic finite element
   operations and data storage / extraction.
  ```bibtex
  @article{Sartori2018,
    Author  = {Sartori, Alberto and Giuliani, Nicola and
               Bardelloni, Mauro and Heltai, Luca},
    Journal = {SoftwareX},
    Pages   = {318--327},
    Title   = {{deal2lkit: A toolkit library for high performance
              programming in deal.II}},
    Doi     = {10.1016/j.softx.2018.09.004},
    Volume  = {7},
    Year    = {2018}
  }
  ```


# Similar projects that inspired this work
------------------------------------------
- `deal.II`
  - [CFL form language for deal.II](https://github.com/masterleinad/CFL) by Daniel Arndt and Guido Kanschat
- Other finite element and finite volume codes
  - [FEniCS](https://fenicsproject.org/): [Unified Form Language](https://github.com/FEniCS/ufl)
  - [NGSolve](https://ngsolve.org/): [Symbolic Integrators](https://docu.ngsolve.org/latest/how_to/symbolic_integrators.html)
  - [OpenFOAM](https://openfoam.com/): [Equation representation](https://cfd.direct/openfoam/user-guide/v6-programming-language-openfoam/)
- Other codes that use expression templates
  - [Sacado](https://trilinos.github.io/sacado.html): [Automatic differentiation using operator overloading](https://github.com/trilinos/Trilinos/tree/master/packages/sacado) 


# Acknowledgements
------------------
- The LaTeX output for the various examples was rendered using the [Interactive LaTeX Editor](https://arachnoid.com/latex/).


# Contributing
--------------
Please read the contributing documentation [here](contributing.md).


# License
---------
This project is licensed under the GNU Lesser General Public License v3.0.
For more information, see the `LICENSE.md` and `COPYING.LESSER` files.

    Weak forms for deal.II: An implementation of a symbolic weak form interface
    for deal.II, using expression templates as well as automatic and symbolic
    differentiation. 

    Copyright (C) 2021 - 2022  Jean-Paul Pelteret

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.