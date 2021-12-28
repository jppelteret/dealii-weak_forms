# Weak forms for deal.II
------------------------
An implementation of a symbolic weak form interface for deal.II, using 
expression templates as well as automatic and symbolic differentiation. 

Author: Jean-Paul Pelteret, 2020 - 2021


# Features
----------

## Functors
- User-defined
  - Scalar function
  - Tensor function
  - Symmetric tensor function
- User-defined (with caching)
  - Scalar function
  - Tensor function
  - Symmetric tensor function
- Wrappers for deal.II `Function`s
  - Scalar function, FunctionParser
  - Tensor function, TensorFunctionParser
- Conversion utilities (local to symbolic)
  - Constant scalar function
  - Constant tensor function
  - Constant symmetric tensor function
- [TODO] No-op


## Spaces
- Test function
- Trial solution
- Field solution
  - Indexed storage for time history or other solution fields 
- Sub-space extractors and views


## Forms
- Standard
  - Linear
  - Bilinear
    - Symmetry flag for local contributions
  - Feature points
    - Form operators involve slots for per-dof calculations and per-quadrature point calculations
      - Can use the per-quadrature point slot as much as possible to minimise number of operations
    - Test function and trial solution may be a composite operation
      - Note: In this case, the composite operation may incur n(dofs)*n(q-points) operations
- Self-linearising
  - Energy functional
    - Automatic differentiation stored energy function
    - Symbolic differentiation free/stored energy function
    - Feature points
      - Variation and linearisation with respect to all field variables
  - Residual
    - Automatic differentiation function for kinetic variable
    - Symbolic differentiation function for kinetic variable
    - Feature points
      - Component selection using test function
      - Test function may be a composite operation (but will not be linearised)
      - Linearisation with respect to all field variables


## Operators
- Symbolic test functions/trial solutions/field solutions
  - Scalar
    - value
    - gradient
    - laplacian
    - Hessian
    - third derivative
    - jump in values
    - jump in gradients
    - jump in Hessians
    - jump in third derivatives
    - average of values
    - average of gradients
    - average of Hessians
  - Vector
    - value
    - gradient
    - symmetric_gradient
    - divergence
    - curl
    - Hessian
    - third derivative
    - jump in values
    - jump in gradients
    - jump in Hessians
    - jump in third derivatives
    - average of values
    - average of gradients
    - average of Hessians
  - Tensor
    - value
    - gradient
    - divergence 
  - Symmetric tensor
    - value
    - divergence
- Function operators
  - Unary
    - General
      -  negation
    -  Scalar operations
       - trignometric operations: sine, cosine, tangent
       - exponential, logarithm
       - square root
       - absolute value
       - [TODO] Other math functions
    - Tensor operations
      - determinant
      - invert
      - transpose
      - symmetrize
    - Tensor contractions
      - Operator * (single contraction for Tensor, double contraction for SymmetricTensor)
      - [TODO] scalar product
      - [TODO] double contract
      - [TODO] general contraction
    - Interface operations (evaluating a function across an interface)
      - [TODO] jump
      - [TODO] average 
  - Binary
    - Addition
    - Subtraction
    - Multiplication
      - Scalar
      - Tensor
    -  Scalar operations
       - power
       - maximum, minimum
       - [TODO] Other math functions
     - Tensor operations
       - Scalar product
       - Cross product
       - Schur product
       - [TODO] Contract
       - [TODO] Double contract
       - Outer product
  - Implicit conversions to functors
    - Arithmetic types
    - Tensor
    - SymmetricTensor
  - Operation modes
    - Quadrature point
    - Shape function @ quadrature point (binary operation with test function / trial solution)
- Form operators
  - Unary
    - negation
  - Binary
    - Addition
    - Subtraction
    - Multiplication
      - [TODO] Scalar


## Integration
- Integration domains
  - Volume
    - Subdomains: Material ID
  - Boundary
    - Subdomains: Boundary ID
  - Interface
    - Subdomains: Manifold ID
    - Inter-cell interfaces for DG
  - [TODO] Custom predicates for the above
- User-defined function integrators
  - Position independent/dependent
  - Volume, boundary, interface (using `mesh_loop`)


## Assemblers
- Matrix-based (using `mesh_loop`)
  - Symmetry flag for global system
    - Exclusion of bilinear form contributions based on field index
  - Ignore DoFs that aren't in DoF group
  - Vectorisation
  - Pre-computation and result caching
- [TODO] Matrix-free

## Output
- Symbolic decorator (customisable)
- ASCII
- LaTeX
  
# Examples
Some examples and output can be found [here](doc/readme/examples.md).
  
# Benchmarks
The results of some preliminary benchmarks can be found [here](doc/readme/benchmarks.md).

# Similar projects that inspired this work
- deal.II
  - [CFL form language for deal.II](https://github.com/masterleinad/CFL) by Daniel Arndt and Guido Kanschat
- Other finite element and finite volume codes
  - [FEniCS](https://fenicsproject.org/): [Unified Form Language](https://github.com/FEniCS/ufl)
  - [NGSolve](https://ngsolve.org/): [Symbolic Integrators](https://docu.ngsolve.org/latest/how_to/symbolic_integrators.html)
  - [OpenFOAM](https://openfoam.com/): [Equation representation](https://cfd.direct/openfoam/user-guide/v6-programming-language-openfoam/)


# Acknowledgements
- The LaTex output for the various examples was rendered using the [Interactive LaTeX Editor](https://arachnoid.com/latex/).


# License
---------
This project is licensed under the GNU Lesser General Public License v3.0.
For more information, see the `LICENSE.md` and `COPYING.LESSER` files.

    Weak forms for deal.II: An implementation of a symbolic weak form interface
    for deal.II, using expression templates as well as automatic and symbolic
    differentiation. 

    Copyright (C) 2021  Jean-Paul Pelteret

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