# Weak forms for deal.II
------------------------
An implementation of a symbolic weak form interface for `deal.II`, using 
expression templates as well as automatic and symbolic differentiation. 

Author: Jean-Paul Pelteret, 2020 - 2022

# Concept
---------
The idea for this library is to offer an abstraction for the discretisation
of finite element weak forms using the `deal.II` open source finite element
library.

What does this mean? Well, instead of writing an assembly loop to assemble into
a matrix and vector
```c++
const double coefficient = 1.0;
const double f           = 1.0;

...
for (const auto &cell : dof_handler.active_cell_iterators())
{
  ...
  for (const unsigned int i : fe_values.dof_indices())
  {
    for (const unsigned int j : fe_values.dof_indices())
      cell_matrix(i, j) +=
                (coefficient *                      // a(x_q)
                 fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                 fe_values.JxW(q_index));

    cell_rhs(i) += (f *                                 // f(x)
                    fe_values.shape_value(i, q_index) * // phi_i(x_q)
                    fe_values.JxW(q_index));            // dx
  }

  constraints.distribute_local_to_global(cell_matrix,
                                         cell_rhs,
                                         local_dof_indices,
                                         system_matrix,
                                         system_rhs);
}
```
with this library you can do it expressively
```c++
  const TestFunction<dim>  test;
  const TrialSolution<dim> trial;
  const ScalarFunctor      mat_coeff("coefficient", "c");
  const ScalarFunctor      rhs_coeff("f", "f");

  const auto mat_coeff_func = mat_coeff.template value<double, dim, spacedim>(
    [](const FEValuesBase<dim, spacedim> &, const unsigned int)
    { return 1.0; });
  const auto rhs_coeff_func = rhs_coeff.template value<double, dim, spacedim>(
    [](const FEValuesBase<dim, spacedim> &, const unsigned int)
    { return 1.0; });

  MatrixBasedAssembler<dim> assembler;

  // LHS contribution: a(grad phi_i, c, grad phi_j)
  assembler += bilinear_form(test.gradient(), 
                             mat_coeff_func,
                             trial.gradient()).dV();
  // RHS contribution: a(phi_i, f)    
  assembler -= linear_form(test.value(), rhs_coeff_func).dV();

  //
  assembler.assemble_system(system_matrix,
                            system_rhs,
                            constraints,
                            dof_handler,
                            qf_cell);
```

Let's identify the key differences between these two paradigms:
- `native deal.II`: 
  
   When writing an assembly loop, one "sees" the weak form at the lowest level,
   i.e. the fully discretised, indexed entries for each shape function from
   the test space and trial solution spaces, as well as field solutions computed
   at quadrature points and the like. One has to know the discretisation
   (i.e. the assembly function has to have access to a specific `DofHandler`,
   `FiniteElement`, `SparseMatrix`, right-hand-side `Vector`, etc) as context 
   (indeed, such assembly functions can be made generic, but this must be
   done by the implementer at some effort -- only something that advanced
   users of the `deal.II` library might consider doing). 

   When extending or modifying the weak form, the user might have to employ
   some effort to cache data (e.g. when using the `Function` classes).
   The speed and correctness of implementation of an assembly loop are 
   subject to the implementer's skill and expertise. It is difficult
   to write an assembly loop that is both readable and highly performance
   orientated. The naive, explicit assembly algorithm becomes even more
   "interesting" when interface terms are introduced, and mesh adaptivity
   makes things even more tricky.

   On the matter of correctness, when dealing with complex constitutive laws
   the automatic and symbolic differentiation capabilities built into `deal.II`
   might be useful, but are another aspect of the library that the user must
   become very familiar with before they can use them *correctly* and
   *effectively*.

- `Symbolic weak form library`:
  
   In contrast, the `weak form` library allows one to write the weak form
   symbolically (perhaps even more "expressively", even closer to the (bi)linear
   notation that one might find in an academic paper). The notion of "assembly"
   is taken care by the library, and only when requested. Each of the forms that
   are (compile-time) generated are, in fact, patterns for assembly (also known
   as integration kernels), and while these kernels describe how to perform the
   numerical computations, they simultaneously retain some text representation
   of their action. So, if you like, you don't even need to do any assembly
   -- you could use this library, sans `Triangulation` or any other complexity,
   to first to assess an implementation though ASCII or LaTeX output before
   doing anything computational with it. Or, if you have two formulations that
   are identical save for a few terms, then perhaps you could implement the
   one and then later add those few extra bilinear/linear forms to fully
   implement the second formulation (as you would if you passed a matrix-vector
   system through two assembly functions). 

   When you do pass the `assembler` some concrete classes to invoke assembly,
   it's action depends on the information passed to it. One `assembler` could
   form the full linear system, or just the system matrix or RHS vector.
   Do you want to assemble using a different `DoFHandler`? No problem -- the
   `assembler`, which has the patterns for assembly of the generic weak form
   that it encapsulates, just repeats the actions with different input data.
   At the end of the day, this part of the library provides "syntactic sugar"
   (i.e. convenience) to the concept of linear system assembly and its
   generalisation. 

   Modifying a weak form in this library might take a few lines less than
   `deal.II` itself, and whatever you do remains a generalisation due to the
   patterning concept. All data structure initialisation and data caching
   is performed on your behalf, so one doesn't need to think about how to
   extract data from the `deal.II` classes and data structures; this is done
   for you *on demand* (lazily), and never unnecessarily.

   The core of the assembly process has been rigourously tested for
   correctness -- the worst thing that a library like this can do is take away
   the low-level functionality from a user and then introduce a bug that
   invalidates the entire thing that the library is designed to do.
   Although no library is bug-free, it is the author's hope that users of this
   work find it to reliably compute what the user has prescribed.

   As an implementational detail, an assembly loop is (currently) performed
   for each bilinear and linear form individually. This is not ideal when
   their are multiple forms contributing to the linear system. However,
   as the implementation of the assembly process is opaque to the user,
   the library is able to perform several optimisations to limit the extent
   to which this impacts the overall assembly time. The vectorisation capabilities
   of modern computers can be exploited to do `SIMD` parallelisation on top
   of the multi-threading which is built in to `deal.II`'s `WorkStream::mesh_loop()`
   concept (and whatever distributed computation the user might be doing using 
   `MPI`). For the classes that are not parallel-friendly, their data is
   extracted as early as possible into parallel data structures, to that
   parallelisation can be used in as many operations as possible. Using
   `WorkStream::mesh_loop()` means that we can offer (some) functionality for
   DG finite elements and other methods that introduce interface terms.

   On the matter of supporting complex physics models, this library offers
   a *further* abstraction to the automatic and symbolic differentiation
   capabilities of `deal.II`. Due to the symbolic nature of this library,
   special `energy_functional` and `residual_view` forms have been implemented
   that are *self linearising*; that is to say, that they understand their
   parameterisation and can generate new forms that encapsulate the linearisation
   of the residual, or the generation of that residual in the first place.
   The use of `AD` or `SD` is then restricted to quadrature-point level
   calculations, which is the point at which they are most efficiently employed.
   The hope is that the current abstraction to `AD` or `SD` allows the user to
   implement some complex constitutive laws without getting to understand all
   of the details of those frameworks.

   Some other interesting features of the library include the mimicry of `deal.II`
   functions for scalars and tensors, so that this library can be most naturally
   used by people who are familiar with the syntax of the `deal.II` linear
   algebra classes. Naturally, integrals can be restricted to specific subdomains,
   boundaries or interfaces. There is a class that wraps solution histories (even
   those tied to other `DoFHandlers`) so, as examples, time discretisation of
   rate-dependent problems is supported, and the solution of one finite
   element problem can be used as the input to another. More features of the
   library are loosly listed below.

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
- Wrappers for `deal.II` `Function`s
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
  - Indexed storage also supports multiple DoFHandlers (e.g. when a field solution for another discretisation is used in the RHS of the one being assembled)
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
       - cross product
       - Schur product
       - outer product
     - Tensor contractions
       - operator * (single contraction for Tensor, double contraction for SymmetricTensor)
       - scalar product
       - contract
       - double contract
       - [TODO] general contraction
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
      - Scalar


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
  - Volume, boundary, interface (using `WorkStream::mesh_loop()`)
- Integral operators
  - Binary
    - Addition
    - Subtraction
    - Multiplication
      - Scalar


## Assemblers
- Matrix-based (using `WorkStream::mesh_loop()`)
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
- The LaTex output for the various examples was rendered using the [Interactive LaTeX Editor](https://arachnoid.com/latex/).

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