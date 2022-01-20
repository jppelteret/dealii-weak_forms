# Class documentation
---------------------

**Note**: This is an intermediate solution to the library documentation, while
doxygen documentation is in preparation.

## Functors
- User-defined (spatially dependent)
  - `ScalarFunctor`: Scalar function
  - `TensorFunctor`: Tensor function
  - `SymmetricTensorFunctor`: Symmetric tensor function
- User-defined (with caching)
  - `ScalarCacheFunctor`: Scalar function
  - `TensorCacheFunctor`: Tensor function
  - `SymmetricTensorCacheFunctor`: Symmetric tensor function
- Wrappers for `deal.II` `Function`s
  - `ScalarFunctionFunctor`: Scalar function, FunctionParser
  - `TensorFunctionFunctor`: Tensor function, TensorFunctionParser
- Conversion utilities (local to symbolic)
  - `constant_scalar()`: Constant scalar function
  - `constant_vector()`: Constant vector function
  - `constant_tensor()`: Constant tensor function
  - `constant_symmetric_tensor()`: Constant symmetric tensor function
- [TODO] No-op


## Spaces
- `TestFunction`: Global test function
- `TrialSolution`: Global trial solution
- `FieldSolution`: Global field solution
  - Indexed storage for time history or other solution fields via the `SolutionStorage` class
  - Indexed storage also supports multiple DoFHandlers (e.g. when a field solution for another discretisation is used in the RHS of the one being assembled)
- Sub-space extractors, used to segment a finite element space into
  sub-components
  - `SubSpaceExtractors::Scalar`
  - `SubSpaceExtractors::Vector`
  - `SubSpaceExtractors::Tensor`
  - `SubSpaceExtractors::SymmetricTensor`
- Sub-space views (extracted from a space), accessing one or more components of
  the finite element space. With a sub-space view, the natural "type" of the
  finite element component is known, and more operators become available for
  use.
  - `SubSpaceViews::Scalar`
  - `SubSpaceViews::Vector`
  - `SubSpaceViews::Tensor`
  - `SubSpaceViews::SymmetricTensor`


## Forms
- Standard
  - `LinearForm`: 
    A class that encapsulates a linear form, composed of a test space operator
    and an arbitrary functor.
    - Convenience function: `linear_form()`
  - `BilinearForm`:
    A class that encapsulates a bilinear form, composed of a test space operator,
    an arbitrary functor, and a trial space operator.
    - Can set a symmetry flag for local contributions
    - Convenience function: `bilinear_form()`
  - Feature points
    - Form operators involve slots for per-dof calculations and per-quadrature point calculations
      - Can use the per-quadrature point slot as much as possible to minimise number of operations
    - Test function and trial solution may be a composite operation
      - Note: In this case, the composite operation may incur n(dofs)*n(q-points) operations
- Self-linearising
  - Accumulation of these forms into an assembler will automatically generate
    additional forms for the linearisation (and, in the case of an energy
    functional, the residual as well). The parameterisation (dictating how many
    additional forms are generated) are automatically deduced.
  - `SelfLinearization::EnergyFunctional`:
    A self-linearising energy functional (as is often defined for variational 
    problems)
    - Feature points
      - Convenience function: `energy_functional_form()`
      - Consumes an `EnergyFunctor`
      - Parameterisation is defined by the energy functor
      - Variation and linearisation with respect to all field variables
    - `EnergyFunctor`: Energy functional
      Functors to parameterise and define self-linearisation finite element
      residuals (or a component of the residual)
      - Convenience function: `energy_functor()`
      - Automatic differentiation stored energy function
      - Symbolic differentiation free/stored energy function
  - `SelfLinearization::ResidualView`:
    A self-linearising energy functional (as is often defined for variational 
    problems)
    - Feature points
      - Convenience function: `residual_form()`
      - Consumes an `ResidualFunctor` or a `ResidualViewFunctor`
      - Finite element component selection using the designated test function
      - Test function may be a composite operation (but will **not** be
        linearised)
      - Parameterisation is defined by the residual functor
      - Linearisation with respect to all field variables
    - `ResidualFunctor`, `ResidualViewFunctor`:
      Functors to parameterise and define self-linearisation finite element
      residuals (or a component of the residual)
      - Convenience functions: `residual_functor()`, `residual_view_functor()`
      - Automatic differentiation function for kinetic variable
      - Symbolic differentiation function for kinetic variable
  - `AD_SD_Functor_Cache`:
    A class that provides a caching mechanism for `SD` calculations.
    Using this class, the results of symbolic operations (e.g. symbolic
    differentiation) can persist across timesteps and Newton iterations, rather
    than being recomputed each time assembly is invoked.
- Forms are integrated over volumes, boundaries and interfaces using the 
  `dV()`, `dA()` and `dI()` class member functions.


## Operators
- Symbolic test functions/trial solutions/field solutions
  - Global `TestFunction`, `TrialSolution`, `FieldSolution`
    - sub-space extraction
    - `value()`: value
    - `gradient()`: gradient
    - `laplacian()`: laplacian
    - `hessian()`: Hessian
    - `third_derivative()`: third derivative
    - `jump_in_values()`: jump in values
    - `jump_in_gradients()`: jump in gradients
    - `jump_in_hessians()`: jump in Hessians
    - `jump_in_third_derivatives()`: jump in third derivatives
    - `average_of_values()`: average of values
    - `average_of_gradients()`: average of gradients
    - `average_of_hessians()`: average of Hessians
  - Scalar (`SubSpaceViews::Scalar` generated by a 
    `TestFunction[SubSpaceExtractors::Scalar]`,
    `TrialSolution[SubSpaceExtractors::Scalar]`, or a
    `FieldSolution[SubSpaceExtractors::Scalar]`)
    - `value()`: value
    - `gradient()`: gradient
    - `laplacian()`: laplacian
    - `hessian()`: Hessian
    - `third_derivative()`: third derivative
    - `jump_in_values()`: jump in values
    - `jump_in_gradients()`: jump in gradients
    - `jump_in_hessians()`: jump in Hessians
    - `jump_in_third_derivatives()`: jump in third derivatives
    - `average_of_values()`: average of values
    - `average_of_gradients()`: average of gradients
    - `average_of_hessians()`: average of Hessians
  - Vector (`SubSpaceViews::Vector` generated by a 
    `TestFunction[SubSpaceExtractors::Vector]`,
    `TrialSolution[SubSpaceExtractors::Vector]`, or a
    `FieldSolution[SubSpaceExtractors::Vector]`)
    - `value()`: value
    - `gradient()`: gradient
    - `symmetric_gradient()`: symmetric gradient
    - `divergence()`: divergence
    - `curl()`: curl
    - `hessian()`: Hessian
    - `third_derivative()`: third derivative
    - `jump_in_values()`: jump in values
    - `jump_in_gradients()`: jump in gradients
    - `jump_in_hessians()`: jump in Hessians
    - `jump_in_third_derivatives()`: jump in third derivatives
    - `average_of_values()`: average of values
    - `average_of_gradients()`: average of gradients
    - `average_of_hessians()`: average of Hessians
  - Tensor (`SubSpaceViews::Tensor` generated by a 
    `TestFunction[SubSpaceExtractors::Tensor]`,
    `TrialSolution[SubSpaceExtractors::Tensor]`, or a
    `FieldSolution[SubSpaceExtractors::Tensor]`)
    - `value()`: value
    - `gradient()`: gradient
    - `divergence()`: divergence
  - Symmetric tensor (`SubSpaceViews::SymmetricTensor` generated by a 
    `TestFunction[SubSpaceExtractors::SymmetricTensor]`,
    `TrialSolution[SubSpaceExtractors::SymmetricTensor]`, or a
    `FieldSolution[SubSpaceExtractors::SymmetricTensor]`)
    - `value()`: value
    - `divergence()`: divergence
- Function operators
  - Unary
    - General
      -  `operator-`: negation
    -  Scalar operations
       - `sin()`, `cos()`, `tan()`: trignometric operations: sine, cosine, tangent
       - `exp()`, `log()`:`exponential, logarithm
       - `sqrt()`:square root
       - `abs()`:absolute value
       - [TODO] Other math functions
    - Tensor operations
      - `determinant()`: determinant
      - `invert()`: invert
      - `transpose()`: transpose
      - `symmetrize()`: symmetrize
    - Interface operations (evaluating a function across an interface)
      - [TODO] jump
      - [TODO] average 
  - Binary
    - `operator+`: Addition
    - `operator-`: Subtraction
    - `operator*`: Multiplication
      - Scalar
      - Tensor
    -  Scalar operations
       - `pow()`: power
       - `max()`, `min()`: maximum, minimum
       - [TODO] Other math functions
     - Tensor operations
       - `cross_product()`: cross product
       - `schur_product()`: Schur product
       - `outer_product`: outer product
     - Tensor contractions
       - `operator*` (single contraction for `Tensor`s, double contraction for `SymmetricTensor`s)
       - `scalar_product()`: scalar product
       - `contract()`: single index contraction
       - `double_contract`: double index double contraction
       - [TODO] general contraction
  - Implicit conversions to functors
    - Arithmetic types, e.g. `double` -> `constant_scalar()`
    - Tensor, e.g. `Tensor` -> `constant_tensor()`
    - SymmetricTensor, e.g. `SymmetricTensor` -> `constant_symmetric_tensor()`
  - Operation modes
    - Quadrature point
    - Shape function @ quadrature point (binary operation with test function / trial solution)
- Form operators
  - Unary
    - `operator-`: negation
  - Binary
    - `operator+`: Addition
    - `operator-`: Subtraction
    - `operator*`: Multiplication
      - Scalar


## Integration
- Integration domains
  - `VolumeIntegral`: A class representing volume integrals
    - Subdomain selection: Material ID (`dealii::types::material_id`)
  - `BoundaryIntegral`: A class representing boundary integrals
    - Subdomain selection: Boundary ID (`dealii::types::boundary_id`)
  - `Interfacentegral`: A class representing interface integrals
    - Inter-cell interfaces for DG FEM
    - Subdomain selection: Manifold ID (`dealii::types::manifold_id`)
  - [TODO] Custom predicates for the above
- `FunctionIntegrator`: User-defined function integrators, used to compute integrals
  of quantities over a domain or subdomain.
  - Position independent/dependent
  - Volume, boundary, interface (using `MeshWorker::mesh_loop()`)
- `Integrator`: Integrator for symbolic functors, used to compute integrals
  of quantities over a domain or subdomain.
  - Volume, boundary, interface (using `MeshWorker::mesh_loop()`)
- Integral operators
  - Binary
    - `operator+`: Addition
    - `operator-`: Subtraction
    - `operator*`: Multiplication
      - Scalar


## Assemblers
- `MatrixBasedAssembler`:
  Matrix-based assembly (using `MeshWorker::mesh_loop()` for multithreading as
  well as `SIMD`)
  - Symmetry flag for global system
    - Exclusion of bilinear form contributions based on field index
  - Ignore DoFs that aren't in DoF group
  - Vectorisation if `AVX` extensions are available
  - Pre-computation and result caching
- [TODO] Matrix-free

## Output
- `SymbolicDecorations`: A (partially customisable) symbolic decorator that
  is used to provide some nomenclature when the expression tree is parsed during
  output
- ASCII using the `as_ascii()` class member functions
- LaTeX using the `as_latex()` class member functions