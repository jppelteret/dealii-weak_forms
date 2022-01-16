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
  // Initialise some data structures, precompute some data for use in
  // the assembly loop, ...

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
with this library you can do it expressively without having to worry about
the structure of the assembly loop, data initialisation or extraction, and
some of the other details
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

  // Take the generic forms, and perform some concrete
  // computations using the given data structures.
  assembler.assemble_system(system_matrix,
                            system_rhs,
                            constraints,
                            dof_handler,
                            qf_cell);
```

If analysing code to debug is a chore, then this library can also spell out,
mathematically, what its going to compute:
- in ACSII format
  ```c++
  const SymbolicDecorations decorator;
  std::cout << assembler.as_ascii(decorator) << std::endl;
  ```
  ```
  0 = #(Grad(d{U}), coefficient, Grad(D{U}))#dV - #(d{U}, f)#dV
  ```
- or in LaTeX format that can be readily 
  [rendered online](https://arachnoid.com/latex/?equ=0%20%3D%20%5Cint%5Cleft%5B%5Cnabla%5Cleft(%5Cdelta%7B%5Cvarphi%7D%5Cright)%20%5Ccdot%20%5Cleft%5B%7Bc%7D%5C%2C%5Cnabla%5Cleft(%5CDelta%7B%5Cvarphi%7D%5Cright)%5Cright%5D%5Cright%5D%5Ctextrm%7BdV%7D%20%0A-%20%5Cint%5Cleft%5B%5Cdelta%7B%5Cvarphi%7D%5C%2C%7Bf%7D%5Cright%5D%5Ctextrm%7BdV%7D%0A)
  or using a minimal number of added packages
  <img src="./doc/readme/images/intro-latex_output-example.png" width="600">
  ```c++
  const SymbolicDecorations decorator;
  std::cout << assembler.as_latex(decorator) << std::endl;
  ```
  ```latex
  0 = \int\left[\nabla\left(\delta{\varphi}\right) \cdot \left[{c}\,\nabla\left(\Delta{\varphi}\right)\right]\right]\textrm{dV} 
  - \int\left[\delta{\varphi}\,{f}\right]\textrm{dV}
  ```

## How is this different from the standard deal.II functionality?

Let's identify the key differences between these two paradigms:
- `native deal.II` (considering the approach typically advocated and
  adopted for "matrix-based" methods): 
  
   When writing an assembly loop, one "sees" the weak form at the lowest level,
   i.e. the fully discretised, indexed entries for each shape function from
   the test space and trial solution spaces, as well as field solutions computed
   at quadrature points and the like. One has to know the discretisation
   (i.e. the assembly function has to have access to a specific `DofHandler`,
   `FiniteElement`, `SparseMatrix`, right-hand-side `Vector`, etc) as context.
   Indeed, such assembly functions can be made generic, but this must be
   done by the implementer at some effort -- only something that advanced
   users of the `deal.II` library might consider doing. This also means that
   the assembly functions might need addition arguments, which might 
   complicate the extension of multi-physics frameworks that use polymorphism
   as part of their design abstraction. None of these (potential) issues are
   insurmountable; they only make the life of the programmer a little more
   challenging.

   When extending or modifying the weak form, the user might have to employ
   some effort to cache data (e.g. when using the `Function` classes).
   The speed and correctness of implementation of an assembly loop are 
   subject to the implementer's skill and expertise. It is difficult
   to write an assembly loop that is both readable and highly performance
   orientated. The naive, explicit assembly algorithm becomes even more
   "interesting" when interface terms are introduced, and mesh adaptivity
   makes things even more tricky. The `MeshWorker::mesh_loop()` concept
   partnered with the `MeshWorker::ScratchData` and `GeneralDataStorage`
   classes can offer a lot of assistance and greatly simplify things here,
   but their use still adds complexity to (and increases the code length of) 
   the assembly method.

   On the matter of correctness, subtle errors related to indexing in the
   assembly loops can be introduced very easily, and sometimes take some
   time to spot. Adding more complexity to the weak form being implemented
   introduces more opportunity for mistakes. Compounding this are the
   formulation of quantities derived from (nonlinear or otherwise complex)
   constitutive laws, and the subsequent consistent linearisation of those
   quantities. To assist in this regard, the automatic and symbolic
   differentiation capabilities built into `deal.II` might be useful, but are
   another aspect of the library that the user must become very familiar with
   before they can use them *correctly* and *effectively*.

- `Symbolic weak form library`:
  
   In contrast, the `weak form` library allows one to write the weak form
   symbolically at a high level. Although only a matter of opinion, perhaps this
   is more expressive than the alternative, as it may be closer to the (bi)linear
   notation that one might find in an academic paper. The notion of "assembly"
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
   of the multi-threading which is built in to `deal.II`'s `MeshWorker::mesh_loop()`
   concept (and whatever distributed computation the user might be doing using 
   `MPI`). For the classes that are not parallel-friendly, their data is
   extracted as early as possible into parallel data structures, to that
   parallelisation can be used in as many operations as possible. Using
   `MeshWorker::mesh_loop()` means that we can offer (some) functionality for
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
   algebra classes. By means of an example, this would be a valid (albeit 
   non-sensical) bilinear form:
   ```c++
    const TestFunction<dim>          test;
    const TrialSolution<dim>         trial;
    const FieldSolution<dim>         field_solution;
    const SubSpaceExtractors::Vector subspace_extractor_v(0,
                                                          "v",
                                                          "\\mathbf{v}");
    const SubSpaceExtractors::Scalar subspace_extractor_p(dim,
                                                          "p_tilde",
                                                          "\\tilde{p}");

    const auto div_test_v   = test[subspace_extractor_v].divergence();
    const auto grad_trial_p = trial[subspace_extractor_p].gradient();
    const auto v            = field_solution[subspace_extractor_v].value();
    const auto grad_v       = field_solution[subspace_extractor_v].gradient();
    const auto hessian_p    = field_solution[subspace_extractor_p].hessian();

    const auto I = dealii::unit_symmetric_tensor<dim>();

    const auto form = 2.0 * bilinear_form(div_test_v * I,            // rank-2
                                          outer_product(v,grad_v),   // rank-3
                                          hessian_p * grad_trial_p); // rank-1
   ```
   
   Note that in the (terrible) example above, we generated the form without
   specifying the integration domain -- these two actions are orthogonal,
   and one form can be integrated in multiple contexts.
   Naturally, integrals can be restricted to specific subdomains,
   boundaries or interfaces. There is a class that wraps solution histories (even
   those tied to other `DoFHandler`s) so, as examples, time discretisation of
   rate-dependent problems is supported, and the solution of one finite
   element problem can be used as the input to another. More features of the
   library are loosely listed below.

## How does this work?

The design paradigm adopted to implement this library is called expression
templating. The library provides some key elements for which their purpose
(or, in this case, action) is well defined. When these elements are used as
part of a composite operation then, assuming the operation is well defined,
the compiler will generate a new class type that is specifically constructed
to perform the desired operation on or between the input elements. Since the
library cannot predict the generated class type (and this is something that
the user should not really be interested in), the `auto` type deduction keyword
is extensively used where these symbolic operators are returned.

Examples of the core elements would be test functions, trial solutions, field
solutions (denoting the values or derivatives of the finite element solution),
so called "views" of the aforementioned types, as well as scalar or tensor
constants, user-defined functions, and the like. Associated with each of these
is a native data type for mathematical construct that it represents. For instance,
the *gradient* of a *vector view* of a *test function* will be a tensor of
rank 2 -- this "result type" is embedded into the generated class.

Composite expressions of these fundamental elements can be formed using many 
standard scalar and tensor mathematical operations. These take the form of
unary and binary operations; each time such an operation is required, a bespoke
class type that understands the *action* of the operation based on the supplied
input operators (acting as an argument or arguments to the function) is
generated by the compiler based on the input arguments (which, themselves may
by composite expressions). The collection of expressions may then be consumed
by a `linear_form()` or a `bilinear_form()`. Linear forms take a test
function (or a composite operation involving a test function) as its first
argument, and another expression as its second argument. Similarly, bilinear
forms take (composite) test function expressions as their first argument, a
general expression as the second argument, and a (composite) trial solution
expression as their third argument. The library checks the type and rank of
object that each expression returns, and ensures that it is compatible with
the desired action. For instance, the two arguments to a linear form (and, 
similarly, all three arguments to a bilinear form) must contract to a scalar
valued result. After that, integration domain for the form is specified and
this provides all of the information required for assembly of the form.

During assembly, the assembler uses the integration domain to understand which
cells or faces to perform evaluation on. On the relevant subset of cells or
faces, it then calls the user-defined operators (e.g. a composite test function)
with some common input arguments. The composition of expressions, also known as
an [*expression tree*](https://en.wikipedia.org/wiki/Binary_expression_tree), 
is then traversed with the output of the leaves acting as input arguments to
the composition (unary and binary math) operators. What results are the various
terms, defined per quadrature point or per DoF index and quadrature point, that
must be contracted to add to the local cell matrix or RHS vector.

## What exactly is the compiler doing?

As was stated previously, the user is not really supposed to be interested in
what the compiler is generating. It is opaque by nature of the programming
paradigm, but this also makes understanding any compile-time error messages
incredibly difficult. So this warrants a brief explanation of the generated
data structures.

The fundamental, human-readable classes implemented in this library include
(but are not limited to):
- `TestFunction`, `TrialSolution`, `FieldSolution`: 
  All relate to finite element fields.
- `SubSpaceExtractors::Scalar`, `SubSpaceExtractors::Vector`,
  `SubSpaceExtractors::Tensor`, `SubSpaceExtractors::SymmetricTensor`:
  Provide a mechanism to produce a *view* of a multi-component finite elemnt
  field.
- `SubSpaceViews::Scalar`, `SubSpaceViews::Vector`, `SubSpaceViews::Tensor`, 
  `SubSpaceViews::SymmetricTensor`:
  Interpret a subset of the finite element field components as a scalar, vector,
  tensor or symmetric tensor type. This permits additional operations being
  performed on the view (e.g. computing the divergence of a vector field is
  permitted, but not a scalar field).
- `ScalarFunctor`, `TensorFunctor`, `SymmetricTensorFunctor`: 
  Are to return a scalar, tensor, or symmetric tensor at a given point in space.
- `LinearForm` and `BiinearForm`: 
  Collect the test function and a per quadrature-point defined functor (and,
  if necessary, the trial solution) that are to be contracted and integrated.

(In the supplied benchmarks and tests, you will see these classes being 
explicitly typed.)

The main (opaque) data classes that this library uses are:
- `SymbolicOp`: Provides a call operator, and gives a meaning or definition
  to some fundamental operation. These act as the *leaves* of the *expression
  tree* that defines any composite operation.
- `UnaryOp`: Implements a mathematical operation by providing a call operator
  which transforms a single input argument to a single result. 
- `BinaryOp`: Implements a mathematical operation by providing a call operator
  which transforms two input arguments to a single result.  

These three symbolic classes perform the data extraction or transformation based
on their input arguments. As the imput arguments are arbitrarily complex, the
generated class has a (template-derived) definition that is similarly arbitrary
complex.

As an illustration of something more concrete, consider the following example:
```c++
const ScalarFunctor sclr("s", "s");
const TestFunction<dim> test;
const SubSpaceExtractors::Vector subspace_extractor_v(0,
                                                      "v",
                                                      "\\mathbf{v}");

const auto s           = sclr.template value<double, dim, spacedim>(
  [](const FEValuesBase<dim, spacedim> &, const unsigned int)
  { return 2.0; });                                                   // (1)
const auto grad_test_v = test[subspace_extractor_v].gradient();       // (2)

const auto neg_grad_test_v = -grad_test_v;                            // (3)
const auto s_times_neg_grad_test_v = s * neg_grad_test_v;             // (4)
```
The variables `test` and `sclr` are classes that can be used to generate
some *leaf* symbolic operators. The first two of the enumerated auto-deduced
class types are just such operators. The latter two are the result of a unary
operation and a binary operation. 

Omitting some (well, actually a lot) of the details for the sake of brevity,
these compile to:
1. A (constant) scalar functor that returns the value `2.0` everywhere is a
   ```c++
   SymbolicOp<ScalarFunctor, SymbolicOpCodes::value>
   ```
   Shown here, the class signature is somewhat indicatative of the fact that
   it represents the `value` of a `ScalarFunctor`.
2. The gradient of a vector-valued test function is a 
   ```c++
   SymbolicOp<SubSpaceViews::Vector<TestFunction>, SymbolicOpCodes::gradient>
   ```
   which again somewhat transparently denotes that it represents the `gradient
    of a `Vector` subspace view of a `TestFunction`.
1. The negation of the gradient of a vector-valued test function is a
   ```c++
   UnaryOp<SymbolicOp<SubSpaceViews::Vector<TestFunction>, SymbolicOpCodes::gradient>, UnaryOpCodes::negate>
   ```
   Composition complicates things slightly. Read from the outside in, this class
   is a unary operation that `negates` an argument; that argument is the object of
   the type described by point `(2)`.
2. Multiplication of the constant scalar function with the negative gradient
   of a vector-valued test function results in a
   ```c++
   BinaryOp<SymbolicOp<ScalarFunctor, SymbolicOpCodes::value>, UnaryOp<SymbolicOp<SubSpaceViews::Vector<TestFunction>, SymbolicOpCodes::gradient>, UnaryOpCodes::negate>, BinaryOpCodes::multiply>
   ```
   Again read from the outside in, this is a binary operation that `multiply`s
   two arguments. The first argument is the same type described in point `(1)`,
   while the second argument is the type described by point `(3)`.

Unfortunately, the concrete class types get more and more complex as the chain
of operations increases. However, as the compiler knows the exact type that
computes operations, the call operators are transparent and the compiler is
in principle able to generate very fast code with which to perform evaluations.


# Features: 
-----------

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

# Class list
------------

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
  - Indexed storage for time history or other solution fields
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
- `Integrator`: User-defined function integrators, used to compute integrals
  of quantities over a domain or subdomain.
  - Position independent/dependent
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