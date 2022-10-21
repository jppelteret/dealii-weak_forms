// ---------------------------------------------------------------------
//
// Copyright (C) 2021 - 2022 by Jean-Paul Pelteret
//
// This file is part of the Weak forms for deal.II library.
//
// The Weak forms for deal.II library is free software; you can use it,
// redistribute it, and/or modify it under the terms of the GNU Lesser
// General Public License as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later
// version. The full text of the license can be found in the file LICENSE
// at the top level of the Weak forms for deal.II distribution.
//
// ---------------------------------------------------------------------

// Test symbolic expression output
// - Residual view functional form

#include <deal.II/differentiation/sd.h>

#include <weak_forms/weak_forms.h>

#include "../weak_forms_tests.h"


template <int dim,
          int spacedim = dim,
          typename TestSpaceOp,
          typename CompositeSymbolicOp>
void
test(const TestSpaceOp &test_space_op, const CompositeSymbolicOp &functor_op)
{
  using namespace dealiiWeakForms::WeakForms;
  using namespace dealiiWeakForms::WeakForms::internal;
  using namespace dealiiWeakForms::WeakForms::Operators::internal;
  CompositeOpHelper<CompositeSymbolicOp>::print(functor_op);

  // A tuple of all field solution operations that appear in the symbolic
  // expression tree. Some of these are duplicated: that's fine, as they'll
  // be a no-op when producing the symbolic substitution maps.
  const auto subspace_field_solution_ops =
    CompositeOpHelper<CompositeSymbolicOp>::get_subspace_field_solution_ops(
      functor_op);

  // Create the residual functor, separately from the residual functional form.
  // We do this because we need to explicitly know the template parameters that
  // are passed to both the residual functor and the residual functional form
  // creation methods.
  const auto residual_func =
    create_residual_functor_from_tuple("R", "R", subspace_field_solution_ops);
  deallog << "residual_func: " << residual_func.as_ascii(SymbolicDecorations())
          << std::endl;

  // Finally create the residual view functional form. The original functor is
  // needed in order to extract the substitution map for the variables other
  // than the field solution components.
  const auto residual_view_form =
    create_residual_view_form_from_residual<dim, spacedim>(residual_func,
                                                           test_space_op,
                                                           functor_op);
  deallog << "residual_view_form: "
          << residual_view_form.as_ascii(SymbolicDecorations()) << std::endl;
}


template <int dim>
void
run()
{
  deallog << "Dim: " << dim << std::endl;

  using namespace dealii;
  using namespace WeakForms;
  using namespace Differentiation;

  constexpr int spacedim = dim;
  using SDNumber_t       = Differentiation::SD::Expression;

  const auto s1 = constant_scalar<dim>(3.0, "s", "s");
  const auto s2 = constant_scalar<dim>(2.0);

  const TestFunction<dim, spacedim>  test_function;
  const FieldSolution<dim, spacedim> field_solution;
  const SubSpaceExtractors::Scalar   subspace_extractor(0, "s", "s");

  const auto ds_val  = test_function[subspace_extractor].value();
  const auto ds_grad = test_function[subspace_extractor].gradient();
  const auto ds_hess = test_function[subspace_extractor].hessian();

  const auto s_val  = field_solution[subspace_extractor].value();
  const auto s_grad = field_solution[subspace_extractor].gradient();
  const auto s_hess = field_solution[subspace_extractor].hessian();

  // Scalar
  {
    deallog << "Scalar" << std::endl;
    const auto functor_op = s1 * s2 + s1 * s_val * s_val + s2 * s_grad * s_grad;
    deallog << "functor_op: " << functor_op.as_expression() << std::endl;

    test<dim, spacedim>(ds_val, functor_op);
  }

  // Vector
  {
    deallog << "Vector" << std::endl;
    const auto functor_op = s1 * s_val * s_grad + s2 * s_hess * s_grad;
    deallog << "functor_op: " << functor_op.as_expression() << std::endl;

    test<dim, spacedim>(ds_grad, functor_op);
  }

  // Tensor
  {
    deallog << "Tensor" << std::endl;
    const auto functor_op =
      s1 * s_val * s_hess + s2 * (s_grad * s_grad) * s_hess;
    deallog << "functor_op: " << functor_op.as_expression() << std::endl;

    test<dim, spacedim>(ds_hess, functor_op);
  }

  deallog << "OK" << std::endl;
}


int
main(int argc, char **argv)
{
  initlog();

  run<2>();
  run<3>();

  deallog << "OK" << std::endl;

  return 0;
}
