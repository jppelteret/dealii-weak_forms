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
// - Subspace field solution: Scalar

#include <deal.II/differentiation/sd.h>

#include <weak_forms/weak_forms.h>

#include "../weak_forms_tests.h"


using namespace dealii;


template <int dim>
void
run()
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;
  using namespace Differentiation;

  constexpr int spacedim = dim;
  using SDNumber_t       = Differentiation::SD::Expression;

  // Scalar
  {
    const FieldSolution<dim, spacedim> field_solution;
    const SubSpaceExtractors::Scalar   subspace_extractor(0, "s", "s");

    const auto s_val  = field_solution[subspace_extractor].value();
    const auto s_grad = field_solution[subspace_extractor].gradient();
    const auto s_lap  = field_solution[subspace_extractor].laplacian();
    const auto s_hess = field_solution[subspace_extractor].hessian();
    const auto s_d3   = field_solution[subspace_extractor].third_derivative();

    deallog << "s_val: " << s_val.as_expression() << std::endl;
    deallog << "s_grad: " << s_grad.as_expression() << std::endl;
    deallog << "s_lap: " << s_lap.as_expression() << std::endl;
    deallog << "s_hess: " << s_hess.as_expression() << std::endl;
    deallog << "s_d3: " << s_d3.as_expression() << std::endl;

    // Check unary and binary ops
    const auto res = -(s_val * s_val) + scalar_product(s_grad, s_grad) +
                     (s_lap * s_lap) + scalar_product(s_hess, s_hess) +
                     scalar_product(s_d3, s_d3);
    deallog << "res: " << res.as_expression() << std::endl;
  }

  // Scalar (jump)
  {
    const FieldSolution<dim, spacedim> field_solution;
    const SubSpaceExtractors::Scalar   subspace_extractor(0, "s", "s");

    const auto s_val  = field_solution[subspace_extractor].jump_in_values();
    const auto s_grad = field_solution[subspace_extractor].jump_in_gradients();
    const auto s_hess = field_solution[subspace_extractor].jump_in_hessians();
    const auto s_d3 =
      field_solution[subspace_extractor].jump_in_third_derivatives();

    deallog << "s_val: " << s_val.as_expression() << std::endl;
    deallog << "s_grad: " << s_grad.as_expression() << std::endl;
    deallog << "s_hess: " << s_hess.as_expression() << std::endl;
    deallog << "s_d3: " << s_d3.as_expression() << std::endl;

    // Check unary and binary ops
    const auto res = -(s_val * s_val) + scalar_product(s_grad, s_grad) +
                     scalar_product(s_hess, s_hess) +
                     scalar_product(s_d3, s_d3);
    deallog << "res: " << res.as_expression() << std::endl;
  }

  // Scalar (average)
  {
    const FieldSolution<dim, spacedim> field_solution;
    const SubSpaceExtractors::Scalar   subspace_extractor(0, "s", "s");

    const auto s_val = field_solution[subspace_extractor].average_of_values();
    const auto s_grad =
      field_solution[subspace_extractor].average_of_gradients();
    const auto s_hess =
      field_solution[subspace_extractor].average_of_hessians();

    deallog << "s_val: " << s_val.as_expression() << std::endl;
    deallog << "s_grad: " << s_grad.as_expression() << std::endl;
    deallog << "s_hess: " << s_hess.as_expression() << std::endl;

    // Check unary and binary ops
    const auto res = -(s_val * s_val) + scalar_product(s_grad, s_grad) +
                     scalar_product(s_hess, s_hess);
    deallog << "res: " << res.as_expression() << std::endl;
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
