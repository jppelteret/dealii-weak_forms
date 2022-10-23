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
// - Subspace field solution: Vector

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

  // Vector
  {
    const FieldSolution<dim, spacedim> field_solution;
    const SubSpaceExtractors::Vector   subspace_extractor(0, "v", "v");

    const auto v_val  = field_solution[subspace_extractor].value();
    const auto v_grad = field_solution[subspace_extractor].gradient();
    const auto v_symm_grad =
      field_solution[subspace_extractor].symmetric_gradient();
    const auto v_div  = field_solution[subspace_extractor].divergence();
    const auto v_curl = field_solution[subspace_extractor].curl();
    const auto v_hess = field_solution[subspace_extractor].hessian();
    const auto v_d3   = field_solution[subspace_extractor].third_derivative();

    deallog << "v_val: " << v_val.as_expression() << std::endl;
    deallog << "v_grad: " << v_grad.as_expression() << std::endl;
    deallog << "v_symm_grad: " << v_symm_grad.as_expression() << std::endl;
    deallog << "v_div: " << v_div.as_expression() << std::endl;
    deallog << "v_curl: " << v_curl.as_expression() << std::endl;
    deallog << "v_hess: " << v_hess.as_expression() << std::endl;
    deallog << "v_d3: " << v_d3.as_expression() << std::endl;

    // Check unary and binary ops
    const auto res = -(v_val * v_val) + scalar_product(v_grad, v_grad) +
                     scalar_product(v_symm_grad, v_symm_grad) +
                     (v_div * v_div) + scalar_product(v_curl, v_curl) +
                     scalar_product(v_hess, v_hess) +
                     scalar_product(v_d3, v_d3);
    deallog << "res: " << res.as_expression() << std::endl;
  }

  // Vector (jump)
  {
    const FieldSolution<dim, spacedim> field_solution;
    const SubSpaceExtractors::Vector   subspace_extractor(0, "v", "v");

    const auto v_val  = field_solution[subspace_extractor].jump_in_values();
    const auto v_grad = field_solution[subspace_extractor].jump_in_gradients();
    const auto v_hess = field_solution[subspace_extractor].jump_in_hessians();
    const auto v_d3 =
      field_solution[subspace_extractor].jump_in_third_derivatives();

    deallog << "v_val: " << v_val.as_expression() << std::endl;
    deallog << "v_grad: " << v_grad.as_expression() << std::endl;
    deallog << "v_hess: " << v_hess.as_expression() << std::endl;
    deallog << "v_d3: " << v_d3.as_expression() << std::endl;

    // Check unary and binary ops
    const auto res = -(v_val * v_val) + scalar_product(v_grad, v_grad) +
                     scalar_product(v_hess, v_hess) +
                     scalar_product(v_d3, v_d3);
    deallog << "res: " << res.as_expression() << std::endl;
  }

  // Vector (average)
  {
    const FieldSolution<dim, spacedim> field_solution;
    const SubSpaceExtractors::Vector   subspace_extractor(0, "v", "v");

    const auto v_val = field_solution[subspace_extractor].average_of_values();
    const auto v_grad =
      field_solution[subspace_extractor].average_of_gradients();
    const auto v_hess =
      field_solution[subspace_extractor].average_of_hessians();

    deallog << "v_val: " << v_val.as_expression() << std::endl;
    deallog << "v_grad: " << v_grad.as_expression() << std::endl;
    deallog << "v_hess: " << v_hess.as_expression() << std::endl;

    // Check unary and binary ops
    const auto res = -(v_val * v_val) + scalar_product(v_grad, v_grad) +
                     scalar_product(v_hess, v_hess);
    deallog << "res: " << res.as_expression() << std::endl;
  }

  deallog << "OK" << std::endl;
}


int
main(int argc, char **argv)
{
  initlog();

  // run<2>(); // Curl only defined in dim == 3
  run<3>();

  deallog << "OK" << std::endl;

  return 0;
}
