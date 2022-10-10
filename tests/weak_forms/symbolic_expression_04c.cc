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
// - Subspace field solution: Tensor

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

  // Tensor
  {
    const FieldSolution<dim, spacedim>  field_solution;
    const SubSpaceExtractors::Tensor<2> subspace_extractor(0, "T", "T");

    const auto T_val  = field_solution[subspace_extractor].value();
    const auto T_grad = field_solution[subspace_extractor].gradient();
    const auto T_div  = field_solution[subspace_extractor].divergence();

    deallog << "T_val: " << T_val.as_expression() << std::endl;
    deallog << "T_grad: " << T_grad.as_expression() << std::endl;
    deallog << "T_div: " << T_div.as_expression() << std::endl;

    // Check unary and binary ops
    const auto res = -scalar_product(T_val, T_val) +
                     scalar_product(T_grad, T_grad) + (T_div * T_div);
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
