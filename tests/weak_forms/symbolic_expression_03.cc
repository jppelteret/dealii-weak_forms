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
// - Binary operations

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
    const auto s1 = constant_scalar<dim>(1.0, "s1", "s1");
    const auto s2 = constant_scalar<dim>(1.0, "s2", "s2");

    deallog << "s1: " << s1.as_expression() << std::endl;
    deallog << "s2: " << s2.as_expression() << std::endl;
    deallog << "(s1+s2): " << (s1 + s2).as_expression() << std::endl;
    deallog << "(s1-s2): " << (s1 - s2).as_expression() << std::endl;
    deallog << "(s1*s2): " << (s1 * s2).as_expression() << std::endl;
    deallog << "(s1/s2): " << (s1 / s2).as_expression() << std::endl;
    deallog << "pow(s1,2): " << pow(s1, 2).as_expression() << std::endl;
    deallog << "max(s1,s2): " << max(s1, s2).as_expression() << std::endl;
    deallog << "min(s1,s2): " << min(s1, s2).as_expression() << std::endl;
  }

  // Vector
  {
    const auto v1 = constant_vector<dim>(Tensor<1, dim>(), "v1", "v1");
    const auto v2 = constant_vector<dim>(Tensor<1, dim>(), "v2", "v2");

    deallog << "v1: " << v1.as_expression() << std::endl;
    deallog << "v2: " << v2.as_expression() << std::endl;
    deallog << "(v1+v2): " << (v1 + v2).as_expression() << std::endl;
    deallog << "(v1-v2): " << (v1 - v2).as_expression() << std::endl;
    deallog << "(v1*v2): " << (v1 * v2).as_expression() << std::endl;
    deallog << "cross_product(v1,v2): " << cross_product(v1, v2).as_expression()
            << std::endl;
    deallog << "schur_product(v1,v2): " << schur_product(v1, v2).as_expression()
            << std::endl;
    deallog << "outer_product(v1,v2): " << outer_product(v1, v2).as_expression()
            << std::endl;
    deallog << "scalar_product(v1,v2): "
            << scalar_product(v1, v2).as_expression() << std::endl;
    deallog << "contract<0, 0>(v1,v2): "
            << contract<0, 0>(v1, v2).as_expression() << std::endl;
  }

  // Tensor
  {
    const auto T1 = constant_tensor<dim>(Tensor<2, dim>(), "T1", "T1");
    const auto T2 = constant_tensor<dim>(Tensor<2, dim>(), "T2", "T2");

    deallog << "T1: " << T1.as_expression() << std::endl;
    deallog << "T2: " << T2.as_expression() << std::endl;
    deallog << "(T1+T2): " << (T1 + T2).as_expression() << std::endl;
    deallog << "(T1-T2): " << (T1 - T2).as_expression() << std::endl;
    deallog << "(T1*T2): " << (T1 * T2).as_expression() << std::endl;
    deallog << "schur_product(T1,T2): " << schur_product(T1, T2).as_expression()
            << std::endl;
    deallog << "outer_product(T1,T2): " << outer_product(T1, T2).as_expression()
            << std::endl;
    deallog << "scalar_product(T1,T2): "
            << scalar_product(T1, T2).as_expression() << std::endl;
    deallog << "contract<0, 0>(T1,T2): "
            << contract<0, 0>(T1, T2).as_expression() << std::endl;
    deallog << "double_contract<0, 1, 1, 0>(T1,T2): "
            << double_contract<0, 1, 1, 0>(T1, T2).as_expression() << std::endl;
  }

  // Symmetric tensor
  {
    const auto S1 =
      constant_symmetric_tensor<dim>(unit_symmetric_tensor<dim>(), "S1", "S1");
    const auto S2 =
      constant_symmetric_tensor<dim>(unit_symmetric_tensor<dim>(), "S2", "S2");

    deallog << "S1: " << S1.as_expression() << std::endl;
    deallog << "S2: " << S2.as_expression() << std::endl;
    deallog << "(S1+S2): " << (S1 + S2).as_expression() << std::endl;
    deallog << "(S1-S2): " << (S1 - S2).as_expression() << std::endl;
    deallog << "(S1*S2): " << (S1 * S2).as_expression() << std::endl;
    deallog << "outer_product(S1,S2): " << outer_product(S1, S2).as_expression()
            << std::endl;
    deallog << "scalar_product(S1,S2): "
            << scalar_product(S1, S2).as_expression() << std::endl;
  }

  deallog << "OK" << std::endl;
}


int
main(int argc, char **argv)
{
  initlog();

  // run<2>(); // Cross-product not valid for dim==2
  run<3>();

  deallog << "OK" << std::endl;

  return 0;
}
