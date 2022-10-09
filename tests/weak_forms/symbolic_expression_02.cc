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
// - Unary operations

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
    const auto s = constant_scalar<dim>(1.0, "s", "s");

    deallog << "s: " << s.as_expression() << std::endl;
    deallog << "-s: " << (-s).as_expression() << std::endl;
    deallog << "normalize(s): " << normalize(s).as_expression() << std::endl;
    deallog << "sin(s): " << sin(s).as_expression() << std::endl;
    deallog << "cos(s): " << cos(s).as_expression() << std::endl;
    deallog << "tan(s): " << tan(s).as_expression() << std::endl;
    deallog << "exp(s): " << exp(s).as_expression() << std::endl;
    deallog << "log(s): " << log(s).as_expression() << std::endl;
    deallog << "sqrt(s): " << sqrt(s).as_expression() << std::endl;
    deallog << "abs(s): " << abs(s).as_expression() << std::endl;
  }

  // Vector
  {
    const auto v = constant_vector<dim>(Tensor<1, dim>(), "v", "v");

    deallog << "v: " << v.as_expression() << std::endl;
    deallog << "-v: " << (-v).as_expression() << std::endl;
    deallog << "normalize(v): " << normalize(v).as_expression() << std::endl;
  }

  // Tensor
  {
    const auto T2 = constant_tensor<dim>(Tensor<2, dim>(), "T2", "T2");

    deallog << "T2: " << (T2).as_expression() << std::endl;
    deallog << "-T2: " << (-T2).as_expression() << std::endl;
    deallog << "normalize(T2): " << normalize(T2).as_expression() << std::endl;
    deallog << "determinant(T2): " << determinant(T2).as_expression()
            << std::endl;
    deallog << "invert(T2): " << invert(T2).as_expression() << std::endl;
    deallog << "transpose(T2): " << transpose(T2).as_expression() << std::endl;
    deallog << "trace(T2): " << trace(T2).as_expression() << std::endl;
    deallog << "symmetrize(T2): " << symmetrize(T2).as_expression()
            << std::endl;
    deallog << "adjugate(T2): " << adjugate(T2).as_expression() << std::endl;
    deallog << "cofactor(T2): " << cofactor(T2).as_expression() << std::endl;

    // TODO: terminate called after throwing an instance of
    // 'SymEngine::SymEngineException' deallog << "l1_norm(T2): " <<
    // l1_norm(T2).as_expression() << std::endl; deallog << "linfty_norm(T2): "
    // << linfty_norm(T2).as_expression() << std::endl;
  }

  // Symmetric tensor
  {
    const auto S2 =
      constant_symmetric_tensor<dim>(unit_symmetric_tensor<dim>(), "S2", "S2");

    deallog << "S2: " << (S2).as_expression() << std::endl;
    deallog << "-S2: " << (-S2).as_expression() << std::endl;
    // deallog << "normalize(S2): " << normalize(S2).as_expression() <<
    // std::endl;
    deallog << "determinant(S2): " << determinant(S2).as_expression()
            << std::endl;
    deallog << "invert(S2): " << invert(S2).as_expression() << std::endl;
    deallog << "transpose(S2): " << transpose(S2).as_expression() << std::endl;
    deallog << "trace(S2): " << trace(S2).as_expression() << std::endl;

    // TODO: terminate called after throwing an instance of
    // 'SymEngine::SymEngineException' deallog << "l1_norm(T2): " <<
    // l1_norm(T2).as_expression() << std::endl; deallog << "linfty_norm(T2): "
    // << linfty_norm(T2).as_expression() << std::endl;
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
