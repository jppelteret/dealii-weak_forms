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


// Check functor form stringization and printing
// - Functor helpers

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <weak_forms/functors.h>
#include <weak_forms/symbolic_decorations.h>
#include <weak_forms/symbolic_operators.h>

#include "../weak_forms_tests.h"


template <int dim, int spacedim = dim, typename NumberType = double>
void
run()
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;

  // Customise the naming convensions, if we wish to.
  const SymbolicDecorations decorator;

  const auto s  = constant_scalar<dim>(1.0);
  const auto v  = constant_vector<dim>(Tensor<1, dim>());
  const auto T2 = constant_tensor<2, dim>(Tensor<2, dim>());
  const auto T3 = constant_tensor<3, dim>(Tensor<3, dim>());
  const auto T4 = constant_tensor<4, dim>(Tensor<4, dim>());
  const auto S2 =
    constant_symmetric_tensor<2, dim>(unit_symmetric_tensor<dim>());
  const auto S4 = constant_symmetric_tensor<4, dim>(identity_tensor<dim>());

  // Test strings
  {
    LogStream::Prefix prefix("string");

    deallog << "Scalar: " << s.as_ascii(decorator) << std::endl;
    deallog << "Vector: " << v.as_ascii(decorator) << std::endl;
    deallog << "Tensor (rank 2): " << T2.as_ascii(decorator) << std::endl;
    deallog << "Tensor (rank 3): " << T3.as_ascii(decorator) << std::endl;
    deallog << "Tensor (rank 4): " << T3.as_ascii(decorator) << std::endl;
    deallog << "SymmetricTensor (rank 2): " << S2.as_ascii(decorator)
            << std::endl;
    deallog << "SymmetricTensor (rank 4): " << S4.as_ascii(decorator)
            << std::endl;

    deallog << std::endl;
  }

  // Test LaTeX
  {
    LogStream::Prefix prefix("LaTeX");

    deallog << "Scalar: " << s.as_latex(decorator) << std::endl;
    deallog << "Vector: " << v.as_latex(decorator) << std::endl;
    deallog << "Tensor (rank 2): " << T2.as_latex(decorator) << std::endl;
    deallog << "Tensor (rank 3): " << T3.as_latex(decorator) << std::endl;
    deallog << "Tensor (rank 4): " << T4.as_latex(decorator) << std::endl;
    deallog << "SymmetricTensor (rank 2): " << S2.as_latex(decorator)
            << std::endl;
    deallog << "SymmetricTensor (rank 4): " << S4.as_latex(decorator)
            << std::endl;

    deallog << std::endl;
  }

  const FE_Q<dim>         fe_cell(1);
  const QGauss<dim>       qf_cell(2);
  FEValues<dim, spacedim> fe_values(fe_cell, qf_cell, update_quadrature_points);

  Triangulation<dim, spacedim> tria;
  GridGenerator::hyper_cube(tria);
  fe_values.reinit(tria.begin_active());

  // Test values
  {
    LogStream::Prefix prefix("values");

    deallog << "Scalar: " << s.template operator()<NumberType>(fe_values)[0]
            << std::endl;
    deallog << "Vector: " << v.template operator()<NumberType>(fe_values)[0]
            << std::endl;
    deallog << "Tensor (rank 2): "
            << T2.template operator()<NumberType>(fe_values)[0] << std::endl;
    deallog << "Tensor (rank 3): "
            << T3.template operator()<NumberType>(fe_values)[0] << std::endl;
    deallog << "Tensor (rank 4): "
            << T4.template operator()<NumberType>(fe_values)[0] << std::endl;
    deallog << "SymmetricTensor (rank 2): "
            << S2.template operator()<NumberType>(fe_values)[0] << std::endl;
    deallog << "SymmetricTensor (rank 4): "
            << S4.template operator()<NumberType>(fe_values)[0] << std::endl;

    deallog << std::endl;
  }

  deallog << "OK" << std::endl << std::endl;
}


int
main()
{
  initlog();

  run<2>();
  run<3>();

  deallog << "OK" << std::endl;
}
