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
// - Functors

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <weak_forms/functors.h>
#include <weak_forms/symbolic_decorations.h>
#include <weak_forms/unary_operators.h>

#include "../weak_forms_tests.h"


template <int dim, int spacedim = dim, typename NumberType = double>
void
run()
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;

  // Customise the naming convensions, if we wish to.
  const SymbolicDecorations decorator;

  const ScalarFunctor                  scalar("s", "s");
  const VectorFunctor<dim>             vector("v", "v");
  const TensorFunctor<2, dim>          tensor2("T2", "T");
  const TensorFunctor<3, dim>          tensor3("T3", "P");
  const TensorFunctor<4, dim>          tensor4("T4", "K");
  const SymmetricTensorFunctor<2, dim> symm_tensor2("S2", "T");
  const SymmetricTensorFunctor<4, dim> symm_tensor4("S4", "K");
  const ScalarFunctionFunctor<dim>     scalar_func("sf", "s");
  const TensorFunctionFunctor<2, dim>  tensor_func2("Tf2", "T");

  // Test strings
  {
    LogStream::Prefix prefix("string");

    deallog << "Scalar: " << scalar.as_ascii(decorator) << std::endl;
    deallog << "Vector: " << vector.as_ascii(decorator) << std::endl;
    deallog << "Tensor (rank 2): " << tensor2.as_ascii(decorator) << std::endl;
    deallog << "Tensor (rank 3): " << tensor3.as_ascii(decorator) << std::endl;
    deallog << "Tensor (rank 4): " << tensor4.as_ascii(decorator) << std::endl;
    deallog << "SymmetricTensor (rank 2): " << symm_tensor2.as_ascii(decorator)
            << std::endl;
    deallog << "SymmetricTensor (rank 4): " << symm_tensor4.as_ascii(decorator)
            << std::endl;

    deallog << "Scalar function: " << scalar_func.as_ascii(decorator)
            << std::endl;
    deallog << "Tensor function (rank 2): " << tensor_func2.as_ascii(decorator)
            << std::endl;

    deallog << std::endl;
  }

  // Test LaTeX
  {
    LogStream::Prefix prefix("LaTeX");

    deallog << "Scalar: " << scalar.as_latex(decorator) << std::endl;
    deallog << "Vector: " << vector.as_latex(decorator) << std::endl;
    deallog << "Tensor (rank 2): " << tensor2.as_latex(decorator) << std::endl;
    deallog << "Tensor (rank 3): " << tensor3.as_latex(decorator) << std::endl;
    deallog << "Tensor (rank 4): " << tensor4.as_latex(decorator) << std::endl;
    deallog << "SymmetricTensor (rank 2): " << symm_tensor2.as_latex(decorator)
            << std::endl;
    deallog << "SymmetricTensor (rank 4): " << symm_tensor4.as_latex(decorator)
            << std::endl;

    deallog << "Scalar function: " << scalar_func.as_latex(decorator)
            << std::endl;
    deallog << "Tensor function (rank 2): " << tensor_func2.as_latex(decorator)
            << std::endl;

    deallog << std::endl;
  }

  const auto s = scalar.template value<NumberType, dim, spacedim>(
    [](const FEValuesBase<dim, spacedim> &, const unsigned int)
    { return 1.0; });
  const auto v = vector.template value<NumberType, dim>(
    [](const FEValuesBase<dim, spacedim> &, const unsigned int)
    { return Tensor<1, spacedim, NumberType>(); });
  const auto T2 = tensor2.template value<NumberType, dim>(
    [](const FEValuesBase<dim, spacedim> &, const unsigned int)
    { return Tensor<2, spacedim, NumberType>(); });
  const auto T3 = tensor3.template value<NumberType, dim>(
    [](const FEValuesBase<dim, spacedim> &, const unsigned int)
    { return Tensor<3, spacedim, NumberType>(); });
  const auto T4 = tensor4.template value<NumberType, dim>(
    [](const FEValuesBase<dim, spacedim> &, const unsigned int)
    { return Tensor<4, spacedim, NumberType>(); });
  const auto S2 = tensor2.template value<NumberType, dim>(
    [](const FEValuesBase<dim, spacedim> &, const unsigned int)
    { return SymmetricTensor<2, spacedim, NumberType>(); });
  const auto S4 = tensor4.template value<NumberType, dim>(
    [](const FEValuesBase<dim, spacedim> &, const unsigned int)
    { return SymmetricTensor<4, spacedim, NumberType>(); });

  const Functions::ConstantFunction<dim, NumberType> constant_scalar_function(
    1);
  const ConstantTensorFunction<2, dim, NumberType> constant_tensor_function(
    unit_symmetric_tensor<dim>());
  const auto sf =
    scalar_func.template value<NumberType, dim>(constant_scalar_function);
  const auto grad_sf =
    scalar_func.template gradient<NumberType, dim>(constant_scalar_function);
  const auto T2f =
    tensor_func2.template value<NumberType, dim>(constant_tensor_function);
  const auto grad_T2f =
    tensor_func2.template gradient<NumberType, dim>(constant_tensor_function);

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

    deallog << "Scalar function : "
            << sf.template operator()<NumberType>(fe_values)[0] << std::endl;
    deallog << "Tensor function (rank 2): "
            << T2f.template operator()<NumberType>(fe_values)[0] << std::endl;

    deallog << "Scalar function gradient : "
            << grad_sf.template operator()<NumberType>(fe_values)[0]
            << std::endl;
    deallog << "Tensor function (rank 2) gradient: "
            << grad_T2f.template operator()<NumberType>(fe_values)[0]
            << std::endl;

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
