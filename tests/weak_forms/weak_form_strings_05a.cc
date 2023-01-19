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
// - Integrals

#include <deal.II/base/function_lib.h>
#include <deal.II/base/tensor_function.h>

#include <weak_forms/bilinear_forms.h>
#include <weak_forms/functors.h>
#include <weak_forms/linear_forms.h>
#include <weak_forms/symbolic_decorations.h>
#include <weak_forms/symbolic_integral.h>
#include <weak_forms/symbolic_operators.h>

#include "../weak_forms_tests.h"


template <int dim, int spacedim = dim, typename NumberType = double>
void
run()
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;

  // Customise the naming convensions, if we wish to.
  const SymbolicDecorations decorator(
    SymbolicNamesAscii(),
    SymbolicNamesLaTeX(),
    FormattingLaTeX(FormattingLaTeX::IntegralFormat::bilinear_form_notation));

  const ScalarFunctor                 scalar("s", "s");
  const ScalarFunctionFunctor<dim>    scalar_func("sf", "s");
  const TensorFunctionFunctor<2, dim> tensor_func2("Tf2", "T");

  const TestFunction<dim, spacedim>  test;
  const TrialSolution<dim, spacedim> trial;
  const FieldSolution<dim, spacedim> soln;

  const auto test_val  = test.value();
  const auto trial_val = trial.value();
  const auto soln_val  = soln.value();

  const VolumeIntegral<>    integral_dV;
  const BoundaryIntegral<>  integral_dA;
  const InterfaceIntegral<> integral_dI;

  const VolumeIntegral<>    integral_sub_dV({1, 2, 3});
  const BoundaryIntegral<>  integral_sub_dA({4, 5, 6});
  const InterfaceIntegral<> integral_sub_dI({7, 8, 9});

  // Test strings
  {
    LogStream::Prefix prefix("string");

    deallog << "Volume integral: " << integral_dV.as_ascii(decorator)
            << std::endl;
    deallog << "Boundary integral: " << integral_dA.as_ascii(decorator)
            << std::endl;
    deallog << "Interface integral: " << integral_dI.as_ascii(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "Volume integral: " << integral_sub_dV.as_ascii(decorator)
            << std::endl;
    deallog << "Boundary integral: " << integral_sub_dA.as_ascii(decorator)
            << std::endl;
    deallog << "Interface integral: " << integral_sub_dI.as_ascii(decorator)
            << std::endl;

    deallog << std::endl;
  }

  // Test LaTeX
  {
    LogStream::Prefix prefix("LaTeX");

    deallog << "Volume integral: " << integral_dV.as_latex(decorator)
            << std::endl;
    deallog << "Boundary integral: " << integral_dA.as_latex(decorator)
            << std::endl;
    deallog << "Interface integral: " << integral_dI.as_latex(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "Volume integral: " << integral_sub_dV.as_latex(decorator)
            << std::endl;
    deallog << "Boundary integral: " << integral_sub_dA.as_latex(decorator)
            << std::endl;
    deallog << "Interface integral: " << integral_sub_dI.as_latex(decorator)
            << std::endl;

    deallog << std::endl;
  }

  const auto s = scalar.template value<NumberType, dim, spacedim>(
    [](const FEValuesBase<dim, spacedim> &, const unsigned int)
    { return 1.0; });

  const Functions::ConstantFunction<dim, NumberType> constant_scalar_function(
    1);
  const ConstantTensorFunction<2, dim, NumberType> constant_tensor_function(
    unit_symmetric_tensor<dim>());
  const auto sf =
    scalar_func.template value<NumberType, dim>(constant_scalar_function);
  const auto T2f =
    tensor_func2.template value<NumberType, dim>(constant_tensor_function);

  const auto l_form  = linear_form(test_val, soln_val);
  const auto bl_form = bilinear_form(test_val, soln_val, trial_val);

  const auto s_dV   = integrate(s, integral_dV);
  const auto T2f_dA = integrate(T2f, integral_dA);
  const auto sf_dI  = integrate(sf, integral_dI);

  const auto blf_dV = integrate(bl_form, integral_dV);
  const auto blf_dA = integrate(bl_form, integral_dA);
  const auto blf_dI = integrate(bl_form, integral_dI);

  const auto lf_dV = integrate(l_form, integral_dV);
  const auto lf_dA = integrate(l_form, integral_dA);
  const auto lf_dI = integrate(l_form, integral_dI);

  // Test values
  {
    LogStream::Prefix prefix("values");

    deallog << "Volume integral: " << s_dV.as_latex(decorator) << std::endl;
    deallog << "Boundary integral: " << T2f_dA.as_latex(decorator) << std::endl;
    deallog << "Interface integral: " << sf_dI.as_latex(decorator) << std::endl;

    deallog << "Integrate function: " << std::endl;
    deallog << "Bilinear form (Volume integral): " << blf_dV.as_latex(decorator)
            << std::endl;
    deallog << "Bilinear form (Boundary integral): "
            << blf_dA.as_latex(decorator) << std::endl;
    deallog << "Bilinear form (Interface integral): "
            << blf_dI.as_latex(decorator) << std::endl;

    deallog << "Linear form (Volume integral): " << lf_dV.as_latex(decorator)
            << std::endl;
    deallog << "Linear form (Boundary integral): " << lf_dA.as_latex(decorator)
            << std::endl;
    deallog << "Linear form (Interface integral): " << lf_dI.as_latex(decorator)
            << std::endl;

    deallog << "Form integral: " << std::endl;
    deallog << "Bilinear form (Volume integral): "
            << bl_form.dV().as_latex(decorator) << std::endl;
    deallog << "Bilinear form (Boundary integral): "
            << bl_form.dA().as_latex(decorator) << std::endl;
    deallog << "Bilinear form (Interface integral): "
            << bl_form.dI().as_latex(decorator) << std::endl;

    deallog << "Linear form (Volume integral): "
            << l_form.dV().as_latex(decorator) << std::endl;
    deallog << "Linear form (Boundary integral): "
            << l_form.dA().as_latex(decorator) << std::endl;
    deallog << "Linear form (Interface integral): "
            << l_form.dI().as_latex(decorator) << std::endl;

    deallog << "Form integral with subregions: " << std::endl;
    deallog << "Bilinear form (Volume integral): "
            << bl_form.dV({1, 2, 3}).as_latex(decorator) << std::endl;
    deallog << "Bilinear form (Boundary integral): "
            << bl_form.dA({4, 5, 6}).as_latex(decorator) << std::endl;
    deallog << "Bilinear form (Interface integral): "
            << bl_form.dI({7, 8, 9}).as_latex(decorator) << std::endl;

    deallog << "Linear form (Volume integral): "
            << l_form.dV({1, 2, 3}).as_latex(decorator) << std::endl;
    deallog << "Linear form (Boundary integral): "
            << l_form.dA({4, 5, 6}).as_latex(decorator) << std::endl;
    deallog << "Linear form (Interface integral): "
            << l_form.dI({7, 8, 9}).as_latex(decorator) << std::endl;

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
