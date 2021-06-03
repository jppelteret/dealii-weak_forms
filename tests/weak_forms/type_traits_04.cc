// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------


// Check type traits for integrals


#include <weak_forms/bilinear_forms.h>
#include <weak_forms/linear_forms.h>
#include <weak_forms/spaces.h>
#include <weak_forms/symbolic_integral.h>
#include <weak_forms/type_traits.h>

#include "../weak_forms_tests.h"


int
main()
{
  initlog();

  using namespace WeakForms;

  constexpr int dim      = 2;
  constexpr int spacedim = 2;

  const TestFunction<dim, spacedim>  test;
  const TrialSolution<dim, spacedim> trial;
  const FieldSolution<dim, spacedim> soln;

  const auto l_form = linear_form(test.value(), soln.value());
  const auto bl_form =
    bilinear_form(test.gradient(), soln.hessian(), trial.gradient());

  const VolumeIntegral    integral_dV;
  const BoundaryIntegral  integral_dA;
  const InterfaceIntegral integral_dI;

  const auto blf_dV = integrate(bl_form, integral_dV);
  const auto blf_dA = integrate(bl_form, integral_dA);
  const auto blf_dI = integrate(bl_form, integral_dI);

  const auto lf_dV = integrate(l_form, integral_dV);
  const auto lf_dA = integrate(l_form, integral_dA);
  const auto lf_dI = integrate(l_form, integral_dI);

  const auto blf_sub_dV = bl_form.dV({1, 2, 3});
  const auto blf_sub_dA = bl_form.dA({4, 5, 6});
  const auto blf_sub_dI = bl_form.dI({7, 8, 9});

  const auto lf_sub_dV = l_form.dV({1, 2, 3});
  const auto lf_sub_dA = l_form.dA({4, 5, 6});
  const auto lf_sub_dI = l_form.dI({7, 8, 9});

  deallog << std::boolalpha;

  // deallog << is_symbolic_integral_op<decltype(blf_dV)>::value << std::endl;
  // // Not working?
  deallog << is_symbolic_integral_op<
               typename std::decay<decltype(blf_dV)>::type>::value
          << std::endl;
  deallog << is_symbolic_integral_op<
               typename std::decay<decltype(blf_dA)>::type>::value
          << std::endl;
  deallog << is_symbolic_integral_op<
               typename std::decay<decltype(blf_dI)>::type>::value
          << std::endl;

  deallog << is_symbolic_integral_op<
               typename std::decay<decltype(lf_dV)>::type>::value
          << std::endl;
  deallog << is_symbolic_integral_op<
               typename std::decay<decltype(lf_dA)>::type>::value
          << std::endl;
  deallog << is_symbolic_integral_op<
               typename std::decay<decltype(lf_dI)>::type>::value
          << std::endl;

  deallog << std::endl;

  deallog << is_symbolic_integral_op<
               typename std::decay<decltype(blf_sub_dV)>::type>::value
          << std::endl;
  deallog << is_symbolic_integral_op<
               typename std::decay<decltype(blf_sub_dA)>::type>::value
          << std::endl;
  deallog << is_symbolic_integral_op<
               typename std::decay<decltype(blf_sub_dI)>::type>::value
          << std::endl;

  deallog << is_symbolic_integral_op<
               typename std::decay<decltype(lf_sub_dV)>::type>::value
          << std::endl;
  deallog << is_symbolic_integral_op<
               typename std::decay<decltype(lf_sub_dA)>::type>::value
          << std::endl;
  deallog << is_symbolic_integral_op<
               typename std::decay<decltype(lf_sub_dI)>::type>::value
          << std::endl;

  deallog << std::endl;

  deallog << "OK" << std::endl;
}
