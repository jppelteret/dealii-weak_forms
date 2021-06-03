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


// Check scalar operator* for integrals


#include <weak_forms/bilinear_forms.h>
#include <weak_forms/binary_integral_operators.h>
#include <weak_forms/binary_operators.h>
#include <weak_forms/linear_forms.h>
#include <weak_forms/mixed_operators.h>
#include <weak_forms/spaces.h>
#include <weak_forms/symbolic_integral.h>
#include <weak_forms/type_traits.h>
#include <weak_forms/unary_integral_operators.h>
#include <weak_forms/unary_operators.h>

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

  // Symbolic integral
  const auto s_blf_dV = bl_form.dV();
  const auto s_lf_dV  = l_form.dV();

  // Unary integral
  const auto u_blf_dV = -s_blf_dV;
  const auto u_lf_dV  = -s_lf_dV;

  // Binary integral
  const auto b_blf_dV = s_blf_dV + s_blf_dV;
  const auto b_lf_dV  = s_lf_dV + s_lf_dV;

  deallog << "OK" << std::endl;
}
