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


// Check scalar operator* for integrals


#include <weak_forms/bilinear_forms.h>
#include <weak_forms/binary_integral_operators.h>
#include <weak_forms/binary_operators.h>
#include <weak_forms/linear_forms.h>
#include <weak_forms/mixed_form_operators.h>
#include <weak_forms/mixed_integral_operators.h>
#include <weak_forms/mixed_operators.h>
#include <weak_forms/spaces.h>
#include <weak_forms/symbolic_decorations.h>
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

  const SymbolicDecorations decorator;

  const double factor = 2.0;


  deallog << "Forms" << std::endl;
  const auto l_form = linear_form(test.value(), soln.value());
  const auto bl_form =
    bilinear_form(test.gradient(), soln.hessian(), trial.gradient());

  deallog << "factor * l_form: " << (factor * l_form).as_ascii(decorator)
          << std::endl;
  deallog << "l_form * factor: " << (l_form * factor).as_ascii(decorator)
          << std::endl;

  deallog << "factor * bl_form: " << (factor * bl_form).as_ascii(decorator)
          << std::endl;
  deallog << "bl_form * factor: " << (bl_form * factor).as_ascii(decorator)
          << std::endl;


  deallog << "Symbolic integral" << std::endl;
  const auto s_lf_dV  = l_form.dV();
  const auto s_blf_dV = bl_form.dV();

  static_assert(
    is_volume_integral_op<typename std::decay<decltype(s_lf_dV)>::type>::value,
    "Expected a volume integral op");
  static_assert(is_symbolic_integral_op<
                  typename std::decay<decltype(s_lf_dV)>::type>::value,
                "Expected a symbolic integral op");
  static_assert(
    is_integral_op<typename std::decay<decltype(s_lf_dV)>::type>::value,
    "Expected an integral op");

  deallog << "factor * s_lf_dV: " << (factor * s_lf_dV).as_ascii(decorator)
          << std::endl;
  deallog << "s_lf_dV * factor: " << (s_lf_dV * factor).as_ascii(decorator)
          << std::endl;

  deallog << "factor * s_blf_dV: " << (factor * s_blf_dV).as_ascii(decorator)
          << std::endl;
  deallog << "s_blf_dV * factor: " << (s_blf_dV * factor).as_ascii(decorator)
          << std::endl;


  deallog << "Unary integral" << std::endl;
  const auto u_lf_dV  = -s_lf_dV;
  const auto u_blf_dV = -s_blf_dV;

  static_assert(
    is_integral_op<typename std::decay<decltype(u_lf_dV)>::type>::value,
    "Expected an integral op");
  static_assert(
    is_unary_op<typename std::decay<decltype(u_lf_dV)>::type>::value,
    "Expected a unary op");
  static_assert(
    is_unary_integral_op<typename std::decay<decltype(u_lf_dV)>::type>::value,
    "Expected a unary integral op");

  deallog << "factor * u_lf_dV: " << (factor * u_lf_dV).as_ascii(decorator)
          << std::endl;
  deallog << "u_lf_dV * factor: " << (u_lf_dV * factor).as_ascii(decorator)
          << std::endl;

  deallog << "factor * u_blf_dV: " << (factor * u_blf_dV).as_ascii(decorator)
          << std::endl;
  deallog << "u_blf_dV * factor: " << (u_blf_dV * factor).as_ascii(decorator)
          << std::endl;


  deallog << "Binary integral: Symbolic" << std::endl;
  const auto bs_lf_dV  = s_lf_dV + s_lf_dV;
  const auto bs_blf_dV = s_blf_dV + s_blf_dV;

  deallog << "factor * bs_lf_dV: " << (factor * bs_lf_dV).as_ascii(decorator)
          << std::endl;
  deallog << "bs_lf_dV * factor: " << (bs_lf_dV * factor).as_ascii(decorator)
          << std::endl;

  deallog << "factor * bs_blf_dV: " << (factor * bs_blf_dV).as_ascii(decorator)
          << std::endl;
  deallog << "bs_blf_dV * factor: " << (bs_blf_dV * factor).as_ascii(decorator)
          << std::endl;


  deallog << "Binary integral: Unary" << std::endl;
  const auto bu_lf_dV  = u_lf_dV + u_lf_dV;
  const auto bu_blf_dV = u_blf_dV + u_blf_dV;

  deallog << "factor * bu_lf_dV: " << (factor * bu_lf_dV).as_ascii(decorator)
          << std::endl;
  deallog << "bu_lf_dV * factor: " << (bu_lf_dV * factor).as_ascii(decorator)
          << std::endl;

  deallog << "factor * bu_blf_dV: " << (factor * bu_blf_dV).as_ascii(decorator)
          << std::endl;
  deallog << "bu_blf_dV * factor: " << (bu_blf_dV * factor).as_ascii(decorator)
          << std::endl;


  deallog << "Binary integral: Unary" << std::endl;
  const auto bb_lf_dV  = bs_lf_dV + bu_lf_dV;
  const auto bb_blf_dV = bs_blf_dV + bu_blf_dV;

  deallog << "factor * bb_lf_dV: " << (factor * bb_lf_dV).as_ascii(decorator)
          << std::endl;
  deallog << "bb_lf_dV * factor: " << (bb_lf_dV * factor).as_ascii(decorator)
          << std::endl;

  deallog << "factor * bb_blf_dV: " << (factor * bb_blf_dV).as_ascii(decorator)
          << std::endl;
  deallog << "bb_blf_dV * factor: " << (bb_blf_dV * factor).as_ascii(decorator)
          << std::endl;

  deallog << "OK" << std::endl;
}
