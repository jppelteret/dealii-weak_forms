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


// Check type traits for integrals


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
  const auto u_blf_dV  = -s_blf_dV;
  const auto u_lf_dV   = -s_lf_dV;
  const auto u2_blf_dV = -u_blf_dV; // double negate
  const auto u2_lf_dV  = -u_lf_dV;  // double negate

  // Binary integral
  const auto b_blf_dV  = s_blf_dV + s_blf_dV;
  const auto b_lf_dV   = s_lf_dV + s_lf_dV;
  const auto b2_blf_dV = s_blf_dV + u_blf_dV;
  const auto b2_lf_dV  = u_lf_dV + s_lf_dV;
  const auto b3_blf_dV = u2_blf_dV + u_blf_dV;
  const auto b3_lf_dV  = u_lf_dV + u2_lf_dV;

  deallog << std::boolalpha;

  {
    LogStream::Prefix prefix("Integral op");

    // Symbolic
    deallog
      << is_integral_op<typename std::decay<decltype(s_blf_dV)>::type>::value
      << std::endl;
    deallog
      << is_integral_op<typename std::decay<decltype(s_lf_dV)>::type>::value
      << std::endl;

    // Unary
    deallog
      << is_integral_op<typename std::decay<decltype(u_blf_dV)>::type>::value
      << std::endl;
    deallog
      << is_integral_op<typename std::decay<decltype(u_lf_dV)>::type>::value
      << std::endl;

    deallog
      << is_integral_op<typename std::decay<decltype(u2_blf_dV)>::type>::value
      << std::endl;
    deallog
      << is_integral_op<typename std::decay<decltype(u2_lf_dV)>::type>::value
      << std::endl;

    // Binary
    deallog
      << is_integral_op<typename std::decay<decltype(b_blf_dV)>::type>::value
      << std::endl;
    deallog
      << is_integral_op<typename std::decay<decltype(b_lf_dV)>::type>::value
      << std::endl;

    deallog
      << is_integral_op<typename std::decay<decltype(b2_blf_dV)>::type>::value
      << std::endl;
    deallog
      << is_integral_op<typename std::decay<decltype(b2_lf_dV)>::type>::value
      << std::endl;

    deallog
      << is_integral_op<typename std::decay<decltype(b3_blf_dV)>::type>::value
      << std::endl;
    deallog
      << is_integral_op<typename std::decay<decltype(b3_lf_dV)>::type>::value
      << std::endl;

    deallog << "OK" << std::endl;
  }


  {
    LogStream::Prefix prefix("Symbolic integral op");

    // Symbolic
    deallog << is_symbolic_integral_op<
                 typename std::decay<decltype(s_blf_dV)>::type>::value
            << std::endl;
    deallog << is_symbolic_integral_op<
                 typename std::decay<decltype(s_lf_dV)>::type>::value
            << std::endl;

    // Unary
    deallog << is_symbolic_integral_op<
                 typename std::decay<decltype(u_blf_dV)>::type>::value
            << std::endl;
    deallog << is_symbolic_integral_op<
                 typename std::decay<decltype(u_lf_dV)>::type>::value
            << std::endl;

    deallog << is_symbolic_integral_op<
                 typename std::decay<decltype(u2_blf_dV)>::type>::value
            << std::endl;
    deallog << is_symbolic_integral_op<
                 typename std::decay<decltype(u2_lf_dV)>::type>::value
            << std::endl;

    // Binary
    deallog << is_symbolic_integral_op<
                 typename std::decay<decltype(b_blf_dV)>::type>::value
            << std::endl;
    deallog << is_symbolic_integral_op<
                 typename std::decay<decltype(b_lf_dV)>::type>::value
            << std::endl;

    deallog << is_symbolic_integral_op<
                 typename std::decay<decltype(b2_blf_dV)>::type>::value
            << std::endl;
    deallog << is_symbolic_integral_op<
                 typename std::decay<decltype(b2_lf_dV)>::type>::value
            << std::endl;

    deallog << is_symbolic_integral_op<
                 typename std::decay<decltype(b3_blf_dV)>::type>::value
            << std::endl;
    deallog << is_symbolic_integral_op<
                 typename std::decay<decltype(b3_lf_dV)>::type>::value
            << std::endl;

    deallog << "OK" << std::endl;
  }

  {
    LogStream::Prefix prefix("Unary integral op");

    // Symbolic
    deallog << is_unary_integral_op<
                 typename std::decay<decltype(s_blf_dV)>::type>::value
            << std::endl;
    deallog << is_unary_integral_op<
                 typename std::decay<decltype(s_lf_dV)>::type>::value
            << std::endl;

    // Unary
    deallog << is_unary_integral_op<
                 typename std::decay<decltype(u_blf_dV)>::type>::value
            << std::endl;
    deallog << is_unary_integral_op<
                 typename std::decay<decltype(u_lf_dV)>::type>::value
            << std::endl;

    deallog << is_unary_integral_op<
                 typename std::decay<decltype(u2_blf_dV)>::type>::value
            << std::endl;
    deallog << is_unary_integral_op<
                 typename std::decay<decltype(u2_lf_dV)>::type>::value
            << std::endl;

    // Binary
    deallog << is_unary_integral_op<
                 typename std::decay<decltype(b_blf_dV)>::type>::value
            << std::endl;
    deallog << is_unary_integral_op<
                 typename std::decay<decltype(b_lf_dV)>::type>::value
            << std::endl;

    deallog << is_unary_integral_op<
                 typename std::decay<decltype(b2_blf_dV)>::type>::value
            << std::endl;
    deallog << is_unary_integral_op<
                 typename std::decay<decltype(b2_lf_dV)>::type>::value
            << std::endl;

    deallog << is_unary_integral_op<
                 typename std::decay<decltype(b3_blf_dV)>::type>::value
            << std::endl;
    deallog << is_unary_integral_op<
                 typename std::decay<decltype(b3_lf_dV)>::type>::value
            << std::endl;

    deallog << "OK" << std::endl;
  }

  {
    LogStream::Prefix prefix("Binary integral op");

    // Symbolic
    deallog << is_binary_integral_op<
                 typename std::decay<decltype(s_blf_dV)>::type>::value
            << std::endl;
    deallog << is_binary_integral_op<
                 typename std::decay<decltype(s_lf_dV)>::type>::value
            << std::endl;

    // Unary
    deallog << is_binary_integral_op<
                 typename std::decay<decltype(u_blf_dV)>::type>::value
            << std::endl;
    deallog << is_binary_integral_op<
                 typename std::decay<decltype(u_lf_dV)>::type>::value
            << std::endl;

    deallog << is_binary_integral_op<
                 typename std::decay<decltype(u2_blf_dV)>::type>::value
            << std::endl;
    deallog << is_binary_integral_op<
                 typename std::decay<decltype(u2_lf_dV)>::type>::value
            << std::endl;

    // Binary
    deallog << is_binary_integral_op<
                 typename std::decay<decltype(b_blf_dV)>::type>::value
            << std::endl;
    deallog << is_binary_integral_op<
                 typename std::decay<decltype(b_lf_dV)>::type>::value
            << std::endl;

    deallog << is_binary_integral_op<
                 typename std::decay<decltype(b2_blf_dV)>::type>::value
            << std::endl;
    deallog << is_binary_integral_op<
                 typename std::decay<decltype(b2_lf_dV)>::type>::value
            << std::endl;

    deallog << is_binary_integral_op<
                 typename std::decay<decltype(b3_blf_dV)>::type>::value
            << std::endl;
    deallog << is_binary_integral_op<
                 typename std::decay<decltype(b3_lf_dV)>::type>::value
            << std::endl;

    deallog << "OK" << std::endl;
  }

  deallog << "OK" << std::endl;
}
