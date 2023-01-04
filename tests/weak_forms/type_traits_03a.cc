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


// Check type traits for space function operations
// - Unary operators


#include <weak_forms/mixed_operators.h>
#include <weak_forms/spaces.h>
#include <weak_forms/subspace_extractors.h>
#include <weak_forms/subspace_views.h>
#include <weak_forms/symbolic_operators.h>
#include <weak_forms/type_traits.h>
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

  const SubSpaceExtractors::Scalar subspace_extractor_sclr(0, "s", "s");
  const SubSpaceExtractors::Vector subspace_extractor_vec(1,
                                                          "u",
                                                          "\\mathbf{u}");

  // Scalar
  const auto test_s_val   = test[subspace_extractor_sclr].value();
  const auto test_s_grad  = test[subspace_extractor_sclr].gradient();
  const auto test_s_hess  = test[subspace_extractor_sclr].hessian();
  const auto trial_s_val  = trial[subspace_extractor_sclr].value();
  const auto trial_s_grad = trial[subspace_extractor_sclr].gradient();
  const auto trial_s_hess = trial[subspace_extractor_sclr].hessian();
  const auto soln_s_val   = soln[subspace_extractor_sclr].value();
  const auto soln_s_grad  = soln[subspace_extractor_sclr].gradient();
  const auto soln_s_hess  = soln[subspace_extractor_sclr].hessian();

  // Vector
  const auto test_v_val   = test[subspace_extractor_vec].value();
  const auto test_v_grad  = test[subspace_extractor_vec].gradient();
  const auto trial_v_val  = trial[subspace_extractor_vec].value();
  const auto trial_v_grad = trial[subspace_extractor_vec].gradient();
  const auto soln_v_val   = soln[subspace_extractor_vec].value();
  const auto soln_v_grad  = soln[subspace_extractor_vec].gradient();

  deallog << std::boolalpha;

  // Standard symbolic op
  {
    LogStream::Prefix prefix("Symbolic op");
    deallog << "has_test_function_op()" << std::endl;
    deallog << has_test_function_op<decltype(test_s_val)>::value << std::endl;
    deallog << has_test_function_op<decltype(test_s_grad)>::value << std::endl;
    deallog << has_test_function_op<decltype(test_v_val)>::value << std::endl;
    deallog << has_test_function_op<decltype(test_v_grad)>::value << std::endl;

    deallog << "has_trial_solution_op()" << std::endl;
    deallog << has_trial_solution_op<decltype(trial_s_val)>::value << std::endl;
    deallog << has_trial_solution_op<decltype(trial_s_grad)>::value
            << std::endl;
    deallog << has_trial_solution_op<decltype(trial_v_val)>::value << std::endl;
    deallog << has_trial_solution_op<decltype(trial_v_grad)>::value
            << std::endl;

    deallog << "has_field_solution_op()" << std::endl;
    deallog << has_field_solution_op<decltype(soln_s_val)>::value << std::endl;
    deallog << has_field_solution_op<decltype(soln_s_grad)>::value << std::endl;
    deallog << has_field_solution_op<decltype(soln_v_val)>::value << std::endl;
    deallog << has_field_solution_op<decltype(soln_v_grad)>::value << std::endl;
  }

  // Negation
  {
    LogStream::Prefix prefix("Negation");
    deallog << "has_test_function_op()" << std::endl;
    deallog << has_test_function_op<decltype(-test_s_val)>::value << std::endl;
    deallog << has_test_function_op<decltype(-test_s_grad)>::value << std::endl;
    deallog << has_test_function_op<decltype(-test_v_val)>::value << std::endl;
    deallog << has_test_function_op<decltype(-test_v_grad)>::value << std::endl;

    deallog << "has_trial_solution_op()" << std::endl;
    deallog << has_trial_solution_op<decltype(-trial_s_val)>::value
            << std::endl;
    deallog << has_trial_solution_op<decltype(-trial_s_grad)>::value
            << std::endl;
    deallog << has_trial_solution_op<decltype(-trial_v_val)>::value
            << std::endl;
    deallog << has_trial_solution_op<decltype(-trial_v_grad)>::value
            << std::endl;

    deallog << "has_field_solution_op()" << std::endl;
    deallog << has_field_solution_op<decltype(-soln_s_val)>::value << std::endl;
    deallog << has_field_solution_op<decltype(-soln_s_grad)>::value
            << std::endl;
    deallog << has_field_solution_op<decltype(-soln_v_val)>::value << std::endl;
    deallog << has_field_solution_op<decltype(-soln_v_grad)>::value
            << std::endl;
  }

  // Square root
  {
    LogStream::Prefix prefix("Square root");

    deallog << "has_field_solution_op()" << std::endl;
    deallog << has_field_solution_op<decltype(sqrt(soln_s_val))>::value
            << std::endl;
  }

  // Determinant
  {
    LogStream::Prefix prefix("Determinant");
    //     deallog << "has_test_function_op()" << std::endl;
    //     deallog <<
    //     has_test_function_op<decltype(determinant(test_s_hess))>::value
    //             << std::endl;
    //     deallog <<
    //     has_test_function_op<decltype(determinant(test_v_grad))>::value
    //             << std::endl;

    //     deallog << "has_trial_solution_op()" << std::endl;
    //     deallog <<
    //     has_trial_solution_op<decltype(determinant(trial_s_hess))>::value
    //             << std::endl;
    //     deallog <<
    //     has_trial_solution_op<decltype(determinant(trial_v_grad))>::value
    //             << std::endl;

    deallog << "has_field_solution_op()" << std::endl;
    deallog << has_field_solution_op<decltype(determinant(soln_s_hess))>::value
            << std::endl;
    deallog << has_field_solution_op<decltype(determinant(soln_v_grad))>::value
            << std::endl;
  }

  // Inverse
  {
    LogStream::Prefix prefix("Invert");
    //     deallog << "has_test_function_op()" << std::endl;
    //     deallog << has_test_function_op<decltype(invert(test_s_hess))>::value
    //             << std::endl;
    //     deallog << has_test_function_op<decltype(invert(test_v_grad))>::value
    //             << std::endl;

    //     deallog << "has_trial_solution_op()" << std::endl;
    //     deallog <<
    //     has_trial_solution_op<decltype(invert(trial_s_hess))>::value
    //             << std::endl;
    //     deallog <<
    //     has_trial_solution_op<decltype(invert(trial_v_grad))>::value
    //             << std::endl;

    deallog << "has_field_solution_op()" << std::endl;
    deallog << has_field_solution_op<decltype(invert(soln_s_hess))>::value
            << std::endl;
    deallog << has_field_solution_op<decltype(invert(soln_v_grad))>::value
            << std::endl;
  }

  // Transpose
  {
    LogStream::Prefix prefix("Transpose");
    deallog << "has_test_function_op()" << std::endl;
    deallog << has_test_function_op<decltype(transpose(test_s_hess))>::value
            << std::endl;
    deallog << has_test_function_op<decltype(transpose(test_v_grad))>::value
            << std::endl;

    deallog << "has_trial_solution_op()" << std::endl;
    deallog << has_trial_solution_op<decltype(transpose(trial_s_hess))>::value
            << std::endl;
    deallog << has_trial_solution_op<decltype(transpose(trial_v_grad))>::value
            << std::endl;

    deallog << "has_field_solution_op()" << std::endl;
    deallog << has_field_solution_op<decltype(transpose(soln_s_hess))>::value
            << std::endl;
    deallog << has_field_solution_op<decltype(transpose(soln_v_grad))>::value
            << std::endl;
  }

  // Symmetrize
  {
    LogStream::Prefix prefix("Symmetrize");
    deallog << "has_test_function_op()" << std::endl;
    deallog << has_test_function_op<decltype(symmetrize(test_s_hess))>::value
            << std::endl;
    deallog << has_test_function_op<decltype(symmetrize(test_v_grad))>::value
            << std::endl;

    deallog << "has_trial_solution_op()" << std::endl;
    deallog << has_trial_solution_op<decltype(symmetrize(trial_s_hess))>::value
            << std::endl;
    deallog << has_trial_solution_op<decltype(symmetrize(trial_v_grad))>::value
            << std::endl;

    deallog << "has_field_solution_op()" << std::endl;
    deallog << has_field_solution_op<decltype(symmetrize(soln_s_hess))>::value
            << std::endl;
    deallog << has_field_solution_op<decltype(symmetrize(soln_v_grad))>::value
            << std::endl;
  }

  // Compound operations
  {
    LogStream::Prefix prefix("Compound");
    deallog << "has_test_function_op()" << std::endl;
    deallog << has_test_function_op<decltype(-(-test_s_val))>::value
            << std::endl;
    deallog << is_test_function_op<decltype(-(-test_s_val))>::value
            << std::endl;
    //     deallog <<
    //     has_test_function_op<decltype(determinant(-test_s_hess))>::value
    //             << std::endl;
    //     deallog << has_test_function_op<decltype(
    //                  -invert(transpose(symmetrize(test_s_hess))))>::value
    //             << std::endl;
    //     deallog << has_test_function_op<decltype(
    //                  -transpose(symmetrize(invert(test_v_grad))))>::value
    //             << std::endl;

    deallog << "has_trial_solution_op()" << std::endl;
    deallog << has_trial_solution_op<decltype(-(-trial_s_val))>::value
            << std::endl;
    deallog << is_trial_solution_op<decltype(-(-trial_s_val))>::value
            << std::endl;
    //     deallog
    //       <<
    //       has_trial_solution_op<decltype(determinant(-trial_s_hess))>::value
    //       << std::endl;
    //     deallog << has_trial_solution_op<decltype(
    //                  -invert(transpose(symmetrize(trial_s_hess))))>::value
    //             << std::endl;
    //     deallog << has_trial_solution_op<decltype(
    //                  -transpose(symmetrize(invert(trial_v_grad))))>::value
    //             << std::endl;

    deallog << "has_field_solution_op()" << std::endl;
    deallog << has_field_solution_op<decltype(-(-soln_s_val))>::value
            << std::endl;
    deallog << is_field_solution_op<decltype(-(-soln_s_val))>::value
            << std::endl;
    deallog << has_field_solution_op<decltype(determinant(-soln_s_hess))>::value
            << std::endl;
    deallog << has_field_solution_op<decltype(
                 -invert(transpose(symmetrize(soln_s_hess))))>::value
            << std::endl;
    deallog << has_field_solution_op<decltype(
                 -transpose(symmetrize(invert(soln_v_grad))))>::value
            << std::endl;
  }

  deallog << "OK" << std::endl;
}
