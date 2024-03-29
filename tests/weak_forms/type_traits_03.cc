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


#include <weak_forms/binary_operators.h>
#include <weak_forms/mixed_operators.h>
#include <weak_forms/spaces.h>
#include <weak_forms/symbolic_operators.h>
#include <weak_forms/type_traits.h>

#include "../weak_forms_tests.h"


int
main()
{
  initlog();

  using namespace WeakForms;

  constexpr int dim      = 2;
  constexpr int spacedim = 2;

  using test_t  = TestFunction<dim, spacedim>;
  using trial_t = TrialSolution<dim, spacedim>;
  using soln_t  = FieldSolution<dim, spacedim>;

  using test_val_t  = decltype(std::declval<test_t>().value());
  using trial_val_t = decltype(std::declval<trial_t>().value());
  using soln_val_t  = decltype(std::declval<soln_t>().value());

  using test_grad_t  = decltype(std::declval<test_t>().gradient());
  using trial_grad_t = decltype(std::declval<trial_t>().gradient());
  using soln_grad_t  = decltype(std::declval<soln_t>().gradient());

  deallog << std::boolalpha;

  // Values: <something> * <solution value>
  {
    LogStream::Prefix prefix("Value: RHS");

    deallog << "is_test_function_op()" << std::endl;
    deallog << is_test_function_op<decltype(std::declval<test_val_t>() *
                                            std::declval<soln_val_t>())>::value
            << std::endl;
    deallog << is_test_function_op<decltype(std::declval<trial_val_t>() *
                                            std::declval<soln_val_t>())>::value
            << std::endl;
    deallog << is_test_function_op<decltype(std::declval<soln_val_t>() *
                                            std::declval<soln_val_t>())>::value
            << std::endl;

    deallog << std::endl;

    deallog << "is_trial_solution_op()" << std::endl;
    deallog << is_trial_solution_op<decltype(std::declval<test_val_t>() *
                                             std::declval<soln_val_t>())>::value
            << std::endl;
    deallog << is_trial_solution_op<decltype(std::declval<trial_val_t>() *
                                             std::declval<soln_val_t>())>::value
            << std::endl;
    deallog << is_trial_solution_op<decltype(std::declval<soln_val_t>() *
                                             std::declval<soln_val_t>())>::value
            << std::endl;

    deallog << std::endl;

    deallog << "is_field_solution_op()" << std::endl;
    deallog << is_field_solution_op<decltype(std::declval<test_val_t>() *
                                             std::declval<soln_val_t>())>::value
            << std::endl;
    deallog << is_field_solution_op<decltype(std::declval<trial_val_t>() *
                                             std::declval<soln_val_t>())>::value
            << std::endl;
    deallog << is_field_solution_op<decltype(std::declval<soln_val_t>() *
                                             std::declval<soln_val_t>())>::value
            << std::endl;

    deallog << std::endl;
  }

  // Values: <solution value> * <something>
  {
    LogStream::Prefix prefix("Value: LHS");

    deallog << "is_test_function_op()" << std::endl;
    deallog << is_test_function_op<decltype(std::declval<soln_val_t>() *
                                            std::declval<test_val_t>())>::value
            << std::endl;
    deallog << is_test_function_op<decltype(std::declval<soln_val_t>() *
                                            std::declval<trial_val_t>())>::value
            << std::endl;
    deallog << is_test_function_op<decltype(std::declval<soln_val_t>() *
                                            std::declval<soln_val_t>())>::value
            << std::endl;

    deallog << std::endl;

    deallog << "is_trial_solution_op()" << std::endl;
    deallog << is_trial_solution_op<decltype(std::declval<soln_val_t>() *
                                             std::declval<test_val_t>())>::value
            << std::endl;
    deallog
      << is_trial_solution_op<decltype(std::declval<soln_val_t>() *
                                       std::declval<trial_val_t>())>::value
      << std::endl;
    deallog << is_trial_solution_op<decltype(std::declval<soln_val_t>() *
                                             std::declval<soln_val_t>())>::value
            << std::endl;

    deallog << std::endl;

    deallog << "is_field_solution_op()" << std::endl;
    deallog << is_field_solution_op<decltype(std::declval<soln_val_t>() *
                                             std::declval<test_val_t>())>::value
            << std::endl;
    deallog
      << is_field_solution_op<decltype(std::declval<soln_val_t>() *
                                       std::declval<trial_val_t>())>::value
      << std::endl;
    deallog << is_field_solution_op<decltype(std::declval<soln_val_t>() *
                                             std::declval<soln_val_t>())>::value
            << std::endl;

    deallog << std::endl;
  }

  // Gradients <something> * <solution gradient>
  {
    LogStream::Prefix prefix("Gradient: RHS");

    deallog << "is_test_function_op()" << std::endl;
    deallog << is_test_function_op<decltype(std::declval<test_val_t>() *
                                            std::declval<soln_grad_t>())>::value
            << std::endl;
    deallog << is_test_function_op<decltype(std::declval<trial_val_t>() *
                                            std::declval<soln_grad_t>())>::value
            << std::endl;
    deallog << is_test_function_op<decltype(std::declval<soln_val_t>() *
                                            std::declval<soln_grad_t>())>::value
            << std::endl;
    deallog << is_test_function_op<decltype(std::declval<soln_grad_t>() *
                                            std::declval<soln_grad_t>())>::value
            << std::endl;

    deallog << std::endl;

    deallog << "is_trial_solution_op()" << std::endl;
    deallog
      << is_trial_solution_op<decltype(std::declval<test_val_t>() *
                                       std::declval<soln_grad_t>())>::value
      << std::endl;
    deallog
      << is_trial_solution_op<decltype(std::declval<trial_val_t>() *
                                       std::declval<soln_grad_t>())>::value
      << std::endl;
    deallog
      << is_trial_solution_op<decltype(std::declval<soln_val_t>() *
                                       std::declval<soln_grad_t>())>::value
      << std::endl;
    deallog
      << is_trial_solution_op<decltype(std::declval<soln_grad_t>() *
                                       std::declval<soln_grad_t>())>::value
      << std::endl;

    deallog << std::endl;

    deallog << "is_field_solution_op()" << std::endl;
    deallog
      << is_field_solution_op<decltype(std::declval<test_val_t>() *
                                       std::declval<soln_grad_t>())>::value
      << std::endl;
    deallog
      << is_field_solution_op<decltype(std::declval<trial_val_t>() *
                                       std::declval<soln_grad_t>())>::value
      << std::endl;
    deallog
      << is_field_solution_op<decltype(std::declval<soln_val_t>() *
                                       std::declval<soln_grad_t>())>::value
      << std::endl;
    deallog
      << is_field_solution_op<decltype(std::declval<soln_grad_t>() *
                                       std::declval<soln_grad_t>())>::value
      << std::endl;

    deallog << std::endl;
  }

  // Gradients: <solution gradient> * <something>
  {
    LogStream::Prefix prefix("Value: LHS");

    deallog << "is_test_function_op()" << std::endl;
    deallog << is_test_function_op<decltype(std::declval<soln_grad_t>() *
                                            std::declval<test_val_t>())>::value
            << std::endl;
    deallog << is_test_function_op<decltype(std::declval<soln_grad_t>() *
                                            std::declval<trial_val_t>())>::value
            << std::endl;
    deallog << is_test_function_op<decltype(std::declval<soln_grad_t>() *
                                            std::declval<soln_val_t>())>::value
            << std::endl;
    deallog << is_test_function_op<decltype(std::declval<soln_grad_t>() *
                                            std::declval<soln_grad_t>())>::value
            << std::endl;

    deallog << std::endl;

    deallog << "is_trial_solution_op()" << std::endl;
    deallog << is_trial_solution_op<decltype(std::declval<soln_grad_t>() *
                                             std::declval<test_val_t>())>::value
            << std::endl;
    deallog
      << is_trial_solution_op<decltype(std::declval<soln_grad_t>() *
                                       std::declval<trial_val_t>())>::value
      << std::endl;
    deallog << is_trial_solution_op<decltype(std::declval<soln_grad_t>() *
                                             std::declval<soln_val_t>())>::value
            << std::endl;
    deallog
      << is_trial_solution_op<decltype(std::declval<soln_grad_t>() *
                                       std::declval<soln_grad_t>())>::value
      << std::endl;


    deallog << std::endl;

    deallog << "is_field_solution_op()" << std::endl;
    deallog << is_field_solution_op<decltype(std::declval<soln_grad_t>() *
                                             std::declval<test_val_t>())>::value
            << std::endl;
    deallog
      << is_field_solution_op<decltype(std::declval<soln_grad_t>() *
                                       std::declval<trial_val_t>())>::value
      << std::endl;
    deallog << is_field_solution_op<decltype(std::declval<soln_grad_t>() *
                                             std::declval<soln_val_t>())>::value
            << std::endl;
    deallog
      << is_field_solution_op<decltype(std::declval<soln_grad_t>() *
                                       std::declval<soln_grad_t>())>::value
      << std::endl;

    deallog << std::endl;
  }

  deallog << "OK" << std::endl;
}
