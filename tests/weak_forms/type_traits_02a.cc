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


// Check type traits for space functions

#include <weak_forms/weak_forms.h>

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

  // Values
  {
    LogStream::Prefix prefix("Value:Symbolic");

    deallog << "is_test_function_op()" << std::endl;
    deallog << is_test_function_op<test_val_t>::value << std::endl;
    deallog << is_test_function_op<trial_val_t>::value << std::endl;
    deallog << is_test_function_op<soln_val_t>::value << std::endl;

    deallog << std::endl;

    deallog << "is_trial_solution_op()" << std::endl;
    deallog << is_trial_solution_op<test_val_t>::value << std::endl;
    deallog << is_trial_solution_op<trial_val_t>::value << std::endl;
    deallog << is_trial_solution_op<soln_val_t>::value << std::endl;

    deallog << std::endl;

    deallog << "is_field_solution_op()" << std::endl;
    deallog << is_field_solution_op<test_val_t>::value << std::endl;
    deallog << is_field_solution_op<trial_val_t>::value << std::endl;
    deallog << is_field_solution_op<soln_val_t>::value << std::endl;

    deallog << std::endl;

    deallog << "is_or_has_test_function_op()" << std::endl;
    deallog << is_or_has_test_function_op<test_val_t>::value << std::endl;
    deallog << is_or_has_test_function_op<trial_val_t>::value << std::endl;
    deallog << is_or_has_test_function_op<soln_val_t>::value << std::endl;

    deallog << std::endl;

    deallog << "is_or_has_trial_solution_op()" << std::endl;
    deallog << is_or_has_trial_solution_op<test_val_t>::value << std::endl;
    deallog << is_or_has_trial_solution_op<trial_val_t>::value << std::endl;
    deallog << is_or_has_trial_solution_op<soln_val_t>::value << std::endl;

    deallog << std::endl;

    deallog << "has_field_solution_op()" << std::endl;
    deallog << has_field_solution_op<test_val_t>::value << std::endl;
    deallog << has_field_solution_op<trial_val_t>::value << std::endl;
    deallog << has_field_solution_op<soln_val_t>::value << std::endl;

    deallog << std::endl;

    // It doesn't make sense for us to have a is_or_has_field_solution_op type
    // trait, but we get the equivalent information by looking at whether the
    // type is evaluated with cached data.
    deallog << "is_or_has_evaluated_with_scratch_data()" << std::endl;
    deallog << is_or_has_evaluated_with_scratch_data<test_val_t>::value
            << std::endl;
    deallog << is_or_has_evaluated_with_scratch_data<trial_val_t>::value
            << std::endl;
    deallog << is_or_has_evaluated_with_scratch_data<soln_val_t>::value
            << std::endl;

    deallog << std::endl;
  }
  {
    LogStream::Prefix prefix("Value:Unary");

    deallog << "is_test_function_op()" << std::endl;
    deallog << is_test_function_op<decltype(-std::declval<test_val_t>())>::value
            << std::endl;
    deallog
      << is_test_function_op<decltype(-std::declval<trial_val_t>())>::value
      << std::endl;
    deallog << is_test_function_op<decltype(-std::declval<soln_val_t>())>::value
            << std::endl;

    deallog << std::endl;

    deallog << "is_trial_solution_op()" << std::endl;
    deallog
      << is_trial_solution_op<decltype(-std::declval<test_val_t>())>::value
      << std::endl;
    deallog
      << is_trial_solution_op<decltype(-std::declval<trial_val_t>())>::value
      << std::endl;
    deallog
      << is_trial_solution_op<decltype(-std::declval<soln_val_t>())>::value
      << std::endl;

    deallog << std::endl;

    deallog << "is_field_solution_op()" << std::endl;
    deallog
      << is_field_solution_op<decltype(-std::declval<test_val_t>())>::value
      << std::endl;
    deallog
      << is_field_solution_op<decltype(-std::declval<trial_val_t>())>::value
      << std::endl;
    deallog
      << is_field_solution_op<decltype(-std::declval<soln_val_t>())>::value
      << std::endl;

    deallog << std::endl;

    deallog << "is_or_has_test_function_op()" << std::endl;
    deallog << is_or_has_test_function_op<decltype(
                 -std::declval<test_val_t>())>::value
            << std::endl;
    deallog << is_or_has_test_function_op<decltype(
                 -std::declval<trial_val_t>())>::value
            << std::endl;
    deallog << is_or_has_test_function_op<decltype(
                 -std::declval<soln_val_t>())>::value
            << std::endl;

    deallog << std::endl;

    deallog << "is_or_has_trial_solution_op()" << std::endl;
    deallog << is_or_has_trial_solution_op<decltype(
                 -std::declval<test_val_t>())>::value
            << std::endl;
    deallog << is_or_has_trial_solution_op<decltype(
                 -std::declval<trial_val_t>())>::value
            << std::endl;
    deallog << is_or_has_trial_solution_op<decltype(
                 -std::declval<soln_val_t>())>::value
            << std::endl;

    deallog << std::endl;

    deallog << "has_field_solution_op()" << std::endl;
    deallog
      << has_field_solution_op<decltype(-std::declval<test_val_t>())>::value
      << std::endl;
    deallog
      << has_field_solution_op<decltype(-std::declval<trial_val_t>())>::value
      << std::endl;
    deallog
      << has_field_solution_op<decltype(-std::declval<soln_val_t>())>::value
      << std::endl;

    deallog << std::endl;

    deallog << "is_or_has_evaluated_with_scratch_data()" << std::endl;
    deallog << is_or_has_evaluated_with_scratch_data<decltype(
                 -std::declval<test_val_t>())>::value
            << std::endl;
    deallog << is_or_has_evaluated_with_scratch_data<decltype(
                 -std::declval<trial_val_t>())>::value
            << std::endl;
    deallog << is_or_has_evaluated_with_scratch_data<decltype(
                 -std::declval<soln_val_t>())>::value
            << std::endl;

    deallog << std::endl;
  }

  // Gradients
  {
    LogStream::Prefix prefix("Gradient:Symbolic");

    deallog << "is_test_function_op()" << std::endl;
    deallog << is_test_function_op<test_grad_t>::value << std::endl;
    deallog << is_test_function_op<trial_grad_t>::value << std::endl;
    deallog << is_test_function_op<soln_grad_t>::value << std::endl;

    deallog << std::endl;

    deallog << "is_trial_solution_op()" << std::endl;
    deallog << is_trial_solution_op<test_grad_t>::value << std::endl;
    deallog << is_trial_solution_op<trial_grad_t>::value << std::endl;
    deallog << is_trial_solution_op<soln_grad_t>::value << std::endl;

    deallog << std::endl;

    deallog << "is_field_solution_op()" << std::endl;
    deallog << is_field_solution_op<test_grad_t>::value << std::endl;
    deallog << is_field_solution_op<trial_grad_t>::value << std::endl;
    deallog << is_field_solution_op<soln_grad_t>::value << std::endl;

    deallog << std::endl;

    deallog << "is_or_has_test_function_op()" << std::endl;
    deallog << is_or_has_test_function_op<test_grad_t>::value << std::endl;
    deallog << is_or_has_test_function_op<trial_grad_t>::value << std::endl;
    deallog << is_or_has_test_function_op<soln_grad_t>::value << std::endl;

    deallog << std::endl;

    deallog << "is_or_has_trial_solution_op()" << std::endl;
    deallog << is_or_has_trial_solution_op<test_grad_t>::value << std::endl;
    deallog << is_or_has_trial_solution_op<trial_grad_t>::value << std::endl;
    deallog << is_or_has_trial_solution_op<soln_grad_t>::value << std::endl;

    deallog << std::endl;

    deallog << "has_field_solution_op()" << std::endl;
    deallog << has_field_solution_op<test_grad_t>::value << std::endl;
    deallog << has_field_solution_op<trial_grad_t>::value << std::endl;
    deallog << has_field_solution_op<soln_grad_t>::value << std::endl;

    deallog << std::endl;

    deallog << "is_or_has_evaluated_with_scratch_data()" << std::endl;
    deallog << is_or_has_evaluated_with_scratch_data<test_grad_t>::value
            << std::endl;
    deallog << is_or_has_evaluated_with_scratch_data<trial_grad_t>::value
            << std::endl;
    deallog << is_or_has_evaluated_with_scratch_data<soln_grad_t>::value
            << std::endl;

    deallog << std::endl;
  }
  {
    LogStream::Prefix prefix("Gradient:Unary");

    deallog << "is_test_function_op()" << std::endl;
    deallog
      << is_test_function_op<decltype(-std::declval<test_grad_t>())>::value
      << std::endl;
    deallog
      << is_test_function_op<decltype(-std::declval<trial_grad_t>())>::value
      << std::endl;
    deallog
      << is_test_function_op<decltype(-std::declval<soln_grad_t>())>::value
      << std::endl;

    deallog << std::endl;

    deallog << "is_trial_solution_op()" << std::endl;
    deallog
      << is_trial_solution_op<decltype(-std::declval<test_grad_t>())>::value
      << std::endl;
    deallog
      << is_trial_solution_op<decltype(-std::declval<trial_grad_t>())>::value
      << std::endl;
    deallog
      << is_trial_solution_op<decltype(-std::declval<soln_grad_t>())>::value
      << std::endl;

    deallog << std::endl;

    deallog << "is_field_solution_op()" << std::endl;
    deallog
      << is_field_solution_op<decltype(-std::declval<test_grad_t>())>::value
      << std::endl;
    deallog
      << is_field_solution_op<decltype(-std::declval<trial_grad_t>())>::value
      << std::endl;
    deallog
      << is_field_solution_op<decltype(-std::declval<soln_grad_t>())>::value
      << std::endl;

    deallog << std::endl;

    deallog << "is_or_has_test_function_op()" << std::endl;
    deallog << is_or_has_test_function_op<decltype(
                 -std::declval<test_grad_t>())>::value
            << std::endl;
    deallog << is_or_has_test_function_op<decltype(
                 -std::declval<trial_grad_t>())>::value
            << std::endl;
    deallog << is_or_has_test_function_op<decltype(
                 -std::declval<soln_grad_t>())>::value
            << std::endl;

    deallog << std::endl;

    deallog << "is_or_has_trial_solution_op()" << std::endl;
    deallog << is_or_has_trial_solution_op<decltype(
                 -std::declval<test_grad_t>())>::value
            << std::endl;
    deallog << is_or_has_trial_solution_op<decltype(
                 -std::declval<trial_grad_t>())>::value
            << std::endl;
    deallog << is_or_has_trial_solution_op<decltype(
                 -std::declval<soln_grad_t>())>::value
            << std::endl;

    deallog << std::endl;

    deallog << "has_field_solution_op()" << std::endl;
    deallog
      << has_field_solution_op<decltype(-std::declval<test_grad_t>())>::value
      << std::endl;
    deallog
      << has_field_solution_op<decltype(-std::declval<trial_grad_t>())>::value
      << std::endl;
    deallog
      << has_field_solution_op<decltype(-std::declval<soln_grad_t>())>::value
      << std::endl;

    deallog << std::endl;

    deallog << "is_or_has_evaluated_with_scratch_data()" << std::endl;
    deallog << is_or_has_evaluated_with_scratch_data<decltype(
                 -std::declval<test_grad_t>())>::value
            << std::endl;
    deallog << is_or_has_evaluated_with_scratch_data<decltype(
                 -std::declval<trial_grad_t>())>::value
            << std::endl;
    deallog << is_or_has_evaluated_with_scratch_data<decltype(
                 -std::declval<soln_grad_t>())>::value
            << std::endl;

    deallog << std::endl;
  }

  deallog << "OK" << std::endl;
}
