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


// Check type traits for space functions


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

  // Values
  {
    LogStream::Prefix prefix("Value");

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
  }

  // Gradients
  {
    LogStream::Prefix prefix("Gradient");

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
  }

  deallog << "OK" << std::endl;
}
