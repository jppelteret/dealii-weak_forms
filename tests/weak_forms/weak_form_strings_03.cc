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


// Check weak form stringization and printing
// - Unary, binary operations with spaces


#include <weak_forms/binary_operators.h>
#include <weak_forms/mixed_operators.h>
#include <weak_forms/spaces.h>
#include <weak_forms/symbolic_decorations.h>
#include <weak_forms/symbolic_operators.h>

#include "../weak_forms_tests.h"


template <int dim, int spacedim = dim, typename NumberType = double>
void
run()
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;

  // Customise the naming convensions, if we wish to.
  const SymbolicDecorations decorator;

  const TestFunction<dim, spacedim>  test_1;
  const TrialSolution<dim, spacedim> trial_1;
  const FieldSolution<dim, spacedim> soln_1;

  const TestFunction<dim, spacedim>  test_2;
  const TrialSolution<dim, spacedim> trial_2;
  const FieldSolution<dim, spacedim> soln_2;

  const auto test_val_1  = value(test_1);
  const auto trial_val_1 = value(trial_1);
  const auto soln_val_1  = value(soln_1);
  const auto soln_grad_1 = gradient(soln_1);

  const auto test_val_2  = value(test_2);
  const auto trial_val_2 = value(trial_2);
  const auto soln_val_2  = value(soln_2);
  const auto soln_grad_2 = gradient(soln_2);

  // What we're going to do here doesn't make much sense, since the test
  // function and trial solution represents the entire finite element space. But
  // this is the basis on which we'll construct the operations for subspaces.

  // Test strings
  {
    LogStream::Prefix prefix("string");

    deallog << "Addition: " << (test_val_1 + test_val_2).as_ascii(decorator)
            << std::endl;

    deallog << "Subtraction: " << (test_val_1 - test_val_2).as_ascii(decorator)
            << std::endl;

    deallog << "Addition: " << (soln_val_1 + soln_val_2).as_ascii(decorator)
            << std::endl;

    deallog << "Multiplication 1: "
            << (test_val_1 * soln_val_1).as_ascii(decorator) << std::endl;

    deallog << "Multiplication 2: "
            << (soln_val_1 * trial_val_1).as_ascii(decorator) << std::endl;

    deallog << "Multiplication 3: "
            << (test_val_1 * soln_grad_1).as_ascii(decorator) << std::endl;

    deallog << "Multiplication 4: "
            << (soln_grad_1 * soln_grad_2).as_ascii(decorator) << std::endl;

    deallog << "Compound 1: "
            << (test_val_1 * (soln_val_1 - soln_val_2) * trial_val_1)
                 .as_ascii(decorator)
            << std::endl;

    deallog << "Compound 2: "
            << ((test_val_1 + test_val_2) * (soln_val_1 + soln_val_2) *
                (trial_val_1 + trial_val_2))
                 .as_ascii(decorator)
            << std::endl;

    deallog << std::endl;
  }

  // Note: These would throw a compile-time error, as they are not permissible.
  //
  // {
  // deallog << "Addition: " << (test_val_1 + soln_val_1).as_ascii(decorator)
  //         << std::endl;

  // deallog << "Subtraction: " << (test_val_1 - soln_val_1).as_ascii(decorator)
  //         << std::endl;

  // deallog << "Multiplication: " << (test_val_1 *
  // soln_val_1).as_ascii(decorator)
  //         << std::endl;

  // deallog << "Compound: " << (test_val_1 * (soln_val_1 - trial_val_1) +
  // soln_val_1).as_ascii(decorator)
  //         << std::endl;
  // }

  // Test LaTeX
  {
    LogStream::Prefix prefix("LaTeX");

    deallog << std::endl;

    deallog << "Addition: " << (test_val_1 + test_val_2).as_latex(decorator)
            << std::endl;

    deallog << "Subtraction: " << (test_val_1 - test_val_2).as_latex(decorator)
            << std::endl;

    deallog << "Addition: " << (soln_val_1 + soln_val_2).as_latex(decorator)
            << std::endl;

    deallog << "Multiplication 1: "
            << (test_val_1 * soln_val_1).as_latex(decorator) << std::endl;

    deallog << "Multiplication 2: "
            << (soln_val_1 * trial_val_1).as_latex(decorator) << std::endl;

    deallog << "Multiplication 3: "
            << (test_val_1 * soln_grad_1).as_latex(decorator) << std::endl;

    deallog << "Multiplication 4: "
            << (soln_grad_1 * soln_grad_2).as_latex(decorator) << std::endl;

    deallog << "Compound 1: "
            << (test_val_1 * (soln_val_1 - soln_val_2) * trial_val_1)
                 .as_latex(decorator)
            << std::endl;

    deallog << "Compound 2: "
            << ((test_val_1 + test_val_2) * (soln_val_1 + soln_val_2) *
                (trial_val_1 + trial_val_2))
                 .as_latex(decorator)
            << std::endl;
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
