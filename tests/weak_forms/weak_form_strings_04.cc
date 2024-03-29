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


// Check weak form stringization and printing
// - Linear, bilinear forms


#include <weak_forms/bilinear_forms.h>
#include <weak_forms/linear_forms.h>
#include <weak_forms/spaces.h>
#include <weak_forms/symbolic_decorations.h>

#include "../weak_forms_tests.h"


template <int dim, int spacedim = dim, typename NumberType = double>
void
run()
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;

  // Customise the naming convensions, if we wish to.
  const SymbolicDecorations decorator;

  const TestFunction<dim, spacedim>  test;
  const TrialSolution<dim, spacedim> trial;
  const FieldSolution<dim, spacedim> soln;

  const auto test_val  = test.value();
  const auto trial_val = trial.value();
  const auto soln_val  = soln.value();

  // Test strings
  {
    LogStream::Prefix prefix("string");

    deallog << "Linear form: "
            << linear_form(test_val, soln_val).as_ascii(decorator) << std::endl;

    deallog << "Bilinear form: "
            << bilinear_form(test_val, soln_val, trial_val).as_ascii(decorator)
            << std::endl;

    deallog << std::endl;
  }

  // Test LaTeX
  {
    LogStream::Prefix prefix("LaTeX");

    deallog << "Linear form: "
            << linear_form(test_val, soln_val).as_latex(decorator) << std::endl;

    deallog << "Bilinear form: "
            << bilinear_form(test_val, soln_val, trial_val).as_latex(decorator)
            << std::endl;

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
