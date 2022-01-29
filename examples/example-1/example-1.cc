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


// Demonstration of how to create and use spaces:
// - Global spaces
// - Linear and bilinear forms
// - Subspace extractors and views

#include <weak_forms/weak_forms.h>


template <int dim, int spacedim = dim>
void
run_global_space()
{
  using namespace dealiiWeakForms::WeakForms;

  // Customise the naming conventions, if we wish to.
  const SymbolicDecorations decorator;

  const TestFunction<dim, spacedim>  test;
  const TrialSolution<dim, spacedim> trial;
  const FieldSolution<dim, spacedim> soln;

  const auto test_val  = test.value();
  const auto trial_val = trial.value();
  const auto soln_val  = soln.value();

  // Test strings
  {
    std::cout << "ASCII output" << std::endl;

    std::cout << "Linear form: "
              << linear_form(test_val, soln_val).as_ascii(decorator)
              << std::endl;

    std::cout
      << "Bilinear form: "
      << bilinear_form(test_val, soln_val, trial_val).as_ascii(decorator)
      << std::endl;

    std::cout << std::endl;
  }

  // Test LaTeX
  {
    std::cout << "LaTeX output" << std::endl;

    std::cout << "Linear form: "
              << linear_form(test_val, soln_val).as_latex(decorator)
              << std::endl;

    std::cout
      << "Bilinear form: "
      << bilinear_form(test_val, soln_val, trial_val).as_latex(decorator)
      << std::endl;

    std::cout << std::endl;
  }
}


template <int dim, int spacedim = dim>
void
run_scalar_subspace()
{
  using namespace dealiiWeakForms::WeakForms;

  // Customise the naming conventions, if we wish to.
  const SymbolicDecorations decorator;

  const SubSpaceExtractors::Scalar subspace_extractor(0, "s", "s");

  const TestFunction<dim, spacedim>  test;
  const TrialSolution<dim, spacedim> trial;
  const FieldSolution<dim, spacedim> soln;

  // Test strings
  {
    std::cout << "ASCII output" << std::endl;

    std::cout << "Linear form: "
              << linear_form(test[subspace_extractor].value(),
                             soln[subspace_extractor].value())
                   .as_ascii(decorator)
              << std::endl;

    std::cout << "Bilinear form: "
              << bilinear_form(test[subspace_extractor].value(),
                               soln[subspace_extractor].value(),
                               trial[subspace_extractor].value())
                   .as_ascii(decorator)
              << std::endl;

    std::cout << std::endl;
  }

  // Test LaTeX
  {
    std::cout << "LaTeX output" << std::endl;

    std::cout << "Linear form: "
              << linear_form(test[subspace_extractor].value(),
                             soln[subspace_extractor].value())
                   .as_latex(decorator)
              << std::endl;

    std::cout << "Bilinear form: "
              << bilinear_form(test[subspace_extractor].value(),
                               soln[subspace_extractor].value(),
                               trial[subspace_extractor].value())
                   .as_latex(decorator)
              << std::endl;

    std::cout << std::endl;
  }
}


int
main()
{
  run_global_space<2>();
  run_scalar_subspace<2>();
}
