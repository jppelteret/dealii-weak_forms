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
// - Sub-Space: Scalar


#include <deal.II/fe/fe_values_extractors.h>

#include <weak_forms/spaces.h>
#include <weak_forms/subspace_extractors.h>
#include <weak_forms/subspace_views.h>
#include <weak_forms/symbolic_decorations.h>
#include <weak_forms/symbolic_operators.h>

#include "../weak_forms_tests.h"


template <int dim,
          int spacedim        = dim,
          typename NumberType = double,
          typename SubSpaceExtractorType>
void
run(const SubSpaceExtractorType &subspace_extractor)
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;

  // Customise the naming convensions, if we wish to.
  const SymbolicDecorations decorator;

  const TestFunction<dim, spacedim>  test;
  const TrialSolution<dim, spacedim> trial;
  const FieldSolution<dim, spacedim> soln;

  const auto test_ss  = test[subspace_extractor];
  const auto trial_ss = trial[subspace_extractor];
  const auto soln_ss  = soln[subspace_extractor];

  // Test strings
  {
    LogStream::Prefix prefix("string");

    deallog << "SPACE CREATION" << std::endl;
    deallog << "Test function: " << test_ss.as_ascii(decorator) << std::endl;
    deallog << "Trial solution: " << trial_ss.as_ascii(decorator) << std::endl;
    deallog << "Solution: " << soln_ss.as_ascii(decorator) << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Value" << std::endl;
    deallog << "Test function: " << test_ss.value().as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: " << trial_ss.value().as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << soln_ss.value().as_ascii(decorator) << std::endl;
    deallog << "Solution (t1): "
            << soln_ss.template value<1>().as_ascii(decorator) << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Gradient" << std::endl;
    deallog << "Test function: " << test_ss.gradient().as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: " << trial_ss.gradient().as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << soln_ss.gradient().as_ascii(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln_ss.template gradient<1>().as_ascii(decorator) << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Laplacian" << std::endl;
    deallog << "Test function: " << test_ss.laplacian().as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: " << trial_ss.laplacian().as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << soln_ss.laplacian().as_ascii(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln_ss.template laplacian<1>().as_ascii(decorator) << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Hessian" << std::endl;
    deallog << "Test function: " << test_ss.hessian().as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: " << trial_ss.hessian().as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << soln_ss.hessian().as_ascii(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln_ss.template hessian<1>().as_ascii(decorator) << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Third derivative" << std::endl;
    deallog << "Test function: "
            << test_ss.third_derivative().as_ascii(decorator) << std::endl;
    deallog << "Trial solution: "
            << trial_ss.third_derivative().as_ascii(decorator) << std::endl;
    deallog << "Solution: " << soln_ss.third_derivative().as_ascii(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln_ss.template third_derivative<1>().as_ascii(decorator)
            << std::endl;

    deallog << std::endl;
  }

  // Test LaTeX
  {
    LogStream::Prefix prefix("LaTeX");

    deallog << "SPACE CREATION" << std::endl;
    deallog << "Test function: " << test_ss.as_latex(decorator) << std::endl;
    deallog << "Trial solution: " << trial_ss.as_latex(decorator) << std::endl;
    deallog << "Solution: " << soln_ss.as_latex(decorator) << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Value" << std::endl;
    deallog << "Test function: " << test_ss.value().as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: " << trial_ss.value().as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << soln_ss.value().as_latex(decorator) << std::endl;
    deallog << "Solution (t1): "
            << soln_ss.template value<1>().as_latex(decorator) << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Gradient" << std::endl;
    deallog << "Test function: " << test_ss.gradient().as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: " << trial_ss.gradient().as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << soln_ss.gradient().as_latex(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln_ss.template gradient<1>().as_latex(decorator) << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Laplacian" << std::endl;
    deallog << "Test function: " << test_ss.laplacian().as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: " << trial_ss.laplacian().as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << soln_ss.laplacian().as_latex(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln_ss.template laplacian<1>().as_latex(decorator) << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Hessian" << std::endl;
    deallog << "Test function: " << test_ss.hessian().as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: " << trial_ss.hessian().as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << soln_ss.hessian().as_latex(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln_ss.template hessian<1>().as_latex(decorator) << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Third derivative" << std::endl;
    deallog << "Test function: "
            << test_ss.third_derivative().as_latex(decorator) << std::endl;
    deallog << "Trial solution: "
            << trial_ss.third_derivative().as_latex(decorator) << std::endl;
    deallog << "Solution: " << soln_ss.third_derivative().as_latex(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln_ss.template third_derivative<1>().as_latex(decorator)
            << std::endl;

    deallog << std::endl;
  }
}


int
main()
{
  initlog();

  const WeakForms::SubSpaceExtractors::Scalar subspace_extractor(0, "s", "s");
  run<2>(subspace_extractor);

  deallog << "OK" << std::endl;
}
