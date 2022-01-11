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
// - Sub-Space: SymmetricTensor


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

    deallog << "SPACE FUNCTIONS: Divergence" << std::endl;
    deallog << "Test function: " << test_ss.divergence().as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: " << trial_ss.divergence().as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << soln_ss.divergence().as_ascii(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln_ss.template divergence<1>().as_ascii(decorator)
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

    deallog << "SPACE FUNCTIONS: Divergence" << std::endl;
    deallog << "Test function: " << test_ss.divergence().as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: " << trial_ss.divergence().as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << soln_ss.divergence().as_latex(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln_ss.template divergence<1>().as_latex(decorator)
            << std::endl;

    deallog << std::endl;
  }
}


int
main()
{
  initlog();

  constexpr int                                              rank = 2;
  const WeakForms::SubSpaceExtractors::SymmetricTensor<rank> subspace_extractor(
    0, "S", "\\mathbf{S}");
  run<2>(subspace_extractor);

  deallog << "OK" << std::endl;
}
