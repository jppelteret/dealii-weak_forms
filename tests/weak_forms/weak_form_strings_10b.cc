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
// - Sub-Space (interface): Vector


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

    deallog << "SPACE FUNCTIONS: Jump in values" << std::endl;
    deallog << "Test function: " << jump_in_values(test_ss).as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: "
            << jump_in_values(trial_ss).as_ascii(decorator) << std::endl;
    //     deallog << "Solution: " <<
    //     jump_in_values(soln_ss).as_ascii(decorator)
    //             << std::endl;
    //     deallog << "Solution (t1): " <<
    //     jump_in_values<1>(soln_ss).as_ascii(decorator)
    //             << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Jump in gradients" << std::endl;
    deallog << "Test function: "
            << jump_in_gradients(test_ss).as_ascii(decorator) << std::endl;
    deallog << "Trial solution: "
            << jump_in_gradients(trial_ss).as_ascii(decorator) << std::endl;
    //     deallog << "Solution: " <<
    //     jump_in_gradients(soln_ss).as_ascii(decorator)
    //             << std::endl;
    //     deallog << "Solution (t1): " <<
    //     jump_in_gradients<1>(soln_ss).as_ascii(decorator)
    //             << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Jump in hessians" << std::endl;
    deallog << "Test function: "
            << jump_in_hessians(test_ss).as_ascii(decorator) << std::endl;
    deallog << "Trial solution: "
            << jump_in_hessians(trial_ss).as_ascii(decorator) << std::endl;
    //     deallog << "Solution: " <<
    //     jump_in_hessians(soln_ss).as_ascii(decorator)
    //             << std::endl;
    //     deallog << "Solution (t1): " <<
    //     jump_in_hessians<1>(soln_ss).as_ascii(decorator)
    //             << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Jump in third derivatives" << std::endl;
    deallog << "Test function: "
            << jump_in_third_derivatives(test_ss).as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: "
            << jump_in_third_derivatives(trial_ss).as_ascii(decorator)
            << std::endl;
    //     deallog << "Solution: " <<
    //     jump_in_third_derivatives(soln_ss).as_ascii(decorator)
    //             << std::endl;
    //     deallog << "Solution (t1): " <<
    //     jump_in_third_derivatives<1>(soln_ss).as_ascii(decorator)
    //             << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Average of values" << std::endl;
    deallog << "Test function: "
            << average_of_values(test_ss).as_ascii(decorator) << std::endl;
    deallog << "Trial solution: "
            << average_of_values(trial_ss).as_ascii(decorator) << std::endl;
    //     deallog << "Solution: " <<
    //     average_of_values(soln_ss).as_ascii(decorator)
    //             << std::endl;
    //     deallog << "Solution (t1): " <<
    //     average_of_values<1>(soln_ss).as_ascii(decorator)
    //             << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Average of gradients" << std::endl;
    deallog << "Test function: "
            << average_of_gradients(test_ss).as_ascii(decorator) << std::endl;
    deallog << "Trial solution: "
            << average_of_gradients(trial_ss).as_ascii(decorator) << std::endl;
    //     deallog << "Solution: " <<
    //     average_of_gradients(soln_ss).as_ascii(decorator)
    //             << std::endl;
    //     deallog << "Solution (t1): " <<
    //     average_of_gradients<1>(soln_ss).as_ascii(decorator)
    //             << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Average of hessians" << std::endl;
    deallog << "Test function: "
            << average_of_hessians(test_ss).as_ascii(decorator) << std::endl;
    deallog << "Trial solution: "
            << average_of_hessians(trial_ss).as_ascii(decorator) << std::endl;
    //     deallog << "Solution: " <<
    //     average_of_hessians(soln_ss).as_ascii(decorator)
    //             << std::endl;
    //     deallog << "Solution (t1): " <<
    //     average_of_hessians<1>(soln_ss).as_ascii(decorator)
    //             << std::endl;

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

    deallog << "SPACE FUNCTIONS: Jump in values" << std::endl;
    deallog << "Test function: " << jump_in_values(test_ss).as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: "
            << jump_in_values(trial_ss).as_latex(decorator) << std::endl;
    //     deallog << "Solution: " <<
    //     jump_in_values(soln_ss).as_latex(decorator)
    //             << std::endl;
    //     deallog << "Solution (t1): " <<
    //     jump_in_values<1>(soln_ss).as_latex(decorator)
    //             << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Jump in gradients" << std::endl;
    deallog << "Test function: "
            << jump_in_gradients(test_ss).as_latex(decorator) << std::endl;
    deallog << "Trial solution: "
            << jump_in_gradients(trial_ss).as_latex(decorator) << std::endl;
    //     deallog << "Solution: " <<
    //     jump_in_gradients(soln_ss).as_latex(decorator)
    //             << std::endl;
    //     deallog << "Solution (t1): " <<
    //     jump_in_gradients<1>(soln_ss).as_latex(decorator)
    //             << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Jump in hessians" << std::endl;
    deallog << "Test function: "
            << jump_in_hessians(test_ss).as_latex(decorator) << std::endl;
    deallog << "Trial solution: "
            << jump_in_hessians(trial_ss).as_latex(decorator) << std::endl;
    //     deallog << "Solution: " <<
    //     jump_in_hessians(soln_ss).as_latex(decorator)
    //             << std::endl;
    //     deallog << "Solution (t1): " <<
    //     jump_in_hessians<1>(soln_ss).as_latex(decorator)
    //             << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Jump in third derivatives" << std::endl;
    deallog << "Test function: "
            << jump_in_third_derivatives(test_ss).as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: "
            << jump_in_third_derivatives(trial_ss).as_latex(decorator)
            << std::endl;
    //     deallog << "Solution: " <<
    //     jump_in_third_derivatives(soln_ss).as_latex(decorator)
    //             << std::endl;
    //     deallog << "Solution (t1): " <<
    //     jump_in_third_derivatives<1>(soln_ss).as_latex(decorator)
    //             << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Average of values" << std::endl;
    deallog << "Test function: "
            << average_of_values(test_ss).as_latex(decorator) << std::endl;
    deallog << "Trial solution: "
            << average_of_values(trial_ss).as_latex(decorator) << std::endl;
    //     deallog << "Solution: " <<
    //     average_of_values(soln_ss).as_latex(decorator)
    //             << std::endl;
    //     deallog << "Solution (t1): " <<
    //     average_of_values<1>(soln_ss).as_latex(decorator)
    //             << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Average of gradients" << std::endl;
    deallog << "Test function: "
            << average_of_gradients(test_ss).as_latex(decorator) << std::endl;
    deallog << "Trial solution: "
            << average_of_gradients(trial_ss).as_latex(decorator) << std::endl;
    //     deallog << "Solution: " <<
    //     average_of_gradients(soln_ss).as_latex(decorator)
    //             << std::endl;
    //     deallog << "Solution (t1): " <<
    //     average_of_gradients<1>(soln_ss).as_latex(decorator)
    //             << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Average of hessians" << std::endl;
    deallog << "Test function: "
            << average_of_hessians(test_ss).as_latex(decorator) << std::endl;
    deallog << "Trial solution: "
            << average_of_hessians(trial_ss).as_latex(decorator) << std::endl;
    //     deallog << "Solution: " <<
    //     average_of_hessians(soln_ss).as_latex(decorator)
    //             << std::endl;
    //     deallog << "Solution (t1): " <<
    //     average_of_hessians<1>(soln_ss).as_latex(decorator)
    //             << std::endl;

    deallog << std::endl;
  }

  deallog << "OK" << std::endl << std::endl;
}


int
main()
{
  initlog();

  const WeakForms::SubSpaceExtractors::Vector subspace_extractor(0,
                                                                 "u",
                                                                 "\\mathbf{u}");
  run<2>(subspace_extractor);

  deallog << "OK" << std::endl;
}