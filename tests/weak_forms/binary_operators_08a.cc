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


// Check that binary math operators work
// - A continuation of binary_operators_05a.cc
// - Subspaces

#include <weak_forms/weak_forms.h>

#include "../weak_forms_tests.h"


template <int dim, int spacedim = dim, typename NumberType = double>
void
run()
{
  using namespace dealiiWeakForms::WeakForms;

  // Customise the naming conventions, if we wish to.
  const SymbolicDecorations decorator;

  const SubSpaceExtractors::Scalar subspace_extractor(0, "s", "s");

  const TestFunction<dim, spacedim>  test;
  const TrialSolution<dim, spacedim> trial;
  const FieldSolution<dim, spacedim> soln;
  const NumberType                   factor(2.0);

  const auto test_val  = test[subspace_extractor].value();
  const auto trial_val = trial[subspace_extractor].value();
  const auto soln_val  = soln[subspace_extractor].value();

  {
    std::cout << "Linear form: "
              << (factor * linear_form(factor * test_val, factor * soln_val))
                   .as_ascii(decorator)
              << std::endl;

    std::cout << "Bilinear form: "
              << (factor * bilinear_form(factor * test_val,
                                         factor * soln_val,
                                         factor * trial_val))
                   .as_ascii(decorator)
              << std::endl;

    std::cout << std::endl;
  }

  {
    std::cout << "Linear form: "
              << (factor *
                  linear_form(factor * test[subspace_extractor].value(),
                              factor * soln[subspace_extractor].value()))
                   .as_ascii(decorator)
              << std::endl;

    std::cout << "Bilinear form: "
              << (factor *
                  bilinear_form(factor * test[subspace_extractor].value(),
                                factor * soln[subspace_extractor].value(),
                                factor * trial[subspace_extractor].value()))
                   .as_ascii(decorator)
              << std::endl;

    std::cout << std::endl;
  }
}


int
main(int argc, char *argv[])
{
  initlog();
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  run<2>();

  deallog << "OK" << std::endl;
}
