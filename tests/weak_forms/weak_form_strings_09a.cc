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


// Check assembly using convenience converter in forms
// - Functor helpers: Scalar, Vector


#include <weak_forms/assembler_matrix_based.h>
#include <weak_forms/bilinear_forms.h>
#include <weak_forms/functors.h>
#include <weak_forms/linear_forms.h>
#include <weak_forms/mixed_operators.h>
#include <weak_forms/spaces.h>
#include <weak_forms/subspace_extractors.h>
#include <weak_forms/subspace_views.h>
#include <weak_forms/symbolic_decorations.h>

#include "../weak_forms_tests.h"


template <int dim, int spacedim = dim, typename NumberType = double>
void
run()
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;

  auto print_assembler = [](const MatrixBasedAssembler<dim> &assembler)
  {
    // Look at what we're going to compute
    const SymbolicDecorations decorator;
    deallog << "Weak form (ascii):\n"
            << assembler.as_ascii(decorator) << std::endl;
    deallog << "Weak form (LaTeX):\n"
            << assembler.as_latex(decorator) << std::endl;
    deallog << "OK" << std::endl;
  };


  // Scalar functor
  {
    LogStream::Prefix prefix("Scalar");

    const double s = 1.0;

    // Symbolic types for test function, trial solution and a coefficient.
    const TestFunction<dim>  test;
    const TrialSolution<dim> trial;

    const WeakForms::SubSpaceExtractors::Scalar subspace_extractor(0, "s", "s");

    const auto test_ss   = test[subspace_extractor];
    const auto test_val  = test_ss.value();
    const auto test_grad = test_ss.gradient();

    const auto trial_ss   = trial[subspace_extractor];
    const auto trial_grad = trial_ss.gradient();

    MatrixBasedAssembler<dim> assembler;
    assembler += bilinear_form(test_grad, s, trial_grad).dV() +
                 linear_form(test_val, s).dA();
    print_assembler(assembler);
  }

  // Vector functor
  {
    LogStream::Prefix prefix("Vector");

    const Tensor<1, dim> v;

    // Symbolic types for test function, trial solution and a coefficient.
    const TestFunction<dim>  test;
    const TrialSolution<dim> trial;

    const WeakForms::SubSpaceExtractors::Scalar subspace_extractor(0, "s", "s");

    const auto test_ss   = test[subspace_extractor];
    const auto test_grad = test_ss.gradient();

    const auto trial_ss   = trial[subspace_extractor];
    const auto trial_hess = trial_ss.hessian();

    MatrixBasedAssembler<dim> assembler;
    assembler += bilinear_form(test_grad, v, trial_hess).dV() +
                 linear_form(test_grad, v).dA();
    print_assembler(assembler);
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
