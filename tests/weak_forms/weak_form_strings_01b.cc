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
// - Sub-Space: Vector


#include <deal.II/fe/fe_values_extractors.h>

#include <weak_forms/spaces.h>
#include <weak_forms/subspace_extractors.h>
#include <weak_forms/subspace_views.h>
#include <weak_forms/symbolic_decorations.h>
#include <weak_forms/symbolic_operators.h>

#include "../weak_forms_tests.h"


template <int dim>
struct RunCurl;

template <>
struct RunCurl<2>
{
  template <typename TestSSType, typename TrialSSType, typename SolutionSSType>
  static void
  as_ascii(const TestSSType &                    test_ss,
           const TrialSSType &                   trial_ss,
           const SolutionSSType &                soln_ss,
           const WeakForms::SymbolicDecorations &decorator)
  {}

  template <typename TestSSType, typename TrialSSType, typename SolutionSSType>
  static void
  as_latex(const TestSSType &                    test_ss,
           const TrialSSType &                   trial_ss,
           const SolutionSSType &                soln_ss,
           const WeakForms::SymbolicDecorations &decorator)
  {}
};

template <>
struct RunCurl<3>
{
  template <typename TestSSType, typename TrialSSType, typename SolutionSSType>
  static void
  as_ascii(const TestSSType &                    test_ss,
           const TrialSSType &                   trial_ss,
           const SolutionSSType &                soln_ss,
           const WeakForms::SymbolicDecorations &decorator)
  {
    using namespace WeakForms;

    deallog << "SPACE FUNCTIONS: Curl" << std::endl;
    deallog << "Test function: " << curl(test_ss).as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: " << curl(trial_ss).as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << curl(soln_ss).as_ascii(decorator) << std::endl;
    deallog << "Solution (t1): " << curl<1>(soln_ss).as_ascii(decorator)
            << std::endl;

    deallog << std::endl;
  }

  template <typename TestSSType, typename TrialSSType, typename SolutionSSType>
  static void
  as_latex(const TestSSType &                    test_ss,
           const TrialSSType &                   trial_ss,
           const SolutionSSType &                soln_ss,
           const WeakForms::SymbolicDecorations &decorator)
  {
    using namespace WeakForms;

    deallog << "SPACE FUNCTIONS: Curl" << std::endl;
    deallog << "Test function: " << curl(test_ss).as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: " << curl(trial_ss).as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << curl(soln_ss).as_latex(decorator) << std::endl;
    deallog << "Solution (t1): " << curl<1>(soln_ss).as_latex(decorator)
            << std::endl;

    deallog << std::endl;
  }
};


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
    deallog << "Test function: " << value(test_ss).as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: " << value(trial_ss).as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << value(soln_ss).as_ascii(decorator) << std::endl;
    deallog << "Solution (t1): " << value<1>(soln_ss).as_ascii(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Gradient" << std::endl;
    deallog << "Test function: " << gradient(test_ss).as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: " << gradient(trial_ss).as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << gradient(soln_ss).as_ascii(decorator)
            << std::endl;
    deallog << "Solution (t1): " << gradient<1>(soln_ss).as_ascii(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Symmetric gradient" << std::endl;
    deallog << "Test function: "
            << symmetric_gradient(test_ss).as_ascii(decorator) << std::endl;
    deallog << "Trial solution: "
            << symmetric_gradient(trial_ss).as_ascii(decorator) << std::endl;
    deallog << "Solution: " << symmetric_gradient(soln_ss).as_ascii(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << symmetric_gradient<1>(soln_ss).as_ascii(decorator) << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Divergence" << std::endl;
    deallog << "Test function: " << divergence(test_ss).as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: " << divergence(trial_ss).as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << divergence(soln_ss).as_ascii(decorator)
            << std::endl;
    deallog << "Solution (t1): " << divergence<1>(soln_ss).as_ascii(decorator)
            << std::endl;

    deallog << std::endl;

    RunCurl<dim>::as_ascii(test_ss, trial_ss, soln_ss, decorator);

    deallog << "SPACE FUNCTIONS: Hessian" << std::endl;
    deallog << "Test function: " << hessian(test_ss).as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: " << hessian(trial_ss).as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << hessian(soln_ss).as_ascii(decorator)
            << std::endl;
    deallog << "Solution (t1): " << hessian<1>(soln_ss).as_ascii(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Third derivative" << std::endl;
    deallog << "Test function: "
            << third_derivative(test_ss).as_ascii(decorator) << std::endl;
    deallog << "Trial solution: "
            << third_derivative(trial_ss).as_ascii(decorator) << std::endl;
    deallog << "Solution: " << third_derivative(soln_ss).as_ascii(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << third_derivative<1>(soln_ss).as_ascii(decorator) << std::endl;

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
    deallog << "Test function: " << value(test_ss).as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: " << value(trial_ss).as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << value(soln_ss).as_latex(decorator) << std::endl;
    deallog << "Solution (t1): " << value<1>(soln_ss).as_latex(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Gradient" << std::endl;
    deallog << "Test function: " << gradient(test_ss).as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: " << gradient(trial_ss).as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << gradient(soln_ss).as_latex(decorator)
            << std::endl;
    deallog << "Solution (t1): " << gradient<1>(soln_ss).as_latex(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Symmetric gradient" << std::endl;
    deallog << "Test function: "
            << symmetric_gradient(test_ss).as_latex(decorator) << std::endl;
    deallog << "Trial solution: "
            << symmetric_gradient(trial_ss).as_latex(decorator) << std::endl;
    deallog << "Solution: " << symmetric_gradient(soln_ss).as_latex(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << symmetric_gradient<1>(soln_ss).as_latex(decorator) << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Divergence" << std::endl;
    deallog << "Test function: " << divergence(test_ss).as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: " << divergence(trial_ss).as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << divergence(soln_ss).as_latex(decorator)
            << std::endl;
    deallog << "Solution (t1): " << divergence<1>(soln_ss).as_latex(decorator)
            << std::endl;

    deallog << std::endl;

    RunCurl<dim>::as_latex(test_ss, trial_ss, soln_ss, decorator);

    deallog << "SPACE FUNCTIONS: Hessian" << std::endl;
    deallog << "Test function: " << hessian(test_ss).as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: " << hessian(trial_ss).as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << hessian(soln_ss).as_latex(decorator)
            << std::endl;
    deallog << "Solution (t1): " << hessian<1>(soln_ss).as_latex(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Third derivative" << std::endl;
    deallog << "Test function: "
            << third_derivative(test_ss).as_latex(decorator) << std::endl;
    deallog << "Trial solution: "
            << third_derivative(trial_ss).as_latex(decorator) << std::endl;
    deallog << "Solution: " << third_derivative(soln_ss).as_latex(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << third_derivative<1>(soln_ss).as_latex(decorator) << std::endl;

    deallog << std::endl;
  }
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
