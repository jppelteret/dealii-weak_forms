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
    deallog << "Test function: " << test_ss.curl().as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: " << trial_ss.curl().as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << soln_ss.curl().as_ascii(decorator) << std::endl;
    deallog << "Solution (t1): "
            << soln_ss.template curl<1>().as_ascii(decorator) << std::endl;

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
    deallog << "Test function: " << test_ss.curl().as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: " << trial_ss.curl().as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << soln_ss.curl().as_latex(decorator) << std::endl;
    deallog << "Solution (t1): "
            << soln_ss.template curl<1>().as_latex(decorator) << std::endl;

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

    deallog << "SPACE FUNCTIONS: Symmetric gradient" << std::endl;
    deallog << "Test function: "
            << test_ss.symmetric_gradient().as_ascii(decorator) << std::endl;
    deallog << "Trial solution: "
            << trial_ss.symmetric_gradient().as_ascii(decorator) << std::endl;
    deallog << "Solution: " << soln_ss.symmetric_gradient().as_ascii(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln_ss.template symmetric_gradient<1>().as_ascii(decorator)
            << std::endl;

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

    RunCurl<dim>::as_ascii(test_ss, trial_ss, soln_ss, decorator);

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

    deallog << "SPACE FUNCTIONS: Symmetric gradient" << std::endl;
    deallog << "Test function: "
            << test_ss.symmetric_gradient().as_latex(decorator) << std::endl;
    deallog << "Trial solution: "
            << trial_ss.symmetric_gradient().as_latex(decorator) << std::endl;
    deallog << "Solution: " << soln_ss.symmetric_gradient().as_latex(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln_ss.template symmetric_gradient<1>().as_latex(decorator)
            << std::endl;

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

    RunCurl<dim>::as_latex(test_ss, trial_ss, soln_ss, decorator);

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

  const WeakForms::SubSpaceExtractors::Vector subspace_extractor(0,
                                                                 "u",
                                                                 "\\mathbf{u}");
  run<2>(subspace_extractor);

  deallog << "OK" << std::endl;
}
