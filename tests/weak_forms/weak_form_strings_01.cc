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
// - Spaces


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

  const TestFunction<dim, spacedim>  test;
  const TrialSolution<dim, spacedim> trial;
  const FieldSolution<dim, spacedim> soln;

  // Test strings
  {
    LogStream::Prefix prefix("string");

    deallog << "SPACE CREATION" << std::endl;
    deallog << "Test function: " << test.as_ascii(decorator) << std::endl;
    deallog << "Trial solution: " << trial.as_ascii(decorator) << std::endl;
    deallog << "Solution: " << soln.as_ascii(decorator) << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Value" << std::endl;
    deallog << "Test function: " << test.value().as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: " << trial.value().as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << soln.value().as_ascii(decorator) << std::endl;
    deallog << "Solution (t1): " << soln.template value<1>().as_ascii(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Gradient" << std::endl;
    deallog << "Test function: " << test.gradient().as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: " << trial.gradient().as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << soln.gradient().as_ascii(decorator) << std::endl;
    deallog << "Solution (t1): "
            << soln.template gradient<1>().as_ascii(decorator) << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Laplacian" << std::endl;
    deallog << "Test function: " << test.laplacian().as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: " << trial.laplacian().as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << soln.laplacian().as_ascii(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln.template laplacian<1>().as_ascii(decorator) << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Hessian" << std::endl;
    deallog << "Test function: " << test.hessian().as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: " << trial.hessian().as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << soln.hessian().as_ascii(decorator) << std::endl;
    deallog << "Solution (t1): "
            << soln.template hessian<1>().as_ascii(decorator) << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Third derivative" << std::endl;
    deallog << "Test function: " << test.third_derivative().as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: "
            << trial.third_derivative().as_ascii(decorator) << std::endl;
    deallog << "Solution: " << soln.third_derivative().as_ascii(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln.template third_derivative<1>().as_ascii(decorator)
            << std::endl;

    deallog << std::endl;
  }

  // Test LaTeX
  {
    LogStream::Prefix prefix("LaTeX");

    deallog << "SPACE CREATION" << std::endl;
    deallog << "Test function: " << test.as_latex(decorator) << std::endl;
    deallog << "Trial solution: " << trial.as_latex(decorator) << std::endl;
    deallog << "Solution: " << soln.as_latex(decorator) << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Value" << std::endl;
    deallog << "Test function: " << test.value().as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: " << trial.value().as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << soln.value().as_latex(decorator) << std::endl;
    deallog << "Solution (t1): " << soln.template value<1>().as_latex(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Gradient" << std::endl;
    deallog << "Test function: " << test.gradient().as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: " << trial.gradient().as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << soln.gradient().as_latex(decorator) << std::endl;
    deallog << "Solution (t1): "
            << soln.template gradient<1>().as_latex(decorator) << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Laplacian" << std::endl;
    deallog << "Test function: " << test.laplacian().as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: " << trial.laplacian().as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << soln.laplacian().as_latex(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln.template laplacian<1>().as_latex(decorator) << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Hessian" << std::endl;
    deallog << "Test function: " << test.hessian().as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: " << trial.hessian().as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << soln.hessian().as_latex(decorator) << std::endl;
    deallog << "Solution (t1): "
            << soln.template hessian<1>().as_latex(decorator) << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Third derivative" << std::endl;
    deallog << "Test function: " << test.third_derivative().as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: "
            << trial.third_derivative().as_latex(decorator) << std::endl;
    deallog << "Solution: " << soln.third_derivative().as_latex(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln.template third_derivative<1>().as_latex(decorator)
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
