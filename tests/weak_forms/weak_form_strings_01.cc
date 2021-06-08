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
    deallog << "Test function: " << value(test).as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: " << value(trial).as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << value(soln).as_ascii(decorator) << std::endl;
    deallog << "Solution (t1): " << value<1>(soln).as_ascii(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Gradient" << std::endl;
    deallog << "Test function: " << gradient(test).as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: " << gradient(trial).as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << gradient(soln).as_ascii(decorator) << std::endl;
    deallog << "Solution (t1): " << gradient<1>(soln).as_ascii(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Laplacian" << std::endl;
    deallog << "Test function: " << laplacian(test).as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: " << laplacian(trial).as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << laplacian(soln).as_ascii(decorator) << std::endl;
    deallog << "Solution (t1): " << laplacian<1>(soln).as_ascii(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Hessian" << std::endl;
    deallog << "Test function: " << hessian(test).as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: " << hessian(trial).as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << hessian(soln).as_ascii(decorator) << std::endl;
    deallog << "Solution (t1): " << hessian<1>(soln).as_ascii(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Third derivative" << std::endl;
    deallog << "Test function: " << third_derivative(test).as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: " << third_derivative(trial).as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << third_derivative(soln).as_ascii(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << third_derivative<1>(soln).as_ascii(decorator) << std::endl;

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
    deallog << "Test function: " << value(test).as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: " << value(trial).as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << value(soln).as_latex(decorator) << std::endl;
    deallog << "Solution (t1): " << value<1>(soln).as_latex(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Gradient" << std::endl;
    deallog << "Test function: " << gradient(test).as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: " << gradient(trial).as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << gradient(soln).as_latex(decorator) << std::endl;
    deallog << "Solution (t1): " << gradient<1>(soln).as_latex(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Laplacian" << std::endl;
    deallog << "Test function: " << laplacian(test).as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: " << laplacian(trial).as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << laplacian(soln).as_latex(decorator) << std::endl;
    deallog << "Solution (t1): " << laplacian<1>(soln).as_latex(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Hessian" << std::endl;
    deallog << "Test function: " << hessian(test).as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: " << hessian(trial).as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << hessian(soln).as_latex(decorator) << std::endl;
    deallog << "Solution (t1): " << hessian<1>(soln).as_latex(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Third derivative" << std::endl;
    deallog << "Test function: " << third_derivative(test).as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: " << third_derivative(trial).as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << third_derivative(soln).as_latex(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << third_derivative<1>(soln).as_latex(decorator) << std::endl;

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
