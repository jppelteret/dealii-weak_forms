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
// - Spaces (interface)


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

    deallog << "SPACE FUNCTIONS: Jump in values" << std::endl;
    deallog << "Test function: " << test.jump_in_values().as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: " << trial.jump_in_values().as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << soln.jump_in_values().as_ascii(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln.template jump_in_values<1>().as_ascii(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Jump in gradients" << std::endl;
    deallog << "Test function: " << test.jump_in_gradients().as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: "
            << trial.jump_in_gradients().as_ascii(decorator) << std::endl;
    deallog << "Solution: " << soln.jump_in_gradients().as_ascii(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln.template jump_in_gradients<1>().as_ascii(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Jump in hessians" << std::endl;
    deallog << "Test function: " << test.jump_in_hessians().as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: "
            << trial.jump_in_hessians().as_ascii(decorator) << std::endl;
    deallog << "Solution: " << soln.jump_in_hessians().as_ascii(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln.template jump_in_hessians<1>().as_ascii(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Jump in third derivatives" << std::endl;
    deallog << "Test function: "
            << test.jump_in_third_derivatives().as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: "
            << trial.jump_in_third_derivatives().as_ascii(decorator)
            << std::endl;
    deallog << "Solution: "
            << soln.jump_in_third_derivatives().as_ascii(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln.template jump_in_third_derivatives<1>().as_ascii(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Average of values" << std::endl;
    deallog << "Test function: " << test.average_of_values().as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: "
            << trial.average_of_values().as_ascii(decorator) << std::endl;
    deallog << "Solution: " << soln.average_of_values().as_ascii(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln.template average_of_values<1>().as_ascii(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Average of gradients" << std::endl;
    deallog << "Test function: "
            << test.average_of_gradients().as_ascii(decorator) << std::endl;
    deallog << "Trial solution: "
            << trial.average_of_gradients().as_ascii(decorator) << std::endl;
    deallog << "Solution: " << soln.average_of_gradients().as_ascii(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln.template average_of_gradients<1>().as_ascii(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Average of hessians" << std::endl;
    deallog << "Test function: "
            << test.average_of_hessians().as_ascii(decorator) << std::endl;
    deallog << "Trial solution: "
            << trial.average_of_hessians().as_ascii(decorator) << std::endl;
    deallog << "Solution: " << soln.average_of_hessians().as_ascii(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln.template average_of_hessians<1>().as_ascii(decorator)
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

    deallog << "SPACE FUNCTIONS: Jump in values" << std::endl;
    deallog << "Test function: " << test.jump_in_values().as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: " << trial.jump_in_values().as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << soln.jump_in_values().as_latex(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln.template jump_in_values<1>().as_latex(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Jump in gradients" << std::endl;
    deallog << "Test function: " << test.jump_in_gradients().as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: "
            << trial.jump_in_gradients().as_latex(decorator) << std::endl;
    deallog << "Solution: " << soln.jump_in_gradients().as_latex(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln.template jump_in_gradients<1>().as_latex(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Jump in hessians" << std::endl;
    deallog << "Test function: " << test.jump_in_hessians().as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: "
            << trial.jump_in_hessians().as_latex(decorator) << std::endl;
    deallog << "Solution: " << soln.jump_in_hessians().as_latex(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln.template jump_in_hessians<1>().as_latex(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Jump in third derivatives" << std::endl;
    deallog << "Test function: "
            << test.jump_in_third_derivatives().as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: "
            << trial.jump_in_third_derivatives().as_latex(decorator)
            << std::endl;
    deallog << "Solution: "
            << soln.jump_in_third_derivatives().as_latex(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln.template jump_in_third_derivatives<1>().as_latex(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Average of values" << std::endl;
    deallog << "Test function: " << test.average_of_values().as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: "
            << trial.average_of_values().as_latex(decorator) << std::endl;
    deallog << "Solution: " << soln.average_of_values().as_latex(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln.template average_of_values<1>().as_latex(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Average of gradients" << std::endl;
    deallog << "Test function: "
            << test.average_of_gradients().as_latex(decorator) << std::endl;
    deallog << "Trial solution: "
            << trial.average_of_gradients().as_latex(decorator) << std::endl;
    deallog << "Solution: " << soln.average_of_gradients().as_latex(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln.template average_of_gradients<1>().as_latex(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Average of hessians" << std::endl;
    deallog << "Test function: "
            << test.average_of_hessians().as_latex(decorator) << std::endl;
    deallog << "Trial solution: "
            << trial.average_of_hessians().as_latex(decorator) << std::endl;
    deallog << "Solution: " << soln.average_of_hessians().as_latex(decorator)
            << std::endl;
    deallog << "Solution (t1): "
            << soln.template average_of_hessians<1>().as_latex(decorator)
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
