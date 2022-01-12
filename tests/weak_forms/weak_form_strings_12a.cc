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
// - Unary math operations: Scalar
//
// Based off of unary_operators_01


#include <weak_forms/functors.h>
#include <weak_forms/mixed_operators.h>
#include <weak_forms/symbolic_decorations.h>
#include <weak_forms/unary_operators.h>

#include "../weak_forms_tests.h"


template <int dim, int spacedim = dim, typename NumberType = double>
void
run()
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;

  // Customise the naming conventions, if we wish to.
  const SymbolicDecorations decorator;

  const ScalarFunctor c1("c", "c");
  const auto          f1 = c1.template value<double, dim, spacedim>(
    [](const FEValuesBase<dim, spacedim> &, const unsigned int)
    { return 2.0; });

  // Test strings
  {
    LogStream::Prefix prefix("string");

    deallog << "Negation: " << (-f1).as_ascii(decorator) << std::endl;

    deallog << "Trigonometric sine: " << sin(f1).as_ascii(decorator)
            << std::endl;

    deallog << "Trigonometric cosine: " << cos(f1).as_ascii(decorator)
            << std::endl;

    deallog << "Trigonometric tangent: " << tan(f1).as_ascii(decorator)
            << std::endl;

    deallog << "Exponential: " << exp(f1).as_ascii(decorator) << std::endl;

    deallog << "Logarithm: " << log(f1).as_ascii(decorator) << std::endl;

    deallog << "Square root: " << sqrt(f1).as_ascii(decorator) << std::endl;

    deallog << "Absolute value: " << abs(-f1).as_ascii(decorator) << std::endl;
  }

  // Test LaTeX
  {
    LogStream::Prefix prefix("LaTeX");

    deallog << "Negation: " << (-f1).as_latex(decorator) << std::endl;

    deallog << "Trigonometric sine: " << sin(f1).as_latex(decorator)
            << std::endl;

    deallog << "Trigonometric cosine: " << cos(f1).as_latex(decorator)
            << std::endl;

    deallog << "Trigonometric tangent: " << tan(f1).as_latex(decorator)
            << std::endl;

    deallog << "Exponential: " << exp(f1).as_latex(decorator) << std::endl;

    deallog << "Logarithm: " << log(f1).as_latex(decorator) << std::endl;

    deallog << "Square root: " << sqrt(f1).as_latex(decorator) << std::endl;

    deallog << "Absolute value: " << abs(-f1).as_latex(decorator) << std::endl;
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
