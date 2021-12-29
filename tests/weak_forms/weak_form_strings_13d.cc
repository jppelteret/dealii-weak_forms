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
// - Binary math operations: Symmetric tensor
//
// Based off of binary_operators_07


#include <weak_forms/binary_operators.h>
#include <weak_forms/functors.h>
#include <weak_forms/mixed_operators.h>
#include <weak_forms/symbolic_decorations.h>

#include "../weak_forms_tests.h"


template <int dim, int spacedim = dim, typename NumberType = double>
void
run()
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;

  // Customise the naming conventions, if we wish to.
  const SymbolicDecorations decorator;

  const SymmetricTensorFunctor<2, dim> S1("S", "S");
  const auto                           f1 = value<double, spacedim>(
    S1,
    [](const FEValuesBase<dim, spacedim> &, const unsigned int)
    {
      SymmetricTensor<2, dim> t;
      for (auto it = t.begin_raw(); it != t.end_raw(); ++it)
        *it = 2.0;
      for (unsigned int i = 0; i < dim; ++i)
        t[i][i] += 1.0;
      return t;
    });

  const SymmetricTensorFunctor<2, dim> S2("U", "U");
  const auto                           f2 = value<double, spacedim>(
    S2,
    [](const FEValuesBase<dim, spacedim> &, const unsigned int)
    {
      SymmetricTensor<2, dim> t;
      for (auto it = t.begin_raw(); it != t.end_raw(); ++it)
        *it = 3.0;
      for (unsigned int i = 0; i < dim; ++i)
        t[i][i] += 1.0;
      return t;
    });

  // Test strings
  {
    LogStream::Prefix prefix("string");

    deallog << "Outer product: " << outer_product(f1, f2).as_ascii(decorator)
            << std::endl;

    // deallog << "Schur product: "
    //           << schur_product(f1, f2).as_ascii(decorator)
    //           << std::endl;

    deallog << "Scalar product: " << scalar_product(f1, f2).as_ascii(decorator)
            << std::endl;
  }

  // Test LaTeX
  {
    LogStream::Prefix prefix("LaTeX");

    deallog << "Outer product: " << outer_product(f1, f2).as_latex(decorator)
            << std::endl;

    // deallog << "Schur product: "
    //           << schur_product(f1, f2).as_latex(decorator)
    //           << std::endl;

    deallog << "Scalar product: " << scalar_product(f1, f2).as_latex(decorator)
            << std::endl;
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