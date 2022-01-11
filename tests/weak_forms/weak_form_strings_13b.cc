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
// - Binary math operations: Vector
//
// Based off of binary_operators_07


#include <weak_forms/binary_operators.h>
#include <weak_forms/functors.h>
#include <weak_forms/mixed_operators.h>
#include <weak_forms/symbolic_decorations.h>

#include "../weak_forms_tests.h"


// Work around the issue that the cross product is not available
// in 2d.
template <int spacedim>
struct CrossProductCheck;

template <>
struct CrossProductCheck<2>
{
  template <typename... Args>
  static void
  as_ascii(const Args &...)
  {}

  template <typename... Args>
  static void
  as_latex(const Args &...)
  {}
};

template <>
struct CrossProductCheck<3>
{
  template <typename F1, typename F2>
  static void
  as_ascii(const F1 &                            f1,
           const F2 &                            f2,
           const WeakForms::SymbolicDecorations &decorator)
  {
    deallog << "Cross product: " << cross_product(f1, f2).as_ascii(decorator)
            << std::endl;
  }

  template <typename F1, typename F2>
  static void
  as_latex(const F1 &                            f1,
           const F2 &                            f2,
           const WeakForms::SymbolicDecorations &decorator)
  {
    deallog << "Cross product: " << cross_product(f1, f2).as_latex(decorator)
            << std::endl;
  }
};


template <int dim, int spacedim = dim, typename NumberType = double>
void
run()
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;

  // Customise the naming conventions, if we wish to.
  const SymbolicDecorations decorator;

  const VectorFunctor<dim> v1("v", "v");
  const auto               f1 = v1.template value<double, spacedim>(
    [](const FEValuesBase<dim, spacedim> &, const unsigned int)
    {
      Tensor<1, dim> t;
      for (auto it = t.begin_raw(); it != t.end_raw(); ++it)
        *it = 2.0;
      return t;
    });

  const VectorFunctor<dim> v2("w", "w");
  const auto               f2 = v2.template value<double, spacedim>(
    [](const FEValuesBase<dim, spacedim> &, const unsigned int)
    {
      Tensor<1, dim> t;
      unsigned int   i = 0;
      for (auto it = t.begin_raw(); it != t.end_raw(); ++it)
        *it = 3.0 + (i++);
      return t;
    });

  // Test strings
  {
    LogStream::Prefix prefix("string");

    CrossProductCheck<spacedim>::as_ascii(f1, f2, decorator);

    deallog << "Outer product: " << outer_product(f1, f2).as_ascii(decorator)
            << std::endl;

    deallog << "Schur product: " << schur_product(f1, f2).as_ascii(decorator)
            << std::endl;

    deallog << "Scalar product: " << scalar_product(f1, f2).as_ascii(decorator)
            << std::endl;

    deallog << "Contract: " << contract<0, 0>(f1, f2).as_ascii(decorator)
            << std::endl;
  }

  // Test LaTeX
  {
    LogStream::Prefix prefix("LaTeX");

    CrossProductCheck<spacedim>::as_latex(f1, f2, decorator);

    deallog << "Outer product: " << outer_product(f1, f2).as_latex(decorator)
            << std::endl;

    deallog << "Schur product: " << schur_product(f1, f2).as_latex(decorator)
            << std::endl;

    deallog << "Scalar product: " << scalar_product(f1, f2).as_latex(decorator)
            << std::endl;

    deallog << "Contract: " << contract<0, 0>(f1, f2).as_latex(decorator)
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
