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
// - Unary, binary operations that require some care to print correctly.
//   These situations arise when the operators render some ambiguity in
//   their interpretation.


#include <weak_forms/binary_operators.h>
#include <weak_forms/mixed_operators.h>
#include <weak_forms/spaces.h>
#include <weak_forms/subspace_extractors.h>
#include <weak_forms/subspace_views.h>
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

  const WeakForms::SubSpaceExtractors::Vector subspace_extractor_v(
    0, "u", "\\mathbf{u}");
  const WeakForms::SubSpaceExtractors::Tensor<2> subspace_extractor_T(
    dim, "T", "\\mathbf{T}");

  const FieldSolution<dim, spacedim> solution;
  const auto                         soln_ss_v = solution[subspace_extractor_v];
  const auto                         soln_ss_T = solution[subspace_extractor_T];

  const auto soln_val_v  = soln_ss_v.value();
  const auto soln_grad_v = soln_ss_v.gradient();
  const auto soln_curl_v = soln_ss_v.curl();

  const auto soln_div_T = soln_ss_T.divergence();

  const auto expr_1 = -(-(-(soln_val_v)));
  const auto expr_2 = -(soln_val_v + (-(soln_val_v)));
  const auto expr_3 = -((-soln_val_v) + soln_val_v);

  const auto expr_4 = soln_val_v * soln_curl_v;
  const auto expr_5 = soln_curl_v * soln_val_v;

  const auto expr_6 = cross_product(soln_div_T, soln_curl_v);
  const auto expr_7 = cross_product(soln_curl_v, soln_div_T);

  //   const auto expr_4 = -(soln_div_v*soln_val_v);

  // Test strings
  {
    LogStream::Prefix prefix("string");

    deallog << std::endl;

    deallog << "Expr 1: " << expr_1.as_ascii(decorator) << std::endl;
    deallog << "Expr 2: " << expr_2.as_ascii(decorator) << std::endl;
    deallog << "Expr 3: " << expr_3.as_ascii(decorator) << std::endl;
    deallog << "Expr 4: " << expr_4.as_ascii(decorator) << std::endl;
    deallog << "Expr 5: " << expr_5.as_ascii(decorator) << std::endl;
    deallog << "Expr 6: " << expr_6.as_ascii(decorator) << std::endl;
    deallog << "Expr 7: " << expr_7.as_ascii(decorator) << std::endl;

    deallog << std::endl;
  }

  // Test LaTeX
  {
    LogStream::Prefix prefix("LaTeX");

    deallog << std::endl;

    deallog << "Expr 1: " << expr_1.as_latex(decorator) << std::endl;
    deallog << "Expr 2: " << expr_2.as_latex(decorator) << std::endl;
    deallog << "Expr 3: " << expr_3.as_latex(decorator) << std::endl;
    deallog << "Expr 4: " << expr_4.as_latex(decorator) << std::endl;
    deallog << "Expr 5: " << expr_5.as_latex(decorator) << std::endl;
    deallog << "Expr 6: " << expr_6.as_latex(decorator) << std::endl;
    deallog << "Expr 7: " << expr_7.as_latex(decorator) << std::endl;
  }

  deallog << "OK" << std::endl << std::endl;
}


int
main()
{
  initlog();

  //   run<2>();
  run<3>();

  deallog << "OK" << std::endl;
}
