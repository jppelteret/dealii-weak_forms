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


// Check functor form stringization and printing
// - Auto-differentiable energy functor
// - Sub-Space: Vector

#include <deal.II/base/function_lib.h>
#include <deal.II/base/tensor_function.h>

#include <weak_forms/energy_functor.h>
#include <weak_forms/functors.h>
#include <weak_forms/spaces.h>
#include <weak_forms/subspace_extractors.h>
#include <weak_forms/subspace_views.h>
#include <weak_forms/symbolic_decorations.h>
#include <weak_forms/symbolic_operators.h>

#include "../weak_forms_tests.h"


template <int dim,
          int spacedim        = dim,
          typename NumberType = double,
          typename SubSpaceExtractorType>
void
run(const SubSpaceExtractorType &subspace_extractor)
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;
  using namespace Differentiation;

  // Customise the naming conventions, if we wish to.
  const SymbolicDecorations decorator;

  const FieldSolution<dim> solution;

  const auto soln_ss   = solution[subspace_extractor];
  const auto soln_val  = soln_ss.value();    // Solution value
  const auto soln_grad = soln_ss.gradient(); // Solution gradient
  const auto soln_symm_grad =
    soln_ss.symmetric_gradient();              // Solution symmetric gradient
  const auto soln_div  = soln_ss.divergence(); // Solution divergence
  const auto soln_curl = soln_ss.curl();       // Solution curl
  const auto soln_hess = soln_ss.hessian();    // Solution hessian
  const auto soln_d3 = soln_ss.third_derivative(); // Solution third derivative

  const auto energy_1 = energy_functor("e", "\\Psi", soln_val);
  const auto energy_2 = energy_functor("e", "\\Psi", soln_val, soln_grad);
  const auto energy_3 =
    energy_functor("e", "\\Psi", soln_val, soln_grad, soln_symm_grad);
  const auto energy_4 =
    energy_functor("e", "\\Psi", soln_val, soln_grad, soln_symm_grad, soln_div);
  const auto energy_5 = energy_functor(
    "e", "\\Psi", soln_val, soln_grad, soln_symm_grad, soln_div, soln_curl);
  const auto energy_6 = energy_functor("e",
                                       "\\Psi",
                                       soln_val,
                                       soln_grad,
                                       soln_symm_grad,
                                       soln_div,
                                       soln_curl,
                                       soln_hess);
  const auto energy_7 = energy_functor("e",
                                       "\\Psi",
                                       soln_val,
                                       soln_grad,
                                       soln_symm_grad,
                                       soln_div,
                                       soln_curl,
                                       soln_hess,
                                       soln_d3);

  // Test strings
  {
    LogStream::Prefix prefix("string");

    deallog << "(v): " << energy_1.as_ascii(decorator) << std::endl;
    deallog << "(v,g): " << energy_2.as_ascii(decorator) << std::endl;
    deallog << "(v,g,sg): " << energy_3.as_ascii(decorator) << std::endl;
    deallog << "(v,g,sg,d): " << energy_4.as_ascii(decorator) << std::endl;
    deallog << "(v,g,sg,d,c): " << energy_5.as_ascii(decorator) << std::endl;
    deallog << "(v,g,sg,d,c,h): " << energy_6.as_ascii(decorator) << std::endl;
    deallog << "(v,g,sg,d,c,h,d3): " << energy_7.as_ascii(decorator)
            << std::endl;

    deallog << std::endl;
  }

  // Test LaTeX
  {
    LogStream::Prefix prefix("LaTeX");

    deallog << "(v): " << energy_1.as_latex(decorator) << std::endl;
    deallog << "(v,g): " << energy_2.as_latex(decorator) << std::endl;
    deallog << "(v,g,sg): " << energy_3.as_latex(decorator) << std::endl;
    deallog << "(v,g,sg,d): " << energy_4.as_latex(decorator) << std::endl;
    deallog << "(v,g,sg,d,c): " << energy_5.as_latex(decorator) << std::endl;
    deallog << "(v,g,sg,d,c,h): " << energy_6.as_latex(decorator) << std::endl;
    deallog << "(v,g,sg,d,c,h,d3): " << energy_7.as_latex(decorator)
            << std::endl;

    deallog << std::endl;
  }

  deallog << "OK" << std::endl << std::endl;
}


int
main()
{
  initlog();

  const WeakForms::SubSpaceExtractors::Vector subspace_extractor(0,
                                                                 "u",
                                                                 "\\mathbf{u}");

  run<3>(subspace_extractor);

  deallog << "OK" << std::endl;
}
