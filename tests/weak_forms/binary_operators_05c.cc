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


// Check tensor operator* for symbolic ops
// - Spaces


#include <weak_forms/binary_operators.h>
#include <weak_forms/mixed_operators.h>
#include <weak_forms/spaces.h>
#include <weak_forms/subspace_extractors.h>
#include <weak_forms/subspace_views.h>
#include <weak_forms/type_traits.h>

#include <complex>

#include "../weak_forms_tests.h"


int
main()
{
  initlog();

  using namespace WeakForms;

  constexpr int dim      = 2;
  constexpr int spacedim = 2;

  const Tensor<2, dim> tensor(unit_symmetric_tensor<dim>());
  // const Tensor<2,dim,std::complex<double>> complex_tensor
  // (unit_symmetric_tensor<dim>());

  const TestFunction<dim, spacedim>  test;
  const TrialSolution<dim, spacedim> trial;
  const FieldSolution<dim, spacedim> soln;

  const SubSpaceExtractors::Scalar subspace_extractor_s(0, "s", "s");
  const SubSpaceExtractors::Vector subspace_extractor_v(0, "u", "\\mathbf{u}");
  constexpr int                    rank = 2;
  const SubSpaceExtractors::Tensor<rank>          subspace_extractor_T(0,
                                                              "T",
                                                              "\\mathbf{T}");
  const SubSpaceExtractors::SymmetricTensor<rank> subspace_extractor_S(
    0, "S", "\\mathbf{S}");

  // Field solution
  {
    const auto a1 = tensor * soln.gradient();
    const auto a2 = soln.gradient() * tensor;

    const auto s1 = tensor * soln[subspace_extractor_s].gradient();
    const auto s2 = soln[subspace_extractor_s].gradient() * tensor;

    const auto v1 = tensor * soln[subspace_extractor_v].value();
    const auto v2 = soln[subspace_extractor_v].value() * tensor;

    const auto T1 = tensor * soln[subspace_extractor_T].value();
    const auto T2 = soln[subspace_extractor_T].value() * tensor;

    const auto S1 = tensor * soln[subspace_extractor_S].value();
    const auto S2 = soln[subspace_extractor_S].value() * tensor;
  }

  // Test function
  {
    const auto a1 = tensor * test.gradient();
    const auto a2 = test.gradient() * tensor;

    const auto s1 = tensor * test[subspace_extractor_s].gradient();
    const auto s2 = test[subspace_extractor_s].gradient() * tensor;

    const auto v1 = tensor * test[subspace_extractor_v].value();
    const auto v2 = test[subspace_extractor_v].value() * tensor;

    const auto T1 = tensor * test[subspace_extractor_T].value();
    const auto T2 = test[subspace_extractor_T].value() * tensor;

    const auto S1 = tensor * test[subspace_extractor_S].value();
    const auto S2 = test[subspace_extractor_S].value() * tensor;
  }

  // Trial function
  {
    const auto a1 = tensor * trial.gradient();
    const auto a2 = trial.gradient() * tensor;

    const auto s1 = tensor * trial[subspace_extractor_s].gradient();
    const auto s2 = trial[subspace_extractor_s].gradient() * tensor;

    const auto v1 = tensor * trial[subspace_extractor_v].value();
    const auto v2 = trial[subspace_extractor_v].value() * tensor;

    const auto T1 = tensor * trial[subspace_extractor_T].value();
    const auto T2 = trial[subspace_extractor_T].value() * tensor;

    const auto S1 = tensor * trial[subspace_extractor_S].value();
    const auto S2 = trial[subspace_extractor_S].value() * tensor;
  }

  deallog << "OK" << std::endl;
}
