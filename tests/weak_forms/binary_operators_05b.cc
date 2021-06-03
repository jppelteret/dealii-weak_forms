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


// Check scalar operator* for symbolic ops
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

  const double scalar = 2.0;
  // const std::complex<double> complex_scalar = std::complex<double>(2.0);

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
    const auto a1 = scalar * soln.value();
    const auto a2 = soln.value() * scalar;

    const auto s1 = scalar * soln[subspace_extractor_s].value();
    const auto s2 = soln[subspace_extractor_s].value() * scalar;

    const auto v1 = scalar * soln[subspace_extractor_v].value();
    const auto v2 = soln[subspace_extractor_v].value() * scalar;

    const auto T1 = scalar * soln[subspace_extractor_T].value();
    const auto T2 = soln[subspace_extractor_T].value() * scalar;

    const auto S1 = scalar * soln[subspace_extractor_S].value();
    const auto S2 = soln[subspace_extractor_S].value() * scalar;
  }

  // Test function
  {
    const auto a1 = scalar * test.value();
    const auto a2 = test.value() * scalar;

    const auto s1 = scalar * test[subspace_extractor_s].value();
    const auto s2 = test[subspace_extractor_s].value() * scalar;

    const auto v1 = scalar * test[subspace_extractor_v].value();
    const auto v2 = test[subspace_extractor_v].value() * scalar;

    const auto T1 = scalar * test[subspace_extractor_T].value();
    const auto T2 = test[subspace_extractor_T].value() * scalar;

    const auto S1 = scalar * test[subspace_extractor_S].value();
    const auto S2 = test[subspace_extractor_S].value() * scalar;
  }

  // Trial function
  {
    const auto a1 = scalar * trial.value();
    const auto a2 = trial.value() * scalar;

    const auto s1 = scalar * trial[subspace_extractor_s].value();
    const auto s2 = trial[subspace_extractor_s].value() * scalar;

    const auto v1 = scalar * trial[subspace_extractor_v].value();
    const auto v2 = trial[subspace_extractor_v].value() * scalar;

    const auto T1 = scalar * trial[subspace_extractor_T].value();
    const auto T2 = trial[subspace_extractor_T].value() * scalar;

    const auto S1 = scalar * trial[subspace_extractor_S].value();
    const auto S2 = trial[subspace_extractor_S].value() * scalar;
  }

  deallog << "OK" << std::endl;
}
