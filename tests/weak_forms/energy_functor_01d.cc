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

// Test that the field extractors are configured correctly for the
// (self-linearizing) energy functor
// - Auto-differentiation
// - Symmetric fields

#include <deal.II/differentiation/ad.h>

#include <weak_forms/weak_forms.h>

#include "../weak_forms_tests.h"
#include "wf_common_tests/energy_and_residual_functor_utilities.h"


using namespace dealii;


template <int dim, typename SubSpaceExtractorType>
void
run(const SubSpaceExtractorType &subspace_extractor)
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;
  using namespace Differentiation;

  constexpr int  spacedim = dim;
  constexpr auto ad_typecode =
    Differentiation::AD::NumberTypes::sacado_dfad_dfad;

  // Symbolic types
  const FieldSolution<dim> solution;

  const auto soln_ss  = solution[subspace_extractor];
  const auto soln_val = soln_ss.value();      // Solution value
  const auto soln_div = soln_ss.divergence(); // Solution divergence

  // Parameterise energy in terms of all possible operations with the space
  const auto energy = energy_functor("e", "\\Psi", soln_val, soln_div);

  // Look at what we're going to compute
  const SymbolicDecorations decorator;

  deallog << "Energy (ascii):\n" << energy.as_ascii(decorator) << std::endl;
  deallog << "Energy (LaTeX):\n" << energy.as_latex(decorator) << std::endl;

  using ADNumber_t =
    typename decltype(energy)::template ad_type<double, ad_typecode>;

  const auto energy_functor = energy.template value<ADNumber_t, dim, spacedim>(
    [](const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
       const std::vector<SolutionExtractionData<dim, spacedim>>
         &                               solution_extraction_data,
       const unsigned int                q_point,
       const Tensor<2, dim, ADNumber_t> &u,
       const Tensor<1, dim, ADNumber_t> &div_u) { return ADNumber_t(0.0); });

  deallog << "Energy functor (ascii):\n"
          << energy_functor.as_ascii(decorator) << std::endl;
  deallog << "Energy functor (LaTeX):\n"
          << energy_functor.as_latex(decorator) << std::endl;

  print_ad_functor_field_args_and_extractors(energy_functor, decorator);

  deallog << "OK" << std::endl;
}


int
main(int argc, char **argv)
{
  initlog();

  const WeakForms::SubSpaceExtractors::SymmetricTensor<2> subspace_extractor(
    0, "S", "\\mathbf{S}");

  run<2>(subspace_extractor);
  run<3>(subspace_extractor);

  deallog << "OK" << std::endl;

  return 0;
}
