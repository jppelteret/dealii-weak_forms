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

// Test that the field extractors are configured correctly for the
// (self-linearizing) energy functor
// - Symbolic differentiation
// - Tensor fields

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

  constexpr int spacedim = dim;
  using SDNumber_t       = typename Differentiation::SD::Expression;

  // Symbolic types
  const FieldSolution<dim> solution;

  const auto soln_ss   = solution[subspace_extractor];
  const auto soln_val  = soln_ss.value();      // Solution value
  const auto soln_grad = soln_ss.gradient();   // Solution gradient
  const auto soln_div  = soln_ss.divergence(); // Solution divergence

  // Parameterise energy in terms of all possible operations with the space
  const auto energy =
    energy_functor("e", "\\Psi", soln_val, soln_grad, soln_div);

  // Look at what we're going to compute
  const SymbolicDecorations decorator;

  deallog << "Energy (ascii):\n" << energy.as_ascii(decorator) << std::endl;
  deallog << "Energy (LaTeX):\n" << energy.as_latex(decorator) << std::endl;

  const auto energy_functor = energy.template value<SDNumber_t, dim, spacedim>(
    [](const Tensor<2, dim, SDNumber_t> &u,
       const Tensor<3, dim, SDNumber_t> &grad_u,
       const Tensor<1, dim, SDNumber_t> &div_u) { return SDNumber_t(0.0); },
    [](const Tensor<2, dim, SDNumber_t> &u,
       const Tensor<3, dim, SDNumber_t> &grad_u,
       const Tensor<1, dim, SDNumber_t> &div_u) {
      return Differentiation::SD::types::substitution_map{};
    },
    [](const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
       const std::vector<std::string> &              solution_names,
       const unsigned int                            q_point) {
      return Differentiation::SD::types::substitution_map{};
    },
    Differentiation::SD::OptimizerType::dictionary,
    Differentiation::SD::OptimizationFlags::optimize_default,
    UpdateFlags::update_default);

  deallog << "Energy functor (ascii):\n"
          << energy_functor.as_ascii(decorator) << std::endl;
  deallog << "Energy functor (LaTeX):\n"
          << energy_functor.as_latex(decorator) << std::endl;

  print_sd_functor_print_field_args_and_symbolic_fields(energy_functor,
                                                        decorator);

  deallog << "OK" << std::endl;
}


int
main(int argc, char **argv)
{
  initlog();

  const WeakForms::SubSpaceExtractors::Tensor<2> subspace_extractor(
    0, "T", "\\mathbf{T}");

  run<2>(subspace_extractor);
  run<3>(subspace_extractor);

  deallog << "OK" << std::endl;

  return 0;
}
