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

// Test that the view extractors are configured correctly for the
// (self-linearizing) residual functor
// - Symbolic differentiation
// - Scalar, vector, tensor, symmetric tensor view
// - Linearize with respect to tensor fields

#include <deal.II/differentiation/ad.h>

#include <weak_forms/weak_forms.h>

#include "../weak_forms_tests.h"
#include "wf_common_tests/energy_and_residual_functor_utilities.h"


using namespace dealii;


template <int dim,
          typename SubSpaceExtractorTestSpace,
          typename SubSpaceExtractorDerivativesType>
void
run(const SubSpaceExtractorTestSpace &      subspace_extractor_test,
    const SubSpaceExtractorDerivativesType &subspace_extractor_lin)
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;
  using namespace Differentiation;

  constexpr int spacedim = dim;
  using SDNumber_t       = typename Differentiation::SD::Expression;

  // Symbolic types
  const TestFunction<dim>  test;
  const FieldSolution<dim> solution;

  const auto test_ss  = test[subspace_extractor_test];
  const auto test_val = test_ss.value();

  const auto soln_ss   = solution[subspace_extractor_lin];
  const auto soln_val  = soln_ss.value();      // Solution value
  const auto soln_grad = soln_ss.gradient();   // Solution gradient
  const auto soln_div  = soln_ss.divergence(); // Solution divergence

  // Parameterise residual in terms of all possible operations with the space
  const auto residual =
    residual_functor("R", "R", soln_val, soln_grad, soln_div);

  // Look at what we're going to compute
  const SymbolicDecorations decorator;

  deallog << "Residual (ascii):\n" << residual.as_ascii(decorator) << std::endl;
  deallog << "Residual (LaTeX):\n" << residual.as_latex(decorator) << std::endl;

  // Now get the view for the residual componet that we're going to linearize.
  const auto residual_view = residual[test_val];

  deallog << "Residual view (ascii):\n"
          << residual_view.as_ascii(decorator) << std::endl;
  deallog << "Residual view (LaTeX):\n"
          << residual_view.as_latex(decorator) << std::endl;

  using Result_t =
    typename decltype(residual_view)::template value_type<SDNumber_t>;
  //   using Result_t = typename SubSpaceExtractorTestSpace::template
  //   value_type<spacedim,SDNumber_t>;

  const auto residual_functor =
    residual_view.template value<SDNumber_t, dim, spacedim>(
      [](const Tensor<2, dim, SDNumber_t> &u,
         const Tensor<3, dim, SDNumber_t> &grad_u,
         const Tensor<1, dim, SDNumber_t> &div_u) { return Result_t{}; },
      [](const Tensor<2, dim, SDNumber_t> &u,
         const Tensor<3, dim, SDNumber_t> &grad_u,
         const Tensor<1, dim, SDNumber_t> &div_u)
      { return Differentiation::SD::types::substitution_map{}; },
      [](const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
         const std::vector<SolutionExtractionData<dim, spacedim>>
           &                solution_extraction_data,
         const unsigned int q_point)
      { return Differentiation::SD::types::substitution_map{}; },
      Differentiation::SD::OptimizerType::dictionary,
      Differentiation::SD::OptimizationFlags::optimize_default,
      UpdateFlags::update_default);

  deallog << "Residual functor (ascii):\n"
          << residual_functor.as_ascii(decorator) << std::endl;
  deallog << "Residual functor (LaTeX):\n"
          << residual_functor.as_latex(decorator) << std::endl;

  print_sd_functor_print_field_args_and_symbolic_fields(residual_functor,
                                                        decorator);

  deallog << "OK" << std::endl;
}


int
main(int argc, char **argv)
{
  initlog();

  const WeakForms::SubSpaceExtractors::Scalar subspace_extractor_s(0, "s", "s");
  const WeakForms::SubSpaceExtractors::Vector subspace_extractor_v(
    0, "v", "\\mathbf{v}");
  const WeakForms::SubSpaceExtractors::Tensor<2> subspace_extractor_T(
    0, "T", "\\mathbf{T}");
  const WeakForms::SubSpaceExtractors::SymmetricTensor<2> subspace_extractor_S(
    0, "S", "\\mathbf{S}");

  {
    LogStream::Prefix prefix("Scalar");
    run<2>(subspace_extractor_s, subspace_extractor_T);
    run<3>(subspace_extractor_s, subspace_extractor_T);
    deallog << "OK" << std::endl;
  }

  {
    LogStream::Prefix prefix("Vector");
    run<2>(subspace_extractor_v, subspace_extractor_T);
    run<3>(subspace_extractor_v, subspace_extractor_T);
    deallog << "OK" << std::endl;
  }

  {
    LogStream::Prefix prefix("Tensor");
    run<2>(subspace_extractor_T, subspace_extractor_T);
    run<3>(subspace_extractor_T, subspace_extractor_T);
    deallog << "OK" << std::endl;
  }

  {
    LogStream::Prefix prefix("SymmetricTensor");
    run<2>(subspace_extractor_S, subspace_extractor_T);
    run<3>(subspace_extractor_S, subspace_extractor_T);
    deallog << "OK" << std::endl;
  }

  deallog << "OK" << std::endl;

  return 0;
}
