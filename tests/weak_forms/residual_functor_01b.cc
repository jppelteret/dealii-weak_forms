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
// - Auto-differentiation
// - Scalar, vector, tensor, symmetric tensor view
// - Linearize with respect to vector fields

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

  constexpr int  spacedim    = dim;
  constexpr auto ad_typecode = Differentiation::AD::NumberTypes::sacado_dfad;

  // Symbolic types
  const TestFunction<dim>  test;
  const FieldSolution<dim> solution;

  const auto test_ss  = test[subspace_extractor_test];
  const auto test_val = test_ss.value();

  const auto soln_ss   = solution[subspace_extractor_lin];
  const auto soln_val  = soln_ss.value();    // Solution value
  const auto soln_grad = soln_ss.gradient(); // Solution gradient
  const auto soln_sym_grad =
    soln_ss.symmetric_gradient();              // Solution symmetric gradient
  const auto soln_div  = soln_ss.divergence(); // Solution divergence
  const auto soln_curl = soln_ss.curl();       // Solution curl
  const auto soln_hess = soln_ss.hessian();    // Solution hessian
  const auto soln_d3 = soln_ss.third_derivative(); // Solution third derivative

  // Parameterise residual in terms of all possible operations with the space
  const auto residual = residual_functor("R",
                                         "R",
                                         soln_val,
                                         soln_grad,
                                         soln_sym_grad,
                                         soln_div,
                                         soln_curl,
                                         soln_hess,
                                         soln_d3);

  // Look at what we're going to compute
  const SymbolicDecorations decorator;

  deallog << "Residual (ascii):\n" << residual.as_ascii(decorator) << std::endl;
  deallog << "Residual (LaTeX):\n" << residual.as_latex(decorator) << std::endl;

  // Now get the view for the residual component that we're going to linearize.
  const auto residual_view = residual[test_val];

  deallog << "Residual view (ascii):\n"
          << residual_view.as_ascii(decorator) << std::endl;
  deallog << "Residual view (LaTeX):\n"
          << residual_view.as_latex(decorator) << std::endl;

  using ADNumber_t =
    typename decltype(residual_view)::template ad_type<double, ad_typecode>;
  using Result_t =
    typename decltype(residual_view)::template value_type<ADNumber_t>;

  const auto residual_functor =
    residual_view.template value<ADNumber_t, dim, spacedim>(
      [](const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
         const std::vector<SolutionExtractionData<dim, spacedim>>
           &                                        solution_extraction_data,
         const unsigned int                         q_point,
         const Tensor<1, dim, ADNumber_t> &         u,
         const Tensor<2, dim, ADNumber_t> &         grad_u,
         const SymmetricTensor<2, dim, ADNumber_t> &symm_grad_u,
         const ADNumber_t &                         div_u,
         const Tensor<1, dim, ADNumber_t> &         curl_u,
         const Tensor<3, dim, ADNumber_t> &         hess_u,
         const Tensor<4, dim, ADNumber_t> &d3_u) { return Result_t{}; });

  deallog << "Residual functor (ascii):\n"
          << residual_functor.as_ascii(decorator) << std::endl;
  deallog << "Residual functor (LaTeX):\n"
          << residual_functor.as_latex(decorator) << std::endl;

  print_ad_functor_field_args_and_extractors(residual_functor, decorator);

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
    // run<2>(subspace_extractor_s, subspace_extractor_v); // Curl not available
    // in 2d
    run<3>(subspace_extractor_s, subspace_extractor_v);
    deallog << "OK" << std::endl;
  }

  {
    LogStream::Prefix prefix("Vector");
    // run<2>(subspace_extractor_v, subspace_extractor_v);
    run<3>(subspace_extractor_v, subspace_extractor_v);
    deallog << "OK" << std::endl;
  }

  {
    LogStream::Prefix prefix("Tensor");
    // run<2>(subspace_extractor_T, subspace_extractor_v);
    run<3>(subspace_extractor_T, subspace_extractor_v);
    deallog << "OK" << std::endl;
  }

  {
    LogStream::Prefix prefix("SymmetricTensor");
    // run<2>(subspace_extractor_S, subspace_extractor_v);
    run<3>(subspace_extractor_S, subspace_extractor_v);
    deallog << "OK" << std::endl;
  }

  deallog << "OK" << std::endl;

  return 0;
}
