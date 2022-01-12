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


// Check [scalar, tensor, symmetric tensor] operator* for symbolic ops
// - Functors


#include <weak_forms/binary_operators.h>
#include <weak_forms/cache_functors.h>
#include <weak_forms/functors.h>
#include <weak_forms/mixed_operators.h>
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


  // === Pre/post multiply by scalar ; post divide by scalar ===

  // Functors
  {
    const auto s = constant_scalar<dim>(2.0);
    const auto T = constant_tensor<2, spacedim>(Tensor<2, dim>{});
    const auto S =
      constant_symmetric_tensor<2, spacedim>(SymmetricTensor<2, dim>{});

    // Multiply
    const auto s1 = scalar * s;
    const auto s2 = s * scalar;

    const auto T1 = scalar * T;
    const auto T2 = T * scalar;

    const auto S1 = scalar * S;
    const auto S2 = S * scalar;

    // Divide
    const auto s3 = s / constant_scalar<dim>(4.0);
    const auto s4 = scalar / s;
    const auto s5 = s / scalar;
  }

  // Function functors
  {
    const ScalarFunctionFunctor<spacedim> s("s", "s");
    const Functions::ConstantFunction<spacedim, double>
               constant_scalar_function(1.0);
    const auto sf = s.template value<double, dim>(constant_scalar_function);

    // const TensorFunctionFunctor<0, spacedim>     t("t", "t");
    const TensorFunctionFunctor<2, spacedim> T("T", "T");
    // const ConstantTensorFunction<0, dim, double>
    // constant_r0_tensor_function(2.0);
    const ConstantTensorFunction<2, dim, double> constant_tensor_function(
      unit_symmetric_tensor<dim>());
    // const auto tf = value<double, dim>(t, constant_r0_tensor_function);
    const auto Tf = T.template value<double, dim>(constant_tensor_function);

    // Multiply
    const auto s1 = scalar * sf;
    const auto s2 = sf * scalar;

    const auto T1 = scalar * Tf;
    const auto T2 = Tf * scalar;

    // Divide
    // const auto t1 = scalar / tf;
    // const auto t2 = tf / scalar;

    const auto T3 = Tf / scalar;
    const auto T4 = Tf / sf;
    // const auto T5 = Tf / tf;
  }

  // Cache functors
  {
    const ScalarCacheFunctor                  s("s", "s");
    const TensorCacheFunctor<2, dim>          T("T", "T");
    const SymmetricTensorCacheFunctor<2, dim> S("S", "T");
    const UpdateFlags update_flags = UpdateFlags::update_default;

    const auto s_func =
      [](MeshWorker::ScratchData<dim, spacedim> &scratch_data,
         const std::vector<SolutionExtractionData<dim, spacedim>>
           &                solution_extraction_data,
         const unsigned int q_point) { return 0.0; };
    const auto sc =
      s.template value<double, dim, spacedim>(s_func, update_flags);

    const auto T_func =
      [](MeshWorker::ScratchData<dim, spacedim> &scratch_data,
         const std::vector<SolutionExtractionData<dim, spacedim>>
           &                solution_extraction_data,
         const unsigned int q_point) { return Tensor<2, dim>(); };
    const auto Tc = T.template value<double, dim>(T_func, update_flags);

    const auto S_func =
      [](MeshWorker::ScratchData<dim, spacedim> &scratch_data,
         const std::vector<SolutionExtractionData<dim, spacedim>>
           &                solution_extraction_data,
         const unsigned int q_point) { return SymmetricTensor<2, dim>(); };
    const auto Sc = S.template value<double, dim>(S_func, update_flags);

    // Multiply
    const auto s1 = scalar * sc;
    const auto s2 = sc * scalar;

    const auto T1 = scalar * Tc;
    const auto T2 = Tc * scalar;

    const auto S1 = scalar * Sc;
    const auto S2 = Sc * scalar;

    // Divide
    const auto s3 = scalar / sc;
    const auto s4 = sc / scalar;

    const auto T3 = Tc / scalar;
    const auto S3 = Sc / scalar;
  }

  deallog << "OK" << std::endl;
}
