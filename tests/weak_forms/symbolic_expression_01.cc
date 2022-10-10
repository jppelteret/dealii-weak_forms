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

// Test symbolic expression output
// - Functors
// - Cache functors

#include <deal.II/differentiation/sd.h>

#include <weak_forms/weak_forms.h>

#include "../weak_forms_tests.h"


using namespace dealii;


template <int dim>
void
run()
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;
  using namespace Differentiation;

  constexpr int spacedim = dim;
  using SDNumber_t       = Differentiation::SD::Expression;

  // Named
  {
    const auto s  = constant_scalar<dim>(1.0, "s", "s");
    const auto v  = constant_vector<dim>(Tensor<1, dim>(), "v", "v");
    const auto T2 = constant_tensor<dim>(Tensor<2, dim>(), "T2", "T2");
    const auto T3 = constant_tensor<dim>(Tensor<3, dim>(), "T3", "T3");
    const auto T4 = constant_tensor<dim>(Tensor<4, dim>(), "T4", "T4");
    const auto S2 =
      constant_symmetric_tensor<dim>(unit_symmetric_tensor<dim>(), "S2", "S2");
    const auto S4 =
      constant_symmetric_tensor<dim>(identity_tensor<dim>(), "S4", "S4");

    deallog << "s: " << s.as_expression() << std::endl;
    deallog << "v: " << v.as_expression() << std::endl;
    deallog << "T2: " << T2.as_expression() << std::endl;
    deallog << "T3: " << T3.as_expression() << std::endl;
    deallog << "T4: " << T4.as_expression() << std::endl;
    deallog << "S2: " << S2.as_expression() << std::endl;
    deallog << "S4: " << S4.as_expression() << std::endl;
  }

  // Unnamed
  {
    const auto s  = constant_scalar<dim>(1.0);
    const auto v  = constant_vector<dim>(Tensor<1, dim>());
    const auto T2 = constant_tensor<dim>(Tensor<2, dim>());
    const auto T3 = constant_tensor<dim>(Tensor<3, dim>());
    const auto T4 = constant_tensor<dim>(Tensor<4, dim>());
    const auto S2 =
      constant_symmetric_tensor<dim>(unit_symmetric_tensor<dim>());
    const auto S4 = constant_symmetric_tensor<dim>(identity_tensor<dim>());

    deallog << "s: " << s.as_expression() << std::endl;
    deallog << "v: " << v.as_expression() << std::endl;
    deallog << "T2: " << T2.as_expression() << std::endl;
    deallog << "T3: " << T3.as_expression() << std::endl;
    deallog << "T4: " << T4.as_expression() << std::endl;
    deallog << "S2: " << S2.as_expression() << std::endl;
    deallog << "S4: " << S4.as_expression() << std::endl;
  }

  // Cache functors
  {
    const ScalarCacheFunctor                  s("s", "s");
    const TensorCacheFunctor<2, dim>          T("T", "T");
    const SymmetricTensorCacheFunctor<2, dim> S("S", "S");
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


    deallog << "sc: " << sc.as_expression() << std::endl;
    deallog << "Tc: " << Tc.as_expression() << std::endl;
    deallog << "Sc: " << Sc.as_expression() << std::endl;
  }

  deallog << "OK" << std::endl;
}


int
main(int argc, char **argv)
{
  initlog();

  run<2>();
  run<3>();

  deallog << "OK" << std::endl;

  return 0;
}
