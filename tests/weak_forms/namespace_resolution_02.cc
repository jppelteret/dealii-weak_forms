// ---------------------------------------------------------------------
//
// Copyright (C) 2022 by Jean-Paul Pelteret
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

// Test that operators work correctly when the namespacing is not
// unconditionally exposed.
// See https://github.com/jppelteret/dealii-weak_forms/issues/63

#include <deal.II/differentiation/ad.h>

#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>

#include <weak_forms/weak_forms.h>


template <int dim>
class Coefficient : public dealii::TensorFunction<4, dim, double>
{
public:
  Coefficient(double lambda = 1.0, double mu = 1.0)
    : lambda(lambda)
    , mu(mu)
  {}

  virtual dealii::Tensor<4, dim, double>
  value(dealii::Point<dim> const & /*p*/) const override
  {
    dealii::Tensor<4, dim, double>        C;
    dealii::SymmetricTensor<2, dim> const I =
      dealii::unit_symmetric_tensor<dim>();

    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        for (unsigned int k = 0; k < dim; ++k)
          for (unsigned int l = 0; l < dim; ++l)
            C[i][j][k][l] = lambda * I[i][j] * I[k][l] +
                            mu * (I[i][k] * I[j][l] + I[i][l] * I[j][k]);

    return C;
  }

private:
  double const lambda;
  double const mu;
};

template <int dim>
class RightHandSide : public dealii::TensorFunction<1, dim, double>
{
public:
  virtual dealii::Tensor<1, dim, double>
  value(dealii::Point<dim> const &p) const override
  {
    dealii::Point<dim> point_1, point_2;
    point_1(0) = 0.5;
    point_2(0) = -0.5;

    dealii::Tensor<1, dim, double> out;

    if (((p - point_1).norm_square() < 0.2 * 0.2) ||
        ((p - point_2).norm_square() < 0.2 * 0.2))
      out[0] = 1.0;
    else
      out[0] = 0.0;

    if (p.norm_square() < 0.2 * 0.2)
      out[1] = 1.0;
    else
      out[1] = 0.0;

    return out;
  }
};


template <int dim>
void
run()
{
  // using namespace dealiiWeakForms;
  // using namespace WeakForms;
  // using namespace Differentiation;
  namespace dealiiWF = dealiiWeakForms::WeakForms;

  int constexpr spacedim = dim;
  auto constexpr ad_typecode =
    dealii::Differentiation::AD::NumberTypes::sacado_dfad_dfad;
  using ADNumber_t =
    typename dealii::Differentiation::AD::NumberTraits<double,
                                                       ad_typecode>::ad_type;

  // Symbolic types for test function, and a coefficient.
  dealiiWF::TestFunction<dim> const          test;
  dealiiWF::FieldSolution<dim> const         solution;
  dealiiWF::SubSpaceExtractors::Vector const subspace_extractor(0,
                                                                "u",
                                                                "\\mathbf{u}");

  dealiiWF::VectorFunctionFunctor<dim> const rhs_coeff("s", "\\mathbf{s}");
  Coefficient<dim> const                     coefficient;
  RightHandSide<dim> const                   rhs;

  auto const test_ss = test[subspace_extractor];
  auto const soln_ss = solution[subspace_extractor];

  auto const test_val  = test_ss.value();
  auto const soln_grad = soln_ss.gradient();

  auto const energy_func = dealiiWF::energy_functor("e", "\\Psi", soln_grad);
  using EnergyADNumber_t =
    typename decltype(energy_func)::template ad_type<double, ad_typecode>;
  static_assert(std::is_same<ADNumber_t, EnergyADNumber_t>::value,
                "Expected identical AD number types");

  auto const energy = energy_func.template value<ADNumber_t, dim, spacedim>(
    [&coefficient](
      dealii::MeshWorker::ScratchData<dim, spacedim> const &scratch_data,
      std::vector<dealiiWF::SolutionExtractionData<dim, spacedim>> const
        &                                            solution_extraction_data,
      unsigned int const                             q_point,
      dealii::Tensor<2, spacedim, ADNumber_t> const &grad_u)
    {
      (void)solution_extraction_data;
      // Sacado is unbelievably annoying. If we don't explicitly
      // cast this return type then we get a segfault.
      // i.e. don't return the result inline!
      dealii::Point<spacedim> const &p =
        scratch_data.get_quadrature_points()[q_point];
      auto const       C      = coefficient.value(p);
      ADNumber_t const energy = 0.5 * dealii::contract3(grad_u, C, grad_u);
      return energy;
    },
    dealii::UpdateFlags::update_quadrature_points);

  dealiiWF::MatrixBasedAssembler<dim> assembler;
  // This line compiles !
  dealiiWF::linear_form(test_val, rhs_coeff.value(rhs));
  // This line does not compile
  assembler += dealiiWF::energy_functional_form(energy).dV() -
               dealiiWF::linear_form(test_val, rhs_coeff.value(rhs)).dV();
}


// Include this down here so that we don't accidentally expose any namespaces
// to the code above.
#include "../weak_forms_tests.h"


int
main()
{
  initlog();

  run<2>();

  deallog << "OK" << std::endl;

  return 0;
}
