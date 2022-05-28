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

#include <weak_forms/weak_forms.h>


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

  // Symbolic types for test function, and a coefficient.
  dealiiWF::TestFunction<dim> const          test;
  dealiiWF::SubSpaceExtractors::Vector const subspace_extractor(0,
                                                                "u",
                                                                "\\mathbf{u}");

  dealiiWF::VectorFunctionFunctor<dim> const rhs_coeff("s", "\\mathbf{s}");
  RightHandSide<dim> const                   rhs;

  auto const test_val = test[subspace_extractor].value();

  dealiiWF::MatrixBasedAssembler<dim> assembler;
  assembler += dealiiWF::linear_form(test_val, rhs_coeff.value(rhs)).dV() -
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
