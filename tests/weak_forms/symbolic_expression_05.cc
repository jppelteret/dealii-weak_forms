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
// - Energy functional form

#include <deal.II/differentiation/sd.h>

#include <weak_forms/weak_forms.h>

#include "../weak_forms_tests.h"


template <int dim, int spacedim = dim, typename CompositeSymbolicOp>
void
test(const CompositeSymbolicOp &functor_op)
{
  using namespace dealiiWeakForms::WeakForms;
  using namespace dealiiWeakForms::WeakForms::internal;
  using namespace dealiiWeakForms::WeakForms::Operators::internal;
  CompositeOpHelper<CompositeSymbolicOp>::print(functor_op);

  // A tuple of all field solution operations that appear in the symbolic
  // expression tree. Some of these are duplicated: that's fine, as they'll
  // be a no-op when producing the symbolic substitution maps.
  const auto subspace_field_solution_ops =
    CompositeOpHelper<CompositeSymbolicOp>::get_subspace_field_solution_ops(
      functor_op);

  // Create the energy functor, separately from the energy functional form.
  // We do this because we need to explicitly know the template parameters that
  // are passed to both the energy functor and the energy functional form
  // creation methods.
  const auto energy_func =
    create_energy_functor_from_tuple("e", "\\Psi", subspace_field_solution_ops);
  deallog << "energy_func: " << energy_func.as_ascii(SymbolicDecorations())
          << std::endl;

  // Finally create the energy functional form. The original functor is needed
  // in order to extract the substitution map for the variables other than the
  // field solution components.
  const auto energy_form =
    create_energy_functional_form_from_energy<dim, spacedim>(energy_func,
                                                             functor_op);
  deallog << "energy_form: " << energy_form.as_ascii(SymbolicDecorations())
          << std::endl;
}


template <int dim>
void
run()
{
  deallog << "Dim: " << dim << std::endl;

  using namespace dealii;
  using namespace WeakForms;
  using namespace Differentiation;

  constexpr int spacedim = dim;
  using SDNumber_t       = Differentiation::SD::Expression;

  const auto s1 = constant_scalar<dim>(3.0, "s", "s");
  const auto s2 = constant_scalar<dim>(2.0);

  const FieldSolution<dim, spacedim> field_solution;
  const SubSpaceExtractors::Scalar   subspace_extractor(0, "s", "s");

  const auto s_val  = field_solution[subspace_extractor].value();
  const auto s_grad = field_solution[subspace_extractor].gradient();

  const auto functor_op = s1 * s2 + s1 * s_val * s_val + s2 * s_grad * s_grad;
  deallog << "functor_op: " << functor_op.as_expression() << std::endl;

  test<dim, spacedim>(functor_op);

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
