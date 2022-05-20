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

#ifndef dealii_weakforms_mixed_form_operators_h
#define dealii_weakforms_mixed_form_operators_h

#include <deal.II/base/config.h>

#include <deal.II/base/template_constraints.h>

#include <weak_forms/bilinear_forms.h>
#include <weak_forms/binary_operators.h>
#include <weak_forms/config.h>
#include <weak_forms/linear_forms.h>
#include <weak_forms/mixed_operators.h>
#include <weak_forms/template_constraints.h>



WEAK_FORMS_NAMESPACE_OPEN


// Arithmetic operations
// These should be in the same namespace as the SymbolicOps, so that ADL
// can be exploited when namespace is not unconditionally exposed.
namespace WeakForms
{
  namespace Operators
  {
    /* ============= Specialization of operators: Linear forms ============= */



    template <typename ScalarType,
              typename TestSpaceOp,
              typename Functor,
              typename = typename WeakForms::is_scalar_type<ScalarType>::type>
    auto
    operator*(const ScalarType &                                 value,
              const WeakForms::LinearForm<TestSpaceOp, Functor> &linear_form)
    {
      return WeakForms::linear_form(linear_form.get_test_space_operation(),
                                    value * linear_form.get_functor());
    }



    template <typename ScalarType,
              typename TestSpaceOp,
              typename Functor,
              typename = typename WeakForms::is_scalar_type<ScalarType>::type>
    auto
    operator*(const WeakForms::LinearForm<TestSpaceOp, Functor> &linear_form,
              const ScalarType &                                 value)
    {
      // Delegate to the other function
      return value * linear_form;
    }



    /* ============ Specialization of operators: Bilinear forms ============ */



    template <typename ScalarType,
              typename TestSpaceOp,
              typename Functor,
              typename TrialSpaceOp,
              typename = typename WeakForms::is_scalar_type<ScalarType>::type>
    auto
    operator*(const ScalarType &value,
              const WeakForms::BilinearForm<TestSpaceOp, Functor, TrialSpaceOp>
                &bilinear_form)
    {
      return WeakForms::bilinear_form(
        bilinear_form.get_test_space_operation(),
        value * bilinear_form.get_functor(),
        bilinear_form.get_trial_space_operation());
    }



    template <typename ScalarType,
              typename TestSpaceOp,
              typename Functor,
              typename TrialSpaceOp,
              typename = typename WeakForms::is_scalar_type<ScalarType>::type>
    auto
    operator*(const WeakForms::BilinearForm<TestSpaceOp, Functor, TrialSpaceOp>
                &               bilinear_form,
              const ScalarType &value)
    {
      // Delegate to the other function
      return value * bilinear_form;
    }


  } // namespace Operators
} // namespace WeakForms


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_mixed_form_operators_h