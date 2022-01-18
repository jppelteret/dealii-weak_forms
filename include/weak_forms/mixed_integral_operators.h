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

#ifndef dealii_weakforms_mixed_integral_operators_h
#define dealii_weakforms_mixed_integral_operators_h

#include <deal.II/base/config.h>

#include <deal.II/base/template_constraints.h>

#include <weak_forms/binary_integral_operators.h>
#include <weak_forms/config.h>
#include <weak_forms/mixed_form_operators.h>
#include <weak_forms/symbolic_integral.h>
#include <weak_forms/symbolic_operators.h>
#include <weak_forms/type_traits.h>
#include <weak_forms/unary_integral_operators.h>

#include <type_traits>


WEAK_FORMS_NAMESPACE_OPEN



/* ============ Specialization of operators: Integral operators ============ */

// Here we have to be sure that we don't use the same template signatures
// as some of the other operator* overloads in mixed_operators.h . So instead
// of using templates directly on the operator, we delegate the work to an
// internal function that we can "more easily" query for its return type
// at compile time.


// Forward declarations
namespace WeakForms
{
  namespace internal
  {
    template <
      typename ScalarType,
      typename SymbolicIntegralOp,
      typename = typename std::enable_if<
        WeakForms::is_scalar_type<ScalarType>::value &&
        WeakForms::is_symbolic_integral_op<SymbolicIntegralOp>::value>::type>
    auto
    ScalarTimesSymbolicIntegralOpImpl(const ScalarType &        value,
                                      const SymbolicIntegralOp &integral);


    template <typename ScalarType,
              typename UnaryIntegralOp,
              typename = typename std::enable_if<
                WeakForms::is_scalar_type<ScalarType>::value &&
                WeakForms::is_unary_integral_op<UnaryIntegralOp>::value>::type>
    auto
    ScalarTimesUnaryIntegralOpImpl(const ScalarType &     value,
                                   const UnaryIntegralOp &integral);


    template <
      typename ScalarType,
      typename BinaryIntegralOp,
      typename = typename std::enable_if<
        WeakForms::is_scalar_type<ScalarType>::value &&
        WeakForms::is_binary_integral_op<BinaryIntegralOp>::value>::type>
    auto
    ScalarTimesBinaryIntegralOpImpl(const ScalarType &      value,
                                    const BinaryIntegralOp &integral);
  } // namespace internal
} // namespace WeakForms



template <typename ScalarType, typename IntegralOp>
typename std::enable_if<
  WeakForms::is_scalar_type<ScalarType>::value &&
    WeakForms::is_symbolic_integral_op<IntegralOp>::value,
  decltype(WeakForms::internal::ScalarTimesSymbolicIntegralOpImpl(
    std::declval<ScalarType>(),
    std::declval<IntegralOp>()))>::type
operator*(const ScalarType &value, const IntegralOp &integral)
{
  return WeakForms::internal::ScalarTimesSymbolicIntegralOpImpl(value,
                                                                integral);
}



template <typename ScalarType, typename IntegralOp>
typename std::enable_if<
  WeakForms::is_scalar_type<ScalarType>::value &&
    WeakForms::is_symbolic_integral_op<IntegralOp>::value,
  decltype(WeakForms::internal::ScalarTimesSymbolicIntegralOpImpl(
    std::declval<ScalarType>(),
    std::declval<IntegralOp>()))>::type
operator*(const IntegralOp &integral, const ScalarType &value)
{
  // Delegate to the other function
  return value * integral;
}



template <typename ScalarType, typename IntegralOp>
typename std::enable_if<
  WeakForms::is_scalar_type<ScalarType>::value &&
    WeakForms::is_unary_integral_op<IntegralOp>::value,
  decltype(WeakForms::internal::ScalarTimesUnaryIntegralOpImpl(
    std::declval<ScalarType>(),
    std::declval<IntegralOp>()))>::type
operator*(const ScalarType &value, const IntegralOp &integral)
{
  return WeakForms::internal::ScalarTimesUnaryIntegralOpImpl(value, integral);
}



template <typename ScalarType, typename IntegralOp>
typename std::enable_if<
  WeakForms::is_scalar_type<ScalarType>::value &&
    WeakForms::is_unary_integral_op<IntegralOp>::value,
  decltype(WeakForms::internal::ScalarTimesUnaryIntegralOpImpl(
    std::declval<ScalarType>(),
    std::declval<IntegralOp>()))>::type
operator*(const IntegralOp &integral, const ScalarType &value)
{
  // Delegate to the other function
  return value * integral;
}



template <typename ScalarType, typename IntegralOp>
typename std::enable_if<
  WeakForms::is_scalar_type<ScalarType>::value &&
    WeakForms::is_binary_integral_op<IntegralOp>::value,
  decltype(WeakForms::internal::ScalarTimesBinaryIntegralOpImpl(
    std::declval<ScalarType>(),
    std::declval<IntegralOp>()))>::type
operator*(const ScalarType &value, const IntegralOp &integral)
{
  return WeakForms::internal::ScalarTimesBinaryIntegralOpImpl(value, integral);
}



template <typename ScalarType, typename IntegralOp>
typename std::enable_if<
  WeakForms::is_scalar_type<ScalarType>::value &&
    WeakForms::is_binary_integral_op<IntegralOp>::value,
  decltype(WeakForms::internal::ScalarTimesBinaryIntegralOpImpl(
    std::declval<ScalarType>(),
    std::declval<IntegralOp>()))>::type
operator*(const IntegralOp &integral, const ScalarType &value)
{
  // Delegate to the other function
  return value * integral;
}



// Declare below the definitions of operator* so that they can be
// recursively used.
namespace WeakForms
{
  namespace internal
  {
    template <typename ScalarType, typename SymbolicIntegralOp, typename>
    DEAL_II_ALWAYS_INLINE inline auto
    ScalarTimesSymbolicIntegralOpImpl(const ScalarType &        value,
                                      const SymbolicIntegralOp &integral)
    {
      using IntegralScalarType = typename SymbolicIntegralOp::ScalarType;
      return integral.get_integral_operation()
        .template integrate<IntegralScalarType>(value *
                                                integral.get_integrand());
    }


    template <typename ScalarType, typename UnaryIntegralOp, typename>
    DEAL_II_ALWAYS_INLINE inline auto
    ScalarTimesUnaryIntegralOpImpl(const ScalarType &     value,
                                   const UnaryIntegralOp &integral)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using IntegrandOpType =
        decltype(value * std::declval<typename UnaryIntegralOp::OpType>());
      using OpType = WeakForms::Operators::UnaryOp<IntegrandOpType,
                                                   UnaryIntegralOp::op_code>;

      return OpType(value * integral.get_operand());
    }


    template <typename ScalarType, typename BinaryIntegralOp, typename>
    DEAL_II_ALWAYS_INLINE inline auto
    ScalarTimesBinaryIntegralOpImpl(const ScalarType &      value,
                                    const BinaryIntegralOp &integral)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using IntegrandLhsOpType =
        decltype(value * std::declval<typename BinaryIntegralOp::LhsOpType>());
      using IntegrandRhsOpType =
        decltype(value * std::declval<typename BinaryIntegralOp::RhsOpType>());
      using OpType = WeakForms::Operators::BinaryOp<IntegrandLhsOpType,
                                                    IntegrandRhsOpType,
                                                    BinaryIntegralOp::op_code>;

      return OpType(value * integral.get_lhs_operand(),
                    value * integral.get_rhs_operand());
    }
  } // namespace internal
} // namespace WeakForms


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_mixed_integral_operators_h
