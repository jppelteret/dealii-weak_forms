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

#ifndef dealii_weakforms_binary_integral_operators_h
#define dealii_weakforms_binary_integral_operators_h

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>

#include <weak_forms/binary_operators.h>
#include <weak_forms/config.h>
#include <weak_forms/operator_evaluators.h>
#include <weak_forms/symbolic_decorations.h>
#include <weak_forms/symbolic_integral.h>
#include <weak_forms/symbolic_operators.h>
#include <weak_forms/type_traits.h>
#include <weak_forms/utilities.h>

#include <type_traits>


WEAK_FORMS_NAMESPACE_OPEN



/* ================== Specialization of binary operators: ================== */
/* ========================== Symbolic integrals =========================== */

// These are picked up by the assembler and dealt with internally.
// The wrappers are just required in order to make these compound
// integral operations valid for the assembler.



namespace WeakForms
{
  namespace Operators
  {
    /**
     * Addition operator for symbolic integrals
     */
    template <typename LhsOp, typename RhsOp>
    class BinaryOp<LhsOp,
                   RhsOp,
                   BinaryOpCodes::add,
                   typename std::enable_if<is_integral_op<LhsOp>::value &&
                                           is_integral_op<RhsOp>::value>::type>
    {
    public:
      using LhsOpType = LhsOp;
      using RhsOpType = RhsOp;

      static_assert(std::is_same<typename LhsOp::ScalarType,
                                 typename RhsOp::ScalarType>::value,
                    "Operands do not have the same scalar type.");
      using ScalarType = typename LhsOp::ScalarType;

      static const enum BinaryOpCodes op_code = BinaryOpCodes::add;

      explicit BinaryOp(const LhsOp &lhs_operand, const RhsOp &rhs_operand)
        : lhs_operand(lhs_operand)
        , rhs_operand(rhs_operand)
      {}

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return lhs_operand.as_ascii(decorator) + " + " +
               rhs_operand.as_ascii(decorator);
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return lhs_operand.as_latex(decorator) + " + " +
               rhs_operand.as_latex(decorator);
      }

      // These need to be exposed for the assembler to accumulate
      // the compound integral expression.
      const LhsOp &
      get_lhs_operand() const
      {
        return lhs_operand;
      }

      const RhsOp &
      get_rhs_operand() const
      {
        return rhs_operand;
      }

    private:
      const LhsOp lhs_operand;
      const RhsOp rhs_operand;
    };



    /**
     * Subtraction operator for symbolic integrals
     */
    template <typename LhsOp, typename RhsOp>
    class BinaryOp<LhsOp,
                   RhsOp,
                   BinaryOpCodes::subtract,
                   typename std::enable_if<is_integral_op<LhsOp>::value &&
                                           is_integral_op<RhsOp>::value>::type>
    {
    public:
      using LhsOpType = LhsOp;
      using RhsOpType = RhsOp;

      static_assert(std::is_same<typename LhsOp::ScalarType,
                                 typename RhsOp::ScalarType>::value,
                    "Operands do not have the same scalar type.");
      using ScalarType = typename LhsOp::ScalarType;

      static const enum BinaryOpCodes op_code = BinaryOpCodes::subtract;

      explicit BinaryOp(const LhsOp &lhs_operand, const RhsOp &rhs_operand)
        : lhs_operand(lhs_operand)
        , rhs_operand(rhs_operand)
      {}

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return lhs_operand.as_ascii(decorator) + " - " +
               rhs_operand.as_ascii(decorator);
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return lhs_operand.as_latex(decorator) + " - " +
               rhs_operand.as_latex(decorator);
      }

      // These need to be exposed for the assembler to accumulate
      // the compound integral expression.
      const LhsOp &
      get_lhs_operand() const
      {
        return lhs_operand;
      }

      const RhsOp &
      get_rhs_operand() const
      {
        return rhs_operand;
      }

    private:
      const LhsOp lhs_operand;
      const RhsOp rhs_operand;
    };



    /**
     * Multiplication operator for symbolic integrals
     */
    // TODO: Test that this is working:
    // Does linear_form(ds, 2.0).dV() == 2.0 * linear_form(ds, 1.0).dV() ?
    // Does bilinear_form(ds, 2.0, Ds).dV() == 2.0 * bilinear_form(ds, 1.0,
    // Ds).dV() ?
    template <typename LhsOp, typename RhsOp>
    class BinaryOp<LhsOp,
                   RhsOp,
                   BinaryOpCodes::multiply,
                   typename std::enable_if<is_integral_op<LhsOp>::value ^
                                           is_integral_op<RhsOp>::value>::type>
    {
      static_assert(!is_unary_integral_op<LhsOp>::value &&
                      !is_unary_integral_op<RhsOp>::value,
                    "Multiplication of symbolic integrals is not permitted.");

    public:
      using LhsOpType = LhsOp;
      using RhsOpType = RhsOp;

      static_assert(std::is_same<typename LhsOp::ScalarType,
                                 typename RhsOp::ScalarType>::value,
                    "Operands do not have the same scalar type.");
      using ScalarType = typename LhsOp::ScalarType;

      static const enum BinaryOpCodes op_code = BinaryOpCodes::multiply;

      explicit BinaryOp(const LhsOp &lhs_operand, const RhsOp &rhs_operand)
        : lhs_operand(lhs_operand)
        , rhs_operand(rhs_operand)
      {}

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return lhs_operand.as_ascii(decorator) + " * " +
               rhs_operand.as_ascii(decorator);
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return lhs_operand.as_latex(decorator) + " " +
               rhs_operand.as_latex(decorator);
      }

      // These need to be exposed for the assembler to accumulate
      // the compound integral expression.
      const LhsOp &
      get_lhs_operand() const
      {
        return lhs_operand;
      }

      const RhsOp &
      get_rhs_operand() const
      {
        return rhs_operand;
      }

    private:
      const LhsOp lhs_operand;
      const RhsOp rhs_operand;
    };

  } // namespace Operators

} // namespace WeakForms



/* ==================== Specialization of type traits ==================== */



#ifndef DOXYGEN


namespace WeakForms
{
  // Can add and subtract integral operations
  template <typename T>
  struct is_binary_integral_op<
    T,
    typename std::enable_if<
      is_binary_op<T>::value &&
      (T::op_code == Operators::BinaryOpCodes::add ||
       T::op_code == Operators::BinaryOpCodes::subtract) &&
      (is_integral_op<typename T::LhsOpType>::value &&
       is_integral_op<typename T::RhsOpType>::value)>::type> : std::true_type
  {};

  // Can only multiply integral operations by arithmetic types.
  template <typename T>
  struct is_binary_integral_op<
    T,
    typename std::enable_if<
      is_binary_op<T>::value &&
      T::op_code == Operators::BinaryOpCodes::multiply &&
      (is_integral_op<typename T::LhsOpType>::value ^
       is_integral_op<typename T::RhsOpType>::value)>::type> : std::true_type
  {};

  // I don't know why, but we need this specialisation here.
  template <typename T>
  struct is_integral_op<
    T,
    typename std::enable_if<is_binary_integral_op<T>::value>::type>
    : std::true_type
  {};

  template <typename T>
  struct operand_requires_braced_decoration<
    T,
    typename std::enable_if<is_integral_op<T>::value>::type> : std::false_type
  {};

} // namespace WeakForms


#endif // DOXYGEN


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_binary_integral_operators_h
