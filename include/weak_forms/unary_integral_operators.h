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

#ifndef dealii_weakforms_unary_integral_operators_h
#define dealii_weakforms_unary_integral_operators_h

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>

#include <weak_forms/config.h>
#include <weak_forms/operator_evaluators.h>
#include <weak_forms/symbolic_decorations.h>
#include <weak_forms/symbolic_integral.h>
#include <weak_forms/type_traits.h>
#include <weak_forms/unary_operators.h>

#include <type_traits>



WEAK_FORMS_NAMESPACE_OPEN



/* ================= Specialization of unary operators: ================= */
/* ======================== Symbolic integrals ========================== */
// These are picked up by the assembler and dealt with internally.
// The wrappers are just required in order to make these compound
// integral operations valid for the assembler.



namespace WeakForms
{
  namespace Operators
  {
    /**
     * Negation operator for symbolic integrals
     */
    template <typename Op>
    class UnaryOp<Op,
                  UnaryOpCodes::negate,
                  typename std::enable_if<is_integral_op<Op>::value>::type>
    {
    public:
      using OpType = Op;

      static const enum UnaryOpCodes op_code = UnaryOpCodes::negate;

      explicit UnaryOp(const Op &operand)
        : operand(operand)
      {}

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return "-" + operand.as_ascii(decorator);
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return "-" + operand.as_latex(decorator);
      }

      const Op &
      get_operand() const
      {
        return operand;
      }

    private:
      const Op operand;
    };

  } // namespace Operators
} // namespace WeakForms



/* ==================== Specialization of type traits ==================== */



#ifndef DOXYGEN


namespace WeakForms
{
  // Can only negate symbolic integrals.
  template <typename T>
  struct is_unary_integral_op<
    T,
    typename std::enable_if<
      is_unary_op<T>::value && is_integral_op<typename T::OpType>::value &&
      T::op_code == Operators::UnaryOpCodes::negate>::type> : std::true_type
  {};

  // I don't know why, but we need this specialisation here.
  template <typename T>
  struct is_integral_op<
    T,
    typename std::enable_if<is_unary_integral_op<T>::value>::type>
    : std::true_type
  {};

} // namespace WeakForms


#endif // DOXYGEN



WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_unary_integral_operators_h
