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

#ifndef dealii_weakforms_sd_expression_internal_h
#define dealii_weakforms_sd_expression_internal_h

#include <deal.II/base/config.h>


#ifdef DEAL_II_WITH_SYMENGINE

#  include <deal.II/base/symmetric_tensor.h>
#  include <deal.II/base/tensor.h>

#  include <deal.II/differentiation/sd.h>

#  include <weak_forms/config.h>
#  include <weak_forms/utilities.h>


WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
  namespace Operators
  {
    namespace internal
    {
      // ===================
      // SD helper functions
      // ===================

      inline std::string
      replace_protected_characters(const std::string &name)
      {
        // Allow SymEngine to parse this field as a string:
        // Required for deserialization.
        // It gets confused when there are numbers in the string name, and
        // we have numbers and some protected characters in the expression
        // name.
        std::string out = name;
        const auto  replace_chars =
          [&out](const char &old_char, const char &new_char)
        { std::replace(out.begin(), out.end(), old_char, new_char); };
        // replace_chars('0', 'A');
        // replace_chars('1', 'B');
        // replace_chars('2', 'C');
        // replace_chars('3', 'D');
        // replace_chars('4', 'E');
        // replace_chars('5', 'F');
        // replace_chars('6', 'G');
        // replace_chars('7', 'H');
        // replace_chars('8', 'I');
        // replace_chars('9', 'J');
        replace_chars(' ', '_');
        replace_chars('(', '_');
        replace_chars(')', '_');
        replace_chars('{', '_');
        replace_chars('}', '_');

        return out;
      }

      template <typename ReturnType>
      typename std::enable_if<
        std::is_same<ReturnType, Differentiation::SD::Expression>::value,
        ReturnType>::type
      make_symbolic(const std::string &name)
      {
        return Differentiation::SD::make_symbol(name);
      }

      template <typename ReturnType>
      typename std::enable_if<
        std::is_same<ReturnType,
                     Tensor<ReturnType::rank,
                            ReturnType::dimension,
                            Differentiation::SD::Expression>>::value,
        ReturnType>::type
      make_symbolic(const std::string &name)
      {
        constexpr int rank = ReturnType::rank;
        constexpr int dim  = ReturnType::dimension;
        return Differentiation::SD::make_tensor_of_symbols<rank, dim>(name);
      }

      template <typename ReturnType>
      typename std::enable_if<
        std::is_same<ReturnType,
                     SymmetricTensor<ReturnType::rank,
                                     ReturnType::dimension,
                                     Differentiation::SD::Expression>>::value,
        ReturnType>::type
      make_symbolic(const std::string &name)
      {
        constexpr int rank = ReturnType::rank;
        constexpr int dim  = ReturnType::dimension;
        return Differentiation::SD::make_symmetric_tensor_of_symbols<rank, dim>(
          name);
      }

      template <typename ExpressionType, typename SymbolicOpField>
      typename SymbolicOpField::template value_type<ExpressionType>
      make_symbolic(const SymbolicOpField &    field,
                    const SymbolicDecorations &decorator)
      {
        using ReturnType =
          typename SymbolicOpField::template value_type<ExpressionType>;

        const std::string name = Utilities::get_deal_II_prefix() + "Field_" +
                                 field.as_ascii(decorator);
        // return make_symbolic<ReturnType>(name);
        return make_symbolic<ReturnType>(replace_protected_characters(name));
      }

    } // namespace internal
  }   // namespace Operators
} // namespace WeakForms


WEAK_FORMS_NAMESPACE_CLOSE


#endif // DEAL_II_WITH_SYMENGINE


#endif // dealii_weakforms_sd_expression_internal_h
