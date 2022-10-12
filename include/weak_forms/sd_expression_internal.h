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
#  include <weak_forms/spaces.h>
#  include <weak_forms/symbolic_decorations.h>
#  include <weak_forms/symbolic_operators.h>
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


      template <typename T, typename... Us>
      struct is_subspace_field_solution_op
      {
        static constexpr bool value =
          is_subspace_field_solution_op<T>::value &&
          is_subspace_field_solution_op<Us...>::value;
      };

      // Scalar and Vector subspaces
      template <template <class> class SubSpaceViewsType,
                typename SpaceType,
                enum WeakForms::Operators::SymbolicOpCodes OpCode,
                types::solution_index                      solution_index>
      struct is_subspace_field_solution_op<WeakForms::Operators::SymbolicOp<
        SubSpaceViewsType<SpaceType>,
        OpCode,
        void,
        WeakForms::internal::SolutionIndex<solution_index>>>
      {
        static constexpr bool value =
          is_field_solution<SubSpaceViewsType<SpaceType>>::value &&
          is_subspace_view<SubSpaceViewsType<SpaceType>>::value;
      };

      // Tensor and SymmetricTensor subspaces
      template <template <int, class> class SubSpaceViewsType,
                int rank,
                typename SpaceType,
                enum WeakForms::Operators::SymbolicOpCodes OpCode,
                types::solution_index                      solution_index>
      struct is_subspace_field_solution_op<WeakForms::Operators::SymbolicOp<
        SubSpaceViewsType<rank, SpaceType>,
        OpCode,
        void,
        WeakForms::internal::SolutionIndex<solution_index>>>
      {
        static constexpr bool value =
          is_field_solution<SubSpaceViewsType<rank, SpaceType>>::value &&
          is_subspace_view<SubSpaceViewsType<rank, SpaceType>>::value;
      };

      template <typename T>
      struct is_subspace_field_solution_op<T> : std::false_type
      {};


      template <typename T, typename U = void>
      struct CompositeOpHelper;

      // Symbolic op (field op)
      template <typename T>
      struct CompositeOpHelper<
        T,
        typename std::enable_if<is_subspace_field_solution_op<T>::value &&
                                !is_unary_op<T>::value &&
                                !is_binary_op<T>::value>::type>
      {
        // static void
        // print(const T &op)
        // {
        //   std::cout << "op (field solution): "
        //             << op.as_ascii(SymbolicDecorations()) << std::endl;
        // }

        static std::tuple<T>
        get_subspace_field_solution_ops(const T &op)
        {
          return std::make_tuple(op);
        }
      };

      // Symbolic op (not a field op)
      template <typename T>
      struct CompositeOpHelper<
        T,
        typename std::enable_if<!is_subspace_field_solution_op<T>::value &&
                                !is_unary_op<T>::value &&
                                !is_binary_op<T>::value>::type>
      {
        // static void
        // print(const T &op)
        // {
        //   std::cout << "op (not field solution): "
        //             << op.as_ascii(SymbolicDecorations()) << std::endl;
        // }

        static std::tuple<>
        get_subspace_field_solution_ops(const T &op)
        {
          // An empty tuple
          return std::make_tuple();
        }
      };

      // Unary op
      template <typename T>
      struct CompositeOpHelper<
        T,
        typename std::enable_if<is_unary_op<T>::value>::type>
      {
        // static void
        // print(const T &op)
        // {
        //   std::cout << "unary op" << std::endl;

        //   using OpType = typename T::OpType;
        //   CompositeOpHelper<OpType>::print(op.get_operand());
        // }

        static auto
        get_subspace_field_solution_ops(const T &op)
        {
          using OpType = typename T::OpType;
          return CompositeOpHelper<OpType>::get_subspace_field_solution_ops(
            op.get_operand());
        }
      };

      // Binary op
      template <typename T>
      struct CompositeOpHelper<
        T,
        typename std::enable_if<is_binary_op<T>::value>::type>
      {
        // static void
        // print(const T &op)
        // {
        //   std::cout << "binary op" << std::endl;

        //   using LhsOpType = typename T::LhsOpType;
        //   using RhsOpType = typename T::RhsOpType;
        //   CompositeOpHelper<LhsOpType>::print(op.get_lhs_operand());
        //   CompositeOpHelper<RhsOpType>::print(op.get_rhs_operand());
        // }

        static auto
        get_subspace_field_solution_ops(const T &op)
        {
          using LhsOpType = typename T::LhsOpType;
          using RhsOpType = typename T::RhsOpType;

          return std::tuple_cat(
            CompositeOpHelper<LhsOpType>::get_subspace_field_solution_ops(
              op.get_lhs_operand()),
            CompositeOpHelper<RhsOpType>::get_subspace_field_solution_ops(
              op.get_rhs_operand()));
        }
      };

    } // namespace internal
  }   // namespace Operators
} // namespace WeakForms


WEAK_FORMS_NAMESPACE_CLOSE


#endif // DEAL_II_WITH_SYMENGINE


#endif // dealii_weakforms_sd_expression_internal_h
