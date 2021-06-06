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

#ifndef dealii_weakforms_binary_operators_h
#define dealii_weakforms_binary_operators_h

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/vectorization.h>

// TODO: Move FeValuesViews::[Scalar/Vector/...]::Output<> into another header??
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/meshworker/scratch_data.h>

#include <boost/core/demangle.hpp> // DEBUGGING

#include <weak_forms/config.h>
#include <weak_forms/operator_evaluators.h>
#include <weak_forms/solution_storage.h>
#include <weak_forms/spaces.h>
#include <weak_forms/symbolic_decorations.h>
#include <weak_forms/symbolic_operators.h>
#include <weak_forms/type_traits.h>
#include <weak_forms/utilities.h>

#include <type_traits>


WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
  namespace Operators
  {
    enum class BinaryOpCodes
    {
      /**
       * Add two operands together.
       */
      add,
      /**
       * Subtract one operand from another.
       */
      subtract,
      /**
       * Multiply two operands together.
       */
      multiply,
      /**
       * Divide two operands together.
       *
       * It is expected that the second operand is a scalar (rank-0) type.
       */
      divide,

      // --- Scalar operations ---
      /**
       * Raise one operand to the power of a second operand.
       */
      power
      // atan2
      // max
      // min

      // --- Tensor operations ---
      /**
       * Cross product (between two vector operands)
       */
      // cross_product
      /**
       * Outer product (between two tensor operands)
       */
      // outer_product

      // --- Tensor contractions ---
      // scalar product,
      // double contract,
      // contract,
    };



    /**
     * Exception denoting that a class requires some specialization
     * in order to be used.
     */
    DeclExceptionMsg(
      ExcRequiresBinaryOperatorSpecialization,
      "This function is called in a class that is expected to be specialized "
      "for binary operations. All binary operators should be specialized, with "
      "a structure matching that of the exemplar class.");



    /**
     * Exception denoting that a class requires some specialization
     * in order to be used.
     */
    template <typename LhsOpType, typename RhsOpType>
    DeclException2(
      ExcRequiresBinaryOperatorSpecialization2,
      LhsOpType,
      RhsOpType,
      << "This function is called in a class that is expected to be specialized "
      << "for binary operations. All binary operators should be specialized, with "
      << "a structure matching that of the exemplar class.\n\n"
      << "LHS op type" << boost::core::demangle(typeid(arg1).name()) << "\n\n"
      << "RHS op type" << boost::core::demangle(typeid(arg2).name()) << "\n");


    /**
     * Exception denoting that a binary operation has not been defined.
     */
    DeclException1(ExcBinaryOperatorNotDefined,
                   enum BinaryOpCodes,
                   << "The binary operator with code " << static_cast<int>(arg1)
                   << " has not been defined.");



    /**
     * @tparam Op
     * @tparam OpCode
     * @tparam UnderlyingType Underlying number type (double, std::complex<double>, etc.).
     * This is necessary because some specializations of the class do not use
     * the number type in the specialization itself, but they may rely on the
     * type in their definitions (e.g. class members).
     */
    template <typename LhsOp,
              typename RhsOp,
              enum BinaryOpCodes OpCode,
              typename UnderlyingType = void>
    class BinaryOp
    {
    public:
      using LhsOpType = LhsOp;
      using RhsOpType = RhsOp;

      static const enum BinaryOpCodes op_code = OpCode;

      explicit BinaryOp(const LhsOp &lhs_operand, const RhsOp &rhs_operand)
        : lhs_operand(lhs_operand)
        , rhs_operand(rhs_operand)
      {
        std::cout << "LHS op type: "
                  << boost::core::demangle(typeid(lhs_operand).name())
                  << std::endl;
        std::cout << "RHS op type: "
                  << boost::core::demangle(typeid(rhs_operand).name())
                  << std::endl;
        AssertThrow(false, ExcRequiresBinaryOperatorSpecialization());
        // AssertThrow(false,
        // ExcRequiresBinaryOperatorSpecialization2<LhsOp,RhsOp>(lhs_operand,
        // rhs_operand));
      }

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        (void)decorator;
        AssertThrow(false, ExcRequiresBinaryOperatorSpecialization());
        return "";
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        (void)decorator;
        AssertThrow(false, ExcRequiresBinaryOperatorSpecialization());
        return "";
      }

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
    }; // class BinaryOp


    namespace internal
    {
      // Assume that everything is compatible to add together or subtract apart
      template <typename LhsOp, typename RhsOp, typename T = void>
      struct has_incompatible_spaces_for_addition_subtraction : std::false_type
      {};


      // Cannot add or subtract a test function and field solution
      template <typename LhsOp, typename RhsOp>
      struct has_incompatible_spaces_for_addition_subtraction<
        LhsOp,
        RhsOp,
        typename std::enable_if<is_test_function_op<LhsOp>::value &&
                                is_field_solution_op<RhsOp>::value>::type>
        : std::true_type
      {};


      // Cannot add or subtract a test function and trial solution
      template <typename LhsOp, typename RhsOp>
      struct has_incompatible_spaces_for_addition_subtraction<
        LhsOp,
        RhsOp,
        typename std::enable_if<is_test_function_op<LhsOp>::value &&
                                (is_trial_solution_op<RhsOp>::value ||
                                 has_trial_solution_op<RhsOp>::value)>::type>
        : std::true_type
      {};


      // Cannot add or subtract a field solution and trial solution
      template <typename LhsOp, typename RhsOp>
      struct has_incompatible_spaces_for_addition_subtraction<
        LhsOp,
        RhsOp,
        typename std::enable_if<is_field_solution_op<LhsOp>::value &&
                                (is_trial_solution_op<RhsOp>::value ||
                                 has_trial_solution_op<RhsOp>::value)>::type>
        : std::true_type
      {};


      // Check a + (b1+b2)
      template <typename LhsOp, typename RhsOp1, typename RhsOp2>
      struct has_incompatible_spaces_for_addition_subtraction<
        LhsOp,
        BinaryOp<RhsOp1, RhsOp2, BinaryOpCodes::add>,
        typename std::enable_if<
          has_incompatible_spaces_for_addition_subtraction<LhsOp,
                                                           RhsOp1>::value ||
          has_incompatible_spaces_for_addition_subtraction<LhsOp, RhsOp2>::
            value>::type> : std::true_type
      {};


      // Check a + (b1-b2)
      template <typename LhsOp, typename RhsOp1, typename RhsOp2>
      struct has_incompatible_spaces_for_addition_subtraction<
        LhsOp,
        BinaryOp<RhsOp1, RhsOp2, BinaryOpCodes::subtract>,
        typename std::enable_if<
          has_incompatible_spaces_for_addition_subtraction<LhsOp,
                                                           RhsOp1>::value ||
          has_incompatible_spaces_for_addition_subtraction<LhsOp, RhsOp2>::
            value>::type> : std::true_type
      {};


      // Deal with the combinatorics of the above by checking both combinations
      // [Lhs,Rhs] and [Rhs,Lhs] together. We negate the condition at the same
      // time.
      template <typename LhsOp, typename RhsOp>
      struct has_compatible_spaces_for_addition_subtraction
        : std::conditional<
            has_incompatible_spaces_for_addition_subtraction<LhsOp,
                                                             RhsOp>::value ||
              has_incompatible_spaces_for_addition_subtraction<RhsOp,
                                                               LhsOp>::value,
            std::false_type,
            std::true_type>::type
      {};

    } // namespace internal

  } // namespace Operators
} // namespace WeakForms



/* ================== Specialization of binary operators: ================== */
/* =================== Integrands of symbolic integrals ==================== */
// TODO: Move this to another file "integral_operators"?


namespace WeakForms
{
  namespace Operators
  {
    // A little bit of CRTP, with a workaround to deal with templates
    // in the derived class.
    // See https://stackoverflow.com/a/45801893
    template <typename Derived>
    struct BinaryOpTypeTraits;

    namespace internal
    {
      template <typename LhsOpType, typename RhsOpType, typename T = void>
      struct binary_op_test_trial_traits;

      template <typename LhsOpType, typename RhsOpType>
      struct binary_op_test_trial_traits<
        LhsOpType,
        RhsOpType,
        typename std::enable_if<
          !is_or_has_test_function_or_trial_solution_op<LhsOpType>::value &&
          !is_or_has_test_function_or_trial_solution_op<RhsOpType>::value>::
          type>
      {
        template <typename T>
        using return_type = std::vector<T>;
      };

      template <typename LhsOpType, typename RhsOpType>
      struct binary_op_test_trial_traits<
        LhsOpType,
        RhsOpType,
        typename std::enable_if<
          is_or_has_test_function_or_trial_solution_op<LhsOpType>::value &&
          !is_or_has_test_function_or_trial_solution_op<RhsOpType>::value>::
          type>
      {
        template <typename T>
        using return_type = std::vector<std::vector<T>>;
      };

      template <typename LhsOpType, typename RhsOpType>
      struct binary_op_test_trial_traits<
        LhsOpType,
        RhsOpType,
        typename std::enable_if<
          is_or_has_test_function_or_trial_solution_op<RhsOpType>::value &&
          !is_or_has_test_function_or_trial_solution_op<LhsOpType>::value>::
          type>
      {
        template <typename T>
        using return_type = std::vector<std::vector<T>>;
      };

      template <typename LhsOpType, typename RhsOpType>
      struct binary_op_test_trial_traits<
        LhsOpType,
        RhsOpType,
        typename std::enable_if<
          is_or_has_test_function_or_trial_solution_op<LhsOpType>::value &&
          is_or_has_test_function_or_trial_solution_op<RhsOpType>::value>::type>
      {
        // static_assert(false, "Both LhsOpType and RhsOpType cannot
        // simultaneously be a test function or trial solution.");
        // static_assert(
        //   LhsOpType::n_components == RhsOpType::n_components,
        //   "LhsOp and RhsOp do not have the same number of components.");
        // static_assert(std::is_same<typename LhsOpType::extractor_type,
        //                            typename
        //                            RhsOpType::extractor_type>::value,
        //               "LhsOp and RhsOp do not have the same extractor
        //               type.");

        template <typename T>
        using return_type = std::vector<std::vector<T>>;
      };
    } // namespace internal


/**
 * A macro to implement the common parts of a binary op type trait class.
 * Note that this should used at the very end of the class definition, as
 * the @p return_type relies on the @p value_type to be defined.
 *
 * What remains to be defined are:
 * - static const enum BinaryOpCodes op_code
 * - static const int rank
 * - template <typename ScalarType> using value_type = ...;
 */
#define DEAL_II_BINARY_OP_TYPE_TRAITS_COMMON_IMPL(LhsOp, RhsOp)                \
  /**                                                                          \
   *                                                                           \
   */                                                                          \
  using LhsOpType = LhsOp;                                                     \
  /**                                                                          \
   *                                                                           \
   */                                                                          \
  using RhsOpType = RhsOp;                                                     \
                                                                               \
  static_assert(                                                               \
    LhsOp::dimension == RhsOp::dimension,                                      \
    "Binary operator requires that operators have the same dimension.");       \
                                                                               \
  static_assert(                                                               \
    LhsOp::space_dimension == RhsOp::space_dimension,                          \
    "Binary operator requires that operators have the same space dimension."); \
                                                                               \
  /**                                                                          \
   * Dimension in which this object operates.                                  \
   */                                                                          \
  static const unsigned int dimension = LhsOp::dimension;                      \
                                                                               \
  /**                                                                          \
   * Dimension of the space in which this object operates.                     \
   */                                                                          \
  static const unsigned int space_dimension = LhsOp::space_dimension;          \
                                                                               \
  /**                                                                          \
   *                                                                           \
   */                                                                          \
  template <typename ScalarType>                                               \
  using return_type =                                                          \
    typename internal::binary_op_test_trial_traits<LhsOpType, RhsOpType>::     \
      template return_type<value_type<ScalarType>>;



    template <typename LhsOp, typename RhsOp>
    struct BinaryOpTypeTraits<BinaryOp<LhsOp, RhsOp, BinaryOpCodes::add>>
    {
      static const enum BinaryOpCodes op_code = BinaryOpCodes::add;

      static_assert(LhsOp::rank == RhsOp::rank,
                    "Addition requires that operators have the same rank.");

      static const int rank = LhsOp::rank;

      template <typename ScalarType>
      using value_type = decltype(
        std::declval<typename LhsOp::template value_type<ScalarType>>() +
        std::declval<typename RhsOp::template value_type<ScalarType>>());

      DEAL_II_BINARY_OP_TYPE_TRAITS_COMMON_IMPL(LhsOp, RhsOp)
    };



    template <typename LhsOp, typename RhsOp>
    struct BinaryOpTypeTraits<BinaryOp<LhsOp, RhsOp, BinaryOpCodes::subtract>>
    {
      static const enum BinaryOpCodes op_code = BinaryOpCodes::subtract;

      static_assert(LhsOp::rank == RhsOp::rank,
                    "Subtraction requires that operators have the same rank.");

      static const int rank = LhsOp::rank;

      template <typename ScalarType>
      using value_type = decltype(
        std::declval<typename LhsOp::template value_type<ScalarType>>() -
        std::declval<typename RhsOp::template value_type<ScalarType>>());

      DEAL_II_BINARY_OP_TYPE_TRAITS_COMMON_IMPL(LhsOp, RhsOp)
    };



    template <typename LhsOp, typename RhsOp>
    struct BinaryOpTypeTraits<BinaryOp<LhsOp, RhsOp, BinaryOpCodes::multiply>>
    {
      static const enum BinaryOpCodes op_code = BinaryOpCodes::multiply;

      template <typename ScalarType>
      using value_type = decltype(
        std::declval<typename LhsOp::template value_type<ScalarType>>() *
        std::declval<typename RhsOp::template value_type<ScalarType>>());

      static const int rank =
        WeakForms::Utilities::ValueHelper<value_type<double>>::rank;

      DEAL_II_BINARY_OP_TYPE_TRAITS_COMMON_IMPL(LhsOp, RhsOp)
    };



    template <typename LhsOp, typename RhsOp>
    struct BinaryOpTypeTraits<BinaryOp<LhsOp, RhsOp, BinaryOpCodes::divide>>
    {
      static const enum BinaryOpCodes op_code = BinaryOpCodes::divide;

      static_assert(
        RhsOp::rank == 0,
        "Division requires that the RHS operand is of rank-0 (i.e. scalar valued).");

      static const int rank = LhsOp::rank;

      template <typename ScalarType>
      using value_type = decltype(
        std::declval<typename LhsOp::template value_type<ScalarType>>() /
        std::declval<typename RhsOp::template value_type<ScalarType>>());

      DEAL_II_BINARY_OP_TYPE_TRAITS_COMMON_IMPL(LhsOp, RhsOp)
    };



    // Using a namespace inside a decltype
    // https://stackoverflow.com/q/50456521
    //
    // See
    // https://dealii.org/developer/doxygen/deal.II/classVectorizedArray.html
    // for operations supported by VectorizedArray
    namespace internal
    {
      template <typename T1, typename T2>
      auto
      pow_impl(const T1 &t1, const T2 &t2)
      {
        using std::pow;
        return pow(t1, t2);
      }
    } // namespace internal



    template <typename LhsOp, typename RhsOp>
    struct BinaryOpTypeTraits<BinaryOp<LhsOp, RhsOp, BinaryOpCodes::power>>
    {
      static const enum BinaryOpCodes op_code = BinaryOpCodes::power;

      static_assert(
        LhsOp::rank == 0,
        "Power requires that the LHS operand is of rank-0 (i.e. scalar valued).");

      static_assert(
        RhsOp::rank == 0,
        "Power requires that the RHS operand is of rank-0 (i.e. scalar valued).");

      static const int rank = 0;

      template <typename ScalarType>
      using value_type = decltype(internal::pow_impl(
        std::declval<typename LhsOp::template value_type<ScalarType>>(),
        std::declval<typename RhsOp::template value_type<ScalarType>>()));

      DEAL_II_BINARY_OP_TYPE_TRAITS_COMMON_IMPL(LhsOp, RhsOp)
    };


#undef DEAL_II_BINARY_OP_TYPE_TRAITS_COMMON_IMPL


    template <typename Derived>
    class BinaryOpBase
    {
    public:
      using LhsOpType = typename BinaryOpTypeTraits<Derived>::LhsOpType;
      using RhsOpType = typename BinaryOpTypeTraits<Derived>::RhsOpType;

      template <typename ScalarType>
      using value_type =
        typename BinaryOpTypeTraits<Derived>::template value_type<ScalarType>;
      template <typename ScalarType>
      using return_type =
        typename BinaryOpTypeTraits<Derived>::template return_type<ScalarType>;

      static const enum BinaryOpCodes op_code =
        BinaryOpTypeTraits<Derived>::op_code;

      static const int dimension = BinaryOpTypeTraits<Derived>::dimension;
      static const int space_dimension =
        BinaryOpTypeTraits<Derived>::space_dimension;

      static const int rank = BinaryOpTypeTraits<Derived>::rank;

      BinaryOpBase(const Derived &derived)
        : derived(derived)
      {}

      UpdateFlags
      get_update_flags() const
      {
        return derived.get_lhs_operand().get_update_flags() |
               derived.get_rhs_operand().get_update_flags();
      }


      // ---- Operators NOT for test functions / trial solutions ---
      // So these are restricted to symbolic ops, functors (standard and
      // cache)  and field solutions as leaf operations.

      template <typename ScalarType>
      auto
      operator()(
        const typename LhsOpType::template return_type<ScalarType> &lhs_value,
        const typename RhsOpType::template return_type<ScalarType> &rhs_value)
        const -> typename std::enable_if<
          !is_or_has_test_function_or_trial_solution_op<LhsOpType>::value &&
            !is_or_has_test_function_or_trial_solution_op<RhsOpType>::value,
          return_type<ScalarType>>::type
      {
        Assert(lhs_value.size() == rhs_value.size(),
               ExcDimensionMismatch(lhs_value.size(), rhs_value.size()));

        return_type<ScalarType> out;
        const unsigned int      n_q_points = lhs_value.size();
        out.reserve(n_q_points);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          out.emplace_back(
            derived.template operator()<ScalarType>(lhs_value[q_point],
                                                    rhs_value[q_point]));

        return out;
      }

      /**
       * Return values at all quadrature points
       *
       * It is expected that this operator never be directly called on a
       * test function or trial solution, but rather that the latter be unpacked
       * manually within the assembler itself.
       * We also cannot expose this function when the operand types are
       * symbolic integrals.
       */
      template <typename ScalarType, int dim, int spacedim>
      auto
      operator()(const FEValuesBase<dim, spacedim> &fe_values) const ->
        typename std::enable_if<
          !is_or_has_test_function_or_trial_solution_op<LhsOpType>::value &&
            !is_or_has_test_function_or_trial_solution_op<RhsOpType>::value &&
            !is_or_has_evaluated_with_scratch_data<LhsOpType>::value &&
            !is_or_has_evaluated_with_scratch_data<RhsOpType>::value,
          return_type<ScalarType>>::type
      {
        return internal::BinaryOpEvaluator<LhsOpType, RhsOpType>::
          template apply<ScalarType>(derived,
                                     derived.get_lhs_operand(),
                                     derived.get_rhs_operand(),
                                     fe_values);
      }

      template <typename ScalarType, int dim, int spacedim>
      auto
      operator()(const FEValuesBase<dim, spacedim> &     fe_values,
                 MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                 const std::vector<std::string> &solution_names) const ->
        typename std::enable_if<
          !is_or_has_test_function_or_trial_solution_op<LhsOpType>::value &&
            !is_or_has_test_function_or_trial_solution_op<RhsOpType>::value &&
            (is_or_has_evaluated_with_scratch_data<LhsOpType>::value ||
             is_or_has_evaluated_with_scratch_data<RhsOpType>::value),
          return_type<ScalarType>>::type
      {
        return internal::BinaryOpEvaluator<LhsOpType, RhsOpType>::
          template apply<ScalarType>(derived,
                                     derived.get_lhs_operand(),
                                     derived.get_rhs_operand(),
                                     fe_values,
                                     scratch_data,
                                     solution_names);
      }


      // ---- Operators for test functions / trial solutions ---
      // So these are for when a test function or trial solution is one or more
      // of the leaf operations. The other leaves may or may not be
      // symbolic ops, functors (standard and cache) and field solutions.

      template <typename ScalarType,
                typename LhsOpType2 = LhsOpType,
                typename RhsOpType2 = RhsOpType>
      return_type<ScalarType>
      operator()(
        const typename LhsOpType::template return_type<ScalarType> &lhs_value,
        const typename RhsOpType::template return_type<ScalarType> &rhs_value,
        typename std::enable_if<
          is_or_has_test_function_or_trial_solution_op<LhsOpType2>::value &&
          is_or_has_test_function_or_trial_solution_op<RhsOpType2>::value>::type
          * = nullptr) const
      {
        Assert(lhs_value.size() == rhs_value.size(),
               ExcDimensionMismatch(lhs_value.size(), rhs_value.size()));

        Assert(lhs_value[0].size() > 0,
               ExcMessage("Uninitialized q-point entry"));
        const unsigned int n_dofs     = lhs_value.size();
        const unsigned int n_q_points = lhs_value[0].size();

        return_type<ScalarType> out(n_dofs);
        for (unsigned int dof_index = 0; dof_index < n_dofs; ++dof_index)
          {
            Assert(lhs_value[dof_index].size() == n_q_points,
                   ExcDimensionMismatch(lhs_value[dof_index].size(),
                                        n_q_points));
            Assert(rhs_value[dof_index].size() == n_q_points,
                   ExcDimensionMismatch(rhs_value[dof_index].size(),
                                        n_q_points));

            out[dof_index].reserve(n_q_points);
          }

        for (unsigned int dof_index = 0; dof_index < n_dofs; ++dof_index)
          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            out[dof_index].emplace_back(derived.template operator()<ScalarType>(
              lhs_value[dof_index][q_point], rhs_value[dof_index][q_point]));

        return out;
      }

      template <typename ScalarType,
                typename LhsOpType2 = LhsOpType,
                typename RhsOpType2 = RhsOpType>
      return_type<ScalarType>
      operator()(
        const typename LhsOpType::template return_type<ScalarType> &lhs_value,
        const typename RhsOpType::template return_type<ScalarType> &rhs_value,
        typename std::enable_if<
          is_or_has_test_function_or_trial_solution_op<LhsOpType2>::value &&
          !is_or_has_test_function_or_trial_solution_op<RhsOpType2>::value>::
          type * = nullptr) const
      {
        const unsigned int n_dofs     = lhs_value.size();
        const unsigned int n_q_points = rhs_value.size();

        return_type<ScalarType> out(n_dofs);
        for (unsigned int dof_index = 0; dof_index < n_dofs; ++dof_index)
          {
            Assert(lhs_value[dof_index].size() == n_q_points,
                   ExcDimensionMismatch(lhs_value[dof_index].size(),
                                        n_q_points));

            out[dof_index].reserve(n_q_points);
          }

        for (unsigned int dof_index = 0; dof_index < n_dofs; ++dof_index)
          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            out[dof_index].emplace_back(derived.template operator()<ScalarType>(
              lhs_value[dof_index][q_point], rhs_value[q_point]));

        return out;
      }

      template <typename ScalarType,
                typename LhsOpType2 = LhsOpType,
                typename RhsOpType2 = RhsOpType>
      return_type<ScalarType>
      operator()(
        const typename LhsOpType::template return_type<ScalarType> &lhs_value,
        const typename RhsOpType::template return_type<ScalarType> &rhs_value,
        typename std::enable_if<
          !is_or_has_test_function_or_trial_solution_op<LhsOpType2>::value &&
          is_or_has_test_function_or_trial_solution_op<RhsOpType2>::value>::type
          * = nullptr) const
      {
        const unsigned int n_dofs     = rhs_value.size();
        const unsigned int n_q_points = lhs_value.size();

        return_type<ScalarType> out(n_dofs);
        for (unsigned int dof_index = 0; dof_index < n_dofs; ++dof_index)
          {
            Assert(rhs_value[dof_index].size() == n_q_points,
                   ExcDimensionMismatch(rhs_value[dof_index].size(),
                                        n_q_points));

            out[dof_index].reserve(n_q_points);
          }

        for (unsigned int dof_index = 0; dof_index < n_dofs; ++dof_index)
          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            out[dof_index].emplace_back(derived.template operator()<ScalarType>(
              lhs_value[q_point], rhs_value[dof_index][q_point]));

        return out;
      }

      template <typename ScalarType, int dim, int spacedim>
      auto
      operator()(const FEValuesBase<dim, spacedim> &fe_values_dofs,
                 const FEValuesBase<dim, spacedim> &fe_values_op) const ->
        typename std::enable_if<
          (is_or_has_test_function_or_trial_solution_op<LhsOpType>::value ||
           is_or_has_test_function_or_trial_solution_op<RhsOpType>::value) &&
            !is_or_has_evaluated_with_scratch_data<LhsOpType>::value &&
            !is_or_has_evaluated_with_scratch_data<RhsOpType>::value,
          return_type<ScalarType>>::type
      {
        return internal::BinaryOpEvaluator<LhsOpType, RhsOpType>::
          template apply<ScalarType>(derived,
                                     derived.get_lhs_operand(),
                                     derived.get_rhs_operand(),
                                     fe_values_dofs,
                                     fe_values_op);
      }

      // Hmm... this is probably a bit inefficient since we mix the solution
      // extraction with the q-point shape function operation
      template <typename ScalarType, int dim, int spacedim>
      auto
      operator()(const FEValuesBase<dim, spacedim> &     fe_values_dofs,
                 const FEValuesBase<dim, spacedim> &     fe_values_op,
                 MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                 const std::vector<std::string> &solution_names) const ->
        typename std::enable_if<
          (is_or_has_test_function_or_trial_solution_op<LhsOpType>::value ||
           is_or_has_test_function_or_trial_solution_op<RhsOpType>::value) &&
            (is_or_has_evaluated_with_scratch_data<LhsOpType>::value ||
             is_or_has_evaluated_with_scratch_data<RhsOpType>::value),
          return_type<ScalarType>>::type
      {
        return internal::BinaryOpEvaluator<LhsOpType, RhsOpType>::
          template apply<ScalarType>(derived,
                                     derived.get_lhs_operand(),
                                     derived.get_rhs_operand(),
                                     fe_values_dofs,
                                     fe_values_op,
                                     scratch_data,
                                     solution_names);
      }

    private:
      const Derived &derived;
    };



/**
 * A macro to implement the common parts of a binary op class.
 * It is expected that the unary op derives from a
 * BinaryOpBase<UnaryOp<LhsOp, RhsOp, BinaryOpCode>> .
 * What remains to be implemented are the public functions:
 *  - as_ascii()
 *  - as_latex()
 *  - operator()
 *
 * @note It is intended that this should used immediately after class
 * definition is opened.
 */
#define DEAL_II_BINARY_OP_COMMON_IMPL(LhsOp, RhsOp, BinaryOpCode)          \
private:                                                                   \
  using Base   = BinaryOpBase<BinaryOp<LhsOp, RhsOp, BinaryOpCode>>;       \
  using Traits = BinaryOpTypeTraits<BinaryOp<LhsOp, RhsOp, BinaryOpCode>>; \
                                                                           \
public:                                                                    \
  using LhsOpType = typename Traits::LhsOpType;                            \
  using RhsOpType = typename Traits::RhsOpType;                            \
                                                                           \
  template <typename ScalarType>                                           \
  using value_type = typename Traits::template value_type<ScalarType>;     \
  template <typename ScalarType>                                           \
  using return_type = typename Traits::template return_type<ScalarType>;   \
                                                                           \
  using Base::dimension;                                                   \
  using Base::op_code;                                                     \
  using Base::rank;                                                        \
  using Base::space_dimension;                                             \
  using Base::get_update_flags;                                            \
  using Base::operator();                                                  \
                                                                           \
  explicit BinaryOp(const LhsOp &lhs_operand, const RhsOp &rhs_operand)    \
    : Base(*this)                                                          \
    , lhs_operand(lhs_operand)                                             \
    , rhs_operand(rhs_operand)                                             \
  {}                                                                       \
                                                                           \
  /**                                                                      \
   * Required to support operands that access objects with a limited       \
   * lifetime, e.g. ScalarFunctionFunctor, TensorFunctionFunctor           \
   */                                                                      \
  BinaryOp(const BinaryOp &rhs)                                            \
    : Base(*this)                                                          \
    , lhs_operand(rhs.lhs_operand)                                         \
    , rhs_operand(rhs.rhs_operand)                                         \
  {}                                                                       \
                                                                           \
  /**                                                                      \
   * Needs to be exposed for the base class to use                         \
   */                                                                      \
  const LhsOp &get_lhs_operand() const                                     \
  {                                                                        \
    return lhs_operand;                                                    \
  }                                                                        \
                                                                           \
  /**                                                                      \
   * Needs to be exposed for the base class to use                         \
   */                                                                      \
  const RhsOp &get_rhs_operand() const                                     \
  {                                                                        \
    return rhs_operand;                                                    \
  }                                                                        \
                                                                           \
private:                                                                   \
  const LhsOp lhs_operand;                                                 \
  const RhsOp rhs_operand;



    /**
     * Addition operator for integrands of symbolic integrals
     */
    template <typename LhsOp, typename RhsOp>
    class BinaryOp<LhsOp,
                   RhsOp,
                   BinaryOpCodes::add,
                   typename std::enable_if<!is_integral_op<LhsOp>::value &&
                                           !is_integral_op<RhsOp>::value>::type>
      : public BinaryOpBase<BinaryOp<LhsOp, RhsOp, BinaryOpCodes::add>>
    {
      static_assert(
        internal::has_compatible_spaces_for_addition_subtraction<LhsOp,
                                                                 RhsOp>::value,
        "It is not permissible to add incompatible spaces together.");

      DEAL_II_BINARY_OP_COMMON_IMPL(LhsOp, RhsOp, BinaryOpCodes::add)

    public:
      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return "[" + lhs_operand.as_ascii(decorator) + " + " +
               rhs_operand.as_ascii(decorator) + "]";
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return Utilities::LaTeX::decorate_term(lhs_operand.as_latex(decorator) +
                                               " + " +
                                               rhs_operand.as_latex(decorator));
      }

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename LhsOp::template value_type<ScalarType> &lhs_value,
        const typename RhsOp::template value_type<ScalarType> &rhs_value) const
      {
        return lhs_value + rhs_value;
      }
    };



    /**
     * Subtraction operator for integrands of symbolic integrals
     */
    template <typename LhsOp, typename RhsOp>
    class BinaryOp<LhsOp,
                   RhsOp,
                   BinaryOpCodes::subtract,
                   typename std::enable_if<!is_integral_op<LhsOp>::value &&
                                           !is_integral_op<RhsOp>::value>::type>
      : public BinaryOpBase<BinaryOp<LhsOp, RhsOp, BinaryOpCodes::subtract>>
    {
      static_assert(
        internal::has_compatible_spaces_for_addition_subtraction<LhsOp,
                                                                 RhsOp>::value,
        "It is not permissible to subtract incompatible spaces from one another.");

      DEAL_II_BINARY_OP_COMMON_IMPL(LhsOp, RhsOp, BinaryOpCodes::subtract)

    public:
      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return "[" + lhs_operand.as_ascii(decorator) + " - " +
               rhs_operand.as_ascii(decorator) + "]";
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return Utilities::LaTeX::decorate_term(lhs_operand.as_latex(decorator) +
                                               " - " +
                                               rhs_operand.as_latex(decorator));
      }

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename LhsOp::template value_type<ScalarType> &lhs_value,
        const typename RhsOp::template value_type<ScalarType> &rhs_value) const
      {
        return lhs_value - rhs_value;
      }
    };



    /**
     * Multiplication operator for integrands of symbolic integrals
     */
    template <typename LhsOp, typename RhsOp>
    class BinaryOp<LhsOp,
                   RhsOp,
                   BinaryOpCodes::multiply,
                   typename std::enable_if<!is_integral_op<LhsOp>::value &&
                                           !is_integral_op<RhsOp>::value>::type>
      : public BinaryOpBase<BinaryOp<LhsOp, RhsOp, BinaryOpCodes::multiply>>
    {
      DEAL_II_BINARY_OP_COMMON_IMPL(LhsOp, RhsOp, BinaryOpCodes::multiply)

    public:
      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return "[" + lhs_operand.as_ascii(decorator) + " * " +
               rhs_operand.as_ascii(decorator) + "]";
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        constexpr unsigned int n_contracting_indices = WeakForms::Utilities::
          FullIndexContraction<LhsOp, RhsOp>::n_contracting_indices;
        const std::string symb_mult =
          Utilities::LaTeX::get_symbol_multiply(n_contracting_indices);
        return Utilities::LaTeX::decorate_term(lhs_operand.as_latex(decorator) +
                                               symb_mult +
                                               rhs_operand.as_latex(decorator));
      }

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename LhsOp::template value_type<ScalarType> &lhs_value,
        const typename RhsOp::template value_type<ScalarType> &rhs_value) const
      {
        return lhs_value * rhs_value;
      }

      // Support test function / trial solution ops
      // In this case, we get back from the test/trial a vector and we multiply
      // it by a single value. So we need to do that in a vector-type operation.
      template <typename ScalarType>
      return_type<ScalarType>
      operator()(
        const typename LhsOp::template return_type<ScalarType> &lhs_value,
        const typename RhsOp::template value_type<ScalarType> & rhs_value) const
      {
        return_type<ScalarType> out;
        const unsigned int      size = lhs_value.size();
        out.reserve(size);

        for (unsigned int i = 0; i < size; ++i)
          out.emplace_back(
            this->template operator()<ScalarType>(lhs_value[i], rhs_value));

        return out;
      }

      // Support test function / trial solution ops
      // In this case, we get back from the test/trial a vector and we multiply
      // it by a single value. So we need to do that in a vector-type operation.
      template <typename ScalarType>
      return_type<ScalarType>
      operator()(
        const typename LhsOp::template value_type<ScalarType> & lhs_value,
        const typename RhsOp::template return_type<ScalarType> &rhs_value) const
      {
        return_type<ScalarType> out;
        const unsigned int      size = rhs_value.size();
        out.reserve(size);

        for (unsigned int i = 0; i < size; ++i)
          out.emplace_back(
            this->template operator()<ScalarType>(lhs_value, rhs_value[i]));

        return out;
      }
    };



    /**
     * Division operator for integrands of symbolic integrals
     */
    template <typename LhsOp, typename RhsOp>
    class BinaryOp<LhsOp,
                   RhsOp,
                   BinaryOpCodes::divide,
                   typename std::enable_if<!is_integral_op<LhsOp>::value &&
                                           !is_integral_op<RhsOp>::value>::type>
      : public BinaryOpBase<BinaryOp<LhsOp, RhsOp, BinaryOpCodes::divide>>
    {
      DEAL_II_BINARY_OP_COMMON_IMPL(LhsOp, RhsOp, BinaryOpCodes::divide)

    public:
      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return "[" + lhs_operand.as_ascii(decorator) + " / " +
               rhs_operand.as_ascii(decorator) + "]";
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return Utilities::LaTeX::decorate_fraction(
          lhs_operand.as_latex(decorator), rhs_operand.as_latex(decorator));
      }

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename LhsOp::template value_type<ScalarType> &lhs_value,
        const typename RhsOp::template value_type<ScalarType> &rhs_value) const
      {
        return lhs_value / rhs_value;
      }

      // Support test function / trial solution ops
      // In this case, we get back from the test/trial a vector and we divide
      // it by a single value. So we need to do that in a vector-type operation.
      template <typename ScalarType>
      return_type<ScalarType>
      operator()(
        const typename LhsOp::template return_type<ScalarType> &lhs_value,
        const typename RhsOp::template value_type<ScalarType> & rhs_value) const
      {
        return_type<ScalarType> out;
        const unsigned int      size = lhs_value.size();
        out.reserve(size);

        for (unsigned int i = 0; i < size; ++i)
          out.emplace_back(
            this->template operator()<ScalarType>(lhs_value[i], rhs_value));

        return out;
      }
    };



    /**
     * Power operator for integrands of symbolic integrals
     */
    template <typename LhsOp, typename RhsOp>
    class BinaryOp<LhsOp,
                   RhsOp,
                   BinaryOpCodes::power,
                   typename std::enable_if<!is_integral_op<LhsOp>::value &&
                                           !is_integral_op<RhsOp>::value>::type>
      : public BinaryOpBase<BinaryOp<LhsOp, RhsOp, BinaryOpCodes::power>>
    {
      DEAL_II_BINARY_OP_COMMON_IMPL(LhsOp, RhsOp, BinaryOpCodes::power)

    public:
      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return "[" + lhs_operand.as_ascii(decorator) + "]^[" +
               rhs_operand.as_ascii(decorator) + "]";
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return Utilities::LaTeX::decorate_power(
          lhs_operand.as_latex(decorator), rhs_operand.as_latex(decorator));
      }

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename LhsOp::template value_type<ScalarType> &lhs_value,
        const typename RhsOp::template value_type<ScalarType> &rhs_value) const
      {
        return internal::pow_impl(lhs_value, rhs_value);
      }
    };


#undef DEAL_II_BINARY_OP_COMMON_IMPL

  } // namespace Operators
} // namespace WeakForms



/* ===================== Define operator overloads ===================== */
// See https://stackoverflow.com/a/12782697 for using multiple parameter packs



#define DEAL_II_BINARY_OP_OF_BINARY_OP(operator_name, binary_op_code)        \
  template <typename LhsOp1,                                                 \
            typename LhsOp2,                                                 \
            enum WeakForms::Operators::BinaryOpCodes LhsOpCode,              \
            typename RhsOp1,                                                 \
            typename RhsOp2,                                                 \
            enum WeakForms::Operators::BinaryOpCodes RhsOpCode>              \
  WeakForms::Operators::BinaryOp<                                            \
    WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode>,               \
    WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode>,               \
    WeakForms::Operators::BinaryOpCodes::binary_op_code>                     \
  operator_name(                                                             \
    const WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode> &lhs_op, \
    const WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode> &rhs_op) \
  {                                                                          \
    using namespace WeakForms;                                               \
    using namespace WeakForms::Operators;                                    \
                                                                             \
    using LhsOpType = BinaryOp<LhsOp1, LhsOp2, LhsOpCode>;                   \
    using RhsOpType = BinaryOp<RhsOp1, RhsOp2, RhsOpCode>;                   \
    using OpType =                                                           \
      BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::binary_op_code>;         \
                                                                             \
    return OpType(lhs_op, rhs_op);                                           \
  }

DEAL_II_BINARY_OP_OF_BINARY_OP(operator+, add)
DEAL_II_BINARY_OP_OF_BINARY_OP(operator-, subtract)
DEAL_II_BINARY_OP_OF_BINARY_OP(operator*, multiply)
DEAL_II_BINARY_OP_OF_BINARY_OP(operator/, divide)
DEAL_II_BINARY_OP_OF_BINARY_OP(pow, power)

#undef DEAL_II_BINARY_OP_OF_BINARY_OP



/* ==================== Specialization of type traits ==================== */



#ifndef DOXYGEN


namespace WeakForms
{
  // template <typename... Args>
  // struct is_binary_op<Operators::BinaryOp<Args...>> : std::true_type
  // {};

  template <typename LhsOp, typename RhsOp, typename UnderlyingType>
  struct is_binary_op<
    Operators::
      BinaryOp<LhsOp, RhsOp, Operators::BinaryOpCodes::add, UnderlyingType>>
    : std::true_type
  {};

  template <typename LhsOp, typename RhsOp, typename UnderlyingType>
  struct is_binary_op<Operators::BinaryOp<LhsOp,
                                          RhsOp,
                                          Operators::BinaryOpCodes::subtract,
                                          UnderlyingType>> : std::true_type
  {};

  template <typename LhsOp, typename RhsOp, typename UnderlyingType>
  struct is_binary_op<Operators::BinaryOp<LhsOp,
                                          RhsOp,
                                          Operators::BinaryOpCodes::multiply,
                                          UnderlyingType>> : std::true_type
  {};

  template <typename LhsOp, typename RhsOp, typename UnderlyingType>
  struct is_binary_op<
    Operators::
      BinaryOp<LhsOp, RhsOp, Operators::BinaryOpCodes::divide, UnderlyingType>>
    : std::true_type
  {};

  template <typename LhsOp, typename RhsOp, typename UnderlyingType>
  struct is_binary_op<
    Operators::
      BinaryOp<LhsOp, RhsOp, Operators::BinaryOpCodes::power, UnderlyingType>>
    : std::true_type
  {};

  template <typename LhsOp,
            typename RhsOp,
            enum Operators::BinaryOpCodes OpCode>
  struct is_field_solution_op<Operators::BinaryOp<LhsOp, RhsOp, OpCode>>
    : std::conditional<
        (is_field_solution_op<LhsOp>::value &&
         !is_or_has_test_function_or_trial_solution_op<RhsOp>::value) ||
          (is_field_solution_op<RhsOp>::value &&
           !is_or_has_test_function_or_trial_solution_op<LhsOp>::value),
        std::true_type,
        std::false_type>::type
  {};

  //   template <int dim, int spacedim, enum Operators::SymbolicOpCodes OpCode>
  //   struct is_test_function<
  //     Operators::BinaryOp<TestFunction<dim, spacedim>, OpCode>> :
  //     std::true_type
  //   {};

  //   template <int dim, int spacedim, enum Operators::SymbolicOpCodes OpCode>
  //   struct is_trial_solution<
  //     Operators::BinaryOp<TrialSolution<dim, spacedim>, OpCode>> :
  //     std::true_type
  //   {};

  //   template <int dim, int spacedim, enum Operators::SymbolicOpCodes OpCode>
  //   struct is_field_solution<
  //     Operators::BinaryOp<FieldSolution<dim, spacedim>, OpCode>> :
  //     std::true_type
  //   {};

} // namespace WeakForms


#endif // DOXYGEN


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_binary_operators_h
