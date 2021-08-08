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

#ifndef dealii_weakforms_unary_operators_h
#define dealii_weakforms_unary_operators_h

#include <deal.II/base/config.h>

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>

#include <deal.II/meshworker/scratch_data.h>

#include <weak_forms/config.h>
#include <weak_forms/numbers.h>
#include <weak_forms/operator_evaluators.h>
#include <weak_forms/operator_utilities.h>
#include <weak_forms/solution_storage.h>
#include <weak_forms/symbolic_decorations.h>
#include <weak_forms/symbolic_operators.h>
#include <weak_forms/type_traits.h>
#include <weak_forms/types.h>

#include <type_traits>



WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
  namespace Operators
  {
    enum class UnaryOpCodes
    {
      /*
       * Negate the current operand.
       */
      negate,

      // --- Scalar operations ---
      /**
       * Compute the square root of a scalar
       */
      square_root,
      // exp
      // log
      // sin
      // cos
      // tan
      // asin
      // acos
      // atan
      // sinh
      // cosh
      // tanh
      // asinh
      // acosh
      // atanh
      // erf
      // erfc
      // abs

      // --- Tensor operations ---
      /**
       * Form the determinant of a tensor
       */
      determinant,
      /**
       * Form the inverse of a tensor
       */
      invert,
      /**
       * Form the transpose of a tensor
       */
      transpose,
      /**
       * Symmetrize a tensor
       */
      symmetrize,

      // --- Interface operations ---
      /**
       * Jump of an operand across an interface
       */
      // jump,
      /**
       * Average of an operand across an interface
       */
      // average,
    };



    /**
     * Exception denoting that a class requires some specialization
     * in order to be used.
     */
    DeclExceptionMsg(
      ExcRequiresUnaryOperatorSpecialization,
      "This function is called in a class that is expected to be specialized "
      "for unary operations. All unary operators should be specialized, "
      "with a structure matching that of the exemplar class.");


    /**
     * Exception denoting that a unary operation has not been defined.
     */
    DeclException1(ExcUnaryOperatorNotDefined,
                   enum UnaryOpCodes,
                   << "The unary operator with code " +
                          dealii::Utilities::to_string(static_cast<int>(arg1)) +
                          " has not been defined.");


    /**
     * @tparam Op
     * @tparam OpCode
     * @tparam UnderlyingType Underlying number type (double, std::complex<double>, etc.).
     * This is necessary because some specializations of the class do not use
     * the number type in the specialization itself, but they may rely on the
     * type in their definitions (e.g. class members).
     * @tparam Args A dumping ground for any other arguments that may be necessary
     * to form a concrete class instance.
     */
    template <typename Op,
              enum UnaryOpCodes OpCode,
              typename UnderlyingType = void,
              typename... Args>
    class UnaryOp
    {
    public:
      explicit UnaryOp(const Op &operand)
        : operand(operand)
      {
        AssertThrow(false, ExcRequiresUnaryOperatorSpecialization());
      }

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        (void)decorator;
        AssertThrow(false, ExcRequiresUnaryOperatorSpecialization());
        return "";
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        (void)decorator;
        AssertThrow(false, ExcRequiresUnaryOperatorSpecialization());
        return "";
      }

    private:
      const Op operand;
    }; // class UnaryOp

  } // namespace Operators


} // namespace WeakForms



/* ================= Specialization of unary operators: ================= */
/* ================== Integrands of symbolic integrals ================== */



namespace WeakForms
{
  namespace Operators
  {
    // A little bit of CRTP, with a workaround to deal with templates
    // in the derived class.
    // See https://stackoverflow.com/a/45801893
    template <typename Derived>
    struct UnaryOpTypeTraits;

    namespace internal
    {
      template <typename OpType, typename U = void>
      struct unary_op_test_trial_traits;

      template <typename OpType>
      struct unary_op_test_trial_traits<
        OpType,
        typename std::enable_if<
          !is_or_has_test_function_or_trial_solution_op<OpType>::value>::type>
      {
        template <typename T>
        using return_type = std::vector<T>;

        template <typename T, std::size_t width>
        using vectorized_return_type =
          typename numbers::VectorizedValue<T>::template type<width>;
      };

      template <typename OpType>
      struct unary_op_test_trial_traits<
        OpType,
        typename std::enable_if<
          is_or_has_test_function_or_trial_solution_op<OpType>::value>::type>
      {
        template <typename T>
        using return_type = std::vector<std::vector<T>>;

        template <typename T, std::size_t width>
        using vectorized_return_type = AlignedVector<
          typename numbers::VectorizedValue<T>::template type<width>>;
      };

    } // namespace internal



/**
 * A macro to implement the common parts of a unary op type trait class.
 *
 * What remains to be defined are:
 * - static const enum UnaryOpCodes op_code
 * - static const int rank
 * - template <typename ScalarType> using value_type = ...;
 *
 * @note This should used at the very end of the class definition, as
 * the @p return_type relies on the @p value_type to be defined.
 */
#define DEAL_II_UNARY_OP_TYPE_TRAITS_COMMON_IMPL(Op)                 \
  /**                                                                \
   *                                                                 \
   */                                                                \
  using OpType = Op;                                                 \
                                                                     \
  /**                                                                \
   * Dimension in which this object operates.                        \
   */                                                                \
  static const unsigned int dimension = Op::dimension;               \
                                                                     \
  /**                                                                \
   * Dimension of the space in which this object operates.           \
   */                                                                \
  static const unsigned int space_dimension = Op::space_dimension;   \
                                                                     \
  /**                                                                \
   *                                                                 \
   */                                                                \
  template <typename ScalarType>                                     \
  using return_type = typename internal::unary_op_test_trial_traits< \
    OpType>::template return_type<value_type<ScalarType>>;           \
                                                                     \
  template <typename ScalarType, std::size_t width>                  \
  using vectorized_value_type = typename numbers::VectorizedValue<   \
    value_type<ScalarType>>::template type<width>;                   \
                                                                     \
  template <typename ScalarType, std::size_t width>                  \
  using vectorized_return_type =                                     \
    typename internal::unary_op_test_trial_traits<                   \
      OpType>::template vectorized_return_type<value_type<ScalarType>, width>;



    template <typename Op>
    struct UnaryOpTypeTraits<UnaryOp<Op, UnaryOpCodes::negate>>
    {
      static const enum UnaryOpCodes op_code = UnaryOpCodes::negate;
      static const int               rank    = Op::rank;

      template <typename ScalarType>
      using value_type =
        decltype(-std::declval<typename Op::template value_type<ScalarType>>());

      // Implement the common part of the class
      DEAL_II_UNARY_OP_TYPE_TRAITS_COMMON_IMPL(Op)
    };



    template <typename Op>
    struct UnaryOpTypeTraits<UnaryOp<Op, UnaryOpCodes::square_root>>
    {
      static const enum UnaryOpCodes op_code = UnaryOpCodes::square_root;

      static_assert(Op::rank == 0,
                    "Invalid operator rank"); // Can only act on scalars
      static const int rank = 0;              // Square root is scalar valued

      template <typename ScalarType>
      using value_type = decltype(
        sqrt(std::declval<typename Op::template value_type<ScalarType>>()));

      // Implement the common part of the class
      DEAL_II_UNARY_OP_TYPE_TRAITS_COMMON_IMPL(Op)
    };



    template <typename Op>
    struct UnaryOpTypeTraits<UnaryOp<Op, UnaryOpCodes::determinant>>
    {
      static const enum UnaryOpCodes op_code = UnaryOpCodes::determinant;
      static_assert(Op::rank == 2,
                    "Invalid operator rank"); // Can only act on rank-2 tensors
      static const int rank = 0;              // Determinant is scalar valued

      template <typename ScalarType>
      using value_type = decltype(determinant(
        std::declval<typename Op::template value_type<ScalarType>>()));

      // Implement the common part of the class
      DEAL_II_UNARY_OP_TYPE_TRAITS_COMMON_IMPL(Op)
    };



    template <typename Op>
    struct UnaryOpTypeTraits<UnaryOp<Op, UnaryOpCodes::invert>>
    {
      static const enum UnaryOpCodes op_code = UnaryOpCodes::invert;

      static_assert(Op::rank == 2 || Op::rank == 4, "Invalid rank");
      static const int rank = Op::rank;

      template <typename ScalarType>
      using value_type = decltype(
        invert(std::declval<typename Op::template value_type<ScalarType>>()));

      // Implement the common part of the class
      DEAL_II_UNARY_OP_TYPE_TRAITS_COMMON_IMPL(Op)
    };



    template <typename Op>
    struct UnaryOpTypeTraits<UnaryOp<Op, UnaryOpCodes::transpose>>
    {
      static const enum UnaryOpCodes op_code = UnaryOpCodes::transpose;

      static_assert(Op::rank == 2 || Op::rank == 4, "Invalid operator rank");
      static const int rank = Op::rank;

      template <typename ScalarType>
      using value_type = decltype(transpose(
        std::declval<typename Op::template value_type<ScalarType>>()));

      // Implement the common part of the class
      DEAL_II_UNARY_OP_TYPE_TRAITS_COMMON_IMPL(Op)
    };



    template <typename Op>
    struct UnaryOpTypeTraits<UnaryOp<Op, UnaryOpCodes::symmetrize>>
    {
      static const enum UnaryOpCodes op_code = UnaryOpCodes::symmetrize;

      static_assert(Op::rank == 2 || Op::rank == 4, "Invalid operator rank");
      static const int rank = Op::rank;

      template <typename ScalarType>
      using value_type = decltype(symmetrize(
        std::declval<typename Op::template value_type<ScalarType>>()));

      // Implement the common part of the class
      DEAL_II_UNARY_OP_TYPE_TRAITS_COMMON_IMPL(Op)
    };


#undef DEAL_II_UNARY_OP_TYPE_TRAITS_COMMON_IMPL


    template <typename Derived>
    class UnaryOpBase
    {
      using Traits = UnaryOpTypeTraits<Derived>;

    public:
      using OpType = typename Traits::OpType;

      static const int               dimension       = Traits::dimension;
      static const int               space_dimension = Traits::space_dimension;
      static const int               rank            = Traits::rank;
      static const enum UnaryOpCodes op_code         = Traits::op_code;

      template <typename ScalarType>
      using value_type = typename Traits::template value_type<ScalarType>;

      template <typename ScalarType>
      using return_type = typename Traits::template return_type<ScalarType>;

      template <typename ScalarType, std::size_t width>
      using vectorized_value_type =
        typename Traits::template vectorized_value_type<ScalarType, width>;

      template <typename ScalarType, std::size_t width>
      using vectorized_return_type =
        typename Traits::template vectorized_return_type<ScalarType, width>;

      UnaryOpBase(const Derived &derived)
        : derived(derived)
      {}

      UpdateFlags
      get_update_flags() const
      {
        return derived.get_operand().get_update_flags();
      }

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename OpType::template value_type<ScalarType> &value) const
      {
        return derived.template operator()<ScalarType>(value);
      }

      // ----- VECTORIZATION -----

      template <typename ScalarType, std::size_t width>
      vectorized_value_type<ScalarType, width>
      operator()(
        const typename OpType::template vectorized_value_type<ScalarType, width>
          &value) const
      {
        return derived.template operator()<ScalarType, width>(value);
      }


      // ---- Operators NOT for test functions / trial solutions ---
      // So these are restricted to symbolic ops, functors (standard and
      // cache) and field solutions as leaf operations.

      template <typename ScalarType>
      auto
      operator()(const typename OpType::template return_type<ScalarType> &value)
        const -> typename std::enable_if<
          !is_or_has_test_function_or_trial_solution_op<OpType>::value,
          return_type<ScalarType>>::type
      {
        return_type<ScalarType> out;
        const unsigned int      n_q_points = value.size();
        out.reserve(n_q_points);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          out.emplace_back(
            derived.template operator()<ScalarType>(value[q_point]));

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
      template <typename ScalarType, typename FEValuesType>
      auto
      operator()(const FEValuesType &fe_values) const ->
        typename std::enable_if<
          internal::is_fe_values_type<FEValuesType>::value &&
            !is_or_has_test_function_or_trial_solution_op<OpType>::value &&
            !is_or_has_evaluated_with_scratch_data<OpType>::value,
          return_type<ScalarType>>::type
      {
        return internal::UnaryOpEvaluator<OpType>::template apply<ScalarType>(
          *this, derived.get_operand(), fe_values);
      }

      template <typename ScalarType,
                typename FEValuesType,
                int dim,
                int spacedim>
      auto
      operator()(const FEValuesType &                    fe_values,
                 MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                 const std::vector<std::string> &solution_names) const ->
        typename std::enable_if<
          internal::is_fe_values_type<FEValuesType>::value &&
            !is_or_has_test_function_or_trial_solution_op<OpType>::value &&
            is_or_has_evaluated_with_scratch_data<OpType>::value,
          return_type<ScalarType>>::type
      {
        return internal::UnaryOpEvaluator<OpType>::template apply<ScalarType>(
          *this,
          derived.get_operand(),
          fe_values,
          scratch_data,
          solution_names);
      }

      // ----- VECTORIZATION -----

      // template <typename ScalarType, std::size_t width>
      // auto
      // operator()(
      //   const typename OpType::template vectorized_return_type<ScalarType,
      //                                                          width> &value)
      //   const -> typename std::enable_if<
      //     !is_or_has_test_function_or_trial_solution_op<OpType>::value,
      //     vectorized_return_type<ScalarType, width>>::type
      // {
      //   return derived.template operator()<ScalarType, width>(value);
      // }

      template <typename ScalarType, std::size_t width, typename FEValuesType>
      auto
      operator()(const FEValuesType &                fe_values,
                 const types::vectorized_qp_range_t &q_point_range) const ->
        typename std::enable_if<
          internal::is_fe_values_type<FEValuesType>::value &&
            !is_or_has_test_function_or_trial_solution_op<OpType>::value &&
            !is_or_has_evaluated_with_scratch_data<OpType>::value,
          vectorized_return_type<ScalarType, width>>::type
      {
        return internal::UnaryOpEvaluator<OpType>::template apply<ScalarType,
                                                                  width>(
          *this, derived.get_operand(), fe_values, q_point_range);
      }

      template <typename ScalarType,
                std::size_t width,
                typename FEValuesType,
                int dim,
                int spacedim>
      auto
      operator()(const FEValuesType &                    fe_values,
                 MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                 const std::vector<std::string> &        solution_names,
                 const types::vectorized_qp_range_t &    q_point_range) const ->
        typename std::enable_if<
          internal::is_fe_values_type<FEValuesType>::value &&
            !is_or_has_test_function_or_trial_solution_op<OpType>::value &&
            is_or_has_evaluated_with_scratch_data<OpType>::value,
          vectorized_return_type<ScalarType, width>>::type
      {
        return internal::UnaryOpEvaluator<
          OpType>::template apply<ScalarType, width>(*this,
                                                     derived.get_operand(),
                                                     fe_values,
                                                     scratch_data,
                                                     solution_names,
                                                     q_point_range);
      }


      // ---- Operators for test functions / trial solutions ---
      // So these are for when a test function or trial solution is one or more
      // of the leaf operations. The other leaves may or may not be
      // symbolic ops, functors (standard and cache) and field solutions.

      template <typename ScalarType>
      auto
      operator()(const typename OpType::template return_type<ScalarType> &value)
        const -> typename std::enable_if<
          is_or_has_test_function_or_trial_solution_op<OpType>::value,
          return_type<ScalarType>>::type
      {
        Assert(value[0].size() > 0, ExcMessage("Uninitialized q-point entry"));
        const unsigned int n_dofs     = value.size();
        const unsigned int n_q_points = value[0].size();

        return_type<ScalarType> out(n_dofs);
        for (unsigned int dof_index = 0; dof_index < n_dofs; ++dof_index)
          {
            Assert(value[dof_index].size() == n_q_points,
                   ExcDimensionMismatch(value[dof_index].size(), n_q_points));

            out[dof_index].reserve(n_q_points);
          }

        for (unsigned int dof_index = 0; dof_index < n_dofs; ++dof_index)
          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            out[dof_index].emplace_back(derived.template operator()<ScalarType>(
              value[dof_index][q_point]));

        return out;
      }

      template <typename ScalarType,
                typename FEValuesTypeDoFs,
                typename FEValuesTypeOp>
      auto
      operator()(const FEValuesTypeDoFs &fe_values_dofs,
                 const FEValuesTypeOp &  fe_values_op) const ->
        typename std::enable_if<
          (internal::is_fe_values_type<FEValuesTypeDoFs>::value &&
           internal::is_fe_values_type<FEValuesTypeOp>::value) &&
            is_or_has_test_function_or_trial_solution_op<OpType>::value &&
            !is_or_has_evaluated_with_scratch_data<OpType>::value,
          return_type<ScalarType>>::type
      {
        return internal::UnaryOpEvaluator<OpType>::template apply<ScalarType>(
          *this, derived.get_operand(), fe_values_dofs, fe_values_op);
      }

      template <typename ScalarType,
                typename FEValuesTypeDoFs,
                typename FEValuesTypeOp,
                int dim,
                int spacedim>
      auto
      operator()(const FEValuesTypeDoFs &                fe_values_dofs,
                 const FEValuesTypeOp &                  fe_values_op,
                 MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                 const std::vector<std::string> &solution_names) const ->
        typename std::enable_if<
          (internal::is_fe_values_type<FEValuesTypeDoFs>::value &&
           internal::is_fe_values_type<FEValuesTypeOp>::value) &&
            is_or_has_test_function_or_trial_solution_op<OpType>::value &&
            is_or_has_evaluated_with_scratch_data<OpType>::value,
          return_type<ScalarType>>::type
      {
        return internal::UnaryOpEvaluator<OpType>::template apply<ScalarType>(
          *this,
          derived.get_operand(),
          fe_values_dofs,
          fe_values_op,
          scratch_data,
          solution_names);
      }

      // ----- VECTORIZATION -----

      template <typename ScalarType, std::size_t width>
      auto
      operator()(
        const typename OpType::template vectorized_return_type<ScalarType,
                                                               width> &value)
        const -> typename std::enable_if<
          is_or_has_test_function_or_trial_solution_op<OpType>::value,
          vectorized_return_type<ScalarType, width>>::type
      {
        const unsigned int                        n_dofs = value.size();
        vectorized_return_type<ScalarType, width> out(n_dofs);

        for (unsigned int dof_index = 0; dof_index < n_dofs; ++dof_index)
          {
            out[dof_index] =
              this->template operator()<ScalarType, width>(value[dof_index]);
          }

        return out;
      }

      template <typename ScalarType,
                std::size_t width,
                typename FEValuesTypeDoFs,
                typename FEValuesTypeOp>
      auto
      operator()(const FEValuesTypeDoFs &            fe_values_dofs,
                 const FEValuesTypeOp &              fe_values_op,
                 const types::vectorized_qp_range_t &q_point_range) const ->
        typename std::enable_if<
          (internal::is_fe_values_type<FEValuesTypeDoFs>::value &&
           internal::is_fe_values_type<FEValuesTypeOp>::value) &&
            is_or_has_test_function_or_trial_solution_op<OpType>::value &&
            !is_or_has_evaluated_with_scratch_data<OpType>::value,
          vectorized_return_type<ScalarType, width>>::type
      {
        return internal::UnaryOpEvaluator<
          OpType>::template apply<ScalarType, width>(*this,
                                                     derived.get_operand(),
                                                     fe_values_dofs,
                                                     fe_values_op,
                                                     q_point_range);
      }

      template <typename ScalarType,
                std::size_t width,
                typename FEValuesTypeDoFs,
                typename FEValuesTypeOp,
                int dim,
                int spacedim>
      auto
      operator()(const FEValuesTypeDoFs &                fe_values_dofs,
                 const FEValuesTypeOp &                  fe_values_op,
                 MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                 const std::vector<std::string> &        solution_names,
                 const types::vectorized_qp_range_t &    q_point_range) const ->
        typename std::enable_if<
          (internal::is_fe_values_type<FEValuesTypeDoFs>::value &&
           internal::is_fe_values_type<FEValuesTypeOp>::value) &&
            is_or_has_test_function_or_trial_solution_op<OpType>::value &&
            is_or_has_evaluated_with_scratch_data<OpType>::value,
          vectorized_return_type<ScalarType, width>>::type
      {
        return internal::UnaryOpEvaluator<
          OpType>::template apply<ScalarType, width>(*this,
                                                     derived.get_operand(),
                                                     fe_values_dofs,
                                                     fe_values_op,
                                                     scratch_data,
                                                     solution_names,
                                                     q_point_range);
      }

    private:
      const Derived &derived;
    };



/**
 * A macro to implement the common parts of a unary op class.
 * It is expected that the unary op derives from a
 * UnaryOpBase<UnaryOp<Op, UnaryOpCode>> .
 *
 * What remains to be implemented are the public functions:
 *  - as_ascii()
 *  - as_latex()
 *  - operator()
 *
 * @note It is intended that this should used immediately after class
 * definition is opened.
 */
#define DEAL_II_UNARY_OP_COMMON_IMPL(Op, UnaryOpCode)                    \
private:                                                                 \
  using Base   = UnaryOpBase<UnaryOp<Op, UnaryOpCode>>;                  \
  using Traits = UnaryOpTypeTraits<UnaryOp<Op, UnaryOpCode>>;            \
                                                                         \
public:                                                                  \
  using OpType = typename Traits::OpType;                                \
                                                                         \
  template <typename ScalarType>                                         \
  using value_type = typename Traits::template value_type<ScalarType>;   \
                                                                         \
  template <typename ScalarType>                                         \
  using return_type = typename Traits::template return_type<ScalarType>; \
                                                                         \
  template <typename ScalarType, std::size_t width>                      \
  using vectorized_value_type =                                          \
    typename Traits::template vectorized_value_type<ScalarType, width>;  \
                                                                         \
  template <typename ScalarType, std::size_t width>                      \
  using vectorized_return_type =                                         \
    typename Traits::template vectorized_return_type<ScalarType, width>; \
                                                                         \
  using Base::dimension;                                                 \
  using Base::op_code;                                                   \
  using Base::rank;                                                      \
  using Base::space_dimension;                                           \
  using Base::get_update_flags;                                          \
  using Base::operator();                                                \
                                                                         \
  explicit UnaryOp(const Op &operand)                                    \
    : Base(*this)                                                        \
    , operand(operand)                                                   \
  {}                                                                     \
                                                                         \
  /**                                                                    \
   * Required to support operands that access objects with a limited     \
   * lifetime, e.g. ScalarFunctionFunctor, TensorFunctionFunctor         \
   */                                                                    \
  UnaryOp(const UnaryOp &rhs)                                            \
    : Base(*this)                                                        \
    , operand(rhs.operand)                                               \
  {}                                                                     \
                                                                         \
  const Op &get_operand() const                                          \
  {                                                                      \
    return operand;                                                      \
  }                                                                      \
                                                                         \
private:                                                                 \
  const Op operand;



    /* ---------------------------- General ---------------------------- */

    /**
     * Negation operator for integrands of symbolic integrals
     */
    template <typename Op>
    class UnaryOp<Op,
                  UnaryOpCodes::negate,
                  typename std::enable_if<!is_integral_op<Op>::value>::type>
      : public UnaryOpBase<UnaryOp<Op, UnaryOpCodes::negate>>
    {
      DEAL_II_UNARY_OP_COMMON_IMPL(Op, UnaryOpCodes::negate)

    public:
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

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename Op::template value_type<ScalarType> &value) const
      {
        return -value;
      }

      template <typename ScalarType, std::size_t width>
      vectorized_value_type<ScalarType, width>
      operator()(
        const typename Op::template vectorized_value_type<ScalarType, width>
          &value) const
      {
        return -value;
      }
    };


    /* ------------------------- Scalar operations ------------------------- */


    /**
     * Square root operator for integrands of symbolic integrals
     *
     * @note Not available for test functions and trial solutions.
     */
    template <typename Op>
    class UnaryOp<Op,
                  UnaryOpCodes::square_root,
                  typename std::enable_if<
                    !is_or_has_test_function_or_trial_solution_op<Op>::value &&
                    !is_integral_op<Op>::value>::type>
      : public UnaryOpBase<UnaryOp<Op, UnaryOpCodes::square_root>>
    {
      static_assert(
        !is_or_has_test_function_or_trial_solution_op<Op>::value,
        "The square root operation is not permitted for test functions or trial solutions.");

      DEAL_II_UNARY_OP_COMMON_IMPL(Op, UnaryOpCodes::square_root)

    public:
      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return decorator.mathop_symbolic_op_operand_as_ascii("sqrt", operand);
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return decorator.mathop_symbolic_op_operand_as_latex(
          Utilities::LaTeX::decorate_latex_op("sqrt"), operand);
      }

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename Op::template value_type<ScalarType> &value) const
      {
        using namespace std;
        return sqrt(value);
      }

      template <typename ScalarType, std::size_t width>
      vectorized_value_type<ScalarType, width>
      operator()(
        const typename Op::template vectorized_value_type<ScalarType, width>
          &value) const
      {
        using namespace std;
        return sqrt(value);
      }
    };


    /* ------------------------- Tensor operations ------------------------- */

    /**
     * Determinant operator for integrands of symbolic integrals
     *
     * @note Not available for test functions and trial solutions.
     */
    template <typename Op>
    class UnaryOp<Op,
                  UnaryOpCodes::determinant,
                  typename std::enable_if<
                    !is_or_has_test_function_or_trial_solution_op<Op>::value &&
                    !is_integral_op<Op>::value>::type>
      : public UnaryOpBase<UnaryOp<Op, UnaryOpCodes::determinant>>
    {
      static_assert(
        !is_or_has_test_function_or_trial_solution_op<Op>::value,
        "The determinant operation is not permitted for test functions or trial solutions.");

      DEAL_II_UNARY_OP_COMMON_IMPL(Op, UnaryOpCodes::determinant)

    public:
      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return decorator.prefixed_parenthesized_symbolic_op_operand_as_ascii(
          "det", operand);
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return decorator.prefixed_parenthesized_symbolic_op_operand_as_latex(
          Utilities::LaTeX::decorate_text("det"), operand);
      }

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename Op::template value_type<ScalarType> &value) const
      {
        return determinant(value);
      }

      template <typename ScalarType, std::size_t width>
      vectorized_value_type<ScalarType, width>
      operator()(
        const typename Op::template vectorized_value_type<ScalarType, width>
          &value) const
      {
        return determinant(value);
      }
    };



    /**
     * Inverse operator for integrands of symbolic integrals
     *
     * @note Not available for test functions and trial solutions.
     */
    template <typename Op>
    class UnaryOp<Op,
                  UnaryOpCodes::invert,
                  typename std::enable_if<!is_integral_op<Op>::value>::type>
      : public UnaryOpBase<UnaryOp<Op, UnaryOpCodes::invert>>
    {
      static_assert(
        !is_or_has_test_function_or_trial_solution_op<Op>::value,
        "The inverse operation is not permitted for test functions or trial solutions.");

      DEAL_II_UNARY_OP_COMMON_IMPL(Op, UnaryOpCodes::invert)

    public:
      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return decorator.prefixed_parenthesized_symbolic_op_operand_as_ascii(
          "inv", operand);
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return decorator
          .suffixed_braced_superscript_symbolic_op_operand_as_latex(operand,
                                                                    "-1");
      }

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename Op::template value_type<ScalarType> &value) const
      {
        return invert(value);
      }

      template <typename ScalarType, std::size_t width>
      vectorized_value_type<ScalarType, width>
      operator()(
        const typename Op::template vectorized_value_type<ScalarType, width>
          &value) const
      {
        return invert(value);
      }
    };



    /**
     * Transpose operator for integrands of symbolic integrals
     */
    template <typename Op>
    class UnaryOp<Op,
                  UnaryOpCodes::transpose,
                  typename std::enable_if<!is_integral_op<Op>::value>::type>
      : public UnaryOpBase<UnaryOp<Op, UnaryOpCodes::transpose>>
    {
      DEAL_II_UNARY_OP_COMMON_IMPL(Op, UnaryOpCodes::transpose)

    public:
      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return decorator.prefixed_parenthesized_symbolic_op_operand_as_ascii(
          "trans", operand);
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return decorator
          .suffixed_braced_superscript_symbolic_op_operand_as_latex(operand,
                                                                    "T");
      }

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename Op::template value_type<ScalarType> &value) const
      {
        return transpose(value);
      }

      template <typename ScalarType, std::size_t width>
      vectorized_value_type<ScalarType, width>
      operator()(
        const typename Op::template vectorized_value_type<ScalarType, width>
          &value) const
      {
        return transpose(value);
      }
    };



    /**
     * Symmetrization operator for integrands of symbolic integrals
     */
    template <typename Op>
    class UnaryOp<Op,
                  UnaryOpCodes::symmetrize,
                  typename std::enable_if<!is_integral_op<Op>::value>::type>
      : public UnaryOpBase<UnaryOp<Op, UnaryOpCodes::symmetrize>>
    {
      DEAL_II_UNARY_OP_COMMON_IMPL(Op, UnaryOpCodes::symmetrize)

    public:
      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return decorator.prefixed_parenthesized_symbolic_op_operand_as_ascii(
          "symm", operand);
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return decorator
          .suffixed_braced_superscript_symbolic_op_operand_as_latex(operand,
                                                                    "S");
      }

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename Op::template value_type<ScalarType> &value) const
      {
        return symmetrize(value);
      }

      template <typename ScalarType, std::size_t width>
      vectorized_value_type<ScalarType, width>
      operator()(
        const typename Op::template vectorized_value_type<ScalarType, width>
          &value) const
      {
        return symmetrize(value);
      }
    };


    /* ------------------------ Tensor contractions ------------------------ */


#undef DEAL_II_UNARY_OP_COMMON_IMPL

  } // namespace Operators
} // namespace WeakForms



/* ===================== Define operator overloads ===================== */



#define DEAL_II_UNARY_OP_OF_UNARY_OP(operator_name, unary_op_code)         \
  template <typename Op, enum WeakForms::Operators::UnaryOpCodes OpCode>   \
  WeakForms::Operators::UnaryOp<                                           \
    WeakForms::Operators::UnaryOp<Op, OpCode>,                             \
    WeakForms::Operators::UnaryOpCodes::unary_op_code>                     \
  operator_name(const WeakForms::Operators::UnaryOp<Op, OpCode> &operand)  \
  {                                                                        \
    using namespace WeakForms;                                             \
    using namespace WeakForms::Operators;                                  \
                                                                           \
    using UnaryOpType = UnaryOp<Op, OpCode>;                               \
    using OpType      = UnaryOp<UnaryOpType, UnaryOpCodes::unary_op_code>; \
                                                                           \
    return OpType(operand);                                                \
  }

DEAL_II_UNARY_OP_OF_UNARY_OP(operator-, negate)
DEAL_II_UNARY_OP_OF_UNARY_OP(sqrt, square_root)
DEAL_II_UNARY_OP_OF_UNARY_OP(determinant, determinant)
DEAL_II_UNARY_OP_OF_UNARY_OP(invert, invert)
DEAL_II_UNARY_OP_OF_UNARY_OP(transpose, transpose)
DEAL_II_UNARY_OP_OF_UNARY_OP(symmetrize, symmetrize)

#undef DEAL_II_UNARY_OP_OF_UNARY_OP



/* ==================== Specialization of type traits ==================== */



#ifndef DOXYGEN


namespace WeakForms
{
  template <typename Op, typename UnderlyingType>
  struct is_unary_op<
    Operators::UnaryOp<Op, Operators::UnaryOpCodes::negate, UnderlyingType>>
    : std::true_type
  {};

  template <typename Op, typename UnderlyingType>
  struct is_unary_op<Operators::UnaryOp<Op,
                                        Operators::UnaryOpCodes::square_root,
                                        UnderlyingType>> : std::true_type
  {};

  template <typename Op, typename UnderlyingType>
  struct is_unary_op<Operators::UnaryOp<Op,
                                        Operators::UnaryOpCodes::determinant,
                                        UnderlyingType>> : std::true_type
  {};

  template <typename Op, typename UnderlyingType>
  struct is_unary_op<
    Operators::UnaryOp<Op, Operators::UnaryOpCodes::invert, UnderlyingType>>
    : std::true_type
  {};

  template <typename Op, typename UnderlyingType>
  struct is_unary_op<
    Operators::UnaryOp<Op, Operators::UnaryOpCodes::transpose, UnderlyingType>>
    : std::true_type
  {};

  template <typename Op, typename UnderlyingType>
  struct is_unary_op<
    Operators::UnaryOp<Op, Operators::UnaryOpCodes::symmetrize, UnderlyingType>>
    : std::true_type
  {};

} // namespace WeakForms


#endif // DOXYGEN



WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_unary_operators_h
