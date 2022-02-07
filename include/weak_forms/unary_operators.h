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
#include <weak_forms/solution_extraction_data.h>
#include <weak_forms/symbolic_decorations.h>
#include <weak_forms/symbolic_operators.h>
#include <weak_forms/template_constraints.h>
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
      /*
       * Normalise the value of the current operand
       */
      normalize,

      // --- Scalar operations ---
      /**
       * Compute the trigonometric sine of a scalar
       */
      sine,
      /**
       * Compute the trigonometric cosine of a scalar
       */
      cosine,
      /**
       * Compute the trigonometric tangent of a scalar
       */
      tangent,
      /**
       * Compute the exponential of a scalar
       */
      exponential,
      /**
       * Compute the logarithm of a scalar
       */
      logarithm,
      /**
       * Compute the square root of a scalar
       */
      square_root,
      /**
       * Compute the absolute value of a scalar
       */
      absolute_value,
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

      // --- Tensor operations ---
      /**
       * Form the determinant of a tensor or symmetric tensor
       */
      determinant,
      /**
       * Form the inverse of a tensor or symmetric tensor
       */
      invert,
      /**
       * Form the transpose of a tensor or symmetric tensor
       */
      transpose,
      /**
       * Form the trace of a tensor or symmetric tensor
       */
      trace,
      /**
       * Symmetrize a tensor
       */
      symmetrize,
      /**
       * Form the adjugate of a tensor
       */
      adjugate,
      /**
       * Form the cofactor of a tensor
       */
      cofactor,
      /**
       * Form the l1-norm of a tensor
       */
      l1_norm,
      /**
       * Form the l(infinity)-norm of a tensor
       */
      linfty_norm,

      // --- Symmetric tensor operations ---

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



      template <typename T>
      void
      switch_zero_for_unit_value_in_denominator(T &denominator)
      {
        (void)denominator;
      }


      template <typename T, std::size_t width>
      VectorizedArray<T, width>
      switch_zero_for_unit_value_in_denominator(
        VectorizedArray<T, width> &denominator)
      {
        DEAL_II_OPENMP_SIMD_PRAGMA
        for (unsigned int v = 0; v < width; v++)
          if (denominator[v] == dealii::internal::NumberType<T>::value(0.0))
            denominator[v] = dealii::internal::NumberType<T>::value(1.0);
      }


      template <typename ScalarType,
                typename = typename WeakForms::is_scalar_type<ScalarType>::type>
      ScalarType
      normalize(const ScalarType &value)
      {
        using namespace std;
        ScalarType norm = std::abs(value);
        if (norm == dealii::internal::NumberType<ScalarType>::value(0.0))
          return norm;

        switch_zero_for_unit_value_in_denominator(norm);
        return value / norm;
      }


      template <typename ScalarType,
                typename = typename WeakForms::is_scalar_type<ScalarType>::type>
      std::complex<ScalarType>
      normalize(const std::complex<ScalarType> &value)
      {
        std::complex<ScalarType> norm = std::abs(value);
        if (norm == dealii::internal::NumberType<ScalarType>::value(0.0))
          return norm;

        switch_zero_for_unit_value_in_denominator(norm);
        return value / norm;
      }


      template <int rank,
                int dim,
                typename ScalarType,
                typename = typename WeakForms::is_scalar_type<ScalarType>::type>
      Tensor<rank, dim, ScalarType>
      normalize(const Tensor<rank, dim, ScalarType> &value)
      {
        ScalarType norm = value.norm();
        if (norm == dealii::internal::NumberType<ScalarType>::value(0.0))
          return Tensor<rank, dim, ScalarType>();

        switch_zero_for_unit_value_in_denominator(norm);
        return value / norm;
      }


      template <int rank,
                int dim,
                typename ScalarType,
                typename = typename WeakForms::is_scalar_type<ScalarType>::type>
      SymmetricTensor<rank, dim, ScalarType>
      normalize(const SymmetricTensor<rank, dim, ScalarType> &value)
      {
        ScalarType norm = value.norm();
        if (norm == dealii::internal::NumberType<ScalarType>::value(0.0))
          return SymmetricTensor<rank, dim, ScalarType>();

        switch_zero_for_unit_value_in_denominator(norm);
        return value / norm;
      }


      template <typename T, std::size_t width>
      VectorizedArray<T, width>
      normalize(const VectorizedArray<T, width> &value)
      {
        VectorizedArray<T, width> out;

        DEAL_II_OPENMP_SIMD_PRAGMA
        for (unsigned int v = 0; v < width; v++)
          {
            if (value[v] != dealii::internal::NumberType<T>::value(0.0))
              out[v] = normalize(value[v]);
            else
              out[v] = dealii::internal::NumberType<T>::value(0.0);
          }

        return out;
      }

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
    struct UnaryOpTypeTraits<UnaryOp<Op, UnaryOpCodes::normalize>>
    {
      static const enum UnaryOpCodes op_code = UnaryOpCodes::normalize;
      static const int               rank    = Op::rank;

      template <typename ScalarType>
      using value_type = decltype(internal::normalize(
        std::declval<typename Op::template value_type<ScalarType>>()));

      // Implement the common part of the class
      DEAL_II_UNARY_OP_TYPE_TRAITS_COMMON_IMPL(Op)
    };



    template <typename Op>
    struct UnaryOpTypeTraits<UnaryOp<Op, UnaryOpCodes::sine>>
    {
      static const enum UnaryOpCodes op_code = UnaryOpCodes::sine;

      static_assert(Op::rank == 0,
                    "Invalid operator rank"); // Can only act on scalars
      static const int rank = 0;              // Result is scalar valued

      template <typename ScalarType>
      using value_type = decltype(
        sin(std::declval<typename Op::template value_type<ScalarType>>()));

      // Implement the common part of the class
      DEAL_II_UNARY_OP_TYPE_TRAITS_COMMON_IMPL(Op)
    };



    template <typename Op>
    struct UnaryOpTypeTraits<UnaryOp<Op, UnaryOpCodes::cosine>>
    {
      static const enum UnaryOpCodes op_code = UnaryOpCodes::cosine;

      static_assert(Op::rank == 0,
                    "Invalid operator rank"); // Can only act on scalars
      static const int rank = 0;              // Result is scalar valued

      template <typename ScalarType>
      using value_type = decltype(
        cos(std::declval<typename Op::template value_type<ScalarType>>()));

      // Implement the common part of the class
      DEAL_II_UNARY_OP_TYPE_TRAITS_COMMON_IMPL(Op)
    };



    template <typename Op>
    struct UnaryOpTypeTraits<UnaryOp<Op, UnaryOpCodes::tangent>>
    {
      static const enum UnaryOpCodes op_code = UnaryOpCodes::tangent;

      static_assert(Op::rank == 0,
                    "Invalid operator rank"); // Can only act on scalars
      static const int rank = 0;              // Result is scalar valued

      template <typename ScalarType>
      using value_type = decltype(
        tan(std::declval<typename Op::template value_type<ScalarType>>()));

      // Implement the common part of the class
      DEAL_II_UNARY_OP_TYPE_TRAITS_COMMON_IMPL(Op)
    };



    template <typename Op>
    struct UnaryOpTypeTraits<UnaryOp<Op, UnaryOpCodes::exponential>>
    {
      static const enum UnaryOpCodes op_code = UnaryOpCodes::exponential;

      static_assert(Op::rank == 0,
                    "Invalid operator rank"); // Can only act on scalars
      static const int rank = 0;              // Result is scalar valued

      template <typename ScalarType>
      using value_type = decltype(
        exp(std::declval<typename Op::template value_type<ScalarType>>()));

      // Implement the common part of the class
      DEAL_II_UNARY_OP_TYPE_TRAITS_COMMON_IMPL(Op)
    };



    template <typename Op>
    struct UnaryOpTypeTraits<UnaryOp<Op, UnaryOpCodes::logarithm>>
    {
      static const enum UnaryOpCodes op_code = UnaryOpCodes::logarithm;

      static_assert(Op::rank == 0,
                    "Invalid operator rank"); // Can only act on scalars
      static const int rank = 0;              // Result is scalar valued

      template <typename ScalarType>
      using value_type = decltype(
        log(std::declval<typename Op::template value_type<ScalarType>>()));

      // Implement the common part of the class
      DEAL_II_UNARY_OP_TYPE_TRAITS_COMMON_IMPL(Op)
    };



    template <typename Op>
    struct UnaryOpTypeTraits<UnaryOp<Op, UnaryOpCodes::square_root>>
    {
      static const enum UnaryOpCodes op_code = UnaryOpCodes::square_root;

      static_assert(Op::rank == 0,
                    "Invalid operator rank"); // Can only act on scalars
      static const int rank = 0;              // Result is scalar valued

      template <typename ScalarType>
      using value_type = decltype(
        sqrt(std::declval<typename Op::template value_type<ScalarType>>()));

      // Implement the common part of the class
      DEAL_II_UNARY_OP_TYPE_TRAITS_COMMON_IMPL(Op)
    };



    template <typename Op>
    struct UnaryOpTypeTraits<UnaryOp<Op, UnaryOpCodes::absolute_value>>
    {
      static const enum UnaryOpCodes op_code = UnaryOpCodes::absolute_value;

      static_assert(Op::rank == 0,
                    "Invalid operator rank"); // Can only act on scalars
      static const int rank = 0;              // Result is scalar valued

      template <typename ScalarType>
      using value_type = decltype(
        abs(std::declval<typename Op::template value_type<ScalarType>>()));

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



    namespace internal
    {
      template <
        typename T,
        typename = typename std::enable_if<
          std::is_same<T, typename EnableIfScalar<T>::type>::value>::type>
      auto
      invert_impl(const T &value)
      {
        return dealii::internal::NumberType<T>::value(1.0) / value;
      }

      template <int rank, int dim, typename NumberType>
      auto
      invert_impl(const Tensor<rank, dim, NumberType> &tensor)
      {
        return dealii::invert(tensor);
      }

      template <int rank, int dim, typename NumberType>
      auto
      invert_impl(const SymmetricTensor<rank, dim, NumberType> &symm_tensor)
      {
        return dealii::invert(symm_tensor);
      }
    } // namespace internal



    template <typename Op>
    struct UnaryOpTypeTraits<UnaryOp<Op, UnaryOpCodes::invert>>
    {
      static const enum UnaryOpCodes op_code = UnaryOpCodes::invert;

      static_assert(Op::rank == 0 || Op::rank == 2 || Op::rank == 4,
                    "Invalid rank");
      static const int rank = Op::rank;

      template <typename ScalarType>
      using value_type = decltype(internal::invert_impl(
        std::declval<typename Op::template value_type<ScalarType>>()));

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



    template <typename Op>
    struct UnaryOpTypeTraits<UnaryOp<Op, UnaryOpCodes::trace>>
    {
      static const enum UnaryOpCodes op_code = UnaryOpCodes::trace;

      static_assert(Op::rank == 2, "Invalid operator rank");
      static const int rank = Op::rank;

      template <typename ScalarType>
      using value_type = decltype(
        trace(std::declval<typename Op::template value_type<ScalarType>>()));

      // Implement the common part of the class
      DEAL_II_UNARY_OP_TYPE_TRAITS_COMMON_IMPL(Op)
    };



    template <typename Op>
    struct UnaryOpTypeTraits<UnaryOp<Op, UnaryOpCodes::adjugate>>
    {
      static const enum UnaryOpCodes op_code = UnaryOpCodes::adjugate;

      static_assert(Op::rank == 2, "Invalid operator rank");
      static const int rank = Op::rank;

      template <typename ScalarType>
      using value_type = decltype(
        adjugate(std::declval<typename Op::template value_type<ScalarType>>()));

      // Implement the common part of the class
      DEAL_II_UNARY_OP_TYPE_TRAITS_COMMON_IMPL(Op)
    };



    template <typename Op>
    struct UnaryOpTypeTraits<UnaryOp<Op, UnaryOpCodes::cofactor>>
    {
      static const enum UnaryOpCodes op_code = UnaryOpCodes::cofactor;

      static_assert(Op::rank == 2, "Invalid operator rank");
      static const int rank = Op::rank;

      template <typename ScalarType>
      using value_type = decltype(
        cofactor(std::declval<typename Op::template value_type<ScalarType>>()));

      // Implement the common part of the class
      DEAL_II_UNARY_OP_TYPE_TRAITS_COMMON_IMPL(Op)
    };



    template <typename Op>
    struct UnaryOpTypeTraits<UnaryOp<Op, UnaryOpCodes::l1_norm>>
    {
      static const enum UnaryOpCodes op_code = UnaryOpCodes::l1_norm;

      static_assert(Op::rank == 2, "Invalid operator rank");
      static const int rank = Op::rank;

      template <typename ScalarType>
      using value_type = decltype(
        l1_norm(std::declval<typename Op::template value_type<ScalarType>>()));

      // Implement the common part of the class
      DEAL_II_UNARY_OP_TYPE_TRAITS_COMMON_IMPL(Op)
    };



    template <typename Op>
    struct UnaryOpTypeTraits<UnaryOp<Op, UnaryOpCodes::linfty_norm>>
    {
      static const enum UnaryOpCodes op_code = UnaryOpCodes::linfty_norm;

      static_assert(Op::rank == 2, "Invalid operator rank");
      static const int rank = Op::rank;

      template <typename ScalarType>
      using value_type = decltype(linfty_norm(
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
                 const std::vector<SolutionExtractionData<dim, spacedim>>
                   &solution_extraction_data) const ->
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
          solution_extraction_data);
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
                 const std::vector<SolutionExtractionData<dim, spacedim>>
                   &                                 solution_extraction_data,
                 const types::vectorized_qp_range_t &q_point_range) const ->
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
                                                     solution_extraction_data,
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
                 const std::vector<SolutionExtractionData<dim, spacedim>>
                   &solution_extraction_data) const ->
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
          solution_extraction_data);
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
                 const std::vector<SolutionExtractionData<dim, spacedim>>
                   &                                 solution_extraction_data,
                 const types::vectorized_qp_range_t &q_point_range) const ->
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
                                                     solution_extraction_data,
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


    /**
     * Normalization operator for integrands of symbolic integrals
     */
    template <typename Op>
    class UnaryOp<Op,
                  UnaryOpCodes::normalize,
                  typename std::enable_if<!is_integral_op<Op>::value>::type>
      : public UnaryOpBase<UnaryOp<Op, UnaryOpCodes::normalize>>
    {
      DEAL_II_UNARY_OP_COMMON_IMPL(Op, UnaryOpCodes::normalize)

    public:
      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto op = operand.as_ascii(decorator);
        return op + "/|" + op + "|";
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const std::string lvert = Utilities::LaTeX::l_vert();
        const std::string rvert = Utilities::LaTeX::r_vert();
        const auto        op    = operand.as_latex(decorator);
        return "\\frac{" + op + "}{" + lvert + op + rvert + "}";
      }

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename Op::template value_type<ScalarType> &value) const
      {
        return internal::normalize(value);
      }

      template <typename ScalarType, std::size_t width>
      vectorized_value_type<ScalarType, width>
      operator()(
        const typename Op::template vectorized_value_type<ScalarType, width>
          &value) const
      {
        return internal::normalize(value);
      }
    };


    /* ------------------------- Scalar operations ------------------------- */

    /**
     * Trigonometric sine operator for integrands of symbolic integrals
     *
     * @note Not available for test functions and trial solutions.
     */
    template <typename Op>
    class UnaryOp<Op,
                  UnaryOpCodes::sine,
                  typename std::enable_if<
                    !is_or_has_test_function_or_trial_solution_op<Op>::value &&
                    !is_integral_op<Op>::value>::type>
      : public UnaryOpBase<UnaryOp<Op, UnaryOpCodes::sine>>
    {
      static_assert(
        !is_or_has_test_function_or_trial_solution_op<Op>::value,
        "The square root operation is not permitted for test functions or trial solutions.");

      DEAL_II_UNARY_OP_COMMON_IMPL(Op, UnaryOpCodes::sine)

    public:
      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return decorator.prefixed_parenthesized_symbolic_op_operand_as_ascii(
          "sin", operand);
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return decorator.prefixed_parenthesized_symbolic_op_operand_as_latex(
          Utilities::LaTeX::decorate_text("sin"), operand);
      }

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename Op::template value_type<ScalarType> &value) const
      {
        using namespace std;
        return sin(value);
      }

      template <typename ScalarType, std::size_t width>
      vectorized_value_type<ScalarType, width>
      operator()(
        const typename Op::template vectorized_value_type<ScalarType, width>
          &value) const
      {
        using namespace std;
        return sin(value);
      }
    };


    /**
     * Trigonometric cosine operator for integrands of symbolic integrals
     *
     * @note Not available for test functions and trial solutions.
     */
    template <typename Op>
    class UnaryOp<Op,
                  UnaryOpCodes::cosine,
                  typename std::enable_if<
                    !is_or_has_test_function_or_trial_solution_op<Op>::value &&
                    !is_integral_op<Op>::value>::type>
      : public UnaryOpBase<UnaryOp<Op, UnaryOpCodes::cosine>>
    {
      static_assert(
        !is_or_has_test_function_or_trial_solution_op<Op>::value,
        "The square root operation is not permitted for test functions or trial solutions.");

      DEAL_II_UNARY_OP_COMMON_IMPL(Op, UnaryOpCodes::cosine)

    public:
      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return decorator.prefixed_parenthesized_symbolic_op_operand_as_ascii(
          "cos", operand);
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return decorator.prefixed_parenthesized_symbolic_op_operand_as_latex(
          Utilities::LaTeX::decorate_text("cos"), operand);
      }

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename Op::template value_type<ScalarType> &value) const
      {
        using namespace std;
        return cos(value);
      }

      template <typename ScalarType, std::size_t width>
      vectorized_value_type<ScalarType, width>
      operator()(
        const typename Op::template vectorized_value_type<ScalarType, width>
          &value) const
      {
        using namespace std;
        return cos(value);
      }
    };


    /**
     * Trigonometric tangent operator for integrands of symbolic integrals
     *
     * @note Not available for test functions and trial solutions.
     */
    template <typename Op>
    class UnaryOp<Op,
                  UnaryOpCodes::tangent,
                  typename std::enable_if<
                    !is_or_has_test_function_or_trial_solution_op<Op>::value &&
                    !is_integral_op<Op>::value>::type>
      : public UnaryOpBase<UnaryOp<Op, UnaryOpCodes::tangent>>
    {
      static_assert(
        !is_or_has_test_function_or_trial_solution_op<Op>::value,
        "The square root operation is not permitted for test functions or trial solutions.");

      DEAL_II_UNARY_OP_COMMON_IMPL(Op, UnaryOpCodes::tangent)

    public:
      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return decorator.prefixed_parenthesized_symbolic_op_operand_as_ascii(
          "tan", operand);
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return decorator.prefixed_parenthesized_symbolic_op_operand_as_latex(
          Utilities::LaTeX::decorate_text("tan"), operand);
      }

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename Op::template value_type<ScalarType> &value) const
      {
        using namespace std;
        return tan(value);
      }

      template <typename ScalarType, std::size_t width>
      vectorized_value_type<ScalarType, width>
      operator()(
        const typename Op::template vectorized_value_type<ScalarType, width>
          &value) const
      {
        using namespace std;
        return tan(value);
      }
    };


    /**
     * Exponential operator for integrands of symbolic integrals
     *
     * @note Not available for test functions and trial solutions.
     */
    template <typename Op>
    class UnaryOp<Op,
                  UnaryOpCodes::exponential,
                  typename std::enable_if<
                    !is_or_has_test_function_or_trial_solution_op<Op>::value &&
                    !is_integral_op<Op>::value>::type>
      : public UnaryOpBase<UnaryOp<Op, UnaryOpCodes::exponential>>
    {
      static_assert(
        !is_or_has_test_function_or_trial_solution_op<Op>::value,
        "The square root operation is not permitted for test functions or trial solutions.");

      DEAL_II_UNARY_OP_COMMON_IMPL(Op, UnaryOpCodes::exponential)

    public:
      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return decorator.prefixed_parenthesized_symbolic_op_operand_as_ascii(
          "exp", operand);
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return decorator.prefixed_parenthesized_symbolic_op_operand_as_latex(
          Utilities::LaTeX::decorate_text("exp"), operand);
      }

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename Op::template value_type<ScalarType> &value) const
      {
        using namespace std;
        return exp(value);
      }

      template <typename ScalarType, std::size_t width>
      vectorized_value_type<ScalarType, width>
      operator()(
        const typename Op::template vectorized_value_type<ScalarType, width>
          &value) const
      {
        using namespace std;
        return exp(value);
      }
    };


    /**
     * Logarithm operator for integrands of symbolic integrals
     *
     * @note Not available for test functions and trial solutions.
     */
    template <typename Op>
    class UnaryOp<Op,
                  UnaryOpCodes::logarithm,
                  typename std::enable_if<
                    !is_or_has_test_function_or_trial_solution_op<Op>::value &&
                    !is_integral_op<Op>::value>::type>
      : public UnaryOpBase<UnaryOp<Op, UnaryOpCodes::logarithm>>
    {
      static_assert(
        !is_or_has_test_function_or_trial_solution_op<Op>::value,
        "The square root operation is not permitted for test functions or trial solutions.");

      DEAL_II_UNARY_OP_COMMON_IMPL(Op, UnaryOpCodes::logarithm)

    public:
      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return decorator.prefixed_parenthesized_symbolic_op_operand_as_ascii(
          "log", operand);
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return decorator.prefixed_parenthesized_symbolic_op_operand_as_latex(
          Utilities::LaTeX::decorate_text("log"), operand);
      }

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename Op::template value_type<ScalarType> &value) const
      {
        using namespace std;
        return log(value);
      }

      template <typename ScalarType, std::size_t width>
      vectorized_value_type<ScalarType, width>
      operator()(
        const typename Op::template vectorized_value_type<ScalarType, width>
          &value) const
      {
        using namespace std;
        return log(value);
      }
    };


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


    /**
     * Absolute value operator for integrands of symbolic integrals
     *
     * @note Not available for test functions and trial solutions.
     */
    template <typename Op>
    class UnaryOp<Op,
                  UnaryOpCodes::absolute_value,
                  typename std::enable_if<
                    !is_or_has_test_function_or_trial_solution_op<Op>::value &&
                    !is_integral_op<Op>::value>::type>
      : public UnaryOpBase<UnaryOp<Op, UnaryOpCodes::absolute_value>>
    {
      static_assert(
        !is_or_has_test_function_or_trial_solution_op<Op>::value,
        "The square root operation is not permitted for test functions or trial solutions.");

      DEAL_II_UNARY_OP_COMMON_IMPL(Op, UnaryOpCodes::absolute_value)

    public:
      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return decorator.prefixed_parenthesized_symbolic_op_operand_as_ascii(
          "abs", operand);
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const std::string lvert = Utilities::LaTeX::l_vert();
        const std::string rvert = Utilities::LaTeX::r_vert();
        return lvert + operand.as_latex(decorator) + rvert;
      }

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename Op::template value_type<ScalarType> &value) const
      {
        using namespace std;
        return abs(value);
      }

      template <typename ScalarType, std::size_t width>
      vectorized_value_type<ScalarType, width>
      operator()(
        const typename Op::template vectorized_value_type<ScalarType, width>
          &value) const
      {
        using namespace std;
        return abs(value);
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
        if (rank == 0)
          {
            return "1/" + operand.as_ascii(decorator);
          }
        else
          {
            return decorator
              .prefixed_parenthesized_symbolic_op_operand_as_ascii("inv",
                                                                   operand);
          }
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        if (rank == 0)
          {
            return Utilities::LaTeX::decorate_fraction(
              "1", operand.as_latex(decorator));
          }
        else
          {
            return decorator
              .suffixed_braced_superscript_symbolic_op_operand_as_latex(operand,
                                                                        "-1");
          }
      }

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename Op::template value_type<ScalarType> &value) const
      {
        return internal::invert_impl(value);
      }

      template <typename ScalarType, std::size_t width>
      vectorized_value_type<ScalarType, width>
      operator()(
        const typename Op::template vectorized_value_type<ScalarType, width>
          &value) const
      {
        return internal::invert_impl(value);
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



    /**
     * Trace operator for integrands of symbolic integrals
     */
    template <typename Op>
    class UnaryOp<Op,
                  UnaryOpCodes::trace,
                  typename std::enable_if<!is_integral_op<Op>::value>::type>
      : public UnaryOpBase<UnaryOp<Op, UnaryOpCodes::trace>>
    {
      DEAL_II_UNARY_OP_COMMON_IMPL(Op, UnaryOpCodes::trace)

    public:
      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return decorator.prefixed_parenthesized_symbolic_op_operand_as_ascii(
          "tr", operand);
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return decorator.prefixed_parenthesized_symbolic_op_operand_as_latex(
          Utilities::LaTeX::decorate_text("tr"), operand);
      }

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename Op::template value_type<ScalarType> &value) const
      {
        return trace(value);
      }

      template <typename ScalarType, std::size_t width>
      vectorized_value_type<ScalarType, width>
      operator()(
        const typename Op::template vectorized_value_type<ScalarType, width>
          &value) const
      {
        return trace(value);
      }
    };



    /**
     * Adjugate operator for integrands of symbolic integrals
     *
     * @note Not available for test functions and trial solutions.
     */
    template <typename Op>
    class UnaryOp<Op,
                  UnaryOpCodes::adjugate,
                  typename std::enable_if<
                    !is_or_has_test_function_or_trial_solution_op<Op>::value &&
                    !is_integral_op<Op>::value>::type>
      : public UnaryOpBase<UnaryOp<Op, UnaryOpCodes::adjugate>>
    {
      static_assert(
        !is_or_has_test_function_or_trial_solution_op<Op>::value,
        "The adjugate operation is not permitted for test functions or trial solutions.");

      DEAL_II_UNARY_OP_COMMON_IMPL(Op, UnaryOpCodes::adjugate)

    public:
      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return decorator.prefixed_parenthesized_symbolic_op_operand_as_ascii(
          "adj", operand);
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return decorator.prefixed_parenthesized_symbolic_op_operand_as_latex(
          Utilities::LaTeX::decorate_text("adj"), operand);
      }

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename Op::template value_type<ScalarType> &value) const
      {
        return adjugate(value);
      }

      template <typename ScalarType, std::size_t width>
      vectorized_value_type<ScalarType, width>
      operator()(
        const typename Op::template vectorized_value_type<ScalarType, width>
          &value) const
      {
        return adjugate(value);
      }
    };



    /**
     * Cofactor operator for integrands of symbolic integrals
     *
     * @note Not available for test functions and trial solutions.
     */
    template <typename Op>
    class UnaryOp<Op,
                  UnaryOpCodes::cofactor,
                  typename std::enable_if<
                    !is_or_has_test_function_or_trial_solution_op<Op>::value &&
                    !is_integral_op<Op>::value>::type>
      : public UnaryOpBase<UnaryOp<Op, UnaryOpCodes::cofactor>>
    {
      static_assert(
        !is_or_has_test_function_or_trial_solution_op<Op>::value,
        "The cofactor operation is not permitted for test functions or trial solutions.");

      DEAL_II_UNARY_OP_COMMON_IMPL(Op, UnaryOpCodes::cofactor)

    public:
      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return decorator.prefixed_parenthesized_symbolic_op_operand_as_ascii(
          "cof", operand);
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return decorator.prefixed_parenthesized_symbolic_op_operand_as_latex(
          Utilities::LaTeX::decorate_text("cof"), operand);
      }

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename Op::template value_type<ScalarType> &value) const
      {
        return cofactor(value);
      }

      template <typename ScalarType, std::size_t width>
      vectorized_value_type<ScalarType, width>
      operator()(
        const typename Op::template vectorized_value_type<ScalarType, width>
          &value) const
      {
        return cofactor(value);
      }
    };



    /**
     * l1-norm operator for integrands of symbolic integrals
     *
     * @note Not available for test functions and trial solutions.
     */
    template <typename Op>
    class UnaryOp<Op,
                  UnaryOpCodes::l1_norm,
                  typename std::enable_if<
                    !is_or_has_test_function_or_trial_solution_op<Op>::value &&
                    !is_integral_op<Op>::value>::type>
      : public UnaryOpBase<UnaryOp<Op, UnaryOpCodes::l1_norm>>
    {
      static_assert(
        !is_or_has_test_function_or_trial_solution_op<Op>::value,
        "The l1-norm operation is not permitted for test functions or trial solutions.");

      DEAL_II_UNARY_OP_COMMON_IMPL(Op, UnaryOpCodes::l1_norm)

    public:
      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return "||" + operand.as_ascii(decorator) + "||_(1)";
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return Utilities::LaTeX::decorate_norm(operand.as_latex(decorator),
                                               "1");
      }

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename Op::template value_type<ScalarType> &value) const
      {
        return l1_norm(value);
      }

      template <typename ScalarType, std::size_t width>
      vectorized_value_type<ScalarType, width>
      operator()(
        const typename Op::template vectorized_value_type<ScalarType, width>
          &value) const
      {
        // static_assert(false,
        //   "Need to first implement std::fabs() for vectorized values.");
        return l1_norm(value);
      }
    };



    /**
     * l(infinity)-norm operator for integrands of symbolic integrals
     *
     * @note Not available for test functions and trial solutions.
     */
    template <typename Op>
    class UnaryOp<Op,
                  UnaryOpCodes::linfty_norm,
                  typename std::enable_if<
                    !is_or_has_test_function_or_trial_solution_op<Op>::value &&
                    !is_integral_op<Op>::value>::type>
      : public UnaryOpBase<UnaryOp<Op, UnaryOpCodes::linfty_norm>>
    {
      static_assert(
        !is_or_has_test_function_or_trial_solution_op<Op>::value,
        "The linfty-norm operation is not permitted for test functions or trial solutions.");

      DEAL_II_UNARY_OP_COMMON_IMPL(Op, UnaryOpCodes::linfty_norm)

    public:
      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return "||" + operand.as_ascii(decorator) + "||_(inf)";
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return Utilities::LaTeX::decorate_norm(operand.as_latex(decorator),
                                               "\\infty");
      }

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename Op::template value_type<ScalarType> &value) const
      {
        return linfty_norm(value);
      }

      template <typename ScalarType, std::size_t width>
      vectorized_value_type<ScalarType, width>
      operator()(
        const typename Op::template vectorized_value_type<ScalarType, width>
          &value) const
      {
        // static_assert(false,
        //   "Need to first implement std::fabs() for vectorized values.");
        return linfty_norm(value);
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

// General operations
DEAL_II_UNARY_OP_OF_UNARY_OP(operator-, negate)
DEAL_II_UNARY_OP_OF_UNARY_OP(normalize, normalize)

// Scalar operations
DEAL_II_UNARY_OP_OF_UNARY_OP(sin, sine)
DEAL_II_UNARY_OP_OF_UNARY_OP(cos, cosine)
DEAL_II_UNARY_OP_OF_UNARY_OP(tan, tangent)
DEAL_II_UNARY_OP_OF_UNARY_OP(exp, exponential)
DEAL_II_UNARY_OP_OF_UNARY_OP(log, logarithm)
DEAL_II_UNARY_OP_OF_UNARY_OP(sqrt, square_root)
DEAL_II_UNARY_OP_OF_UNARY_OP(abs, absolute_value)

// Tensor operations
DEAL_II_UNARY_OP_OF_UNARY_OP(determinant, determinant)
DEAL_II_UNARY_OP_OF_UNARY_OP(invert, invert)
DEAL_II_UNARY_OP_OF_UNARY_OP(transpose, transpose)
DEAL_II_UNARY_OP_OF_UNARY_OP(symmetrize, symmetrize)
DEAL_II_UNARY_OP_OF_UNARY_OP(trace, trace)
DEAL_II_UNARY_OP_OF_UNARY_OP(adjugate, adjugate)
DEAL_II_UNARY_OP_OF_UNARY_OP(cofactor, cofactor)
DEAL_II_UNARY_OP_OF_UNARY_OP(l1_norm, l1_norm)
DEAL_II_UNARY_OP_OF_UNARY_OP(linfty_norm, linfty_norm)

#undef DEAL_II_UNARY_OP_OF_UNARY_OP



/* ==================== Specialization of type traits ==================== */



#ifndef DOXYGEN


namespace WeakForms
{
  // Scalar operations

  template <typename Op, typename UnderlyingType>
  struct is_unary_op<
    Operators::UnaryOp<Op, Operators::UnaryOpCodes::negate, UnderlyingType>>
    : std::true_type
  {};

  template <typename Op, typename UnderlyingType>
  struct is_unary_op<
    Operators::UnaryOp<Op, Operators::UnaryOpCodes::normalize, UnderlyingType>>
    : std::true_type
  {};

  template <typename Op, typename UnderlyingType>
  struct is_unary_op<
    Operators::UnaryOp<Op, Operators::UnaryOpCodes::sine, UnderlyingType>>
    : std::true_type
  {};

  template <typename Op, typename UnderlyingType>
  struct is_unary_op<
    Operators::UnaryOp<Op, Operators::UnaryOpCodes::cosine, UnderlyingType>>
    : std::true_type
  {};

  template <typename Op, typename UnderlyingType>
  struct is_unary_op<
    Operators::UnaryOp<Op, Operators::UnaryOpCodes::tangent, UnderlyingType>>
    : std::true_type
  {};

  template <typename Op, typename UnderlyingType>
  struct is_unary_op<Operators::UnaryOp<Op,
                                        Operators::UnaryOpCodes::exponential,
                                        UnderlyingType>> : std::true_type
  {};

  template <typename Op, typename UnderlyingType>
  struct is_unary_op<
    Operators::UnaryOp<Op, Operators::UnaryOpCodes::logarithm, UnderlyingType>>
    : std::true_type
  {};

  template <typename Op, typename UnderlyingType>
  struct is_unary_op<Operators::UnaryOp<Op,
                                        Operators::UnaryOpCodes::square_root,
                                        UnderlyingType>> : std::true_type
  {};

  template <typename Op, typename UnderlyingType>
  struct is_unary_op<Operators::UnaryOp<Op,
                                        Operators::UnaryOpCodes::absolute_value,
                                        UnderlyingType>> : std::true_type
  {};

  // Tensor operations

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

  template <typename Op, typename UnderlyingType>
  struct is_unary_op<
    Operators::UnaryOp<Op, Operators::UnaryOpCodes::trace, UnderlyingType>>
    : std::true_type
  {};

  template <typename Op, typename UnderlyingType>
  struct is_unary_op<
    Operators::UnaryOp<Op, Operators::UnaryOpCodes::adjugate, UnderlyingType>>
    : std::true_type
  {};

  template <typename Op, typename UnderlyingType>
  struct is_unary_op<
    Operators::UnaryOp<Op, Operators::UnaryOpCodes::cofactor, UnderlyingType>>
    : std::true_type
  {};

  template <typename Op, typename UnderlyingType>
  struct is_unary_op<
    Operators::UnaryOp<Op, Operators::UnaryOpCodes::l1_norm, UnderlyingType>>
    : std::true_type
  {};

  template <typename Op, typename UnderlyingType>
  struct is_unary_op<Operators::UnaryOp<Op,
                                        Operators::UnaryOpCodes::linfty_norm,
                                        UnderlyingType>> : std::true_type
  {};

} // namespace WeakForms


#endif // DOXYGEN



WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_unary_operators_h
