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

#ifndef dealii_weakforms_cell_face_subface_operators_h
#define dealii_weakforms_cell_face_subface_operators_h

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>

#include <deal.II/differentiation/sd.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_values.h>

#include <weak_forms/config.h>
#include <weak_forms/numbers.h>
#include <weak_forms/sd_expression_internal.h>
#include <weak_forms/symbolic_decorations.h>
#include <weak_forms/symbolic_operators.h>
#include <weak_forms/type_traits.h>
#include <weak_forms/types.h>


WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
#ifndef DOXYGEN

  /**
   * Exception denoting that a class requires some specialization
   * in order to be used.
   */
  DeclExceptionMsg(
    ExcNotCastableToFEFaceValuesBase,
    "The input FEValuesBase object cannot be cast to an  FEFaceValuesBase "
    "object. This is required for attributes on a cell face to be retrieved.");

#endif // DOXYGEN


  /* --------------- Cell face and cell subface operators --------------- */

  /**
   * @brief A functor that represents the normal of the cell, as evaluated at a boundary or an interface.
   *
   * An example of usage:
   * @code {.cpp}
   * const Normal<spacedim> normal{};
   * const auto N = normal.value();
   * @endcode
   *
   * @tparam dim The dimension in which the scalar is being evaluated.
   * @tparam spacedim The spatial dimension in which the scalar is being evaluated.
   */
  template <int dim, int spacedim = dim>
  class Normal
  {
  public:
    Normal() = default;

    /**
     * Dimension in which this object operates.
     */
    static const unsigned int dimension = dim;

    /**
     * Dimension of the space in which this object operates.
     */
    static const unsigned int space_dimension = spacedim;

    /**
     * Rank of this object operates.
     */
    static const unsigned int rank = 1;

    template <typename ScalarType>
    using value_type = Tensor<rank, spacedim, ScalarType>;

    // Methods to promote this class to a SymbolicOp

    auto
    value() const;

    // ----  Ascii ----

    std::string
    as_ascii(const SymbolicDecorations &decorator) const
    {
      return decorator.symbolic_op_operand_as_ascii(*this);
    }

    std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const
    {
      const auto &naming = decorator.get_naming_ascii().geometry;
      return naming.normal;
    }

    virtual std::string
    get_field_ascii(const SymbolicDecorations &decorator) const
    {
      (void)decorator;
      return "";
    }

    // ---- LaTeX ----

    std::string
    as_latex(const SymbolicDecorations &decorator) const
    {
      return decorator.symbolic_op_operand_as_latex(*this);
    }

    std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const
    {
      const auto &naming = decorator.get_naming_latex().geometry;
      return naming.normal;
    }

    virtual std::string
    get_field_latex(const SymbolicDecorations &decorator) const
    {
      (void)decorator;
      return "";
    }
  };


  /* ---------------Cell, cell face and cell subface operators ---------------
   */

  // See
  // https://dealii.org/developer/doxygen/deal.II/classFEValues.html
  // https://dealii.org/developer/doxygen/deal.II/classFEFaceValues.html
  // https://dealii.org/developer/doxygen/deal.II/classFESubfaceValues.html

  // Jacobian

  // Jacobian (pushed forward)

  // Inverse jacobian

} // namespace WeakForms



/* ================== Specialization of unary operators ================== */



namespace WeakForms
{
  namespace Operators
  {
#ifdef DEAL_II_WITH_SYMENGINE

/**
 * A macro that performs a conversion of the functor to a symbolic
 * expression type.
 */
#  define DEAL_II_SYMBOLIC_EXPRESSION_CONVERSION_COMMON_IMPL()                 \
    value_type<dealii::Differentiation::SD::Expression> as_expression(         \
      const SymbolicDecorations &decorator = SymbolicDecorations()) const      \
                                                                               \
    {                                                                          \
      return WeakForms::Operators::internal::make_symbolic<                    \
        value_type<dealii::Differentiation::SD::Expression>>(                  \
        this->as_ascii(decorator));                                            \
    }                                                                          \
                                                                               \
    Differentiation::SD::types::substitution_map get_symbol_registration_map() \
      const                                                                    \
    {                                                                          \
      return Differentiation::SD::make_symbol_map(this->as_expression());      \
    }                                                                          \
                                                                               \
    Differentiation::SD::types::substitution_map                               \
    get_intermediate_substitution_map() const                                  \
    {                                                                          \
      return Differentiation::SD::types::substitution_map{};                   \
    }                                                                          \
                                                                               \
    Differentiation::SD::types::substitution_map get_substitution_map(         \
      const MeshWorker::ScratchData<dim, spacedim> &scratch_data,              \
      const std::vector<SolutionExtractionData<dim, spacedim>>                 \
        &                solution_extraction_data,                             \
      const unsigned int q_point) const                                        \
    {                                                                          \
      (void)solution_extraction_data;                                          \
      const auto &fe_values = scratch_data.get_current_fe_values();            \
      return Differentiation::SD::make_substitution_map(                       \
        this->as_expression(),                                                 \
        this->template operator()<ResultScalarType>(fe_values)[q_point]);      \
    }

#else // DEAL_II_WITH_SYMENGINE

/**
 * A dummy macro.
 */
#  define DEAL_II_SYMBOLIC_EXPRESSION_CONVERSION_COMMON_IMPL() ;

#endif // DEAL_II_WITH_SYMENGINE


    /* --------------- Cell face and cell subface operators --------------- */

    /**
     * Extract the normals from a cell face or interface.
     */
    template <int dim, int spacedim>
    class SymbolicOp<Normal<dim, spacedim>, SymbolicOpCodes::value>
    {
      using Op = Normal<dim, spacedim>;

      // Normals are always defined as a tensor of doubles.
      using ResultScalarType = double;

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = dim;

      /**
       * Dimension of the space in which this object operates.
       */
      static const unsigned int space_dimension = spacedim;

      static const int rank = Op::rank;
      static_assert(rank == 1, "Expected a rank 1 operation");

      template <typename ResultScalarType>
      using value_type = typename Op::template value_type<ResultScalarType>;

      template <typename ResultScalarType>
      using return_type = std::vector<value_type<ResultScalarType>>;

      template <typename ResultScalarType, std::size_t width>
      using vectorized_value_type = typename numbers::VectorizedValue<
        value_type<ResultScalarType>>::template type<width>;

      template <typename ResultScalarType, std::size_t width>
      using vectorized_return_type = typename numbers::VectorizedValue<
        value_type<ResultScalarType>>::template type<width>;

      explicit SymbolicOp(const Op &operand)
        : operand(operand)
      {}

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_ascii().differential_operators;
        return decorator.decorate_with_operator_ascii(
          naming.value, operand.as_ascii(decorator));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_latex().differential_operators;
        return decorator.decorate_with_operator_latex(
          naming.value, operand.as_latex(decorator));
      }

      // =======

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_normal_vectors;
      }

      /**
       * Return normals at all quadrature points
       */
      template <typename ResultScalarType>
      const return_type<ResultScalarType> &
      operator()(const FEValuesBase<dim, spacedim> &fe_face_values) const
      {
        Assert((dynamic_cast<const FEFaceValuesBase<dim, spacedim> *>(
                 &fe_face_values)),
               ExcNotCastableToFEFaceValuesBase());
        return static_cast<const FEFaceValuesBase<dim, spacedim> &>(
                 fe_face_values)
          .get_normal_vectors();
      }

      template <typename ResultScalarType, std::size_t width>
      vectorized_return_type<ResultScalarType, width>
      operator()(const FEValuesBase<dim, spacedim> & fe_face_values,
                 const types::vectorized_qp_range_t &q_point_range) const
      {
        vectorized_return_type<ResultScalarType, width> out;
        Assert(q_point_range.size() <= width,
               ExcIndexRange(q_point_range.size(), 0, width));

        DEAL_II_OPENMP_SIMD_PRAGMA
        for (unsigned int i = 0; i < q_point_range.size(); ++i)
          if (q_point_range[i] < fe_face_values.n_quadrature_points)
            numbers::set_vectorized_values(
              out,
              i,
              this->template operator()<ResultScalarType>(
                fe_face_values)[q_point_range[i]]);

        return out;
      }

      /**
       * Return normals at all quadrature points
       */
      template <typename ResultScalarType>
      const return_type<ResultScalarType> &
      operator()(
        const FEInterfaceValues<dim, spacedim> &fe_interface_values) const
      {
        return fe_interface_values.get_normal_vectors();
      }

      template <typename ResultScalarType, std::size_t width>
      vectorized_return_type<ResultScalarType, width>
      operator()(const FEInterfaceValues<dim, spacedim> &fe_interface_values,
                 const types::vectorized_qp_range_t &    q_point_range) const
      {
        vectorized_return_type<ResultScalarType, width> out;
        Assert(q_point_range.size() <= width,
               ExcIndexRange(q_point_range.size(), 0, width));

        DEAL_II_OPENMP_SIMD_PRAGMA
        for (unsigned int i = 0; i < q_point_range.size(); ++i)
          if (q_point_range[i] < fe_interface_values.n_quadrature_points)
            numbers::set_vectorized_values(
              out,
              i,
              this->template operator()<ResultScalarType>(
                fe_interface_values)[q_point_range[i]]);

        return out;
      }

      DEAL_II_SYMBOLIC_EXPRESSION_CONVERSION_COMMON_IMPL()

    private:
      const Op operand;
    };

#undef DEAL_II_SYMBOLIC_EXPRESSION_CONVERSION_COMMON_IMPL

  } // namespace Operators
} // namespace WeakForms



/* ==================== Class method definitions ==================== */



namespace WeakForms
{
  template <int dim, int spacedim>
  DEAL_II_ALWAYS_INLINE inline auto
  Normal<dim, spacedim>::value() const
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = Normal<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::value>;

    const auto &operand = *this;
    return OpType(operand);
  }
} // namespace WeakForms



/* ==================== Specialization of type traits ==================== */



#ifndef DOXYGEN


namespace WeakForms
{
  template <int dim, int spacedim, enum Operators::SymbolicOpCodes OpCode>
  struct is_boundary_op<Operators::SymbolicOp<Normal<dim, spacedim>, OpCode>>
    : std::true_type
  {};

  template <int dim, int spacedim, enum Operators::SymbolicOpCodes OpCode>
  struct is_interface_op<Operators::SymbolicOp<Normal<dim, spacedim>, OpCode>>
    : std::true_type
  {};

  template <int dim, int spacedim, enum Operators::SymbolicOpCodes OpCode>
  struct is_cell_geometry_op<
    Operators::SymbolicOp<Normal<dim, spacedim>, OpCode>> : std::true_type
  {};

} // namespace WeakForms


#endif // DOXYGEN


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_cell_face_subface_operators_h
