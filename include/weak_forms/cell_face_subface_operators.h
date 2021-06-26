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

#ifndef dealii_weakforms_cell_face_subface_operators_h
#define dealii_weakforms_cell_face_subface_operators_h

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_values.h>

#include <weak_forms/config.h>
#include <weak_forms/numbers.h>
#include <weak_forms/symbolic_decorations.h>
#include <weak_forms/symbolic_operators.h>
#include <weak_forms/type_traits.h>
#include <weak_forms/types.h>


WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
  /**
   * Exception denoting that a class requires some specialization
   * in order to be used.
   */
  DeclExceptionMsg(
    ExcNotCastableToFEFaceValuesBase,
    "The input FEValuesBase object cannot be cast to an  FEFaceValuesBase "
    "object. This is required for attributes on a cell face to be retrieved.");



  /* --------------- Cell face and cell subface operators --------------- */

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
    using value_type = Tensor<rank, spacedim, double>;

    template <typename ScalarType, std::size_t width>
    using vectorized_value_type = typename numbers::VectorizedValue<
      value_type<ScalarType>>::template type<width>;

    // Call operator to promote this class to a SymbolicOp
    auto
    operator()() const;

    // Let's give our users a nicer syntax to work with this
    // templated call operator.
    auto
    value() const
    {
      return this->operator()();
    }

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
    /* --------------- Cell face and cell subface operators --------------- */

    /**
     * Extract the normals from a cell face or interface.
     */
    template <int dim, int spacedim>
    class SymbolicOp<Normal<dim, spacedim>, SymbolicOpCodes::value>
    {
      using Op = Normal<dim, spacedim>;

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

      template <typename ResultScalarType, std::size_t width>
      using vectorized_value_type =
        typename Op::template vectorized_value_type<ResultScalarType, width>;

      template <typename ResultScalarType>
      using return_type = std::vector<value_type<ResultScalarType>>;

      template <typename ResultScalarType, std::size_t width>
      using vectorized_return_type =
        vectorized_value_type<ResultScalarType, width>;

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
      template <typename ResultScalarType, int dim2>
      const return_type<ResultScalarType> &
      operator()(const FEValuesBase<dim2, spacedim> &fe_face_values) const
      {
        Assert((dynamic_cast<const FEFaceValuesBase<dim2, spacedim> *>(
                 &fe_face_values)),
               ExcNotCastableToFEFaceValuesBase());
        return static_cast<const FEFaceValuesBase<dim2, spacedim> &>(
                 fe_face_values)
          .get_normal_vectors();
      }

      template <typename ResultScalarType, std::size_t width, int dim2>
      vectorized_return_type<ResultScalarType, width>
      operator()(const FEValuesBase<dim2, spacedim> &fe_face_values,
                 const types::vectorized_qp_range_t &q_point_range) const
      {
        vectorized_return_type<ResultScalarType, width> out;
        Assert(q_point_range.size() <= width,
               ExcIndexRange(q_point_range.size(), 0, width));

        DEAL_II_OPENMP_SIMD_PRAGMA
        for (unsigned int i = 0; i < q_point_range.size(); ++i)
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
      template <typename ResultScalarType, int dim2>
      const return_type<ResultScalarType> &
      operator()(
        const FEInterfaceValues<dim2, spacedim> &fe_interface_values) const
      {
        return fe_interface_values.get_normal_vectors();
      }

      template <typename ResultScalarType, std::size_t width, int dim2>
      vectorized_return_type<ResultScalarType, width>
      operator()(const FEInterfaceValues<dim2, spacedim> &fe_interface_values,
                 const types::vectorized_qp_range_t &     q_point_range) const
      {
        vectorized_return_type<ResultScalarType, width> out;
        Assert(q_point_range.size() <= width,
               ExcIndexRange(q_point_range.size(), 0, width));

        DEAL_II_OPENMP_SIMD_PRAGMA
        for (unsigned int i = 0; i < q_point_range.size(); ++i)
          numbers::set_vectorized_values(
            out,
            i,
            this->template operator()<ResultScalarType>(
              fe_interface_values)[q_point_range[i]]);

        return out;
      }

    private:
      const Op operand;
    };

  } // namespace Operators
} // namespace WeakForms



/* ======================== Convenience functions ======================== */



namespace WeakForms
{
  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<WeakForms::Normal<dim, spacedim>,
                                   WeakForms::Operators::SymbolicOpCodes::value>
  value(const WeakForms::Normal<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = Normal<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::value>;

    return OpType(operand);
  }

} // namespace WeakForms


/* ==================== Class method definitions ==================== */


namespace WeakForms
{
  template <int dim, int spacedim>
  auto
  Normal<dim, spacedim>::operator()() const
  {
    return WeakForms::value(*this);
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

} // namespace WeakForms


#endif // DOXYGEN


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_cell_face_subface_operators_h
