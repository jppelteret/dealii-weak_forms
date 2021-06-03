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

#ifndef dealii_weakforms_tensor_operators_h
#define dealii_weakforms_tensor_operators_h


// TODO: Base this on the approach taken in BinaryOps



// #include <deal.II/base/config.h>
#include <weak_forms/config.h>

// #include <deal.II/base/exceptions.h>
// #include <deal.II/base/symmetric_tensor.h>

// #include <deal.II/fe/fe_values.h>

// #include <weak_forms/symbolic_decorations.h>
// #include <weak_forms/symbolic_operators.h>


// WEAK_FORMS_NAMESPACE_OPEN


// #ifndef DOXYGEN

// // Forward declarations
// namespace WeakForms
// {
//   /* --------------- Operators acting on tensors --------------- */

//   template <int rank, int spacedim>
//   class Symmetrize;



//   template <int rank, int spacedim, typename Functor>
//   WeakForms::Operators::SymbolicOp<Symmetrize<rank, spacedim>,
//                                 WeakForms::Operators::SymbolicOpCodes::value,
//                                 Functor>
//   value(const Symmetrize<rank, spacedim> &operand, const Functor
//   &functor_op);

// } // namespace WeakForms

// #endif // DOXYGEN



// namespace WeakForms
// {
//   // TODO:
//   // - Transpose
//   // - Contract
//   // - Double contract
//   // - Scalar product
//   // - Determinant
//   // - Invert
//   // - Symmetrize
//   // - Cross product -- > This is a binary op!


//   /* --------------- Operators acting on tensors --------------- */

//   template <int rank_, int spacedim>
//   class Symmetrize
//   {
//   public:
//     Symmetrize() = default;

//     /**
//      * Dimension of the space in which this object operates.
//      */
//     static const unsigned int space_dimension = spacedim;

//     /**
//      * Rank of this object operates.
//      */
//     static const unsigned int rank = rank_;

//     template <typename ScalarType>
//     using value_type = SymmetricTensor<rank, spacedim, ScalarType>;

//     // Call operator to promote this class to a SymbolicOp
//     template<typename Functor>
//     auto
//     operator()(const Functor &functor_op) const
//     {
//       return WeakForms::value(*this, functor_op);
//     }

//     // Let's give our users a nicer syntax to work with this
//     // templated call operator.
//     template<typename Functor>
//     auto
//     value(const Functor &functor_op) const
//     {
//       return this->operator()(functor_op);
//     }

//     // ----  Ascii ----

//     std::string
//     as_ascii(const SymbolicDecorations &decorator) const
//     {
//       return decorator.symbolic_op_operand_as_ascii(*this);
//     }

//     std::string
//     get_symbol_ascii(const SymbolicDecorations &decorator) const
//     {
//       return decorator.naming_ascii.symmetrize;
//     }

//     virtual std::string
//     get_field_ascii(const SymbolicDecorations &decorator) const
//     {
//       return "";
//     }

//     // ---- LaTeX ----

//     std::string
//     as_latex(const SymbolicDecorations &decorator) const
//     {
//       return decorator.symbolic_op_operand_as_latex(*this);
//     }

//     std::string
//     get_symbol_latex(const SymbolicDecorations &decorator) const
//     {
//       return decorator.naming_latex.symmetrize;
//     }

//     virtual std::string
//     get_field_latex(const SymbolicDecorations &decorator) const
//     {
//       return "";
//     }
//   };

// } // namespace WeakForms



// /* ================== Specialization of unary operators ================== */



// namespace WeakForms
// {
//   namespace Operators
//   {
//     /* --------------- Cell face and cell subface operators ---------------
//     */

//     /**
//      * Extract the Symmetrizes from a cell face.
//      */
//     template <int rank, int spacedim, typename Functor>
//     class SymbolicOp<Symmetrize<rank, spacedim>, SymbolicOpCodes::value>
//     {
//       using Op = Symmetrize<rank, spacedim>;

//     public:
//       static const int rank = Op::rank;

//       template <typename ResultScalarType>
//       using value_type = typename Op::template value_type<ResultScalarType>;

//       template <typename ResultScalarType>
//       using return_type = std::vector<value_type<ResultScalarType>>;

//       explicit SymbolicOp(const Op &operand, const Functor &functor_op)
//         : operand(operand)
//         , functor_op (functor_op)
//       {}

//       std::string
//       as_ascii(const SymbolicDecorations &decorator) const
//       {
//         const auto &naming =
//         decorator.get_naming_ascii().differential_operators; return
//         decorator.decorate_with_operator_ascii(
//           naming.value, operand.as_ascii(decorator));
//       }

//       std::string
//       as_latex(const SymbolicDecorations &decorator) const
//       {
//         const auto &naming =
//         decorator.get_naming_latex().differential_operators; return
//         decorator.decorate_with_operator_latex(
//           naming.value, operand.as_latex(decorator));
//       }

//       // =======

//       UpdateFlags
//       get_update_flags() const
//       {
//         return UpdateFlags::update_default;
//       }

//       /**
//        * Return symmetrized result of functor operations at all quadrature
//        points
//        */
//       template <typename ResultScalarType, int dim>
//       return_type<ResultScalarType>
//       operator()(const FEValuesBase<dim, spacedim> &fe_face_values) const
//       {
//         Assert((dynamic_cast<const FEFaceValuesBase<dim, spacedim> *>(
//                  &fe_face_values)),
//                ExcNotCastableToFEFaceValuesBase());
//         return static_cast<const FEFaceValuesBase<dim, spacedim> &>(
//                  fe_face_values)
//           .get_Symmetrize_vectors();
//       }

//     private:
//       const Op operand;
//       const Functor functor_op;
//     };

//   } // namespace Operators
// } // namespace WeakForms



// /* ======================== Convenience functions ======================== */



// namespace WeakForms
// {
//   template <int rank, int spacedim, typename Functor>
//   WeakForms::Operators::SymbolicOp<WeakForms::Symmetrize<spacedim>,
//                                 WeakForms::Operators::SymbolicOpCodes::value>
//   value(const WeakForms::Symmetrize<spacedim> &operand, const Functor &
//   functor)
//   {
//     using namespace WeakForms;
//     using namespace WeakForms::Operators;

//     using Op     = Symmetrize<rank, spacedim>;
//     using OpType = SymbolicOp<Op, SymbolicOpCodes::value, Functor>;

//     return OpType(operand, functor);
//   }

// } // namespace WeakForms



// /* ==================== Specialization of type traits ==================== */



// #ifndef DOXYGEN


// namespace WeakForms
// {} // namespace WeakForms


// #endif // DOXYGEN


// WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_tensor_operators_h
