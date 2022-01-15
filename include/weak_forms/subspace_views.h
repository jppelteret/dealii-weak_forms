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

#ifndef dealii_weakforms_subspace_views_h
#define dealii_weakforms_subspace_views_h

#include <deal.II/base/config.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/meshworker/scratch_data.h>

#include <weak_forms/config.h>
#include <weak_forms/solution_extraction_data.h>
#include <weak_forms/spaces.h>
#include <weak_forms/symbolic_decorations.h>
#include <weak_forms/symbolic_operators.h>
#include <weak_forms/type_traits.h>
#include <weak_forms/types.h>


WEAK_FORMS_NAMESPACE_OPEN


#ifndef DOXYGEN

// Forward declarations
namespace WeakForms
{
  namespace SelfLinearization
  {
    namespace internal
    {
      struct ConvertTo;
    } // namespace internal
  }   // namespace SelfLinearization



  namespace internal
  {
    /* ----- Finite element subspaces: Test functions and trial solutions -----
     */


    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::value>
    value(const SubSpaceViewsType<SpaceType> &operand);


    template <template <int, class> class SubSpaceViewsType,
              int rank,
              typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<rank, SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::value>
    value(const SubSpaceViewsType<rank, SpaceType> &operand);


    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::gradient>
    gradient(const SubSpaceViewsType<SpaceType> &operand);


    template <template <int, class> class SubSpaceViewsType,
              int rank,
              typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<rank, SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::gradient>
    gradient(const SubSpaceViewsType<rank, SpaceType> &operand);


    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::symmetric_gradient>
    symmetric_gradient(const SubSpaceViewsType<SpaceType> &operand);


    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::divergence>
    divergence(const SubSpaceViewsType<SpaceType> &operand);


    template <template <int, class> class SubSpaceViewsType,
              int rank,
              typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<rank, SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::divergence>
    divergence(const SubSpaceViewsType<rank, SpaceType> &operand);


    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::curl>
    curl(const SubSpaceViewsType<SpaceType> &operand);


    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::laplacian>
    laplacian(const SubSpaceViewsType<SpaceType> &operand);


    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::hessian>
    hessian(const SubSpaceViewsType<SpaceType> &operand);


    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::third_derivative>
    third_derivative(const SubSpaceViewsType<SpaceType> &operand);



    /* -- Finite element subspaces: Test functions and trial solutions
     * (interface)
     * -- */


    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::jump_in_values>
    jump_in_values(const SubSpaceViewsType<SpaceType> &operand);


    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::jump_in_gradients>
    jump_in_gradients(const SubSpaceViewsType<SpaceType> &operand);


    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::jump_in_hessians>
    jump_in_hessians(const SubSpaceViewsType<SpaceType> &operand);


    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::jump_in_third_derivatives>
    jump_in_third_derivatives(const SubSpaceViewsType<SpaceType> &operand);



    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::average_of_values>
    average_of_values(const SubSpaceViewsType<SpaceType> &operand);


    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::average_of_gradients>
    average_of_gradients(const SubSpaceViewsType<SpaceType> &operand);


    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::average_of_hessians>
    average_of_hessians(const SubSpaceViewsType<SpaceType> &operand);



    /* ------------- Finite element subspaces: Field solutions ------------- */


    template <types::solution_index solution_index = 0,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::value,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    value(const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand);


    template <types::solution_index solution_index = 0,
              template <int, class>
              class SubSpaceViewsType,
              int rank,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<rank, FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::value,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    value(const SubSpaceViewsType<rank, FieldSolution<dim, spacedim>> &operand);


    template <types::solution_index solution_index = 0,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::gradient,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    gradient(const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand);


    template <types::solution_index solution_index = 0,
              template <int, class>
              class SubSpaceViewsType,
              int rank,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<rank, FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::gradient,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    gradient(
      const SubSpaceViewsType<rank, FieldSolution<dim, spacedim>> &operand);


    template <types::solution_index solution_index = 0,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::symmetric_gradient,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    symmetric_gradient(
      const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand);


    template <types::solution_index solution_index = 0,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::divergence,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    divergence(const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand);


    template <types::solution_index solution_index = 0,
              template <int, class>
              class SubSpaceViewsType,
              int rank,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<rank, FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::divergence,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    divergence(
      const SubSpaceViewsType<rank, FieldSolution<dim, spacedim>> &operand);


    template <types::solution_index solution_index = 0,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::curl,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    curl(const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand);


    template <types::solution_index solution_index = 0,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::laplacian,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    laplacian(const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand);


    template <types::solution_index solution_index = 0,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::hessian,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    hessian(const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand);


    template <types::solution_index solution_index = 0,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::third_derivative,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    third_derivative(
      const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand);



    /* -------- Finite element subspaces: Field solutions (interface) --------
     */


    template <types::solution_index solution_index = 0,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::jump_in_values,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    jump_in_values(
      const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand);


    template <types::solution_index solution_index = 0,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::jump_in_gradients,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    jump_in_gradients(
      const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand);


    template <types::solution_index solution_index = 0,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::jump_in_hessians,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    jump_in_hessians(
      const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand);


    template <types::solution_index solution_index = 0,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::jump_in_third_derivatives,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    jump_in_third_derivatives(
      const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand);


    template <types::solution_index solution_index = 0,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::average_of_values,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    average_of_values(
      const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand);


    template <types::solution_index solution_index = 0,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::average_of_gradients,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    average_of_gradients(
      const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand);


    template <types::solution_index solution_index = 0,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::average_of_hessians,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    average_of_hessians(
      const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand);
  } // namespace internal

} // namespace WeakForms

#endif // DOXYGEN



namespace WeakForms
{
  namespace SubSpaceViews
  {
    template <typename SpaceType, typename ExtractorType>
    class SubSpaceViewBase
    {
    public:
      using extractor_type = ExtractorType;

      virtual ~SubSpaceViewBase() = default;

      virtual SubSpaceViewBase *
      clone() const = 0;

      types::field_index
      get_field_index() const
      {
        return space.get_field_index();
      }

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return space.as_ascii(decorator);
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return space.as_latex(decorator);
      }

      std::string
      get_field_ascii(const SymbolicDecorations &decorator) const
      {
        return space.get_field_ascii(decorator);
      }

      std::string
      get_field_latex(const SymbolicDecorations &decorator) const
      {
        return space.get_field_latex(decorator);
      }

      std::string
      get_symbol_ascii(const SymbolicDecorations &decorator) const
      {
        return space.get_symbol_ascii(decorator);
      }

      std::string
      get_symbol_latex(const SymbolicDecorations &decorator) const
      {
        return space.get_symbol_latex(decorator);
      }

      const ExtractorType &
      get_extractor() const
      {
        return extractor;
      }

    protected:
      // Allow access to get_space()
      friend WeakForms::SelfLinearization::internal::ConvertTo;

      // Only want this to be a base class providing common implementation
      // for concrete views
      explicit SubSpaceViewBase(const SpaceType &    space,
                                const ExtractorType &extractor)
        : space(space)
        , extractor(extractor)
      {}

      SubSpaceViewBase(const SubSpaceViewBase &) = default;

      const SpaceType &
      get_space() const
      {
        return space;
      }

    private:
      const SpaceType     space;
      const ExtractorType extractor;
    };


    namespace internal
    {
      template <typename ExtractorType>
      struct FEValuesExtractorHelper;

      template <>
      struct FEValuesExtractorHelper<FEValuesExtractors::Scalar>
      {
        static const int rank = 0;

        template <int dim, int spacedim>
        using view_type = FEValuesViews::Scalar<dim, spacedim>;

        template <int dim, int spacedim>
        using interface_view_type = FEInterfaceViews::Scalar<dim, spacedim>;

        static unsigned int
        first_component(const FEValuesExtractors::Scalar &extractor)
        {
          return extractor.component;
        }
      };

      template <>
      struct FEValuesExtractorHelper<FEValuesExtractors::Vector>
      {
        static const int rank = 1;

        template <int dim, int spacedim>
        using view_type = FEValuesViews::Vector<dim, spacedim>;

        template <int dim, int spacedim>
        using interface_view_type = FEInterfaceViews::Vector<dim, spacedim>;

        static unsigned int
        first_component(const FEValuesExtractors::Vector &extractor)
        {
          return extractor.first_vector_component;
        }
      };

      template <int rank_>
      struct FEValuesExtractorHelper<FEValuesExtractors::Tensor<rank_>>
      {
        static const int rank = rank_;

        template <int dim, int spacedim>
        using view_type = FEValuesViews::Tensor<rank_, dim, spacedim>;

        // template <int dim, int spacedim>
        // using interface_view_type = std::nullptr_t;

        static unsigned int
        first_component(const FEValuesExtractors::Tensor<rank_> &extractor)
        {
          return extractor.first_tensor_component;
        }
      };

      template <int rank_>
      struct FEValuesExtractorHelper<FEValuesExtractors::SymmetricTensor<rank_>>
      {
        static const int rank = rank_;

        template <int dim, int spacedim>
        using view_type = FEValuesViews::SymmetricTensor<rank_, dim, spacedim>;

        // template <int dim, int spacedim>
        // using interface_view_type = std::nullptr_t;

        static unsigned int
        first_component(
          const FEValuesExtractors::SymmetricTensor<rank_> &extractor)
        {
          return extractor.first_tensor_component;
        }
      };
    } // namespace internal


/**
 * A macro to implement the common parts of a subspace view class.
 * It is expected that the unary op derives from a
 * SubSpaceViewBase<SpaceType, FEValuesExtractors::<TYPE>> .
 *
 * What remains to be defined are:
 * - static const int rank
 *
 * @note The @p ClassName should match the type that is used in the
 * FEValuesExtractors and FEValuesViews namespaces.
 *
 * @note It is intended that this should used immediately after class
 * definition is opened.
 */
#define DEAL_II_SUBSPACE_VIEW_COMMON_IMPL(ClassName,                        \
                                          SpaceType_,                       \
                                          FEValuesExtractorType)            \
private:                                                                    \
  using Base_t = SubSpaceViewBase<SpaceType_, FEValuesExtractorType>;       \
                                                                            \
public:                                                                     \
  /**                                                                       \
   *                                                                        \
   */                                                                       \
  using SpaceType = SpaceType_;                                             \
                                                                            \
  /**                                                                       \
   *                                                                        \
   */                                                                       \
  using extractor_type = typename Base_t::extractor_type;                   \
                                                                            \
  /**                                                                       \
   * Dimension in which this object operates.                               \
   */                                                                       \
  static const unsigned int dimension = SpaceType::dimension;               \
                                                                            \
  /**                                                                       \
   * Dimension of the subspace in which this object operates.               \
   */                                                                       \
  static const unsigned int space_dimension = SpaceType::space_dimension;   \
  /**                                                                       \
   * Rank of subspace                                                       \
   */                                                                       \
  static const int rank =                                                   \
    internal::FEValuesExtractorHelper<FEValuesExtractorType>::rank;         \
                                                                            \
  /**                                                                       \
   *                                                                        \
   */                                                                       \
  using FEValuesViewsType = typename internal::FEValuesExtractorHelper<     \
    FEValuesExtractorType>::template view_type<dimension, space_dimension>; \
                                                                            \
  explicit ClassName(const SpaceType &            space,                    \
                     const FEValuesExtractorType &extractor)                \
    : Base_t(space, extractor)                                              \
  {}                                                                        \
                                                                            \
  ClassName(const ClassName &) = default;                                   \
                                                                            \
  virtual ClassName *clone() const override                                 \
  {                                                                         \
    return new ClassName(*this);                                            \
  }


    template <typename SpaceType_>
    class Scalar final
      : public SubSpaceViewBase<SpaceType_, FEValuesExtractors::Scalar>
    {
      DEAL_II_SUBSPACE_VIEW_COMMON_IMPL(Scalar,
                                        SpaceType_,
                                        FEValuesExtractors::Scalar)
      static_assert(rank == SpaceType::rank,
                    "Unexpected rank in parent space.");

    public:
      template <typename ScalarType>
      using value_type =
        typename FEValuesViewsType::template solution_value_type<ScalarType>;

      template <typename ScalarType>
      using gradient_type =
        typename FEValuesViewsType::template solution_gradient_type<ScalarType>;

      template <typename ScalarType>
      using hessian_type =
        typename FEValuesViewsType::template solution_hessian_type<ScalarType>;

      template <typename ScalarType>
      using laplacian_type =
        typename FEValuesViewsType::template solution_laplacian_type<
          ScalarType>;

      template <typename ScalarType>
      using third_derivative_type =
        typename FEValuesViewsType::template solution_third_derivative_type<
          ScalarType>;


      // Methods to promote this class to a SymbolicOp:
      // Operators: Test functions, trial solutions, and field solutions

      auto
      value() const
      {
        return WeakForms::internal::value(*this);
      }

      auto
      gradient() const
      {
        return WeakForms::internal::gradient(*this);
      }

      auto
      laplacian() const
      {
        return WeakForms::internal::laplacian(*this);
      }

      auto
      hessian() const
      {
        return WeakForms::internal::hessian(*this);
      }

      auto
      third_derivative() const
      {
        return WeakForms::internal::third_derivative(*this);
      }

      // Operators: Field solutions only

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      value() const
      {
        return WeakForms::internal::value<solution_index>(*this);
      }

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      gradient() const
      {
        return WeakForms::internal::gradient<solution_index>(*this);
      }

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      laplacian() const
      {
        return WeakForms::internal::laplacian<solution_index>(*this);
      }

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      hessian() const
      {
        return WeakForms::internal::hessian<solution_index>(*this);
      }

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      third_derivative() const
      {
        return WeakForms::internal::third_derivative<solution_index>(*this);
      }

      // Methods to promote this class to a SymbolicOp:
      // Interface

      auto
      jump_in_values() const
      {
        return WeakForms::internal::jump_in_values(*this);
      }

      auto
      jump_in_gradients() const
      {
        return WeakForms::internal::jump_in_gradients(*this);
      }

      auto
      jump_in_hessians() const
      {
        return WeakForms::internal::jump_in_hessians(*this);
      }

      auto
      jump_in_third_derivatives() const
      {
        return WeakForms::internal::jump_in_third_derivatives(*this);
      }

      auto
      average_of_values() const
      {
        return WeakForms::internal::average_of_values(*this);
      }

      auto
      average_of_gradients() const
      {
        return WeakForms::internal::average_of_gradients(*this);
      }

      auto
      average_of_hessians() const
      {
        return WeakForms::internal::average_of_hessians(*this);
      }

      // Operators: Field solutions only (interface)

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      jump_in_values() const
      {
        return WeakForms::internal::jump_in_values<solution_index>(*this);
      }

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      jump_in_gradients() const
      {
        return WeakForms::internal::jump_in_gradients<solution_index>(*this);
      }

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      jump_in_hessians() const
      {
        return WeakForms::internal::jump_in_hessians<solution_index>(*this);
      }

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      jump_in_third_derivatives() const
      {
        return WeakForms::internal::jump_in_third_derivatives<solution_index>(
          *this);
      }

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      average_of_values() const
      {
        return WeakForms::internal::average_of_values<solution_index>(*this);
      }

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      average_of_gradients() const
      {
        return WeakForms::internal::average_of_gradients<solution_index>(*this);
      }

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      average_of_hessians() const
      {
        return WeakForms::internal::average_of_hessians<solution_index>(*this);
      }
    };


    template <typename SpaceType_>
    class Vector final
      : public SubSpaceViewBase<SpaceType_, FEValuesExtractors::Vector>
    {
      DEAL_II_SUBSPACE_VIEW_COMMON_IMPL(Vector,
                                        SpaceType_,
                                        FEValuesExtractors::Vector)

    public:
      template <typename ScalarType>
      using value_type =
        typename FEValuesViewsType::template solution_value_type<ScalarType>;

      template <typename ScalarType>
      using gradient_type =
        typename FEValuesViewsType::template solution_gradient_type<ScalarType>;

      template <typename ScalarType>
      using symmetric_gradient_type =
        typename FEValuesViewsType::template solution_symmetric_gradient_type<
          ScalarType>;

      template <typename ScalarType>
      using divergence_type =
        typename FEValuesViewsType::template solution_divergence_type<
          ScalarType>;

      template <typename ScalarType>
      using curl_type =
        typename FEValuesViewsType::template solution_curl_type<ScalarType>;

      template <typename ScalarType>
      using hessian_type =
        typename FEValuesViewsType::template solution_hessian_type<ScalarType>;

      template <typename ScalarType>
      using third_derivative_type =
        typename FEValuesViewsType::template solution_third_derivative_type<
          ScalarType>;

      // Methods to promote this class to a SymbolicOp:
      // Operators: Test functions, trial solutions, and field solutions

      auto
      value() const
      {
        return WeakForms::internal::value(*this);
      }

      auto
      gradient() const
      {
        return WeakForms::internal::gradient(*this);
      }

      auto
      symmetric_gradient() const
      {
        return WeakForms::internal::symmetric_gradient(*this);
      }

      auto
      divergence() const
      {
        return WeakForms::internal::divergence(*this);
      }

      auto
      curl() const
      {
        return WeakForms::internal::curl(*this);
      }

      auto
      hessian() const
      {
        return WeakForms::internal::hessian(*this);
      }

      auto
      third_derivative() const
      {
        return WeakForms::internal::third_derivative(*this);
      }

      // Operators: Field solutions only

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      value() const
      {
        return WeakForms::internal::value<solution_index>(*this);
      }

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      gradient() const
      {
        return WeakForms::internal::gradient<solution_index>(*this);
      }

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      symmetric_gradient() const
      {
        return WeakForms::internal::symmetric_gradient<solution_index>(*this);
      }

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      divergence() const
      {
        return WeakForms::internal::divergence<solution_index>(*this);
      }

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      curl() const
      {
        return WeakForms::internal::curl<solution_index>(*this);
      }

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      hessian() const
      {
        return WeakForms::internal::hessian<solution_index>(*this);
      }

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      third_derivative() const
      {
        return WeakForms::internal::third_derivative<solution_index>(*this);
      }

      // Methods to promote this class to a SymbolicOp:
      // Interface

      auto
      jump_in_values() const
      {
        return WeakForms::internal::jump_in_values(*this);
      }

      auto
      jump_in_gradients() const
      {
        return WeakForms::internal::jump_in_gradients(*this);
      }

      auto
      jump_in_hessians() const
      {
        return WeakForms::internal::jump_in_hessians(*this);
      }

      auto
      jump_in_third_derivatives() const
      {
        return WeakForms::internal::jump_in_third_derivatives(*this);
      }

      auto
      average_of_values() const
      {
        return WeakForms::internal::average_of_values(*this);
      }

      auto
      average_of_gradients() const
      {
        return WeakForms::internal::average_of_gradients(*this);
      }

      auto
      average_of_hessians() const
      {
        return WeakForms::internal::average_of_hessians(*this);
      }

      // Operators: Field solutions only (interface)

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      jump_in_values() const
      {
        return WeakForms::internal::jump_in_values<solution_index>(*this);
      }

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      jump_in_gradients() const
      {
        return WeakForms::internal::jump_in_gradients<solution_index>(*this);
      }

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      jump_in_hessians() const
      {
        return WeakForms::internal::jump_in_hessians<solution_index>(*this);
      }

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      jump_in_third_derivatives() const
      {
        return WeakForms::internal::jump_in_third_derivatives<solution_index>(
          *this);
      }

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      average_of_values() const
      {
        return WeakForms::internal::average_of_values<solution_index>(*this);
      }

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      average_of_gradients() const
      {
        return WeakForms::internal::average_of_gradients<solution_index>(*this);
      }

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      average_of_hessians() const
      {
        return WeakForms::internal::average_of_hessians<solution_index>(*this);
      }
    };


    template <int rank_, typename SpaceType_>
    class Tensor final
      : public SubSpaceViewBase<SpaceType_, FEValuesExtractors::Tensor<rank_>>
    {
      DEAL_II_SUBSPACE_VIEW_COMMON_IMPL(Tensor,
                                        SpaceType_,
                                        FEValuesExtractors::Tensor<rank_>)

    public:
      template <typename ScalarType>
      using value_type =
        typename FEValuesViewsType::template solution_value_type<ScalarType>;

      template <typename ScalarType>
      using gradient_type =
        typename FEValuesViewsType::template solution_gradient_type<ScalarType>;

      template <typename ScalarType>
      using divergence_type =
        typename FEValuesViewsType::template solution_divergence_type<
          ScalarType>;

      // Methods to promote this class to a SymbolicOp:
      // Operators: Test functions, trial solutions, and field solutions

      auto
      value() const
      {
        return WeakForms::internal::value(*this);
      }

      auto
      gradient() const
      {
        return WeakForms::internal::gradient(*this);
      }

      auto
      divergence() const
      {
        return WeakForms::internal::divergence(*this);
      }

      // Operators: Field solutions only

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      value() const
      {
        return WeakForms::internal::value<solution_index>(*this);
      }

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      gradient() const
      {
        return WeakForms::internal::gradient<solution_index>(*this);
      }

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      divergence() const
      {
        return WeakForms::internal::divergence<solution_index>(*this);
      }

      // Interfaces
      // - Not yet implemented in deal.II
    };


    template <int rank_, typename SpaceType_>
    class SymmetricTensor final
      : public SubSpaceViewBase<SpaceType_,
                                FEValuesExtractors::SymmetricTensor<rank_>>
    {
      DEAL_II_SUBSPACE_VIEW_COMMON_IMPL(
        SymmetricTensor,
        SpaceType_,
        FEValuesExtractors::SymmetricTensor<rank_>)

    public:
      template <typename ScalarType>
      using value_type =
        typename FEValuesViewsType::template solution_value_type<ScalarType>;

      template <typename ScalarType>
      using divergence_type =
        typename FEValuesViewsType::template solution_divergence_type<
          ScalarType>;

      // Methods to promote this class to a SymbolicOp:
      // Operators: Test functions, trial solutions, and field solutions

      auto
      value() const
      {
        return WeakForms::internal::value(*this);
      }

      auto
      divergence() const
      {
        return WeakForms::internal::divergence(*this);
      }

      // Operators: Field solutions only

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      value() const
      {
        return WeakForms::internal::value<solution_index>(*this);
      }

      template <
        types::solution_index solution_index = 0,
        typename T                           = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      divergence() const
      {
        return WeakForms::internal::divergence<solution_index>(*this);
      }

      // Interfaces
      // - Not yet implemented in deal.II
    };

#undef DEAL_II_SUBSPACE_VIEW_COMMON_IMPL

  } // namespace SubSpaceViews

} // namespace WeakForms



/* ================== Specialization of unary operators ================== */



namespace WeakForms
{
  namespace Operators
  {
    namespace internal
    {
      /**
       * A struct that holds the type of extractor that corresponds to the
       * return type of a specific operation that is performed on a space.
       *
       * @tparam SubSpaceViewsType The space on which the operation is performed.
       * @tparam SymbolicOpCodes The operation that is to be performed.
       */
      template <typename SubSpaceViewsType, enum SymbolicOpCodes>
      struct SymbolicOpExtractor;


      // Scalar
      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Scalar<SpaceType>,
                                 SymbolicOpCodes::value>
      {
        using type = FEValuesExtractors::Scalar;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Scalar<SpaceType>,
                                 SymbolicOpCodes::gradient>
      {
        using type = FEValuesExtractors::Vector;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Scalar<SpaceType>,
                                 SymbolicOpCodes::hessian>
      {
        using type = FEValuesExtractors::Tensor<2>;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Scalar<SpaceType>,
                                 SymbolicOpCodes::laplacian>
      {
        using type = FEValuesExtractors::Scalar;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Scalar<SpaceType>,
                                 SymbolicOpCodes::third_derivative>
      {
        using type = FEValuesExtractors::Tensor<3>;
      };


      // Vector
      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Vector<SpaceType>,
                                 SymbolicOpCodes::value>
      {
        using type = FEValuesExtractors::Vector;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Vector<SpaceType>,
                                 SymbolicOpCodes::gradient>
      {
        using type = FEValuesExtractors::Tensor<2>;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Vector<SpaceType>,
                                 SymbolicOpCodes::symmetric_gradient>
      {
        using type = FEValuesExtractors::SymmetricTensor<2>;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Vector<SpaceType>,
                                 SymbolicOpCodes::divergence>
      {
        using type = FEValuesExtractors::Scalar;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Vector<SpaceType>,
                                 SymbolicOpCodes::curl>
      {
        using type = FEValuesExtractors::Vector;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Vector<SpaceType>,
                                 SymbolicOpCodes::hessian>
      {
        using type = FEValuesExtractors::Tensor<3>;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Vector<SpaceType>,
                                 SymbolicOpCodes::third_derivative>
      {
        using type = FEValuesExtractors::Tensor<4>;
      };


      // Tensor
      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Tensor<2, SpaceType>,
                                 SymbolicOpCodes::value>
      {
        using type = FEValuesExtractors::Tensor<2>;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Tensor<2, SpaceType>,
                                 SymbolicOpCodes::divergence>
      {
        using type = FEValuesExtractors::Vector;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Tensor<2, SpaceType>,
                                 SymbolicOpCodes::gradient>
      {
        using type = FEValuesExtractors::Tensor<3>;
      };


      // Symmetric Tensor
      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::SymmetricTensor<2, SpaceType>,
                                 SymbolicOpCodes::value>
      {
        using type = FEValuesExtractors::SymmetricTensor<2>;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::SymmetricTensor<2, SpaceType>,
                                 SymbolicOpCodes::divergence>
      {
        using type = FEValuesExtractors::Vector;
      };
    } // namespace internal


    /* ---- Finite element spaces: Test functions and trial solutions ---- */


/**
 * A macro to implement the common parts of a symbolic op class
 * for test functions and trial solution subspaces.
 * It is expected that the unary op derives from a
 * SymbolicOp[TYPE]Base<SubSpaceViewsType> .
 *
 * @note It is intended that this should used immediately after class
 * definition is opened.
 */
#define DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SUBSPACE_COMMON_IMPL(                 \
  SymbolicOpBaseType, SubSpaceViewsType, SymbolicOpCode)                     \
private:                                                                     \
  using Base_t  = SymbolicOpBaseType<SubSpaceViewsType>;                     \
  using View_t  = SubSpaceViewsType;                                         \
  using Space_t = typename View_t::SpaceType;                                \
  using SymbolicOpExtractor_t =                                              \
    internal::SymbolicOpExtractor<SubSpaceViewsType, SymbolicOpCode>;        \
  using typename Base_t::Op;                                                 \
                                                                             \
public:                                                                      \
  /**                                                                        \
   * Dimension in which this object operates.                                \
   */                                                                        \
  static const unsigned int dimension = View_t::dimension;                   \
                                                                             \
  /**                                                                        \
   * Dimension of the subspace in which this object operates.                \
   */                                                                        \
  static const unsigned int space_dimension = View_t::space_dimension;       \
                                                                             \
  template <typename ScalarType>                                             \
  using value_type = typename Base_t::template value_type<ScalarType>;       \
                                                                             \
  template <typename ScalarType>                                             \
  using qp_value_type = typename Base_t::template qp_value_type<ScalarType>; \
                                                                             \
  template <typename ScalarType>                                             \
  using return_type = typename Base_t::template dof_value_type<ScalarType>;  \
                                                                             \
  template <typename ScalarType, std::size_t width>                          \
  using vectorized_value_type =                                              \
    typename Base_t::template vectorized_value_type<ScalarType, width>;      \
                                                                             \
  template <typename ScalarType, std::size_t width>                          \
  using vectorized_qp_value_type =                                           \
    typename Base_t::template vectorized_qp_value_type<ScalarType, width>;   \
                                                                             \
  template <typename ScalarType, std::size_t width>                          \
  using vectorized_return_type =                                             \
    typename Base_t::template vectorized_dof_value_type<ScalarType, width>;  \
                                                                             \
  explicit SymbolicOp(const Op &operand)                                     \
    : Base_t(operand)                                                        \
  {}                                                                         \
                                                                             \
  /**                                                                        \
   * Return all shape function values all quadrature points.                 \
   *                                                                         \
   * The outer index is the shape function, and the inner index              \
   * is the quadrature point.                                                \
   *                                                                         \
   * @tparam ScalarType                                                      \
   * @param fe_values_dofs                                                   \
   * @param fe_values_op                                                     \
   * @return return_type<ScalarType>                                         \
   */                                                                        \
  template <typename ScalarType>                                             \
  return_type<ScalarType> operator()(                                        \
    const FEValuesBase<dimension, space_dimension> &fe_values_dofs,          \
    const FEValuesBase<dimension, space_dimension> &fe_values_op) const      \
  {                                                                          \
    return_type<ScalarType> out(fe_values_dofs.dofs_per_cell);               \
                                                                             \
    for (const auto dof_index : fe_values_dofs.dof_indices())                \
      {                                                                      \
        out[dof_index].reserve(fe_values_op.n_quadrature_points);            \
                                                                             \
        for (const auto q_point : fe_values_op.quadrature_point_indices())   \
          out[dof_index].emplace_back(this->template operator()<ScalarType>( \
            fe_values_op, dof_index, q_point));                              \
      }                                                                      \
                                                                             \
    return out;                                                              \
  }                                                                          \
                                                                             \
  template <typename ScalarType, std::size_t width>                          \
  vectorized_return_type<ScalarType, width> operator()(                      \
    const FEValuesBase<dimension, space_dimension> &fe_values_dofs,          \
    const FEValuesBase<dimension, space_dimension> &fe_values_op,            \
    const types::vectorized_qp_range_t &            q_point_range) const                 \
  {                                                                          \
    vectorized_return_type<ScalarType, width> out(                           \
      fe_values_dofs.dofs_per_cell);                                         \
                                                                             \
    Assert(q_point_range.size() <= width,                                    \
           ExcIndexRange(q_point_range.size(), 0, width));                   \
                                                                             \
    for (const auto dof_index : fe_values_dofs.dof_indices())                \
      {                                                                      \
        DEAL_II_OPENMP_SIMD_PRAGMA                                           \
        for (unsigned int i = 0; i < q_point_range.size(); ++i)              \
          numbers::set_vectorized_values(                                    \
            out[dof_index],                                                  \
            i,                                                               \
            this->template operator()<ScalarType>(fe_values_op,              \
                                                  dof_index,                 \
                                                  q_point_range[i]));        \
      }                                                                      \
                                                                             \
    return out;                                                              \
  }                                                                          \
                                                                             \
protected:                                                                   \
  /**                                                                        \
   * The extractor corresponding to the view itself                          \
   */                                                                        \
  using view_extractor_type = typename View_t::extractor_type;               \
                                                                             \
  const view_extractor_type &get_extractor() const                           \
  {                                                                          \
    return this->get_operand().get_extractor();                              \
  }


    /**
     * Extract the shape function values from a finite element subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType,
     *         e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     */
    template <typename SubSpaceViewsType>
    class SymbolicOp<
      SubSpaceViewsType,
      SymbolicOpCodes::value,
      typename std::enable_if<
        is_test_function<typename SubSpaceViewsType::SpaceType>::value ||
        is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public SymbolicOpValueBase<SubSpaceViewsType>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SUBSPACE_COMMON_IMPL(
        SymbolicOpValueBase,
        SubSpaceViewsType,
        SymbolicOpCodes::value)

    protected:
      // Return single entry
      template <typename ScalarType>
      value_type<ScalarType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                              dof_index,
                 const unsigned int                              q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values[get_extractor()].value(dof_index, q_point);
      }
    };



    /**
     * Extract the shape function gradients from a finite element subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     */
    template <typename SubSpaceViewsType>
    class SymbolicOp<
      SubSpaceViewsType,
      SymbolicOpCodes::gradient,
      typename std::enable_if<
        is_test_function<typename SubSpaceViewsType::SpaceType>::value ||
        is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public SymbolicOpGradientBase<SubSpaceViewsType>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SUBSPACE_COMMON_IMPL(
        SymbolicOpGradientBase,
        SubSpaceViewsType,
        SymbolicOpCodes::gradient)

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<View_t, SubSpaceViews::Scalar<Space_t>>::value ||
          std::is_same<View_t, SubSpaceViews::Vector<Space_t>>::value ||
          std::is_same<View_t,
                       SubSpaceViews::Tensor<View_t::rank, Space_t>>::value,
        "The selected subspace view does not support the gradient operation.");

    protected:
      // Return single entry
      template <typename ScalarType>
      value_type<ScalarType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                              dof_index,
                 const unsigned int                              q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values[get_extractor()].gradient(dof_index, q_point);
      }
    };



    /**
     * Extract the shape function symmetric gradients from a finite element
     * subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     */
    template <typename SubSpaceViewsType>
    class SymbolicOp<
      SubSpaceViewsType,
      SymbolicOpCodes::symmetric_gradient,
      typename std::enable_if<
        is_test_function<typename SubSpaceViewsType::SpaceType>::value ||
        is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public SymbolicOpSymmetricGradientBase<SubSpaceViewsType>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SUBSPACE_COMMON_IMPL(
        SymbolicOpSymmetricGradientBase,
        SubSpaceViewsType,
        SymbolicOpCodes::symmetric_gradient)

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<View_t, SubSpaceViews::Vector<Space_t>>::value,
        "The selected subspace view does not support the symmetric gradient operation.");

    protected:
      // Return single entry
      template <typename ScalarType>
      value_type<ScalarType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                              dof_index,
                 const unsigned int                              q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values[get_extractor()].symmetric_gradient(dof_index,
                                                             q_point);
      }
    };



    /**
     * Extract the shape function divergences from a finite element subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Vector
     * @tparam SpaceType A space type, specifically a test space or trial space
     */
    template <typename SubSpaceViewsType>
    class SymbolicOp<
      SubSpaceViewsType,
      SymbolicOpCodes::divergence,
      typename std::enable_if<
        is_test_function<typename SubSpaceViewsType::SpaceType>::value ||
        is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public SymbolicOpDivergenceBase<SubSpaceViewsType>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SUBSPACE_COMMON_IMPL(
        SymbolicOpDivergenceBase,
        SubSpaceViewsType,
        SymbolicOpCodes::divergence)

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<View_t, SubSpaceViews::Vector<Space_t>>::value ||
          std::is_same<View_t,
                       SubSpaceViews::Tensor<View_t::rank, Space_t>>::value ||
          std::is_same<
            View_t,
            SubSpaceViews::SymmetricTensor<View_t::rank, Space_t>>::value,
        "The selected subspace view does not support the divergence operation.");

    protected:
      // Return single entry
      template <typename ScalarType>
      value_type<ScalarType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                              dof_index,
                 const unsigned int                              q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values[get_extractor()].divergence(dof_index, q_point);
      }
    };



    /**
     * Extract the shape function divergences from a finite element subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Vector
     * @tparam SpaceType A space type, specifically a test space or trial space
     */
    template <typename SubSpaceViewsType>
    class SymbolicOp<
      SubSpaceViewsType,
      SymbolicOpCodes::curl,
      typename std::enable_if<
        is_test_function<typename SubSpaceViewsType::SpaceType>::value ||
        is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public SymbolicOpCurlBase<SubSpaceViewsType>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SUBSPACE_COMMON_IMPL(SymbolicOpCurlBase,
                                                          SubSpaceViewsType,
                                                          SymbolicOpCodes::curl)

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<View_t, SubSpaceViews::Vector<Space_t>>::value,
        "The selected subspace view does not support the curls operation.");

    protected:
      // Return single entry
      template <typename ScalarType>
      value_type<ScalarType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                              dof_index,
                 const unsigned int                              q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values[get_extractor()].curl(dof_index, q_point);
      }
    };



    /**
     * Extract the shape function Laplacians from a finite element subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Vector
     * @tparam SpaceType A space type, specifically a test space or trial space
     */
    template <typename SubSpaceViewsType>
    class SymbolicOp<
      SubSpaceViewsType,
      SymbolicOpCodes::laplacian,
      typename std::enable_if<
        is_test_function<typename SubSpaceViewsType::SpaceType>::value ||
        is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public SymbolicOpLaplacianBase<SubSpaceViewsType>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SUBSPACE_COMMON_IMPL(
        SymbolicOpLaplacianBase,
        SubSpaceViewsType,
        SymbolicOpCodes::laplacian)

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<View_t, SubSpaceViews::Scalar<Space_t>>::value,
        "The selected subspace view does not support the Laplacian operation.");

    protected:
      // Return single entry
      template <typename ScalarType>
      value_type<ScalarType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                              dof_index,
                 const unsigned int                              q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return trace(fe_values[get_extractor()].hessian(dof_index, q_point));
      }
    };



    /**
     * Extract the shape function Hessians from a finite element subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Vector
     * @tparam SpaceType A space type, specifically a test space or trial space
     */
    template <typename SubSpaceViewsType>
    class SymbolicOp<
      SubSpaceViewsType,
      SymbolicOpCodes::hessian,
      typename std::enable_if<
        is_test_function<typename SubSpaceViewsType::SpaceType>::value ||
        is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public SymbolicOpHessianBase<SubSpaceViewsType>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SUBSPACE_COMMON_IMPL(
        SymbolicOpHessianBase,
        SubSpaceViewsType,
        SymbolicOpCodes::hessian)

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<View_t, SubSpaceViews::Scalar<Space_t>>::value ||
          std::is_same<View_t, SubSpaceViews::Vector<Space_t>>::value,
        "The selected subspace view does not support the Hessian operation.");

    protected:
      // Return single entry
      template <typename ScalarType>
      value_type<ScalarType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                              dof_index,
                 const unsigned int                              q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values[get_extractor()].hessian(dof_index, q_point);
      }
    };



    /**
     * Extract the shape function third derivatives from a finite element
     * subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Vector
     * @tparam SpaceType A space type, specifically a test space or trial space
     */
    template <typename SubSpaceViewsType>
    class SymbolicOp<
      SubSpaceViewsType,
      SymbolicOpCodes::third_derivative,
      typename std::enable_if<
        is_test_function<typename SubSpaceViewsType::SpaceType>::value ||
        is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public SymbolicOpThirdDerivativeBase<SubSpaceViewsType>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SUBSPACE_COMMON_IMPL(
        SymbolicOpThirdDerivativeBase,
        SubSpaceViewsType,
        SymbolicOpCodes::third_derivative)

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<View_t, SubSpaceViews::Scalar<Space_t>>::value ||
          std::is_same<View_t, SubSpaceViews::Vector<Space_t>>::value,
        "The selected subspace view does not support the third derivative operation.");

    protected:
      // Return single entry
      template <typename ScalarType>
      value_type<ScalarType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                              dof_index,
                 const unsigned int                              q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values[get_extractor()].third_derivative(dof_index, q_point);
      }
    };


#undef DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SUBSPACE_COMMON_IMPL


    /* -- Finite element spaces: Test functions and trial solutions (interfaces)
     * -- */


/**
 * A macro to implement the common parts of a symbolic op class
 * for test functions and trial solution subspaces.
 * It is expected that the unary op derives from a
 * SymbolicOp[TYPE]Base<SubSpaceViewsType> .
 *
 * @note It is intended that this should used immediately after class
 * definition is opened.
 */
#define DEAL_II_SYMBOLIC_OP_TEST_TRIAL_INTERFACE_SUBSPACE_COMMON_IMPL(                     \
  SymbolicOpBaseType, SubSpaceViewsType, SymbolicOpCode)                                   \
private:                                                                                   \
  using Base_t  = SymbolicOpBaseType<SubSpaceViewsType>;                                   \
  using View_t  = SubSpaceViewsType;                                                       \
  using Space_t = typename View_t::SpaceType;                                              \
  using SymbolicOpExtractor_t =                                                            \
    internal::SymbolicOpExtractor<SubSpaceViewsType, SymbolicOpCode>;                      \
  using typename Base_t::Op;                                                               \
                                                                                           \
public:                                                                                    \
  /**                                                                                      \
   * Dimension in which this object operates.                                              \
   */                                                                                      \
  static const unsigned int dimension = View_t::dimension;                                 \
                                                                                           \
  /**                                                                                      \
   * Dimension of the subspace in which this object operates.                              \
   */                                                                                      \
  static const unsigned int space_dimension = View_t::space_dimension;                     \
                                                                                           \
  template <typename ScalarType>                                                           \
  using value_type = typename Base_t::template value_type<ScalarType>;                     \
                                                                                           \
  template <typename ScalarType>                                                           \
  using qp_value_type = typename Base_t::template qp_value_type<ScalarType>;               \
                                                                                           \
  template <typename ScalarType>                                                           \
  using return_type = typename Base_t::template dof_value_type<ScalarType>;                \
                                                                                           \
  template <typename ScalarType, std::size_t width>                                        \
  using vectorized_value_type =                                                            \
    typename Base_t::template vectorized_value_type<ScalarType, width>;                    \
                                                                                           \
  template <typename ScalarType, std::size_t width>                                        \
  using vectorized_return_type =                                                           \
    typename Base_t::template vectorized_dof_value_type<ScalarType, width>;                \
                                                                                           \
  explicit SymbolicOp(const Op &operand)                                                   \
    : Base_t(operand)                                                                      \
  {}                                                                                       \
                                                                                           \
  /**                                                                                      \
   * Return all shape function values all quadrature points.                               \
   *                                                                                       \
   * The outer index is the shape function, and the inner index                            \
   * is the quadrature point.                                                              \
   *                                                                                       \
   * @tparam ScalarType                                                                    \
   * @param fe_values_dofs                                                                 \
   * @param fe_values_op                                                                   \
   * @return return_type<ScalarType>                                                       \
   */                                                                                      \
  template <typename ScalarType>                                                           \
  return_type<ScalarType> operator()(                                                      \
    const FEInterfaceValues<dimension, space_dimension> &fe_interface_values)              \
    const                                                                                  \
  {                                                                                        \
    return_type<ScalarType> out(                                                           \
      fe_interface_values.n_current_interface_dofs());                                     \
                                                                                           \
    for (const auto interface_dof_index : fe_interface_values.dof_indices())               \
      {                                                                                    \
        out[interface_dof_index].reserve(                                                  \
          fe_interface_values.n_quadrature_points);                                        \
                                                                                           \
        for (const auto q_point :                                                          \
             fe_interface_values.quadrature_point_indices())                               \
          out[interface_dof_index].emplace_back(                                           \
            this->template operator()<ScalarType>(fe_interface_values,                     \
                                                  interface_dof_index,                     \
                                                  q_point));                               \
      }                                                                                    \
                                                                                           \
    return out;                                                                            \
  }                                                                                        \
                                                                                           \
  template <typename ScalarType>                                                           \
  return_type<ScalarType> operator()(                                                      \
    const FEInterfaceValues<dimension, space_dimension>                                    \
      &fe_interface_values_dofs,                                                           \
    const FEInterfaceValues<dimension, space_dimension>                                    \
      &fe_interface_values_op) const                                                       \
  {                                                                                        \
    Assert(                                                                                \
      &fe_interface_values_dofs == &fe_interface_values_op,                                \
      ExcMessage(                                                                          \
        "Expected exactly the same FEInterfaceValues object for the DoFs and Operator.")); \
    (void)fe_interface_values_op;                                                          \
    return this->template operator()<ScalarType>(fe_interface_values_dofs);                \
  }                                                                                        \
                                                                                           \
  template <typename ScalarType, std::size_t width>                                        \
  vectorized_return_type<ScalarType, width> operator()(                                    \
    const FEInterfaceValues<dimension, space_dimension> &fe_interface_values,              \
    const types::vectorized_qp_range_t &                 q_point_range) const                               \
  {                                                                                        \
    vectorized_return_type<ScalarType, width> out(                                         \
      fe_interface_values.n_current_interface_dofs());                                     \
                                                                                           \
    Assert(q_point_range.size() <= width,                                                  \
           ExcIndexRange(q_point_range.size(), 0, width));                                 \
                                                                                           \
    for (const auto interface_dof_index : fe_interface_values.dof_indices())               \
      {                                                                                    \
        DEAL_II_OPENMP_SIMD_PRAGMA                                                         \
        for (unsigned int i = 0; i < q_point_range.size(); ++i)                            \
          numbers::set_vectorized_values(                                                  \
            out[interface_dof_index],                                                      \
            i,                                                                             \
            this->template operator()<ScalarType>(fe_interface_values,                     \
                                                  interface_dof_index,                     \
                                                  q_point_range[i]));                      \
      }                                                                                    \
                                                                                           \
    return out;                                                                            \
  }                                                                                        \
                                                                                           \
  template <typename ScalarType, std::size_t width>                                        \
  vectorized_return_type<ScalarType, width> operator()(                                    \
    const FEInterfaceValues<dimension, space_dimension>                                    \
      &fe_interface_values_dofs,                                                           \
    const FEInterfaceValues<dimension, space_dimension>                                    \
      &                                 fe_interface_values_op,                            \
    const types::vectorized_qp_range_t &q_point_range) const                               \
  {                                                                                        \
    Assert(                                                                                \
      &fe_interface_values_dofs == &fe_interface_values_op,                                \
      ExcMessage(                                                                          \
        "Expected exactly the same FEInterfaceValues object for the DoFs and Operator.")); \
    (void)fe_interface_values_op;                                                          \
    return this->template operator()<ScalarType, width>(                                   \
      fe_interface_values_dofs, q_point_range);                                            \
  }                                                                                        \
                                                                                           \
                                                                                           \
protected:                                                                                 \
  /**                                                                                      \
   * The extractor corresponding to the view itself                                        \
   */                                                                                      \
  using view_extractor_type = typename View_t::extractor_type;                             \
                                                                                           \
  const view_extractor_type &get_extractor() const                                         \
  {                                                                                        \
    return this->get_operand().get_extractor();                                            \
  }


    /**
     * Extract the jump in shape function values from a finite element subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType,
     *         e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     */
    template <typename SubSpaceViewsType>
    class SymbolicOp<
      SubSpaceViewsType,
      SymbolicOpCodes::jump_in_values,
      typename std::enable_if<
        is_test_function<typename SubSpaceViewsType::SpaceType>::value ||
        is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public SymbolicOpJumpValueBase<SubSpaceViewsType>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_INTERFACE_SUBSPACE_COMMON_IMPL(
        SymbolicOpJumpValueBase,
        SubSpaceViewsType,
        SymbolicOpCodes::jump_in_values)

    protected:
      // Return single entry
      template <typename ScalarType, int dim, int spacedim>
      value_type<ScalarType>
      operator()(const FEInterfaceValues<dim, spacedim> &fe_interface_values,
                 const unsigned int                      interface_dof_index,
                 const unsigned int                      q_point) const
      {
        Assert(interface_dof_index <
                 fe_interface_values.n_current_interface_dofs(),
               ExcIndexRange(interface_dof_index,
                             0,
                             fe_interface_values.n_current_interface_dofs()));
        Assert(q_point < fe_interface_values.n_quadrature_points,
               ExcIndexRange(q_point,
                             0,
                             fe_interface_values.n_quadrature_points));

        return fe_interface_values[get_extractor()].jump_in_values(
          interface_dof_index, q_point);
      }
    };


    /**
     * Extract the jump in shape function gradients from a finite element
     * subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType,
     *         e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     */
    template <typename SubSpaceViewsType>
    class SymbolicOp<
      SubSpaceViewsType,
      SymbolicOpCodes::jump_in_gradients,
      typename std::enable_if<
        is_test_function<typename SubSpaceViewsType::SpaceType>::value ||
        is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public SymbolicOpJumpGradientBase<SubSpaceViewsType>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_INTERFACE_SUBSPACE_COMMON_IMPL(
        SymbolicOpJumpGradientBase,
        SubSpaceViewsType,
        SymbolicOpCodes::jump_in_gradients)

    protected:
      // Return single entry
      template <typename ScalarType, int dim, int spacedim>
      value_type<ScalarType>
      operator()(const FEInterfaceValues<dim, spacedim> &fe_interface_values,
                 const unsigned int                      interface_dof_index,
                 const unsigned int                      q_point) const
      {
        Assert(interface_dof_index <
                 fe_interface_values.n_current_interface_dofs(),
               ExcIndexRange(interface_dof_index,
                             0,
                             fe_interface_values.n_current_interface_dofs()));
        Assert(q_point < fe_interface_values.n_quadrature_points,
               ExcIndexRange(q_point,
                             0,
                             fe_interface_values.n_quadrature_points));

        return fe_interface_values[get_extractor()].jump_in_gradients(
          interface_dof_index, q_point);
      }
    };


    /**
     * Extract the jump in shape function Hessians from a finite element
     * subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType,
     *         e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     */
    template <typename SubSpaceViewsType>
    class SymbolicOp<
      SubSpaceViewsType,
      SymbolicOpCodes::jump_in_hessians,
      typename std::enable_if<
        is_test_function<typename SubSpaceViewsType::SpaceType>::value ||
        is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public SymbolicOpJumpHessianBase<SubSpaceViewsType>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_INTERFACE_SUBSPACE_COMMON_IMPL(
        SymbolicOpJumpHessianBase,
        SubSpaceViewsType,
        SymbolicOpCodes::jump_in_hessians)

    protected:
      // Return single entry
      template <typename ScalarType, int dim, int spacedim>
      value_type<ScalarType>
      operator()(const FEInterfaceValues<dim, spacedim> &fe_interface_values,
                 const unsigned int                      interface_dof_index,
                 const unsigned int                      q_point) const
      {
        Assert(interface_dof_index <
                 fe_interface_values.n_current_interface_dofs(),
               ExcIndexRange(interface_dof_index,
                             0,
                             fe_interface_values.n_current_interface_dofs()));
        Assert(q_point < fe_interface_values.n_quadrature_points,
               ExcIndexRange(q_point,
                             0,
                             fe_interface_values.n_quadrature_points));

        return fe_interface_values[get_extractor()].jump_in_hessians(
          interface_dof_index, q_point);
      }
    };


    /**
     * Extract the jump in shape function third derivatives from a finite
     * element subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType,
     *         e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     */
    template <typename SubSpaceViewsType>
    class SymbolicOp<
      SubSpaceViewsType,
      SymbolicOpCodes::jump_in_third_derivatives,
      typename std::enable_if<
        is_test_function<typename SubSpaceViewsType::SpaceType>::value ||
        is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public SymbolicOpJumpThirdDerivativeBase<SubSpaceViewsType>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_INTERFACE_SUBSPACE_COMMON_IMPL(
        SymbolicOpJumpThirdDerivativeBase,
        SubSpaceViewsType,
        SymbolicOpCodes::jump_in_third_derivatives)

    protected:
      // Return single entry
      template <typename ScalarType, int dim, int spacedim>
      value_type<ScalarType>
      operator()(const FEInterfaceValues<dim, spacedim> &fe_interface_values,
                 const unsigned int                      interface_dof_index,
                 const unsigned int                      q_point) const
      {
        Assert(interface_dof_index <
                 fe_interface_values.n_current_interface_dofs(),
               ExcIndexRange(interface_dof_index,
                             0,
                             fe_interface_values.n_current_interface_dofs()));
        Assert(q_point < fe_interface_values.n_quadrature_points,
               ExcIndexRange(q_point,
                             0,
                             fe_interface_values.n_quadrature_points));

        return fe_interface_values[get_extractor()].jump_in_third_derivatives(
          interface_dof_index, q_point);
      }
    };



    /**
     * Extract the average of shape function values from a finite element
     * subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType,
     *         e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     */
    template <typename SubSpaceViewsType>
    class SymbolicOp<
      SubSpaceViewsType,
      SymbolicOpCodes::average_of_values,
      typename std::enable_if<
        is_test_function<typename SubSpaceViewsType::SpaceType>::value ||
        is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public SymbolicOpAverageValueBase<SubSpaceViewsType>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_INTERFACE_SUBSPACE_COMMON_IMPL(
        SymbolicOpAverageValueBase,
        SubSpaceViewsType,
        SymbolicOpCodes::average_of_values)

    protected:
      // Return single entry
      template <typename ScalarType, int dim, int spacedim>
      value_type<ScalarType>
      operator()(const FEInterfaceValues<dim, spacedim> &fe_interface_values,
                 const unsigned int                      interface_dof_index,
                 const unsigned int                      q_point) const
      {
        Assert(interface_dof_index <
                 fe_interface_values.n_current_interface_dofs(),
               ExcIndexRange(interface_dof_index,
                             0,
                             fe_interface_values.n_current_interface_dofs()));
        Assert(q_point < fe_interface_values.n_quadrature_points,
               ExcIndexRange(q_point,
                             0,
                             fe_interface_values.n_quadrature_points));

        return fe_interface_values[get_extractor()].average_of_values(
          interface_dof_index, q_point);
      }
    };


    /**
     * Extract the average of shape function gradients from a finite element
     * subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType,
     *         e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     */
    template <typename SubSpaceViewsType>
    class SymbolicOp<
      SubSpaceViewsType,
      SymbolicOpCodes::average_of_gradients,
      typename std::enable_if<
        is_test_function<typename SubSpaceViewsType::SpaceType>::value ||
        is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public SymbolicOpAverageGradientBase<SubSpaceViewsType>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_INTERFACE_SUBSPACE_COMMON_IMPL(
        SymbolicOpAverageGradientBase,
        SubSpaceViewsType,
        SymbolicOpCodes::average_of_gradients)

    protected:
      // Return single entry
      template <typename ScalarType, int dim, int spacedim>
      value_type<ScalarType>
      operator()(const FEInterfaceValues<dim, spacedim> &fe_interface_values,
                 const unsigned int                      interface_dof_index,
                 const unsigned int                      q_point) const
      {
        Assert(interface_dof_index <
                 fe_interface_values.n_current_interface_dofs(),
               ExcIndexRange(interface_dof_index,
                             0,
                             fe_interface_values.n_current_interface_dofs()));
        Assert(q_point < fe_interface_values.n_quadrature_points,
               ExcIndexRange(q_point,
                             0,
                             fe_interface_values.n_quadrature_points));

        return fe_interface_values[get_extractor()].average_of_gradients(
          interface_dof_index, q_point);
      }
    };


    /**
     * Extract the average of shape function Hessians from a finite element
     * subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType,
     *         e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     */
    template <typename SubSpaceViewsType>
    class SymbolicOp<
      SubSpaceViewsType,
      SymbolicOpCodes::average_of_hessians,
      typename std::enable_if<
        is_test_function<typename SubSpaceViewsType::SpaceType>::value ||
        is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public SymbolicOpAverageHessianBase<SubSpaceViewsType>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_INTERFACE_SUBSPACE_COMMON_IMPL(
        SymbolicOpAverageHessianBase,
        SubSpaceViewsType,
        SymbolicOpCodes::average_of_hessians)

    protected:
      // Return single entry
      template <typename ScalarType, int dim, int spacedim>
      value_type<ScalarType>
      operator()(const FEInterfaceValues<dim, spacedim> &fe_interface_values,
                 const unsigned int                      interface_dof_index,
                 const unsigned int                      q_point) const
      {
        Assert(interface_dof_index <
                 fe_interface_values.n_current_interface_dofs(),
               ExcIndexRange(interface_dof_index,
                             0,
                             fe_interface_values.n_current_interface_dofs()));
        Assert(q_point < fe_interface_values.n_quadrature_points,
               ExcIndexRange(q_point,
                             0,
                             fe_interface_values.n_quadrature_points));

        return fe_interface_values[get_extractor()].average_of_hessians(
          interface_dof_index, q_point);
      }
    };


#undef DEAL_II_SYMBOLIC_OP_TEST_TRIAL_INTERFACE_SUBSPACE_COMMON_IMPL


    /* ------------ Finite element spaces: Solution fields ------------ */


#define DEAL_II_SYMBOLIC_OP_FIELD_SOLUTION_SUBSPACE_COMMON_IMPL(               \
  SymbolicOpBaseType, SubSpaceViewsType, solution_index, SymbolicOpCode)       \
private:                                                                       \
  using Base_t = SymbolicOpBaseType<SubSpaceViewsType, solution_index>;        \
  using View_t = SubSpaceViewsType;                                            \
  using SymbolicOpExtractor_t =                                                \
    typename internal::SymbolicOpExtractor<SubSpaceViewsType, SymbolicOpCode>; \
  using typename Base_t::Op;                                                   \
                                                                               \
public:                                                                        \
  /**                                                                          \
   * Dimension in which this object operates.                                  \
   */                                                                          \
  static const unsigned int dimension = View_t::dimension;                     \
                                                                               \
  /**                                                                          \
   * Dimension of the subspace in which this object operates.                  \
   */                                                                          \
  static const unsigned int space_dimension = View_t::space_dimension;         \
                                                                               \
  template <typename ScalarType>                                               \
  using value_type = typename Base_t::template value_type<ScalarType>;         \
                                                                               \
  template <typename ScalarType>                                               \
  using return_type = typename Base_t::template qp_value_type<ScalarType>;     \
                                                                               \
  template <typename ScalarType, std::size_t width>                            \
  using vectorized_return_type =                                               \
    typename Base_t::template vectorized_qp_value_type<ScalarType, width>;     \
                                                                               \
  explicit SymbolicOp(const Op &operand)                                       \
    : Base_t(operand)                                                          \
  {}                                                                           \
                                                                               \
  template <typename ScalarType, std::size_t width>                            \
  vectorized_return_type<ScalarType, width> operator()(                        \
    MeshWorker::ScratchData<dimension, space_dimension> &scratch_data,         \
    const std::vector<SolutionExtractionData<dimension, space_dimension>>      \
      &                                 solution_extraction_data,              \
    const types::vectorized_qp_range_t &q_point_range) const                   \
  {                                                                            \
    vectorized_return_type<ScalarType, width> out;                             \
    Assert(q_point_range.size() <= width,                                      \
           ExcIndexRange(q_point_range.size(), 0, width));                     \
                                                                               \
    DEAL_II_OPENMP_SIMD_PRAGMA                                                 \
    for (unsigned int i = 0; i < q_point_range.size(); ++i)                    \
      numbers::set_vectorized_values(                                          \
        out,                                                                   \
        i,                                                                     \
        this->template operator()<ScalarType>(                                 \
          scratch_data, solution_extraction_data)[q_point_range[i]]);          \
                                                                               \
    return out;                                                                \
  }                                                                            \
                                                                               \
protected:                                                                     \
  /**                                                                          \
   * The extractor corresponding to the view itself                            \
   */                                                                          \
  using view_extractor_type = typename View_t::extractor_type;                 \
                                                                               \
  const view_extractor_type &get_extractor() const                             \
  {                                                                            \
    return this->get_operand().get_extractor();                                \
  }


    /**
     * Extract the solution values from the discretised solution field subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a solution field
     */
    template <typename SubSpaceViewsType, types::solution_index solution_index>
    class SymbolicOp<SubSpaceViewsType,
                     SymbolicOpCodes::value,
                     typename std::enable_if<is_field_solution<
                       typename SubSpaceViewsType::SpaceType>::value>::type,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpValueBase<SubSpaceViewsType, solution_index>
    {
      DEAL_II_SYMBOLIC_OP_FIELD_SOLUTION_SUBSPACE_COMMON_IMPL(
        SymbolicOpValueBase,
        SubSpaceViewsType,
        solution_index,
        SymbolicOpCodes::value)

    public:
      // Return solution values at all quadrature points
      template <typename ScalarType>
      const return_type<ScalarType> &
      operator()(
        MeshWorker::ScratchData<dimension, space_dimension> &scratch_data,
        const std::vector<SolutionExtractionData<dimension, space_dimension>>
          &solution_extraction_data) const
      {
        Assert(solution_index < solution_extraction_data.size(),
               ExcIndexRange(solution_index,
                             0,
                             solution_extraction_data.size()));

        (void)scratch_data;
        if (solution_extraction_data[solution_index].uses_external_dofhandler)
          {
            Assert(&scratch_data != &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }
        else
          {
            Assert(&scratch_data == &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }

        return solution_extraction_data[solution_index]
          .get_scratch_data()
          .template get_values<view_extractor_type, ScalarType>(
            solution_extraction_data[solution_index].solution_name,
            get_extractor());
      }
    };



    /**
     * Extract the solution gradients from the discretised solution field
     * subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a solution field
     */
    template <typename SubSpaceViewsType, types::solution_index solution_index>
    class SymbolicOp<SubSpaceViewsType,
                     SymbolicOpCodes::gradient,
                     typename std::enable_if<is_field_solution<
                       typename SubSpaceViewsType::SpaceType>::value>::type,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpGradientBase<SubSpaceViewsType, solution_index>
    {
      DEAL_II_SYMBOLIC_OP_FIELD_SOLUTION_SUBSPACE_COMMON_IMPL(
        SymbolicOpGradientBase,
        SubSpaceViewsType,
        solution_index,
        SymbolicOpCodes::gradient)

    public:
      // Return solution gradients at all quadrature points
      template <typename ScalarType>
      const return_type<ScalarType> &
      operator()(
        MeshWorker::ScratchData<dimension, space_dimension> &scratch_data,
        const std::vector<SolutionExtractionData<dimension, space_dimension>>
          &solution_extraction_data) const
      {
        Assert(solution_index < solution_extraction_data.size(),
               ExcIndexRange(solution_index,
                             0,
                             solution_extraction_data.size()));

        (void)scratch_data;
        if (solution_extraction_data[solution_index].uses_external_dofhandler)
          {
            Assert(&scratch_data != &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }
        else
          {
            Assert(&scratch_data == &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }

        return solution_extraction_data[solution_index]
          .get_scratch_data()
          .template get_gradients<view_extractor_type, ScalarType>(
            solution_extraction_data[solution_index].solution_name,
            get_extractor());
      }
    };



    /**
     * Extract the solution symmetric gradients from the discretised solution
     * field subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a solution field
     */
    template <typename SubSpaceViewsType, types::solution_index solution_index>
    class SymbolicOp<SubSpaceViewsType,
                     SymbolicOpCodes::symmetric_gradient,
                     typename std::enable_if<is_field_solution<
                       typename SubSpaceViewsType::SpaceType>::value>::type,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpSymmetricGradientBase<SubSpaceViewsType,
                                               solution_index>
    {
      DEAL_II_SYMBOLIC_OP_FIELD_SOLUTION_SUBSPACE_COMMON_IMPL(
        SymbolicOpSymmetricGradientBase,
        SubSpaceViewsType,
        solution_index,
        SymbolicOpCodes::symmetric_gradient)

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<
          View_t,
          SubSpaceViews::Vector<typename SubSpaceViewsType::SpaceType>>::value,
        "The selected subspace view does not support the symmetric gradient operation.");

    public:
      // Return solution symmetric gradients at all quadrature points
      template <typename ScalarType>
      const return_type<ScalarType> &
      operator()(
        MeshWorker::ScratchData<dimension, space_dimension> &scratch_data,
        const std::vector<SolutionExtractionData<dimension, space_dimension>>
          &solution_extraction_data) const
      {
        Assert(solution_index < solution_extraction_data.size(),
               ExcIndexRange(solution_index,
                             0,
                             solution_extraction_data.size()));

        (void)scratch_data;
        if (solution_extraction_data[solution_index].uses_external_dofhandler)
          {
            Assert(&scratch_data != &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }
        else
          {
            Assert(&scratch_data == &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }

        return solution_extraction_data[solution_index]
          .get_scratch_data()
          .template get_symmetric_gradients<view_extractor_type, ScalarType>(
            solution_extraction_data[solution_index].solution_name,
            get_extractor());
      }
    };



    /**
     * Extract the solution divergences from the discretised solution field
     * subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a solution field
     */
    template <typename SubSpaceViewsType, types::solution_index solution_index>
    class SymbolicOp<SubSpaceViewsType,
                     SymbolicOpCodes::divergence,
                     typename std::enable_if<is_field_solution<
                       typename SubSpaceViewsType::SpaceType>::value>::type,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpDivergenceBase<SubSpaceViewsType, solution_index>
    {
      DEAL_II_SYMBOLIC_OP_FIELD_SOLUTION_SUBSPACE_COMMON_IMPL(
        SymbolicOpDivergenceBase,
        SubSpaceViewsType,
        solution_index,
        SymbolicOpCodes::divergence)

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<View_t,
                     SubSpaceViews::Vector<
                       typename SubSpaceViewsType::SpaceType>>::value ||
          std::is_same<View_t,
                       SubSpaceViews::Tensor<
                         View_t::rank,
                         typename SubSpaceViewsType::SpaceType>>::value ||
          std::is_same<View_t,
                       SubSpaceViews::SymmetricTensor<
                         View_t::rank,
                         typename SubSpaceViewsType::SpaceType>>::value,
        "The selected subspace view does not support the divergence operation.");

    public:
      // Return solution divergences at all quadrature points
      template <typename ScalarType>
      const return_type<ScalarType> &
      operator()(
        MeshWorker::ScratchData<dimension, space_dimension> &scratch_data,
        const std::vector<SolutionExtractionData<dimension, space_dimension>>
          &solution_extraction_data) const
      {
        Assert(solution_index < solution_extraction_data.size(),
               ExcIndexRange(solution_index,
                             0,
                             solution_extraction_data.size()));

        (void)scratch_data;
        if (solution_extraction_data[solution_index].uses_external_dofhandler)
          {
            Assert(&scratch_data != &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }
        else
          {
            Assert(&scratch_data == &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }

        return solution_extraction_data[solution_index]
          .get_scratch_data()
          .template get_divergences<view_extractor_type, ScalarType>(
            solution_extraction_data[solution_index].solution_name,
            get_extractor());
      }
    };



    /**
     * Extract the solution curls from the discretised solution field subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a solution field
     */
    template <typename SubSpaceViewsType, types::solution_index solution_index>
    class SymbolicOp<SubSpaceViewsType,
                     SymbolicOpCodes::curl,
                     typename std::enable_if<is_field_solution<
                       typename SubSpaceViewsType::SpaceType>::value>::type,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpCurlBase<SubSpaceViewsType, solution_index>
    {
      DEAL_II_SYMBOLIC_OP_FIELD_SOLUTION_SUBSPACE_COMMON_IMPL(
        SymbolicOpCurlBase,
        SubSpaceViewsType,
        solution_index,
        SymbolicOpCodes::curl)

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<
          View_t,
          SubSpaceViews::Vector<typename SubSpaceViewsType::SpaceType>>::value,
        "The selected subspace view does not support the curl operation.");

      // In dim==2, the curl operation returns a interestingly dimensioned
      // tensor that is not easily compatible with this framework.
      static_assert(
        dimension == 3,
        "The curl operation for the selected subspace view is only implemented in 3d.");

    public:
      // Return solution symmetric gradients at all quadrature points
      template <typename ScalarType>
      const return_type<ScalarType> &
      operator()(
        MeshWorker::ScratchData<dimension, space_dimension> &scratch_data,
        const std::vector<SolutionExtractionData<dimension, space_dimension>>
          &solution_extraction_data) const
      {
        Assert(solution_index < solution_extraction_data.size(),
               ExcIndexRange(solution_index,
                             0,
                             solution_extraction_data.size()));

        (void)scratch_data;
        if (solution_extraction_data[solution_index].uses_external_dofhandler)
          {
            Assert(&scratch_data != &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }
        else
          {
            Assert(&scratch_data == &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }

        return solution_extraction_data[solution_index]
          .get_scratch_data()
          .template get_curls<view_extractor_type, ScalarType>(
            solution_extraction_data[solution_index].solution_name,
            get_extractor());
      }
    };



    /**
     * Extract the solution Laplacians from the discretised solution field
     * subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a solution field
     */
    template <typename SubSpaceViewsType, types::solution_index solution_index>
    class SymbolicOp<SubSpaceViewsType,
                     SymbolicOpCodes::laplacian,
                     typename std::enable_if<is_field_solution<
                       typename SubSpaceViewsType::SpaceType>::value>::type,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpLaplacianBase<SubSpaceViewsType, solution_index>
    {
      DEAL_II_SYMBOLIC_OP_FIELD_SOLUTION_SUBSPACE_COMMON_IMPL(
        SymbolicOpLaplacianBase,
        SubSpaceViewsType,
        solution_index,
        SymbolicOpCodes::laplacian)

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<
          View_t,
          SubSpaceViews::Scalar<typename SubSpaceViewsType::SpaceType>>::value,
        "The selected subspace view does not support the Laplacian operation.");

    public:
      // Return solution Laplacian at all quadrature points
      template <typename ScalarType>
      const return_type<ScalarType> &
      operator()(
        MeshWorker::ScratchData<dimension, space_dimension> &scratch_data,
        const std::vector<SolutionExtractionData<dimension, space_dimension>>
          &solution_extraction_data) const
      {
        Assert(solution_index < solution_extraction_data.size(),
               ExcIndexRange(solution_index,
                             0,
                             solution_extraction_data.size()));

        (void)scratch_data;
        if (solution_extraction_data[solution_index].uses_external_dofhandler)
          {
            Assert(&scratch_data != &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }
        else
          {
            Assert(&scratch_data == &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }

        return solution_extraction_data[solution_index]
          .get_scratch_data()
          .template get_laplacians<view_extractor_type, ScalarType>(
            solution_extraction_data[solution_index].solution_name,
            get_extractor());
      }
    };



    /**
     * Extract the solution Hessians from the discretised solution field
     * subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a solution field
     */
    template <typename SubSpaceViewsType, types::solution_index solution_index>
    class SymbolicOp<SubSpaceViewsType,
                     SymbolicOpCodes::hessian,
                     typename std::enable_if<is_field_solution<
                       typename SubSpaceViewsType::SpaceType>::value>::type,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpHessianBase<SubSpaceViewsType, solution_index>
    {
      DEAL_II_SYMBOLIC_OP_FIELD_SOLUTION_SUBSPACE_COMMON_IMPL(
        SymbolicOpHessianBase,
        SubSpaceViewsType,
        solution_index,
        SymbolicOpCodes::hessian)

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<View_t,
                     SubSpaceViews::Scalar<
                       typename SubSpaceViewsType::SpaceType>>::value ||
          std::is_same<View_t,
                       SubSpaceViews::Vector<
                         typename SubSpaceViewsType::SpaceType>>::value,
        "The selected subspace view does not support the Hessian operation.");

    public:
      // Return solution symmetric gradients at all quadrature points
      template <typename ScalarType>
      const return_type<ScalarType> &
      operator()(
        MeshWorker::ScratchData<dimension, space_dimension> &scratch_data,
        const std::vector<SolutionExtractionData<dimension, space_dimension>>
          &solution_extraction_data) const
      {
        Assert(solution_index < solution_extraction_data.size(),
               ExcIndexRange(solution_index,
                             0,
                             solution_extraction_data.size()));

        (void)scratch_data;
        if (solution_extraction_data[solution_index].uses_external_dofhandler)
          {
            Assert(&scratch_data != &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }
        else
          {
            Assert(&scratch_data == &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }

        return solution_extraction_data[solution_index]
          .get_scratch_data()
          .template get_hessians<view_extractor_type, ScalarType>(
            solution_extraction_data[solution_index].solution_name,
            get_extractor());
      }
    };



    /**
     * Extract the solution third derivatives from the discretised solution
     * field subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a solution field
     */
    template <typename SubSpaceViewsType, types::solution_index solution_index>
    class SymbolicOp<SubSpaceViewsType,
                     SymbolicOpCodes::third_derivative,
                     typename std::enable_if<is_field_solution<
                       typename SubSpaceViewsType::SpaceType>::value>::type,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpThirdDerivativeBase<SubSpaceViewsType, solution_index>
    {
      DEAL_II_SYMBOLIC_OP_FIELD_SOLUTION_SUBSPACE_COMMON_IMPL(
        SymbolicOpThirdDerivativeBase,
        SubSpaceViewsType,
        solution_index,
        SymbolicOpCodes::third_derivative)

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<View_t,
                     SubSpaceViews::Scalar<
                       typename SubSpaceViewsType::SpaceType>>::value ||
          std::is_same<View_t,
                       SubSpaceViews::Vector<
                         typename SubSpaceViewsType::SpaceType>>::value,
        "The selected subspace view does not support the third derivative operation.");

    public:
      // Return solution third derivatives at all quadrature points
      template <typename ScalarType>
      const return_type<ScalarType> &
      operator()(
        MeshWorker::ScratchData<dimension, space_dimension> &scratch_data,
        const std::vector<SolutionExtractionData<dimension, space_dimension>>
          &solution_extraction_data) const
      {
        Assert(solution_index < solution_extraction_data.size(),
               ExcIndexRange(solution_index,
                             0,
                             solution_extraction_data.size()));

        (void)scratch_data;
        if (solution_extraction_data[solution_index].uses_external_dofhandler)
          {
            Assert(&scratch_data != &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }
        else
          {
            Assert(&scratch_data == &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }

        return solution_extraction_data[solution_index]
          .get_scratch_data()
          .template get_third_derivatives<view_extractor_type, ScalarType>(
            solution_extraction_data[solution_index].solution_name,
            get_extractor());
      }
    };



    /* -------- Finite element spaces: Solution fields (interfaces) -------- */



    template <typename SubSpaceViewsType, types::solution_index solution_index>
    class SymbolicOp<SubSpaceViewsType,
                     SymbolicOpCodes::jump_in_values,
                     typename std::enable_if<is_field_solution<
                       typename SubSpaceViewsType::SpaceType>::value>::type,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpJumpValueBase<SubSpaceViewsType, solution_index>
    {
      DEAL_II_SYMBOLIC_OP_FIELD_SOLUTION_SUBSPACE_COMMON_IMPL(
        SymbolicOpJumpValueBase,
        SubSpaceViewsType,
        solution_index,
        SymbolicOpCodes::jump_in_values)

    public:
      // Return solution values at all quadrature points
      template <typename ScalarType>
      const return_type<ScalarType> &
      operator()(
        MeshWorker::ScratchData<dimension, space_dimension> &scratch_data,
        const std::vector<SolutionExtractionData<dimension, space_dimension>>
          &solution_extraction_data) const
      {
        Assert(solution_index < solution_extraction_data.size(),
               ExcIndexRange(solution_index,
                             0,
                             solution_extraction_data.size()));

        (void)scratch_data;
        if (solution_extraction_data[solution_index].uses_external_dofhandler)
          {
            Assert(&scratch_data != &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }
        else
          {
            Assert(&scratch_data == &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }

        return solution_extraction_data[solution_index]
          .get_scratch_data()
          .template get_jumps_in_values<view_extractor_type, ScalarType>(
            solution_extraction_data[solution_index].solution_name,
            get_extractor());
      }
    };


    template <typename SubSpaceViewsType, types::solution_index solution_index>
    class SymbolicOp<SubSpaceViewsType,
                     SymbolicOpCodes::average_of_values,
                     typename std::enable_if<is_field_solution<
                       typename SubSpaceViewsType::SpaceType>::value>::type,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpAverageValueBase<SubSpaceViewsType, solution_index>
    {
      DEAL_II_SYMBOLIC_OP_FIELD_SOLUTION_SUBSPACE_COMMON_IMPL(
        SymbolicOpAverageValueBase,
        SubSpaceViewsType,
        solution_index,
        SymbolicOpCodes::average_of_values)

    public:
      // Return solution values at all quadrature points
      template <typename ScalarType>
      const return_type<ScalarType> &
      operator()(
        MeshWorker::ScratchData<dimension, space_dimension> &scratch_data,
        const std::vector<SolutionExtractionData<dimension, space_dimension>>
          &solution_extraction_data) const
      {
        Assert(solution_index < solution_extraction_data.size(),
               ExcIndexRange(solution_index,
                             0,
                             solution_extraction_data.size()));

        (void)scratch_data;
        if (solution_extraction_data[solution_index].uses_external_dofhandler)
          {
            Assert(&scratch_data != &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }
        else
          {
            Assert(&scratch_data == &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }

        return solution_extraction_data[solution_index]
          .get_scratch_data()
          .template get_averages_of_values<view_extractor_type, ScalarType>(
            solution_extraction_data[solution_index].solution_name,
            get_extractor());
      }
    };


    template <typename SubSpaceViewsType, types::solution_index solution_index>
    class SymbolicOp<SubSpaceViewsType,
                     SymbolicOpCodes::jump_in_gradients,
                     typename std::enable_if<is_field_solution<
                       typename SubSpaceViewsType::SpaceType>::value>::type,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpJumpGradientBase<SubSpaceViewsType, solution_index>
    {
      DEAL_II_SYMBOLIC_OP_FIELD_SOLUTION_SUBSPACE_COMMON_IMPL(
        SymbolicOpJumpGradientBase,
        SubSpaceViewsType,
        solution_index,
        SymbolicOpCodes::jump_in_gradients)

    public:
      // Return solution values at all quadrature points
      template <typename ScalarType>
      const return_type<ScalarType> &
      operator()(
        MeshWorker::ScratchData<dimension, space_dimension> &scratch_data,
        const std::vector<SolutionExtractionData<dimension, space_dimension>>
          &solution_extraction_data) const
      {
        Assert(solution_index < solution_extraction_data.size(),
               ExcIndexRange(solution_index,
                             0,
                             solution_extraction_data.size()));

        (void)scratch_data;
        if (solution_extraction_data[solution_index].uses_external_dofhandler)
          {
            Assert(&scratch_data != &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }
        else
          {
            Assert(&scratch_data == &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }

        return solution_extraction_data[solution_index]
          .get_scratch_data()
          .template get_jumps_in_gradients<view_extractor_type, ScalarType>(
            solution_extraction_data[solution_index].solution_name,
            get_extractor());
      }
    };


    template <typename SubSpaceViewsType, types::solution_index solution_index>
    class SymbolicOp<SubSpaceViewsType,
                     SymbolicOpCodes::average_of_gradients,
                     typename std::enable_if<is_field_solution<
                       typename SubSpaceViewsType::SpaceType>::value>::type,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpAverageGradientBase<SubSpaceViewsType, solution_index>
    {
      DEAL_II_SYMBOLIC_OP_FIELD_SOLUTION_SUBSPACE_COMMON_IMPL(
        SymbolicOpAverageGradientBase,
        SubSpaceViewsType,
        solution_index,
        SymbolicOpCodes::average_of_gradients)

    public:
      // Return solution values at all quadrature points
      template <typename ScalarType>
      const return_type<ScalarType> &
      operator()(
        MeshWorker::ScratchData<dimension, space_dimension> &scratch_data,
        const std::vector<SolutionExtractionData<dimension, space_dimension>>
          &solution_extraction_data) const
      {
        Assert(solution_index < solution_extraction_data.size(),
               ExcIndexRange(solution_index,
                             0,
                             solution_extraction_data.size()));

        (void)scratch_data;
        if (solution_extraction_data[solution_index].uses_external_dofhandler)
          {
            Assert(&scratch_data != &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }
        else
          {
            Assert(&scratch_data == &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }

        return solution_extraction_data[solution_index]
          .get_scratch_data()
          .template get_averages_of_gradients<view_extractor_type, ScalarType>(
            solution_extraction_data[solution_index].solution_name,
            get_extractor());
      }
    };


    template <typename SubSpaceViewsType, types::solution_index solution_index>
    class SymbolicOp<SubSpaceViewsType,
                     SymbolicOpCodes::jump_in_hessians,
                     typename std::enable_if<is_field_solution<
                       typename SubSpaceViewsType::SpaceType>::value>::type,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpJumpHessianBase<SubSpaceViewsType, solution_index>
    {
      DEAL_II_SYMBOLIC_OP_FIELD_SOLUTION_SUBSPACE_COMMON_IMPL(
        SymbolicOpJumpHessianBase,
        SubSpaceViewsType,
        solution_index,
        SymbolicOpCodes::jump_in_hessians)

    public:
      // Return solution values at all quadrature points
      template <typename ScalarType>
      const return_type<ScalarType> &
      operator()(
        MeshWorker::ScratchData<dimension, space_dimension> &scratch_data,
        const std::vector<SolutionExtractionData<dimension, space_dimension>>
          &solution_extraction_data) const
      {
        Assert(solution_index < solution_extraction_data.size(),
               ExcIndexRange(solution_index,
                             0,
                             solution_extraction_data.size()));

        (void)scratch_data;
        if (solution_extraction_data[solution_index].uses_external_dofhandler)
          {
            Assert(&scratch_data != &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }
        else
          {
            Assert(&scratch_data == &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }

        return solution_extraction_data[solution_index]
          .get_scratch_data()
          .template get_jumps_in_hessians<view_extractor_type, ScalarType>(
            solution_extraction_data[solution_index].solution_name,
            get_extractor());
      }
    };


    template <typename SubSpaceViewsType, types::solution_index solution_index>
    class SymbolicOp<SubSpaceViewsType,
                     SymbolicOpCodes::average_of_hessians,
                     typename std::enable_if<is_field_solution<
                       typename SubSpaceViewsType::SpaceType>::value>::type,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpAverageHessianBase<SubSpaceViewsType, solution_index>
    {
      DEAL_II_SYMBOLIC_OP_FIELD_SOLUTION_SUBSPACE_COMMON_IMPL(
        SymbolicOpAverageHessianBase,
        SubSpaceViewsType,
        solution_index,
        SymbolicOpCodes::average_of_hessians)

    public:
      // Return solution values at all quadrature points
      template <typename ScalarType>
      const return_type<ScalarType> &
      operator()(
        MeshWorker::ScratchData<dimension, space_dimension> &scratch_data,
        const std::vector<SolutionExtractionData<dimension, space_dimension>>
          &solution_extraction_data) const
      {
        Assert(solution_index < solution_extraction_data.size(),
               ExcIndexRange(solution_index,
                             0,
                             solution_extraction_data.size()));

        (void)scratch_data;
        if (solution_extraction_data[solution_index].uses_external_dofhandler)
          {
            Assert(&scratch_data != &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }
        else
          {
            Assert(&scratch_data == &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }

        return solution_extraction_data[solution_index]
          .get_scratch_data()
          .template get_averages_of_hessians<view_extractor_type, ScalarType>(
            solution_extraction_data[solution_index].solution_name,
            get_extractor());
      }
    };


    template <typename SubSpaceViewsType, types::solution_index solution_index>
    class SymbolicOp<SubSpaceViewsType,
                     SymbolicOpCodes::jump_in_third_derivatives,
                     typename std::enable_if<is_field_solution<
                       typename SubSpaceViewsType::SpaceType>::value>::type,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpJumpThirdDerivativeBase<SubSpaceViewsType,
                                                 solution_index>
    {
      DEAL_II_SYMBOLIC_OP_FIELD_SOLUTION_SUBSPACE_COMMON_IMPL(
        SymbolicOpJumpThirdDerivativeBase,
        SubSpaceViewsType,
        solution_index,
        SymbolicOpCodes::jump_in_third_derivatives)

    public:
      // Return solution values at all quadrature points
      template <typename ScalarType>
      const return_type<ScalarType> &
      operator()(
        MeshWorker::ScratchData<dimension, space_dimension> &scratch_data,
        const std::vector<SolutionExtractionData<dimension, space_dimension>>
          &solution_extraction_data) const
      {
        Assert(solution_index < solution_extraction_data.size(),
               ExcIndexRange(solution_index,
                             0,
                             solution_extraction_data.size()));

        (void)scratch_data;
        if (solution_extraction_data[solution_index].uses_external_dofhandler)
          {
            Assert(&scratch_data != &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }
        else
          {
            Assert(&scratch_data == &solution_extraction_data[solution_index]
                                       .get_scratch_data(),
                   ExcInternalError());
          }

        return solution_extraction_data[solution_index]
          .get_scratch_data()
          .template get_jumps_in_third_derivatives<view_extractor_type,
                                                   ScalarType>(
            solution_extraction_data[solution_index].solution_name,
            get_extractor());
      }
    };


#undef DEAL_II_SYMBOLIC_OP_FIELD_SOLUTION_SUBSPACE_COMMON_IMPL


  } // namespace Operators
} // namespace WeakForms



#ifndef DOXYGEN



/* ======================== Convenience functions ======================== */



namespace WeakForms
{
  namespace internal
  {
    /* ----- Finite element subspaces: Test functions and trial solutions -----
     */

    /**
     * @brief Value variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::value>
    value(const SubSpaceViewsType<SpaceType> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op     = SubSpaceViewsType<SpaceType>;
      using OpType = SymbolicOp<Op, SymbolicOpCodes::value>;

      return OpType(operand);
    }


    /**
     * @brief Value variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    template <template <int, class> class SubSpaceViewsType,
              int rank,
              typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<rank, SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::value>
    value(const SubSpaceViewsType<rank, SpaceType> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op     = SubSpaceViewsType<rank, SpaceType>;
      using OpType = SymbolicOp<Op, SymbolicOpCodes::value>;

      return OpType(operand);
    }


    /**
     * @brief Gradient variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::gradient>
    gradient(const SubSpaceViewsType<SpaceType> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op     = SubSpaceViewsType<SpaceType>;
      using OpType = SymbolicOp<Op, SymbolicOpCodes::gradient>;

      return OpType(operand);
    }


    /**
     * @brief Gradient variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    template <template <int, class> class SubSpaceViewsType,
              int rank,
              typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<rank, SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::gradient>
    gradient(const SubSpaceViewsType<rank, SpaceType> &operand)
    {
      static_assert(
        std::is_same<SubSpaceViewsType<rank, SpaceType>,
                     SubSpaceViews::Tensor<rank, SpaceType>>::value,
        "The selected subspace view does not support the gradient operation.");

      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op     = SubSpaceViewsType<rank, SpaceType>;
      using OpType = SymbolicOp<Op, SymbolicOpCodes::gradient>;

      return OpType(operand);
    }


    /**
     * @brief Symmetric gradient variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::symmetric_gradient>
    symmetric_gradient(const SubSpaceViewsType<SpaceType> &operand)
    {
      static_assert(
        std::is_same<SubSpaceViewsType<SpaceType>,
                     SubSpaceViews::Vector<SpaceType>>::value,
        "The selected subspace view does not support the symmetric gradient operation.");

      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op     = SubSpaceViewsType<SpaceType>;
      using OpType = SymbolicOp<Op, SymbolicOpCodes::symmetric_gradient>;

      return OpType(operand);
    }


    /**
     * @brief Symmetric gradient variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    // template <template<int, class> typename SubSpaceViewsType, int rank,
    // typename SpaceType>
    // WeakForms::Operators::SymbolicOp<SubSpaceViewsType<rank, SpaceType>,
    //                               WeakForms::Operators::SymbolicOpCodes::symmetric_gradient>
    // symmetric_gradient(const SubSpaceViewsType<rank, SpaceType> &operand)
    // {
    //   static_assert(false, "Tensor and SymmetricTensor subspace views do not
    //   support the symmetric gradient operation.");

    //   using namespace WeakForms;
    //   using namespace WeakForms::Operators;

    //   using Op     = SubSpaceViewsType<rank, SpaceType>;
    //   using OpType = SymbolicOp<Op, SymbolicOpCodes::symmetric_gradient>;

    //   return OpType(operand);
    // }


    /**
     * @brief Divergence variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::divergence>
    divergence(const SubSpaceViewsType<SpaceType> &operand)
    {
      static_assert(
        std::is_same<SubSpaceViewsType<SpaceType>,
                     SubSpaceViews::Vector<SpaceType>>::value,
        "The selected subspace view does not support the divergence operation.");
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op     = SubSpaceViewsType<SpaceType>;
      using OpType = SymbolicOp<Op, SymbolicOpCodes::divergence>;

      return OpType(operand);
    }


    /**
     * @brief Divergence variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    template <template <int, class> class SubSpaceViewsType,
              int rank,
              typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<rank, SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::divergence>
    divergence(const SubSpaceViewsType<rank, SpaceType> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op     = SubSpaceViewsType<rank, SpaceType>;
      using OpType = SymbolicOp<Op, SymbolicOpCodes::divergence>;

      return OpType(operand);
    }


    /**
     * @brief Curl variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::curl>
    curl(const SubSpaceViewsType<SpaceType> &operand)
    {
      static_assert(
        std::is_same<SubSpaceViewsType<SpaceType>,
                     SubSpaceViews::Vector<SpaceType>>::value,
        "The selected subspace view does not support the curl operation.");
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op     = SubSpaceViewsType<SpaceType>;
      using OpType = SymbolicOp<Op, SymbolicOpCodes::curl>;

      return OpType(operand);
    }


    /**
     * @brief Curl variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    // template <template<int, class> typename SubSpaceViewsType, int rank,
    // typename SpaceType>
    // WeakForms::Operators::SymbolicOp<SubSpaceViewsType<rank, SpaceType>,
    //                               WeakForms::Operators::SymbolicOpCodes::curl>
    // curl(const SubSpaceViewsType<rank, SpaceType> &operand)
    // {
    //   static_assert(false, "Tensor and SymmetricTensor subspace views do not
    //   support the symmetric gradient operation.");

    //   using namespace WeakForms;
    //   using namespace WeakForms::Operators;

    //   using Op     = SubSpaceViewsType<rank, SpaceType>;
    //   using OpType = SymbolicOp<Op, SymbolicOpCodes::curl>;

    //   return OpType(operand);
    // }


    /**
     * @brief Laplacian variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::laplacian>
    laplacian(const SubSpaceViewsType<SpaceType> &operand)
    {
      static_assert(
        std::is_same<SubSpaceViewsType<SpaceType>,
                     SubSpaceViews::Scalar<SpaceType>>::value,
        "The selected subspace view does not support the Laplacian operation.");
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op     = SubSpaceViewsType<SpaceType>;
      using OpType = SymbolicOp<Op, SymbolicOpCodes::laplacian>;

      return OpType(operand);
    }


    // /**
    //  * @brief Laplacian variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
    //  *
    //  * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
    //  * @tparam SpaceType A space type, specifically a test space or trial space
    //  * @param operand
    //  * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
    //  * WeakForms::Operators::SymbolicOpCodes::value>
    //  */
    // template <template<int, class> typename SubSpaceViewsType, int rank,
    // typename SpaceType>
    // WeakForms::Operators::SymbolicOp<SubSpaceViewsType<rank,SpaceType>,
    //                               WeakForms::Operators::SymbolicOpCodes::laplacian>
    // laplacian(const SubSpaceViewsType<rank,SpaceType> &operand)
    // {
    //   using namespace WeakForms;
    //   using namespace WeakForms::Operators;

    //   using Op     = SubSpaceViewsType<rank,SpaceType>;
    //   using OpType = SymbolicOp<Op, SymbolicOpCodes::laplacian>;

    //   return OpType(operand);
    // }


    /**
     * @brief Hessian variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::hessian>
    hessian(const SubSpaceViewsType<SpaceType> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op     = SubSpaceViewsType<SpaceType>;
      using OpType = SymbolicOp<Op, SymbolicOpCodes::hessian>;

      return OpType(operand);
    }


    // /**
    //  * @brief Hessian variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
    //  *
    //  * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
    //  * @tparam SpaceType A space type, specifically a test space or trial space
    //  * @param operand
    //  * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
    //  * WeakForms::Operators::SymbolicOpCodes::value>
    //  */
    // template <template<int, class> typename SubSpaceViewsType, int rank,
    // typename SpaceType>
    // WeakForms::Operators::SymbolicOp<SubSpaceViewsType<rank,SpaceType>,
    //                               WeakForms::Operators::SymbolicOpCodes::hessian>
    // hessian(const SubSpaceViewsType<rank,SpaceType> &operand)
    // {
    //   using namespace WeakForms;
    //   using namespace WeakForms::Operators;

    //   using Op     = SubSpaceViewsType<rank,SpaceType>;
    //   using OpType = SymbolicOp<Op, SymbolicOpCodes::hessian>;

    //   return OpType(operand);
    // }


    /**
     * @brief Third derivative variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::third_derivative>
    third_derivative(const SubSpaceViewsType<SpaceType> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op     = SubSpaceViewsType<SpaceType>;
      using OpType = SymbolicOp<Op, SymbolicOpCodes::third_derivative>;

      return OpType(operand);
    }


    // /**
    //  * @brief Laplacian variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
    //  *
    //  * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
    //  * @tparam SpaceType A space type, specifically a test space or trial space
    //  * @param operand
    //  * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
    //  * WeakForms::Operators::SymbolicOpCodes::value>
    //  */
    // template <template<int, class> typename SubSpaceViewsType, int rank,
    // typename SpaceType>
    // WeakForms::Operators::SymbolicOp<SubSpaceViewsType<rank,SpaceType>,
    //                               WeakForms::Operators::SymbolicOpCodes::third_derivative>
    // third_derivative(const SubSpaceViewsType<rank,SpaceType> &operand)
    // {
    //   using namespace WeakForms;
    //   using namespace WeakForms::Operators;

    //   using Op     = SubSpaceViewsType<rank,SpaceType>;
    //   using OpType = SymbolicOp<Op, SymbolicOpCodes::third_derivative>;

    //   return OpType(operand);
    // }



    /* -- Finite element subspaces: Test functions and trial solutions
     * (interface)
     * -- */

    /**
     * @brief Jump in values variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::jump_in_values>
     */
    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::jump_in_values>
    jump_in_values(const SubSpaceViewsType<SpaceType> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op     = SubSpaceViewsType<SpaceType>;
      using OpType = SymbolicOp<Op, SymbolicOpCodes::jump_in_values>;

      return OpType(operand);
    }


    /**
     * @brief Jump in gradient variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::jump_in_gradients>
     */
    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::jump_in_gradients>
    jump_in_gradients(const SubSpaceViewsType<SpaceType> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op     = SubSpaceViewsType<SpaceType>;
      using OpType = SymbolicOp<Op, SymbolicOpCodes::jump_in_gradients>;

      return OpType(operand);
    }


    /**
     * @brief Jump in Hessians variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::jump_in_hessians>
     */
    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::jump_in_hessians>
    jump_in_hessians(const SubSpaceViewsType<SpaceType> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op     = SubSpaceViewsType<SpaceType>;
      using OpType = SymbolicOp<Op, SymbolicOpCodes::jump_in_hessians>;

      return OpType(operand);
    }


    /**
     * @brief Jump in third derivatives variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::jump_in_third_derivatives>
     */
    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::jump_in_third_derivatives>
    jump_in_third_derivatives(const SubSpaceViewsType<SpaceType> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op     = SubSpaceViewsType<SpaceType>;
      using OpType = SymbolicOp<Op, SymbolicOpCodes::jump_in_third_derivatives>;

      return OpType(operand);
    }



    /**
     * @brief Average of values variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::average_of_values>
     */
    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::average_of_values>
    average_of_values(const SubSpaceViewsType<SpaceType> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op     = SubSpaceViewsType<SpaceType>;
      using OpType = SymbolicOp<Op, SymbolicOpCodes::average_of_values>;

      return OpType(operand);
    }


    /**
     * @brief Average of gradient variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::average_of_gradients>
     */
    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::average_of_gradients>
    average_of_gradients(const SubSpaceViewsType<SpaceType> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op     = SubSpaceViewsType<SpaceType>;
      using OpType = SymbolicOp<Op, SymbolicOpCodes::average_of_gradients>;

      return OpType(operand);
    }


    /**
     * @brief Average of Hessians variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::average_of_hessians>
     */
    template <template <class> class SubSpaceViewsType, typename SpaceType>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<SpaceType>,
      WeakForms::Operators::SymbolicOpCodes::average_of_hessians>
    average_of_hessians(const SubSpaceViewsType<SpaceType> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op     = SubSpaceViewsType<SpaceType>;
      using OpType = SymbolicOp<Op, SymbolicOpCodes::average_of_hessians>;

      return OpType(operand);
    }



    /* ------------- Finite element subspaces: Field solutions ------------- */

    /**
     * @brief Value variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    template <types::solution_index solution_index,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::value,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    value(const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op = SubSpaceViewsType<FieldSolution<dim, spacedim>>;
      using OpType =
        SymbolicOp<Op,
                   SymbolicOpCodes::value,
                   void,
                   WeakForms::internal::SolutionIndex<solution_index>>;

      return OpType(operand);
    }


    /**
     * @brief Value variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    template <types::solution_index solution_index,
              template <int, class>
              class SubSpaceViewsType,
              int rank,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<rank, FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::value,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    value(const SubSpaceViewsType<rank, FieldSolution<dim, spacedim>> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op = SubSpaceViewsType<rank, FieldSolution<dim, spacedim>>;
      using OpType =
        SymbolicOp<Op,
                   SymbolicOpCodes::value,
                   void,
                   WeakForms::internal::SolutionIndex<solution_index>>;

      return OpType(operand);
    }


    /**
     * @brief Gradient variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    template <types::solution_index solution_index,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::gradient,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    gradient(const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op = SubSpaceViewsType<FieldSolution<dim, spacedim>>;
      using OpType =
        SymbolicOp<Op,
                   SymbolicOpCodes::gradient,
                   void,
                   WeakForms::internal::SolutionIndex<solution_index>>;

      return OpType(operand);
    }


    /**
     * @brief Gradient variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    template <types::solution_index solution_index,
              template <int, class>
              class SubSpaceViewsType,
              int rank,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<rank, FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::gradient,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    gradient(
      const SubSpaceViewsType<rank, FieldSolution<dim, spacedim>> &operand)
    {
      static_assert(
        std::is_same<
          SubSpaceViewsType<rank, FieldSolution<dim, spacedim>>,
          SubSpaceViews::Tensor<rank, FieldSolution<dim, spacedim>>>::value,
        "The selected subspace view does not support the gradient operation.");

      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op = SubSpaceViewsType<rank, FieldSolution<dim, spacedim>>;
      using OpType =
        SymbolicOp<Op,
                   SymbolicOpCodes::gradient,
                   void,
                   WeakForms::internal::SolutionIndex<solution_index>>;

      return OpType(operand);
    }


    /**
     * @brief Symmetric gradient variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    template <types::solution_index solution_index,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::symmetric_gradient,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    symmetric_gradient(
      const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand)
    {
      static_assert(
        std::is_same<
          SubSpaceViewsType<FieldSolution<dim, spacedim>>,
          SubSpaceViews::Vector<FieldSolution<dim, spacedim>>>::value,
        "The selected subspace view does not support the symmetric gradient operation.");

      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op = SubSpaceViewsType<FieldSolution<dim, spacedim>>;
      using OpType =
        SymbolicOp<Op,
                   SymbolicOpCodes::symmetric_gradient,
                   void,
                   WeakForms::internal::SolutionIndex<solution_index>>;

      return OpType(operand);
    }


    /**
     * @brief Symmetric gradient variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    // template <template<int, class> typename SubSpaceViewsType, int rank,
    // int dim, int spacedim>
    // WeakForms::Operators::SymbolicOp<SubSpaceViewsType<rank,
    // FieldSolution<dim,spacedim>>,
    //                               WeakForms::Operators::SymbolicOpCodes::symmetric_gradient>
    // symmetric_gradient(const SubSpaceViewsType<rank,
    // FieldSolution<dim,spacedim>> &operand)
    // {
    //   static_assert(false, "Tensor and SymmetricTensor subspace views do not
    //   support the symmetric gradient operation.");

    //   using namespace WeakForms;
    //   using namespace WeakForms::Operators;

    //   using Op     = SubSpaceViewsType<rank, FieldSolution<dim,spacedim>>;
    //   using OpType = SymbolicOp<Op, SymbolicOpCodes::symmetric_gradient>;

    //   return OpType(operand);
    // }


    /**
     * @brief Divergence variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    template <types::solution_index solution_index,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::divergence,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    divergence(const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand)
    {
      static_assert(
        std::is_same<
          SubSpaceViewsType<FieldSolution<dim, spacedim>>,
          SubSpaceViews::Vector<FieldSolution<dim, spacedim>>>::value,
        "The selected subspace view does not support the divergence operation.");
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op = SubSpaceViewsType<FieldSolution<dim, spacedim>>;
      using OpType =
        SymbolicOp<Op,
                   SymbolicOpCodes::divergence,
                   void,
                   WeakForms::internal::SolutionIndex<solution_index>>;

      return OpType(operand);
    }


    /**
     * @brief Divergence variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    template <types::solution_index solution_index,
              template <int, class>
              class SubSpaceViewsType,
              int rank,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<rank, FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::divergence,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    divergence(
      const SubSpaceViewsType<rank, FieldSolution<dim, spacedim>> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op = SubSpaceViewsType<rank, FieldSolution<dim, spacedim>>;
      using OpType =
        SymbolicOp<Op,
                   SymbolicOpCodes::divergence,
                   void,
                   WeakForms::internal::SolutionIndex<solution_index>>;

      return OpType(operand);
    }


    /**
     * @brief Curl variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    template <types::solution_index solution_index,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::curl,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    curl(const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand)
    {
      static_assert(
        std::is_same<
          SubSpaceViewsType<FieldSolution<dim, spacedim>>,
          SubSpaceViews::Vector<FieldSolution<dim, spacedim>>>::value,
        "The selected subspace view does not support the curl operation.");
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op = SubSpaceViewsType<FieldSolution<dim, spacedim>>;
      using OpType =
        SymbolicOp<Op,
                   SymbolicOpCodes::curl,
                   void,
                   WeakForms::internal::SolutionIndex<solution_index>>;

      return OpType(operand);
    }


    /**
     * @brief Curl variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    // template <template<int, class> typename SubSpaceViewsType, int rank,
    // int dim, int spacedim>
    // WeakForms::Operators::SymbolicOp<SubSpaceViewsType<rank,
    // FieldSolution<dim,spacedim>>,
    //                               WeakForms::Operators::SymbolicOpCodes::curl>
    // curl(const SubSpaceViewsType<rank, FieldSolution<dim,spacedim>> &operand)
    // {
    //   static_assert(false, "Tensor and SymmetricTensor subspace views do not
    //   support the symmetric gradient operation.");

    //   using namespace WeakForms;
    //   using namespace WeakForms::Operators;

    //   using Op     = SubSpaceViewsType<rank, FieldSolution<dim,spacedim>>;
    //   using OpType = SymbolicOp<Op, SymbolicOpCodes::curl>;

    //   return OpType(operand);
    // }


    /**
     * @brief Laplacian variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    template <types::solution_index solution_index,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::laplacian,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    laplacian(const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand)
    {
      static_assert(
        std::is_same<
          SubSpaceViewsType<FieldSolution<dim, spacedim>>,
          SubSpaceViews::Scalar<FieldSolution<dim, spacedim>>>::value,
        "The selected subspace view does not support the Laplacian operation.");
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op = SubSpaceViewsType<FieldSolution<dim, spacedim>>;
      using OpType =
        SymbolicOp<Op,
                   SymbolicOpCodes::laplacian,
                   void,
                   WeakForms::internal::SolutionIndex<solution_index>>;

      return OpType(operand);
    }


    // /**
    //  * @brief Laplacian variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
    //  *
    //  * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
    //  * @tparam SpaceType A space type, specifically a test space or trial space
    //  * @param operand
    //  * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
    //  * WeakForms::Operators::SymbolicOpCodes::value>
    //  */
    // template <template<int, class> typename SubSpaceViewsType, int rank,
    // int dim, int spacedim>
    // WeakForms::Operators::SymbolicOp<SubSpaceViewsType<rank,FieldSolution<dim,spacedim>>,
    //                               WeakForms::Operators::SymbolicOpCodes::laplacian>
    // laplacian(const SubSpaceViewsType<rank,FieldSolution<dim,spacedim>>
    // &operand)
    // {
    //   using namespace WeakForms;
    //   using namespace WeakForms::Operators;

    //   using Op     = SubSpaceViewsType<rank,FieldSolution<dim,spacedim>>;
    //   using OpType = SymbolicOp<Op, SymbolicOpCodes::laplacian>;

    //   return OpType(operand);
    // }


    /**
     * @brief Hessian variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    template <types::solution_index solution_index,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::hessian,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    hessian(const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op = SubSpaceViewsType<FieldSolution<dim, spacedim>>;
      using OpType =
        SymbolicOp<Op,
                   SymbolicOpCodes::hessian,
                   void,
                   WeakForms::internal::SolutionIndex<solution_index>>;

      return OpType(operand);
    }


    // /**
    //  * @brief Hessian variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
    //  *
    //  * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
    //  * @tparam SpaceType A space type, specifically a test space or trial space
    //  * @param operand
    //  * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
    //  * WeakForms::Operators::SymbolicOpCodes::value>
    //  */
    // template <template<int, class> typename SubSpaceViewsType, int rank,
    // int dim, int spacedim>
    // WeakForms::Operators::SymbolicOp<SubSpaceViewsType<rank,FieldSolution<dim,spacedim>>,
    //                               WeakForms::Operators::SymbolicOpCodes::hessian>
    // hessian(const SubSpaceViewsType<rank,FieldSolution<dim,spacedim>>
    // &operand)
    // {
    //   using namespace WeakForms;
    //   using namespace WeakForms::Operators;

    //   using Op     = SubSpaceViewsType<rank,FieldSolution<dim,spacedim>>;
    //   using OpType = SymbolicOp<Op, SymbolicOpCodes::hessian>;

    //   return OpType(operand);
    // }


    /**
     * @brief Third derivative variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
     * @tparam SpaceType A space type, specifically a test space or trial space
     * @param operand
     * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
     * WeakForms::Operators::SymbolicOpCodes::value>
     */
    template <types::solution_index solution_index,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::third_derivative,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    third_derivative(
      const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op = SubSpaceViewsType<FieldSolution<dim, spacedim>>;
      using OpType =
        SymbolicOp<Op,
                   SymbolicOpCodes::third_derivative,
                   void,
                   WeakForms::internal::SolutionIndex<solution_index>>;

      return OpType(operand);
    }


    // /**
    //  * @brief Laplacian variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
    //  *
    //  * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
    //  * @tparam SpaceType A space type, specifically a test space or trial space
    //  * @param operand
    //  * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
    //  * WeakForms::Operators::SymbolicOpCodes::value>
    //  */
    // template <template<int, class> typename SubSpaceViewsType, int rank,
    // int dim, int spacedim>
    // WeakForms::Operators::SymbolicOp<SubSpaceViewsType<rank,FieldSolution<dim,spacedim>>,
    //                               WeakForms::Operators::SymbolicOpCodes::third_derivative>
    // third_derivative(const
    // SubSpaceViewsType<rank,FieldSolution<dim,spacedim>> &operand)
    // {
    //   using namespace WeakForms;
    //   using namespace WeakForms::Operators;

    //   using Op     = SubSpaceViewsType<rank,FieldSolution<dim,spacedim>>;
    //   using OpType = SymbolicOp<Op, SymbolicOpCodes::third_derivative>;

    //   return OpType(operand);
    // }



    /* -------- Finite element subspaces: Field solutions (interface) --------
     */


    template <types::solution_index solution_index,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::jump_in_values,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    jump_in_values(
      const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op = SubSpaceViewsType<FieldSolution<dim, spacedim>>;
      using OpType =
        SymbolicOp<Op,
                   SymbolicOpCodes::jump_in_values,
                   void,
                   WeakForms::internal::SolutionIndex<solution_index>>;

      return OpType(operand);
    }


    template <types::solution_index solution_index,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::jump_in_gradients,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    jump_in_gradients(
      const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op = SubSpaceViewsType<FieldSolution<dim, spacedim>>;
      using OpType =
        SymbolicOp<Op,
                   SymbolicOpCodes::jump_in_gradients,
                   void,
                   WeakForms::internal::SolutionIndex<solution_index>>;

      return OpType(operand);
    }


    template <types::solution_index solution_index,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::jump_in_hessians,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    jump_in_hessians(
      const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op = SubSpaceViewsType<FieldSolution<dim, spacedim>>;
      using OpType =
        SymbolicOp<Op,
                   SymbolicOpCodes::jump_in_hessians,
                   void,
                   WeakForms::internal::SolutionIndex<solution_index>>;

      return OpType(operand);
    }


    template <types::solution_index solution_index,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::jump_in_third_derivatives,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    jump_in_third_derivatives(
      const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op = SubSpaceViewsType<FieldSolution<dim, spacedim>>;
      using OpType =
        SymbolicOp<Op,
                   SymbolicOpCodes::jump_in_third_derivatives,
                   void,
                   WeakForms::internal::SolutionIndex<solution_index>>;

      return OpType(operand);
    }


    template <types::solution_index solution_index,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::average_of_values,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    average_of_values(
      const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op = SubSpaceViewsType<FieldSolution<dim, spacedim>>;
      using OpType =
        SymbolicOp<Op,
                   SymbolicOpCodes::average_of_values,
                   void,
                   WeakForms::internal::SolutionIndex<solution_index>>;

      return OpType(operand);
    }


    template <types::solution_index solution_index,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::average_of_gradients,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    average_of_gradients(
      const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op = SubSpaceViewsType<FieldSolution<dim, spacedim>>;
      using OpType =
        SymbolicOp<Op,
                   SymbolicOpCodes::average_of_gradients,
                   void,
                   WeakForms::internal::SolutionIndex<solution_index>>;

      return OpType(operand);
    }


    template <types::solution_index solution_index,
              template <class>
              class SubSpaceViewsType,
              int dim,
              int spacedim>
    WeakForms::Operators::SymbolicOp<
      SubSpaceViewsType<FieldSolution<dim, spacedim>>,
      WeakForms::Operators::SymbolicOpCodes::average_of_hessians,
      void,
      WeakForms::internal::SolutionIndex<solution_index>>
    average_of_hessians(
      const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand)
    {
      using namespace WeakForms;
      using namespace WeakForms::Operators;

      using Op = SubSpaceViewsType<FieldSolution<dim, spacedim>>;
      using OpType =
        SymbolicOp<Op,
                   SymbolicOpCodes::average_of_hessians,
                   void,
                   WeakForms::internal::SolutionIndex<solution_index>>;

      return OpType(operand);
    }

  } // namespace internal
} // namespace WeakForms



/* ==================== Specialization of type traits ==================== */


namespace WeakForms
{
  // Subspace views

  template <int dim, int spacedim>
  struct is_subspace_view<SubSpaceViews::Scalar<TestFunction<dim, spacedim>>>
    : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_subspace_view<SubSpaceViews::Scalar<TrialSolution<dim, spacedim>>>
    : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_subspace_view<SubSpaceViews::Scalar<FieldSolution<dim, spacedim>>>
    : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_subspace_view<SubSpaceViews::Vector<TestFunction<dim, spacedim>>>
    : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_subspace_view<SubSpaceViews::Vector<TrialSolution<dim, spacedim>>>
    : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_subspace_view<SubSpaceViews::Vector<FieldSolution<dim, spacedim>>>
    : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_subspace_view<
    SubSpaceViews::Tensor<rank, TestFunction<dim, spacedim>>> : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_subspace_view<
    SubSpaceViews::Tensor<rank, TrialSolution<dim, spacedim>>> : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_subspace_view<
    SubSpaceViews::Tensor<rank, FieldSolution<dim, spacedim>>> : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_subspace_view<
    SubSpaceViews::SymmetricTensor<rank, TestFunction<dim, spacedim>>>
    : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_subspace_view<
    SubSpaceViews::SymmetricTensor<rank, TrialSolution<dim, spacedim>>>
    : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_subspace_view<
    SubSpaceViews::SymmetricTensor<rank, FieldSolution<dim, spacedim>>>
    : std::true_type
  {};



  // Decorators

  template <int dim, int spacedim>
  struct is_test_function<SubSpaceViews::Scalar<TestFunction<dim, spacedim>>>
    : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_trial_solution<SubSpaceViews::Scalar<TrialSolution<dim, spacedim>>>
    : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_field_solution<SubSpaceViews::Scalar<FieldSolution<dim, spacedim>>>
    : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_test_function<SubSpaceViews::Vector<TestFunction<dim, spacedim>>>
    : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_trial_solution<SubSpaceViews::Vector<TrialSolution<dim, spacedim>>>
    : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_field_solution<SubSpaceViews::Vector<FieldSolution<dim, spacedim>>>
    : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_test_function<
    SubSpaceViews::Tensor<rank, TestFunction<dim, spacedim>>> : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_trial_solution<
    SubSpaceViews::Tensor<rank, TrialSolution<dim, spacedim>>> : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_field_solution<
    SubSpaceViews::Tensor<rank, FieldSolution<dim, spacedim>>> : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_test_function<
    SubSpaceViews::SymmetricTensor<rank, TestFunction<dim, spacedim>>>
    : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_trial_solution<
    SubSpaceViews::SymmetricTensor<rank, TrialSolution<dim, spacedim>>>
    : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_field_solution<
    SubSpaceViews::SymmetricTensor<rank, FieldSolution<dim, spacedim>>>
    : std::true_type
  {};



  // Symbolic operations: Subspace views

  template <int dim, int spacedim, enum Operators::SymbolicOpCodes OpCode>
  struct is_test_function_op<
    Operators::SymbolicOp<SubSpaceViews::Scalar<TestFunction<dim, spacedim>>,
                          OpCode>> : std::true_type
  {};

  template <int dim, int spacedim, enum Operators::SymbolicOpCodes OpCode>
  struct is_trial_solution_op<
    Operators::SymbolicOp<SubSpaceViews::Scalar<TrialSolution<dim, spacedim>>,
                          OpCode>> : std::true_type
  {};

  template <int                             dim,
            int                             spacedim,
            enum Operators::SymbolicOpCodes OpCode,
            types::solution_index           solution_index>
  struct is_field_solution_op<
    Operators::SymbolicOp<SubSpaceViews::Scalar<FieldSolution<dim, spacedim>>,
                          OpCode,
                          void,
                          WeakForms::internal::SolutionIndex<solution_index>>>
    : std::true_type
  {};

  template <int dim, int spacedim, enum Operators::SymbolicOpCodes OpCode>
  struct is_test_function_op<
    Operators::SymbolicOp<SubSpaceViews::Vector<TestFunction<dim, spacedim>>,
                          OpCode>> : std::true_type
  {};

  template <int dim, int spacedim, enum Operators::SymbolicOpCodes OpCode>
  struct is_trial_solution_op<
    Operators::SymbolicOp<SubSpaceViews::Vector<TrialSolution<dim, spacedim>>,
                          OpCode>> : std::true_type
  {};

  template <int                             dim,
            int                             spacedim,
            enum Operators::SymbolicOpCodes OpCode,
            types::solution_index           solution_index>
  struct is_field_solution_op<
    Operators::SymbolicOp<SubSpaceViews::Vector<FieldSolution<dim, spacedim>>,
                          OpCode,
                          void,
                          WeakForms::internal::SolutionIndex<solution_index>>>
    : std::true_type
  {};

  template <int                             rank,
            int                             dim,
            int                             spacedim,
            enum Operators::SymbolicOpCodes OpCode>
  struct is_test_function_op<Operators::SymbolicOp<
    SubSpaceViews::Tensor<rank, TestFunction<dim, spacedim>>,
    OpCode>> : std::true_type
  {};

  template <int                             rank,
            int                             dim,
            int                             spacedim,
            enum Operators::SymbolicOpCodes OpCode>
  struct is_trial_solution_op<Operators::SymbolicOp<
    SubSpaceViews::Tensor<rank, TrialSolution<dim, spacedim>>,
    OpCode>> : std::true_type
  {};

  template <int                             rank,
            int                             dim,
            int                             spacedim,
            enum Operators::SymbolicOpCodes OpCode,
            types::solution_index           solution_index>
  struct is_field_solution_op<Operators::SymbolicOp<
    SubSpaceViews::Tensor<rank, FieldSolution<dim, spacedim>>,
    OpCode,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>> : std::true_type
  {};

  template <int                             rank,
            int                             dim,
            int                             spacedim,
            enum Operators::SymbolicOpCodes OpCode>
  struct is_test_function_op<Operators::SymbolicOp<
    SubSpaceViews::SymmetricTensor<rank, TestFunction<dim, spacedim>>,
    OpCode>> : std::true_type
  {};

  template <int                             rank,
            int                             dim,
            int                             spacedim,
            enum Operators::SymbolicOpCodes OpCode>
  struct is_trial_solution_op<Operators::SymbolicOp<
    SubSpaceViews::SymmetricTensor<rank, TrialSolution<dim, spacedim>>,
    OpCode>> : std::true_type
  {};

  template <int                             rank,
            int                             dim,
            int                             spacedim,
            enum Operators::SymbolicOpCodes OpCode,
            types::solution_index           solution_index>
  struct is_field_solution_op<Operators::SymbolicOp<
    SubSpaceViews::SymmetricTensor<rank, FieldSolution<dim, spacedim>>,
    OpCode,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>> : std::true_type
  {};



  // Interface operations
  // - Already implemented in spaces.h as a generic trait for all jump/average
  // operations

} // namespace WeakForms


#endif // DOXYGEN


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_spaces_h
