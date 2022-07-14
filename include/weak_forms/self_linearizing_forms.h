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

#ifndef dealii_weakforms_self_linearizing_forms_h
#define dealii_weakforms_self_linearizing_forms_h

#include <deal.II/base/config.h>

#include <weak_forms/config.h>

// #include <boost/core/demangle.hpp>

// TODO: Are all of these needed?
#include <weak_forms/assembler_base.h>
#include <weak_forms/bilinear_forms.h>
#include <weak_forms/differentiation.h>
#include <weak_forms/energy_functor.h>
#include <weak_forms/residual_functor.h>
// #include <weak_forms/functors.h> // Needed?
#include <weak_forms/linear_forms.h>
#include <weak_forms/solution_extraction_data.h>
#include <weak_forms/subspace_extractors.h>
#include <weak_forms/subspace_views.h>
#include <weak_forms/symbolic_integral.h>
#include <weak_forms/symbolic_operators.h>
#include <weak_forms/type_traits.h>

#include <string>
#include <type_traits>


WEAK_FORMS_NAMESPACE_OPEN

#if defined(DEAL_II_WITH_SYMENGINE) || \
  defined(DEAL_II_WITH_AUTO_DIFFERENTIATION)

namespace WeakForms
{
  namespace SelfLinearization
  {
    /**
     * A special form that consumes an energy functor and produces
     * both the associated linear form(s) and consistently linearized
     * bilinear form(s) associated with the energy functional.
     *
     * The @p EnergyFunctional form is supplied with the finite element fields upon
     * which the @p Functor is parameterized. It then self-linearizes the discrete
     * problem (i.e. at the finite element level) to produce the linear and
     * bilinear forms. One linear form is generated for each variable upon which
     * the energy functor is parameterized; each of these linear forms is then
     * linearized with respect to all other variables. So, if the energy functor
     * is parameterized by `n` arguments, then `n` linear forms and `n^2`
     * bilinear forms will be generated.
     *
     * This class, however, doesn't directly know how the energy functor
     * itself is to be linearized: the derivatives of the energy functor
     * with respect to the various field parameters are computed by the energy
     * functor itself. We employ automatic or symbolic differentiation to
     * perform that task. The local description of the energy (i.e. at the
     * quadrature point level) is given by the @p EnergyFunctor.
     *
     * This is fair trade between the convenience of compile-time
     * expansions for the derivatives of the energy functional, and some
     * run-time derivatives of the (potentially complex) constitutive
     * laws that the @p EnergyFunctor describes. The functor is only evaluated
     * at quadrature points, so the computational cost associated with
     * the calculation of those derivatives is kept to a minimum.
     * It also means that we can take care of most of the bookkeeping and
     * implementational details surrounding AD and SD. The user then needs a
     * "minimal" understanding of how these parts of the framework work in order
     * to use this feature.
     *
     * @tparam EnergyFunctor A class that is recognised to be a energy functor
     *         operation, as well as a functor that is either AD compatible
     *         SD compatible (i.e. can exploit either automatic or symbolic
     *         differentiation).
     */
    template <typename EnergyFunctor>
    class EnergyFunctional
    {
      static_assert(is_energy_functor_op<EnergyFunctor>::value,
                    "Expected an EnergyFunctor.");
      static_assert(
        is_ad_functor_op<EnergyFunctor>::value ||
          is_sd_functor_op<EnergyFunctor>::value,
        "The SelfLinearizing::EnergyFunctional class is designed to work with AD or SD functors.");

      // static_assert(
      //   is_symbolic_op<Functor>::value,
      //   "The SelfLinearizing::EnergyFunctional class is designed to work a
      //   unary operation as a functor.");

    public:
      EnergyFunctional(const EnergyFunctor &functor_op)
        : functor_op(functor_op)
      {}

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return "SelfLinearizingEnergyFunctional(" +
               functor_op.as_ascii(decorator) + ")";
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return "SelfLinearizingEnergyFunctional(" +
               functor_op.as_latex(decorator) + ")";
      }

      // ===== Section: Construct assembly operation =====

      UpdateFlags
      get_update_flags() const
      {
        return unpack_update_flags(get_field_args());
      }

      const EnergyFunctor &
      get_functor() const
      {
        return functor_op;
      }

      const auto &
      get_field_args() const
      {
        return get_functor().get_field_args();
      }

      // ===== Section: Integration =====

      template <typename ScalarType = double>
      auto
      dV() const
      {
        return VolumeIntegral().template integrate<ScalarType>(*this);
      }

      template <typename ScalarType = double>
      auto
      dV(const typename VolumeIntegral::subdomain_t subdomain) const
      {
        return dV<ScalarType>(
          std::set<typename VolumeIntegral::subdomain_t>{subdomain});
      }

      template <typename ScalarType = double>
      auto
      dV(const std::set<typename VolumeIntegral::subdomain_t> &subdomains) const
      {
        return VolumeIntegral(subdomains).template integrate<ScalarType>(*this);
      }

      template <typename ScalarType = double>
      auto
      dA() const
      {
        return BoundaryIntegral().template integrate<ScalarType>(*this);
      }

      template <typename ScalarType = double>
      auto
      dA(const typename BoundaryIntegral::subdomain_t boundary) const
      {
        return dA<ScalarType>(
          std::set<typename BoundaryIntegral::subdomain_t>{boundary});
      }

      template <typename ScalarType = double>
      auto
      dA(const std::set<typename BoundaryIntegral::subdomain_t> &boundaries)
        const
      {
        return BoundaryIntegral(boundaries)
          .template integrate<ScalarType>(*this);
      }

      template <typename ScalarType = double>
      auto
      dI() const
      {
        return InterfaceIntegral().template integrate<ScalarType>(*this);
      }

      template <typename ScalarType = double>
      auto
      dI(const typename InterfaceIntegral::subdomain_t interface) const
      {
        return dI<ScalarType>(
          std::set<typename InterfaceIntegral::subdomain_t>{interface});
      }

      template <typename ScalarType = double>
      auto
      dI(const std::set<typename InterfaceIntegral::subdomain_t> &interfaces)
        const
      {
        return InterfaceIntegral(interfaces)
          .template integrate<ScalarType>(*this);
      }

    private:
      const EnergyFunctor functor_op;

      // =============
      // AD operations
      // =============

#  ifdef DEAL_II_WITH_AUTO_DIFFERENTIATION

      template <typename AssemblerScalar_t,
                std::size_t FieldIndex,
                typename SymbolicOpField,
                typename T = EnergyFunctor>
      auto
      get_functor_first_derivative(
        const SymbolicOpField &field,
        typename std::enable_if<is_ad_functor_op<T>::value>::type * =
          nullptr) const
      {
        constexpr int dim      = SymbolicOpField::dimension;
        constexpr int spacedim = SymbolicOpField::space_dimension;

        using EnergyFunctorScalar_t = typename EnergyFunctor::scalar_type;
        using FieldValue_t =
          typename SymbolicOpField::template value_type<AssemblerScalar_t>;
        using DiffOpResult_t = WeakForms::internal::Differentiation::
          DiffOpResult<EnergyFunctorScalar_t, FieldValue_t>;

        using DiffOpValue_t = typename DiffOpResult_t::type;
        using DiffOpFunction_t =
          typename DiffOpResult_t::template function_type<dim>;

        static_assert(
          std::is_same<std::vector<DiffOpValue_t>,
                       typename DiffOpFunction_t::result_type>::value,
          "Expected same result type.");

        // For AD types, the derivative_extractor will be a FEValues::Extractor.
        const EnergyFunctor &functor = this->get_functor();
        const auto &         derivative_extractor =
          functor.template get_derivative_extractor<FieldIndex>(
            field); // Row extractor

        // The functor may only be temporary, so pass it in as a copy.
        // The extractor is specific to this operation, so it definitely
        // must be passed by copy.
        return DiffOpResult_t::template get_functor<dim, spacedim>(
          "Df",
          "D(f)",
          [functor, derivative_extractor](
            MeshWorker::ScratchData<dim, spacedim> &scratch_data,
            const std::vector<SolutionExtractionData<dim, spacedim>>
              &solution_extraction_data)
          {
            (void)solution_extraction_data;
            // We need to fetch the helper from Scratch (rather than passing
            // it into this lambda function) to avoid working with the same copy
            // of this object on multiple threads.
            const auto &helper = functor.get_ad_helper(scratch_data);
            // The return result from the differentiation is also not shared
            // between threads. But we can reuse the same object many times
            // since its stored in Scratch.
            const std::vector<Vector<EnergyFunctorScalar_t>> &gradients =
              functor.get_gradients(scratch_data);

            std::vector<DiffOpValue_t>         out;
            const FEValuesBase<dim, spacedim> &fe_values =
              scratch_data.get_current_fe_values();
            out.reserve(fe_values.n_quadrature_points);

            for (const auto q_point : fe_values.quadrature_point_indices())
              out.emplace_back(
                helper.extract_gradient_component(gradients[q_point],
                                                  derivative_extractor));

            return out;
          },
          functor,
          field);
      }

      template <typename AssemblerScalar_t,
                std::size_t FieldIndex_1,
                std::size_t FieldIndex_2,
                typename SymbolicOpField_1,
                typename SymbolicOpField_2,
                typename T = EnergyFunctor>
      auto
      get_functor_second_derivative(
        const SymbolicOpField_1 &field_1,
        const SymbolicOpField_2 &field_2,
        typename std::enable_if<is_ad_functor_op<T>::value>::type * =
          nullptr) const
      {
        static_assert(SymbolicOpField_1::dimension ==
                        SymbolicOpField_2::dimension,
                      "Dimension mismatch");
        static_assert(SymbolicOpField_1::space_dimension ==
                        SymbolicOpField_2::space_dimension,
                      "Space dimension mismatch");

        constexpr int dim      = SymbolicOpField_1::dimension;
        constexpr int spacedim = SymbolicOpField_1::space_dimension;

        using EnergyFunctorScalar_t = typename EnergyFunctor::scalar_type;
        using FieldValue_1_t =
          typename SymbolicOpField_1::template value_type<AssemblerScalar_t>;
        using FieldValue_2_t =
          typename SymbolicOpField_2::template value_type<AssemblerScalar_t>;
        using FirstDiffOpResult_t = WeakForms::internal::Differentiation::
          DiffOpResult<EnergyFunctorScalar_t, FieldValue_1_t>;
        using SecondDiffOpResult_t = WeakForms::internal::Differentiation::
          DiffOpResult<typename FirstDiffOpResult_t::type, FieldValue_2_t>;

        using DiffOpValue_t = typename SecondDiffOpResult_t::type;
        using DiffOpFunction_t =
          typename SecondDiffOpResult_t::template function_type<dim>;

        static_assert(
          std::is_same<std::vector<DiffOpValue_t>,
                       typename DiffOpFunction_t::result_type>::value,
          "Expected same result type.");

        const EnergyFunctor &functor = this->get_functor();
        const auto &         derivative_1_extractor =
          functor.template get_derivative_extractor<FieldIndex_1>(
            field_1); // Row extractor
        const auto &derivative_2_extractor =
          functor.template get_derivative_extractor<FieldIndex_2>(
            field_2); // Column extractor

        // The functor may only be temporary, so pass it in as a copy.
        // The extractors are specific to this operation, so they definitely
        // must be passed by copy.
        return SecondDiffOpResult_t::template get_functor<dim, spacedim>(
          "D2f",
          "D^{2}(f)",
          [functor, derivative_1_extractor, derivative_2_extractor](
            MeshWorker::ScratchData<dim, spacedim> &scratch_data,
            const std::vector<SolutionExtractionData<dim, spacedim>>
              &solution_extraction_data)
          {
            (void)solution_extraction_data;
            // We need to fetch the helper from Scratch (rather than passing
            // it into this lambda function) to avoid working with the same copy
            // of this object on multiple threads.
            const auto &helper = functor.get_ad_helper(scratch_data);
            // The return result from the differentiation is also not shared
            // between threads. But we can reuse the same object many times
            // since its stored in Scratch.
            const std::vector<FullMatrix<EnergyFunctorScalar_t>> &hessians =
              functor.get_hessians(scratch_data);

            std::vector<DiffOpValue_t>         out;
            const FEValuesBase<dim, spacedim> &fe_values =
              scratch_data.get_current_fe_values();
            out.reserve(fe_values.n_quadrature_points);

            for (const auto q_point : fe_values.quadrature_point_indices())
              out.emplace_back(
                helper.extract_hessian_component(hessians[q_point],
                                                 derivative_1_extractor,
                                                 derivative_2_extractor));

            return out;
          },
          functor,
          field_1,
          field_2);
      }

#  endif // DEAL_II_WITH_AUTO_DIFFERENTIATION

      // =============
      // SD operations
      // =============

#  ifdef DEAL_II_WITH_SYMENGINE

      template <typename AssemblerScalar_t,
                std::size_t FieldIndex,
                typename SymbolicOpField,
                typename T = EnergyFunctor>
      auto
      get_functor_first_derivative(
        const SymbolicOpField &field,
        typename std::enable_if<is_sd_functor_op<T>::value>::type * =
          nullptr) const
      {
        constexpr int dim      = SymbolicOpField::dimension;
        constexpr int spacedim = SymbolicOpField::space_dimension;

        // SD expressions can represent anything, so it doesn't make sense to
        // ask the functor for this type. We expect the result to be castable
        // into the Assembler's scalar type.
        using EnergyFunctorScalar_t = AssemblerScalar_t;
        using FieldValue_t =
          typename SymbolicOpField::template value_type<AssemblerScalar_t>;
        using DiffOpResult_t = WeakForms::internal::Differentiation::
          DiffOpResult<EnergyFunctorScalar_t, FieldValue_t>;

        using DiffOpValue_t = typename DiffOpResult_t::type;
        using DiffOpFunction_t =
          typename DiffOpResult_t::template function_type<dim>;

        static_assert(
          std::is_same<std::vector<DiffOpValue_t>,
                       typename DiffOpFunction_t::result_type>::value,
          "Expected same result type.");

        // For SD types, the derivative_extractor an SD::Expression or tensor of
        // expressions that correspond to the solution field that is being
        // derived with respect to.
        const EnergyFunctor &functor = this->get_functor();
        const auto &         first_derivative =
          functor.template get_symbolic_first_derivative<FieldIndex>();

        // The functor may only be temporary, so pass it in as a copy.
        // The extractor is specific to this operation, so it definitely
        // must be passed by copy.
        return DiffOpResult_t::template get_functor<dim, spacedim>(
          "Df",
          "D(f)",
          [functor, first_derivative](
            MeshWorker::ScratchData<dim, spacedim> &scratch_data,
            const std::vector<SolutionExtractionData<dim, spacedim>>
              &solution_extraction_data)
          {
            (void)solution_extraction_data;
            // We need to fetch the optimizer from Scratch (rather than passing
            // it into this lambda function) to avoid working with the same copy
            // of this object on multiple threads.
            const auto &optimizer =
              functor.template get_batch_optimizer<EnergyFunctorScalar_t>(
                scratch_data);
            // The return result from the differentiation is also not shared
            // between threads. But we can reuse the same object many times
            // since its stored in Scratch.
            const std::vector<std::vector<EnergyFunctorScalar_t>>
              &evaluated_dependent_functions =
                functor.template get_evaluated_dependent_functions<
                  EnergyFunctorScalar_t>(scratch_data);

            std::vector<DiffOpValue_t>         out;
            const FEValuesBase<dim, spacedim> &fe_values =
              scratch_data.get_current_fe_values();
            out.reserve(fe_values.n_quadrature_points);

            // Note: We should not use the evaluated variables that are stored
            // in the optimizer itself. They will only store the values computed
            // at the last call to optimizer.substitute(), which should be the
            // values at the last evaluated quadrature point.
            // We rather follow the same approach as for AD, and store the
            // evaluated variables elsewhere until we want to evaluate them
            // with some centralized optimizer.
            for (const auto q_point : fe_values.quadrature_point_indices())
              out.emplace_back(
                optimizer.extract(first_derivative,
                                  evaluated_dependent_functions[q_point]));

            return out;
          },
          functor,
          field);
      }

      template <typename AssemblerScalar_t,
                std::size_t FieldIndex_1,
                std::size_t FieldIndex_2,
                typename SymbolicOpField_1,
                typename SymbolicOpField_2,
                typename T = EnergyFunctor>
      auto
      get_functor_second_derivative(
        const SymbolicOpField_1 &field_1,
        const SymbolicOpField_2 &field_2,
        typename std::enable_if<is_sd_functor_op<T>::value>::type * =
          nullptr) const
      {
        static_assert(SymbolicOpField_1::dimension ==
                        SymbolicOpField_2::dimension,
                      "Dimension mismatch");
        static_assert(SymbolicOpField_1::space_dimension ==
                        SymbolicOpField_2::space_dimension,
                      "Space dimension mismatch");

        constexpr int dim      = SymbolicOpField_1::dimension;
        constexpr int spacedim = SymbolicOpField_1::space_dimension;

        // SD expressions can represent anything, so it doesn't make sense to
        // ask the functor for this type. We expect the result to be castable
        // into the Assembler's scalar type.
        using EnergyFunctorScalar_t = AssemblerScalar_t;
        using FieldValue_1_t =
          typename SymbolicOpField_1::template value_type<AssemblerScalar_t>;
        using FieldValue_2_t =
          typename SymbolicOpField_2::template value_type<AssemblerScalar_t>;
        using FirstDiffOpResult_t = WeakForms::internal::Differentiation::
          DiffOpResult<EnergyFunctorScalar_t, FieldValue_1_t>;
        using SecondDiffOpResult_t = WeakForms::internal::Differentiation::
          DiffOpResult<typename FirstDiffOpResult_t::type, FieldValue_2_t>;

        using DiffOpValue_t = typename SecondDiffOpResult_t::type;
        using DiffOpFunction_t =
          typename SecondDiffOpResult_t::template function_type<dim>;

        static_assert(
          std::is_same<std::vector<DiffOpValue_t>,
                       typename DiffOpFunction_t::result_type>::value,
          "Expected same result type.");

        const EnergyFunctor &functor = this->get_functor();
        const auto &         second_derivative =
          functor.template get_symbolic_second_derivative<FieldIndex_1,
                                                          FieldIndex_2>();

        // The functor may only be temporary, so pass it in as a copy.
        // The extractors are specific to this operation, so they definitely
        // must be passed by copy.
        return SecondDiffOpResult_t::template get_functor<dim, spacedim>(
          "D2f",
          "D^{2}(f)",
          [functor, second_derivative](
            MeshWorker::ScratchData<dim, spacedim> &scratch_data,
            const std::vector<SolutionExtractionData<dim, spacedim>>
              &solution_extraction_data)
          {
            (void)solution_extraction_data;
            // We need to fetch the optimizer from Scratch (rather than passing
            // it into this lambda function) to avoid working with the same copy
            // of this object on multiple threads.
            const auto &optimizer =
              functor.template get_batch_optimizer<EnergyFunctorScalar_t>(
                scratch_data);
            // The return result from the differentiation is also not shared
            // between threads. But we can reuse the same object many times
            // since its stored in Scratch.
            const std::vector<std::vector<EnergyFunctorScalar_t>>
              &evaluated_dependent_functions =
                functor.template get_evaluated_dependent_functions<
                  EnergyFunctorScalar_t>(scratch_data);

            std::vector<DiffOpValue_t>         out;
            const FEValuesBase<dim, spacedim> &fe_values =
              scratch_data.get_current_fe_values();
            out.reserve(fe_values.n_quadrature_points);

            // Note: We should not use the evaluated variables that are stored
            // in the optimizer itself. They will only store the values computed
            // at the last call to optimizer.substitute(), which should be the
            // values at the last evaluated quadrature point.
            // We rather follow the same approach as for AD, and store the
            // evaluated variables elsewhere until we want to evaluate them
            // with some centralized optimizer.
            for (const auto q_point : fe_values.quadrature_point_indices())
              out.emplace_back(
                optimizer.extract(second_derivative,
                                  evaluated_dependent_functions[q_point]));

            return out;
          },
          functor,
          field_1,
          field_2);
      }

#  endif // DEAL_II_WITH_SYMENGINE

      // =============================
      // Self-linearization operations
      // =============================

      // Provide access to accumulation function
      template <int dim,
                int spacedim,
                typename ScalarType,
                bool use_vectorization,

                std::size_t width>
      friend class WeakForms::AssemblerBase;

      template <enum WeakForms::internal::AccumulationSign OpSign,
                typename AssemblerType,
                typename IntegralType>
      void
      accumulate_into(AssemblerType &     assembler,
                      const IntegralType &integral_operation) const
      {
        unpack_accumulate_linear_form_into<OpSign>(assembler,
                                                   integral_operation,
                                                   get_field_args());

        unpack_accumulate_bilinear_form_into<OpSign>(assembler,
                                                     integral_operation,
                                                     get_field_args(),
                                                     get_field_args());
      }

      // === Recursive function ===
      // All patterns constructed below follow the approach
      // laid out here:
      // https://stackoverflow.com/a/6894436

      // Get update flags from a unary op
      template <std::size_t I = 0, typename... SymbolicOpType>
        inline typename std::enable_if <
        I<sizeof...(SymbolicOpType), UpdateFlags>::type
        unpack_update_flags(const std::tuple<SymbolicOpType...>
                              &symbolic_op_field_solutions) const
      {
        return std::get<I>(symbolic_op_field_solutions).get_update_flags() |
               unpack_update_flags<I + 1, SymbolicOpType...>(
                 symbolic_op_field_solutions);
      }

      // Get update flags from a unary op: End point
      template <std::size_t I = 0, typename... SymbolicOpType>
      inline typename std::enable_if<I == sizeof...(SymbolicOpType),
                                     UpdateFlags>::type
      unpack_update_flags(
        const std::tuple<SymbolicOpType...> &symbolic_op_field_solution) const
      {
        // Do nothing
        (void)symbolic_op_field_solution;
        return UpdateFlags::update_default;
      }

      // Create linear forms
      template <enum WeakForms::internal::AccumulationSign OpSign,
                std::size_t                                I = 0,
                typename AssemblerType,
                typename IntegralType,
                typename... SymbolicOpType>
        inline typename std::enable_if <
        I<sizeof...(SymbolicOpType), void>::type
        unpack_accumulate_linear_form_into(
          AssemblerType &                      assembler,
          const IntegralType &                 integral_operation,
          const std::tuple<SymbolicOpType...> &symbolic_op_field_solutions)
          const
      {
        using AssemblerScalar_t = typename AssemblerType::scalar_type;

        const auto &field_solution_test =
          std::get<I>(symbolic_op_field_solutions);
        const auto test_function =
          internal::ConvertTo::test_function(field_solution_test);

        const auto linear_form = WeakForms::linear_form(
          test_function,
          get_functor_first_derivative<AssemblerScalar_t, I>(
            field_solution_test));
        const auto integrated_linear_form =
          integral_operation.template integrate<AssemblerScalar_t>(linear_form);

        if (OpSign == WeakForms::internal::AccumulationSign::plus)
          {
            assembler += integrated_linear_form;
          }
        else
          {
            Assert(OpSign == WeakForms::internal::AccumulationSign::minus,
                   ExcInternalError());
            assembler -= integrated_linear_form;
          }

        // Move on to the next form:
        // This effectively traverses the list of dependent fields, creating the
        // linear forms associated with the residual starting from first to
        // last.
        unpack_accumulate_linear_form_into<OpSign, I + 1>(
          assembler, integral_operation, symbolic_op_field_solutions);
      }

      // Create linear forms: End point
      template <enum WeakForms::internal::AccumulationSign OpSign,
                std::size_t                                I = 0,
                typename AssemblerType,
                typename IntegralType,
                typename... SymbolicOpType>
      inline typename std::enable_if<I == sizeof...(SymbolicOpType), void>::type
      unpack_accumulate_linear_form_into(
        AssemblerType &                      assembler,
        const IntegralType &                 integral_operation,
        const std::tuple<SymbolicOpType...> &symbolic_op_field_solutions) const
      {
        // Do nothing
        (void)assembler;
        (void)integral_operation;
        (void)symbolic_op_field_solutions;
      }

      // Create bilinear forms
      template <enum WeakForms::internal::AccumulationSign OpSign,
                std::size_t                                I = 0,
                std::size_t                                J = 0,
                typename AssemblerType,
                typename IntegralType,
                typename... SymbolicOpType_1,
                typename... SymbolicOpType_2>
          inline typename std::enable_if < I < sizeof...(SymbolicOpType_1) &&
        J<sizeof...(SymbolicOpType_2), void>::type
        unpack_accumulate_bilinear_form_into(
          AssemblerType &                        assembler,
          const IntegralType &                   integral_operation,
          const std::tuple<SymbolicOpType_1...> &symbolic_op_field_solutions_1,
          const std::tuple<SymbolicOpType_2...> &symbolic_op_field_solutions_2)
          const
      {
        using AssemblerScalar_t = typename AssemblerType::scalar_type;

        const auto &field_solution_test =
          std::get<I>(symbolic_op_field_solutions_1);
        const auto &field_solution_trial =
          std::get<J>(symbolic_op_field_solutions_2);

        // We only allow one solution index, namely that pertaining to
        // the current timestep / Newton iterate and active DoFHandler
        // to be linearized.
        if (field_solution_trial.solution_index !=
            numbers::linearizable_solution_index)
          return;

        const auto test_function =
          internal::ConvertTo::test_function(field_solution_test);
        const auto trial_solution =
          internal::ConvertTo::trial_solution(field_solution_trial);

        // Since we derive from a potential, we can expect the contributions
        // to the linear system to be symmetric.
        const auto bilinear_form =
          WeakForms::bilinear_form(
            test_function,
            get_functor_second_derivative<AssemblerScalar_t, I, J>(
              field_solution_test, field_solution_trial),
            trial_solution)
            .symmetrize();
        const auto integrated_bilinear_form =
          integral_operation.template integrate<AssemblerScalar_t>(
            bilinear_form);

        if (OpSign == WeakForms::internal::AccumulationSign::plus)
          {
            assembler += integrated_bilinear_form;
          }
        else
          {
            Assert(OpSign == WeakForms::internal::AccumulationSign::minus,
                   ExcInternalError());
            assembler -= integrated_bilinear_form;
          }

        // Move on to the next forms:
        // This effectively traverses the list of dependent fields, generating
        // the bilinear forms associated with the linearization.
        //
        // Step 1: Linearize this linear form with respect to all field
        // variables. This basically traverses the row subblock of the system
        // and produces all column subblocks.
        unpack_accumulate_bilinear_form_into<OpSign, I, J + 1>(
          assembler,
          integral_operation,
          symbolic_op_field_solutions_1,
          symbolic_op_field_solutions_2);
        // Step 2: Only move on to the next row if we're at the zeroth column.
        // This is because the above operation traverses all columns in a row.
        if (J == 0)
          unpack_accumulate_bilinear_form_into<OpSign, I + 1, J>(
            assembler,
            integral_operation,
            symbolic_op_field_solutions_1,
            symbolic_op_field_solutions_2);
      }

      // Create bilinear forms: End point
      template <enum WeakForms::internal::AccumulationSign OpSign,
                std::size_t                                I = 0,
                std::size_t                                J = 0,
                typename AssemblerType,
                typename IntegralType,
                typename... SymbolicOpType_1,
                typename... SymbolicOpType_2>
      inline typename std::enable_if<I == sizeof...(SymbolicOpType_1) ||
                                       J == sizeof...(SymbolicOpType_2),
                                     void>::type
      unpack_accumulate_bilinear_form_into(
        AssemblerType &                        assembler,
        const IntegralType &                   integral_operation,
        const std::tuple<SymbolicOpType_1...> &symbolic_op_field_solutions_1,
        const std::tuple<SymbolicOpType_2...> &symbolic_op_field_solutions_2)
        const
      {
        // Do nothing
        (void)assembler;
        (void)integral_operation;
        (void)symbolic_op_field_solutions_1;
        (void)symbolic_op_field_solutions_2;
      }
    }; // class EnergyFunctional


    /**
     * A special form that consumes a residual view functor and produces
     * both the associated linear form (essentially a duplication of the
     * residual form) and its consistently linearized bilinear form(s).
     *
     * The @p ResidualView form is supplied with the finite element fields upon
     * which the @p Functor is parameterized. It then self-linearizes the discrete
     * residual (i.e. at the finite element level) to produce the
     * bilinear form(s). One linear form is generated for component of the
     * residual that the residual view represents; this linear form is then
     * linearized with respect to all other variables upon which the residual
     * view functor is parameterized. So, if the residual view functor is
     * parameterized by `n` arguments, then `1` linear form and `n` bilinear
     * forms will be generated.
     *
     * This class, however, doesn't directly know how the residual view functor
     * itself is to be linearized: the derivatives of the residual functor
     * with respect to the various field parameters are computed by the residual
     * functor itself. We employ automatic or symbolic differentiation to
     * perform that task. The local description of the residual (i.e. at the
     * quadrature point level) is given by the @p ResidualViewFunctor.
     *
     * This is fair trade between the convenience of compile-time
     * expansions for the derivatives of the residual functor, and some
     * run-time derivatives of the (potentially complex) constitutive
     * laws that the @p ResidualViewFunctor describes. The functor is only evaluated
     * at quadrature points, so the computational cost associated with
     * the calculation of those derivatives is kept to a minimum.
     * It also means that we can take care of most of the bookkeeping and
     * implementational details surrounding AD and SD. The user then needs a
     * "minimal" understanding of how these parts of the framework work in order
     * to use this feature.
     *
     * @tparam ResidualViewFunctor A class that is recognised to be a residual
     *         view functor operation, as well as a functor that is either AD
     *         compatible SD compatible (i.e. can exploit either automatic or
     *         symbolic differentiation).
     */
    template <typename ResidualViewFunctor>
    class ResidualView
    {
      static_assert(is_residual_functor_op<ResidualViewFunctor>::value,
                    "Expected a ResidualViewFunctor.");
      static_assert(
        is_ad_functor_op<ResidualViewFunctor>::value ||
          is_sd_functor_op<ResidualViewFunctor>::value,
        "The SelfLinearizing::ResidualView class is designed to work with AD or SD functors.");

      // static_assert(
      //   is_symbolic_op<ResidualViewFunctor>::value,
      //   "The SelfLinearizing::ResidualView class is designed to work a
      //   unary operation as a functor.");

    public:
      ResidualView(const ResidualViewFunctor &functor_op)
        : functor_op(functor_op)
      {}

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return "SelfLinearizingResidualView(" + functor_op.as_ascii(decorator) +
               ")";
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return "SelfLinearizingResidualView(" + functor_op.as_latex(decorator) +
               ")";
      }

      // ===== Section: Construct assembly operation =====

      UpdateFlags
      get_update_flags() const
      {
        return unpack_update_flags(get_field_args());
      }

      const ResidualViewFunctor &
      get_functor() const
      {
        return functor_op;
      }

      const auto &
      get_test_function() const
      {
        return get_functor().get_test_function();
      }

      const auto &
      get_field_args() const
      {
        return get_functor().get_field_args();
      }

      // ===== Section: Integration =====

      template <typename ScalarType = double>
      auto
      dV() const
      {
        return VolumeIntegral().template integrate<ScalarType>(*this);
      }

      template <typename ScalarType = double>
      auto
      dV(const typename VolumeIntegral::subdomain_t subdomain) const
      {
        return dV<ScalarType>(
          std::set<typename VolumeIntegral::subdomain_t>{subdomain});
      }

      template <typename ScalarType = double>
      auto
      dV(const std::set<typename VolumeIntegral::subdomain_t> &subdomains) const
      {
        return VolumeIntegral(subdomains).template integrate<ScalarType>(*this);
      }

      template <typename ScalarType = double>
      auto
      dA() const
      {
        return BoundaryIntegral().template integrate<ScalarType>(*this);
      }

      template <typename ScalarType = double>
      auto
      dA(const typename BoundaryIntegral::subdomain_t boundary) const
      {
        return dA<ScalarType>(
          std::set<typename BoundaryIntegral::subdomain_t>{boundary});
      }

      template <typename ScalarType = double>
      auto
      dA(const std::set<typename BoundaryIntegral::subdomain_t> &boundaries)
        const
      {
        return BoundaryIntegral(boundaries)
          .template integrate<ScalarType>(*this);
      }

      template <typename ScalarType = double>
      auto
      dI() const
      {
        return InterfaceIntegral().template integrate<ScalarType>(*this);
      }

      template <typename ScalarType = double>
      auto
      dI(const typename InterfaceIntegral::subdomain_t interface) const
      {
        return dI<ScalarType>(
          std::set<typename InterfaceIntegral::subdomain_t>{interface});
      }

      template <typename ScalarType = double>
      auto
      dI(const std::set<typename InterfaceIntegral::subdomain_t> &interfaces)
        const
      {
        return InterfaceIntegral(interfaces)
          .template integrate<ScalarType>(*this);
      }

    private:
      const ResidualViewFunctor functor_op;

      // =============
      // AD operations
      // =============

#  ifdef DEAL_II_WITH_AUTO_DIFFERENTIATION

      template <typename AssemblerScalar_t, typename T = ResidualViewFunctor>
      auto
      get_functor_value(
        typename std::enable_if<is_ad_functor_op<T>::value>::type * =
          nullptr) const
      {
        constexpr int dim      = ResidualViewFunctor::dimension;
        constexpr int spacedim = ResidualViewFunctor::space_dimension;

        using ResidualViewFunctorScalar_t =
          typename ResidualViewFunctor::scalar_type;
        using ResidualViewFunctorValue_t =
          typename ResidualViewFunctor::template value_type<
            ResidualViewFunctorScalar_t>;
        using FieldValue_t =
          typename ResidualViewFunctor::template value_type<AssemblerScalar_t>;

        // A little no-op cheat to get us the same interface to the functor
        using ValueOpResult_t = WeakForms::internal::Differentiation::
          DiffOpResult<ResidualViewFunctorValue_t, ResidualViewFunctorScalar_t>;

        using ValueOpValue_t = typename ValueOpResult_t::type;
        using ValueOpFunction_t =
          typename ValueOpResult_t::template function_type<dim>;

        static_assert(
          std::is_same<std::vector<ValueOpValue_t>,
                       typename ValueOpFunction_t::result_type>::value,
          "Expected same result type.");

        // For AD types, the residual_extractor will be a FEValues::Extractor.
        const ResidualViewFunctor &functor = this->get_functor();
        const auto &               residual_extractor =
          functor.get_residual_extractor(); // Row extractor

        // The functor may only be temporary, so pass it in as a copy.
        // The extractor is specific to this operation, so it definitely
        // must be passed by copy.
        return ValueOpResult_t::template get_functor<dim, spacedim>(
          "f",
          "f",
          [functor, residual_extractor](
            MeshWorker::ScratchData<dim, spacedim> &scratch_data,
            const std::vector<SolutionExtractionData<dim, spacedim>>
              &solution_extraction_data)
          {
            (void)solution_extraction_data;
            // We need to fetch the helper from Scratch (rather than passing
            // it into this lambda function) to avoid working with the same copy
            // of this object on multiple threads.
            const auto &helper = functor.get_ad_helper(scratch_data);
            // The return result from the differentiation is also not shared
            // between threads. But we can reuse the same object many times
            // since its stored in Scratch.
            const std::vector<Vector<ResidualViewFunctorScalar_t>> &values =
              functor.get_values(scratch_data);

            std::vector<ValueOpValue_t>        out;
            const FEValuesBase<dim, spacedim> &fe_values =
              scratch_data.get_current_fe_values();
            out.reserve(fe_values.n_quadrature_points);

            for (const auto q_point : fe_values.quadrature_point_indices())
              out.emplace_back(
                helper.extract_value_component(values[q_point],
                                               residual_extractor));

            return out;
          },
          functor);
      }

      template <typename AssemblerScalar_t,
                std::size_t FieldIndex,
                typename SymbolicOpField,
                typename T = ResidualViewFunctor>
      auto
      get_functor_first_derivative(
        const SymbolicOpField &field,
        typename std::enable_if<is_ad_functor_op<T>::value>::type * =
          nullptr) const
      {
        constexpr int dim      = ResidualViewFunctor::dimension;
        constexpr int spacedim = ResidualViewFunctor::space_dimension;

        static_assert(SymbolicOpField::dimension == dim, "Dimension mismatch.");
        static_assert(SymbolicOpField::space_dimension == spacedim,
                      "Spatial dimension mismatch.");

        using ResidualViewFunctorScalar_t =
          typename ResidualViewFunctor::scalar_type;
        using ResidualViewFunctorValue_t =
          typename ResidualViewFunctor::template value_type<
            ResidualViewFunctorScalar_t>;
        using FieldValue_t =
          typename SymbolicOpField::template value_type<AssemblerScalar_t>;
        using DiffOpResult_t = WeakForms::internal::Differentiation::
          DiffOpResult<ResidualViewFunctorValue_t, FieldValue_t>;

        using DiffOpValue_t = typename DiffOpResult_t::type;
        using DiffOpFunction_t =
          typename DiffOpResult_t::template function_type<dim>;

        static_assert(
          std::is_same<std::vector<DiffOpValue_t>,
                       typename DiffOpFunction_t::result_type>::value,
          "Expected same result type.");

        // For AD types, the derivative_extractor will be a FEValues::Extractor.
        const ResidualViewFunctor &functor = this->get_functor();
        const auto &               residual_extractor =
          functor.get_residual_extractor(); // Row extractor
        const auto &derivative_extractor =
          functor.template get_derivative_extractor<FieldIndex>(
            field); // Column extractor

        // The functor may only be temporary, so pass it in as a copy.
        // The extractor is specific to this operation, so it definitely
        // must be passed by copy.
        return DiffOpResult_t::template get_functor<dim, spacedim>(
          "Df",
          "D(f)",
          [functor, residual_extractor, derivative_extractor](
            MeshWorker::ScratchData<dim, spacedim> &scratch_data,
            const std::vector<SolutionExtractionData<dim, spacedim>>
              &solution_extraction_data)
          {
            (void)solution_extraction_data;
            // We need to fetch the helper from Scratch (rather than passing
            // it into this lambda function) to avoid working with the same copy
            // of this object on multiple threads.
            const auto &helper = functor.get_ad_helper(scratch_data);
            // The return result from the differentiation is also not shared
            // between threads. But we can reuse the same object many times
            // since its stored in Scratch.
            const std::vector<FullMatrix<ResidualViewFunctorScalar_t>>
              &jacobians = functor.get_jacobians(scratch_data);

            std::vector<DiffOpValue_t>         out;
            const FEValuesBase<dim, spacedim> &fe_values =
              scratch_data.get_current_fe_values();
            out.reserve(fe_values.n_quadrature_points);

            for (const auto q_point : fe_values.quadrature_point_indices())
              out.emplace_back(helper.extract_jacobian_component(
                jacobians[q_point], residual_extractor, derivative_extractor));

            return out;
          },
          functor,
          field);
      }

#  endif // DEAL_II_WITH_AUTO_DIFFERENTIATION

      // =============
      // SD operations
      // =============

#  ifdef DEAL_II_WITH_SYMENGINE

      template <typename AssemblerScalar_t, typename T = ResidualViewFunctor>
      auto
      get_functor_value(
        typename std::enable_if<is_sd_functor_op<T>::value>::type * =
          nullptr) const
      {
        constexpr int dim      = ResidualViewFunctor::dimension;
        constexpr int spacedim = ResidualViewFunctor::space_dimension;

        // SD expressions can represent anything, so it doesn't make sense to
        // ask the functor for this type. We expect the result to be castable
        // into the Assembler's scalar type.
        using ResidualViewFunctorScalar_t = AssemblerScalar_t;
        using ResidualViewFunctorValue_t =
          typename ResidualViewFunctor::template value_type<
            ResidualViewFunctorScalar_t>;
        using FieldValue_t =
          typename ResidualViewFunctor::template value_type<AssemblerScalar_t>;

        // A little no-op cheat to get us the same interface to the functor
        using ValueOpResult_t = WeakForms::internal::Differentiation::
          DiffOpResult<ResidualViewFunctorValue_t, ResidualViewFunctorScalar_t>;

        using ValueOpValue_t = typename ValueOpResult_t::type;
        using ValueOpFunction_t =
          typename ValueOpResult_t::template function_type<dim>;

        static_assert(
          std::is_same<std::vector<ValueOpValue_t>,
                       typename ValueOpFunction_t::result_type>::value,
          "Expected same result type.");

        // For SD types, the derivative_extractor an SD::Expression or tensor of
        // expressions that correspond to the solution field that is being
        // derived with respect to.
        const ResidualViewFunctor &functor = this->get_functor();
        const auto &               value   = functor.get_symbolic_residual();

        // The functor may only be temporary, so pass it in as a copy.
        // The extractor is specific to this operation, so it definitely
        // must be passed by copy.
        return ValueOpResult_t::template get_functor<dim, spacedim>(
          "f",
          "f",
          [functor,
           value](MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                  const std::vector<SolutionExtractionData<dim, spacedim>>
                    &solution_extraction_data)
          {
            (void)solution_extraction_data;
            // We need to fetch the optimizer from Scratch (rather than passing
            // it into this lambda function) to avoid working with the same copy
            // of this object on multiple threads.
            const auto &optimizer =
              functor.template get_batch_optimizer<ResidualViewFunctorScalar_t>(
                scratch_data);
            // The return result from the differentiation is also not shared
            // between threads. But we can reuse the same object many times
            // since its stored in Scratch.
            const std::vector<std::vector<ResidualViewFunctorScalar_t>>
              &evaluated_dependent_functions =
                functor.template get_evaluated_dependent_functions<
                  ResidualViewFunctorScalar_t>(scratch_data);

            std::vector<ValueOpValue_t>        out;
            const FEValuesBase<dim, spacedim> &fe_values =
              scratch_data.get_current_fe_values();
            out.reserve(fe_values.n_quadrature_points);

            // Note: We should not use the evaluated variables that are stored
            // in the optimizer itself. They will only store the values computed
            // at the last call to optimizer.substitute(), which should be the
            // values at the last evaluated quadrature point.
            // We rather follow the same approach as for AD, and store the
            // evaluated variables elsewhere until we want to evaluate them
            // with some centralized optimizer.
            for (const auto q_point : fe_values.quadrature_point_indices())
              out.emplace_back(
                optimizer.extract(value,
                                  evaluated_dependent_functions[q_point]));

            return out;
          },
          functor);
      }

      template <typename AssemblerScalar_t,
                std::size_t FieldIndex,
                typename SymbolicOpField,
                typename T = ResidualViewFunctor>
      auto
      get_functor_first_derivative(
        const SymbolicOpField &field,
        typename std::enable_if<is_sd_functor_op<T>::value>::type * =
          nullptr) const
      {
        constexpr int dim      = ResidualViewFunctor::dimension;
        constexpr int spacedim = ResidualViewFunctor::space_dimension;

        static_assert(SymbolicOpField::dimension == dim, "Dimension mismatch.");
        static_assert(SymbolicOpField::space_dimension == spacedim,
                      "Spatial dimension mismatch.");

        // SD expressions can represent anything, so it doesn't make sense to
        // ask the functor for this type. We expect the result to be castable
        // into the Assembler's scalar type.
        using ResidualViewFunctorScalar_t = AssemblerScalar_t;
        using ResidualViewFunctorValue_t =
          typename ResidualViewFunctor::template value_type<
            ResidualViewFunctorScalar_t>;
        using FieldValue_t =
          typename SymbolicOpField::template value_type<AssemblerScalar_t>;
        using DiffOpResult_t = WeakForms::internal::Differentiation::
          DiffOpResult<ResidualViewFunctorValue_t, FieldValue_t>;

        using DiffOpValue_t = typename DiffOpResult_t::type;
        using DiffOpFunction_t =
          typename DiffOpResult_t::template function_type<dim>;

        static_assert(
          std::is_same<std::vector<DiffOpValue_t>,
                       typename DiffOpFunction_t::result_type>::value,
          "Expected same result type.");

        // For SD types, the derivative_extractor an SD::Expression or tensor of
        // expressions that correspond to the solution field that is being
        // derived with respect to.
        const ResidualViewFunctor &functor = this->get_functor();
        const auto &               first_derivative =
          functor.template get_symbolic_first_derivative<FieldIndex>();

        // The functor may only be temporary, so pass it in as a copy.
        // The extractor is specific to this operation, so it definitely
        // must be passed by copy.
        return DiffOpResult_t::template get_functor<dim, spacedim>(
          "Df",
          "D(f)",
          [functor, first_derivative](
            MeshWorker::ScratchData<dim, spacedim> &scratch_data,
            const std::vector<SolutionExtractionData<dim, spacedim>>
              &solution_extraction_data)
          {
            (void)solution_extraction_data;
            // We need to fetch the optimizer from Scratch (rather than passing
            // it into this lambda function) to avoid working with the same copy
            // of this object on multiple threads.
            const auto &optimizer =
              functor.template get_batch_optimizer<ResidualViewFunctorScalar_t>(
                scratch_data);
            // The return result from the differentiation is also not shared
            // between threads. But we can reuse the same object many times
            // since its stored in Scratch.
            const std::vector<std::vector<ResidualViewFunctorScalar_t>>
              &evaluated_dependent_functions =
                functor.template get_evaluated_dependent_functions<
                  ResidualViewFunctorScalar_t>(scratch_data);

            std::vector<DiffOpValue_t>         out;
            const FEValuesBase<dim, spacedim> &fe_values =
              scratch_data.get_current_fe_values();
            out.reserve(fe_values.n_quadrature_points);

            // Note: We should not use the evaluated variables that are stored
            // in the optimizer itself. They will only store the values computed
            // at the last call to optimizer.substitute(), which should be the
            // values at the last evaluated quadrature point.
            // We rather follow the same approach as for AD, and store the
            // evaluated variables elsewhere until we want to evaluate them
            // with some centralized optimizer.
            for (const auto q_point : fe_values.quadrature_point_indices())
              out.emplace_back(
                optimizer.extract(first_derivative,
                                  evaluated_dependent_functions[q_point]));

            return out;
          },
          functor,
          field);
      }

#  endif // DEAL_II_WITH_SYMENGINE

      // =============================
      // Self-linearization operations
      // =============================

      // Provide access to accumulation function
      template <int dim,
                int spacedim,
                typename ScalarType,
                bool use_vectorization,

                std::size_t width>
      friend class WeakForms::AssemblerBase;

      template <enum WeakForms::internal::AccumulationSign OpSign,
                typename AssemblerType,
                typename IntegralType>
      void
      accumulate_into(AssemblerType &     assembler,
                      const IntegralType &integral_operation) const
      {
        accumulate_linear_form_into<OpSign>(assembler, integral_operation);
        unpack_accumulate_bilinear_form_into<OpSign>(assembler,
                                                     integral_operation,
                                                     get_field_args());
      }

      // === Recursive function ===
      // All patterns constructed below follow the approach
      // laid out here:
      // https://stackoverflow.com/a/6894436

      // Get update flags from a unary op
      template <std::size_t I = 0, typename... SymbolicOpType>
        inline typename std::enable_if <
        I<sizeof...(SymbolicOpType), UpdateFlags>::type
        unpack_update_flags(const std::tuple<SymbolicOpType...>
                              &symbolic_op_field_solutions) const
      {
        return std::get<I>(symbolic_op_field_solutions).get_update_flags() |
               unpack_update_flags<I + 1, SymbolicOpType...>(
                 symbolic_op_field_solutions);
      }

      // Get update flags from a unary op: End point
      template <std::size_t I = 0, typename... SymbolicOpType>
      inline typename std::enable_if<I == sizeof...(SymbolicOpType),
                                     UpdateFlags>::type
      unpack_update_flags(
        const std::tuple<SymbolicOpType...> &symbolic_op_field_solution) const
      {
        // Do nothing
        (void)symbolic_op_field_solution;
        return UpdateFlags::update_default;
      }

      // Create linear form
      template <enum WeakForms::internal::AccumulationSign OpSign,
                typename AssemblerType,
                typename IntegralType>
      void
      accumulate_linear_form_into(AssemblerType &     assembler,
                                  const IntegralType &integral_operation) const
      {
        using AssemblerScalar_t = typename AssemblerType::scalar_type;

        const auto test_function = get_test_function();

        const auto linear_form =
          WeakForms::linear_form(test_function,
                                 get_functor_value<AssemblerScalar_t>());
        const auto integrated_linear_form =
          integral_operation.template integrate<AssemblerScalar_t>(linear_form);

        if (OpSign == WeakForms::internal::AccumulationSign::plus)
          {
            assembler += integrated_linear_form;
          }
        else
          {
            Assert(OpSign == WeakForms::internal::AccumulationSign::minus,
                   ExcInternalError());
            assembler -= integrated_linear_form;
          }
      }

      // Create bilinear forms
      template <enum WeakForms::internal::AccumulationSign OpSign,
                std::size_t                                J = 0,
                typename AssemblerType,
                typename IntegralType,
                typename... SymbolicOpType>
        inline typename std::enable_if <
        J<sizeof...(SymbolicOpType), void>::type
        unpack_accumulate_bilinear_form_into(
          AssemblerType &                      assembler,
          const IntegralType &                 integral_operation,
          const std::tuple<SymbolicOpType...> &symbolic_op_field_solutions)
          const
      {
        using AssemblerScalar_t = typename AssemblerType::scalar_type;

        const auto &field_solution_trial =
          std::get<J>(symbolic_op_field_solutions);

        // We only allow one solution index, namely that pertaining to
        // the current timestep / Newton iterate and active DoFHandler
        // to be linearized.
        if (field_solution_trial.solution_index !=
            numbers::linearizable_solution_index)
          return;

        const auto test_function = get_test_function();
        const auto trial_solution =
          internal::ConvertTo::trial_solution(field_solution_trial);

        const auto bilinear_form = WeakForms::bilinear_form(
          test_function,
          get_functor_first_derivative<AssemblerScalar_t, J>(
            field_solution_trial),
          trial_solution);
        const auto integrated_bilinear_form =
          integral_operation.template integrate<AssemblerScalar_t>(
            bilinear_form);

        if (OpSign == WeakForms::internal::AccumulationSign::plus)
          {
            assembler += integrated_bilinear_form;
          }
        else
          {
            Assert(OpSign == WeakForms::internal::AccumulationSign::minus,
                   ExcInternalError());
            assembler -= integrated_bilinear_form;
          }

        // Move on to the next forms:
        // This effectively traverses the list of dependent fields, generating
        // the bilinear forms associated with the linearization.
        unpack_accumulate_bilinear_form_into<OpSign, J + 1>(
          assembler, integral_operation, symbolic_op_field_solutions);
      }

      // Create bilinear forms: End point
      template <enum WeakForms::internal::AccumulationSign OpSign,
                std::size_t                                J = 0,
                typename AssemblerType,
                typename IntegralType,
                typename... SymbolicOpType>
      inline typename std::enable_if<J == sizeof...(SymbolicOpType), void>::type
      unpack_accumulate_bilinear_form_into(
        AssemblerType &                      assembler,
        const IntegralType &                 integral_operation,
        const std::tuple<SymbolicOpType...> &symbolic_op_field_solutions) const
      {
        // Do nothing
        (void)assembler;
        (void)integral_operation;
        (void)symbolic_op_field_solutions;
      }
    }; // class ResidualView
  }    // namespace SelfLinearization

} // namespace WeakForms



/* ======================== Convenience functions ======================== */


namespace WeakForms
{
  /**
   * A convenience function that generates a self-linearizing energy
   * functional form from an energy functor.
   *
   * For more information about the self-linearizing form that is created,
   * please refer to the documentation of the
   * SelfLinearization::EnergyFunctional class.
   *
   * @tparam EnergyFunctor A class that is recognised to be a energy functor
   *         operation, as well as a functor that is either AD compatible
   *         SD compatible (i.e. can exploit either automatic or symbolic
   *         differentiation).
   * @param functor_op An energy functor that is to be converted to a form.
   * @return SelfLinearization::EnergyFunctional<EnergyFunctor>
   */
  template <typename EnergyFunctor,
            typename = typename std::enable_if<
              is_energy_functor_op<EnergyFunctor>::value &&
              (is_ad_functor_op<EnergyFunctor>::value ||
               is_sd_functor_op<EnergyFunctor>::value)>::type>
  SelfLinearization::EnergyFunctional<EnergyFunctor>
  energy_functional_form(const EnergyFunctor &functor_op)
  {
    return SelfLinearization::EnergyFunctional<EnergyFunctor>(functor_op);
  }


  /**
   * A convenience function that generates a self-linearizing residual
   * form from a residual view functor.
   *
   * For more information about the self-linearizing form that is created,
   * please refer to the documentation of the SelfLinearization::ResidualView
   * class.
   *
   * @tparam ResidualViewFunctor A class that is recognised to be a residual view
   *         functor operation, as well as a functor that is either AD
   *         compatible SD compatible (i.e. can exploit either automatic or
   *         symbolic differentiation).
   * @param functor_op An residual functor that is to be converted to a form.
   * @return SelfLinearization::ResidualView<ResidualViewFunctor>
   */
  template <typename ResidualViewFunctor,
            typename = typename std::enable_if<
              is_residual_functor_op<ResidualViewFunctor>::value &&
              (is_ad_functor_op<ResidualViewFunctor>::value ||
               is_sd_functor_op<ResidualViewFunctor>::value)>::type>
  SelfLinearization::ResidualView<ResidualViewFunctor>
  residual_form(const ResidualViewFunctor &functor_op)
  {
    return SelfLinearization::ResidualView<ResidualViewFunctor>(functor_op);
  }

} // namespace WeakForms



/* ==================== Specialization of type traits ==================== */



#  ifndef DOXYGEN


namespace WeakForms
{
  template <typename EnergyFunctor>
  struct is_self_linearizing_form<
    SelfLinearization::EnergyFunctional<EnergyFunctor>> : std::true_type
  {};


  template <typename ResidualViewFunctor>
  struct is_self_linearizing_form<
    SelfLinearization::ResidualView<ResidualViewFunctor>> : std::true_type
  {};

} // namespace WeakForms


#  endif // DOXYGEN


#endif // defined(DEAL_II_WITH_SYMENGINE) ||
       // defined(DEAL_II_WITH_AUTO_DIFFERENTIATION)

WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_self_linearizing_forms_h
