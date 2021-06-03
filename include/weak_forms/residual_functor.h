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

#ifndef dealii_weakforms_residual_functor_h
#define dealii_weakforms_residual_functor_h

#include <deal.II/base/config.h>

#include <deal.II/algorithms/general_data_storage.h>

#include <deal.II/base/exceptions.h>

#include <deal.II/differentiation/ad.h>
#include <deal.II/differentiation/sd.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/meshworker/scratch_data.h>

#include <weak_forms/ad_sd_functor_internal.h>
#include <weak_forms/config.h>
#include <weak_forms/differentiation.h>
#include <weak_forms/functors.h>
#include <weak_forms/residual_functor.h>
#include <weak_forms/solution_storage.h>
#include <weak_forms/subspace_extractors.h>
#include <weak_forms/subspace_views.h>
#include <weak_forms/type_traits.h>
#include <weak_forms/utilities.h>

#include <tuple>


WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
  template <typename TestSpaceOp, typename... SymbolicOpsSubSpaceFieldSolution>
  class ResidualViewFunctor;


  template <typename... SymbolicOpsSubSpaceFieldSolution>
  class ResidualFunctor : public WeakForms::Functor<1>
  {
    using Base = WeakForms::Functor<1>; // Residual is a vector
  public:
    ResidualFunctor(
      const std::string &symbol_ascii,
      const std::string &symbol_latex,
      const SymbolicOpsSubSpaceFieldSolution &... symbolic_op_field_solutions)
      : Base(symbol_ascii, symbol_latex)
      , symbolic_op_field_solutions(symbolic_op_field_solutions...)
    {}

    // ----  Ascii ----

    virtual std::string
    as_ascii(const SymbolicDecorations &decorator) const override
    {
      return Base::as_ascii(decorator) + "(" +
             decorator.unary_field_ops_as_ascii(get_field_args()) + ")";
    }

    virtual std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const override
    {
      (void)decorator;
      return symbol_ascii;
    }

    // ---- LaTeX ----

    virtual std::string
    as_latex(const SymbolicDecorations &decorator) const override
    {
      return Utilities::LaTeX::decorate_function_with_arguments(
        Base::as_latex(decorator),
        decorator.unary_field_ops_as_latex(get_field_args()));
    }

    virtual std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const override
    {
      (void)decorator;
      return symbol_latex;
    }

    template <typename TestSpaceOp>
    ResidualViewFunctor<TestSpaceOp, SymbolicOpsSubSpaceFieldSolution...>
    operator[](const TestSpaceOp &test_space_op) const
    {
      return ResidualViewFunctor<TestSpaceOp,
                                 SymbolicOpsSubSpaceFieldSolution...>(
        symbol_ascii, symbol_latex, *this, test_space_op);
    }

    // Independent fields
    const std::tuple<SymbolicOpsSubSpaceFieldSolution...> &
    get_field_args() const
    {
      return symbolic_op_field_solutions;
    }

  private:
    const std::tuple<SymbolicOpsSubSpaceFieldSolution...>
      symbolic_op_field_solutions;
  };



  template <typename TestSpaceOp, typename... SymbolicOpsSubSpaceFieldSolution>
  class ResidualViewFunctor : public WeakForms::Functor<1>
  {
    static_assert(is_or_has_test_function_op<TestSpaceOp>::value,
                  "Expected a test function.");
    using Base = WeakForms::Functor<1>; // Residual is a vector

  public:
    /**
     * Dimension in which this object operates.
     */
    static const unsigned int dimension = TestSpaceOp::dimension;

    /**
     * Dimension of the subspace in which this object operates.
     */
    static const unsigned int space_dimension = TestSpaceOp::space_dimension;

    using ResidualFunctorType =
      ResidualFunctor<SymbolicOpsSubSpaceFieldSolution...>;

    // Note: The dimension of the value_type is embedded in the
    // test function op itself.
    template <typename ADorSDNumberType>
    using value_type =
      typename TestSpaceOp::template value_type<ADorSDNumberType>;

    template <typename ScalarType,
              enum Differentiation::AD::NumberTypes ADNumberTypeCode>
    using ad_type =
      typename Differentiation::AD::NumberTraits<ScalarType,
                                                 ADNumberTypeCode>::ad_type;

    template <typename ADNumberType, int dim, int spacedim = dim>
    using ad_function_type = std::function<value_type<ADNumberType>(
      const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      const std::vector<std::string> &              solution_names,
      const unsigned int                            q_point,
      const typename SymbolicOpsSubSpaceFieldSolution::template value_type<
        ADNumberType> &... field_solutions)>;

    template <typename ScalarType>
    using sd_type               = Differentiation::SD::Expression;
    using substitution_map_type = Differentiation::SD::types::substitution_map;

    template <typename SDNumberType, int dim, int spacedim = dim>
    using sd_function_type = std::function<value_type<SDNumberType>(
      const typename SymbolicOpsSubSpaceFieldSolution::template value_type<
        SDNumberType> &... field_solutions)>;

    template <typename SDNumberType, int dim, int spacedim = dim>
    using sd_intermediate_substitution_function_type =
      std::function<substitution_map_type(
        const typename SymbolicOpsSubSpaceFieldSolution::template value_type<
          SDNumberType> &... field_solutions)>;

    // This also allows the user to encode symbols/parameters in terms of
    // the (symbolic) field variables, for which we'll supply the values.
    template <typename SDNumberType, int dim, int spacedim = dim>
    using sd_register_symbols_function_type =
      std::function<substitution_map_type(
        const typename SymbolicOpsSubSpaceFieldSolution::template value_type<
          SDNumberType> &... field_solutions)>;

    template <typename SDNumberType, int dim, int spacedim = dim>
    using sd_substitution_function_type = std::function<substitution_map_type(
      const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      const std::vector<std::string> &              solution_names,
      const unsigned int                            q_point)>;

    explicit ResidualViewFunctor(const std::string &        symbol_ascii,
                                 const std::string &        symbol_latex,
                                 const ResidualFunctorType &residual,
                                 const TestSpaceOp &        test_space_op)
      : Base(symbol_ascii, symbol_latex)
      , residual(residual)
      , test_space_op(test_space_op)
    {}

    ResidualViewFunctor(const ResidualViewFunctor &) = default;

    // ----  Ascii ----

    virtual std::string
    as_ascii(const SymbolicDecorations &decorator) const override
    {
      return Base::as_ascii(decorator) + "_[" +
             test_space_op.as_ascii(decorator) + "]" + "(" +
             decorator.unary_field_ops_as_ascii(get_field_args()) + ")";
    }

    virtual std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const override
    {
      (void)decorator;
      return symbol_ascii;
    }

    // ---- LaTeX ----

    virtual std::string
    as_latex(const SymbolicDecorations &decorator) const override
    {
      return Utilities::LaTeX::decorate_function_with_arguments(
        Utilities::LaTeX::decorate_subscript(Base::as_latex(decorator),
                                             test_space_op.as_latex(decorator)),
        decorator.unary_field_ops_as_latex(get_field_args()));
    }

    virtual std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const override
    {
      (void)decorator;
      return symbol_latex;
    }

    // Call operator to promote this class to a SymbolicOp
    template <typename ADNumberType, int dim, int spacedim = dim>
    auto
    operator()(const ad_function_type<ADNumberType, dim, spacedim> &function,
               const UpdateFlags update_flags) const;

    template <typename SDNumberType, int dim, int spacedim = dim>
    auto
    operator()(
      const sd_function_type<SDNumberType, dim, spacedim> &function,
      const sd_register_symbols_function_type<SDNumberType, dim, spacedim>
        symbol_registration_map,
      const sd_substitution_function_type<SDNumberType, dim, spacedim>
        substitution_map,
      const sd_intermediate_substitution_function_type<SDNumberType,
                                                       dim,
                                                       spacedim>
                                                        intermediate_substitution_map,
      const enum Differentiation::SD::OptimizerType     optimization_method,
      const enum Differentiation::SD::OptimizationFlags optimization_flags,
      const UpdateFlags                                 update_flags) const;

    // Let's give our users a nicer syntax to work with this
    // templated call operator.
    template <typename ADNumberType, int dim, int spacedim = dim>
    auto
    value(const ad_function_type<ADNumberType, dim, spacedim> &function,
          const UpdateFlags update_flags) const
    {
      return this->template operator()<ADNumberType, dim, spacedim>(
        function, update_flags);
    }

    template <typename ADNumberType, int dim, int spacedim = dim>
    auto
    value(const ad_function_type<ADNumberType, dim, spacedim> &function) const
    {
      return this->template operator()<ADNumberType, dim, spacedim>(
        function, UpdateFlags::update_default);
    }

    template <typename SDNumberType, int dim, int spacedim = dim>
    auto
    value(const sd_function_type<SDNumberType, dim, spacedim> &function,
          const sd_register_symbols_function_type<SDNumberType, dim, spacedim>
            symbol_registration_map,
          const sd_substitution_function_type<SDNumberType, dim, spacedim>
            substitution_map,
          const sd_intermediate_substitution_function_type<SDNumberType,
                                                           dim,
                                                           spacedim>
                                                            intermediate_substitution_map,
          const enum Differentiation::SD::OptimizerType     optimization_method,
          const enum Differentiation::SD::OptimizationFlags optimization_flags,
          const UpdateFlags                                 update_flags) const;

    template <typename SDNumberType, int dim, int spacedim = dim>
    auto
    value(const sd_function_type<SDNumberType, dim, spacedim> &function,
          const enum Differentiation::SD::OptimizerType     optimization_method,
          const enum Differentiation::SD::OptimizationFlags optimization_flags)
      const
    {
      const sd_register_symbols_function_type<SDNumberType, dim, spacedim>
        dummy_symbol_registration_map;
      const sd_substitution_function_type<SDNumberType, dim, spacedim>
        dummy_substitution_map;
      const sd_intermediate_substitution_function_type<SDNumberType,
                                                       dim,
                                                       spacedim>
        dummy_intermediate_substitution_map;

      return this->template operator()<SDNumberType, dim, spacedim>(
        function,
        dummy_symbol_registration_map,
        dummy_substitution_map,
        dummy_intermediate_substitution_map,
        optimization_method,
        optimization_flags,
        UpdateFlags::update_default);
    }

    template <typename SDNumberType, int dim, int spacedim = dim>
    auto
    value(const sd_function_type<SDNumberType, dim, spacedim> &function,
          const sd_register_symbols_function_type<SDNumberType, dim, spacedim>
            symbol_registration_map,
          const sd_substitution_function_type<SDNumberType, dim, spacedim>
                                                            substitution_map,
          const enum Differentiation::SD::OptimizerType     optimization_method,
          const enum Differentiation::SD::OptimizationFlags optimization_flags,
          const UpdateFlags                                 update_flags) const
    {
      const sd_intermediate_substitution_function_type<SDNumberType,
                                                       dim,
                                                       spacedim>
        dummy_intermediate_substitution_map;

      return this->template operator()<SDNumberType, dim, spacedim>(
        function,
        symbol_registration_map,
        substitution_map,
        dummy_intermediate_substitution_map,
        optimization_method,
        optimization_flags,
        update_flags);
    }

    template <typename SDNumberType, int dim, int spacedim = dim>
    auto
    value(const sd_function_type<SDNumberType, dim, spacedim> &function,
          const sd_register_symbols_function_type<SDNumberType, dim, spacedim>
            symbol_registration_map,
          const sd_substitution_function_type<SDNumberType, dim, spacedim>
                                                            substitution_map,
          const enum Differentiation::SD::OptimizerType     optimization_method,
          const enum Differentiation::SD::OptimizationFlags optimization_flags)
      const
    {
      const sd_intermediate_substitution_function_type<SDNumberType,
                                                       dim,
                                                       spacedim>
        dummy_intermediate_substitution_map;

      return this->template operator()<SDNumberType, dim, spacedim>(
        function,
        symbol_registration_map,
        substitution_map,
        dummy_intermediate_substitution_map,
        optimization_method,
        optimization_flags,
        UpdateFlags::update_default);
    }

    typename Operators::internal::SpaceOpComponentInfo<
      TestSpaceOp>::extractor_type
    get_test_space_extractor() const
    {
      const unsigned int first_component = 0;
      return typename Operators::internal::SpaceOpComponentInfo<
        TestSpaceOp>::extractor_type(first_component);
    }

    const TestSpaceOp &
    get_test_function() const
    {
      return test_space_op;
    }

    // Independent fields
    const std::tuple<SymbolicOpsSubSpaceFieldSolution...> &
    get_field_args() const
    {
      return residual.get_field_args();
    }

  protected:
    // Allow access to get_space()
    // friend WeakForms::SelfLinearization::internal::ConvertTo;

    const ResidualFunctorType &
    get_residual_functor() const
    {
      return residual;
    }

  private:
    const ResidualFunctorType residual;
    const TestSpaceOp         test_space_op;
  };

} // namespace WeakForms



// /* ======================== Convenience functions ======================== */



namespace WeakForms
{
  /**
   * Shortcut so that we don't need to do something like this:
   *
   * <code>
   * const FieldSolution<dim> solution;
   * const WeakForms::SubSpaceExtractors::Scalar subspace_extractor(0, "s",
   * "s");
   *
   * const auto soln_ss   = solution[subspace_extractor];
   * const auto soln_val  = soln_ss.value();    // Solution value
   * const auto soln_grad = soln_ss.gradient(); // Solution gradient
   * ...
   *
   * const ResidualFunctor<decltype(soln_val), decltype(soln_grad), ...>
   * residual("R", "R", soln_val, soln_grad, ...);
   * </code>
   */
  template <typename... SymbolicOpsSubSpaceFieldSolution>
  ResidualFunctor<SymbolicOpsSubSpaceFieldSolution...>
  residual_functor(
    const std::string &symbol_ascii,
    const std::string &symbol_latex,
    const SymbolicOpsSubSpaceFieldSolution &... symbolic_op_field_solutions)
  {
    return ResidualFunctor<SymbolicOpsSubSpaceFieldSolution...>(
      symbol_ascii, symbol_latex, symbolic_op_field_solutions...);
  }


  template <typename TestSpaceOp, typename... SymbolicOpsSubSpaceFieldSolution>
  ResidualViewFunctor<SymbolicOpsSubSpaceFieldSolution...>
  residual_view_functor(
    const std::string &symbol_ascii,
    const std::string &symbol_latex,
    const TestSpaceOp &test_space_op,
    const SymbolicOpsSubSpaceFieldSolution &... symbolic_op_field_solutions)
  {
    return residual_functor(symbol_ascii,
                            symbol_latex,
                            symbolic_op_field_solutions...)[test_space_op];
  }
} // namespace WeakForms



// /* ================== Specialization of unary operators ================== */



namespace WeakForms
{
  namespace Operators
  {
    /* ------------------------ Functors: Custom ------------------------ */

    /**
     * Extract the value from a residual view functor.
     *
     * Variant for auto-differentiable number.
     */
    template <typename ADNumberType,
              int dim,
              int spacedim,
              typename TestSpaceOp,
              typename... SymbolicOpsSubSpaceFieldSolution>
    class SymbolicOp<
      ResidualViewFunctor<TestSpaceOp, SymbolicOpsSubSpaceFieldSolution...>,
      SymbolicOpCodes::value,
      typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type,
      ADNumberType,
      WeakForms::internal::DimPack<dim, spacedim>>
    {
      static_assert(is_or_has_test_function_op<TestSpaceOp>::value,
                    "Expected a test function.");
      static_assert(TestSpaceOp::dimension == dim, "Dimension mismatch.");
      static_assert(TestSpaceOp::space_dimension == spacedim,
                    "Spatial dimension mismatch.");
      static_assert(Differentiation::AD::is_ad_number<ADNumberType>::value,
                    "Expected an AD number.");

      using Op =
        ResidualViewFunctor<TestSpaceOp, SymbolicOpsSubSpaceFieldSolution...>;

      using OpHelper_t = internal::SymbolicOpsSubSpaceFieldSolutionHelper<
        SymbolicOpsSubSpaceFieldSolution...>;

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = Op::dimension;

      /**
       * Dimension of the subspace in which this object operates.
       */
      static const unsigned int space_dimension = Op::space_dimension;

      using scalar_type =
        typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type;

      static constexpr enum Differentiation::AD::NumberTypes ADNumberTypeCode =
        Differentiation::AD::ADNumberTraits<ADNumberType>::type_code;

      using ad_helper_type = Differentiation::AD::
        VectorFunction<spacedim, ADNumberTypeCode, scalar_type>;
      using ad_type = typename ad_helper_type::ad_type;

      static_assert(
        std::is_same<typename Differentiation::AD::
                       NumberTraits<scalar_type, ADNumberTypeCode>::ad_type,
                     ADNumberType>::value,
        "AD types not the same.");
      static_assert(std::is_same<ad_type, ADNumberType>::value,
                    "AD types not the same.");

      template <typename ResultScalarType>
      using value_type = typename Op::template value_type<ResultScalarType>;

      template <typename ResultScalarType>
      using function_type =
        typename Op::template ad_function_type<ResultScalarType, dim, spacedim>;

      template <typename ResultScalarType>
      using return_type = void;
      // using return_type = std::vector<value_type<ResultScalarType>>;

      using residual_type    = value_type<ad_type>;
      using ad_function_type = function_type<ad_type>;

      static const int rank = Op::rank;

      static const enum SymbolicOpCodes op_code = SymbolicOpCodes::value;

      explicit SymbolicOp(const Op &              operand,
                          const ad_function_type &function,
                          const UpdateFlags       update_flags)
        : operand(operand)
        , function(function)
        , update_flags(update_flags)
        , extractors(OpHelper_t::get_initialized_extractors())
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
        return update_flags;
      }

      const ad_helper_type &
      get_ad_helper(
        const MeshWorker::ScratchData<dim, spacedim> &scratch_data) const
      {
        const GeneralDataStorage &cache =
          scratch_data.get_general_data_storage();

        return cache.get_object_with_name<ad_helper_type>(get_name_ad_helper());
      }

      const TestSpaceOp &
      get_test_function() const
      {
        return get_op().get_test_function();
      }

      auto
      get_residual_extractor() const
      {
        return get_op().get_test_space_extractor();
      }

      template <std::size_t FieldIndex, typename SymbolicOpField>
      typename Operators::internal::SpaceOpComponentInfo<
        SymbolicOpField>::extractor_type
      get_derivative_extractor(const SymbolicOpField &) const
      {
        static_assert(FieldIndex < OpHelper_t::n_operators(),
                      "Index out of bounds.");
        return std::get<FieldIndex>(get_field_extractors());
      }

      const std::vector<Vector<scalar_type>> &
      get_values(
        const MeshWorker::ScratchData<dim, spacedim> &scratch_data) const
      {
        const GeneralDataStorage &cache =
          scratch_data.get_general_data_storage();

        return cache.get_object_with_name<std::vector<Vector<scalar_type>>>(
          get_name_value());
      }

      const std::vector<FullMatrix<scalar_type>> &
      get_jacobians(
        const MeshWorker::ScratchData<dim, spacedim> &scratch_data) const
      {
        const GeneralDataStorage &cache =
          scratch_data.get_general_data_storage();

        return cache.get_object_with_name<std::vector<FullMatrix<scalar_type>>>(
          get_name_jacobian());
      }

      /**
       * Return values at all quadrature points
       */
      template <typename ResultScalarType, int dim2>
      return_type<ResultScalarType>
      operator()(MeshWorker::ScratchData<dim2, spacedim> &scratch_data,
                 const std::vector<std::string> &         solution_names) const
      {
        // Follow the recipe described in the documentation:
        // - Initialize helper.
        // - Register independent variables and set the values for all fields.
        // - Extract the sensitivities.
        // - Use sensitivities in AD functor.
        // - Register the definition of the residual field.
        // - Compute residual field value, Jacobian, etc.
        // - Later, extract the desired components of the residual field value,
        //   Jacobian etc.

        // Note: All user functions have the same parameterization, so we can
        // use the same ADHelper for each of them. This does not restrict the
        // user to use the same definition for the field itself at each
        // QP!
        ad_helper_type &ad_helper = get_mutable_ad_helper(scratch_data);
        std::vector<Vector<scalar_type>> &values =
          get_mutable_values(scratch_data, ad_helper);
        std::vector<FullMatrix<scalar_type>> &Dvalues =
          get_mutable_jacobians(scratch_data, ad_helper);

        const FEValuesBase<dim, spacedim> &fe_values =
          scratch_data.get_current_fe_values();

        // In the HP case, we might traverse between cells with a different
        // number of quadrature points. So we need to resize the output data
        // accordingly.
        if (values.size() != fe_values.n_quadrature_points ||
            Dvalues.size() != fe_values.n_quadrature_points)
          {
            values.resize(fe_values.n_quadrature_points,
                          Vector<scalar_type>(
                            ad_helper.n_dependent_variables()));
            Dvalues.resize(
              fe_values.n_quadrature_points,
              FullMatrix<scalar_type>(ad_helper.n_dependent_variables(),
                                      ad_helper.n_independent_variables()));
          }

        for (const auto q_point : fe_values.quadrature_point_indices())
          {
            ad_helper.reset();

            // Register the independent variables. The actual field solution at
            // the quadrature point is fetched from the scratch_data cache. It
            // is paired with its counterpart extractor, which should not have
            // any indiced overlapping with the extractors for the other fields
            // in the field_args.
            OpHelper_t::ad_register_independent_variables(
              ad_helper,
              scratch_data,
              solution_names,
              q_point,
              get_field_args(),
              get_field_extractors());

            // Evaluate the functor to compute the residual field value.
            // To do this, we extract all sensitivities and pass them directly
            // in the user-provided function.
            const residual_type residual_field_value =
              OpHelper_t::ad_call_function(ad_helper,
                                           function,
                                           scratch_data,
                                           solution_names,
                                           q_point,
                                           get_field_extractors());

            // Register the definition of the field value
            ad_helper.register_dependent_variable(residual_field_value,
                                                  get_residual_extractor());

            // Store the output function value, its gradient and linearization.
            ad_helper.compute_values(values[q_point]);
            ad_helper.compute_jacobian(Dvalues[q_point]);
          }
      }

      const Op &
      get_op() const
      {
        return operand;
      }

      // Independent fields
      const typename OpHelper_t::field_args_t &
      get_field_args() const
      {
        // Get the unary op field solutions from the ResidualFunctor
        return get_op().get_field_args();
      }

      const typename OpHelper_t::field_extractors_t &
      get_field_extractors() const
      {
        return extractors;
      }

    private:
      const Op               operand;
      const ad_function_type function;
      // Some additional update flags that the user might require in order to
      // evaluate their AD function (e.g. UpdateFlags::update_quadrature_points)
      const UpdateFlags update_flags;

      const typename OpHelper_t::field_extractors_t
        extractors; // FEValuesExtractors to work with multi-component fields

      std::string
      get_name_ad_helper() const
      {
        const SymbolicDecorations decorator;
        return internal::get_deal_II_prefix() + "ResidualFunctor_ADHelper_" +
               operand.as_ascii(decorator);
      }

      std::string
      get_name_value() const
      {
        const SymbolicDecorations decorator;
        return internal::get_deal_II_prefix() +
               "ResidualFunctor_ADHelper_Values_" + operand.as_ascii(decorator);
      }

      std::string
      get_name_jacobian() const
      {
        const SymbolicDecorations decorator;
        return internal::get_deal_II_prefix() +
               "ResidualFunctor_ADHelper_Jacobians_" +
               operand.as_ascii(decorator);
      }

      ad_helper_type &
      get_mutable_ad_helper(
        MeshWorker::ScratchData<dim, spacedim> &scratch_data) const
      {
        GeneralDataStorage &cache = scratch_data.get_general_data_storage();
        const std::string   name_ad_helper = get_name_ad_helper();

        // Unfortunately we cannot perform a check like this because the
        // ScratchData is reused by many cells during the mesh loop. So
        // there's no real way to verify that the user is not accidentally
        // re-using an object because they forget to uniquely name the
        // ResidualFunctor upon which this op is based.
        //
        // Assert(!(cache.stores_object_with_name(name_ad_helper)),
        //        ExcMessage("ADHelper is already present in the cache."));

        const unsigned int n_dependent_variables =
          Operators::internal::SpaceOpComponentInfo<TestSpaceOp>::n_components;
        const unsigned int n_independent_variables =
          OpHelper_t::get_n_components();
        return cache.get_or_add_object_with_name<ad_helper_type>(
          name_ad_helper, n_independent_variables, n_dependent_variables);
      }

      std::vector<Vector<scalar_type>> &
      get_mutable_values(MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                         const ad_helper_type &ad_helper) const
      {
        GeneralDataStorage &cache = scratch_data.get_general_data_storage();
        const FEValuesBase<dim, spacedim> &fe_values =
          scratch_data.get_current_fe_values();

        return cache
          .get_or_add_object_with_name<std::vector<Vector<scalar_type>>>(
            get_name_value(),
            fe_values.n_quadrature_points,
            Vector<scalar_type>(ad_helper.n_dependent_variables()));
      }

      std::vector<FullMatrix<scalar_type>> &
      get_mutable_jacobians(
        MeshWorker::ScratchData<dim, spacedim> &scratch_data,
        const ad_helper_type &                  ad_helper) const
      {
        GeneralDataStorage &cache = scratch_data.get_general_data_storage();
        const FEValuesBase<dim, spacedim> &fe_values =
          scratch_data.get_current_fe_values();

        return cache
          .get_or_add_object_with_name<std::vector<FullMatrix<scalar_type>>>(
            get_name_jacobian(),
            fe_values.n_quadrature_points,
            FullMatrix<scalar_type>(ad_helper.n_dependent_variables(),
                                    ad_helper.n_independent_variables()));
      }
    };



    /**
     * Extract the value from a residual functor.
     *
     * Variant for symbolic expressions.
     */
    template <int dim,
              int spacedim,
              typename TestSpaceOp,
              typename... SymbolicOpsSubSpaceFieldSolution>
    class SymbolicOp<
      ResidualViewFunctor<TestSpaceOp, SymbolicOpsSubSpaceFieldSolution...>,
      SymbolicOpCodes::value,
      void,
      Differentiation::SD::Expression,
      WeakForms::internal::DimPack<dim, spacedim>>
    {
      static_assert(is_or_has_test_function_op<TestSpaceOp>::value,
                    "Expected a test function.");
      static_assert(TestSpaceOp::dimension == dim, "Dimension mismatch.");
      static_assert(TestSpaceOp::space_dimension == spacedim,
                    "Spatial dimension mismatch.");

      // All template parameter types must be unary operators
      // for subspaces of a field solution.
      static_assert(
        internal::TemplateRestrictions::
          EnforceIsSymbolicOpSubspaceFieldSolution<
            SymbolicOpsSubSpaceFieldSolution...>::value,
        "Template arguments must be unary operation subspace field solutions. "
        "You might have used a test function or trial solution, or perhaps "
        "have not used a sub-space extractor.");

      // We cannot permit multiple instance of the same unary operations
      // as a part of the template parameter pack. This would imply that
      // we want the user to define a functor that takes in multiple instances
      // of the same field variable, which does not make sense.
      // static_assert(internal::TemplateRestrictions::EnforceNoDuplicates<
      //                 SymbolicOpsSubSpaceFieldSolution...>::value,
      //               "No duplicate types allowed.");

      using Op =
        ResidualViewFunctor<TestSpaceOp, SymbolicOpsSubSpaceFieldSolution...>;

      using OpHelper_t = internal::SymbolicOpsSubSpaceFieldSolutionHelper<
        SymbolicOpsSubSpaceFieldSolution...>;

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = Op::dimension;

      /**
       * Dimension of the subspace in which this object operates.
       */
      static const unsigned int space_dimension = Op::space_dimension;

      using scalar_type =
        std::nullptr_t; // SD expressions can represent anything
      template <typename ReturnType>
      using sd_helper_type = Differentiation::SD::BatchOptimizer<ReturnType>;
      using sd_type        = Differentiation::SD::Expression;
      using substitution_map_type =
        Differentiation::SD::types::substitution_map;

      template <typename ResultScalarType>
      using value_type = typename Op::template value_type<ResultScalarType>;

      template <typename ResultScalarType>
      using function_type =
        typename Op::template sd_function_type<ResultScalarType, dim, spacedim>;

      template <typename ResultScalarType>
      using return_type = void;

      using residual_type    = value_type<sd_type>;
      using sd_function_type = function_type<sd_type>;
      using sd_intermediate_substitution_function_type =
        typename Op::template sd_intermediate_substitution_function_type<
          sd_type,
          dim,
          spacedim>;
      // TODO: If this needs a template <typename ResultScalarType> then the
      // entire unary op must get one.
      using sd_register_symbols_function_type = typename Op::
        template sd_register_symbols_function_type<sd_type, dim, spacedim>;
      using sd_substitution_function_type = typename Op::
        template sd_substitution_function_type<sd_type, dim, spacedim>;

      static const int rank = 0;

      static const enum SymbolicOpCodes op_code = SymbolicOpCodes::value;

      explicit SymbolicOp(
        const Op &                               operand,
        const sd_function_type &                 function,
        const sd_register_symbols_function_type &user_symbol_registration_map,
        const sd_substitution_function_type &    user_substitution_map,
        const sd_intermediate_substitution_function_type
          &user_intermediate_substitution_map,
        const enum Differentiation::SD::OptimizerType     optimization_method,
        const enum Differentiation::SD::OptimizationFlags optimization_flags,
        const UpdateFlags                                 update_flags)
        : operand(operand)
        , function(function)
        , user_symbol_registration_map(user_symbol_registration_map)
        , user_substitution_map(user_substitution_map)
        , user_intermediate_substitution_map(user_intermediate_substitution_map)
        , optimization_method(optimization_method)
        , optimization_flags(optimization_flags)
        , update_flags(update_flags)
        , symbolic_fields(OpHelper_t::template get_symbolic_fields<sd_type>(
            get_field_args(),
            SymbolicDecorations()))
        , residual(
            OpHelper_t::template sd_call_function<sd_type>(function,
                                                           symbolic_fields))
        , first_derivatives(
            OpHelper_t::template sd_differentiate<sd_type>(residual,
                                                           symbolic_fields))
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
        return update_flags;
      }

      template <typename ResultScalarType>
      const sd_helper_type<ResultScalarType> &
      get_batch_optimizer(
        const MeshWorker::ScratchData<dim, spacedim> &scratch_data) const
      {
        const GeneralDataStorage &cache =
          scratch_data.get_general_data_storage();

        return cache.get_object_with_name<sd_helper_type<ResultScalarType>>(
          get_name_sd_batch_optimizer());
      }

      const auto &
      get_symbolic_residual() const
      {
        return residual;
      }

      template <std::size_t FieldIndex>
      const auto &
      get_symbolic_first_derivative() const
      {
        static_assert(FieldIndex < OpHelper_t::n_operators(),
                      "Index out of bounds.");
        return std::get<FieldIndex>(first_derivatives);
      }


      template <typename ResultScalarType>
      const std::vector<std::vector<ResultScalarType>> &
      get_evaluated_dependent_functions(
        const MeshWorker::ScratchData<dim, spacedim> &scratch_data) const
      {
        const GeneralDataStorage &cache =
          scratch_data.get_general_data_storage();

        return cache
          .get_object_with_name<std::vector<std::vector<ResultScalarType>>>(
            get_name_evaluated_dependent_functions());
      }

      /**
       * Return values at all quadrature points
       */
      template <typename ResultScalarType, int dim2>
      return_type<ResultScalarType>
      operator()(MeshWorker::ScratchData<dim2, spacedim> &scratch_data,
                 const std::vector<std::string> &         solution_names) const
      {
        // Follow the recipe described in the documentation:
        // - Define some independent variables.
        // - Compute symbolic expressions that are dependent on the independent
        //   variables.
        // - Create a optimizer to evaluate the dependent functions.
        // - Register symbols that represent independent variables.
        // - Register symbolic expressions that represent dependent functions.
        // - Optimize: Determine computationally efficient code path for
        //   evaluation.
        // - Substitute: Pass the optimizer the numeric values that thee
        //   independent variables to represent.
        // - Extract the numeric equivalent of the dependent functions from the
        //   optimizer.

        // Note: All user functions have the same parameterization, so on the
        // face of it we can use the same BatchOptimizer for each of them. In
        // theory the user can encode the QPoint into the field function: this
        // current implementation restricts the user to use the same definition
        // for the field itself at each QP.
        sd_helper_type<ResultScalarType> &batch_optimizer =
          get_mutable_sd_batch_optimizer<ResultScalarType>(scratch_data);
        if (batch_optimizer.optimized() == false)
          {
            Assert(batch_optimizer.n_independent_variables() == 0,
                   ExcMessage(
                     "Expected the batch optimizer to be uninitialized."));
            Assert(batch_optimizer.n_dependent_variables() == 0,
                   ExcMessage(
                     "Expected the batch optimizer to be uninitialized."));
            Assert(batch_optimizer.values_substituted() == false,
                   ExcMessage(
                     "Expected the batch optimizer to be uninitialized."));

            // Create and register field variables (the independent variables).
            // We deal with the fields before the user data just in case
            // the users try to overwrite these field symbols. It shouldn't
            // happen, but this way its not possible to do overwrite what's
            // already in the map.
            Differentiation::SD::types::substitution_map symbol_map =
              OpHelper_t::template sd_get_symbol_map<sd_type>(
                get_symbolic_fields());
            if (user_symbol_registration_map)
              {
                Differentiation::SD::add_to_symbol_map(
                  symbol_map,
                  OpHelper_t::template sd_call_function<sd_type>(
                    user_symbol_registration_map, get_symbolic_fields()));
              }
            batch_optimizer.register_symbols(symbol_map);

            // The next typical few steps that precede function resistration
            // have already been performed in the class constructor:
            // - Evaluate the functor to compute the total stored field.
            // - Compute the first derivatives of the field function.
            // - If there's some intermediate substitution to be done (modifying
            // the first derivatives), then do it before computing the second
            // derivatives.
            // (Why the intermediate substitution? If the first derivatives
            // represent the partial derivatives, then this substitution may be
            // done to ensure that the consistent linearization is given by the
            // second derivatives.)
            // - Differentiate the first derivatives (perhaps a modified form)
            // to get the second derivatives.

            // Register the dependent variables.
            OpHelper_t::template sd_register_functions<sd_type, residual_type>(
              batch_optimizer, residual);
            OpHelper_t::template sd_register_functions<sd_type, residual_type>(
              batch_optimizer, first_derivatives);

            // Finalize the optimizer.
            batch_optimizer.optimize();
          }

        // Check that we've actually got a state that we can do some work with.
        Assert(batch_optimizer.n_independent_variables() > 0,
               ExcMessage("Expected the batch optimizer to be initialized."));
        Assert(batch_optimizer.n_dependent_variables() > 0,
               ExcMessage("Expected the batch optimizer to be initialized."));

        std::vector<std::vector<ResultScalarType>>
          &evaluated_dependent_functions =
            get_mutable_evaluated_dependent_functions<ResultScalarType>(
              scratch_data, batch_optimizer);

        const FEValuesBase<dim, spacedim> &fe_values =
          scratch_data.get_current_fe_values();

        // In the HP case, we might traverse between cells with a different
        // number of quadrature points. So we need to resize the output data
        // accordingly.
        if (evaluated_dependent_functions.size() !=
            fe_values.n_quadrature_points)
          {
            evaluated_dependent_functions.resize(
              fe_values.n_quadrature_points,
              std::vector<ResultScalarType>(
                batch_optimizer.n_dependent_variables()));
          }

        for (const auto q_point : fe_values.quadrature_point_indices())
          {
            // Substitute the field variables and whatever user symbols
            // are defined.
            // First we do the values from finite element fields,
            // followed by the values for user parameters, etc.
            Differentiation::SD::types::substitution_map substitution_map =
              OpHelper_t::template sd_get_substitution_map<sd_type,
                                                           ResultScalarType>(
                scratch_data,
                solution_names,
                q_point,
                get_symbolic_fields(),
                get_field_args());
            if (user_substitution_map)
              {
                Differentiation::SD::add_to_substitution_map(
                  substitution_map,
                  user_substitution_map(scratch_data, solution_names, q_point));
              }

            // Perform the value substitution at this quadrature point
            batch_optimizer.substitute(substitution_map);

            // Extract evaluated data to be retrieved later.
            evaluated_dependent_functions[q_point] = batch_optimizer.evaluate();
          }
      }

      const TestSpaceOp &
      get_test_function() const
      {
        return get_op().get_test_function();
      }

      const typename OpHelper_t::template field_values_t<sd_type> &
      get_symbolic_fields() const
      {
        return symbolic_fields;
      }

      // Independent fields
      const typename OpHelper_t::field_args_t &
      get_field_args() const
      {
        // Get the unary op field solutions from the ResidualFunctor
        return get_op().get_field_args();
      }

    private:
      const Op               operand;
      const sd_function_type function;

      const sd_register_symbols_function_type user_symbol_registration_map;
      const sd_substitution_function_type     user_substitution_map;
      const sd_intermediate_substitution_function_type
        user_intermediate_substitution_map;

      const enum Differentiation::SD::OptimizerType     optimization_method;
      const enum Differentiation::SD::OptimizationFlags optimization_flags;

      // Some additional update flags that the user might require in order to
      // evaluate their SD function (e.g. UpdateFlags::update_quadrature_points)
      const UpdateFlags update_flags;

      // Independent variables
      const typename OpHelper_t::template field_values_t<sd_type>
        symbolic_fields;

      // Dependent variables
      const residual_type residual;
      const typename OpHelper_t::
        template first_derivatives_value_t<sd_type, residual_type>
          first_derivatives;

      std::string
      get_name_sd_batch_optimizer() const
      {
        const SymbolicDecorations decorator;
        return internal::get_deal_II_prefix() +
               "ResidualFunctor_SDBatchOptimizer_" +
               operand.as_ascii(decorator);
      }

      std::string
      get_name_evaluated_dependent_functions() const
      {
        const SymbolicDecorations decorator;
        return internal::get_deal_II_prefix() +
               "ResidualFunctor_ADHelper_Evaluated_Dependent_Functions" +
               operand.as_ascii(decorator);
      }

      template <typename ResultScalarType>
      sd_helper_type<ResultScalarType> &
      get_mutable_sd_batch_optimizer(
        MeshWorker::ScratchData<dim, spacedim> &scratch_data) const
      {
        GeneralDataStorage &cache = scratch_data.get_general_data_storage();
        const std::string   name_sd_batch_optimizer =
          get_name_sd_batch_optimizer();

        // Unfortunately we cannot perform a check like this because the
        // ScratchData is reused by many cells during the mesh loop. So
        // there's no real way to verify that the user is not accidentally
        // re-using an object because they forget to uniquely name the
        // ResidualFunctor upon which this op is based.
        //
        // Assert(!(cache.stores_object_with_name(name_sd_batch_optimizer)),
        //        ExcMessage("SDBatchOptimizer is already present in the
        //        cache."));

        return cache
          .get_or_add_object_with_name<sd_helper_type<ResultScalarType>>(
            name_sd_batch_optimizer, optimization_method, optimization_flags);
      }

      template <typename ResultScalarType>
      std::vector<std::vector<ResultScalarType>> &
      get_mutable_evaluated_dependent_functions(
        MeshWorker::ScratchData<dim, spacedim> &scratch_data,
        const sd_helper_type<ResultScalarType> &batch_optimizer) const
      {
        GeneralDataStorage &cache = scratch_data.get_general_data_storage();
        const FEValuesBase<dim, spacedim> &fe_values =
          scratch_data.get_current_fe_values();

        return cache.get_or_add_object_with_name<
          std::vector<std::vector<ResultScalarType>>>(
          get_name_evaluated_dependent_functions(),
          fe_values.n_quadrature_points,
          std::vector<ResultScalarType>(
            batch_optimizer.n_dependent_variables()));
      }

      const Op &
      get_op() const
      {
        return operand;
      }
    };

  } // namespace Operators
} // namespace WeakForms



// /* ======================== Convenience functions ======================== */



namespace WeakForms
{
  template <typename ADNumberType,
            int dim,
            int spacedim = dim,
            typename TestSpaceOp,
            typename... SymbolicOpsSubSpaceFieldSolution,
            typename = typename std::enable_if<
              Differentiation::AD::is_ad_number<ADNumberType>::value>::type>
  WeakForms::Operators::SymbolicOp<
    WeakForms::ResidualViewFunctor<TestSpaceOp,
                                   SymbolicOpsSubSpaceFieldSolution...>,
    WeakForms::Operators::SymbolicOpCodes::value,
    typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type,
    ADNumberType,
    internal::DimPack<dim, spacedim>>
  value(
    const WeakForms::ResidualViewFunctor<TestSpaceOp,
                                         SymbolicOpsSubSpaceFieldSolution...>
      &operand,
    const typename WeakForms::
      ResidualViewFunctor<TestSpaceOp, SymbolicOpsSubSpaceFieldSolution...>::
        template ad_function_type<ADNumberType, dim, spacedim> &function,
    const UpdateFlags                                           update_flags)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op =
      WeakForms::ResidualViewFunctor<TestSpaceOp,
                                     SymbolicOpsSubSpaceFieldSolution...>;
    using ScalarType =
      typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type;
    using OpType = SymbolicOp<Op,
                              SymbolicOpCodes::value,
                              ScalarType,
                              ADNumberType,
                              WeakForms::internal::DimPack<dim, spacedim>>;

    return OpType(operand, function, update_flags);
  }



  template <typename SDNumberType,
            int dim,
            int spacedim = dim,
            typename TestSpaceOp,
            typename... SymbolicOpsSubSpaceFieldSolution,
            typename = typename std::enable_if<
              Differentiation::SD::is_sd_number<SDNumberType>::value>::type>
  WeakForms::Operators::SymbolicOp<
    WeakForms::ResidualViewFunctor<TestSpaceOp,
                                   SymbolicOpsSubSpaceFieldSolution...>,
    WeakForms::Operators::SymbolicOpCodes::value,
    void,
    SDNumberType,
    internal::DimPack<dim, spacedim>>
  value(
    const WeakForms::ResidualViewFunctor<TestSpaceOp,
                                         SymbolicOpsSubSpaceFieldSolution...>
      &operand,
    const typename WeakForms::
      ResidualViewFunctor<TestSpaceOp, SymbolicOpsSubSpaceFieldSolution...>::
        template sd_function_type<SDNumberType, dim, spacedim> &function,
    const typename WeakForms::
      ResidualViewFunctor<TestSpaceOp, SymbolicOpsSubSpaceFieldSolution...>::
        template sd_register_symbols_function_type<SDNumberType, dim, spacedim>
          symbol_registration_map,
    const typename WeakForms::
      ResidualViewFunctor<TestSpaceOp, SymbolicOpsSubSpaceFieldSolution...>::
        template sd_substitution_function_type<SDNumberType, dim, spacedim>
          substitution_map,
    const typename WeakForms::
      ResidualViewFunctor<TestSpaceOp, SymbolicOpsSubSpaceFieldSolution...>::
        template sd_intermediate_substitution_function_type<SDNumberType,
                                                            dim,
                                                            spacedim>
                                                      intermediate_substitution_map,
    const enum Differentiation::SD::OptimizerType     optimization_method,
    const enum Differentiation::SD::OptimizationFlags optimization_flags,
    const UpdateFlags                                 update_flags)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op =
      WeakForms::ResidualViewFunctor<TestSpaceOp,
                                     SymbolicOpsSubSpaceFieldSolution...>;
    using OpType = SymbolicOp<Op,
                              SymbolicOpCodes::value,
                              void,
                              SDNumberType,
                              WeakForms::internal::DimPack<dim, spacedim>>;

    return OpType(operand,
                  function,
                  symbol_registration_map,
                  substitution_map,
                  intermediate_substitution_map,
                  optimization_method,
                  optimization_flags,
                  update_flags);
  }
} // namespace WeakForms



// /* ==================== Specialization of type traits ==================== */



// /* ==================== Class method definitions ==================== */

namespace WeakForms
{
  template <typename TestSpaceOp, typename... SymbolicOpsSubSpaceFieldSolution>
  template <typename ADNumberType, int dim, int spacedim>
  auto
  ResidualViewFunctor<TestSpaceOp, SymbolicOpsSubSpaceFieldSolution...>::
  operator()(const typename WeakForms::ResidualViewFunctor<
               TestSpaceOp,
               SymbolicOpsSubSpaceFieldSolution...>::
               template ad_function_type<ADNumberType, dim, spacedim> &function,
             const UpdateFlags update_flags) const
  {
    return WeakForms::value<ADNumberType, dim, spacedim>(*this,
                                                         function,
                                                         update_flags);
  }


  template <typename TestSpaceOp, typename... SymbolicOpsSubSpaceFieldSolution>
  template <typename SDNumberType, int dim, int spacedim>
  auto
  ResidualViewFunctor<TestSpaceOp, SymbolicOpsSubSpaceFieldSolution...>::
  operator()(
    const typename WeakForms::
      ResidualViewFunctor<TestSpaceOp, SymbolicOpsSubSpaceFieldSolution...>::
        template sd_function_type<SDNumberType, dim, spacedim> &function,
    const typename WeakForms::
      ResidualViewFunctor<TestSpaceOp, SymbolicOpsSubSpaceFieldSolution...>::
        template sd_register_symbols_function_type<SDNumberType, dim, spacedim>
          symbol_registration_map,
    const typename WeakForms::
      ResidualViewFunctor<TestSpaceOp, SymbolicOpsSubSpaceFieldSolution...>::
        template sd_substitution_function_type<SDNumberType, dim, spacedim>
          substitution_map,
    const typename WeakForms::
      ResidualViewFunctor<TestSpaceOp, SymbolicOpsSubSpaceFieldSolution...>::
        template sd_intermediate_substitution_function_type<SDNumberType,
                                                            dim,
                                                            spacedim>
                                                      intermediate_substitution_map,
    const enum Differentiation::SD::OptimizerType     optimization_method,
    const enum Differentiation::SD::OptimizationFlags optimization_flags,
    const UpdateFlags                                 update_flags) const
  {
    return WeakForms::value<SDNumberType, dim, spacedim>(
      *this,
      function,
      symbol_registration_map,
      substitution_map,
      intermediate_substitution_map,
      optimization_method,
      optimization_flags,
      update_flags);
  }
} // namespace WeakForms



#ifndef DOXYGEN


namespace WeakForms
{
  // ======= AD =======


  template <typename ADNumberType,
            int dim,
            int spacedim,
            typename TestSpaceOp,
            typename... SymbolicOpsSubSpaceFieldSolution>
  struct is_ad_functor_op<Operators::SymbolicOp<
    ResidualViewFunctor<TestSpaceOp, SymbolicOpsSubSpaceFieldSolution...>,
    Operators::SymbolicOpCodes::value,
    typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type,
    ADNumberType,
    internal::DimPack<dim, spacedim>>> : std::true_type
  {};


  template <typename ADNumberType,
            int dim,
            int spacedim,
            typename TestSpaceOp,
            typename... SymbolicOpsSubSpaceFieldSolution>
  struct is_residual_functor_op<Operators::SymbolicOp<
    ResidualViewFunctor<TestSpaceOp, SymbolicOpsSubSpaceFieldSolution...>,
    Operators::SymbolicOpCodes::value,
    typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type,
    ADNumberType,
    internal::DimPack<dim, spacedim>>> : std::true_type
  {};


  // ======= SD =======


  template <int dim,
            int spacedim,
            typename TestSpaceOp,
            typename... SymbolicOpsSubSpaceFieldSolution>
  struct is_sd_functor_op<Operators::SymbolicOp<
    ResidualViewFunctor<TestSpaceOp, SymbolicOpsSubSpaceFieldSolution...>,
    Operators::SymbolicOpCodes::value,
    void,
    Differentiation::SD::Expression,
    WeakForms::internal::DimPack<dim, spacedim>>> : std::true_type
  {};


  template <int dim,
            int spacedim,
            typename TestSpaceOp,
            typename... SymbolicOpsSubSpaceFieldSolution>
  struct is_residual_functor_op<Operators::SymbolicOp<
    ResidualViewFunctor<TestSpaceOp, SymbolicOpsSubSpaceFieldSolution...>,
    Operators::SymbolicOpCodes::value,
    void,
    Differentiation::SD::Expression,
    WeakForms::internal::DimPack<dim, spacedim>>> : std::true_type
  {};

} // namespace WeakForms


#endif // DOXYGEN


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_residual_functor_h
