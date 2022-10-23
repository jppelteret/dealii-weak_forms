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

#ifndef dealii_weakforms_energy_functor_h
#define dealii_weakforms_energy_functor_h

#include <deal.II/base/config.h>

#include <deal.II/algorithms/general_data_storage.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/multithread_info.h>

#include <deal.II/differentiation/ad.h>
#include <deal.II/differentiation/sd.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/meshworker/scratch_data.h>

#include <weak_forms/ad_sd_functor_cache.h>
#include <weak_forms/ad_sd_functor_internal.h>
#include <weak_forms/config.h>
#include <weak_forms/differentiation.h>
#include <weak_forms/functors.h>
#include <weak_forms/solution_extraction_data.h>
#include <weak_forms/type_traits.h>
#include <weak_forms/utilities.h>

#include <functional>
#include <mutex>
#include <thread>
#include <tuple>


WEAK_FORMS_NAMESPACE_OPEN

#if defined(DEAL_II_WITH_SYMENGINE) || \
  defined(DEAL_II_WITH_AUTO_DIFFERENTIATION)


namespace WeakForms
{
  /**
   * @brief A class that represents the point-wise decomposition of an energy functional.
   *
   * An instance of this class  can also be easily created using the
   * energy_functor() convenience function.
   *
   * The parameterization of the energy functor is provided by its template
   * argument(s). This selection then defines the arguments that must be passed
   * into the functions that compute the energy at any quadrature point.
   * Depending on which variant of the EnergyFunctor::value() function is called
   * to assign a definition to the energy, one of two modes will have been
   * chosen to evaluate both this function as well as its various derivatives
   * with respect to the field variables: either automatic differentiation (AD)
   * or symbolic differentiation (SD) will be utilized for this task. The former
   * is likely more user-friendly, while the latter offers the possibility of
   * more performance when the definition of the energy is itself complex, or
   * has complex or lengthy derivatives.
   *
   * An example use of this class (using the AD technique for brevity) is
   * as follows:
   * @code {.cpp}
   * using namespace WeakForms;
   * constexpr int dim = ...;
   * constexpr int spacedim = ...;
   *
   * // Define the field solution and an extractor to get a view into one
   * // of its components.
   * const FieldSolution<dim> solution;
   * const SubSpaceExtractors::Scalar subspace_extractor(0, "s", "s");
   *
   * // Extract subspace of field solution; namely operators that
   * // represent its value and gradient.
   * const auto soln_s_val  = solution[subspace_extractor].value();
   * const auto soln_s_grad = solution[subspace_extractor].gradient();
   *
   * // Parameterize an energy in terms of all field solution's value and
   * // gradient. This is used as a factory to define energy functionals with
   * // the given parameterization and naming convention.
   * // Using the provided convenience function, the types
   * // `decltype(soln_s_val)` and `decltype(soln_s_grad)` are collectively
   * // passed as the EnergyFunctor class template argument
   * // `SymbolicOpsSubSpaceFieldSolution`.
   * const auto energy_func
   *   = energy_functor("e", "\\Psi", soln_s_val, soln_s_grad);
   *
   * // Choose an auto-differentiable number as the scalar type for the energy
   * // that we'll define next.
   * constexpr auto ad_typecode =
   *   dealii::Differentiation::AD::NumberTypes::sacado_dfad_dfad;
   * using ADNumber_t =
   *   typename decltype(energy_func)::template ad_type<double, ad_typecode>;
   *
   * // Now create a specific instance of an energy functional: this not only
   * // provides the definition of the point-wise energy to be considered, but
   * // also a means to differentiate it with respect to its arguments.
   * // This is achieved with a call to EnergyFunctor::value(), with the
   * // differentiable number type as the first template argument.
   * // The definition of the energy can be supplied using a lambda function
   * // (for auto-differentiable numbers, only one lambda function will need
   * // to be provided):
   * // the first three arguments that the lambda function must take in are
   * // always the same;
   * // the arguments that follow are exactly the local value types for the
   * // field solution that were supplied in the call to energy_functor() a
   * // few lines above. Since we passed in the view to a scalar field value,
   * // followed by a scalar field gradient, the two arguments to this
   * // lambda function will be a (scalar) ADNumber_t followed by a rank-1
   * // tensor of ADNumber_t. These arguments will point to valid data that is
   * // extracted from the associated field solution and the energy will only
   * // be made to be sensitive (in a differentiable sense) to them.
   * // All of the initialization of the AD or SD values is done automatically,
   * // so one need only concentrate on the definition to be evaluated.
   * const auto energy = energy_func.template value<ADNumber_t, dim, spacedim>(
   *   [](const dealii::MeshWorker::ScratchData<dim, spacedim> &scratch_data,
   *      const std::vector<SolutionExtractionData<dim, spacedim>>
   *        &                                       solution_extraction_data,
   *      const unsigned int                        q_point,
   *      const ADNumber_t &                        u,
   *      const dealii::Tensor<1, dim, ADNumber_t> &grad_u)
   * {
   *   const ADNumber_t psi = ...;
   *   return psi;
   * });
   *
   * // Now for assembly...
   * MatrixBasedAssembler<spacedim> assembler;
   *
   * // To make use of this energy, we simply accumulate its integral (thereby
   * // formally defining an energy functional in terms of the point-wise
   * // energy) into an assembler using a
   * // SelfLinearization::EnergyFunctionalForm.
   * // This can be done using the energy_functional_form() convenience
   * // function. At this point, the energy functional is translated into the
   * // appropriate linear and bilinear forms at compile time, and where
   * // run-time differentiation of the point-wise energy contributions with
   * // respect to its field arguments is also configured.
   * assembler += energy_functional_form(energy).dV();
   * @endcode
   *
   * In this specific example, the energy functional generates the following
   * forms (this can be inspected by printing the `assembler`):
   * @f{align}{
   * \psi \left( s, \nabla s \right)
   * \quad \Rightarrow 0
   * &= \left( \delta s, \dfrac{d \psi \left( s, \nabla s \right)}{d s} \right)
   *  + \left( \delta \nabla s, \dfrac{d \psi \left( s, \nabla s \right)}{d
   *    \nabla s} \right) \\
   * &+ a \left( \delta s, \dfrac{d^{2} \psi \left( s, \nabla s \right)}{d s^2}
   *    . \Delta s \right)
   *  + a \left( \delta s, \dfrac{d^{2} \psi \left( s, \nabla s \right)}{d s . d
   *    \nabla s} \cdot \Delta \nabla s \right) \\
   * &+ a \left( \delta \nabla s, \dfrac{d^{2} \psi \left( s, \nabla s
   *    \right)}{d \nabla s . ds} . \Delta s \right)
   *  + a \left( \delta \nabla s, \dfrac{d^{2} \psi \left( s, \nabla s
   *    \right)}{d \nabla s \otimes d \nabla s} \cdot \Delta \nabla s \right)
   * @f}
   * where @f$ \delta \left( \bullet \right) @f$ represents a variation (or test
   * function, for linear problems) and @f$ \Delta \left( \bullet \right) @f$
   * represents the solution increment (or trial solution, for linear problems).
   * The linear forms, which are the first two terms on the right of the
   * equation, get transferred to the right-hand side of the linear system
   * @f$ \mathbf{K}\cdot\mathbf{d} = \mathbf{f} @f$, with a sign change. The
   * bilinear forms are assembled to the system matrix on the left-hand side of
   * the equation with the same sign.
   *
   * (With respect to the formulae presented above, we follow the notation of
   * T.J.R. Hughes "The Finite Element Method: Linear  Static and Dynamic Finite
   * Element Analysis", 2000 to denote symmetric, bilinear forms. Important here
   * is that the quantity @f$ a \left( \bullet, \bullet \right) @f$ contributes
   * to the linear system matrix, while @f$ \left( \bullet, \bullet \right) @f$
   * contribute to the right-hand side vector.)
   *
   * @tparam SymbolicOpsSubSpaceFieldSolution A variadic template that represents
   * the component(s) of the field solutions that parameterize the energy
   * functional. Each argument captures either a field, or one of its
   * derivatives. Each are treated as independent variables as they are later
   * used in the construction of linear and bilinear form(s). Note that one may
   * not use the global field solution as an argument; instead, this class
   * requires views to each component of the solution in order to be useful.
   */
  template <typename... SymbolicOpsSubSpaceFieldSolution>
  class EnergyFunctor : public WeakForms::Functor<0>
  {
    using Base = WeakForms::Functor<0>;

  public:
    template <typename ADorSDNumberType>
    using value_type = ADorSDNumberType;

    template <typename ScalarType,
              enum Differentiation::AD::NumberTypes ADNumberTypeCode>
    using ad_type =
      typename Differentiation::AD::NumberTraits<ScalarType,
                                                 ADNumberTypeCode>::ad_type;

    template <typename ADNumberType, int dim, int spacedim = dim>
    using ad_function_type = std::function<value_type<ADNumberType>(
      const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      const std::vector<SolutionExtractionData<dim, spacedim>>
        &                solution_extraction_data,
      const unsigned int q_point,
      const typename SymbolicOpsSubSpaceFieldSolution::template value_type<
        ADNumberType> &...field_solutions)>;

#  ifdef DEAL_II_WITH_SYMENGINE

    template <typename ScalarType>
    using sd_type               = Differentiation::SD::Expression;
    using substitution_map_type = Differentiation::SD::types::substitution_map;

    template <typename SDNumberType, int dim, int spacedim = dim>
    using sd_function_type = std::function<value_type<SDNumberType>(
      const typename SymbolicOpsSubSpaceFieldSolution::template value_type<
        SDNumberType> &...field_solutions)>;

    template <typename SDNumberType, int dim, int spacedim = dim>
    using sd_intermediate_substitution_function_type =
      std::function<substitution_map_type(
        const typename SymbolicOpsSubSpaceFieldSolution::template value_type<
          SDNumberType> &...field_solutions)>;

    // This also allows the user to encode symbols/parameters in terms of
    // the (symbolic) field variables, for which we'll supply the values.
    template <typename SDNumberType, int dim, int spacedim = dim>
    using sd_register_symbols_function_type =
      std::function<substitution_map_type(
        const typename SymbolicOpsSubSpaceFieldSolution::template value_type<
          SDNumberType> &...field_solutions)>;

    template <typename SDNumberType, int dim, int spacedim = dim>
    using sd_substitution_function_type = std::function<substitution_map_type(
      const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      const std::vector<SolutionExtractionData<dim, spacedim>>
        &                solution_extraction_data,
      const unsigned int q_point)>;

#  endif // DEAL_II_WITH_SYMENGINE


    EnergyFunctor(
      const std::string &symbol_ascii,
      const std::string &symbol_latex,
      const SymbolicOpsSubSpaceFieldSolution &...symbolic_op_field_solutions)
      : Base(symbol_ascii, symbol_latex)
      , symbolic_op_field_solutions(symbolic_op_field_solutions...)
    {}

    EnergyFunctor(const std::string &symbol_ascii,
                  const std::string &symbol_latex,
                  const std::tuple<SymbolicOpsSubSpaceFieldSolution...>
                    &symbolic_op_field_solutions)
      : Base(symbol_ascii, symbol_latex)
      , symbolic_op_field_solutions(symbolic_op_field_solutions)
    {}

    // ----  Ascii ----

    virtual std::string
    as_ascii(const SymbolicDecorations &decorator) const override
    {
      return Base::as_ascii(decorator) + "(" +
             decorator.unary_field_ops_as_ascii<true>(get_field_args()) + ")";
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
        decorator.unary_field_ops_as_latex<true>(get_field_args()));
    }

    virtual std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const override
    {
      (void)decorator;
      return symbol_latex;
    }

    // Methods to promote this class to a SymbolicOp

#  ifdef DEAL_II_WITH_AUTO_DIFFERENTIATION

    template <typename ADNumberType, int dim, int spacedim = dim>
    auto
    value(const ad_function_type<ADNumberType, dim, spacedim> &function,
          const UpdateFlags update_flags) const;

    template <typename ADNumberType, int dim, int spacedim = dim>
    auto
    value(const ad_function_type<ADNumberType, dim, spacedim> &function) const
    {
      return this->template value<ADNumberType, dim, spacedim>(
        function, UpdateFlags::update_default);
    }

#  endif // DEAL_II_WITH_AUTO_DIFFERENTIATION

#  ifdef DEAL_II_WITH_SYMENGINE

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

      return this->template value<SDNumberType, dim, spacedim>(
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

      return this->template value<SDNumberType, dim, spacedim>(
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

      return this->template value<SDNumberType, dim, spacedim>(
        function,
        symbol_registration_map,
        substitution_map,
        dummy_intermediate_substitution_map,
        optimization_method,
        optimization_flags,
        UpdateFlags::update_default);
    }

#  endif // DEAL_II_WITH_SYMENGINE

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

} // namespace WeakForms



/* ======================== Convenience functions ======================== */



namespace WeakForms
{
  /**
   * @brief A convenience function for creating an EnergyFunctor.
   *
   * It is, essentially, a shortcut so that we can do this:
   * @code
   * const FieldSolution<dim>                    solution;
   * const WeakForms::SubSpaceExtractors::Scalar subspace_extractor(
   *   0, "s", "s");
   *
   * const auto soln_ss   = solution[subspace_extractor];
   * const auto soln_val  = soln_ss.value();    // Solution value
   * const auto soln_grad = soln_ss.gradient(); // Solution gradient
   * // ... etc.
   *
   * // Parameterise energy in terms of all possible operations with the space
   * const auto energy = energy_functor("e", "\\Psi", soln_val, soln_grad, ...);
   * @endcode
   * instead of the last call being this more complicated expression:
   * @code
   * const EnergyFunctor<decltype(soln_val), decltype(soln_grad), ...>
   *   energy("e", "\\Psi", soln_val, soln_grad, ...);
   * @endcode
   *
   * @tparam SymbolicOpsSubSpaceFieldSolution A variadic template that lists the types of field solution operations that this functor is sensitive to (i.e. the association to the input arguments for differentiation).
   * @param symbol_ascii The ASCII representation of the value.
   * @param symbol_latex  The LaTeX representation of the value.
   * @param symbolic_op_field_solutions The field solution operations that this functor is sensitive to (i.e. the association to the input arguments for differentiation).
   *
   * \ingroup functors convenience_functions
   */
  template <typename... SymbolicOpsSubSpaceFieldSolution>
  EnergyFunctor<SymbolicOpsSubSpaceFieldSolution...>
  energy_functor(
    const std::string &symbol_ascii,
    const std::string &symbol_latex,
    const SymbolicOpsSubSpaceFieldSolution &...symbolic_op_field_solutions)
  {
    return EnergyFunctor<SymbolicOpsSubSpaceFieldSolution...>(
      symbol_ascii, symbol_latex, symbolic_op_field_solutions...);
  }
} // namespace WeakForms



/* ================== Specialization of unary operators ================== */



namespace WeakForms
{
  namespace Operators
  {
    /* ------------------------ Functors: Custom ------------------------ */

#  ifdef DEAL_II_WITH_AUTO_DIFFERENTIATION

    /**
     * Extract the value from a energy functor.
     *
     * Variant for auto-differentiable number.
     */
    template <typename ADNumberType,
              int dim,
              int spacedim,
              typename... SymbolicOpsSubSpaceFieldSolution>
    class SymbolicOp<
      EnergyFunctor<SymbolicOpsSubSpaceFieldSolution...>,
      SymbolicOpCodes::value,
      typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type,
      ADNumberType,
      WeakForms::internal::DimPack<dim, spacedim>>
    {
      static_assert(Differentiation::AD::is_ad_number<ADNumberType>::value,
                    "Expected an AD number.");

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

      using Op = EnergyFunctor<SymbolicOpsSubSpaceFieldSolution...>;

      using OpHelper_t = internal::SymbolicOpsSubSpaceFieldSolutionADHelper<
        SymbolicOpsSubSpaceFieldSolution...>;

    public:
      using scalar_type =
        typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type;

      static constexpr enum Differentiation::AD::NumberTypes ADNumberTypeCode =
        Differentiation::AD::ADNumberTraits<ADNumberType>::type_code;

      using ad_helper_type = Differentiation::AD::
        ScalarFunction<spacedim, ADNumberTypeCode, scalar_type>;
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

      using energy_type      = value_type<ad_type>;
      using ad_function_type = function_type<ad_type>;

      static const int rank = 0;

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
          AD_SD_Functor_Cache::get_cache(scratch_data);

        return cache.get_object_with_name<ad_helper_type>(get_name_ad_helper());
      }

      template <std::size_t FieldIndex, typename SymbolicOpField>
      typename Operators::internal::SpaceOpComponentInfo<
        SymbolicOpField>::extractor_type
      get_derivative_extractor(const SymbolicOpField &) const
      {
        static_assert(FieldIndex < OpHelper_t::n_operators(),
                      "Index out of bounds.");
        return std::get<FieldIndex>(get_field_extractors());

        // TODO: Remove obsolete implementation in OpHelper_t
        // return OpHelper_t::get_initialized_extractor(field,
        // get_field_args());
      }

      const std::vector<Vector<scalar_type>> &
      get_gradients(
        const MeshWorker::ScratchData<dim, spacedim> &scratch_data) const
      {
        const GeneralDataStorage &cache =
          AD_SD_Functor_Cache::get_cache(scratch_data);

        return cache.get_object_with_name<std::vector<Vector<scalar_type>>>(
          get_name_gradient());
      }

      const std::vector<FullMatrix<scalar_type>> &
      get_hessians(
        const MeshWorker::ScratchData<dim, spacedim> &scratch_data) const
      {
        const GeneralDataStorage &cache =
          AD_SD_Functor_Cache::get_cache(scratch_data);

        return cache.get_object_with_name<std::vector<FullMatrix<scalar_type>>>(
          get_name_hessian());
      }

      /**
       * Return values at all quadrature points
       */
      template <typename ResultScalarType>
      return_type<ResultScalarType>
      operator()(MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                 const std::vector<SolutionExtractionData<dim, spacedim>>
                   &solution_extraction_data) const
      {
        // Follow the recipe described in the documentation:
        // - Initialize helper.
        // - Register independent variables and set the values for all fields.
        // - Extract the sensitivities.
        // - Use sensitivities in AD functor.
        // - Register the definition of the total stored energy.
        // - Compute gradient, linearization, etc.
        // - Later, extract the desired components of the gradient,
        //   linearization etc.

        // Note: All user functions have the same parameterization, so we can
        // use the same ADHelper for each of them. This does not restrict the
        // user to use the same definition for the energy itself at each QP!
        ad_helper_type &ad_helper = get_mutable_ad_helper(scratch_data);
        std::vector<Vector<scalar_type>> &Dpsi =
          get_mutable_gradients(scratch_data, ad_helper);
        std::vector<FullMatrix<scalar_type>> &D2psi =
          get_mutable_hessians(scratch_data, ad_helper);

        const FEValuesBase<dim, spacedim> &fe_values =
          scratch_data.get_current_fe_values();

        // In the HP case, we might traverse between cells with a different
        // number of quadrature points. So we need to resize the output data
        // accordingly.
        if (Dpsi.size() != fe_values.n_quadrature_points ||
            D2psi.size() != fe_values.n_quadrature_points)
          {
            Dpsi.resize(fe_values.n_quadrature_points,
                        Vector<scalar_type>(ad_helper.n_dependent_variables()));
            D2psi.resize(
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
              solution_extraction_data,
              q_point,
              get_field_args(),
              get_field_extractors());

            // Evaluate the functor to compute the total stored energy.
            // To do this, we extract all sensitivities and pass them directly
            // in the user-provided function.
            const energy_type psi =
              OpHelper_t::ad_call_function(ad_helper,
                                           function,
                                           scratch_data,
                                           solution_extraction_data,
                                           q_point,
                                           get_field_extractors());

            // Register the definition of the total stored energy
            ad_helper.register_dependent_variable(psi);

            // Store the output function value, its gradient and linearization.
            ad_helper.compute_gradient(Dpsi[q_point]);
            ad_helper.compute_hessian(D2psi[q_point]);
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
        // Get the unary op field solutions from the EnergyFunctor
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
        return Utilities::get_deal_II_prefix() + "EnergyFunctor_ADHelper_" +
               operand.as_ascii(decorator);
      }

      std::string
      get_name_gradient() const
      {
        const SymbolicDecorations decorator;
        return Utilities::get_deal_II_prefix() +
               "EnergyFunctor_ADHelper_Gradients_" +
               operand.as_ascii(decorator);
      }

      std::string
      get_name_hessian() const
      {
        const SymbolicDecorations decorator;
        return Utilities::get_deal_II_prefix() +
               "EnergyFunctor_ADHelper_Hessians_" + operand.as_ascii(decorator);
      }

      ad_helper_type &
      get_mutable_ad_helper(
        MeshWorker::ScratchData<dim, spacedim> &scratch_data) const
      {
        GeneralDataStorage &cache =
          AD_SD_Functor_Cache::get_cache(scratch_data);
        const std::string name_ad_helper = get_name_ad_helper();

        // Unfortunately we cannot perform a check like this because the
        // ScratchData is reused by many cells during the mesh loop. So
        // there's no real way to verify that the user is not accidentally
        // re-using an object because they forget to uniquely name the
        // EnergyFunctor upon which this op is based.
        //
        // Assert(!(cache.stores_object_with_name(name_ad_helper)),
        //        ExcMessage("ADHelper is already present in the cache."));

        const unsigned int n_independent_variables =
          OpHelper_t::get_n_components();
        return cache.get_or_add_object_with_name<ad_helper_type>(
          name_ad_helper, n_independent_variables);
      }

      std::vector<Vector<scalar_type>> &
      get_mutable_gradients(
        MeshWorker::ScratchData<dim, spacedim> &scratch_data,
        const ad_helper_type &                  ad_helper) const
      {
        GeneralDataStorage &cache =
          AD_SD_Functor_Cache::get_cache(scratch_data);
        const FEValuesBase<dim, spacedim> &fe_values =
          scratch_data.get_current_fe_values();

        return cache
          .get_or_add_object_with_name<std::vector<Vector<scalar_type>>>(
            get_name_gradient(),
            fe_values.n_quadrature_points,
            Vector<scalar_type>(ad_helper.n_dependent_variables()));
      }

      std::vector<FullMatrix<scalar_type>> &
      get_mutable_hessians(MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                           const ad_helper_type &ad_helper) const
      {
        GeneralDataStorage &cache =
          AD_SD_Functor_Cache::get_cache(scratch_data);
        const FEValuesBase<dim, spacedim> &fe_values =
          scratch_data.get_current_fe_values();

        return cache
          .get_or_add_object_with_name<std::vector<FullMatrix<scalar_type>>>(
            get_name_hessian(),
            fe_values.n_quadrature_points,
            FullMatrix<scalar_type>(ad_helper.n_dependent_variables(),
                                    ad_helper.n_independent_variables()));
      }
    };

#  endif // DEAL_II_WITH_AUTO_DIFFERENTIATION


#  ifdef DEAL_II_WITH_SYMENGINE


    /**
     * Extract the value from a energy functor.
     *
     * Variant for symbolic expressions.
     */
    template <int dim,
              int spacedim,
              typename... SymbolicOpsSubSpaceFieldSolution>
    class SymbolicOp<EnergyFunctor<SymbolicOpsSubSpaceFieldSolution...>,
                     SymbolicOpCodes::value,
                     void,
                     Differentiation::SD::Expression,
                     WeakForms::internal::DimPack<dim, spacedim>>
    {
      using Op = EnergyFunctor<SymbolicOpsSubSpaceFieldSolution...>;

      using OpHelper_t = internal::SymbolicOpsSubSpaceFieldSolutionSDHelper<
        SymbolicOpsSubSpaceFieldSolution...>;

    public:
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

      using energy_type = value_type<sd_type>;

      template <typename ResultScalarType>
      using return_type = void;

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
        , name_sd_batch_optimizer(get_name_sd_batch_optimizer(operand))
        , name_evaluated_dependent_functions(
            get_name_evaluated_dependent_functions(operand))
        , symbolic_fields(OpHelper_t::template get_symbolic_fields<sd_type>(
            get_field_args(),
            SymbolicDecorations()))
        , psi(OpHelper_t::template sd_call_function<sd_type>(function,
                                                             symbolic_fields))
        , first_derivatives(
            OpHelper_t::template sd_differentiate<sd_type>(psi,
                                                           symbolic_fields))
        , second_derivatives(
            OpHelper_t::template sd_substitute_and_differentiate<sd_type>(
              first_derivatives,
              OpHelper_t::template sd_call_substitution_function<sd_type>(
                user_intermediate_substitution_map,
                symbolic_fields),
              symbolic_fields))
      {
        OpHelper_t::sd_assert_hash_computed(first_derivatives);
        OpHelper_t::sd_assert_hash_computed(second_derivatives);
      }

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
          AD_SD_Functor_Cache::get_cache(scratch_data);

        return cache.get_object_with_name<sd_helper_type<ResultScalarType>>(
          name_sd_batch_optimizer);
      }

      template <std::size_t FieldIndex>
      const auto &
      get_symbolic_first_derivative() const
      {
        static_assert(FieldIndex < OpHelper_t::n_operators(),
                      "Index out of bounds.");
        return std::get<FieldIndex>(first_derivatives);
      }

      template <std::size_t FieldIndex_1, std::size_t FieldIndex_2>
      const auto &
      get_symbolic_second_derivative() const
      {
        static_assert(FieldIndex_1 < OpHelper_t::n_operators(),
                      "Row index out of bounds.");
        static_assert(FieldIndex_2 < OpHelper_t::n_operators(),
                      "Column index out of bounds.");
        // Get the row tuple, then the column entry in that row tuple.
        return std::get<FieldIndex_2>(
          std::get<FieldIndex_1>(second_derivatives));
      }


      template <typename ResultScalarType>
      const std::vector<std::vector<ResultScalarType>> &
      get_evaluated_dependent_functions(
        const MeshWorker::ScratchData<dim, spacedim> &scratch_data) const
      {
        const GeneralDataStorage &cache =
          AD_SD_Functor_Cache::get_cache(scratch_data);

        return cache
          .get_object_with_name<std::vector<std::vector<ResultScalarType>>>(
            name_evaluated_dependent_functions);
      }

      /**
       * Return values at all quadrature points
       */
      template <typename ResultScalarType>
      return_type<ResultScalarType>
      operator()(MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                 const std::vector<SolutionExtractionData<dim, spacedim>>
                   &solution_extraction_data) const
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
        // theory the user can encode the QPoint into the energy function: this
        // current implementation restricts the user to use the same definition
        // for the energy itself at each QP.
        const auto initialize_optimizer =
          [this](sd_helper_type<ResultScalarType> &batch_optimizer)
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

          // The next typical few steps that precede function registration
          // have already been performed in the class constructor:
          // - Evaluate the functor to compute the total stored energy.
          // - Compute the first derivatives of the energy function.
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
          OpHelper_t::template sd_register_functions<sd_type, energy_type>(
            batch_optimizer, first_derivatives);
          OpHelper_t::template sd_register_functions<sd_type, energy_type>(
            batch_optimizer, second_derivatives);
        };

        // Create and, if necessary, optimize a BatchOptimizer instance.
        // If the user has specified a cache, then this is only done once
        // per number of ScratchData times over the entire lifetime of the
        // cache.
        // If the user has not specified a cache, then this is done once
        // per number of ScratchData times for each assembly step (i.e.
        // with an additional multiplication factor like number of timesteps
        // times number of Newton iterations).
        sd_helper_type<ResultScalarType> &batch_optimizer =
          get_mutable_sd_batch_optimizer<ResultScalarType>(scratch_data);
        if (batch_optimizer.optimized() == false)
          {
            initialize_optimizer(batch_optimizer);

            // Finalize the optimizer.
            // If using the LLVM optimiser in a threaded environment, we have to
            // stagger initialisation so as not to cause some race condition
            // that leads to a segfault.
            if (optimization_method ==
                  Differentiation::SD::OptimizerType::llvm &&
                MultithreadInfo::is_running_single_threaded() == false)
              {
                static std::mutex           mutex;
                std::lock_guard<std::mutex> lock(mutex);
                batch_optimizer.optimize();
              }
            else
              {
                batch_optimizer.optimize();
              }
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
                solution_extraction_data,
                q_point,
                get_symbolic_fields(),
                get_field_args());
            if (user_substitution_map)
              {
                Differentiation::SD::add_to_substitution_map(
                  substitution_map,
                  user_substitution_map(scratch_data,
                                        solution_extraction_data,
                                        q_point));
              }

#    ifdef DEBUG
            // Add a check from  BatchOptimizer<ReturnType>::substitute()
            const Differentiation::SD::types::symbol_vector symbol_sub_vec =
              Differentiation::SD::Utilities::extract_symbols(substitution_map);
            const Differentiation::SD::types::symbol_vector symbol_vec =
              batch_optimizer.get_independent_symbols();
            Assert(symbol_sub_vec.size() == symbol_vec.size(),
                   ExcDimensionMismatch(symbol_sub_vec.size(),
                                        symbol_vec.size()));
            for (unsigned int i = 0; i < symbol_sub_vec.size(); ++i)
              {
                if (!dealii::numbers::values_are_equal(symbol_sub_vec[i],
                                                       symbol_vec[i]))
                  {
                    std::cout << "i: " << i
                              << " ; symbol_sub_vec[i]: " << symbol_sub_vec[i]
                              << " ; symbol_vec[i]: " << symbol_vec[i]
                              << std::endl;
                  }
                Assert(
                  dealii::numbers::values_are_equal(symbol_sub_vec[i],
                                                    symbol_vec[i]),
                  ExcMessage(
                    "The input substitution map is either incomplete, or does "
                    "not match that used in the register_symbols() call. "
                    "If you are using an AD/SD cache, check that the cached "
                    "symbol is a numerical value that changes over time or "
                    "the evaluation history."));
              }
#    endif

            // Perform the value substitution at this quadrature point
            batch_optimizer.substitute(substitution_map);

            // Extract evaluated data to be retrieved later.
            evaluated_dependent_functions[q_point] = batch_optimizer.evaluate();
          }
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
        // Get the unary op field solutions from the EnergyFunctor
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

      // Naming
      const std::string name_sd_batch_optimizer;
      const std::string name_evaluated_dependent_functions;

      // Independent variables
      const typename OpHelper_t::template field_values_t<sd_type>
        symbolic_fields;

      // Dependent variables
      const energy_type psi; // The energy
      const typename OpHelper_t::template first_derivatives_value_t<sd_type,
                                                                    energy_type>
        first_derivatives;
      const typename OpHelper_t::
        template second_derivatives_value_t<sd_type, energy_type>
          second_derivatives;

      static std::string
      get_name_sd_batch_optimizer(const Op &operand)
      {
        const std::hash<std::string> hash_fn;
        const SymbolicDecorations    decorator;
        return Utilities::get_deal_II_prefix() +
               "EnergyFunctor_SDBatchOptimizer_" +
               std::to_string(hash_fn(operand.as_ascii(decorator)));
      }

      static std::string
      get_name_evaluated_dependent_functions(const Op &operand)
      {
        const std::hash<std::string> hash_fn;
        const SymbolicDecorations    decorator;
        return Utilities::get_deal_II_prefix() +
               "EnergyFunctor_ADHelper_Evaluated_Dependent_Functions_" +
               std::to_string(hash_fn(operand.as_ascii(decorator)));
      }

      template <typename ResultScalarType>
      sd_helper_type<ResultScalarType> &
      get_mutable_sd_batch_optimizer(
        MeshWorker::ScratchData<dim, spacedim> &scratch_data) const
      {
        GeneralDataStorage &cache =
          AD_SD_Functor_Cache::get_cache(scratch_data);

        // Unfortunately we cannot perform a check like this because the
        // ScratchData is reused by many cells during the mesh loop. So
        // there's no real way to verify that the user is not accidentally
        // re-using an object because they forget to uniquely name the
        // EnergyFunctor upon which this op is based.
        //
        // Assert(!(cache.stores_object_with_name(name_sd_batch_optimizer)),
        //        ExcMessage("SDBatchOptimizer is already present in the
        //        cache."));

        // Work around a GCC bug, where it cannot disambiguate between a lvalue
        // and rvalue template parameter in
        // GeneralDataStorage::get_or_add_object_with_name()
        enum Differentiation::SD::OptimizerType nc_optimization_method =
          this->optimization_method;
        enum Differentiation::SD::OptimizationFlags nc_optimization_flags =
          this->optimization_flags;

        return cache
          .get_or_add_object_with_name<sd_helper_type<ResultScalarType>>(
            name_sd_batch_optimizer,
            std::move(nc_optimization_method),
            std::move(nc_optimization_flags));
      }

      template <typename ResultScalarType>
      std::vector<std::vector<ResultScalarType>> &
      get_mutable_evaluated_dependent_functions(
        MeshWorker::ScratchData<dim, spacedim> &scratch_data,
        const sd_helper_type<ResultScalarType> &batch_optimizer) const
      {
        GeneralDataStorage &cache =
          AD_SD_Functor_Cache::get_cache(scratch_data);
        const FEValuesBase<dim, spacedim> &fe_values =
          scratch_data.get_current_fe_values();

        return cache.get_or_add_object_with_name<
          std::vector<std::vector<ResultScalarType>>>(
          name_evaluated_dependent_functions,
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

#  endif // DEAL_II_WITH_SYMENGINE

  } // namespace Operators
} // namespace WeakForms



/* ==================== Specialization of type traits ==================== */



/* ==================== Class method definitions ==================== */

namespace WeakForms
{
#  ifdef DEAL_II_WITH_AUTO_DIFFERENTIATION

  template <typename... SymbolicOpsSubSpaceFieldSolution>
  template <typename ADNumberType, int dim, int spacedim>
  DEAL_II_ALWAYS_INLINE inline auto
  EnergyFunctor<SymbolicOpsSubSpaceFieldSolution...>::value(
    const typename WeakForms::EnergyFunctor<
      SymbolicOpsSubSpaceFieldSolution...>::
      template ad_function_type<ADNumberType, dim, spacedim> &function,
    const UpdateFlags update_flags) const
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op = EnergyFunctor<SymbolicOpsSubSpaceFieldSolution...>;
    using ScalarType =
      typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type;
    using OpType = SymbolicOp<Op,
                              SymbolicOpCodes::value,
                              ScalarType,
                              ADNumberType,
                              WeakForms::internal::DimPack<dim, spacedim>>;

    const auto &operand = *this;
    return OpType(operand, function, update_flags);
  }

#  endif // DEAL_II_WITH_AUTO_DIFFERENTIATION

#  ifdef DEAL_II_WITH_SYMENGINE


  template <typename... SymbolicOpsSubSpaceFieldSolution>
  template <typename SDNumberType, int dim, int spacedim>
  DEAL_II_ALWAYS_INLINE inline auto
  EnergyFunctor<SymbolicOpsSubSpaceFieldSolution...>::value(
    const typename WeakForms::EnergyFunctor<
      SymbolicOpsSubSpaceFieldSolution...>::
      template sd_function_type<SDNumberType, dim, spacedim> &function,
    const typename WeakForms::EnergyFunctor<
      SymbolicOpsSubSpaceFieldSolution...>::
      template sd_register_symbols_function_type<SDNumberType, dim, spacedim>
        symbol_registration_map,
    const typename WeakForms::EnergyFunctor<
      SymbolicOpsSubSpaceFieldSolution...>::
      template sd_substitution_function_type<SDNumberType, dim, spacedim>
        substitution_map,
    const typename WeakForms::EnergyFunctor<
      SymbolicOpsSubSpaceFieldSolution...>::
      template sd_intermediate_substitution_function_type<SDNumberType,
                                                          dim,
                                                          spacedim>
                                                  intermediate_substitution_map,
    const enum Differentiation::SD::OptimizerType optimization_method,
    const enum Differentiation::SD::OptimizationFlags optimization_flags,
    const UpdateFlags                                 update_flags) const
  {
    assertOptimizerSettings(optimization_method, optimization_flags);

    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = EnergyFunctor<SymbolicOpsSubSpaceFieldSolution...>;
    using OpType = SymbolicOp<Op,
                              SymbolicOpCodes::value,
                              void,
                              SDNumberType,
                              WeakForms::internal::DimPack<dim, spacedim>>;

    const auto &operand = *this;
    return OpType(operand,
                  function,
                  symbol_registration_map,
                  substitution_map,
                  intermediate_substitution_map,
                  optimization_method,
                  optimization_flags,
                  update_flags);
  }

#  endif // DEAL_II_WITH_SYMENGINE

} // namespace WeakForms



#  ifndef DOXYGEN


namespace WeakForms
{
  // ======= AD =======

#    ifdef DEAL_II_WITH_AUTO_DIFFERENTIATION

  template <typename ADNumberType,
            int dim,
            int spacedim,
            typename... SymbolicOpsSubSpaceFieldSolution>
  struct is_ad_functor_op<Operators::SymbolicOp<
    EnergyFunctor<SymbolicOpsSubSpaceFieldSolution...>,
    Operators::SymbolicOpCodes::value,
    typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type,
    ADNumberType,
    internal::DimPack<dim, spacedim>>> : std::true_type
  {};


  template <typename ADNumberType,
            int dim,
            int spacedim,
            typename... SymbolicOpsSubSpaceFieldSolution>
  struct is_energy_functor_op<Operators::SymbolicOp<
    EnergyFunctor<SymbolicOpsSubSpaceFieldSolution...>,
    Operators::SymbolicOpCodes::value,
    typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type,
    ADNumberType,
    internal::DimPack<dim, spacedim>>> : std::true_type
  {};

#    endif // DEAL_II_WITH_AUTO_DIFFERENTIATION


  // ======= SD =======


#    ifdef DEAL_II_WITH_SYMENGINE


  template <int dim, int spacedim, typename... SymbolicOpsSubSpaceFieldSolution>
  struct is_sd_functor_op<
    Operators::SymbolicOp<EnergyFunctor<SymbolicOpsSubSpaceFieldSolution...>,
                          Operators::SymbolicOpCodes::value,
                          void,
                          Differentiation::SD::Expression,
                          WeakForms::internal::DimPack<dim, spacedim>>>
    : std::true_type
  {};


  template <int dim, int spacedim, typename... SymbolicOpsSubSpaceFieldSolution>
  struct is_energy_functor_op<
    Operators::SymbolicOp<EnergyFunctor<SymbolicOpsSubSpaceFieldSolution...>,
                          Operators::SymbolicOpCodes::value,
                          void,
                          Differentiation::SD::Expression,
                          WeakForms::internal::DimPack<dim, spacedim>>>
    : std::true_type
  {};


#    endif // DEAL_II_WITH_SYMENGINE

} // namespace WeakForms


#  endif // DOXYGEN


#endif // defined(DEAL_II_WITH_SYMENGINE) ||
       // defined(DEAL_II_WITH_AUTO_DIFFERENTIATION)

WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_energy_functor_h
