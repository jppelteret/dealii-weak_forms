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

#ifndef dealii_weakforms_bilinear_forms_h
#define dealii_weakforms_bilinear_forms_h

#include <deal.II/base/config.h>

#include <deal.II/base/template_constraints.h>
#include <deal.II/base/types.h>

#include <deal.II/fe/fe_update_flags.h>

#include <weak_forms/config.h>
#include <weak_forms/functors.h>
#include <weak_forms/spaces.h>
#include <weak_forms/symbolic_integral.h>
#include <weak_forms/template_constraints.h>
#include <weak_forms/type_traits.h>
#include <weak_forms/utilities.h>


WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
  /**
   * A set of filters that are applied during the assembly process for bilinear
   * forms.
   *
   * These filters are used to skip some assembly contributions, or to extract
   * certain components of vectorial or tensorial shape functions. To understand
   * what these filters are to do, it is helpful to first define some notation:
   * - `I` and `J` refer to the local degree of freedom indices respectively
   *   associated with the test and trial shape functions.
   * - `component_i` and `component_j` are, respectively, the index of the `I`th
   *   and `J`th shape functions within the finite element (system).
   *   The component of the `K`th shape function would correspond to
   *   `finite_element.system_to_component_index(K).first;`
   * - `multiplicity_I` and `multiplicity_J` are, respectively, the copy of the
   *   base finite element within the finite element (system) that corresponds
   *   with the `I`th and `J`th degree of freedom.
   *   The multiplicity of the `K`th shape function is the equivalent to what is
   *   returned by `finite_element.system_to_base_index(K).first.second;`.
   *
   * Also refer to the introduction of the deal.II
   * [FiniteElement](https://dealii.org/developer/doxygen/deal.II/classFiniteElement.html)
   * class for further insights.
   */
  enum BilinearFormComponentFilter
  {
    /**
     * Apply no filters.
     *
     * @note The use of this filter alone results in optimum performance code
     * being executed by the assembler.
     */
    form_components_default = 0,
    /**
     * Of a vectorial or tensorial basis function result (i.e. fully evaluated
     * shape function operator), extract its "multiplicity index" component
     * that is associated with the `I`th local shape function.
     *
     * @note This extraction operation always occurs in preference to others.
     * So if it is combined with other filters acting on index `I`, then the
     * extraction of this component of the (tensorial) shape function operation
     * result will happen first.
     */
    multiplicity_I = 0x0001,
    /**
     * Of a vectorial or tensorial basis function result (i.e. fully evaluated
     * shape function operator), extract its "multiplicity index" component
     * that is associated with the `J`th local shape function.
     *
     * @note This extraction operation always occurs in preference to others.
     * So if it is combined with other filters acting on index `J`, then the
     * extraction of this component of the (tensorial) shape function operation
     * result will happen first.
     */
    multiplicity_J = 0x0002,
    /**
     * Of a vectorial or tensorial basis function result (i.e. the fully
     * evaluated test shape function operator), extract its "component index"
     * component that is associated with the `I`th local shape function.
     * This is the equivalent of computing the inner product
     * @f$ \left[ \bullet \right]^{I} \cdot \mathbf{e}_{i} @f$
     * where @f$ \left[ \bullet \right]^{I} @f$ denotes the qualified test
     * shape function operator and @f$ \mathbf{e}_{i} @f$ is the @f$ i @f$th
     * Cartesian coordinate direction.
     *
     * @note The use of this filter will result in a slight performance penalty,
     * as it will inhibit certain optimisations from being performed during the
     * assembly process.
     *
     * @warning This can only be called if all shape functions in the finite
     * element (system) are primitive.
     */
    dof_I_component_i = 0x0004,
    /**
     * Of a vectorial or tensorial basis function result (i.e. the fully
     * evaluated test shape function operator), extract its "component index"
     * component that is associated with the `J`th local shape function.
     * This is the equivalent of computing the inner product
     * @f$ \left[ \bullet \right]^{I} \cdot \mathbf{e}_{j} @f$
     * where @f$ \left[ \bullet \right]^{I} @f$ denotes the qualified test
     * shape function operator and @f$ \mathbf{e}_{j} @f$ is the @f$ j @f$th
     * Cartesian coordinate direction.
     *
     * Notice that, for this filter, there is a cross-indexing of the shape
     * function operator and  the component that is to be extracted.
     *
     * @note The use of this filter will result in a slight performance penalty,
     * as it will inhibit certain optimisations from being performed during the
     * assembly process.
     *
     * @warning This can only be called if all shape functions in the finite
     * element (system) are primitive.
     */
    dof_I_component_j = 0x0008,
    /**
     * Of a vectorial or tensorial basis function result (i.e. the fully
     * evaluated trial shape function operator), extract its "component index"
     * component that is associated with the `I`th local shape function.
     * This is the equivalent of computing the inner product
     * @f$ \left[ \bullet \right]^{J} \cdot \mathbf{e}_{i} @f$
     * where @f$ \left[ \bullet \right]^{J} @f$ denotes the qualified trial
     * shape function operator and @f$ \mathbf{e}_{i} @f$ is the @f$ i @f$th
     * Cartesian coordinate direction.
     *
     * Notice that, for this filter, there is a cross-indexing of the shape
     * function operator and  the component that is to be extracted.
     *
     * @note The use of this filter will result in a slight performance penalty,
     * as it will inhibit certain optimisations from being performed during the
     * assembly process.
     *
     * @warning This can only be called if all shape functions in the finite
     * element (system) are primitive.
     */
    dof_J_component_i = 0x0010,
    /**
     * Of a vectorial or tensorial basis function result (i.e. the fully
     * evaluated trial shape function operator), extract its "component index"
     * component that is associated with the `J`th local shape function.
     * This is the equivalent of computing the inner product
     * @f$ \left[ \bullet \right]^{J} \cdot \mathbf{e}_{j} @f$
     * where @f$ \left[ \bullet \right]^{J} @f$ denotes the qualified trial
     * shape function operator and @f$ \mathbf{e}_{j} @f$ is the @f$ j @f$th
     * Cartesian coordinate direction.
     *
     * @note The use of this filter will result in a slight performance penalty,
     * as it will inhibit certain optimisations from being performed during the
     * assembly process.
     *
     * @warning This can only be called if all shape functions in the finite
     * element (system) are primitive.
     */
    dof_J_component_j = 0x0020,
    /**
     * Skip all contributions for which the shape function component is not
     * equal for `component_i` and `component_j` of the `I`th and `J`th degrees
     * of freedom. That is to say, that the associated Cartesian coordinate
     * directions @f$ \mathbf{e}_{i} @f$ and @f$ \mathbf{e}_{j} @f$ are
     * identical.
     *
     * @note If used in isolation, then use of this filter still results in
     * optimum performance code being executed by the assembler.
     */
    local_shape_function_kronecker_delta = 0x0040
  };

  constexpr inline BilinearFormComponentFilter
  operator|(const BilinearFormComponentFilter f1,
            const BilinearFormComponentFilter f2)
  {
    return static_cast<BilinearFormComponentFilter>(
      static_cast<unsigned int>(f1) | static_cast<unsigned int>(f2));
  }

  constexpr inline BilinearFormComponentFilter &
  operator|=(BilinearFormComponentFilter &     f1,
             const BilinearFormComponentFilter f2)
  {
    f1 = f1 | f2;
    return f1;
  }

  constexpr inline BilinearFormComponentFilter
  operator&(const BilinearFormComponentFilter f1,
            const BilinearFormComponentFilter f2)
  {
    return static_cast<BilinearFormComponentFilter>(
      static_cast<unsigned int>(f1) & static_cast<unsigned int>(f2));
  }

  constexpr inline BilinearFormComponentFilter &
  operator&=(BilinearFormComponentFilter &     f1,
             const BilinearFormComponentFilter f2)
  {
    f1 = f1 & f2;
    return f1;
  }


#ifndef DOXYGEN

  constexpr inline bool
  has_no_component_filter(const BilinearFormComponentFilter &flags)
  {
    return flags == form_components_default;
  }

  constexpr inline bool
  has_kronecker_delta_property(const BilinearFormComponentFilter &flags)
  {
    return flags & local_shape_function_kronecker_delta;
  }

  constexpr inline bool
  has_multiplicity_filter_flag(const BilinearFormComponentFilter &flags)
  {
    if (flags & multiplicity_I)
      return true;
    if (flags & multiplicity_J)
      return true;

    return false;
  }

  constexpr inline bool
  has_dof_component_filter_flag(const BilinearFormComponentFilter &flags)
  {
    if (flags & dof_I_component_i)
      return true;
    if (flags & dof_I_component_j)
      return true;
    if (flags & dof_J_component_i)
      return true;
    if (flags & dof_J_component_j)
      return true;

    return false;
  }

  constexpr inline BilinearFormComponentFilter
  get_test_dof_component_filter_flags(const BilinearFormComponentFilter &flags)
  {
    BilinearFormComponentFilter out = form_components_default;

    if (flags & multiplicity_I)
      out |= multiplicity_I;
    if (flags & dof_I_component_i)
      out |= dof_I_component_i;
    if (flags & dof_I_component_j)
      out |= dof_I_component_j;

    return out;
  }

  constexpr inline BilinearFormComponentFilter
  get_trial_dof_component_filter_flags(const BilinearFormComponentFilter &flags)
  {
    BilinearFormComponentFilter out = form_components_default;

    if (flags & multiplicity_J)
      out |= multiplicity_J;
    if (flags & dof_J_component_i)
      out |= dof_J_component_i;
    if (flags & dof_J_component_j)
      out |= dof_J_component_j;

    return out;
  }

#endif // DOXYGEN


  namespace internal
  {
    template <BilinearFormComponentFilter ComponentFilterFlags>
    constexpr void
    check_flag_consistency()
    {
      static_assert(
        !((ComponentFilterFlags & dof_I_component_i) &&
          (ComponentFilterFlags & dof_I_component_j)),
        "Cannot filter test function shape functions on both the i-th and j-th component.");
      static_assert(
        !((ComponentFilterFlags & dof_J_component_i) &&
          (ComponentFilterFlags & dof_J_component_j)),
        "Cannot filter trial solution shape functions on both the i-th and j-th component.");
    }
  } // namespace internal



  /**
   * @brief A class that represents any linear form.
   *
   * A bilinear form is a construct that represents the inner product of a
   * test function, an arbitrary functor, and a trial solution. It has a special
   * meaning in a mathematical sense; see
   * <a
   * href="https://en.wikipedia.org/wiki/Bilinear_form">this Wikipedia
   * entry</a>.
   *
   * An instance of this class is not typically created directly by a user, but
   * rather would be preferrably generated by one of the bilinear_form()
   * convenience functions.
   *
   * An example of usage:
   * @code {.cpp}
   * const TestFunction<dim, spacedim>  test;
   * const TrialSolution<dim, spacedim> trial;
   * const auto                         test_space_op  = test.value();
   * const auto                         trial_space_op = trial.value();
   *
   * const auto functor_op = constant_scalar<dim>(1.0);
   *
   * using TestSpaceOp  = decltype(test_space_op);
   * using Functor      = decltype(s);
   * using TrialSpaceOp = decltype(trial_space_op);
   *
   * const auto blf
   *   = BilinearForm<TestSpaceOp, Functor, TrialSpaceOp>(
   *       test_space_op, functor_op, trial_space_op);
   * @endcode
   *
   * @tparam TestSpaceOp_ A symbolic operator that represents a test function
   * (test space) operation. It may exactly represent the test function or a
   * view into a multi-component test function (or some differential operation
   * involving either of these), or some more complex operation (e.g. a unary,
   * binary or composite operation) that involves a test function.
   * @tparam Functor_ A functor that represents a function with which the inner
   * product is to be taken with respect to both the test function and trial
   * solution.
   * @tparam TrialSpaceOp_ A symbolic operator that represents a trial solution
   * (trial space) operation. It may exactly represent the trial solution or a
   * view into a multi-component trial solution (or some differential operation
   * involving either of these), or some more complex operation (e.g. a unary,
   * binary or composite operation) that involves a trial solution.
   * @tparam ComponentFilterFlags A set of flags that describe some special set
   * of operations that are to be performed when assembling a local element
   * matrix from the bilinear form.
   *
   * \ingroup forms
   */
  template <typename TestSpaceOp_,
            typename Functor_,
            typename TrialSpaceOp_,
            BilinearFormComponentFilter ComponentFilterFlags =
              BilinearFormComponentFilter::form_components_default>
  class BilinearForm
  {
    static_assert(is_or_has_test_function_op<TestSpaceOp_>::value,
                  "Expected a test function.");
    static_assert(
      is_valid_form_functor<Functor_>::value,
      "Expected the functor to a linear form to either be a SymbolicOp, or that the unary or binary operation does not include a test function or trial solution as a lead operation.");
    static_assert(!is_symbolic_integral_op<Functor_>::value,
                  "Functor cannot be an integral.");
    static_assert(is_or_has_trial_solution_op<TrialSpaceOp_>::value,
                  "Expected a trial solution.");

  public:
    using TestSpaceOp  = TestSpaceOp_;
    using Functor      = Functor_;
    using TrialSpaceOp = TrialSpaceOp_;

    explicit BilinearForm(const TestSpaceOp & test_space_op,
                          const Functor &     functor_op,
                          const TrialSpaceOp &trial_space_op)
      : test_space_op(test_space_op)
      , functor_op(functor_op)
      , trial_space_op(trial_space_op)
      , local_contribution_symmetry_flag(false)
    {}

    std::string
    as_ascii(const SymbolicDecorations &decorator) const
    {
      const std::string sym = (is_symmetric() ? "sym" : "");
      const std::string k_delta =
        (has_kronecker_delta_property(ComponentFilterFlags) ? "D^(IJ)" : "");

      const auto contract_with_basis =
        [&decorator](const std::string &component,
                     const bool         shape_function_component = false)
      {
        const auto &naming = decorator.get_naming_ascii().geometry;

        // TODO: get this from a VectorFunctor?
        if (shape_function_component)
          return " . <" + naming.basis + ">^(" + component + ")";
        else
          return " . <" + naming.basis + ">_(" + component + ")";
      };

      // From the assembly point of view, the order of the test/trial
      // function manipulations is important!
      // See extract_test_shape_component() in assembler_base.h
      std::string test_op =
        decorator.brace_term_when_required_ascii(test_space_op);
      if (ComponentFilterFlags & BilinearFormComponentFilter::multiplicity_I)
        test_op += contract_with_basis("I", true);
      if (ComponentFilterFlags & BilinearFormComponentFilter::dof_I_component_i)
        test_op += contract_with_basis("i");
      if (ComponentFilterFlags & BilinearFormComponentFilter::dof_I_component_j)
        test_op += contract_with_basis("j");

      std::string trial_op =
        decorator.brace_term_when_required_ascii(trial_space_op);
      if (ComponentFilterFlags & BilinearFormComponentFilter::multiplicity_J)
        trial_op += contract_with_basis("J", true);
      if (ComponentFilterFlags & BilinearFormComponentFilter::dof_J_component_i)
        trial_op += contract_with_basis("i");
      if (ComponentFilterFlags & BilinearFormComponentFilter::dof_J_component_j)
        trial_op += contract_with_basis("j");

      return sym + "(" + test_op + ", " + functor_op.as_ascii(decorator) +
             ", " + trial_op + ")" + k_delta;
    }

    std::string
    as_latex(const SymbolicDecorations &decorator) const
    {
      const std::string lbrace = Utilities::LaTeX::l_square_brace();
      const std::string rbrace = Utilities::LaTeX::r_square_brace();

      const auto contract_with_basis =
        [&decorator](const std::string &component,
                     unsigned int &     contraction_counter,
                     const bool         shape_function_component = false)
      {
        const std::string symb_mult_tf =
          Utilities::LaTeX::get_symbol_multiply(1);
        ++contraction_counter;

        const auto &naming = decorator.get_naming_latex().geometry;

        // TODO: get this from a VectorFunctor?
        if (shape_function_component)
          return symb_mult_tf + naming.basis + "^{" + component + "}";
        else
          return symb_mult_tf + naming.basis + "_{" + component + "}";
      };

      // From the assembly point of view, the order of the test/trial
      // function manipulations is important!
      // See extract_test_shape_component() in assembler_base.h
      std::string test_op =
        decorator.brace_term_when_required_latex(test_space_op);
      unsigned int n_reduced_indices_tf = 0;
      if (ComponentFilterFlags & BilinearFormComponentFilter::multiplicity_I)
        test_op += contract_with_basis("I", n_reduced_indices_tf, true);
      if (ComponentFilterFlags & BilinearFormComponentFilter::dof_I_component_i)
        test_op += contract_with_basis("i", n_reduced_indices_tf);
      if (ComponentFilterFlags & BilinearFormComponentFilter::dof_I_component_j)
        test_op += contract_with_basis("j", n_reduced_indices_tf);
      if (n_reduced_indices_tf > 0)
        test_op = lbrace + test_op + rbrace;

      std::string trial_op =
        decorator.brace_term_when_required_latex(trial_space_op);
      unsigned int n_reduced_indices_ts = 0;
      if (ComponentFilterFlags & BilinearFormComponentFilter::multiplicity_J)
        trial_op += contract_with_basis("J", n_reduced_indices_ts, true);
      if (ComponentFilterFlags & BilinearFormComponentFilter::dof_J_component_i)
        trial_op += contract_with_basis("i", n_reduced_indices_ts);
      if (ComponentFilterFlags & BilinearFormComponentFilter::dof_J_component_j)
        trial_op += contract_with_basis("j", n_reduced_indices_ts);
      if (n_reduced_indices_ts > 0)
        trial_op = lbrace + trial_op + rbrace;

      const bool use_bilinear_form_notation =
        (decorator.get_formatting_latex().get_integral_format() ==
         FormattingLaTeX::IntegralFormat::bilinear_form_notation);

      // If the functor is scalar valued, then we need to be a bit careful about
      // what the test and trial space ops are (e.g. rank > 0)
      std::string standard_bilinear_form;
      if (Functor::rank == 0)
        {
          const int n_contracting_indices_tt =
            WeakForms::Utilities::FullIndexContraction<TestSpaceOp,
                                                       TrialSpaceOp>::
              get_reduced_n_contracting_indices(n_reduced_indices_tf,
                                                n_reduced_indices_ts);
          Assert(n_contracting_indices_tt >= 0,
                 ExcMessage("Invalid index contraction"));

          // Incorporate the extra indices already contracted out as a part of
          // the bilinear form component filter operations
          const std::string symb_mult_tt =
            (use_bilinear_form_notation ?
               ", " :
               Utilities::LaTeX::get_symbol_multiply(n_contracting_indices_tt));
          const std::string symb_mult_sclr =
            Utilities::LaTeX::get_symbol_multiply(Functor::rank);

          standard_bilinear_form =
            test_op + symb_mult_tt +
            decorator.brace_term_when_required_latex(functor_op) +
            symb_mult_sclr + trial_op;
        }
      else
        {
          const int n_contracting_indices_tf =
            WeakForms::Utilities::FullIndexContraction<TestSpaceOp, Functor>::
              get_reduced_n_contracting_indices(n_reduced_indices_tf, 0);
          const int n_contracting_indices_ft =
            WeakForms::Utilities::FullIndexContraction<Functor, TrialSpaceOp>::
              get_reduced_n_contracting_indices(0, n_reduced_indices_ts);

          Assert(n_contracting_indices_tf >= 0,
                 ExcMessage("Invalid index contraction"));
          Assert(n_contracting_indices_ft >= 0,
                 ExcMessage("Invalid index contraction"));

          // Incorporate the extra indices already contracted out as a part of
          // the bilinear form component filter operations
          const std::string symb_mult_tf =
            (use_bilinear_form_notation ?
               ", " :
               Utilities::LaTeX::get_symbol_multiply(n_contracting_indices_tf));
          const std::string symb_mult_ft =
            Utilities::LaTeX::get_symbol_multiply(n_contracting_indices_ft);

          standard_bilinear_form =
            test_op + symb_mult_tf +
            decorator.brace_term_when_required_latex(functor_op) +
            symb_mult_ft + trial_op;
        }

      if (is_symmetric() || has_kronecker_delta_property(ComponentFilterFlags))
        {
          const std::string sym = (is_symmetric() ? "^{S}" : "");
          const std::string k_delta =
            (has_kronecker_delta_property(ComponentFilterFlags) ?
               "\\delta^{IJ}" :
               "");

          return lbrace + standard_bilinear_form + rbrace + sym + k_delta;
        }
      else
        return standard_bilinear_form;
    }

    // ===== Filters =====

    bool
    is_symmetric() const
    {
      return local_contribution_symmetry_flag;
    }

    static constexpr BilinearFormComponentFilter
    get_component_filter_flags()
    {
      return ComponentFilterFlags;
    }

    // Indicate that the contribution that comes from this form is symmetric.
    //
    // Note: We return this object to facilitate operation chaining.
    BilinearForm &
    symmetrize()
    {
      local_contribution_symmetry_flag = true;
      return *this;
    }

    // Indicate that the contribution that comes from this form
    // for specific combinations of shape functions components as
    // ascertained during the assembly process.
    //
    // The return type is equivalent to
    // BilinearForm<TestSpaceOp, Functor, TrialSpaceOp, ComponentFilterFlags | Flags>
    //
    // Note: We return this object to facilitate operation chaining.
    template <BilinearFormComponentFilter Flags>
    auto
    component_filter()
    {
      internal::check_flag_consistency<ComponentFilterFlags | Flags>();

      return BilinearForm<TestSpaceOp,
                          Functor,
                          TrialSpaceOp,
                          ComponentFilterFlags | Flags>(
        this->get_test_space_operation(),
        this->get_functor(),
        this->get_trial_space_operation());
    }

    // Indicate that the contribution that comes from this form
    // only participates when the shape function components of the
    // test function and trial solution spaces are identical.
    //
    // The return type is equivalent to
    // BilinearForm<TestSpaceOp, Functor, TrialSpaceOp, ComponentFilterFlags | BilinearFormComponentFilter::local_shape_function_kronecker_delta>
    //
    // Note: We return this object to facilitate operation chaining.
    auto
    delta_IJ()
    {
      return this->template component_filter<
        BilinearFormComponentFilter::local_shape_function_kronecker_delta>();
    }

    // ===== Section: Integration =====

    template <
      typename ScalarType    = double,
      typename PredicateType = types::default_volume_integral_predicate_t>
    auto
    dV() const
    {
      return VolumeIntegral<PredicateType>().template integrate<ScalarType>(
        *this);
    }

    template <
      typename ScalarType    = double,
      typename PredicateType = types::default_volume_integral_predicate_t>
    auto
    dV(
      const typename VolumeIntegral<PredicateType>::subdomain_t subdomain) const
    {
      return dV<ScalarType>(
        std::set<typename VolumeIntegral<PredicateType>::subdomain_t>{
          subdomain});
    }

    template <
      typename ScalarType    = double,
      typename PredicateType = types::default_volume_integral_predicate_t>
    auto
    dV(const std::set<typename VolumeIntegral<PredicateType>::subdomain_t>
         &subdomains) const
    {
      return VolumeIntegral<PredicateType>(subdomains)
        .template integrate<ScalarType>(*this);
    }

    template <
      typename ScalarType    = double,
      typename PredicateType = types::default_boundary_integral_predicate_t>
    auto
    dA() const
    {
      return BoundaryIntegral<PredicateType>().template integrate<ScalarType>(
        *this);
      // return integrate<ScalarType>(*this, BoundaryIntegral<PredicateType>());
    }

    template <
      typename ScalarType    = double,
      typename PredicateType = types::default_boundary_integral_predicate_t>
    auto
    dA(const typename BoundaryIntegral<PredicateType>::subdomain_t boundary)
      const
    {
      return dA<ScalarType>(
        std::set<typename BoundaryIntegral<PredicateType>::subdomain_t>{
          boundary});
    }

    template <
      typename ScalarType    = double,
      typename PredicateType = types::default_boundary_integral_predicate_t>
    auto
    dA(const std::set<typename BoundaryIntegral<PredicateType>::subdomain_t>
         &boundaries) const
    {
      return BoundaryIntegral<PredicateType>(boundaries)
        .template integrate<ScalarType>(*this);
    }

    template <
      typename ScalarType    = double,
      typename PredicateType = types::default_interface_integral_predicate_t>
    auto
    dI() const
    {
      return InterfaceIntegral<PredicateType>().template integrate<ScalarType>(
        *this);
    }

    template <
      typename ScalarType    = double,
      typename PredicateType = types::default_interface_integral_predicate_t>
    auto
    dI(const typename InterfaceIntegral<PredicateType>::subdomain_t interface)
      const
    {
      return dI<ScalarType>(
        std::set<typename InterfaceIntegral<PredicateType>::subdomain_t>{
          interface});
    }

    template <
      typename ScalarType    = double,
      typename PredicateType = types::default_interface_integral_predicate_t>
    auto
    dI(const std::set<typename InterfaceIntegral<PredicateType>::subdomain_t>
         &interfaces) const
    {
      return InterfaceIntegral<PredicateType>(interfaces)
        .template integrate<ScalarType>(*this);
    }

    // ===== Section: Construct assembly operation =====

    UpdateFlags
    get_update_flags() const
    {
      return test_space_op.get_update_flags() | functor_op.get_update_flags() |
             trial_space_op.get_update_flags();
    }

    const TestSpaceOp &
    get_test_space_operation() const
    {
      return test_space_op;
    }

    const Functor &
    get_functor() const
    {
      return functor_op;
    }

    const TrialSpaceOp &
    get_trial_space_operation() const
    {
      return trial_space_op;
    }

  private:
    const TestSpaceOp  test_space_op;
    const Functor      functor_op;
    const TrialSpaceOp trial_space_op;
    bool local_contribution_symmetry_flag; // Indicate whether or not this local
                                           // contribution is a symmetric one
  };

} // namespace WeakForms



/* ======================== Convenience functions ======================== */



namespace WeakForms
{
  /**
   * @brief A convenience function that is used to create bilinear forms.
   *
   * @tparam TestSpaceOp A symbolic operator that represents a test function
   * (test space) operation. It may exactly represent the test function or a
   * view into a multi-component test function (or some differential operation
   * involving either of these), or some more complex operation (e.g. a unary,
   * binary or composite operation) that involves a test function.
   * @tparam Functor A functor that represents a function with which the inner
   * product is to be taken with respect to both the test function and trial
   * solution.
   * @tparam TrialSpaceOp A symbolic operator that represents a trial solution
   * (trial space) operation. It may exactly represent the trial solution or a
   * view into a multi-component trial solution (or some differential operation
   * involving either of these), or some more complex operation (e.g. a unary,
   * binary or composite operation) that involves a trial solution.
   * @param test_space_op A symbolic operation that represents some test
   * function (or a derivative thereof).
   * @param functor_op A symbolic operation that represents some general functor.
   * @param trial_space_op A symbolic operation that represents some trial
   * solution (or a derivative thereof).
   *
   * \ingroup forms convenience_functions
   */
  template <
    typename TestSpaceOp,
    typename Functor,
    typename TrialSpaceOp,
    typename = typename std::enable_if<is_valid_form_functor<Functor>::value &&
                                       !is_scalar_type<Functor>::value>::type>
  BilinearForm<TestSpaceOp, Functor, TrialSpaceOp>
  bilinear_form(const TestSpaceOp & test_space_op,
                const Functor &     functor_op,
                const TrialSpaceOp &trial_space_op)
  {
    return BilinearForm<TestSpaceOp, Functor, TrialSpaceOp>(test_space_op,
                                                            functor_op,
                                                            trial_space_op);
  }



  /**
   * @brief A convenience function that is used to create bilinear forms.
   *
   * This variant takes in a scalar value for the functor.
   *
   * @tparam TestSpaceOp A symbolic operator that represents a test function
   * (test space) operation. It may exactly represent the test function or a
   * view into a multi-component test function (or some differential operation
   * involving either of these), or some more complex operation (e.g. a unary,
   * binary or composite operation) that involves a test function.
   * @tparam ScalarType A scalar type (e.g. a float, double, complex number).
   * @tparam TrialSpaceOp A symbolic operator that represents a trial solution
   * (trial space) operation. It may exactly represent the trial solution or a
   * view into a multi-component trial solution (or some differential operation
   * involving either of these), or some more complex operation (e.g. a unary,
   * binary or composite operation) that involves a trial solution.
   * @param test_space_op A symbolic operation that represents some test
   * function (or a derivative thereof).
   * @param value A (spatially constant) scalar value.
   * @param trial_space_op A symbolic operation that represents some trial
   * solution (or a derivative thereof).
   *
   * \ingroup forms convenience_functions
   */
  template <typename TestSpaceOp,
            typename ScalarType,
            typename TrialSpaceOp,
            typename = typename std::enable_if<
              !is_valid_form_functor<ScalarType>::value &&
              is_scalar_type<ScalarType>::value>::type>
  auto
  bilinear_form(const TestSpaceOp & test_space_op,
                const ScalarType &  value,
                const TrialSpaceOp &trial_space_op)
  {
    constexpr int dim      = TestSpaceOp::dimension;
    constexpr int spacedim = TestSpaceOp::space_dimension;
    // Delegate to the other function
    return bilinear_form(test_space_op,
                         constant_scalar<dim, spacedim>(value),
                         trial_space_op);
  }



  /**
   * @brief A convenience function that is used to create bilinear forms.
   *
   * This variant takes in a tensor value for the functor.
   *
   * @tparam TestSpaceOp A symbolic operator that represents a test function
   * (test space) operation. It may exactly represent the test function or a
   * view into a multi-component test function (or some differential operation
   * involving either of these), or some more complex operation (e.g. a unary,
   * binary or composite operation) that involves a test function.
   * @tparam rank The rank of the tensor that is returned upon evaluation.
   * @tparam spacedim The spatial dimension of the input tensor.
   * @tparam ScalarType The underlying scalar type for each component of the
   * tensor.
   * @tparam TrialSpaceOp A symbolic operator that represents a trial solution
   * (trial space) operation. It may exactly represent the trial solution or a
   * view into a multi-component trial solution (or some differential operation
   * involving either of these), or some more complex operation (e.g. a unary,
   * binary or composite operation) that involves a trial solution.
   * @param test_space_op A symbolic operation that represents some test
   * function (or a derivative thereof).
   * @param value A (spatially constant) tensor.
   * @param trial_space_op A symbolic operation that represents some trial
   * solution (or a derivative thereof).
   *
   * \ingroup forms convenience_functions
   */
  template <typename TestSpaceOp,
            int rank,
            int spacedim,
            typename ScalarType,
            typename TrialSpaceOp,
            typename = typename is_scalar_type<ScalarType>::type>
  auto
  bilinear_form(const TestSpaceOp &                       test_space_op,
                const Tensor<rank, spacedim, ScalarType> &value,
                const TrialSpaceOp &                      trial_space_op)
  {
    constexpr int dim = TestSpaceOp::dimension;
    // Delegate to the other function
    return bilinear_form(test_space_op,
                         constant_tensor<dim>(value),
                         trial_space_op);
  }



  /**
   * @brief A convenience function that is used to create bilinear forms.
   *
   * This variant takes in a symmetric tensor value for the functor.
   *
   * @tparam TestSpaceOp A symbolic operator that represents a test function
   * (test space) operation. It may exactly represent the test function or a
   * view into a multi-component test function (or some differential operation
   * involving either of these), or some more complex operation (e.g. a unary,
   * binary or composite operation) that involves a test function.
   * @tparam rank The rank of the tensor that is returned upon evaluation.
   * @tparam spacedim The spatial dimension of the input tensor.
   * @tparam ScalarType The underlying scalar type for each component of the
   * tensor.
   * @tparam TrialSpaceOp A symbolic operator that represents a trial solution
   * (trial space) operation. It may exactly represent the trial solution or a
   * view into a multi-component trial solution (or some differential operation
   * involving either of these), or some more complex operation (e.g. a unary,
   * binary or composite operation) that involves a trial solution.
   * @param test_space_op A symbolic operation that represents some test
   * function (or a derivative thereof).
   * @param value A (spatially constant) symmetric tensor.
   * @param trial_space_op A symbolic operation that represents some trial
   * solution (or a derivative thereof).
   *
   * \ingroup forms convenience_functions
   */
  template <typename TestSpaceOp,
            int rank,
            int spacedim,
            typename ScalarType,
            typename TrialSpaceOp,
            typename = typename is_scalar_type<ScalarType>::type>
  auto
  bilinear_form(const TestSpaceOp &test_space_op,
                const SymmetricTensor<rank, spacedim, ScalarType> &value,
                const TrialSpaceOp &trial_space_op)
  {
    constexpr int dim = TestSpaceOp::dimension;
    // Delegate to the other function
    return bilinear_form(test_space_op,
                         constant_symmetric_tensor<dim>(value),
                         trial_space_op);
  }



  /**
   * @brief A convenience function that is used to create bilinear forms.
   *
   * This specialised variant with no functor specified. We assume a unity
   * scalar functor is the equivalent operation.
   *
   * @tparam TestSpaceOp A symbolic operator that represents a test function
   * (test space) operation. It may exactly represent the test function or a
   * view into a multi-component test function (or some differential operation
   * involving either of these), or some more complex operation (e.g. a unary,
   * binary or composite operation) that involves a test function.
   * @tparam TrialSpaceOp A symbolic operator that represents a trial solution
   * (trial space) operation. It may exactly represent the trial solution or a
   * view into a multi-component trial solution (or some differential operation
   * involving either of these), or some more complex operation (e.g. a unary,
   * binary or composite operation) that involves a trial solution.
   * @param test_space_op A symbolic operation that represents some test
   * function (or a derivative thereof).
   * @param trial_space_op A symbolic operation that represents some trial
   * solution (or a derivative thereof).
   *
   * \ingroup forms convenience_functions
   */
  template <typename TestSpaceOp, typename TrialSpaceOp>
  auto
  bilinear_form(const TestSpaceOp & test_space_op,
                const TrialSpaceOp &trial_space_op)
  {
    // Delegate to the other function
    return bilinear_form(test_space_op, 1.0, trial_space_op);
  }

} // namespace WeakForms



/* ==================== Specialization of type traits ==================== */



#ifndef DOXYGEN


namespace WeakForms
{
  template <typename TestSpaceOp,
            typename Functor,
            typename TrialSpaceOp,
            BilinearFormComponentFilter ComponentFilter>
  struct is_bilinear_form<
    BilinearForm<TestSpaceOp, Functor, TrialSpaceOp, ComponentFilter>>
    : std::true_type
  {};

} // namespace WeakForms


#endif // DOXYGEN


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_bilinear_forms_h
