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
  enum BilinearFormComponentFilter
  {
    form_components_default              = 0,
    dof_I_component_i                    = 0x0001,
    dof_I_component_j                    = 0x0002,
    dof_J_component_i                    = 0x0004,
    dof_J_component_j                    = 0x0008,
    local_shape_function_kronecker_delta = 0x0010
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

  constexpr inline bool
  has_kronecker_delta_property(const BilinearFormComponentFilter &flags)
  {
    return flags & local_shape_function_kronecker_delta;
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


  namespace internal
  {
    template <BilinearFormComponentFilter Flags>
    constexpr void
    check_flag_consistency()
    {
      static_assert(!((Flags & dof_I_component_i) &&
                      (Flags & dof_I_component_j)),
                    "Invalid flag combination.");
      static_assert(!((Flags & dof_J_component_i) &&
                      (Flags & dof_J_component_j)),
                    "Invalid flag combination.");
    }
  } // namespace internal



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
      // TODO: has_kronecker_delta_property()
      // TODO: is_symmetric()
      return "(" + test_space_op.as_ascii(decorator) + ", " +
             functor_op.as_ascii(decorator) + ", " +
             trial_space_op.as_ascii(decorator) + ")";
    }

    std::string
    as_latex(const SymbolicDecorations &decorator) const
    {
      const std::string lbrace = Utilities::LaTeX::l_square_brace();
      const std::string rbrace = Utilities::LaTeX::r_square_brace();

      // If the functor is scalar valued, then we need to be a bit careful about
      // what the test and trial space ops are (e.g. rank > 0)
      if (Functor::rank == 0)
        {
          constexpr unsigned int n_contracting_indices_tt =
            WeakForms::Utilities::FullIndexContraction<
              TestSpaceOp,
              TrialSpaceOp>::n_contracting_indices;

          const std::string symb_mult_tt =
            Utilities::LaTeX::get_symbol_multiply(n_contracting_indices_tt);
          const std::string symb_mult_sclr =
            Utilities::LaTeX::get_symbol_multiply(Functor::rank);

          // TODO: has_kronecker_delta_property()
          // TODO: is_symmetric()
          return lbrace + test_space_op.as_latex(decorator) + symb_mult_tt +
                 lbrace + functor_op.as_latex(decorator) + symb_mult_sclr +
                 trial_space_op.as_latex(decorator) + rbrace + rbrace;
        }
      else
        {
          constexpr unsigned int n_contracting_indices_tf =
            WeakForms::Utilities::FullIndexContraction<TestSpaceOp, Functor>::
              n_contracting_indices;
          constexpr unsigned int n_contracting_indices_ft =
            WeakForms::Utilities::FullIndexContraction<Functor, TrialSpaceOp>::
              n_contracting_indices;
          const std::string symb_mult_tf =
            Utilities::LaTeX::get_symbol_multiply(n_contracting_indices_tf);
          const std::string symb_mult_ft =
            Utilities::LaTeX::get_symbol_multiply(n_contracting_indices_ft);

          // TODO: has_kronecker_delta_property()
          // TODO: is_symmetric()
          return lbrace + test_space_op.as_latex(decorator) + symb_mult_tf +
                 functor_op.as_latex(decorator) + symb_mult_ft +
                 trial_space_op.as_latex(decorator) + rbrace;
        }
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
      return integrate<ScalarType>(*this, BoundaryIntegral());
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
    dA(const std::set<typename BoundaryIntegral::subdomain_t> &boundaries) const
    {
      return BoundaryIntegral(boundaries).template integrate<ScalarType>(*this);
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
    dI(
      const std::set<typename InterfaceIntegral::subdomain_t> &interfaces) const
    {
      return InterfaceIntegral(interfaces)
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
  // template <typename TestSpaceOp, typename TrialSpaceOp>
  // BilinearForm<TestSpaceOp, NoOp, TrialSpaceOp>
  // bilinear_form(const TestSpaceOp & test_space_op,
  //               const TrialSpaceOp &trial_space_op)
  // {
  //   return BilinearForm<TestSpaceOp, NoOp, TrialSpaceOp>(test_space_op,
  //                                                        trial_space_op);
  // }

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


  // A specialised variant with no functor specified. We assume a unity
  // scalar functor is the equivalent operation.
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
