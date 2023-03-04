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

#ifndef dealii_weakforms_linear_forms_h
#define dealii_weakforms_linear_forms_h

#include <deal.II/base/config.h>

#include <deal.II/base/template_constraints.h>

#include <weak_forms/config.h>
#include <weak_forms/functors.h>
#include <weak_forms/spaces.h>
#include <weak_forms/symbolic_integral.h>
#include <weak_forms/template_constraints.h>
#include <weak_forms/type_traits.h>


WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
  /**
   * @brief A class that represents any linear form.
   *
   * A linear form is a construct that represents the inner product of a
   * test function and an arbitrary functor. It has a special meaning in a
   * mathematical sense; see
   * <a
   * href="https://en.wikipedia.org/wiki/Linear_form">this Wikipedia entry</a>.
   *
   * An instance of this class is not typically created directly by a user, but
   * rather would be preferably generated by one of the linear_form()
   * convenience functions.
   *
   * An example of usage:
   * @code {.cpp}
   * const TestFunction<dim, spacedim> test;
   * const auto                        test_space_op = test.value();
   *
   * const auto functor_op = constant_scalar<dim>(1.0);
   *
   * using TestSpaceOp = decltype(test_space_op);
   * using Functor     = decltype(s);
   *
   * const auto lf
   *   = LinearForm<TestSpaceOp, Functor>(test_space_op, functor_op);
   * @endcode
   *
   * @tparam TestSpaceOp_ A symbolic operator that represents a test function
   * (test space) operation. It may exactly represent the test function or a
   * view into a multi-component test function (or some differential operation
   * involving either of these), or some more complex operation (e.g. a unary,
   * binary or composite operation) that involves a test function.
   * @tparam Functor_ A functor that represents a function with which the inner
   * product is to be taken with respect to the test function. (In some texts,
   * it might be said that this function is "tested" with the test function.)
   *
   * \ingroup forms
   */
  template <typename TestSpaceOp_, typename Functor_>
  class LinearForm
  {
    static_assert(is_or_has_test_function_op<TestSpaceOp_>::value,
                  "Expected a test function.");
    static_assert(
      is_valid_form_functor<Functor_>::value,
      "Expected the functor to a linear form to either be a SymbolicOp, or that the unary or binary operation does not include a test function or trial solution as a lead operation.");
    static_assert(!is_symbolic_integral_op<Functor_>::value,
                  "Functor cannot be an integral.");

  public:
    using TestSpaceOp = TestSpaceOp_;
    using Functor     = Functor_;

    explicit LinearForm(const TestSpaceOp &test_space_op,
                        const Functor &    functor_op)
      : test_space_op(test_space_op)
      , functor_op(functor_op)
    {}

    std::string
    as_ascii(const SymbolicDecorations &decorator) const
    {
      return "(" + test_space_op.as_ascii(decorator) + ", " +
             functor_op.as_ascii(decorator) + ")";
    }

    std::string
    as_latex(const SymbolicDecorations &decorator) const
    {
      constexpr unsigned int n_contracting_indices_tf = WeakForms::Utilities::
        FullIndexContraction<TestSpaceOp, Functor>::n_contracting_indices;

      const bool use_bilinear_form_notation =
        (decorator.get_formatting_latex().get_integral_format() ==
         FormattingLaTeX::IntegralFormat::bilinear_form_notation);

      const std::string symb_mult_tf =
        (use_bilinear_form_notation ?
           ", " :
           Utilities::LaTeX::get_symbol_multiply(n_contracting_indices_tf));

      return decorator.brace_term_when_required_latex(test_space_op) +
             symb_mult_tf +
             decorator.brace_term_when_required_latex(functor_op);
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
      return test_space_op.get_update_flags() | functor_op.get_update_flags();
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

  private:
    const TestSpaceOp test_space_op;
    const Functor     functor_op;
  };

} // namespace WeakForms



/* ======================== Convenience functions ======================== */



namespace WeakForms
{
  /**
   * @brief A convenience function that is used to create linear forms.
   *
   * @tparam TestSpaceOp A symbolic operator that represents a test function
   * (test space) operation. It may exactly represent the test function or a
   * view into a multi-component test function (or some differential operation
   * involving either of these), or some more complex operation (e.g. a unary,
   * binary or composite operation) that involves a test function.
   * @tparam Functor A functor that represents a function with which the inner
   * product is to be taken with respect to both the test function and trial
   * solution.
   * @param test_space_op A symbolic operation that represents some test
   * function (or a derivative thereof).
   * @param functor_op A symbolic operation that represents some general functor.
   *
   * \ingroup forms convenience_functions
   */
  template <
    typename TestSpaceOp,
    typename Functor,
    typename = typename std::enable_if<is_valid_form_functor<Functor>::value &&
                                       !is_scalar_type<Functor>::value>::type>
  LinearForm<TestSpaceOp, Functor>
  linear_form(const TestSpaceOp &test_space_op, const Functor &functor_op)
  {
    return LinearForm<TestSpaceOp, Functor>(test_space_op, functor_op);
  }



  /**
   * @brief A convenience function that is used to create linear forms.
   *
   * This variant takes in a scalar value for the functor.
   *
   * @tparam TestSpaceOp A symbolic operator that represents a test function
   * (test space) operation. It may exactly represent the test function or a
   * view into a multi-component test function (or some differential operation
   * involving either of these), or some more complex operation (e.g. a unary,
   * binary or composite operation) that involves a test function.
   * @tparam ScalarType A scalar type (e.g. a float, double, complex number).
   * @param test_space_op A symbolic operation that represents some test
   * function (or a derivative thereof).
   * @param value A (spatially constant) scalar value.
   *
   * \ingroup forms convenience_functions
   */
  template <typename TestSpaceOp,
            typename ScalarType,
            typename = typename std::enable_if<
              !is_valid_form_functor<ScalarType>::value &&
              is_scalar_type<ScalarType>::value>::type>
  auto
  linear_form(const TestSpaceOp &test_space_op, const ScalarType &value)
  {
    constexpr int dim      = TestSpaceOp::dimension;
    constexpr int spacedim = TestSpaceOp::space_dimension;
    // Delegate to the other function
    return linear_form(test_space_op, constant_scalar<dim, spacedim>(value));
  }



  /**
   * @brief A convenience function that is used to create linear forms.
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
   * @param test_space_op A symbolic operation that represents some test
   * function (or a derivative thereof).
   * @param value A (spatially constant) tensor.
   *
   * \ingroup forms convenience_functions
   */
  template <typename TestSpaceOp,
            int rank,
            int spacedim,
            typename ScalarType,
            typename = typename is_scalar_type<ScalarType>::type>
  auto
  linear_form(const TestSpaceOp &                       test_space_op,
              const Tensor<rank, spacedim, ScalarType> &value)
  {
    constexpr int dim = TestSpaceOp::dimension;
    // Delegate to the other function
    return linear_form(test_space_op, constant_tensor<dim>(value));
  }



  /**
   * @brief A convenience function that is used to create linear forms.
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
   * @param test_space_op A symbolic operation that represents some test
   * function (or a derivative thereof).
   * @param value A (spatially constant) symmetric tensor.
   *
   * \ingroup forms convenience_functions
   */
  template <typename TestSpaceOp,
            int rank,
            int spacedim,
            typename ScalarType,
            typename = typename is_scalar_type<ScalarType>::type>
  auto
  linear_form(const TestSpaceOp &                                test_space_op,
              const SymmetricTensor<rank, spacedim, ScalarType> &value)
  {
    constexpr int dim = TestSpaceOp::dimension;
    // Delegate to the other function
    return linear_form(test_space_op, constant_symmetric_tensor<dim>(value));
  }

} // namespace WeakForms



/* ==================== Specialization of type traits ==================== */



#ifndef DOXYGEN


namespace WeakForms
{
  template <typename TestSpaceOp, typename Functor>
  struct is_linear_form<LinearForm<TestSpaceOp, Functor>> : std::true_type
  {};

} // namespace WeakForms


#endif // DOXYGEN


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_linear_forms_h
