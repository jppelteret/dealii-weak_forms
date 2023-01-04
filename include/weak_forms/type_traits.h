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

#ifndef dealii_weakforms_type_traits_h
#define dealii_weakforms_type_traits_h

#include <deal.II/base/config.h>

#include <weak_forms/config.h>

#include <type_traits>


WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
  /**
    @brief A type trait that checks to see if a class is a symbolic operator that represents an operation involving the test function (or a component thereof).
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T>
  struct is_test_function : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator that represents an operation involving the trial solution (or a component thereof).
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T>
  struct is_trial_solution : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator that represents an operation involving the field solution (or a component thereof).
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T>
  struct is_field_solution : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a view into the global test function, trial solution, or field solution.
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T>
  struct is_subspace_view : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is an elementary or composite (unary or binary) symbolic operator that involves the test function (or a view into it).
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T>
  struct is_test_function_op : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is an elementary or composite (unary or binary) symbolic operator that involves the trial solution (or a view into it).
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T>
  struct is_trial_solution_op : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is an elementary or composite (unary or binary) symbolic operator that involves the field solution (or a view into it).
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T>
  struct is_field_solution_op : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator that involves a geometry operation of a discrete finite element (cell).
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T>
  struct is_cell_geometry_op : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator that represents an elementary operation involving a functor.
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T>
  struct is_functor_op : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator that represents an elementary operation involving a cached functor.
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T>
  struct is_cache_functor_op : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator that is automatically differentiable.
   *
   * The scalar type associated with this class must be one that is an
   * auto-differentiable number, e.g. one provided by the Trilinos Sacado or
   * ADOL-C libraries.
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T>
  struct is_ad_functor_op : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator that is symbolically differentiable.
   *
   * The scalar type associated with this class must be one that is an
   * symbolically differentiable number, e.g. one provided by the SymEngine
   * library.
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T>
  struct is_sd_functor_op : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator represents an energy functor.
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T>
  struct is_energy_functor_op : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator represents a residual functor.
   *
   * Specifically, this type trait will evaluate to `std::true_type` when the
   * symbolic operator is a ResidualViewFunctor, which extracts a subset of
   * fields associated with the overall residual.
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T>
  struct is_residual_functor_op : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator that must be evaluated using deal.II's MeshWorker::ScratchData class.
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T, typename U = void>
  struct is_evaluated_with_scratch_data : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator that can be sensibly used in conjunction with a scalar arithmetic operation.
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T, typename = void>
  struct is_compatible_with_scalar_arithmetic : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator that can be evaluated on a boundary.
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T>
  struct is_boundary_op : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator that can be evaluated on an interface.
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T>
  struct is_interface_op : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a valid elementary of composite (unary or binary) symbolic operator that can be passed into, and evaluated within, a linear or binary form.
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T, typename U = void>
  struct is_valid_form_functor : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator that represents a binary form.
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T>
  struct is_bilinear_form : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator that represents a linear form.
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T>
  struct is_linear_form : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator that represents a form that has the ability to linearise itself.
   *
   * In practice, this means that the form must represent an energy functor or
   * residual view functor operation.
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T>
  struct is_self_linearizing_form : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator that represents a volume integral (domain).
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T>
  struct is_volume_integral_op : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator that represents a boundary integral (domain).
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T>
  struct is_boundary_integral_op : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator that represents an interface integral (domain).
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T>
  struct is_interface_integral_op : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator that represents a domain of integration.
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T>
  struct is_valid_integration_domain : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator that represents a unary operator.
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T, typename U = void>
  struct is_unary_op : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator that represents a binary operator.
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T>
  struct is_binary_op : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator that represents another elementary operator that has been integrated.
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T, typename U = void>
  struct is_symbolic_integral_op : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator that represents a unary operator that has been integrated.
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T, typename U = void>
  struct is_unary_integral_op : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator that represents a binary operator that has been integrated.
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T, typename U = void>
  struct is_binary_integral_op : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a class is a symbolic operator that represents any operator that has been integrated.
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T, typename U = void>
  struct is_integral_op : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a operator is a differential operator.
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T, typename U = void>
  struct is_continuous_differential_operator : std::false_type
  {};


  /**
   * @brief A type trait that checks to see if a operator requires bracing being parse for LaTeX output.
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T, typename U = void>
  struct operand_requires_braced_decoration : std::false_type
  {};

} // namespace WeakForms


#ifndef DOXYGEN


namespace WeakForms
{
// A trait that declares that one or more leaf operations
// is a test function operation.
// The left itself will not be marked as one.
#  define DEAL_II_EXPRESSION_TREE_TYPE_TRAIT(expression_tree_type_trait,   \
                                             singular_type_trait)          \
    template <typename T, typename U = void>                               \
    struct expression_tree_type_trait : std::false_type                    \
    {};                                                                    \
                                                                           \
    template <typename T>                                                  \
    struct expression_tree_type_trait<                                     \
      T,                                                                   \
      typename std::enable_if<                                             \
        is_unary_op<T>::value &&                                           \
        singular_type_trait<typename T::OpType>::value>::type>             \
      : std::true_type                                                     \
    {};                                                                    \
                                                                           \
    template <typename T>                                                  \
    struct expression_tree_type_trait<                                     \
      T,                                                                   \
      typename std::enable_if<                                             \
        is_unary_op<T>::value &&                                           \
        !singular_type_trait<typename T::OpType>::value &&                 \
        expression_tree_type_trait<typename T::OpType>::value>::type>      \
      : std::true_type                                                     \
    {};                                                                    \
                                                                           \
    template <typename T>                                                  \
    struct expression_tree_type_trait<                                     \
      T,                                                                   \
      typename std::enable_if<                                             \
        is_binary_op<T>::value &&                                          \
        (singular_type_trait<typename T::LhsOpType>::value ||              \
         singular_type_trait<typename T::RhsOpType>::value)>::type>        \
      : std::true_type                                                     \
    {};                                                                    \
                                                                           \
    template <typename T>                                                  \
    struct expression_tree_type_trait<                                     \
      T,                                                                   \
      typename std::enable_if<                                             \
        is_binary_op<T>::value &&                                          \
        !singular_type_trait<typename T::LhsOpType>::value &&              \
        !singular_type_trait<typename T::RhsOpType>::value &&              \
        (expression_tree_type_trait<typename T::LhsOpType>::value ||       \
         expression_tree_type_trait<typename T::RhsOpType>::value)>::type> \
      : std::true_type                                                     \
    {};

  // A trait that declares that one or more leaf operations
  // is a test function operation.
  // The left itself will not be marked as one.
  //
  // These types are, in essence, modifiers for test functions.
  // e.g. like taking the transpose or negation of test function,
  // or even nesting it in a binary op.
  DEAL_II_EXPRESSION_TREE_TYPE_TRAIT(has_test_function_op, is_test_function_op)

  // A trait that declares that one or more leaf operations
  // is a trial solution operation.
  // The left itself will not be marked as one.
  //
  // These types are, in essence, modifiers for trial solutions
  // e.g. like taking the transpose or negation of trial solution,
  // or even nesting it in a binary op.
  DEAL_II_EXPRESSION_TREE_TYPE_TRAIT(has_trial_solution_op,
                                     is_trial_solution_op)

  // A trait that declares that one or more leaf operations
  // is a field solution operation.
  // The left itself will not be marked as one.
  //
  // These types are, in essence, modifiers for field_solutions
  // e.g. like taking the transpose or negation of field solution,
  // or even nesting it in a binary op.
  DEAL_II_EXPRESSION_TREE_TYPE_TRAIT(has_field_solution_op,
                                     is_field_solution_op)


  // A trait that declares that one or more leaf operations
  // is one that must be evaluated using scratch data.
  // The left itself will not be marked as one.
  DEAL_II_EXPRESSION_TREE_TYPE_TRAIT(has_evaluated_with_scratch_data,
                                     is_evaluated_with_scratch_data)


  // A trait that declares that one or more leaf operations
  // is a operation that can only be performed on a boundary.
  // The left itself will not be marked as one.
  DEAL_II_EXPRESSION_TREE_TYPE_TRAIT(has_boundary_op, is_boundary_op)


  // A trait that declares that one or more leaf operations
  // is a operation that can only be performed on an interface.
  // The left itself will not be marked as one.
  DEAL_II_EXPRESSION_TREE_TYPE_TRAIT(has_interface_op, is_interface_op)

  // Integrals
  DEAL_II_EXPRESSION_TREE_TYPE_TRAIT(has_symbolic_integral_op,
                                     is_symbolic_integral_op)
  DEAL_II_EXPRESSION_TREE_TYPE_TRAIT(has_unary_integral_op,
                                     is_unary_integral_op)
  DEAL_II_EXPRESSION_TREE_TYPE_TRAIT(has_binary_integral_op,
                                     is_binary_integral_op)

#  undef DEAL_II_EXPRESSION_TREE_TYPE_TRAIT


#  define DEAL_II_TYPE_TRAIT_OR_COMBINER(compound_type_trait,                 \
                                         type_trait_1,                        \
                                         type_trait_2)                        \
    template <typename T, typename U = void>                                  \
    struct compound_type_trait : std::false_type                              \
    {};                                                                       \
                                                                              \
    template <typename T>                                                     \
    struct compound_type_trait<                                               \
      T,                                                                      \
      typename std::enable_if<type_trait_1<T>::value ||                       \
                              type_trait_2<T>::value>::type> : std::true_type \
    {};


  DEAL_II_TYPE_TRAIT_OR_COMBINER(is_or_has_test_function_op,
                                 is_test_function_op,
                                 has_test_function_op)

  DEAL_II_TYPE_TRAIT_OR_COMBINER(is_or_has_trial_solution_op,
                                 is_trial_solution_op,
                                 has_trial_solution_op)

  DEAL_II_TYPE_TRAIT_OR_COMBINER(is_test_function_or_trial_solution_op,
                                 is_test_function_op,
                                 is_trial_solution_op)

  DEAL_II_TYPE_TRAIT_OR_COMBINER(has_test_function_or_trial_solution_op,
                                 has_test_function_op,
                                 has_trial_solution_op)

  DEAL_II_TYPE_TRAIT_OR_COMBINER(is_or_has_test_function_or_trial_solution_op,
                                 is_test_function_or_trial_solution_op,
                                 has_test_function_or_trial_solution_op)

  DEAL_II_TYPE_TRAIT_OR_COMBINER(is_or_has_evaluated_with_scratch_data,
                                 is_evaluated_with_scratch_data,
                                 has_evaluated_with_scratch_data)

  DEAL_II_TYPE_TRAIT_OR_COMBINER(is_or_has_boundary_op,
                                 is_boundary_op,
                                 has_boundary_op)

  DEAL_II_TYPE_TRAIT_OR_COMBINER(is_or_has_interface_op,
                                 is_interface_op,
                                 has_interface_op)

  // Integrals
  DEAL_II_TYPE_TRAIT_OR_COMBINER(is_or_has_symbolic_integral_op,
                                 is_symbolic_integral_op,
                                 has_symbolic_integral_op)

  DEAL_II_TYPE_TRAIT_OR_COMBINER(is_or_has_unary_integral_op,
                                 is_unary_integral_op,
                                 has_unary_integral_op)

  DEAL_II_TYPE_TRAIT_OR_COMBINER(is_or_has_binary_integral_op,
                                 is_binary_integral_op,
                                 has_binary_integral_op)

  DEAL_II_TYPE_TRAIT_OR_COMBINER(is_symbolic_or_unary_integral_op,
                                 is_symbolic_integral_op,
                                 is_unary_integral_op)


#  undef DEAL_II_TYPE_TRAIT_OR_COMBINER

} // namespace WeakForms


namespace WeakForms
{
  // TODO: Add test for this
  template <typename T>
  struct is_evaluated_with_scratch_data<
    T,
    typename std::enable_if<is_field_solution_op<T>::value ||
                            is_cache_functor_op<T>::value>::type>
    : std::true_type
  {};

  // These are treated specially, so we note this explicitly.
  template <typename T>
  struct is_evaluated_with_scratch_data<
    T,
    typename std::enable_if<is_ad_functor_op<T>::value ||
                            is_sd_functor_op<T>::value>::type> : std::false_type
  {};


  template <typename T>
  struct is_valid_form_functor<
    T,
    typename std::enable_if<
      (is_functor_op<T>::value || is_cache_functor_op<T>::value ||
       is_field_solution_op<T>::value) &&
      !(is_unary_op<T>::value || is_binary_op<T>::value)>::type>
    : std::true_type
  {};

  template <typename T>
  struct is_valid_form_functor<
    T,
    typename std::enable_if<
      (is_unary_op<T>::value || is_binary_op<T>::value) &&
      !is_or_has_test_function_or_trial_solution_op<T>::value>::type>
    : std::true_type
  {};

  // template <typename T>
  // struct is_valid_form_functor<
  //   T,
  //   typename std::enable_if<
  //     is_binary_op<T>::value &&
  //     !is_or_has_test_function_or_trial_solution_op<T>::value>::type>
  //   : std::true_type
  // {};

  // template <typename T>
  // struct is_integral_op<
  //   T,
  //   typename std::enable_if<is_symbolic_integral_op<T>::value ||
  //                           is_unary_integral_op<T>::value ||
  //                           is_binary_integral_op<T>::value>::value>
  //   : std::true_type
  // {};

  // template <typename T>
  // struct is_integral_op<
  //   T,
  //   typename std::enable_if<is_or_has_symbolic_integral_op<T>::value ||
  //                           is_or_has_unary_integral_op<T>::value ||
  //                           is_or_has_binary_integral_op<T>::value>::value>
  //   : std::true_type
  // {};


  // Arithmetic

  template <typename T>
  struct is_compatible_with_scalar_arithmetic<
    T,
    typename std::enable_if<
      (is_test_function_op<T>::value || is_trial_solution_op<T>::value ||
       is_field_solution_op<T>::value || is_cell_geometry_op<T>::value ||
       is_functor_op<T>::value || is_cache_functor_op<T>::value) &&
      !(is_unary_op<T>::value || is_binary_op<T>::value)>::type>
    : std::true_type
  {};

  template <typename T>
  struct is_compatible_with_scalar_arithmetic<
    T,
    typename std::enable_if<is_symbolic_integral_op<T>::value>::type>
    : std::false_type
  {};

  template <typename T>
  struct is_compatible_with_scalar_arithmetic<
    T,
    typename std::enable_if<
      is_unary_op<T>::value && !is_unary_integral_op<T>::value &&
      is_compatible_with_scalar_arithmetic<typename T::OpType>::value>::type>
    : std::true_type
  {};

  template <typename T>
  struct is_compatible_with_scalar_arithmetic<
    T,
    typename std::enable_if<
      is_binary_op<T>::value && !is_binary_integral_op<T>::value &&
      is_compatible_with_scalar_arithmetic<typename T::LhsOpType>::value &&
      is_compatible_with_scalar_arithmetic<typename T::RhsOpType>::value>::type>
    : std::true_type
  {};

  template <typename T, typename = void>
  struct is_compatible_with_tensor_arithmetic : std::false_type
  {};

  template <typename T>
  struct is_compatible_with_tensor_arithmetic<
    T,
    typename std::enable_if<
      (is_test_function_op<T>::value || is_trial_solution_op<T>::value ||
       is_field_solution_op<T>::value || is_cell_geometry_op<T>::value ||
       is_functor_op<T>::value || is_cache_functor_op<T>::value) &&
      !(is_unary_op<T>::value || is_binary_op<T>::value)>::type>
    : std::true_type
  {};

  template <typename T>
  struct is_compatible_with_tensor_arithmetic<
    T,
    typename std::enable_if<is_symbolic_integral_op<T>::value>::type>
    : std::false_type
  {};

  template <typename T>
  struct is_compatible_with_tensor_arithmetic<
    T,
    typename std::enable_if<
      is_unary_op<T>::value && !is_unary_integral_op<T>::value &&
      is_compatible_with_tensor_arithmetic<typename T::OpType>::value>::type>
    : std::true_type
  {};

  template <typename T>
  struct is_compatible_with_tensor_arithmetic<
    T,
    typename std::enable_if<
      is_binary_op<T>::value && !is_binary_integral_op<T>::value &&
      is_compatible_with_tensor_arithmetic<typename T::LhsOpType>::value &&
      is_compatible_with_tensor_arithmetic<typename T::RhsOpType>::value>::type>
    : std::true_type
  {};

} // namespace WeakForms


#endif // DOXYGEN


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_type_traits_h
