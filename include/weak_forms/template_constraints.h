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

#ifndef dealii_weakforms_template_constraints_h
#define dealii_weakforms_template_constraints_h

#include <deal.II/base/config.h>

#include <deal.II/base/template_constraints.h>

#include <weak_forms/config.h>

#include <type_traits>


WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
  /**
   * @brief A type trait that checks to see if a class is a scalar type.
   *
   * It is, essentially, a check to see if type @p T is a floating point or
   * integer type,  a complex number, or a vectorised number. This corresponds
   * closely with the check performed by
   * <a
   * href="https://www.dealii.org/current/doxygen/deal.II/structEnableIfScalar.html">`dealii::EnableIfScalar`</a>.
   *
   * @tparam T The class type to be tested for the trait.
   */
  template <typename T>
  class is_scalar_type
  {
    // See has_begin_and_end() in template_constraints.h
    // and https://stackoverflow.com/a/10722840

    template <typename A>
    static constexpr auto
    test(int) -> decltype(
      std::declval<
        typename EnableIfScalar<typename std::decay<A>::type>::type>(),
      std::true_type())
    {
      return true;
    }

    template <typename A>
    static std::false_type
    test(...);

  public:
    using type = decltype(test<T>(0));

    static const bool value = type::value;
  };



  /**
   * @brief A type trait that checks to see two classes are both scalar types.
   *
   * @tparam T A class type to be tested for the trait.
   * @tparam U A class type to be tested for the trait.
   */
  template <typename T, typename U, typename = void>
  struct are_scalar_types : std::false_type
  {};



#ifndef DOXYGEN

  template <typename T, typename U>
  struct are_scalar_types<
    T,
    U,
    typename std::enable_if<is_scalar_type<T>::value &&
                            is_scalar_type<U>::value>::type> : std::true_type
  {};

#endif // DOXYGEN

} // namespace WeakForms


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_template_constraints_h