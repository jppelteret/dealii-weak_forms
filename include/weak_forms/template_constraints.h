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

    // TODO: This is replicated in energy_functor.h
    template <typename T>
    class is_scalar_type
    {
      // See has_begin_and_end() in template_constraints.h
      // and https://stackoverflow.com/a/10722840

      template <typename A>
      static constexpr auto
      test(int) -> decltype(std::declval<typename EnableIfScalar<typename std::decay<A>::type>::type>(),
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


    template <typename T, typename U, typename = void>
    struct are_scalar_types : std::false_type
    {};


    template <typename T, typename U>
    struct are_scalar_types<
      T,
      U,
      typename std::enable_if<is_scalar_type<T>::value &&
                              is_scalar_type<U>::value>::type> : std::true_type
    {};

} // namespace WeakForms


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_template_constraints_h