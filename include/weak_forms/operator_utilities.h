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

#ifndef dealii_weakforms_operator_utilities_h
#define dealii_weakforms_operator_utilities_h

#include <deal.II/base/config.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_values.h>

#include <weak_forms/config.h>



WEAK_FORMS_NAMESPACE_OPEN



namespace WeakForms
{
  namespace Operators
  {
    namespace internal
    {
      template <typename T, typename U = void>
      struct is_fe_values_type : std::false_type
      {};


      template <template <int, int> class FEValuesType, int dim, int spacedim>
      struct is_fe_values_type<FEValuesType<dim, spacedim>,
                               typename std::enable_if<std::is_base_of<
                                 FEValuesBase<dim, spacedim>,
                                 FEValuesType<dim, spacedim>>::value>::type>
        : std::true_type
      {};


      template <template <int, int> class FEValuesType, int dim, int spacedim>
      struct is_fe_values_type<
        FEValuesType<dim, spacedim>,
        typename std::enable_if<
          std::is_same<FEValuesType<dim, spacedim>,
                       FEInterfaceValues<dim, spacedim>>::value>::type>
        : std::true_type
      {};

    } // namespace internal
  }   // namespace Operators
} // namespace WeakForms



WEAK_FORMS_NAMESPACE_CLOSE


#endif // dealii_weakforms_operator_utilities_h