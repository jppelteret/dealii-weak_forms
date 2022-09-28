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

#ifndef dealii_weakforms_types_h
#define dealii_weakforms_types_h

#include <deal.II/base/std_cxx20/iota_view.h>

#include <weak_forms/config.h>


WEAK_FORMS_NAMESPACE_OPEN

namespace WeakForms
{
  namespace types
  {
    /**
     * @brief A type that represents the starting index of a finite element field.
     *
     * This makes the most sense in the context problems involving multiple
     * solution fields captured within the same finite element system.
     */
    using field_index = unsigned int;


    /**
     * @brief A type that represents a user-defined index associated with a solution vector.
     *
     * A field solution operation becomes associated with a solution vector
     * through this index. In the context of time-dependent problems, the
     * @p solution_index could be used to reference multiple solution histories
     * such that the rate of change of the solution can be reconstructed.
     */
    using solution_index = unsigned int;


    /**
     * @brief A type that holds the quadrature point indices upon which some vectorized operation must be performed.
     *
     * It is, effectively, a mask for the indices of interest when an operator
     * or fields is to be evaluated using some mechanism that is compatible with
     * the SIMD parallel processing paradigm.
     */
    using vectorized_qp_range_t =
      std_cxx20::ranges::iota_view<unsigned int, unsigned int>;
  } // namespace types
} // namespace WeakForms

WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_types_h