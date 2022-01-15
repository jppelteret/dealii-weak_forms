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
    using field_index = unsigned int;

    using solution_index = unsigned int;

    using vectorized_qp_range_t =
      std_cxx20::ranges::iota_view<unsigned int, unsigned int>;
  } // namespace types
} // namespace WeakForms

WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_types_h