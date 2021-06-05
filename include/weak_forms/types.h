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

#ifndef dealii_weakforms_types_h
#define dealii_weakforms_types_h

#include <weak_forms/config.h>


WEAK_FORMS_NAMESPACE_OPEN

namespace WeakForms
{
  namespace types
  {
    using field_index = unsigned int;
  }


  namespace numbers
  {
    const types::field_index invalid_field_index =
      static_cast<types::field_index>(-1);
  }

} // namespace WeakForms

WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_types_h