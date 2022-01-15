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

#ifndef _weak_forms_config_h
#define _weak_forms_config_h

#include <deal.II/base/config.h>

#define WEAK_FORMS_NAMESPACE_NAME dealiiWeakForms

#define WEAK_FORMS_NAMESPACE_OPEN     \
  namespace WEAK_FORMS_NAMESPACE_NAME \
  {
#define WEAK_FORMS_NAMESPACE_CLOSE } // namespace dealiiWeakForms


// Unconditionally import the deal.II namespace
WEAK_FORMS_NAMESPACE_OPEN
using namespace dealii;
WEAK_FORMS_NAMESPACE_CLOSE

#endif // _weak_forms_config_h