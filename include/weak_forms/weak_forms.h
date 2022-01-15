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

#ifndef dealii_weakforms_weakforms_h
#define dealii_weakforms_weakforms_h

#include <deal.II/base/config.h>

#include <weak_forms/config.h>

// Grouped by function:

// Utilities
// #include <weak_forms/operators.h> // ? TODO: Remove
#include <weak_forms/numbers.h>
#include <weak_forms/symbolic_decorations.h>
#include <weak_forms/type_traits.h>
#include <weak_forms/types.h>

// Functors and spaces to be used inside of weak forms
#include <weak_forms/cache_functors.h>
#include <weak_forms/functors.h>
#include <weak_forms/spaces.h>

// Subspaces
#include <weak_forms/subspace_extractors.h>
#include <weak_forms/subspace_views.h>

// Operators that operate and give values to functors and spaces
#include <weak_forms/binary_operators.h>
#include <weak_forms/cell_face_subface_operators.h>
#include <weak_forms/mixed_operators.h>
#include <weak_forms/symbolic_operators.h>
#include <weak_forms/unary_operators.h>

// The actual forms themselves
#include <weak_forms/bilinear_forms.h>
#include <weak_forms/linear_forms.h>
#include <weak_forms/mixed_form_operators.h>

// Self-linearizing functors and forms
// (using template meta-programming in conjunction with AD/SD)
#include <weak_forms/ad_sd_functor_cache.h>
#include <weak_forms/energy_functor.h>
#include <weak_forms/residual_functor.h>
#include <weak_forms/self_linearizing_forms.h>

// Common tools for assembly
#include <weak_forms/integrator.h>
#include <weak_forms/solution_extraction_data.h>
#include <weak_forms/solution_storage.h>
#include <weak_forms/symbolic_integral.h>
// Operate on symbolic integrals
#include <weak_forms/binary_integral_operators.h>
#include <weak_forms/mixed_integral_operators.h>
#include <weak_forms/unary_integral_operators.h>

// Assembly
#include <weak_forms/assembler_matrix_based.h>
// #include <weak_forms/assembler_matrix_free.h>


#endif // dealii_weakforms_weakforms_h
