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

#ifndef dealii_weakforms_tests_utilities_h
#define dealii_weakforms_tests_utilities_h


#include <deal.II/base/exceptions.h>

#include <regex>
#include <string>
#include <vector>


DeclException4(ExcMatrixEntriesNotEqual,
               int,
               int,
               double,
               double,
               << "Matrix entries are different (exemplar). "
               << "(R,C) = (" << arg1 << "," << arg2 << "). "
               << "Blessed value: " << arg3 << "; "
               << "Other value: " << arg4 << ".");

DeclException2(ExcIteratorRowIndexNotEqual,
               int,
               int,
               << "Iterator row index mismatch. "
               << "  Iterator 1: " << arg1 << "  Iterator 2: " << arg2);

DeclException2(ExcIteratorColumnIndexNotEqual,
               int,
               int,
               << "Iterator column index mismatch. "
               << "  Iterator 1: " << arg1 << "  Iterator 2: " << arg2);


DeclException3(ExcVectorEntriesNotEqual,
               int,
               double,
               double,
               << "Vector entries are different (exemplar). "
               << "(R) = (" << arg1 << "). "
               << "Blessed value: " << arg2 << "; "
               << "Other value: " << arg3 << ".");


std::string
strip_off_namespace(std::string demangled_type)
{
  const std::vector<std::string> names{
    "dealii::WeakForms::Operators::", "dealii::WeakForms::", "dealii::"};

  for (const auto &name : names)
    demangled_type = std::regex_replace(demangled_type, std::regex(name), "");

  return demangled_type;
}

#endif // dealii_weakforms_tests_utilities_h
