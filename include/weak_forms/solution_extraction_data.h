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

#ifndef dealii_weakforms_solution_extraction_data_h
#define dealii_weakforms_solution_extraction_data_h

#include <deal.II/base/config.h>

#include <deal.II/meshworker/scratch_data.h>

#include <weak_forms/config.h>

#include <string>


WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
  template <int dim, int spacedim = dim>
  struct SolutionExtractionData
  {
    SolutionExtractionData(MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                           const std::string &solution_name,
                           const bool         uses_external_dofhandler)
      : solution_name(solution_name)
      , uses_external_dofhandler(uses_external_dofhandler)
      , scratch_data(&scratch_data)
    {}

    MeshWorker::ScratchData<dim, spacedim> &
    get_scratch_data()
    {
      return *scratch_data;
    }

    MeshWorker::ScratchData<dim, spacedim> &
    get_scratch_data() const
    {
      return *scratch_data;
    }

    const std::string solution_name;
    const bool        uses_external_dofhandler;

  private:
    MeshWorker::ScratchData<dim, spacedim> *const scratch_data;
  }; // struct SolutionExtractionData
} // namespace WeakForms


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_solution_extraction_data_h
