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

#ifndef dealii_weakforms_solution_vectors_h
#define dealii_weakforms_solution_vectors_h

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>

#include <deal.II/meshworker/scratch_data.h>

#include <weak_forms/config.h>
#include <weak_forms/types.h>

#include <string>


WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
  template <typename VectorType>
  class SolutionStorage
  {
  public:
    using ptr_type = const VectorType *;

    // Store nothing
    explicit SolutionStorage()
    {}

    explicit SolutionStorage(const std::vector<ptr_type> &   solution_vectors,
                             const std::vector<std::string> &solution_names)
      : solution_names(solution_names)
      , solution_vectors(solution_vectors)
    {}

    explicit SolutionStorage(const VectorType &solution_vector,
                             const std::string name = "solution")
      : SolutionStorage({&solution_vector}, create_name_vector(name, 1))
    {}

    explicit SolutionStorage(const std::vector<ptr_type> &solution_vectors,
                             const std::string            name = "solution")
      : SolutionStorage(solution_vectors,
                        create_name_vector(name, solution_vectors.size()))
    {}

    SolutionStorage(const SolutionStorage &) = default;
    SolutionStorage(SolutionStorage &&)      = default;
    ~SolutionStorage()                       = default;

    std::size_t
    n_solution_vectors() const
    {
      Assert(solution_names.size() == solution_vectors.size(),
             ExcDimensionMismatch(solution_names.size(),
                                  solution_vectors.size()));
      return solution_vectors.size();
    }

    const std::vector<std::string> &
    get_solution_names() const
    {
      return solution_names;
    }

    template <int dim, int spacedim>
    void
    extract_local_dof_values(
      MeshWorker::ScratchData<dim, spacedim> &scratch_data) const
    {
      for (unsigned int t = 0; t < n_solution_vectors(); ++t)
        {
          scratch_data.extract_local_dof_values(get_solution_name(t),
                                                get_solution_vector(t));
        }
    }

  private:
    // Recommended order (0 = index default):
    // - 0: Current solution
    // - 1: Previous solution
    // - 2: Previous-previous solution
    // - ...
    // OR
    // - 0: Current solution
    // - 1: Solution first time derivative
    // - 2: Solution second time derivative
    // - ...
    const std::vector<std::string> solution_names;
    const std::vector<ptr_type>    solution_vectors;

    const std::string &
    get_solution_name(const types::solution_index index) const
    {
      Assert(index < solution_names.size(),
             ExcIndexRange(index, 0, solution_names.size()));
      return solution_names[index];
    }

    // TEMP: Move to private section?
    const VectorType &
    get_solution_vector(const types::solution_index index) const
    {
      Assert(index < solution_vectors.size(),
             ExcIndexRange(index, 0, solution_vectors.size()));
      Assert(solution_vectors[index], ExcNotInitialized());
      return *(solution_vectors[index]);
    }

    static std::vector<std::string>
    create_name_vector(const std::string &name, const unsigned int n_entries)
    {
      std::vector<std::string> out;
      out.reserve(n_entries);

      for (unsigned int index = 0; index < n_entries; ++index)
        {
          if (index == 0)
            out.push_back(name);
          else
            out.push_back(name + "_t" + dealii::Utilities::to_string(index));
        }

      return out;
    }
  }; // class SolutionStorage
} // namespace WeakForms


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_solution_vectors_h
