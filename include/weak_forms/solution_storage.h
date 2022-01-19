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

#ifndef dealii_weakforms_solution_storage_h
#define dealii_weakforms_solution_storage_h

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>

#include <deal.II/meshworker/scratch_data.h>

#include <weak_forms/config.h>
#include <weak_forms/solution_extraction_data.h>
#include <weak_forms/types.h>
#include <weak_forms/utilities.h>

#include <initializer_list>
#include <string>


WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
  template <typename VectorType, typename DoFHandlerType = std::nullptr_t>
  class SolutionStorage
  {
  public:
    using solution_ptr_type   = const VectorType *;
    using dofhandler_ptr_type = const DoFHandlerType *;

    // Store nothing
    explicit SolutionStorage()
    {}

    explicit SolutionStorage(
      const std::vector<solution_ptr_type> &  solution_vectors,
      const std::vector<dofhandler_ptr_type> &dof_handlers,
      const std::vector<std::string> &        solution_names)
      : solution_names(solution_names)
      , solution_vectors(solution_vectors)
      , dof_handlers(dof_handlers)
    {
      AssertThrow((!std::is_same<VectorType, std::nullptr_t>::value),
                  ExcMessage(
                    "Cannot create a solution storage for vectors when the "
                    "vector has no type. Use the default constructor instead, "
                    "or specify the type using the template parameter."));
    }

    explicit SolutionStorage(
      const std::vector<solution_ptr_type> &  solution_vectors,
      const std::vector<dofhandler_ptr_type> &dof_handlers,
      const std::string                       name = "solution")
      : SolutionStorage(solution_vectors,
                        dof_handlers,
                        create_name_vector(name, solution_vectors.size()))
    {}

    template <typename DH = DoFHandlerType,
              typename    = typename std::enable_if<
                std::is_same<DH, std::nullptr_t>::value>::type>
    explicit SolutionStorage(
      const std::vector<solution_ptr_type> &solution_vectors,
      const std::vector<std::string> &      solution_names)
      : SolutionStorage(solution_vectors,
                        std::vector<dofhandler_ptr_type>(solution_names.size(),
                                                         nullptr),
                        solution_names)
    {}

    template <typename DH = DoFHandlerType,
              typename    = typename std::enable_if<
                std::is_same<DH, std::nullptr_t>::value>::type>
    explicit SolutionStorage(
      const std::vector<solution_ptr_type> &solution_vectors,
      const std::string                     name = "solution")
      : SolutionStorage(solution_vectors,
                        create_name_vector(name, solution_vectors.size()))
    {}

    template <typename DH = DoFHandlerType,
              typename    = typename std::enable_if<
                std::is_same<DH, std::nullptr_t>::value>::type>
    explicit SolutionStorage(
      const std::initializer_list<solution_ptr_type> &solution_vectors,
      const std::string                               name = "solution")
      : SolutionStorage(std::vector<solution_ptr_type>(solution_vectors), name)
    {}

    template <typename DH = DoFHandlerType,
              typename    = typename std::enable_if<
                std::is_same<DH, std::nullptr_t>::value>::type>
    explicit SolutionStorage(const VectorType &solution_vector,
                             const std::string name = "solution")
      : SolutionStorage({&solution_vector}, create_name_vector(name, 1))
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

    // This function permits us to have field solutions that are associated
    // with another DoFHandler. Each ScratchData is bound to a DoFHandler
    // through the cell that it is reinit'd with.

    template <int dim,
              int spacedim,
              template <int, int>
              class DoFHandlerType2,
              typename DH = DoFHandlerType>
    typename std::enable_if<!std::is_same<DH, std::nullptr_t>::value>::type
    initialize(MeshWorker::ScratchData<dim, spacedim> &scratch_data,
               const FEValues<dim, spacedim> &         fe_values,
               const DoFHandlerType2<dim, spacedim> &  dof_handler) const
    {
      static_assert(
        std::is_same<DoFHandlerType, DoFHandlerType2<dim, spacedim>>::value,
        "Solution DoFHandler type and used DoFHandler type must be the same.");

      initialize(scratch_data, dof_handler);

      for (unsigned int t = 0; t < n_solution_vectors(); ++t)
        {
          if (uses_same_dof_handler(t, dof_handler) == false)
            {
              // For this cell, initialise scratch data object that we can use
              // to extract data associated with an external DoFHandler object
              // whenever we need to.
              GeneralDataStorage &cache =
                scratch_data.get_general_data_storage();
              MeshWorker::ScratchData<dim, spacedim> &external_scratch_data =
                cache.template get_object_with_name<
                  MeshWorker::ScratchData<dim, spacedim>>(
                  get_name_solution_scratch(t));

              const DoFHandlerType &external_dof_handler = get_dof_handler(t);
              const Triangulation<dim, spacedim> &triangulation =
                external_dof_handler.get_triangulation();
              Assert(
                &triangulation == &dof_handler.get_triangulation(),
                ExcMessage(
                  "DoFHandlers are associated with different triangulations."));

              const typename Triangulation<dim, spacedim>::cell_iterator cell =
                fe_values.get_cell();
              const typename DoFHandlerType::active_cell_iterator external_cell(
                &triangulation,
                cell->level(),
                cell->index(),
                &external_dof_handler);

              external_scratch_data.reinit(external_cell);
            }
        }
    }

    template <int dim,
              int spacedim,
              template <int, int>
              class DoFHandlerType2,
              typename DH = DoFHandlerType>
    typename std::enable_if<std::is_same<DH, std::nullptr_t>::value>::type
    initialize(MeshWorker::ScratchData<dim, spacedim> &scratch_data,
               const FEValues<dim, spacedim> &         fe_values,
               const DoFHandlerType2<dim, spacedim> &  dof_handler) const
    {
      (void)scratch_data;
      (void)fe_values;
      (void)dof_handler;
    }

    template <int dim,
              int spacedim,
              template <int, int>
              class DoFHandlerType2,
              typename DH = DoFHandlerType>
    typename std::enable_if<!std::is_same<DH, std::nullptr_t>::value>::type
    initialize(MeshWorker::ScratchData<dim, spacedim> &scratch_data,
               const FEValues<dim, spacedim> &         fe_values,
               const FEFaceValues<dim, spacedim> &     fe_face_values,
               const DoFHandlerType2<dim, spacedim> &  dof_handler) const
    {
      static_assert(
        std::is_same<DoFHandlerType, DoFHandlerType2<dim, spacedim>>::value,
        "Solution DoFHandler type and used DoFHandler type must be the same.");

      initialize(scratch_data, dof_handler);

      for (unsigned int t = 0; t < n_solution_vectors(); ++t)
        {
          if (uses_same_dof_handler(t, dof_handler) == false)
            {
              // For this cell and face, initialise scratch data object that we
              // can use to extract data associated with an external DoFHandler
              // object whenever we need to.
              GeneralDataStorage &cache =
                scratch_data.get_general_data_storage();
              MeshWorker::ScratchData<dim, spacedim> &external_scratch_data =
                cache.template get_object_with_name<
                  MeshWorker::ScratchData<dim, spacedim>>(
                  get_name_solution_scratch(t));

              const DoFHandlerType &external_dof_handler = get_dof_handler(t);
              const Triangulation<dim, spacedim> &triangulation =
                external_dof_handler.get_triangulation();
              Assert(
                &triangulation == &dof_handler.get_triangulation(),
                ExcMessage(
                  "DoFHandlers are associated with different triangulations."));

              const typename Triangulation<dim, spacedim>::cell_iterator cell =
                fe_values.get_cell();
              const typename DoFHandlerType::active_cell_iterator external_cell(
                &triangulation,
                cell->level(),
                cell->index(),
                &external_dof_handler);

              external_scratch_data.reinit(external_cell);
              external_scratch_data.reinit(external_cell,
                                           fe_face_values.get_face_number());
            }
        }
    }

    template <int dim,
              int spacedim,
              template <int, int>
              class DoFHandlerType2,
              typename DH = DoFHandlerType>
    typename std::enable_if<std::is_same<DH, std::nullptr_t>::value>::type
    initialize(MeshWorker::ScratchData<dim, spacedim> &scratch_data,
               const FEValues<dim, spacedim> &         fe_values,
               const FEFaceValues<dim, spacedim> &     fe_face_values,
               const DoFHandlerType2<dim, spacedim> &  dof_handler) const
    {
      (void)scratch_data;
      (void)fe_values;
      (void)fe_face_values;
      (void)dof_handler;
    }

    template <int dim,
              int spacedim,
              template <int, int>
              class DoFHandlerType2,
              typename DH = DoFHandlerType>
    typename std::enable_if<!std::is_same<DH, std::nullptr_t>::value>::type
    initialize(MeshWorker::ScratchData<dim, spacedim> &scratch_data,
               const FEInterfaceValues<dim, spacedim> &fe_interface_values,
               const DoFHandlerType2<dim, spacedim> &  dof_handler) const
    {
      static_assert(
        std::is_same<DoFHandlerType, DoFHandlerType2<dim, spacedim>>::value,
        "Solution DoFHandler type and used DoFHandler type must be the same.");

      initialize(scratch_data, dof_handler);

      for (unsigned int t = 0; t < n_solution_vectors(); ++t)
        {
          if (uses_same_dof_handler(t, dof_handler) == false)
            {
              // For this interface, initialise scratch data object that we can
              // use to extract data associated with an external DoFHandler
              // object whenever we need to.
              GeneralDataStorage &cache =
                scratch_data.get_general_data_storage();
              MeshWorker::ScratchData<dim, spacedim> &external_scratch_data =
                cache.template get_object_with_name<
                  MeshWorker::ScratchData<dim, spacedim>>(
                  get_name_solution_scratch(t));

              const DoFHandlerType &external_dof_handler = get_dof_handler(t);
              const Triangulation<dim, spacedim> &triangulation =
                external_dof_handler.get_triangulation();
              Assert(
                &triangulation == &dof_handler.get_triangulation(),
                ExcMessage(
                  "DoFHandlers are associated with different triangulations."));

              const typename Triangulation<dim, spacedim>::cell_iterator cell =
                fe_interface_values.get_cell(0);
              const typename Triangulation<dim, spacedim>::cell_iterator
                neighbour_cell = fe_interface_values.get_cell(1);

              const typename DoFHandlerType::active_cell_iterator external_cell(
                &triangulation,
                cell->level(),
                cell->index(),
                &external_dof_handler);
              const typename DoFHandlerType::active_cell_iterator
                external_neighbour_cell(&triangulation,
                                        neighbour_cell->level(),
                                        neighbour_cell->index(),
                                        &dof_handler);

              const unsigned int face =
                fe_interface_values.get_fe_face_values(0).get_face_number();
              const unsigned int neighbour_face =
                fe_interface_values.get_fe_face_values(1).get_face_number();
              const unsigned int subface =
                fe_interface_values.get_fe_face_values(0).get_face_index();
              const unsigned int neighbour_subface =
                fe_interface_values.get_fe_face_values(1).get_face_index();

              external_scratch_data.reinit(external_cell,
                                           face,
                                           subface,
                                           external_neighbour_cell,
                                           neighbour_face,
                                           neighbour_subface);
            }
        }
    }

    template <int dim,
              int spacedim,
              template <int, int>
              class DoFHandlerType2,
              typename DH = DoFHandlerType>
    typename std::enable_if<std::is_same<DH, std::nullptr_t>::value>::type
    initialize(MeshWorker::ScratchData<dim, spacedim> &scratch_data,
               const FEInterfaceValues<dim, spacedim> &fe_interface_values,
               const DoFHandlerType2<dim, spacedim> &  dof_handler) const
    {
      (void)scratch_data;
      (void)fe_interface_values;
      (void)dof_handler;
    }

    template <int dim,
              int spacedim,
              template <int, int>
              class DoFHandlerType2,
              typename DH = DoFHandlerType>
    typename std::enable_if<!std::is_same<DH, std::nullptr_t>::value>::type
    extract_local_dof_values(
      MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      const DoFHandlerType2<dim, spacedim> &  dof_handler) const
    {
      std::vector<SolutionExtractionData<dim, spacedim>>
        solution_extraction_data =
          get_solution_extraction_data(scratch_data, dof_handler);
      Assert(solution_extraction_data.size() == n_solution_vectors(),
             ExcDimensionMismatch(solution_extraction_data.size(),
                                  n_solution_vectors()));

      for (unsigned int t = 0; t < n_solution_vectors(); ++t)
        {
          Assert(solution_extraction_data[t].solution_name ==
                   get_solution_name(t),
                 ExcInternalError());
          if (solution_extraction_data[t].uses_external_dofhandler == false)
            {
              Assert(&solution_extraction_data[t].get_scratch_data() ==
                       &scratch_data,
                     ExcInternalError());
            }

          MeshWorker::ScratchData<dim, spacedim> &solution_scratch_data =
            solution_extraction_data[t].get_scratch_data();
          solution_scratch_data.extract_local_dof_values(
            get_solution_name(t), get_solution_vector(t));
        }
    }

    template <int dim,
              int spacedim,
              template <int, int>
              class DoFHandlerType2,
              typename DH = DoFHandlerType>
    typename std::enable_if<std::is_same<DH, std::nullptr_t>::value>::type
    extract_local_dof_values(
      MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      const DoFHandlerType2<dim, spacedim> &  dof_handler) const
    {
      (void)dof_handler;
      for (unsigned int t = 0; t < n_solution_vectors(); ++t)
        {
          Assert(
            dof_handlers[t] == nullptr,
            ExcMessage(
              "Unexpected association of solution with an external DoFHandler."));

          scratch_data.extract_local_dof_values(get_solution_name(t),
                                                get_solution_vector(t));
        }
    }

    /**
     * Returns a vector to a scratch
     */
    template <int dim, int spacedim, template <int, int> class DoFHandlerType2>
    std::vector<SolutionExtractionData<dim, spacedim>>
    get_solution_extraction_data(
      MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      const DoFHandlerType2<dim, spacedim> &  dof_handler) const
    {
      std::vector<SolutionExtractionData<dim, spacedim>>
        solution_extraction_data;
      solution_extraction_data.reserve(n_solution_vectors());

      for (unsigned int t = 0; t < n_solution_vectors(); ++t)
        {
          if (uses_same_dof_handler(t, dof_handler))
            {
              constexpr const bool uses_external_dofhandler = false;
              solution_extraction_data.emplace_back(scratch_data,
                                                    get_solution_name(t),
                                                    uses_external_dofhandler);
            }
          else
            {
              Assert(
                (!std::is_same<DoFHandlerType, std::nullptr_t>::value),
                ExcMessage(
                  "Cannot extract from an external scratch data object when it "
                  "is not associated with another DoFHandler. You need to use the "
                  "constructor that takes in a DoFHandler in order to use this "
                  "functionality."));

              GeneralDataStorage &cache =
                scratch_data.get_general_data_storage();
              MeshWorker::ScratchData<dim, spacedim> &external_scratch_data =
                cache.template get_object_with_name<
                  MeshWorker::ScratchData<dim, spacedim>>(
                  get_name_solution_scratch(t));

              constexpr const bool uses_external_dofhandler = true;
              solution_extraction_data.emplace_back(external_scratch_data,
                                                    get_solution_name(t),
                                                    uses_external_dofhandler);
            }
        }

      return solution_extraction_data;
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
    const std::vector<std::string>       solution_names;
    const std::vector<solution_ptr_type> solution_vectors;

    /**
     * Permit each of the solution vectors to be associated with a DoFHandler
     * that is not necessarily the one used in the main assembly loop. This
     * allows us to use external finite element solutions as fields for source
     * terms and the like. If the "default" option, namely associating this
     * solution field with the primary DoFHandler, is to be made then the
     * entry can just be null-initialised.
     */
    const std::vector<dofhandler_ptr_type> dof_handlers;

    std::string
    get_name_solution_scratch(const types::solution_index index) const
    {
      return Utilities::get_deal_II_prefix() + "Solution_Storage_Scratch_" +
             std::to_string(index);
    }

    const std::string &
    get_solution_name(const types::solution_index index) const
    {
      Assert(index < solution_names.size(),
             ExcIndexRange(index, 0, solution_names.size()));
      return solution_names[index];
    }

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

    template <int dim,
              int spacedim,
              template <int, int>
              class DoFHandlerType2,
              typename DH = DoFHandlerType>
    typename std::enable_if<!std::is_same<DH, std::nullptr_t>::value>::type
    initialize(MeshWorker::ScratchData<dim, spacedim> &scratch_data,
               const DoFHandlerType2<dim, spacedim> &  dof_handler) const
    {
      static_assert(
        std::is_same<DoFHandlerType, DoFHandlerType2<dim, spacedim>>::value,
        "Solution DoFHandler type and used DoFHandler type must be the same.");

      for (unsigned int t = 0; t < n_solution_vectors(); ++t)
        {
          if (uses_same_dof_handler(t, dof_handler) == false)
            {
              // Create and store another scratch data object that we can use to
              // extract data associated with an external DoFHandler object
              // whenever we need to.
              GeneralDataStorage &cache =
                scratch_data.get_general_data_storage();

              cache.template get_or_add_object_with_name<
                MeshWorker::ScratchData<dim, spacedim>>(
                get_name_solution_scratch(t),
                scratch_data.get_mapping(),
                get_dof_handler(t).get_fe(),
                scratch_data.get_cell_quadrature(),
                scratch_data.get_cell_update_flags(),
                scratch_data.get_face_quadrature(),
                scratch_data.get_face_update_flags());
            }
          else
            {
              Assert(t < dof_handlers.size(),
                     ExcIndexRange(t, 0, dof_handlers.size()));
              Assert(
                dof_handlers[t] == nullptr ||
                  &get_dof_handler(t) == &dof_handler,
                ExcMessage(
                  "Unexpected association of solution with an external DoFHandler."));
            }
        }
    }

    template <int dim,
              int spacedim,
              template <int, int>
              class DoFHandlerType2,
              typename DH = DoFHandlerType>
    typename std::enable_if<std::is_same<DH, std::nullptr_t>::value>::type
    initialize(MeshWorker::ScratchData<dim, spacedim> &scratch_data,
               const DoFHandlerType2<dim, spacedim> &  dof_handler) const
    {
      // Nothing to do: All solution vectors should be associated with the
      // primary DoFHandler that is used in the assembly loop

      (void)dof_handler;
      (void)scratch_data;
    }

    template <int dim,
              int spacedim,
              template <int, int>
              class DoFHandlerType2,
              typename DH = DoFHandlerType>
    typename std::enable_if<!std::is_same<DH, std::nullptr_t>::value,
                            bool>::type
    uses_same_dof_handler(
      const types::solution_index           index,
      const DoFHandlerType2<dim, spacedim> &dof_handler) const
    {
      Assert(index < dof_handlers.size(),
             ExcIndexRange(index, 0, dof_handlers.size()));

      // No checks if null-initialised.
      // In this case, we assume that the input DoFHandler is the
      // one that we want to use.
      if (!dof_handlers[index])
        return true;

      return dof_handlers[index] == &dof_handler;
    }

    template <int dim,
              int spacedim,
              template <int, int>
              class DoFHandlerType2,
              typename DH = DoFHandlerType>
    typename std::enable_if<std::is_same<DH, std::nullptr_t>::value, bool>::type
    uses_same_dof_handler(
      const types::solution_index           index,
      const DoFHandlerType2<dim, spacedim> &dof_handler) const
    {
      (void)index;
      (void)dof_handler;

      return true;
    }

    const DoFHandlerType &
    get_dof_handler(const types::solution_index index) const
    {
      Assert(index < dof_handlers.size(),
             ExcIndexRange(index, 0, dof_handlers.size()));
      Assert(dof_handlers[index], ExcNotInitialized());

      return *(dof_handlers[index]);
    }

  }; // class SolutionStorage


  namespace internal
  {
    // Utility functions to help with template arguments of the
    // assemble_system() method being void / std::null_ptr_t.

    template <typename FEValuesType,
              typename DoFHandlerType,
              typename VectorType,
              typename SSDType,
              int dim,
              int spacedim>
    typename std::enable_if<std::is_same<typename std::decay<VectorType>::type,
                                         std::nullptr_t>::value>::type
    initialize(MeshWorker::ScratchData<dim, spacedim> &    scratch_data,
               const FEValuesType &                        fe_values,
               const DoFHandlerType &                      dof_handler,
               const SolutionStorage<VectorType, SSDType> &solution_storage)
    {
      static_assert(
        std::is_same<typename std::decay<SSDType>::type, std::nullptr_t>::value,
        "Expected DoFHandler type for solution storage to be null type.");
      (void)scratch_data;
      (void)fe_values;
      (void)dof_handler;
      (void)solution_storage;

      // Void pointer; do nothing.
      AssertThrow(false, ExcUnexpectedFunctionCall());
    }

    // Valid for both FEValues and FEInterfaceValues
    template <typename FEValuesType,
              typename DoFHandlerType,
              typename VectorType,
              typename SSDType,
              int dim,
              int spacedim>
    typename std::enable_if<!std::is_same<typename std::decay<VectorType>::type,
                                          std::nullptr_t>::value>::type
    initialize(MeshWorker::ScratchData<dim, spacedim> &    scratch_data,
               const FEValuesType &                        fe_values,
               const DoFHandlerType &                      dof_handler,
               const SolutionStorage<VectorType, SSDType> &solution_storage)
    {
      solution_storage.initialize(scratch_data, fe_values, dof_handler);
    }

    template <typename FEValuesType,
              typename FEFaceValuesType,
              typename DoFHandlerType,
              typename VectorType,
              typename SSDType,
              int dim,
              int spacedim>
    typename std::enable_if<std::is_same<typename std::decay<VectorType>::type,
                                         std::nullptr_t>::value>::type
    initialize(MeshWorker::ScratchData<dim, spacedim> &    scratch_data,
               const FEValuesType &                        fe_values,
               const FEFaceValuesType &                    fe_face_values,
               const DoFHandlerType &                      dof_handler,
               const SolutionStorage<VectorType, SSDType> &solution_storage)
    {
      static_assert(
        std::is_same<typename std::decay<SSDType>::type, std::nullptr_t>::value,
        "Expected DoFHandler type for solution storage to be null type.");
      (void)scratch_data;
      (void)fe_values;
      (void)fe_face_values;
      (void)dof_handler;
      (void)solution_storage;

      // Void pointer; do nothing.
      AssertThrow(false, ExcUnexpectedFunctionCall());
    }

    template <typename FEValuesType,
              typename FEFaceValuesType,
              typename DoFHandlerType,
              typename VectorType,
              typename SSDType,
              int dim,
              int spacedim>
    typename std::enable_if<!std::is_same<typename std::decay<VectorType>::type,
                                          std::nullptr_t>::value>::type
    initialize(MeshWorker::ScratchData<dim, spacedim> &    scratch_data,
               const FEValuesType &                        fe_values,
               const FEFaceValuesType &                    fe_face_values,
               const DoFHandlerType &                      dof_handler,
               const SolutionStorage<VectorType, SSDType> &solution_storage)
    {
      solution_storage.initialize(scratch_data,
                                  fe_values,
                                  fe_face_values,
                                  dof_handler);
    }

    template <typename DoFHandlerType,
              typename VectorType,
              typename SSDType,
              int dim,
              int spacedim>
    typename std::enable_if<std::is_same<typename std::decay<VectorType>::type,
                                         std::nullptr_t>::value>::type
    extract_solution_local_dof_values(
      MeshWorker::ScratchData<dim, spacedim> &    scratch_data,
      const DoFHandlerType &                      dof_handler,
      const SolutionStorage<VectorType, SSDType> &solution_storage)
    {
      static_assert(
        std::is_same<typename std::decay<SSDType>::type, std::nullptr_t>::value,
        "Expected DoFHandler type for solution storage to be null type.");
      (void)scratch_data;
      (void)dof_handler;
      (void)solution_storage;

      // Void pointer; do nothing.
      AssertThrow(false, ExcUnexpectedFunctionCall());
    }

    template <typename DoFHandlerType,
              typename VectorType,
              typename SSDType,
              int dim,
              int spacedim>
    typename std::enable_if<!std::is_same<typename std::decay<VectorType>::type,
                                          std::nullptr_t>::value>::type
    extract_solution_local_dof_values(
      MeshWorker::ScratchData<dim, spacedim> &    scratch_data,
      const DoFHandlerType &                      dof_handler,
      const SolutionStorage<VectorType, SSDType> &solution_storage)
    {
      solution_storage.extract_local_dof_values(scratch_data, dof_handler);
    }
  } // namespace internal
} // namespace WeakForms


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_solution_storage_h
