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

#ifndef dealii_weakforms_integrator_h
#define dealii_weakforms_integrator_h

#include <deal.II/base/config.h>

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>

#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/meshworker/copy_data.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/meshworker/scratch_data.h>

#include <weak_forms/config.h>
#include <weak_forms/template_constraints.h>

#include <functional>


WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
  template <int spacedim, typename Type, typename U = void>
  class Integrator;

  // Integrator for user-defined functions
  template <int spacedim, typename ReturnType>
  class Integrator<
    spacedim,
    ReturnType,
    typename std::enable_if<is_scalar_type<ReturnType>::value>::type>
  {
    /**
     * Type definition for functions that are independent of position.
     *
     * @note We switch the type definition based on whether or not the function
     * returns some values .
     */
    using IntegrandPositionIndependent =
      std::function<void(const unsigned int       n_q_points,
                         std::vector<ReturnType> &values)>;

    using IntegrandPositionDependent =
      std::function<void(const std::vector<Point<spacedim>> &quadrature_points,
                         std::vector<ReturnType> &           values)>;

  public:
    /**
     * Construct a new Integral object
     *
     * @param integrand
     */
    Integrator(const IntegrandPositionIndependent &integrand,
               const MPI_Comm *const               mpi_communicator = nullptr)
      : integrand_position_independent(integrand)
      , integrand_position_dependent(nullptr)
      , mpi_communicator(mpi_communicator)
    {}

    /**
     * Construct a new Integral object
     *
     * @param integrand
     */
    Integrator(const IntegrandPositionDependent &integrand,
               const MPI_Comm *const             mpi_communicator = nullptr)
      : integrand_position_independent(nullptr)
      , integrand_position_dependent(integrand)
      , mpi_communicator(mpi_communicator)
    {}

    /**
     * Construct a new Integral object
     *
     * @param function
     */
    Integrator(const Function<spacedim, ReturnType> &function,
               const MPI_Comm *const                 mpi_communicator = nullptr)
      : integrand_position_independent(nullptr)
      , integrand_position_dependent(
          [&function](const std::vector<Point<spacedim>> &points,
                      std::vector<ReturnType> &           values)
          { return function.value_list(points, values); })
      , mpi_communicator(mpi_communicator)
    {}

    // SECTION: Volume integrals

    /**
     * Integrate on a volume.
     *
     * @tparam spacedim
     * @tparam DoFHandlerType
     * @param cell_quadrature
     * @param dof_handler
     * @return ReturnType
     */
    template <int dim, template <int, int> class DoFHandlerType>
    ReturnType
    dV(const Quadrature<dim> &              cell_quadrature,
       const DoFHandlerType<dim, spacedim> &dof_handler)
    {
      const auto filtered_iterator_range =
        filter_iterators(dof_handler.active_cell_iterators(),
                         IteratorFilters::LocallyOwnedCell());

      return dV(cell_quadrature,
                dof_handler,
                get_update_flags_cell(),
                filtered_iterator_range);
    }

    template <int dim, template <int, int> class DoFHandlerType>
    ReturnType
    dV(const Quadrature<dim> &                     cell_quadrature,
       const DoFHandlerType<dim, spacedim> &       dof_handler,
       const std::set<dealii::types::material_id> &subdomains)
    {
      const auto filtered_iterator_range =
        filter_iterators(dof_handler.active_cell_iterators(),
                         IteratorFilters::LocallyOwnedCell(),
                         IteratorFilters::MaterialIdEqualTo(subdomains));

      return dV(cell_quadrature,
                dof_handler,
                get_update_flags_cell(),
                filtered_iterator_range);
    }

    // SECTION: Boundary integrals

    /**
     * Integrate on a boundary.
     *
     * @tparam spacedim
     * @tparam DoFHandlerType
     * @param face_quadrature
     * @param dof_handler
     * @return ReturnType
     */
    template <int dim, template <int, int> class DoFHandlerType>
    ReturnType
    dA(const Quadrature<dim - 1> &          face_quadrature,
       const DoFHandlerType<dim, spacedim> &dof_handler)
    {
      return dA(face_quadrature,
                dof_handler,
                std::set<dealii::types::boundary_id>{});
    }

    template <int dim, template <int, int> class DoFHandlerType>
    ReturnType
    dA(const Quadrature<dim - 1> &                 face_quadrature,
       const DoFHandlerType<dim, spacedim> &       dof_handler,
       const std::set<dealii::types::boundary_id> &boundaries)
    {
      // if (boundaries.size() > 1)
      //   {
      //     Assert(boundaries.find(numbers::invalid_boundary_id) ==
      //              boundaries.end(),
      //            ExcMessage(
      //              "Cannot integrate over a subset of the boundary if "
      //              "the entire boundary has been marked for integration."));
      //   }
      const auto filtered_iterator_range =
        filter_iterators(dof_handler.active_cell_iterators(),
                         IteratorFilters::LocallyOwnedCell());

      return dA(face_quadrature,
                dof_handler,
                get_update_flags_face(),
                boundaries,
                filtered_iterator_range);
    }

  private:
    /**
     *
     *
     */
    const IntegrandPositionIndependent integrand_position_independent;

    /**
     *
     *
     */
    const IntegrandPositionDependent integrand_position_dependent;

    const MPI_Comm *const mpi_communicator;

    UpdateFlags
    get_update_flags_cell() const
    {
      UpdateFlags update_flags = update_JxW_values;
      if (integrand_position_dependent)
        update_flags |= update_quadrature_points;

      return update_flags;
    }

    UpdateFlags
    get_update_flags_face() const
    {
      UpdateFlags update_flags = update_JxW_values;
      if (integrand_position_dependent)
        update_flags |= update_quadrature_points;

      return update_flags;
    }

    template <int dim,
              template <int, int>
              class DoFHandlerType,
              typename BaseIterator>
    ReturnType
    dV(const Quadrature<dim> &              cell_quadrature,
       const DoFHandlerType<dim, spacedim> &dof_handler,
       const UpdateFlags                    update_flags_cell,
       const IteratorRange<FilteredIterator<BaseIterator>>
         filtered_iterator_range)
    {
      using ScratchData      = MeshWorker::ScratchData<dim, spacedim>;
      using CellIteratorType = decltype(dof_handler.begin_active());
      struct CopyData : MeshWorker::CopyData<0, 0, 0>
      {
        using Base = MeshWorker::CopyData<0, 0, 0>;

        CopyData()
          : Base(0)
          , cell_integral(0.0)
        {}

        ReturnType cell_integral;
      };

      ScratchData scratch(dof_handler.get_fe(),
                          cell_quadrature,
                          update_flags_cell);
      CopyData    copy;

      const auto &integrand_pd = this->integrand_position_dependent;
      const auto &integrand_pi = this->integrand_position_independent;

      // Consistency check: Only one of these may be defined for the
      // worker function to make any sense.
      if (integrand_pd)
        {
          Assert(!integrand_pi, ExcInternalError());
        }
      else if (integrand_pi)
        {
          Assert(!integrand_pd, ExcInternalError());
        }

      // A function that dictates where the locally integrated quantity
      // is accumulated into
      auto destination = [](CopyData &copy_data) -> ReturnType & {
        return copy_data.cell_integral;
      };

      // Note: CopyData is reset by mesh_loop()
      auto cell_worker = [&integrand_pd,
                          &integrand_pi,
                          &destination](const CellIteratorType &cell,
                                        ScratchData &           scratch_data,
                                        CopyData &              copy_data)
      {
        const auto &fe_values = scratch_data.reinit(cell);

        // Get values to be integrated
        std::vector<ReturnType> values(fe_values.n_quadrature_points);
        if (integrand_pi)
          integrand_pi(fe_values.n_quadrature_points, values);
        else
          integrand_pd(fe_values.get_quadrature_points(), values);

        ReturnType &cell_integral = destination(copy_data);
        for (const unsigned int q_point : fe_values.quadrature_point_indices())
          {
            cell_integral += values[q_point] * fe_values.JxW(q_point);
          }
      };

      ReturnType integral =
        dealii::internal::NumberType<ReturnType>::value(0.0);
      auto copier = [&integral](const CopyData &copy_data)
      { integral += copy_data.cell_integral; };

      MeshWorker::mesh_loop(filtered_iterator_range,
                            cell_worker,
                            copier,
                            scratch,
                            copy,
                            MeshWorker::assemble_own_cells);

      if (mpi_communicator)
        Utilities::MPI::sum(integral, *mpi_communicator);

      return integral;
    }

    template <int dim,
              template <int, int>
              class DoFHandlerType,
              typename BaseIterator>
    ReturnType
    dA(const Quadrature<dim - 1> &                 face_quadrature,
       const DoFHandlerType<dim, spacedim> &       dof_handler,
       const UpdateFlags                           update_flags_face,
       const std::set<dealii::types::boundary_id> &boundaries,
       const IteratorRange<FilteredIterator<BaseIterator>>
         filtered_iterator_range)
    {
      using ScratchData      = MeshWorker::ScratchData<dim, spacedim>;
      using CellIteratorType = decltype(dof_handler.begin_active());
      struct CopyData : MeshWorker::CopyData<0, 0, 0>
      {
        using Base = MeshWorker::CopyData<0, 0, 0>;

        CopyData()
          : Base(0)
          , face_integral(0.0)
        {}

        ReturnType face_integral;
      };

      const Quadrature<dim> cell_quadrature;
      const UpdateFlags     update_flags_cell = update_default;
      ScratchData           scratch(dof_handler.get_fe(),
                          cell_quadrature,
                          update_flags_cell,
                          face_quadrature,
                          update_flags_face);
      CopyData              copy;

      std::function<void(const CellIteratorType &, ScratchData &, CopyData &)>
        empty_cell_worker;
      // std::function<void(
      //   const decltype(cell) &, const unsigned int &, ScratchData &, CopyData
      //   &)> empty_boundary_worker;

      // Check to see if we need to filter on the boundary ID or not.
      // If there are no boundaries marked for integration, then it is
      // implied that we want to integrate on the whole boundary.
      const bool must_filter_bid = (boundaries.empty() == false);

      const auto &integrand_pd = this->integrand_position_dependent;
      const auto &integrand_pi = this->integrand_position_independent;

      // Consistency check: Only one of these may be defined for the
      // worker function to make any sense.
      if (integrand_pd)
        {
          Assert(!integrand_pi, ExcInternalError());
        }
      else if (integrand_pi)
        {
          Assert(!integrand_pd, ExcInternalError());
        }

      // A function that dictates where the locally integrated quantity
      // is accumulated into
      auto destination = [](CopyData &copy_data) -> ReturnType & {
        return copy_data.face_integral;
      };

      // Note: CopyData is reset by mesh_loop()
      auto boundary_worker = [&must_filter_bid,
                              &boundaries,
                              &integrand_pd,
                              &integrand_pi,
                              &destination](const CellIteratorType &cell,
                                            const unsigned int      face,
                                            ScratchData &scratch_data,
                                            CopyData &   copy_data)
      {
        Assert(cell->face(face)->at_boundary(), ExcInternalError());

        // Check to see if we're going to work on a boundary of interest.
        // We only do this if we know that we're wanting to integrate over
        // a subset of the boundary faces.
        if (must_filter_bid &&
            boundaries.find(cell->face(face)->boundary_id()) ==
              boundaries.end())
          {
            return;
          }

        const auto &fe_face_values = scratch_data.reinit(cell, face);

        // Get values to be integrated
        std::vector<ReturnType> values(fe_face_values.n_quadrature_points);
        if (integrand_pi)
          integrand_pi(fe_face_values.n_quadrature_points, values);
        else
          integrand_pd(fe_face_values.get_quadrature_points(), values);

        ReturnType &face_integral = destination(copy_data);
        for (const unsigned int q_point :
             fe_face_values.quadrature_point_indices())
          {
            face_integral += values[q_point] * fe_face_values.JxW(q_point);
          }
      };

      ReturnType integral =
        dealii::internal::NumberType<ReturnType>::value(0.0);
      auto copier = [&integral](const CopyData &copy_data)
      { integral += copy_data.face_integral; };

      MeshWorker::mesh_loop(filtered_iterator_range,
                            empty_cell_worker,
                            copier,
                            scratch,
                            copy,
                            MeshWorker::assemble_boundary_faces,
                            boundary_worker);

      if (mpi_communicator)
        Utilities::MPI::sum(integral, *mpi_communicator);

      return integral;
    }

    // dI
    // {
    //         std::function<void(const CellIteratorType &, ScratchData &,
    //         CopyData &)>
    //     empty_cell_worker;
    // std::function<void(
    //   const decltype(cell) &, const unsigned int &, ScratchData &, CopyData
    //   &)> empty_boundary_worker;
    // }
  };

} // namespace WeakForms


WEAK_FORMS_NAMESPACE_CLOSE


#endif // dealii_weakforms_integrator_h
