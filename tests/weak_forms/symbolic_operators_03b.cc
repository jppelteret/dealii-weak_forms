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


// Check that symbolic operators work on interfaces
// - Trial function, test function (vector-valued finite element)
// - Field solution (vector-valued finite element)

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/meshworker/scratch_data.h>

#include <deal.II/numerics/vector_tools.h>

#include <weak_forms/spaces.h>
#include <weak_forms/subspace_extractors.h>
#include <weak_forms/subspace_views.h>
#include <weak_forms/symbolic_operators.h>

#include "../weak_forms_tests.h"

template <int dim>
class DiscontinuousFunction : public Function<dim>
{
public:
  DiscontinuousFunction(const unsigned int n_components = 1)
    : Function<dim>(n_components)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    const double value = (p[0] * p[1] > 0 ? +3 : -1); // Non-trivial average
    return value * (1.0 + std::sin(p[0]) * std::cos(p[1]));
  }
};


template <int dim, int spacedim = dim, typename NumberType = double>
void
run()
{
  LogStream::Prefix prefix("Dim " + Utilities::to_string(dim));
  std::cout << "Dim: " << dim << std::endl;

  const FESystem<dim, spacedim> fe(FE_DGQ<dim, spacedim>(3), dim);
  const QGauss<spacedim>        qf_cell(fe.degree + 1);
  const QGauss<spacedim - 1>    qf_face(fe.degree + 1);

  Triangulation<dim, spacedim> triangulation;
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(1);

  DoFHandler<dim, spacedim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  Vector<double> solution(dof_handler.n_dofs());
  // VectorTools::interpolate(dof_handler,
  //                          DiscontinuousFunction<spacedim>(
  //                            fe.n_components()),
  //                          solution);
  // Don't interpolate, as points precisely on the interfaces get evaluated and
  // we don't end up with a jump in the values across it.
  AffineConstraints<double> constraints;
  constraints.close();
  VectorTools::project(dof_handler,
                       constraints,
                       QGauss<spacedim>(fe.degree + 3),
                       DiscontinuousFunction<spacedim>(fe.n_components()),
                       solution);

  const UpdateFlags update_flags_interface =
    update_values | update_gradients | update_hessians | update_3rd_derivatives;

  const auto          cell = dof_handler.begin_active();
  std::vector<double> local_dof_values(fe.dofs_per_cell);
  cell->get_dof_values(solution,
                       local_dof_values.begin(),
                       local_dof_values.end());

  MeshWorker::ScratchData<dim, spacedim> scratch_data(
    fe, qf_cell, update_default, qf_face, update_flags_interface);

  const WeakForms::SolutionStorage<Vector<double>> solution_storage(solution);
  const std::vector<std::string> &                 solution_names =
    solution_storage.get_solution_names();

  const auto test =
    [&scratch_data, &solution_names](
      const FEInterfaceValues<dim, spacedim> &fe_interface_values,
      const std::string &                     type,
      const bool                              is_on_interface)
  {
    const unsigned int dof_index =
      fe_interface_values.n_current_interface_dofs() - 1;
    const unsigned int q_point = fe_interface_values.n_quadrature_points - 1;
    std::cout << "dof_index: " << dof_index << " ; q_point: " << q_point
              << std::endl;

    {
      const std::string title = "Test function: " + type;
      std::cout << title << std::endl;
      deallog << title << std::endl;

      using namespace WeakForms;
      const TestFunction<dim, spacedim> test;
      const SubSpaceExtractors::Vector  subspace_extractor(0,
                                                          "u",
                                                          "\\mathbf{u}");

      std::cout << "Jump in values: "
                << (test[subspace_extractor].jump_in_values().template
                    operator()<NumberType>(
                      fe_interface_values))[dof_index][q_point]
                << std::endl;
      std::cout << "Jump in gradients: "
                << (test[subspace_extractor].jump_in_gradients().template
                    operator()<NumberType>(
                      fe_interface_values))[dof_index][q_point]
                << std::endl;
      std::cout << "Jump in Hessians: "
                << (test[subspace_extractor].jump_in_hessians().template
                    operator()<NumberType>(
                      fe_interface_values))[dof_index][q_point]
                << std::endl;
      std::cout << "Jump in third derivatives: "
                << (test[subspace_extractor]
                      .jump_in_third_derivatives()
                      .template operator()<NumberType>(
                        fe_interface_values))[dof_index][q_point]
                << std::endl;

      std::cout << "Average of values: "
                << (test[subspace_extractor].average_of_values().template
                    operator()<NumberType>(
                      fe_interface_values))[dof_index][q_point]
                << std::endl;
      std::cout << "Average of gradients: "
                << (test[subspace_extractor].average_of_gradients().template
                    operator()<NumberType>(
                      fe_interface_values))[dof_index][q_point]
                << std::endl;
      std::cout << "Average of Hessians: "
                << (test[subspace_extractor].average_of_hessians().template
                    operator()<NumberType>(
                      fe_interface_values))[dof_index][q_point]
                << std::endl;

      deallog << "OK" << std::endl;
    }

    {
      const std::string title = "Trial solution: " + type;
      std::cout << title << std::endl;
      deallog << title << std::endl;

      using namespace WeakForms;
      const TrialSolution<dim, spacedim> trial;
      const SubSpaceExtractors::Vector   subspace_extractor(0,
                                                          "u",
                                                          "\\mathbf{u}");

      std::cout << "Jump in values: "
                << (trial[subspace_extractor].jump_in_values().template
                    operator()<NumberType>(
                      fe_interface_values))[dof_index][q_point]
                << std::endl;
      std::cout << "Jump in gradients: "
                << (trial[subspace_extractor].jump_in_gradients().template
                    operator()<NumberType>(
                      fe_interface_values))[dof_index][q_point]
                << std::endl;
      std::cout << "Jump in Hessians: "
                << (trial[subspace_extractor].jump_in_hessians().template
                    operator()<NumberType>(
                      fe_interface_values))[dof_index][q_point]
                << std::endl;
      std::cout << "Jump in third derivatives: "
                << (trial[subspace_extractor]
                      .jump_in_third_derivatives()
                      .template operator()<NumberType>(
                        fe_interface_values))[dof_index][q_point]
                << std::endl;

      std::cout << "Average of values: "
                << (trial[subspace_extractor].average_of_values().template
                    operator()<NumberType>(
                      fe_interface_values))[dof_index][q_point]
                << std::endl;
      std::cout << "Average of gradients: "
                << (trial[subspace_extractor].average_of_gradients().template
                    operator()<NumberType>(
                      fe_interface_values))[dof_index][q_point]
                << std::endl;
      std::cout << "Average of Hessians: "
                << (trial[subspace_extractor].average_of_hessians().template
                    operator()<NumberType>(
                      fe_interface_values))[dof_index][q_point]
                << std::endl;

      deallog << "OK" << std::endl;
    }

    // Scratch data will only be correctly initialised if on an interface
    if (is_on_interface)
      {
        const std::string title = "Field solution: " + type;
        std::cout << title << std::endl;
        deallog << title << std::endl;

        using namespace WeakForms;
        const FieldSolution<dim, spacedim> field_solution;
        const SubSpaceExtractors::Vector   subspace_extractor(0,
                                                            "u",
                                                            "\\mathbf{u}");

        std::cout
          << "Jump in values: "
          << (field_solution[subspace_extractor].jump_in_values().template
              operator()<NumberType>(scratch_data, solution_names))[q_point]
          << std::endl;
        std::cout
          << "Jump in gradients: "
          << (field_solution[subspace_extractor].jump_in_gradients().template
              operator()<NumberType>(scratch_data, solution_names))[q_point]
          << std::endl;
        std::cout
          << "Jump in Hessians: "
          << (field_solution[subspace_extractor].jump_in_hessians().template
              operator()<NumberType>(scratch_data, solution_names))[q_point]
          << std::endl;
        std::cout << "Jump in third derivatives: "
                  << (field_solution[subspace_extractor]
                        .jump_in_third_derivatives()
                        .template operator()<NumberType>(
                          scratch_data, solution_names))[q_point]
                  << std::endl;

        std::cout
          << "Average of values: "
          << (field_solution[subspace_extractor].average_of_values().template
              operator()<NumberType>(scratch_data, solution_names))[q_point]
          << std::endl;
        std::cout
          << "Average of gradients: "
          << (field_solution[subspace_extractor].average_of_gradients().template
              operator()<NumberType>(scratch_data, solution_names))[q_point]
          << std::endl;
        std::cout
          << "Average of Hessians: "
          << (field_solution[subspace_extractor].average_of_hessians().template
              operator()<NumberType>(scratch_data, solution_names))[q_point]
          << std::endl;

        deallog << "OK" << std::endl;
      }
  };

  // Find a boundary face
  {
    FEInterfaceValues<dim, spacedim> fe_interface_values(
      fe, qf_face, update_flags_interface);
    unsigned int face = 0;
    while (!cell->face(face)->at_boundary())
      {
        ++face;
      }
    fe_interface_values.reinit(cell, face);

    solution_storage.extract_local_dof_values(scratch_data);
    test(fe_interface_values, "Face", false);
  }

  // Find an interface face
  {
    unsigned int face = 0;
    while (cell->face(face)->at_boundary())
      {
        ++face;
      }
    const unsigned int subface                = numbers::invalid_unsigned_int;
    const auto         cell_neighbour         = cell->neighbor(face);
    const unsigned int cell_neighbour_face    = cell->neighbor_face_no(face);
    const unsigned int cell_neighbour_subface = numbers::invalid_unsigned_int;
    const FEInterfaceValues<dim, spacedim> &fe_interface_values =
      scratch_data.reinit(cell,
                          face,
                          subface,
                          cell_neighbour,
                          cell_neighbour_face,
                          cell_neighbour_subface);

    solution_storage.extract_local_dof_values(scratch_data);
    test(fe_interface_values, "Interface", true);
  }

  deallog << "OK" << std::endl;
}


int
main(int argc, char *argv[])
{
  initlog();
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  run<2>();
  run<3>();

  deallog << "OK" << std::endl;
}
