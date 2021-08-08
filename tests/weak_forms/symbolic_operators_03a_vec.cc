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


// Check that symbolic operators work on interfaces (vectorised)
// - Trial function, test function (scalar-valued finite element)
// - [field solution (scalar-valued finite element)] -- NOT IMPLEMENTED YET

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
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


template <int dim, int spacedim = dim, typename NumberType = double>
void
run()
{
  LogStream::Prefix prefix("Dim " + Utilities::to_string(dim));
  std::cout << "Dim: " << dim << std::endl;

  const FE_DGQ<dim, spacedim> fe(3);
  const QGauss<spacedim>      qf_cell(fe.degree + 1);
  const QGauss<spacedim - 1>  qf_face(fe.degree + 1);

  Triangulation<dim, spacedim> triangulation;
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(1);

  DoFHandler<dim, spacedim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  Vector<double> solution(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler,
                           Functions::CosineFunction<spacedim>(
                             fe.n_components()),
                           solution);

  const UpdateFlags update_flags_interface =
    update_values | update_gradients | update_hessians | update_3rd_derivatives;
  FEInterfaceValues<dim, spacedim> fe_interface_values(fe,
                                                       qf_face,
                                                       update_flags_interface);

  const auto          cell = dof_handler.begin_active();
  std::vector<double> local_dof_values(fe.dofs_per_cell);
  cell->get_dof_values(solution,
                       local_dof_values.begin(),
                       local_dof_values.end());

  const auto test =
    [](const FEInterfaceValues<dim, spacedim> &fe_interface_values,
       const std::string &                     type)
  {
    const unsigned int dof_index =
      fe_interface_values.n_current_interface_dofs() - 1;

    constexpr std::size_t width =
      dealii::internal::VectorizedArrayWidthSpecifier<double>::max_width;
    const WeakForms::types::vectorized_qp_range_t q_point_range(0, width);

    std::cout << "dof_index: " << dof_index << " ; q_point range: [" << 0 << ","
              << width << ")" << std::endl;

    {
      const std::string title = "Test function: " + type;
      std::cout << title << std::endl;
      deallog << title << std::endl;

      using namespace WeakForms;
      const TestFunction<dim, spacedim> test;
      const SubSpaceExtractors::Scalar  subspace_extractor(0, "s", "s");

      std::cout << "Jump in values: "
                << (test[subspace_extractor].jump_in_values().template
                    operator()<NumberType, width>(fe_interface_values,
                                                  q_point_range))[dof_index]
                << std::endl;
      std::cout << "Jump in gradients: "
                << (test[subspace_extractor].jump_in_gradients().template
                    operator()<NumberType, width>(fe_interface_values,
                                                  q_point_range))[dof_index]
                << std::endl;
      std::cout << "Jump in Hessians: "
                << (test[subspace_extractor].jump_in_hessians().template
                    operator()<NumberType, width>(fe_interface_values,
                                                  q_point_range))[dof_index]
                << std::endl;
      std::cout << "Jump in third derivatives: "
                << (test[subspace_extractor]
                      .jump_in_third_derivatives()
                      .template operator()<NumberType, width>(
                        fe_interface_values, q_point_range))[dof_index]
                << std::endl;

      std::cout << "Average of values: "
                << (test[subspace_extractor].average_of_values().template
                    operator()<NumberType, width>(fe_interface_values,
                                                  q_point_range))[dof_index]
                << std::endl;
      std::cout << "Jump in gradients: "
                << (test[subspace_extractor].average_of_gradients().template
                    operator()<NumberType, width>(fe_interface_values,
                                                  q_point_range))[dof_index]
                << std::endl;
      std::cout << "Jump in Hessians: "
                << (test[subspace_extractor].average_of_hessians().template
                    operator()<NumberType, width>(fe_interface_values,
                                                  q_point_range))[dof_index]
                << std::endl;

      deallog << "OK" << std::endl;
    }

    {
      const std::string title = "Trial solution: " + type;
      std::cout << title << std::endl;
      deallog << title << std::endl;

      using namespace WeakForms;
      const TrialSolution<dim, spacedim> trial;
      const SubSpaceExtractors::Scalar   subspace_extractor(0, "s", "s");

      std::cout << "Jump in values: "
                << (trial[subspace_extractor].jump_in_values().template
                    operator()<NumberType, width>(fe_interface_values,
                                                  q_point_range))[dof_index]
                << std::endl;
      std::cout << "Jump in gradients: "
                << (trial[subspace_extractor].jump_in_gradients().template
                    operator()<NumberType, width>(fe_interface_values,
                                                  q_point_range))[dof_index]
                << std::endl;
      std::cout << "Jump in Hessians: "
                << (trial[subspace_extractor].jump_in_hessians().template
                    operator()<NumberType, width>(fe_interface_values,
                                                  q_point_range))[dof_index]
                << std::endl;
      std::cout << "Jump in third derivatives: "
                << (trial[subspace_extractor]
                      .jump_in_third_derivatives()
                      .template operator()<NumberType, width>(
                        fe_interface_values, q_point_range))[dof_index]
                << std::endl;

      std::cout << "Average of values: "
                << (trial[subspace_extractor].average_of_values().template
                    operator()<NumberType, width>(fe_interface_values,
                                                  q_point_range))[dof_index]
                << std::endl;
      std::cout << "Jump in gradients: "
                << (trial[subspace_extractor].average_of_gradients().template
                    operator()<NumberType, width>(fe_interface_values,
                                                  q_point_range))[dof_index]
                << std::endl;
      std::cout << "Jump in Hessians: "
                << (trial[subspace_extractor].average_of_hessians().template
                    operator()<NumberType, width>(fe_interface_values,
                                                  q_point_range))[dof_index]
                << std::endl;

      deallog << "OK" << std::endl;
    }
  };

  // Find a boundary face
  unsigned int face = 0;
  while (!cell->face(face)->at_boundary())
    {
      ++face;
    }
  fe_interface_values.reinit(cell, face);
  test(fe_interface_values, "Face");

  // Find an interface face
  face = 0;
  while (cell->face(face)->at_boundary())
    {
      ++face;
    }
  const unsigned int subface                = numbers::invalid_unsigned_int;
  const auto         cell_neighbour         = cell->neighbor(face);
  const unsigned int cell_neighbour_face    = cell->neighbor_face_no(face);
  const unsigned int cell_neighbour_subface = numbers::invalid_unsigned_int;
  fe_interface_values.reinit(cell,
                             face,
                             subface,
                             cell_neighbour,
                             cell_neighbour_face,
                             cell_neighbour_subface);
  test(fe_interface_values, "Interface");

  // Not implemented
  //   {
  //     const std::string title = "Field solution";
  //     std::cout << title << std::endl;
  //     deallog << title << std::endl;

  //     using namespace WeakForms;
  //     const FieldSolution<dim, spacedim> field_solution;

  //     std::cout << "Value: "
  //               << (field_solution.value().template operator()<NumberType,
  //               width>(
  //                    fe_values, local_dof_values))[q_point]
  //               << std::endl;
  //     std::cout << "Gradient: "
  //               << (field_solution.gradient().template
  //               operator()<NumberType, width>(
  //                    fe_values, local_dof_values))[q_point]
  //               << std::endl;
  //     std::cout << "Laplacian: "
  //               << (field_solution.laplacian().template
  //               operator()<NumberType, width>(
  //                    fe_values, local_dof_values))[q_point]
  //               << std::endl;
  //     std::cout << "Hessian: "
  //               << (field_solution.hessian().template operator()<NumberType,
  //               width>(
  //                    fe_values, local_dof_values))[q_point]
  //               << std::endl;
  //     std::cout << "Third derivative: "
  //               << (field_solution.third_derivative().template
  //                   operator()<NumberType, width>(fe_values,
  //                   local_dof_values))[q_point]
  //               << std::endl;

  //     deallog << "OK" << std::endl;
  //   }

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
