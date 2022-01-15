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


// Check that symbolic operators work
// - Cell face subface operators

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <weak_forms/cell_face_subface_operators.h>
#include <weak_forms/symbolic_operators.h>

#include "../weak_forms_tests.h"


template <int dim, int spacedim = dim, typename NumberType = double>
void
run()
{
  LogStream::Prefix prefix("Dim " + Utilities::to_string(dim));
  std::cout << "Dim: " << dim << std::endl;

  const FE_Q<dim, spacedim>  fe(1);
  const QGauss<spacedim>     qf_cell(fe.degree + 1);
  const QGauss<spacedim - 1> qf_face(fe.degree + 1);

  Triangulation<dim, spacedim> triangulation;
  GridGenerator::hyper_cube(triangulation);

  DoFHandler<dim, spacedim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  const UpdateFlags           update_flags_cell = update_quadrature_points;
  const UpdateFlags           update_flags_face = update_normal_vectors;
  FEValues<dim, spacedim>     fe_values(fe, qf_cell, update_flags_cell);
  FEFaceValues<dim, spacedim> fe_face_values(fe, qf_face, update_flags_face);
  fe_values.reinit(dof_handler.begin_active());
  fe_face_values.reinit(dof_handler.begin_active(), 0);

  const unsigned int q_point = 0;

  {
    const std::string title = "Normal";
    std::cout << title << std::endl;
    deallog << title << std::endl;

    using namespace WeakForms;

    const Normal<spacedim> normal{};
    const auto             functor = normal.value();

    std::cout << "Value: "
              << (functor.template operator()<NumberType>(
                   fe_face_values))[q_point]
              << std::endl;

    deallog << "OK" << std::endl;
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
