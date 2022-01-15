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


// Check that integrator works for volumes and boundaries

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <weak_forms/integrator.h>

#include "../weak_forms_tests.h"


template <int dim, int spacedim = dim>
void
run()
{
  deallog << "Dim: " << dim << std::endl;

  const FE_Q<dim, spacedim> fe(1);
  const QGauss<dim>         cell_quadrature(fe.degree + 1);
  const QGauss<dim - 1>     face_quadrature(fe.degree + 1);

  Triangulation<dim, spacedim> triangulation;
  GridGenerator::subdivided_hyper_cube(triangulation, 4, 0.0, 1.0);

  DoFHandler<dim, spacedim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  Functions::ConstantFunction<spacedim, double> unity(1.0);

  // Volume integral
  {
    const double volume =
      WeakForms::Integrator<dim, double>(unity).dV(cell_quadrature,
                                                   dof_handler);
    deallog << "Volume: " << volume << std::endl;

    double reference_volume = 0.0;
    for (auto &cell : dof_handler.active_cell_iterators())
      reference_volume += cell->measure();

    Assert(std::abs(volume - reference_volume) < 1e-6,
           ExcMessage("Volumes do not match. Reference value: " +
                      Utilities::to_string(reference_volume) +
                      "; Calculated value: " + Utilities::to_string(volume)));
  }

  // Boundary integral
  {
    const double area =
      WeakForms::Integrator<dim, double>(unity).dA(face_quadrature,
                                                   dof_handler);
    deallog << "Area: " << area << std::endl;

    double reference_area = 0.0;
    for (auto &cell : dof_handler.active_cell_iterators())
      for (const unsigned int face : GeometryInfo<dim>::face_indices())
        if (cell->face(face)->at_boundary())
          {
            reference_area += cell->face(face)->measure();
          }

    Assert(std::abs(area - reference_area) < 1e-6,
           ExcMessage("Areas do not match. Reference value: " +
                      Utilities::to_string(reference_area) +
                      "; Calculated value: " + Utilities::to_string(area)));
  }

  deallog << "OK" << std::endl;
}


int
main()
{
  initlog();

  run<2>();
  run<3>();

  deallog << "OK" << std::endl;
}
