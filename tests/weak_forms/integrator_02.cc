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


// Check that integrator works for partial volumes, partial boundaries
// and for manifolds

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

  // Colour some cells, boundaries and manifolds
  const types::material_id mat_id_1 = 1;
  const types::material_id mat_id_2 = 2;
  for (auto &cell : triangulation.active_cell_iterators())
    {
      if (cell->center()[0] < 0.5)
        cell->set_material_id(mat_id_1);
      else
        cell->set_material_id(mat_id_2);
    }

  const types::material_id b_id_1 = 20;
  const types::material_id b_id_2 = 21;
  const types::material_id m_id   = 10;
  for (auto &cell : triangulation.active_cell_iterators())
    {
      for (const unsigned int face : GeometryInfo<dim>::face_indices())
        {
          if (cell->face(face)->at_boundary())
            {
              if (cell->center()[0] < 0.5)
                cell->face(face)->set_all_boundary_ids(b_id_1);
              else
                cell->face(face)->set_all_boundary_ids(b_id_2);
            }
          else if (cell->neighbor(face)->material_id() != cell->material_id())
            cell->face(face)->set_all_manifold_ids(m_id);
        }
    }

  DoFHandler<dim, spacedim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  Functions::ConstantFunction<spacedim, double> unity(1.0);

  // Volume integral (partial)
  {
    const double volume_1 =
      WeakForms::Integrator<dim, double>(unity).dV(cell_quadrature,
                                                   dof_handler,
                                                   {mat_id_1});
    const double volume_2 =
      WeakForms::Integrator<dim, double>(unity).dV(cell_quadrature,
                                                   dof_handler,
                                                   {mat_id_2});

    deallog << "Volume: " << volume_1 << " in material " << mat_id_1
            << std::endl;
    deallog << "Volume: " << volume_2 << " in material " << mat_id_2
            << std::endl;
  }

  // Boundary integral (partial)
  {
    const double area_1 =
      WeakForms::Integrator<dim, double>(unity).dA(face_quadrature,
                                                   dof_handler,
                                                   {b_id_1});
    const double area_2 =
      WeakForms::Integrator<dim, double>(unity).dA(face_quadrature,
                                                   dof_handler,
                                                   {b_id_2});

    deallog << "Area: " << area_1 << " on boundary " << b_id_1 << std::endl;
    deallog << "Area: " << area_2 << " on boundary " << b_id_2 << std::endl;
  }

  // Interface integral
  {}

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
