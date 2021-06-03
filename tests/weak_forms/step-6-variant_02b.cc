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

// This test replicates step-6.
// It is used as a baseline for the weak form tests.
//
// This particular variant reorganises the assembly, and uses FEValuesViews.
// Note that the structure of the assembly is different to step-6-variant_01b.


#include <deal.II/fe/fe_values_extractors.h>

#include "../weak_forms_tests.h"
#include "wf_common_tests/step-6.h"


using namespace dealii;


template <int dim>
class Step6 : public Step6_Base<dim>
{
public:
  Step6();

protected:
  void
  assemble_system() override;
};


template <int dim>
Step6<dim>::Step6()
  : Step6_Base<dim>()
{}


template <int dim>
void
Step6<dim>::assemble_system()
{
  FEValues<dim> fe_values(this->fe,
                          this->qf_cell,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = this->fe.dofs_per_cell;

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  auto coefficient = [](const Point<dim> &p) -> double {
    if (p.square() < 0.5 * 0.5)
      return 20;
    else
      return 1;
  };

  FEValuesExtractors::Scalar phi(0);

  for (const auto &cell : this->dof_handler.active_cell_iterators())
    {
      cell_matrix = 0;
      cell_rhs    = 0;

      fe_values.reinit(cell);

      // We organise everything with the quadrature point as the last index.
      // We do this because in the weak forms we plan on assemblying for all
      // DoFs (i,j) over all quadrature points q. This is the natural order to
      // integrate (via accumulation) into a scalar variable. If we were to
      // keep the quadrature point loop on the outside then we'd naturally want
      // integrate the entire cell matrix / vector contribution at a quadrature
      // point, which is perhaps not the most efficient thing to do.
      std::vector<double>              c_q(fe_values.n_quadrature_points);
      const std::vector<double> &      JxW_q = fe_values.get_JxW_values();
      std::vector<std::vector<double>> Nx_K_q(
        dofs_per_cell, std::vector<double>(fe_values.n_quadrature_points));
      std::vector<std::vector<Tensor<1, dim>>> grad_Nx_K_q(
        dofs_per_cell,
        std::vector<Tensor<1, dim>>(fe_values.n_quadrature_points));
      std::vector<std::vector<Tensor<1, dim>>> c_q_grad_Nx_K_q(
        dofs_per_cell,
        std::vector<Tensor<1, dim>>(fe_values.n_quadrature_points));

      for (const unsigned int q_point : fe_values.quadrature_point_indices())
        {
          c_q[q_point] = coefficient(fe_values.quadrature_point(q_point));
        }
      for (const unsigned int k : fe_values.dof_indices())
        {
          for (const unsigned int q_point :
               fe_values.quadrature_point_indices())
            {
              Nx_K_q[k][q_point]      = fe_values.shape_value(k, q_point);
              grad_Nx_K_q[k][q_point] = fe_values.shape_grad(k, q_point);

              // Precompute quantities of the form
              //   C(q)*Nx(J,q)
              // i.e. those with the trial solution on the right.
              // These quantities are repeatedly computed in the inner loop.
              // We want to keep the test functions free of precomputation, so
              // that the structure fits in with matrix-free as much as
              // possible.
              c_q_grad_Nx_K_q[k][q_point] =
                c_q[q_point] * grad_Nx_K_q[k][q_point];
            }
        }

      for (const unsigned int i : fe_values.dof_indices())
        {
          for (const unsigned int j : fe_values.dof_indices())
            for (const unsigned int q_point :
                 fe_values.quadrature_point_indices())
              {
                cell_matrix(i, j) +=
                  (grad_Nx_K_q[i][q_point] *     // grad phi_i(x_q)
                   c_q_grad_Nx_K_q[j][q_point] * // a(x_q) * grad phi_j(x_q)
                   JxW_q[q_point]);              // dx
              }

          for (const unsigned int q_point :
               fe_values.quadrature_point_indices())
            {
              cell_rhs(i) += (1.0 *                // f(x)
                              Nx_K_q[i][q_point] * // phi_i(x_q)
                              JxW_q[q_point]);     // dx
            }
        }

      cell->get_dof_indices(local_dof_indices);
      this->constraints.distribute_local_to_global(cell_matrix,
                                                   cell_rhs,
                                                   local_dof_indices,
                                                   this->system_matrix,
                                                   this->system_rhs);
    }
}


int
main(int argc, char **argv)
{
  initlog();
  deallog << std::setprecision(9);

  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  try
    {
      Step6<2> laplace_problem_2d;
      laplace_problem_2d.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  deallog << "OK" << std::endl;

  return 0;
}
