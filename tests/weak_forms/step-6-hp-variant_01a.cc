// ---------------------------------------------------------------------
//
// Copyright (C) 2005 - 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

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

// This test replicates step-6.
// It is used as a baseline for the weak form tests.
// - hp variant


#include "../weak_forms_tests.h"
#include "wf_common_tests/step-6-hp.h"


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
  hp::FEValues<dim> hp_fe_values(this->fe_collection,
                                 this->qf_collection_cell,
                                 update_values | update_gradients |
                                   update_quadrature_points |
                                   update_JxW_values);

  RightHandSide<dim> rhs_function;

  FullMatrix<double> cell_matrix;
  Vector<double>     cell_rhs;

  std::vector<types::global_dof_index> local_dof_indices;

  for (const auto &cell : this->dof_handler.active_cell_iterators())
    {
      const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();

      cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_matrix = 0;

      cell_rhs.reinit(dofs_per_cell);
      cell_rhs = 0;

      hp_fe_values.reinit(cell);

      const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

      std::vector<double> rhs_values(fe_values.n_quadrature_points);
      rhs_function.value_list(fe_values.get_quadrature_points(), rhs_values);

      for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points;
           ++q_point)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_point) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_point) * // grad phi_j(x_q)
                 fe_values.JxW(q_point));           // dx

            cell_rhs(i) += (fe_values.shape_value(i, q_point) * // phi_i(x_q)
                            rhs_values[q_point] *               // f(x_q)
                            fe_values.JxW(q_point));            // dx
          }

      local_dof_indices.resize(dofs_per_cell);
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
