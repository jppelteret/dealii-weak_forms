// ---------------------------------------------------------------------
//
// Copyright (C) 2018 - 2021 by the deal.II authors
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

// This test replicates step-58.
// It is used as a baseline for the weak form tests.


#include "../weak_forms_tests.h"
#include "wf_common_tests/step-58.h"


using namespace dealii;


template <int dim>
class Step58 : public Step58_Base<dim>
{
public:
  Step58();

protected:
  void
  assemble_matrices() override;
};


template <int dim>
Step58<dim>::Step58()
  : Step58_Base<dim>()
{}


template <int dim>
void
Step58<dim>::assemble_matrices()
{
  const QGauss<dim> quadrature_formula(this->fe.degree + 1);

  FEValues<dim> fe_values(this->fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = this->fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<std::complex<double>> cell_matrix_lhs(dofs_per_cell,
                                                   dofs_per_cell);
  FullMatrix<std::complex<double>> cell_matrix_rhs(dofs_per_cell,
                                                   dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<double>                  potential_values(n_q_points);
  const Potential<dim>                 potential;

  for (const auto &cell : this->dof_handler.active_cell_iterators())
    {
      cell_matrix_lhs = std::complex<double>(0.);
      cell_matrix_rhs = std::complex<double>(0.);

      fe_values.reinit(cell);

      potential.value_list(fe_values.get_quadrature_points(), potential_values);

      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        {
          for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
              for (unsigned int l = 0; l < dofs_per_cell; ++l)
                {
                  const std::complex<double> i = {0, 1};

                  cell_matrix_lhs(k, l) +=
                    (-i * fe_values.shape_value(k, q_index) *
                       fe_values.shape_value(l, q_index) +
                     this->time_step / 4 * fe_values.shape_grad(k, q_index) *
                       fe_values.shape_grad(l, q_index) +
                     this->time_step / 2 * potential_values[q_index] *
                       fe_values.shape_value(k, q_index) *
                       fe_values.shape_value(l, q_index)) *
                    fe_values.JxW(q_index);

                  cell_matrix_rhs(k, l) +=
                    (-i * fe_values.shape_value(k, q_index) *
                       fe_values.shape_value(l, q_index) -
                     this->time_step / 4 * fe_values.shape_grad(k, q_index) *
                       fe_values.shape_grad(l, q_index) -
                     this->time_step / 2 * potential_values[q_index] *
                       fe_values.shape_value(k, q_index) *
                       fe_values.shape_value(l, q_index)) *
                    fe_values.JxW(q_index);
                }
            }
        }

      cell->get_dof_indices(local_dof_indices);
      this->constraints.distribute_local_to_global(cell_matrix_lhs,
                                                   local_dof_indices,
                                                   this->system_matrix);
      this->constraints.distribute_local_to_global(cell_matrix_rhs,
                                                   local_dof_indices,
                                                   this->rhs_matrix);
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
      Step58<2> nse;
      nse.run();
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
