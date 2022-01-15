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


// This test replicates step-23, this time using the typical (explicit)
// approach to assembly.

#include "../weak_forms_tests.h"
#include "wf_common_tests/step-23.h"

namespace Step23
{
  using namespace dealii;

  template <int dim>
  class Step23 : public Step23_Base<dim>
  {
  public:
    Step23()
      : Step23_Base<dim>()
      , extractor(0)
    {
      constraints.close();
    }

  protected:
    const FEValuesExtractors::Scalar extractor;
    AffineConstraints<double>        constraints;

    void
    assemble_u() override;
    void
    assemble_v() override;
  };

  template <int dim>
  void
  Step23<dim>::assemble_u()
  {
    this->matrix_u   = 0;
    this->system_rhs = 0;

    FEValues<dim> fe_values(this->fe,
                            this->quadrature,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = this->fe.n_dofs_per_cell();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    RightHandSide<dim>  rhs_function;
    std::vector<double> rhs_values(fe_values.n_quadrature_points);
    std::vector<double> rhs_values_t1(fe_values.n_quadrature_points);

    std::vector<double> solution_values_u_t1(fe_values.n_quadrature_points);
    std::vector<double> solution_values_v_t1(fe_values.n_quadrature_points);
    std::vector<Tensor<1, dim>> solution_gradients_u_t1(
      fe_values.n_quadrature_points);

    for (const auto &cell : this->dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;
        cell_rhs    = 0;
        fe_values.reinit(cell);

        rhs_function.set_time(this->time);
        rhs_function.value_list(fe_values.get_quadrature_points(), rhs_values);
        rhs_function.set_time(this->time - this->time_step);
        rhs_function.value_list(fe_values.get_quadrature_points(),
                                rhs_values_t1);

        fe_values[extractor].get_function_values(this->old_solution_u,
                                                 solution_values_u_t1);
        fe_values[extractor].get_function_values(this->old_solution_v,
                                                 solution_values_v_t1);
        fe_values[extractor].get_function_gradients(this->old_solution_u,
                                                    solution_gradients_u_t1);

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          {
            const double current_coefficient = rhs_values[q_index];
            for (const unsigned int i : fe_values.dof_indices())
              {
                for (const unsigned int j : fe_values.dof_indices())
                  {
                    cell_matrix(i, j) += fe_values.shape_value(i, q_index) *
                                         fe_values.shape_value(j, q_index) *
                                         fe_values.JxW(q_index);
                    cell_matrix(i, j) += (this->theta * this->theta *
                                          this->time_step * this->time_step) *
                                         fe_values.shape_grad(i, q_index) *
                                         fe_values.shape_grad(j, q_index) *
                                         fe_values.JxW(q_index);
                  }

                cell_rhs(i) += fe_values[extractor].value(i, q_index) *
                               solution_values_u_t1[q_index] *
                               fe_values.JxW(q_index);
                cell_rhs(i) +=
                  this->time_step * fe_values[extractor].value(i, q_index) *
                  solution_values_v_t1[q_index] * fe_values.JxW(q_index);
                cell_rhs(i) -= (this->theta * (1 - this->theta) *
                                this->time_step * this->time_step) *
                               fe_values[extractor].gradient(i, q_index) *
                               solution_gradients_u_t1[q_index] *
                               fe_values.JxW(q_index);

                // Forcing term * (theta*time_step)
                cell_rhs(i) += (this->theta * this->time_step) *
                               (this->theta * this->time_step) *
                               fe_values.shape_value(i, q_index) *
                               rhs_values[q_index] * fe_values.JxW(q_index);
                cell_rhs(i) += (this->theta * this->time_step) *
                               ((1 - this->theta) * this->time_step) *
                               fe_values.shape_value(i, q_index) *
                               rhs_values_t1[q_index] * fe_values.JxW(q_index);
              }
          }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix,
                                               cell_rhs,
                                               local_dof_indices,
                                               this->matrix_u,
                                               this->system_rhs);
      }
  }

  template <int dim>
  void
  Step23<dim>::assemble_v()
  {
    this->matrix_v   = 0;
    this->system_rhs = 0;

    FEValues<dim> fe_values(this->fe,
                            this->quadrature,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = this->fe.n_dofs_per_cell();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    RightHandSide<dim>  rhs_function;
    std::vector<double> rhs_values(fe_values.n_quadrature_points);
    std::vector<double> rhs_values_t1(fe_values.n_quadrature_points);

    std::vector<double> solution_values_v_t1(fe_values.n_quadrature_points);
    std::vector<Tensor<1, dim>> solution_gradients_u(
      fe_values.n_quadrature_points);
    std::vector<Tensor<1, dim>> solution_gradients_u_t1(
      fe_values.n_quadrature_points);

    for (const auto &cell : this->dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;
        cell_rhs    = 0;
        fe_values.reinit(cell);

        rhs_function.set_time(this->time);
        rhs_function.value_list(fe_values.get_quadrature_points(), rhs_values);
        rhs_function.set_time(this->time - this->time_step);
        rhs_function.value_list(fe_values.get_quadrature_points(),
                                rhs_values_t1);

        fe_values[extractor].get_function_values(this->old_solution_v,
                                                 solution_values_v_t1);
        fe_values[extractor].get_function_gradients(this->solution_u,
                                                    solution_gradients_u);
        fe_values[extractor].get_function_gradients(this->old_solution_u,
                                                    solution_gradients_u_t1);

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          {
            for (const unsigned int i : fe_values.dof_indices())
              {
                for (const unsigned int j : fe_values.dof_indices())
                  {
                    cell_matrix(i, j) += fe_values.shape_value(i, q_index) *
                                         fe_values.shape_value(j, q_index) *
                                         fe_values.JxW(q_index);
                  }

                cell_rhs(i) -= (this->theta * this->time_step) *
                               fe_values[extractor].gradient(i, q_index) *
                               solution_gradients_u[q_index] *
                               fe_values.JxW(q_index);
                cell_rhs(i) += fe_values[extractor].value(i, q_index) *
                               solution_values_v_t1[q_index] *
                               fe_values.JxW(q_index);
                cell_rhs(i) -= (this->time_step * (1 - this->theta)) *
                               fe_values[extractor].gradient(i, q_index) *
                               solution_gradients_u_t1[q_index] *
                               fe_values.JxW(q_index);

                // Forcing term
                cell_rhs(i) += (this->theta * this->time_step) *
                               fe_values.shape_value(i, q_index) *
                               rhs_values[q_index] * fe_values.JxW(q_index);
                cell_rhs(i) += ((1 - this->theta) * this->time_step) *
                               fe_values.shape_value(i, q_index) *
                               rhs_values_t1[q_index] * fe_values.JxW(q_index);
              }
          }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix,
                                               cell_rhs,
                                               local_dof_indices,
                                               this->matrix_v,
                                               this->system_rhs);
      }
  }
} // namespace Step23


int
main(int argc, char **argv)
{
  initlog();

  deallog.depth_file(1);

  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  try
    {
      Step23::Step23<2> wave_equation_solver;
      wave_equation_solver.run();
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

  return 0;
}
