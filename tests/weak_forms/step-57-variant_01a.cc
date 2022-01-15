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

// This test replicates step-57.
// It is used as a baseline for the weak form tests.

#include "../weak_forms_tests.h"
#include "wf_common_tests/step-57.h"

namespace Step57
{
  template <int dim>
  class Step57 : public StationaryNavierStokes<dim>
  {
  public:
    Step57(const unsigned int degree)
      : StationaryNavierStokes<dim>(degree)
    {}

  protected:
    void
    assemble(const bool initial_step, const bool assemble_matrix) override;
  };


  template <int dim>
  void
  Step57<dim>::assemble(const bool initial_step, const bool assemble_matrix)
  {
    if (assemble_matrix)
      this->system_matrix = 0;

    this->system_rhs = 0;

    QGauss<dim> quadrature_formula(this->degree + 2);

    FEValues<dim> fe_values(this->fe,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);

    const unsigned int dofs_per_cell = this->fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


    std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
    std::vector<Tensor<2, dim>> present_velocity_gradients(n_q_points);
    std::vector<double>         present_pressure_values(n_q_points);

    std::vector<double>         div_phi_u(dofs_per_cell);
    std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
    std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double>         phi_p(dofs_per_cell);

    for (const auto &cell : this->dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);

        local_matrix = 0;
        local_rhs    = 0;

        fe_values[velocities].get_function_values(this->evaluation_point,
                                                  present_velocity_values);

        fe_values[velocities].get_function_gradients(
          this->evaluation_point, present_velocity_gradients);

        fe_values[pressure].get_function_values(this->evaluation_point,
                                                present_pressure_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                div_phi_u[k]  = fe_values[velocities].divergence(k, q);
                grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                phi_u[k]      = fe_values[velocities].value(k, q);
                phi_p[k]      = fe_values[pressure].value(k, q);
              }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                if (assemble_matrix)
                  {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        local_matrix(i, j) +=
                          (this->viscosity *
                             scalar_product(grad_phi_u[j], grad_phi_u[i]) +
                           present_velocity_gradients[q] * phi_u[j] * phi_u[i] +
                           grad_phi_u[j] * present_velocity_values[q] *
                             phi_u[i] -
                           div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j] +
                           this->gamma * div_phi_u[j] * div_phi_u[i] +
                           phi_p[i] * phi_p[j]) *
                          fe_values.JxW(q);
                      }
                  }

                double present_velocity_divergence =
                  trace(present_velocity_gradients[q]);
                local_rhs(i) +=
                  (-this->viscosity *
                     scalar_product(present_velocity_gradients[q],
                                    grad_phi_u[i]) -
                   present_velocity_gradients[q] * present_velocity_values[q] *
                     phi_u[i] +
                   present_pressure_values[q] * div_phi_u[i] +
                   present_velocity_divergence * phi_p[i] -
                   this->gamma * present_velocity_divergence * div_phi_u[i]) *
                  fe_values.JxW(q);
              }
          }

        cell->get_dof_indices(local_dof_indices);

        const AffineConstraints<double> &constraints_used =
          initial_step ? this->nonzero_constraints : this->zero_constraints;

        if (assemble_matrix)
          {
            constraints_used.distribute_local_to_global(local_matrix,
                                                        local_rhs,
                                                        local_dof_indices,
                                                        this->system_matrix,
                                                        this->system_rhs);
          }
        else
          {
            constraints_used.distribute_local_to_global(local_rhs,
                                                        local_dof_indices,
                                                        this->system_rhs);
          }
      }

    if (assemble_matrix)
      {
        this->pressure_mass_matrix.reinit(this->sparsity_pattern.block(1, 1));
        this->pressure_mass_matrix.copy_from(this->system_matrix.block(1, 1));

        this->system_matrix.block(1, 1) = 0;
      }
  }

} // namespace Step57

int
main(int argc, char **argv)
{
  initlog();
  deallog << std::setprecision(9);

  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  using namespace dealii;
  try
    {
      const unsigned int dim                       = 2;
      const unsigned int degree                    = 1;
      const unsigned int n_local_refinement_levels = 1;

      Step57::Step57<dim> flow(degree);
      flow.run(n_local_refinement_levels);
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
