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

// Transient, uncoupled curl-curl problem.
//
// This test uses an external field, associated with a second DoFHandler, as
// a current source term on the RHS.

#include "../weak_forms_tests.h"
#include "wf_common_tests/step-transient_curl_curl.h"

namespace StepTransientCurlCurl
{
  template <int dim>
  class StepTransientCurlCurl : public StepTransientCurlCurl_Base<dim>
  {
  public:
    StepTransientCurlCurl(const std::string &input_file)
      : StepTransientCurlCurl_Base<dim>(input_file)
    {
      this->parameters.wire_excitation = "Voltage";
    }

  protected:
    virtual void
    assemble_system_esp(
      TrilinosWrappers::SparseMatrix & system_matrix_esp,
      TrilinosWrappers::MPI::Vector &  system_rhs_esp,
      const AffineConstraints<double> &constraints_esp) override;

    virtual void
    assemble_system_mvp(
      TrilinosWrappers::SparseMatrix & system_matrix_mvp,
      TrilinosWrappers::MPI::Vector &  system_rhs_mvp,
      const AffineConstraints<double> &constraints_mvp) override;
  };


  // @sect4{StepTransientCurlCurl_Base::assemble_system}

  template <int dim>
  void
  StepTransientCurlCurl<dim>::assemble_system_esp(
    TrilinosWrappers::SparseMatrix & system_matrix_esp,
    TrilinosWrappers::MPI::Vector &  system_rhs_esp,
    const AffineConstraints<double> &constraints_esp)
  {
    TimerOutput::Scope timer_scope(this->computing_timer, "Assembly: ESP");
    system_matrix_esp = 0.0;
    system_rhs_esp    = 0.0;

    FEValues<dim> fe_values(this->mapping,
                            this->fe_esp,
                            this->qf_cell_esp,
                            update_gradients | update_quadrature_points |
                              update_JxW_values);

    typename DoFHandler<dim>::active_cell_iterator
      cell = this->dof_handler_esp.begin_active(),
      endc = this->dof_handler_esp.end();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned() == false)
          continue;

        fe_values.reinit(cell);
        const unsigned int &n_q_points      = fe_values.n_quadrature_points;
        const unsigned int &n_dofs_per_cell = fe_values.dofs_per_cell;

        FullMatrix<double> cell_matrix(n_dofs_per_cell, n_dofs_per_cell);
        Vector<double>     cell_rhs(n_dofs_per_cell);

        std::vector<double> conductivity_coefficient_values(n_q_points);
        this->function_material_conductivity_coefficients.value_list(
          fe_values.get_quadrature_points(), conductivity_coefficient_values);

        // Pre-compute QP data
        std::vector<std::vector<Tensor<1, dim>>> qp_grad_Nx(
          n_q_points, std::vector<Tensor<1, dim>>(n_dofs_per_cell));
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
              {
                qp_grad_Nx[q_point][k] =
                  fe_values[this->esp_extractor].gradient(k, q_point);
              }
          }

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            const std::vector<Tensor<1, dim>> &grad_Nx = qp_grad_Nx[q_point];
            const double &                     JxW     = fe_values.JxW(q_point);

            AssertThrow(
              std::abs(conductivity_coefficient_values[q_point]) > 1e-9,
              ExcMessage(
                "Electric conductivity coefficient must be non-zero."));

            const double &sigma = conductivity_coefficient_values[q_point];

            for (unsigned int I = 0; I < n_dofs_per_cell; ++I)
              {
                for (unsigned int J = 0; J <= I; ++J)
                  {
                    cell_matrix(I, J) +=
                      (grad_Nx[I] * sigma * grad_Nx[J]) * JxW;
                  }
              }
          }


        // Finally, we need to copy the lower half of the local matrix into the
        // upper half:
        for (unsigned int I = 0; I < n_dofs_per_cell; ++I)
          for (unsigned int J = I + 1; J < n_dofs_per_cell; ++J)
            cell_matrix(I, J) = cell_matrix(J, I);

        std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);
        constraints_esp.distribute_local_to_global(cell_matrix,
                                                   cell_rhs,
                                                   local_dof_indices,
                                                   system_matrix_esp,
                                                   system_rhs_esp);
      }

    system_matrix_esp.compress(VectorOperation::add);
    system_rhs_esp.compress(VectorOperation::add);
  }


  template <int dim>
  void
  StepTransientCurlCurl<dim>::assemble_system_mvp(
    TrilinosWrappers::SparseMatrix & system_matrix_mvp,
    TrilinosWrappers::MPI::Vector &  system_rhs_mvp,
    const AffineConstraints<double> &constraints_mvp)
  {
    TimerOutput::Scope timer_scope(this->computing_timer, "Assembly: MVP");
    system_matrix_mvp = 0.0;
    system_rhs_mvp    = 0.0;

    FEValues<dim> fe_values(this->mapping,
                            this->fe_mvp,
                            this->qf_cell_mvp,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    FEValues<dim> fe_values_esp(this->mapping,
                                this->fe_esp,
                                this->qf_cell_esp,
                                update_gradients);

    typename DoFHandler<dim>::active_cell_iterator
      cell = this->dof_handler_mvp.begin_active(),
      endc = this->dof_handler_mvp.end();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned() == false)
          continue;

        fe_values.reinit(cell);
        const unsigned int &n_q_points      = fe_values.n_quadrature_points;
        const unsigned int &n_dofs_per_cell = fe_values.dofs_per_cell;

        FullMatrix<double> cell_matrix(n_dofs_per_cell, n_dofs_per_cell);
        Vector<double>     cell_rhs(n_dofs_per_cell);

        if (cell->material_id() == this->parameters.mid_wire)
          {
            Assert(this->geometry.within_wire(cell->center()) == true,
                   ExcMessage("Expected cell to be in wire."));
          }
        else
          {
            Assert(this->geometry.within_wire(cell->center()) == false,
                   ExcMessage("Expected cell not to be in wire."));
          }

        std::vector<double>         permeability_coefficient_values(n_q_points);
        std::vector<double>         conductivity_coefficient_values(n_q_points);
        std::vector<Tensor<1, dim>> source_values(n_q_points);
        this->function_material_permeability_coefficients.value_list(
          fe_values.get_quadrature_points(), permeability_coefficient_values);
        this->function_material_conductivity_coefficients.value_list(
          fe_values.get_quadrature_points(), conductivity_coefficient_values);

        // this->function_free_current_density.value_list(
        //   fe_values.get_quadrature_points(), source_values);
        Assert(this->parameters.use_voltage_excitation(), ExcInternalError());
        if (cell->material_id() == this->parameters.mid_wire)
          {
            typename DoFHandler<dim>::active_cell_iterator cell_esp(
              &this->triangulation,
              cell->level(),
              cell->index(),
              &this->dof_handler_esp);
            fe_values_esp.reinit(cell_esp);

            fe_values_esp[this->esp_extractor].get_function_gradients(
              this->solution_esp, source_values);
            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
              source_values[q_point] *= -1.0;
          }

        std::vector<Tensor<1, dim>> solution_mvp_values(n_q_points);
        std::vector<Tensor<1, dim>> solution_mvp_curls(n_q_points);
        std::vector<Tensor<1, dim>> solution_mvp_values_t1(n_q_points);
        std::vector<Tensor<1, dim>> d_solution_mvp_dt_values(n_q_points);
        fe_values[this->mvp_extractor].get_function_values(this->solution_mvp,
                                                           solution_mvp_values);
        fe_values[this->mvp_extractor].get_function_curls(this->solution_mvp,
                                                          solution_mvp_curls);
        fe_values[this->mvp_extractor].get_function_values(
          this->solution_mvp_t1, solution_mvp_values_t1);
        fe_values[this->mvp_extractor].get_function_values(
          this->d_solution_mvp_dt, d_solution_mvp_dt_values);

        // Pre-compute QP data
        std::vector<std::vector<Tensor<1, dim>>> qp_Nx(
          n_q_points, std::vector<Tensor<1, dim>>(n_dofs_per_cell));
        std::vector<std::vector<Tensor<1, dim>>> qp_curl_Nx(
          n_q_points, std::vector<Tensor<1, dim>>(n_dofs_per_cell));
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
              {
                qp_Nx[q_point][k] =
                  fe_values[this->mvp_extractor].value(k, q_point);
                qp_curl_Nx[q_point][k] =
                  fe_values[this->mvp_extractor].curl(k, q_point);
              }
          }

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            const std::vector<Tensor<1, dim>> &Nx      = qp_Nx[q_point];
            const std::vector<Tensor<1, dim>> &curl_Nx = qp_curl_Nx[q_point];
            const double &                     JxW     = fe_values.JxW(q_point);

            AssertThrow(
              std::abs(permeability_coefficient_values[q_point]) > 1e-9,
              ExcMessage(
                "Magnetic permeability coefficient must be non-zero."));
            AssertThrow(
              std::abs(conductivity_coefficient_values[q_point]) > 1e-9,
              ExcMessage(
                "Electric conductivity coefficient must be non-zero."));

            const double inv_mu_r_mu_0 =
              1.0 / permeability_coefficient_values[q_point];
            const double &sigma    = conductivity_coefficient_values[q_point];
            const double &dt       = this->parameters.delta_t;
            const double  sigma_dt = sigma / dt;
            const double  kappa =
              (this->parameters.regularisation_parameter / dim) * inv_mu_r_mu_0;

            const Tensor<1, dim> &A      = solution_mvp_values[q_point];
            const Tensor<1, dim> &curl_A = solution_mvp_curls[q_point];
            const Tensor<1, dim> &A_t1   = solution_mvp_values_t1[q_point];
            const Tensor<1, dim> &dA_dt  = d_solution_mvp_dt_values[q_point];

            // Uniform current through wire
            // Note: J_f must be divergence free!
            const Tensor<1, dim> &J_f = source_values[q_point];

            for (unsigned int I = 0; I < n_dofs_per_cell; ++I)
              {
                for (unsigned int J = 0; J <= I; ++J)
                  {
                    if (cell->material_id() == this->parameters.mid_wire)
                      {
                        cell_matrix(I, J) +=
                          (curl_Nx[I] * inv_mu_r_mu_0 * curl_Nx[J] +
                           Nx[I] * sigma_dt * Nx[J]) *
                          JxW;
                      }
                    else
                      {
                        cell_matrix(I, J) +=
                          (curl_Nx[I] * inv_mu_r_mu_0 * curl_Nx[J] +
                           Nx[I] * kappa * Nx[J]) *
                          JxW;
                      }
                  }

                if (cell->material_id() == this->parameters.mid_wire)
                  {
                    cell_rhs(I) -= (curl_Nx[I] * inv_mu_r_mu_0 * curl_A +
                                    Nx[I] * sigma * dA_dt) *
                                   JxW;
                  }
                else
                  {
                    cell_rhs(I) -= (curl_Nx[I] * inv_mu_r_mu_0 * curl_A +
                                    Nx[I] * kappa * A) *
                                   JxW;
                  }

                // Source term
                cell_rhs(I) += (Nx[I] * J_f) * JxW;
              }
          }


        // Finally, we need to copy the lower half of the local matrix into the
        // upper half:
        for (unsigned int I = 0; I < n_dofs_per_cell; ++I)
          for (unsigned int J = I + 1; J < n_dofs_per_cell; ++J)
            cell_matrix(I, J) = cell_matrix(J, I);

        std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);
        constraints_mvp.distribute_local_to_global(cell_matrix,
                                                   cell_rhs,
                                                   local_dof_indices,
                                                   system_matrix_mvp,
                                                   system_rhs_mvp);
      }

    system_matrix_mvp.compress(VectorOperation::add);
    system_rhs_mvp.compress(VectorOperation::add);
  }

} // namespace StepTransientCurlCurl

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());
  mpi_initlog();
  deallog << std::setprecision(9);

  try
    {
      ConditionalOStream pcout(
        std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

      const std::string input_file(SOURCE_DIR
                                   "/prm/parameters-step-curl_curl.prm");

      {
        const std::string title = "Running in 3-d...";
        const std::string divider(title.size(), '=');

        pcout << divider << std::endl
              << title << std::endl
              << divider << std::endl;

        StepTransientCurlCurl::StepTransientCurlCurl<3> curl_curl_3d(
          input_file);
        curl_curl_3d.run();
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl
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
                << std::endl
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
