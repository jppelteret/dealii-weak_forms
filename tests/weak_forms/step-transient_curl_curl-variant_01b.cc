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

// Transient, uncoupled curl-curl problem: Assembly using composite weak forms
// This test replicates step-transient_curl_curl exactly.

#include <weak_forms/weak_forms.h>

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
    {}

  protected:
    virtual void
    assemble_system(TrilinosWrappers::SparseMatrix & system_matrix,
                    TrilinosWrappers::MPI::Vector &  system_rhs,
                    const AffineConstraints<double> &constraints) override;
  };


  // @sect4{StepTransientCurlCurl_Base::assemble_system}

  template <int dim>
  void
  StepTransientCurlCurl<dim>::assemble_system(
    TrilinosWrappers::SparseMatrix & system_matrix,
    TrilinosWrappers::MPI::Vector &  system_rhs,
    const AffineConstraints<double> &constraints)
  {
    using namespace WeakForms;
    constexpr int spacedim = dim;

    TimerOutput::Scope timer_scope(this->computing_timer, "Assembly");
    system_matrix = 0.0;
    system_rhs    = 0.0;

    // Symbolic types for test function, trial solution and a coefficient.
    const TestFunction<dim, spacedim>  test;
    const TrialSolution<dim, spacedim> trial;
    const FieldSolution<dim, spacedim> field_solution;
    const SubSpaceExtractors::Vector   subspace_extractor_A(0,
                                                          0,
                                                          "A",
                                                          "\\mathbf{A}");

    // Test function (subspaced)
    const auto test_A      = test[subspace_extractor_A].value();
    const auto test_curl_A = test[subspace_extractor_A].curl();

    // Test function (subspaced)
    const auto trial_A      = trial[subspace_extractor_A].value();
    const auto trial_curl_A = trial[subspace_extractor_A].curl();

    // Create storage for the solution vectors that may be referenced
    // by the weak forms
    using VectorType = TrilinosWrappers::MPI::Vector;
    const SolutionStorage<VectorType> solution_storage(
      {&this->solution, &this->solution_t1});

    // Field solution
    constexpr WeakForms::types::solution_index solution_index_t1 = 1;
    const auto                                 A_t1 =
      field_solution[subspace_extractor_A].template value<solution_index_t1>();

    // Functions
    const ScalarFunctor                   timestep("dt", "\\Delta t");
    const ScalarFunctionFunctor<spacedim> permeability("mu(x)", "\\mu");
    const ScalarFunctionFunctor<spacedim> conductivity("sigma(x)", "\\sigma");
    const VectorFunctionFunctor<spacedim> current_source("Jf(x)",
                                                         "\\mathbf{J}^{free}");

    const auto dt = timestep.value<double, dim, spacedim>(
      [this](const FEValuesBase<dim, spacedim> &, const unsigned int)
      { return this->parameters.delta_t; });

    const auto mu =
      permeability.value(this->function_material_permeability_coefficients);
    const auto sigma =
      conductivity.value(this->function_material_conductivity_coefficients);
    const auto J_f = current_source.value(this->function_free_current_density);

    // Assembly
    MatrixBasedAssembler<dim> assembler;
    assembler +=
      bilinear_form(test_curl_A, 1.0 / mu, trial_curl_A).symmetrize().dV() +
      bilinear_form(test_A, sigma / dt, trial_A).symmetrize().dV();
    assembler -= linear_form(test_A, (sigma / dt) * A_t1).dV() +
                 linear_form(test_A, J_f).dV();
    // assembler.symmetrize();

    // Look at what we're going to compute
    const SymbolicDecorations decorator;
    static bool               output = true;
    if (output)
      {
        std::cout << "\n\n" << std::endl;
        std::cout << "Weak form (ascii):\n"
                  << assembler.as_ascii(decorator) << std::endl;
        std::cout << "Weak form (LaTeX):\n"
                  << assembler.as_latex(decorator) << std::endl;
        std::cout << "\n\n" << std::endl;
        output = false;
      }

    // Now we pass in concrete objects to get data from
    // and assemble into.
    assembler.assemble_system(system_matrix,
                              system_rhs,
                              solution_storage,
                              constraints,
                              this->dof_handler,
                              this->qf_cell);

    // {
    // FEValues<dim> fe_values(this->mapping,
    //                         this->fe,
    //                         this->qf_cell,
    //                         update_values | update_gradients |
    //                           update_quadrature_points | update_JxW_values);

    // typename DoFHandler<dim>::active_cell_iterator cell = this->dof_handler
    //                                                         .begin_active(),
    //                                                endc =
    //                                                  this->dof_handler.end();
    // for (; cell != endc; ++cell)
    //   {
    //     //    if (cell->is_locally_owned() == false) continue;
    //     if (cell->subdomain_id() != this->this_mpi_process)
    //       continue;

    //     fe_values.reinit(cell);
    //     const unsigned int &n_q_points      = fe_values.n_quadrature_points;
    //     const unsigned int &n_dofs_per_cell = fe_values.dofs_per_cell;

    //     FullMatrix<double> cell_matrix(n_dofs_per_cell, n_dofs_per_cell);
    //     Vector<double>     cell_rhs(n_dofs_per_cell);

    //     std::vector<double> permeability_coefficient_values(n_q_points);
    //     std::vector<double> conductivity_coefficient_values(n_q_points);
    //     std::vector<Tensor<1, dim>> source_values(n_q_points);
    //     this->function_material_permeability_coefficients.value_list(
    //       fe_values.get_quadrature_points(),
    //       permeability_coefficient_values);
    //     this->function_material_conductivity_coefficients.value_list(
    //       fe_values.get_quadrature_points(),
    //       conductivity_coefficient_values);
    //     this->function_free_current_density.value_list(
    //       fe_values.get_quadrature_points(), source_values);

    //     // std::vector<Tensor<1, dim>> solution_curls(n_q_points);
    //     std::vector<Tensor<1, dim>> solution_values_t1(n_q_points);
    //     // std::vector<Tensor<1, dim>> d_solution_dt_values(n_q_points);
    //     // fe_values[this->mvp_fe].get_function_curls(this->solution,
    //     // solution_curls);
    //     fe_values[this->mvp_fe].get_function_values(this->solution_t1,
    //                                                 solution_values_t1);
    //     // fe_values[this->mvp_fe].get_function_values(this->d_solution_dt,
    //     //                                       d_solution_dt_values);

    //     // Pre-compute QP data
    //     std::vector<std::vector<Tensor<1, dim>>> qp_Nx(
    //       n_q_points, std::vector<Tensor<1, dim>>(n_dofs_per_cell));
    //     std::vector<std::vector<Tensor<1, dim>>> qp_curl_Nx(
    //       n_q_points, std::vector<Tensor<1, dim>>(n_dofs_per_cell));
    //     for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    //       {
    //         for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
    //           {
    //             qp_Nx[q_point][k] = fe_values[this->mvp_fe].value(k,
    //             q_point); qp_curl_Nx[q_point][k] =
    //               fe_values[this->mvp_fe].curl(k, q_point);
    //           }
    //       }

    //     for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    //       {
    //         const std::vector<Tensor<1, dim>> &Nx      = qp_Nx[q_point];
    //         const std::vector<Tensor<1, dim>> &curl_Nx = qp_curl_Nx[q_point];
    //         const double &                     JxW     =
    //         fe_values.JxW(q_point);

    //         AssertThrow(
    //           std::abs(permeability_coefficient_values[q_point]) > 1e-9,
    //           ExcMessage(
    //             "Magnetic permeability coefficient must be non-zero."));
    //         AssertThrow(
    //           std::abs(conductivity_coefficient_values[q_point]) > 1e-9,
    //           ExcMessage(
    //             "Electric conductivity coefficient must be non-zero."));

    //         const double inv_mu_r_mu_0 =
    //           1.0 / permeability_coefficient_values[q_point];
    //         const double &sigma    =
    //         conductivity_coefficient_values[q_point]; const double &dt =
    //         this->parameters.delta_t; const double  sigma_dt = sigma / dt;

    //         // const Tensor<1, dim> &curl_A = solution_curls[q_point];
    //         const Tensor<1, dim> &A_t1 = solution_values_t1[q_point];
    //         // const Tensor<1, dim> &dA_dt  = d_solution_dt_values[q_point];

    //         // Uniform current through wire
    //         // Note: J_f must be divergence free!
    //         const Tensor<1, dim> &J_f = source_values[q_point];

    //         for (unsigned int I = 0; I < n_dofs_per_cell; ++I)
    //           {
    //             for (unsigned int J = 0; J <= I; ++J)
    //               {
    //                 cell_matrix(I, J) +=
    //                   (
    //                     curl_Nx[I] * inv_mu_r_mu_0 * curl_Nx[J]
    //                   //  + Nx[I] * sigma_dt * Nx[J]
    //                    ) *
    //                   JxW;
    //               }

    //             // For the linear problem, this is the contribution from the
    //             // rate dependent term that comes from its time
    //             discretisation.
    //             // cell_rhs(I) += (Nx[I] * sigma_dt * A_t1) * JxW;

    //             // // For the incremental non-linear problem, we'd add these
    //             terms
    //             // // cell_rhs(I) -= (curl_Nx[I] * inv_mu_r_mu_0 * curl_A) *
    //             JxW;
    //             // // cell_rhs(I) -= (Nx[I] * sigma * dA_dt) * JxW;

    //             // cell_rhs(I) += (Nx[I] * J_f) * JxW;
    //           }
    //       }


    //     // Finally, we need to copy the lower half of the local matrix into
    //     the
    //     // upper half:
    //     for (unsigned int I = 0; I < n_dofs_per_cell; ++I)
    //       for (unsigned int J = I + 1; J < n_dofs_per_cell; ++J)
    //         cell_matrix(I, J) = cell_matrix(J, I);

    //     std::vector<dealii::types::global_dof_index>
    //     local_dof_indices(n_dofs_per_cell);
    //     cell->get_dof_indices(local_dof_indices);
    //     constraints.distribute_local_to_global(
    //       cell_matrix, cell_rhs, local_dof_indices, system_matrix,
    //       system_rhs);
    //   }
    // }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
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
