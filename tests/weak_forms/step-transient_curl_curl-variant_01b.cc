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
    AssertThrow(false, ExcPureFunctionCalled());
  }


  template <int dim>
  void
  StepTransientCurlCurl<dim>::assemble_system_mvp(
    TrilinosWrappers::SparseMatrix & system_matrix_mvp,
    TrilinosWrappers::MPI::Vector &  system_rhs_mvp,
    const AffineConstraints<double> &constraints_mvp)
  {
    using namespace WeakForms;
    constexpr int spacedim = dim;

    TimerOutput::Scope timer_scope(this->computing_timer, "Assembly: MVP");
    system_matrix_mvp = 0.0;
    system_rhs_mvp    = 0.0;

    // Symbolic types for test function, trial solution_mvp and a coefficient.
    const TestFunction<dim, spacedim>  test;
    const TrialSolution<dim, spacedim> trial;
    const FieldSolution<dim, spacedim> field_solution;
    const SubSpaceExtractors::Vector   subspace_extractor_A(0,
                                                          this->mvp_extractor,
                                                          "A",
                                                          "\\mathbf{A}");

    // Test function (subspaced)
    const auto test_A      = test[subspace_extractor_A].value();
    const auto test_curl_A = test[subspace_extractor_A].curl();

    // Test function (subspaced)
    const auto trial_A      = trial[subspace_extractor_A].value();
    const auto trial_curl_A = trial[subspace_extractor_A].curl();

    // Create storage for the solution_mvp vectors that may be referenced
    // by the weak forms
    using VectorType = TrilinosWrappers::MPI::Vector;
    const SolutionStorage<VectorType> solution_storage(
      {&this->solution_mvp, &this->solution_mvp_t1, &this->d_solution_mvp_dt});

    // Field solution
    constexpr WeakForms::types::solution_index solution_mvp_index = 0;
    const auto                                 curl_A =
      field_solution[subspace_extractor_A].template curl<solution_mvp_index>();

    constexpr WeakForms::types::solution_index d_solution_mvp_dt_index_t1 = 2;
    const auto dA_dt = field_solution[subspace_extractor_A]
                         .template value<d_solution_mvp_dt_index_t1>();

    // Functions
    const ScalarFunctionFunctor<spacedim> permeability("mu(x)", "\\mu");
    const ScalarFunctionFunctor<spacedim> conductivity("sigma(x)", "\\sigma");
    const VectorFunctionFunctor<spacedim> current_source("Jf(x)",
                                                         "\\mathbf{J}^{free}");

    const auto dt =
      constant_scalar<dim>(this->parameters.delta_t, "dt", "\\Delta t");
    const auto mu =
      permeability.value(this->function_material_permeability_coefficients);
    const auto nu = 1.0 / mu;
    const auto sigma =
      conductivity.value(this->function_material_conductivity_coefficients);
    const auto kappa =
      (this->parameters.regularisation_parameter / spacedim) * nu;
    const auto J_f = current_source.value(this->function_free_current_density);

    // Check current running through boundary
    const Normal<spacedim> normal{};
    const auto             N       = normal.value();
    const auto             J_dot_N = J_f * N;
    const double           I_total =
      WeakForms::Integrator<dim, decltype(J_dot_N)>(J_dot_N)
        .template dA<double>(
          this->dof_handler_mvp,
          this->qf_cell_mvp,
          this->qf_face_mvp,
          {this->parameters.bid_wire_inlet} /*input boundary*/);
    // std::cout << "I_total: " << I_total << std::endl;

    // Assembly
    MatrixBasedAssembler<dim> assembler;
    // assembler.symmetrize();

    // Common contributions
    assembler += linear_form(test_curl_A, nu * curl_A).dV() +
                 bilinear_form(test_curl_A, nu, trial_curl_A).symmetrize().dV();

    // Magneto-static region
    const auto &mat_air = this->parameters.mid_surroundings;
    assembler += bilinear_form(test_A, kappa, trial_A).symmetrize().dV(mat_air);

    // Transient / conducting magnetic region
    const auto &mat_wire = this->parameters.mid_wire;
    assembler +=
      linear_form(test_A, sigma * dA_dt).dV(mat_wire) +
      bilinear_form(test_A, sigma / dt, trial_A).symmetrize().dV(mat_wire);

    // Source term
    assembler -= linear_form(test_A, J_f).dV(mat_wire);

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
    assembler.assemble_system(system_matrix_mvp,
                              system_rhs_mvp,
                              solution_storage,
                              constraints_mvp,
                              this->dof_handler_mvp,
                              this->qf_cell_mvp);
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
