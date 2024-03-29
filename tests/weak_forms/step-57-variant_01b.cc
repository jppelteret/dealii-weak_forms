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

// Incompressible Navier-Stokes flow problem: Assembly using composite weak
// forms.
// This test replicates step-57 exactly.

#include <weak_forms/weak_forms.h>

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

    using namespace WeakForms;
    constexpr int spacedim = dim;

    // Symbolic types for test function, trial solution and a coefficient.
    const TestFunction<dim, spacedim>  test;
    const TrialSolution<dim, spacedim> trial;
    const FieldSolution<dim, spacedim> field_solution;
    const SubSpaceExtractors::Vector   subspace_extractor_v(0,
                                                          "v",
                                                          "\\mathbf{v}");
    const SubSpaceExtractors::Scalar   subspace_extractor_p(spacedim,
                                                          "p_tilde",
                                                          "\\tilde{p}");

    // Test function (subspaces)
    const auto test_v      = test[subspace_extractor_v].value();
    const auto div_test_v  = test[subspace_extractor_v].divergence();
    const auto grad_test_v = test[subspace_extractor_v].gradient();
    const auto test_p      = test[subspace_extractor_p].value();

    // Trial solution (subspaces)
    const auto trial_v      = trial[subspace_extractor_v].value();
    const auto div_trial_v  = trial[subspace_extractor_v].divergence();
    const auto grad_trial_v = trial[subspace_extractor_v].gradient();
    const auto trial_p      = trial[subspace_extractor_p].value();

    // Field solution
    const auto v      = field_solution[subspace_extractor_v].value();
    const auto div_v  = field_solution[subspace_extractor_v].divergence();
    const auto grad_v = field_solution[subspace_extractor_v].gradient();
    const auto p      = field_solution[subspace_extractor_p].value();

    // Constitutive parameters
    const auto &viscosity = this->viscosity;
    const auto &gamma     = this->gamma;

    // Assembly
    MatrixBasedAssembler<dim> assembler;

    // Cell LHS to assemble:
    //   viscosity * scalar_product(grad_phi_u[j], grad_phi_u[i])
    // + present_velocity_gradients[q] * phi_u[j] * phi_u[i]
    // + grad_phi_u[j] * present_velocity_values[q] * phi_u[i]
    // - div_phi_u[i] * phi_p[j]
    // - phi_p[i] * div_phi_u[j]
    // + gamma * div_phi_u[j] * div_phi_u[i]
    // + phi_p[i] * phi_p[j]
    assembler += bilinear_form(grad_test_v, viscosity, grad_trial_v).dV() +
                 bilinear_form(test_v, grad_v, trial_v).dV() +
                 bilinear_form(test_v, v, transpose(grad_trial_v)).dV() -
                 bilinear_form(div_test_v, 1, trial_p).dV() -
                 bilinear_form(test_p, 1, div_trial_v).dV() +
                 bilinear_form(div_test_v, gamma, div_trial_v).dV() +
                 bilinear_form(test_p, 1, trial_p).dV();

    // Cell RHS to assemble:
    // - viscosity * scalar_product(present_velocity_gradients[q],grad_phi_u[i])
    // - present_velocity_gradients[q] * present_velocity_values[q] * phi_u[i]
    // + present_pressure_values[q] * div_phi_u[i]
    // + present_velocity_divergence * phi_p[i]
    // - gamma * present_velocity_divergence * div_phi_u[i]
    assembler -=
      -linear_form(grad_test_v, constant_scalar<dim>(viscosity) * grad_v).dV() -
      linear_form(test_v, grad_v * v).dV() + linear_form(div_test_v, p).dV() +
      linear_form(test_p, div_v).dV() -
      linear_form(div_test_v, constant_scalar<dim>(gamma) * div_v).dV();

    // Look at what we're going to compute
    const SymbolicDecorations decorator;
    static bool               output = true;
    if (output)
      {
        deallog << "\n" << std::endl;
        deallog << "Weak form (ascii):\n"
                << assembler.as_ascii(decorator) << std::endl;
        deallog << "Weak form (LaTeX):\n"
                << assembler.as_latex(decorator) << std::endl;
        deallog << "\n" << std::endl;
        output = false;
      }

    // Now we pass in concrete objects to get data from
    // and assemble into.
    const AffineConstraints<double> &constraints_used =
      initial_step ? this->nonzero_constraints : this->zero_constraints;
    const QGauss<dim> quadrature_formula(this->degree + 2);
    const auto &      solution_vector = this->evaluation_point;
    if (assemble_matrix)
      {
        assembler.assemble_system(this->system_matrix,
                                  this->system_rhs,
                                  solution_vector,
                                  constraints_used,
                                  this->dof_handler,
                                  quadrature_formula);

        this->pressure_mass_matrix.reinit(this->sparsity_pattern.block(1, 1));
        this->pressure_mass_matrix.copy_from(this->system_matrix.block(1, 1));

        this->system_matrix.block(1, 1) = 0;
      }
    else
      {
        assembler.assemble_rhs_vector(this->system_rhs,
                                      solution_vector,
                                      constraints_used,
                                      this->dof_handler,
                                      quadrature_formula);
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
