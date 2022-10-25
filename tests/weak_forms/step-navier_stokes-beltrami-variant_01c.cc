/* $Id: NavierStokes-Beltrami.cc 2008-04-15 10:54:52CET martinkr $ */
/* Author: Martin Kronbichler, Uppsala University, 2008 */
/*    $Id: NavierStokes-Beltrami.cc 2008-04-15 10:54:52CET martinkr $ */
/*    Version: $Name$                                             */
/*                                                                */
/*    Copyright (C) 2008 by the author                            */
/*                                                                */
/*    This file is subject to QPL and may not be  distributed     */
/*    without copyright and license information. Please refer     */
/*    to the file deal.II/doc/license.html for the  text  and     */
/*    further information on this license.                        */

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

// This program implements an algorithm for the incompressible Navier-Stokes
// equations solving the whole system at once, i.e., without any projection
// equation.
// This test replicates step-navier_stokes-beltrami exactly.
//
// This variant has no stablisation enabled.
//
// The implementation follows non-linearized form described in the paper of
// Bazilevs:
//   Variational multiscale residual-based turbulence modeling for large eddy
//   simulation of incompressible flows
//   Linearisation:   Equation 73
//   Right hand side: Equation 72
// as opposed to the explicit formulas given in equations 101-108.


#include <weak_forms/weak_forms.h>

#include "../weak_forms_tests.h"
#include "wf_common_tests/step-navier_stokes-beltrami.h"

namespace StepNavierStokesBeltrami
{
  template <int dim>
  class NavierStokesProblem : public NavierStokesProblemBase<dim>
  {
  public:
    NavierStokesProblem()
      : NavierStokesProblemBase<dim>(0 /*stabilization*/)
    {}

  protected:
    void
    assemble_system() override;
  };

  template <int dim>
  void
  NavierStokesProblem<dim>::assemble_system()
  {
    using namespace WeakForms;
    constexpr int spacedim = dim;

    this->system_matrix = 0;
    this->system_rhs    = 0;

    // Symbolic types for test function, trial solution
    const TestFunction<dim, spacedim>  test;
    const TrialSolution<dim, spacedim> trial;
    const FieldSolution<dim, spacedim> field_solution;

    // Subspace extractors
    const SubSpaceExtractors::Vector subspace_extractor_v(0, 0, "v", "v");
    const SubSpaceExtractors::Scalar subspace_extractor_p(1, dim, "p", "p");

    // Test function (subspaced)
    const auto test_ss_v        = test[subspace_extractor_v];
    const auto test_ss_p        = test[subspace_extractor_p];
    const auto test_v           = test_ss_v.value();
    const auto grad_test_v      = test_ss_v.gradient();
    const auto symm_grad_test_v = test_ss_v.symmetric_gradient();
    const auto div_test_v       = test_ss_v.divergence();
    const auto test_p           = test_ss_p.value();

    // Trial solution (subspaced)
    const auto trial_ss_v        = trial[subspace_extractor_v];
    const auto trial_ss_p        = trial[subspace_extractor_p];
    const auto trial_v           = trial_ss_v.value();
    const auto grad_trial_v      = trial_ss_v.gradient();
    const auto symm_grad_trial_v = trial_ss_v.symmetric_gradient();
    const auto div_trial_v       = trial_ss_v.divergence();
    const auto trial_p           = trial_ss_p.value();

    // Create storage for the solution vectors that may be referenced
    // by the weak forms
    const SolutionStorage<Vector<double>> solution_storage(
      {&this->solution, &this->solution_old_scaled});

    // Field solution (subspaced)
    constexpr WeakForms::types::solution_index solution_index_v    = 0;
    constexpr WeakForms::types::solution_index solution_index_v_t1 = 1;

    const auto v =
      field_solution[subspace_extractor_v].template value<solution_index_v>();
    const auto v_t1 = field_solution[subspace_extractor_v]
                        .template value<solution_index_v_t1>();
    const auto div_v = field_solution[subspace_extractor_v]
                         .template divergence<solution_index_v>();

    // Constants
    const auto nu = constant_scalar<dim>(this->nu, "nu", "\\nu");
    const auto tau =
      constant_scalar<dim>(this->time_step_weight, "tau", "\\tau");

    // Functors
    const RightHandSideTF<dim>       rhs(this->time);
    const VectorFunctionFunctor<dim> rhs_coeff("s", "\\mathbf{s}");

    // Assembly
    MatrixBasedAssembler<dim> assembler;

    // Either of the two lines marked with a (*) could be used.
    // The first one produces exactly the same result as the parent
    // test. The second variant has a slightly different convergnece
    // history, but the result is still the same.
    assembler += bilinear_form(test_v, tau, trial_v).dV();
    // assembler -=
    //   bilinear_form(grad_test_v, outer_product(trial_v, v)).dV(); // (*)
    assembler -=
      bilinear_form(grad_test_v, outer_product(v, trial_v)).dV(); // (*)
    assembler +=
      bilinear_form(symm_grad_test_v, 2.0 * nu, symm_grad_trial_v).dV();
    assembler -= bilinear_form(div_test_v, trial_p).dV();
    assembler += bilinear_form(test_p, div_trial_v).dV();

    assembler -= linear_form(test_v, rhs_coeff.value(rhs) + v_t1).dV();

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
    const auto &      constraints = this->hanging_node_and_pressure_constraints;
    const QGauss<dim> quadrature_formula(this->degree + 2);
    assembler.assemble_system(this->system_matrix,
                              this->system_rhs,
                              solution_storage,
                              constraints,
                              this->dof_handler,
                              quadrature_formula);

    this->hanging_node_and_pressure_constraints.condense(this->system_matrix);
    this->hanging_node_and_pressure_constraints.condense(this->system_rhs);
  }

} // namespace StepNavierStokesBeltrami


int
main(int argc, char **argv)
{
  initlog();
  deallog << std::setprecision(9);

  deallog.depth_file(1);

  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  using namespace dealii;
  try
    {
      const int                                          dim = 2;
      StepNavierStokesBeltrami::NavierStokesProblem<dim> navier_stokes_problem;
      navier_stokes_problem.run();
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
