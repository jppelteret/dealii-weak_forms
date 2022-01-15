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


// Wave equation: Assembly using composite weak forms
// This test replicates step-23.

#include <weak_forms/weak_forms.h>

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
    using namespace WeakForms;
    constexpr int spacedim = dim;

    this->matrix_u   = 0;
    this->system_rhs = 0;

    // Symbolic types for test function, trial solution and a coefficient.
    const TestFunction<dim, spacedim>  test;
    const TrialSolution<dim, spacedim> trial;
    const FieldSolution<dim, spacedim> field_solution;
    const SubSpaceExtractors::Scalar   subspace_extractor_u(0,
                                                          extractor,
                                                          "u",
                                                          "u");
    const SubSpaceExtractors::Scalar   subspace_extractor_v(0,
                                                          extractor,
                                                          "v",
                                                          "v");

    // Test function (subspaced)
    const auto test_ss_u   = test[subspace_extractor_u];
    const auto test_u      = test_ss_u.value();
    const auto grad_test_u = test_ss_u.gradient();

    // Trial solution (subspaced)
    const auto trial_ss_u   = trial[subspace_extractor_u];
    const auto trial_u      = trial_ss_u.value();
    const auto grad_trial_u = trial_ss_u.gradient();

    // Create storage for the solution vectors that may be referenced
    // by the weak forms
    const SolutionStorage<Vector<double>> solution_storage(
      {&this->solution_u,
       &this->old_solution_u,
       &this->solution_v,
       &this->old_solution_v});

    // Field solution (subspaced)
    constexpr WeakForms::types::solution_index solution_index_u    = 0;
    constexpr WeakForms::types::solution_index solution_index_u_t1 = 1;
    constexpr WeakForms::types::solution_index solution_index_v    = 2;
    constexpr WeakForms::types::solution_index solution_index_v_t1 = 3;

    const auto u_t1 = field_solution[subspace_extractor_u]
                        .template value<solution_index_u_t1>();
    const auto grad_u_t1 = field_solution[subspace_extractor_u]
                             .template gradient<solution_index_u_t1>();
    const auto v_t1 = field_solution[subspace_extractor_v]
                        .template value<solution_index_v_t1>();

    // Field variables
    const ScalarFunctionFunctor<dim> rhs_coeff("f", "f");
    const ScalarFunctionFunctor<dim> rhs_coeff_t1("f_t1", "f_{t-1}");
    RightHandSide<dim>               rhs_function;
    RightHandSide<dim>               rhs_function_t1;
    rhs_function.set_time(this->time);
    rhs_function_t1.set_time(this->time - this->time_step);
    const auto rhs    = rhs_coeff.value(rhs_function);
    const auto rhs_t1 = rhs_coeff_t1.value(rhs_function_t1);

    // Assembly
    MatrixBasedAssembler<dim> assembler;
    assembler += bilinear_form(test_u, 1.0, trial_u).dV() +
                 bilinear_form(grad_test_u,
                               this->theta * this->theta * this->time_step *
                                 this->time_step,
                               grad_trial_u)
                   .dV();
    assembler -= linear_form(test_u, u_t1).dV() +
                 linear_form(test_u, this->time_step * v_t1).dV() -
                 linear_form(grad_test_u,
                             (this->theta * (1 - this->theta) *
                              this->time_step * this->time_step) *
                               grad_u_t1)
                   .dV();
    assembler -=
      linear_form(test_u, (this->theta * this->time_step) * rhs).dV() +
      linear_form(test_u,
                  (this->theta * this->time_step) * (1 - this->theta) *
                    this->time_step * rhs_t1)
        .dV();

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
    assembler.assemble_system(this->matrix_u,
                              this->system_rhs,
                              solution_storage,
                              constraints,
                              this->dof_handler,
                              this->quadrature);
  }

  template <int dim>
  void
  Step23<dim>::assemble_v()
  {
    using namespace WeakForms;
    constexpr int spacedim = dim;

    this->matrix_v   = 0;
    this->system_rhs = 0;

    // Symbolic types for test function, trial solution and a coefficient.
    const TestFunction<dim, spacedim>  test;
    const TrialSolution<dim, spacedim> trial;
    const FieldSolution<dim, spacedim> field_solution;
    const SubSpaceExtractors::Scalar   subspace_extractor_u(0,
                                                          extractor,
                                                          "u",
                                                          "u");
    const SubSpaceExtractors::Scalar   subspace_extractor_v(0,
                                                          extractor,
                                                          "v",
                                                          "v");

    // Test function (subspaced)
    const auto test_ss_v   = test[subspace_extractor_v];
    const auto test_v      = test_ss_v.value();
    const auto grad_test_v = test_ss_v.gradient();

    // Trial solution (subspaced)
    const auto trial_ss_v   = trial[subspace_extractor_v];
    const auto trial_v      = trial_ss_v.value();
    const auto grad_trial_v = trial_ss_v.gradient();

    // Create storage for the solution vectors that may be referenced
    // by the weak forms
    const SolutionStorage<Vector<double>> solution_storage(
      {&this->solution_u,
       &this->old_solution_u,
       &this->solution_v,
       &this->old_solution_v});

    // Field solution (subspaced)
    constexpr WeakForms::types::solution_index solution_index_u    = 0;
    constexpr WeakForms::types::solution_index solution_index_u_t1 = 1;
    constexpr WeakForms::types::solution_index solution_index_v    = 2;
    constexpr WeakForms::types::solution_index solution_index_v_t1 = 3;

    const auto grad_u = field_solution[subspace_extractor_u]
                          .template gradient<solution_index_u>();
    const auto grad_u_t1 = field_solution[subspace_extractor_u]
                             .template gradient<solution_index_u_t1>();
    const auto v_t1 = field_solution[subspace_extractor_v]
                        .template value<solution_index_v_t1>();

    // Field variables
    const ScalarFunctionFunctor<dim> rhs_coeff("f", "f");
    const ScalarFunctionFunctor<dim> rhs_coeff_t1("f_t1", "f_{t-1}");
    RightHandSide<dim>               rhs_function;
    RightHandSide<dim>               rhs_function_t1;
    rhs_function.set_time(this->time);
    rhs_function_t1.set_time(this->time - this->time_step);
    const auto rhs    = rhs_coeff.value(rhs_function);
    const auto rhs_t1 = rhs_coeff_t1.value(rhs_function_t1);

    // Assembly
    MatrixBasedAssembler<dim> assembler;
    assembler += bilinear_form(test_v, 1.0, trial_v).dV();
    assembler -=
      -linear_form(grad_test_v, (this->theta * this->time_step) * grad_u).dV() +
      linear_form(test_v, v_t1).dV() -
      linear_form(grad_test_v,
                  (this->time_step * (1 - this->theta)) * grad_u_t1)
        .dV();
    assembler -=
      linear_form(test_v, (this->theta * this->time_step) * rhs).dV() +
      linear_form(test_v, (1 - this->theta) * this->time_step * rhs_t1).dV();

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
    assembler.assemble_system(this->matrix_v,
                              this->system_rhs,
                              solution_storage,
                              constraints,
                              this->dof_handler,
                              this->quadrature);
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
