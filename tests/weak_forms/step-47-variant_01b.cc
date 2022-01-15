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

// Biharmonic problem: Assembly using composite weak forms.
// This test replicates step-47 exactly.

#include <weak_forms/weak_forms.h>

#include "../weak_forms_tests.h"
#include "wf_common_tests/step-47.h"

namespace Step47
{
  template <int dim>
  class Step47 : public BiharmonicProblem<dim>
  {
  public:
    Step47(const unsigned int degree)
      : BiharmonicProblem<dim>(degree)
    {}

  protected:
    void
    assemble_system() override;
  };


  template <int dim>
  void
  Step47<dim>::assemble_system()
  {
    using namespace WeakForms;
    constexpr int spacedim = dim;

    // Symbolic types for test function, trial solution and a coefficient.
    const TestFunction<dim, spacedim>  test;
    const TrialSolution<dim, spacedim> trial;

    // Test function
    const auto test_value         = test.value();
    const auto test_gradient      = test.gradient();
    const auto test_hessian       = test.hessian();
    const auto test_ave_hessian   = test.average_of_hessians();
    const auto test_jump_gradient = test.jump_in_gradients();

    // Trial solution
    const auto trial_gradient      = trial.gradient();
    const auto trial_hessian       = trial.hessian();
    const auto trial_ave_hessian   = trial.average_of_hessians();
    const auto trial_jump_gradient = trial.jump_in_gradients();

    // Boundaries and interfaces
    const Normal<spacedim> normal{};
    const auto             N = normal.value();

    // Functions
    const ExactSolution::RightHandSide<dim> right_hand_side;
    const ScalarFunctionFunctor<spacedim>   rhs_function(
      "f(x)", "f\\left(\\mathbf{X}\\right)");

    const ExactSolution::Solution<dim>    exact_solution;
    const ScalarFunctionFunctor<spacedim> exact_solution_function(
      "u(x)", "u\\left(\\mathbf{X}\\right)");

    const ScalarFunctor gamma_over_h_functor("gamma/h", "\\frac{\\gamma}{h}");
    const auto          gamma_over_h =
      gamma_over_h_functor.template value<double, dim, spacedim>(
        [this](const FEValuesBase<dim, spacedim> &fe_values,
               const unsigned int                 q_point)
        {
          Assert((dynamic_cast<const FEFaceValuesBase<dim, spacedim> *const>(
                   &fe_values)),
                 ExcMessage("Cannot cast to FEFaceValues."));
          const auto &fe_face_values =
            static_cast<const FEFaceValuesBase<dim, spacedim> &>(fe_values);

          const auto &cell = fe_face_values.get_cell();
          const auto  f    = fe_face_values.get_face_number();

          const unsigned int p = fe_face_values.get_fe().degree;
          const double       gamma_over_h =
            (1.0 * p * (p + 1) /
             cell->extent_in_direction(
               GeometryInfo<dim>::unit_normal_direction[f]));

          return gamma_over_h;
        },
        [this](const FEInterfaceValues<dim, spacedim> &fe_interface_values,
               const unsigned int                      q_point)
        {
          Assert(fe_interface_values.at_boundary() == false,
                 ExcInternalError());

          const auto cell =
            fe_interface_values.get_fe_face_values(0).get_cell();
          const auto ncell =
            fe_interface_values.get_fe_face_values(1).get_cell();
          const auto f =
            fe_interface_values.get_fe_face_values(0).get_face_number();
          const auto nf =
            fe_interface_values.get_fe_face_values(1).get_face_number();

          const unsigned int p = fe_interface_values.get_fe().degree;
          const double       gamma_over_h =
            std::max((1.0 * p * (p + 1) /
                      cell->extent_in_direction(
                        GeometryInfo<dim>::unit_normal_direction[f])),
                     (1.0 * p * (p + 1) /
                      ncell->extent_in_direction(
                        GeometryInfo<dim>::unit_normal_direction[nf])));

          return gamma_over_h;
        });

    // Assembly
    MatrixBasedAssembler<dim> assembler;

    // Cell LHS to assemble:
    //   (nabla^2 phi_i(x) * nabla^2 phi_j(x)).dV
    // - ({grad^2 v n n} * [grad u n]).dI
    // - ({grad^2 u n n} * [grad v n]).dI
    // + (gamma/h [grad v n] * [grad u n]).dI
    // - ({grad^2 v n n} * [grad u n]).dA
    // - ({grad^2 u n n} * [grad v n]).dA
    // + (gamma/h [grad v n] * [grad u n]).dA
    assembler +=
      bilinear_form(test_hessian, 1.0, trial_hessian).dV() -
      bilinear_form(N * test_ave_hessian * N, 1.0, trial_jump_gradient * N)
        .dI() -
      bilinear_form(test_jump_gradient * N, 1.0, N * trial_ave_hessian * N)
        .dI() +
      bilinear_form(test_jump_gradient * N,
                    gamma_over_h,
                    trial_jump_gradient * N)
        .dI() -
      bilinear_form(N * test_hessian * N, 1.0, trial_gradient * N).dA() -
      bilinear_form(test_gradient * N, 1.0, N * trial_hessian * N).dA() +
      bilinear_form(test_gradient * N, gamma_over_h, trial_gradient * N).dA();

    // Cell RHS to assemble:
    //   (phi_i(x) * f(x)).dV
    // - ({grad^2 v n n} * (grad u_exact . n)).dA
    // + (gamma/h [grad v n] * (grad u_exact . n)).dA
    assembler -=
      linear_form(test_value, rhs_function.value(right_hand_side)).dV() -
      linear_form(N * test_hessian * N,
                  exact_solution_function.gradient(exact_solution) * N)
        .dA() +
      linear_form(gamma_over_h * test_gradient * N,
                  exact_solution_function.gradient(exact_solution) * N)
        .dA();

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
    const unsigned int quadrature_degree =
      this->dof_handler.get_fe().degree + 1;
    const QGauss<dim>     cell_quadrature(quadrature_degree);
    const QGauss<dim - 1> face_quadrature(quadrature_degree);
    assembler.assemble_system(this->system_matrix,
                              this->system_rhs,
                              this->constraints,
                              this->dof_handler,
                              cell_quadrature,
                              face_quadrature);
  }

} // namespace Step47

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
      const unsigned int fe_degree                 = 2;
      const unsigned int n_local_refinement_levels = 4;

      Assert(fe_degree >= 2,
             ExcMessage("The C0IP formulation for the biharmonic problem "
                        "only works if one uses elements of polynomial "
                        "degree at least 2."));

      Step47::Step47<dim> biharmonic_problem(fe_degree);
      biharmonic_problem.run(n_local_refinement_levels);
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
