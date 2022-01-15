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

// The Symmetric interior penalty Galerkin (SIPG) method for Poisson's
// equation: Assembly using composite weak forms
// This test replicates step-74 exactly.

#include <weak_forms/weak_forms.h>

#include "../weak_forms_tests.h"
#include "wf_common_tests/step-74.h"

namespace Step74
{
  template <int dim>
  class Step74 : public SIPGLaplace<dim>
  {
    using ScratchData = typename SIPGLaplace<dim>::ScratchData;

  public:
    Step74(const TestCase &test_case)
      : SIPGLaplace<dim>(test_case)
    {}

  protected:
    void
    assemble_system() override;
  };


  template <int dim>
  void
  Step74<dim>::assemble_system()
  {
    using namespace WeakForms;
    constexpr int spacedim = dim;

    // Symbolic types for test function, trial solution and a coefficient.
    const TestFunction<dim, spacedim>  test;
    const TrialSolution<dim, spacedim> trial;

    // Test function
    const auto test_value        = test.value();
    const auto test_gradient     = test.gradient();
    const auto test_ave_gradient = test.average_of_gradients();
    const auto test_jump_values  = test.jump_in_values();

    // Trial solution
    const auto trial_value        = trial.value();
    const auto trial_gradient     = trial.gradient();
    const auto trial_ave_gradient = trial.average_of_gradients();
    const auto trial_jump_values  = trial.jump_in_values();

    // Boundaries and interfaces
    const Normal<spacedim> normal{};
    const auto             N = normal.value();

    // Functions
    const ScalarFunctor diffusion_coeff("nu", "\\nu");
    const auto nu = diffusion_coeff.template value<double, dim, spacedim>(
      [this](const FEValuesBase<dim, spacedim> &, const unsigned int)
      { return this->diffusion_coefficient; },
      [this](const FEInterfaceValues<dim, spacedim> &, const unsigned int)
      { return this->diffusion_coefficient; });

    const ScalarFunctionFunctor<spacedim> right_hand_side(
      "f(x)", "f\\left(\\mathbf{X}\\right)");
    const auto rhs_value = right_hand_side.value(*this->rhs_function);

    const ScalarFunctionFunctor<spacedim> exact_solution_function(
      "u(x)", "u\\left(\\mathbf{X}\\right)");
    const auto exact_solution_value =
      exact_solution_function.value(*this->exact_solution);

    const ScalarFunctor sigma_functor("sigma",
                                      "\\sigma\\left(\\mathbf{X}\\right)");
    const auto sigma = sigma_functor.template value<double, dim, spacedim>(
      [this](const FEValuesBase<dim, spacedim> &fe_values,
             const unsigned int                 q_point)
      {
        Assert((dynamic_cast<const FEFaceValuesBase<dim, spacedim> *const>(
                 &fe_values)),
               ExcMessage("Cannot cast to FEFaceValues."));
        const auto &fe_face_values =
          static_cast<const FEFaceValuesBase<dim, spacedim> &>(fe_values);

        const auto &cell    = fe_face_values.get_cell();
        const auto  face_no = fe_face_values.get_face_number();

        const double extent1 = cell->measure() / cell->face(face_no)->measure();
        const double penalty =
          get_penalty_factor(this->degree, extent1, extent1);
        return penalty;
      },
      [this](const FEInterfaceValues<dim, spacedim> &fe_interface_values,
             const unsigned int                      q_point)
      {
        Assert(fe_interface_values.at_boundary() == false, ExcInternalError());

        const auto cell  = fe_interface_values.get_fe_face_values(0).get_cell();
        const auto ncell = fe_interface_values.get_fe_face_values(1).get_cell();
        const auto f =
          fe_interface_values.get_fe_face_values(0).get_face_number();
        const auto nf =
          fe_interface_values.get_fe_face_values(1).get_face_number();

        const double extent1 = cell->measure() / cell->face(f)->measure();
        const double extent2 = ncell->measure() / ncell->face(nf)->measure();
        const double penalty =
          get_penalty_factor(this->degree, extent1, extent2);

        return penalty;
      });

    // Assembly
    MatrixBasedAssembler<dim> assembler;

    // Cell LHS to assemble:
    //   (grad v_h * nu * grad u_h).dV
    // - ([v_h] * nu * ( (grad u_h) . n)).dI
    // - ( ((grad v_h) . n) * nu * [u_h]).dI
    // + ([v_h] * (nu * sigma) * [u_h]).dI
    // - (v_h * nu * (grad u_h . n)).dA
    // - ((grad v_h . n) * nu * u_h).dA
    // + (v_h * (nu * sigma) * u_h).dA
    assembler +=
      bilinear_form(test_gradient, nu, trial_gradient).dV() -
      bilinear_form(test_jump_values, nu, trial_ave_gradient * N).dI() -
      bilinear_form(test_ave_gradient * N, nu, trial_jump_values).dI() +
      bilinear_form(test_jump_values, nu * sigma, trial_jump_values).dI() -
      bilinear_form(test_value, nu, trial_gradient * N).dA() -
      bilinear_form(test_gradient * N, nu, trial_value).dA() +
      bilinear_form(test_value, nu * sigma, trial_value).dA();

    // Cell RHS to assemble:
    //   (v_h * f).dV
    // - ((grad v_h . n) * (nu * u_exact)).dA
    // + (v_h * (nu * sigma * u_exact)).dA
    assembler -=
      linear_form(test_value, rhs_value).dV() -
      linear_form(test_gradient * N, nu * exact_solution_value).dA() +
      linear_form(test_value, nu * sigma * exact_solution_value).dA();

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
    AffineConstraints<double> constraints;
    constraints.close();
    assembler.assemble_system(this->system_matrix,
                              this->system_rhs,
                              constraints,
                              this->dof_handler,
                              this->quadrature,
                              this->face_quadrature);
  }

} // namespace Step74

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
      const Step74::TestCase test_case = Step74::TestCase::l_singularity;
      const int              dim       = 2;

      Step74::Step74<dim> problem(test_case);
      problem.run();
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
