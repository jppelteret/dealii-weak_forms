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

// Finite strain elasticity problem: Assembly using composite weak forms
// This test replicates step-44 exactly.
//
// In this variant:
// - the shape function slots in the bilinear and linear forms
// have the solution field embedded in a binary operation. So the
// implementation takes place in the fully referential setting.
// - there is a local symmetric flag set on some of the bilinear form
// contributions

#include <weak_forms/weak_forms.h>

#include "../weak_forms_tests.h"
#include "wf_common_tests/step-44.h"

namespace Step44
{
  template <int dim>
  class Step44 : public Step44_Base<dim>
  {
  public:
    Step44(const std::string &input_file)
      : Step44_Base<dim>(input_file)
    {}

  protected:
    void
    assemble_system(const BlockVector<double> &solution_delta) override;
  };
  template <int dim>
  void
  Step44<dim>::assemble_system(const BlockVector<double> &solution_delta)
  {
    using namespace WeakForms;
    constexpr int spacedim = dim;

    this->timer.enter_subsection("Assemble system");
    std::cout << " ASM_SYS " << std::flush;
    this->tangent_matrix = 0.0;
    this->system_rhs     = 0.0;
    const BlockVector<double> solution_total(
      this->get_total_solution(solution_delta));

    // Symbolic types for test function, trial solution and a coefficient.
    const TestFunction<dim, spacedim>  test;
    const TrialSolution<dim, spacedim> trial;
    const FieldSolution<dim, spacedim> field_solution;
    const SubSpaceExtractors::Vector   subspace_extractor_u(0,
                                                          "u",
                                                          "\\mathbf{u}");
    const SubSpaceExtractors::Scalar   subspace_extractor_p(spacedim,
                                                          "p_tilde",
                                                          "\\tilde{p}");
    const SubSpaceExtractors::Scalar   subspace_extractor_J(spacedim + 1,
                                                          "J_tilde",
                                                          "\\tilde{J}");

    // Test function (subspaced)
    const auto test_ss_u = test[subspace_extractor_u];
    const auto test_ss_p = test[subspace_extractor_p];
    const auto test_ss_J = test[subspace_extractor_J];

    const auto test_u      = test_ss_u.value();
    const auto test_p      = test_ss_p.value();
    const auto test_J      = test_ss_J.value();
    const auto grad_test_u = test_ss_u.gradient();

    // Trial solution (subspaces)
    const auto trial_ss_u = trial[subspace_extractor_u];
    const auto trial_ss_p = trial[subspace_extractor_p];
    const auto trial_ss_J = trial[subspace_extractor_J];

    const auto grad_trial_u = trial_ss_u.gradient();
    const auto trial_p      = trial_ss_p.value();
    const auto trial_J      = trial_ss_J.value();

    // Field solution
    const auto grad_u  = field_solution[subspace_extractor_u].gradient();
    const auto p_tilde = field_solution[subspace_extractor_p].value();
    const auto J_tilde = field_solution[subspace_extractor_J].value();

    // Field variables
    const ScalarFunctor det_F_symb("det_F", "det(\\mathbf{F})");
    const ScalarFunctor dPsi_vol_dJ_symb("dPsi_vol_dJ",
                                         "\\frac{d \\Psi^{vol}(J)}{dJ}");
    const ScalarFunctor d2Psi_vol_dJ2_symb(
      "d2Psi_vol_dJ2", "\\frac{d^{2} \\Psi^{vol}(J)}{dJ^{2}}");
    const SymmetricTensorFunctor<2, spacedim> C_inv_symb("C_inv",
                                                         "\\mathbf{C}^{-1}");
    const SymmetricTensorFunctor<2, spacedim> I_symb(
      "I", "\\mathbf{I}"); // Identity tensor
    const SymmetricTensorFunctor<2, spacedim> S_symb(
      "S", "\\mathbf{S}"); // Piola Kirchhoff stress
    const SymmetricTensorFunctor<4, spacedim> H_tot_symb(
      "H^{tot}", "\\mathcal{H}^{tot}"); // Total elasticity tensor (mat + geo
                                        // contributions)

    const auto det_F = det_F_symb.template value<double, dim, spacedim>(
      [this](const FEValuesBase<dim, spacedim> &fe_values,
             const unsigned int                 q_point) {
        const auto &cell = fe_values.get_cell();
        const auto &qph  = this->quadrature_point_history;
        const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
          qph.get_data(cell);
        return lqph[q_point]->get_det_F();
      });
    const auto dPsi_vol_dJ =
      dPsi_vol_dJ_symb.template value<double, dim, spacedim>(
        [this](const FEValuesBase<dim, spacedim> &fe_values,
               const unsigned int                 q_point) {
          const auto &cell = fe_values.get_cell();
          const auto &qph  = this->quadrature_point_history;
          const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
            qph.get_data(cell);
          return lqph[q_point]->get_dPsi_vol_dJ();
        });
    const auto d2Psi_vol_dJ2 =
      d2Psi_vol_dJ2_symb.template value<double, dim, spacedim>(
        [this](const FEValuesBase<dim, spacedim> &fe_values,
               const unsigned int                 q_point) {
          const auto &cell = fe_values.get_cell();
          const auto &qph  = this->quadrature_point_history;
          const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
            qph.get_data(cell);
          return lqph[q_point]->get_d2Psi_vol_dJ2();
        });
    const auto C_inv = C_inv_symb.template value<double, dim>(
      [this](const FEValuesBase<dim, spacedim> &fe_values,
             const unsigned int                 q_point) {
        const auto &cell = fe_values.get_cell();
        const auto &qph  = this->quadrature_point_history;
        const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
          qph.get_data(cell);
        return lqph[q_point]->get_C_inv();
      });
    const auto I = I_symb.template value<double, dim>(
      [](const FEValuesBase<dim, spacedim> &fe_values, const unsigned int) {
        return Physics::Elasticity::StandardTensors<dim>::I;
      });
    const auto S = S_symb.template value<double, dim>(
      [this](const FEValuesBase<dim, spacedim> &fe_values,
             const unsigned int                 q_point) {
        const auto &cell = fe_values.get_cell();
        const auto &qph  = this->quadrature_point_history;
        const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
          qph.get_data(cell);
        return lqph[q_point]->get_S();
      });
    const auto H_tot = H_tot_symb.template value<double, dim>(
      [this](const FEValuesBase<dim, spacedim> &fe_values,
             const unsigned int                 q_point) {
        const auto &cell = fe_values.get_cell();
        const auto &qph  = this->quadrature_point_history;
        const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
          qph.get_data(cell);
        return lqph[q_point]->get_H_mat_geo();
      });

    // Boundary conditions
    const types::boundary_id traction_boundary_id = 6;
    const ScalarFunctor      p_symb("p", "p"); // Applied pressure
    const Normal<spacedim>   normal{};

    const auto p = p_symb.template value<double, dim, spacedim>(
      [this](const FEValuesBase<dim, spacedim> &, const unsigned int) {
        static const double p0 =
          -4.0 / (this->parameters.scale * this->parameters.scale);
        const double time_ramp = (this->time.current() / this->time.end());
        const double pressure  = p0 * this->parameters.p_p0 * time_ramp;
        return pressure;
      });
    const auto N = normal.value();

    // Variations and linearisations
    const auto F  = I + grad_u;
    const auto dF = grad_test_u;
    const auto DF = grad_trial_u;
    const auto dE = symmetrize(transpose(F) * dF);
    const auto DE = symmetrize(transpose(F) * DF);

    // Note: This is also valid, but produces a tensor in the LaTeX output
    // const auto F = grad_u + Physics::Elasticity::StandardTensors<dim>::I;

    // Standard approach to incorporating geometric stiffness
    // const SymmetricTensor<2, dim> DdE_IJ = symmetrize(transpose(Grad_Nx_u_J)
    // * Grad_Nx_u_I); data.cell_matrix(I, J) += (DdE_IJ * S) * JxW;

    // Assembly
    MatrixBasedAssembler<dim> assembler;
    assembler +=
      bilinear_form(dE, H_tot, DE).symmetrize().dV()                     // K_uu
      + bilinear_form(dE, det_F * C_inv, trial_p).dV()                   // K_up
      + bilinear_form(test_p, det_F * C_inv, DE).dV()                    // K_pu
      - bilinear_form(test_p, 1.0, trial_J).dV()                         // K_pJ
      - bilinear_form(test_J, 1.0, trial_p).dV()                         // K_Jp
      + bilinear_form(test_J, d2Psi_vol_dJ2, trial_J).symmetrize().dV(); // K_JJ
    assembler += linear_form(dE, S).dV()                                 // r_u
                 + linear_form(test_p, det_F - J_tilde).dV()             // r_p
                 + linear_form(test_J, dPsi_vol_dJ - p_tilde).dV();      // r_J
    assembler -= linear_form(test_u, N * p).dA(traction_boundary_id);    // f_u

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
    const QGauss<dim>     qf_cell(this->fe.degree + 1);
    const QGauss<dim - 1> qf_face(this->fe.degree + 1);
    assembler.assemble_system(this->tangent_matrix,
                              this->system_rhs,
                              solution_total,
                              this->constraints,
                              this->dof_handler_ref,
                              qf_cell,
                              qf_face);

    this->timer.leave_subsection();
  }
} // namespace Step44

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
      const unsigned int  dim = 3;
      Step44::Step44<dim> solid(SOURCE_DIR "/prm/parameters-step-44.prm");
      solid.run();
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
