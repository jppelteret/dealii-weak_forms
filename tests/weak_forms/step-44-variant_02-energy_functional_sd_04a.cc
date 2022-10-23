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

// Finite strain elasticity problem: Assembly using self-linearizing energy
// functional weak form in conjunction with symbolic differentiation. The
// internal energy is calculated by hand (not retrieved from LQPH), and the
// external energy is also supplied.
// The energy functional form is recovered directly from a symbolic function.
// This test replicates step-44 exactly.
// - Optimizer type: LLVM
// - Optimization method: All
// - AD/SD Cache

#include <deal.II/differentiation/sd.h>

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
    WeakForms::AD_SD_Functor_Cache ad_sd_cache;

    void
    assemble_system(const BlockVector<double> &solution_delta) override;
  };


// Warning due to an unnecessary lambda capture, but if the
// capture is removed then we get a compilation error.
#pragma GCC diagnostic push
#if defined(__clang__)
#  pragma GCC diagnostic ignored "-Wunused-lambda-capture"
#endif

  template <int dim>
  void
  Step44<dim>::assemble_system(const BlockVector<double> &solution_delta)
  {
    using namespace WeakForms;
    using namespace Differentiation;

    constexpr int spacedim = dim;
    using SDNumber_t       = Differentiation::SD::Expression;

    constexpr Differentiation::SD::OptimizerType optimizer_type =
      Differentiation::SD::OptimizerType::llvm;
    constexpr Differentiation::SD::OptimizationFlags optimization_flags =
      Differentiation::SD::OptimizationFlags::optimize_default;

    this->timer.enter_subsection("Assemble system");
    std::cout << " ASM_SYS " << std::flush;
    this->tangent_matrix = 0.0;
    this->system_rhs     = 0.0;
    const BlockVector<double> solution_total(
      this->get_total_solution(solution_delta));

    // Symbolic types for test function, and the field solution.
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

    // Field solution (subspaces)
    const auto u       = field_solution[subspace_extractor_u].value();
    const auto Grad_u  = field_solution[subspace_extractor_u].gradient();
    const auto p_tilde = field_solution[subspace_extractor_p].value();
    const auto J_tilde = field_solution[subspace_extractor_J].value();

    const auto I =
      constant_symmetric_tensor<spacedim>(unit_symmetric_tensor<spacedim>(),
                                          "I",
                                          "I");
    const auto F     = I + Grad_u;
    const auto det_F = determinant(F);
    const auto C     = symmetrize(transpose(F) * F);
    const auto C_bar = pow(det_F, -2.0 / spacedim) * C;

    // Geometry
    const Normal<spacedim> normal{};
    const auto             N = normal.value();

    // Internal energy
    const double mu    = this->parameters.mu;
    const double nu    = this->parameters.nu;
    const double kappa = (2.0 * mu * (1.0 + nu)) / (3.0 * (1.0 - 2.0 * nu));
    const double c_1   = mu / 2.0;

    // - Isochoric part
    const auto psi_CpJ_iso =
      c_1 * (trace(C_bar) - constant_scalar<spacedim>(spacedim));
    // - Volumetric part
    const auto psi_CpJ_vol =
      (kappa / 4.0) * (J_tilde * J_tilde - 1.0 - 2.0 * log(J_tilde));
    // Penalisation term
    const auto psi_CpJ_pen = p_tilde * (det_F - J_tilde);

    const auto internal_energy = psi_CpJ_iso + psi_CpJ_vol + psi_CpJ_pen;

    // External energy
    static const double p0 =
      -4.0 / (this->parameters.scale * this->parameters.scale);
    const double time_ramp = (this->time.current() / this->time.end());
    const double pressure  = p0 * this->parameters.p_p0 * time_ramp;

    // The pressure changes at each time step, but the Sd cache keeps the
    // original value / symbol. So we now have to explicitly make this symbolic
    // so that there is not a clash when its evaluated after the first timestep.
    const auto external_energy =
      -u *
      (constant_scalar<spacedim>(pressure, "p_ext", "p^{\\text{ext}}") * N);

    // Boundary conditions
    const dealii::types::boundary_id traction_boundary_id = 6;

    // Assembly
    MatrixBasedAssembler<dim> assembler(ad_sd_cache);
    assembler += energy_functional_form<dim, spacedim>("e^{int}",
                                                       "\\Psi^{\text{int}}",
                                                       internal_energy)
                   .dV() +
                 energy_functional_form<dim, spacedim>("e^{ext}",
                                                       "\\Psi^{\text{ext}}",
                                                       external_energy)
                   .dA(traction_boundary_id);
    assembler.symmetrize();

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


#pragma GCC diagnostic pop // "-Wunused-lambda-capture"
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
