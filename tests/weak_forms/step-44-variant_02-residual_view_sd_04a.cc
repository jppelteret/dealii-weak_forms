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

// Finite strain elasticity problem: Assembly using self-linearizing residual
// weak form in conjunction with symbolic differentiation.
// The residual view form is recovered directly from a symbolic function.
// This test replicates step-44 exactly.
// - Optimizer type: LLVM
// - Optimization method: All
// - Parameter file: parameters-step-44-refined_short.prm
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


  template <int dim>
  void
  Step44<dim>::assemble_system(const BlockVector<double> &solution_delta)
  {
    using namespace WeakForms;
    using namespace Differentiation;

    this->timer.enter_subsection("Assemble system");
    std::cout << " ASM_SYS " << std::flush;
    this->tangent_matrix = 0.0;
    this->system_rhs     = 0.0;
    const BlockVector<double> solution_total(
      this->get_total_solution(solution_delta));

    constexpr int spacedim = dim;
    using SDNumber_t       = Differentiation::SD::Expression;

    constexpr Differentiation::SD::OptimizerType optimizer_type =
      Differentiation::SD::OptimizerType::llvm;
    constexpr Differentiation::SD::OptimizationFlags optimization_flags =
      Differentiation::SD::OptimizationFlags::optimize_default;

    // Symbolic types for test function, and the field solution.
    const TestFunction<dim, spacedim>  test;
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
    const auto Grad_test_u = test_ss_u.gradient();
    const auto test_p      = test_ss_p.value();
    const auto test_J      = test_ss_J.value();

    // Field solution (subspaces)
    const auto u       = field_solution[subspace_extractor_u].value();
    const auto Grad_u  = field_solution[subspace_extractor_u].gradient();
    const auto p_tilde = field_solution[subspace_extractor_p].value();
    const auto J_tilde = field_solution[subspace_extractor_J].value();

    const auto I =
      constant_symmetric_tensor<spacedim>(unit_symmetric_tensor<spacedim>(),
                                          "I",
                                          "I");
    const auto F = I + Grad_u;

    // Geometry
    const Normal<spacedim> normal{};
    const auto             N = normal.value();

    // Residual
    const auto residual_func_u = residual_functor("R", "R", Grad_u, p_tilde);
    // const auto residual_func_p = residual_functor("R", "R", Grad_u, J_tilde);
    const auto residual_func_J = residual_functor("R", "R", p_tilde, J_tilde);
    const auto residual_ss_u   = residual_func_u[Grad_test_u];
    // const auto residual_ss_p   = residual_func_p[test_p];
    const auto residual_ss_J = residual_func_J[test_J];

    // Instead of re-rewriting the kinetic variables in full (as was done for
    // the energy density in step-44-variant_02-energy_functional_sd_01), we'll
    // cheat and fetch the definition from a prototypical QP.
    const auto &cell = this->dof_handler_ref.begin_active();
    const auto &qph  = this->quadrature_point_history;
    const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
      qph.get_data(cell);
    const auto &lqph_q_point = lqph[0];

    const auto residual_u =
      residual_ss_u.template value<SDNumber_t, dim, spacedim>(
        [lqph_q_point, &spacedim](const Tensor<2, spacedim, SDNumber_t> &Grad_u,
                                  const SDNumber_t &p_tilde)
        {
          const Tensor<2, spacedim, SDNumber_t> F =
            Grad_u + Physics::Elasticity::StandardTensors<dim>::I;
          const Tensor<2, spacedim, SDNumber_t> P =
            lqph_q_point->get_P(F, p_tilde);
          return P;
        },
        [](const Tensor<2, spacedim, SDNumber_t> &Grad_u,
           const SDNumber_t &                     p_tilde)
        {
          // Due to our shortcut, we've not made the constitutive
          // parameters symbolic.
          return Differentiation::SD::types::substitution_map{};
        },
        [](const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
           const std::vector<SolutionExtractionData<dim, spacedim>>
             &                solution_extraction_data,
           const unsigned int q_point)
        { return Differentiation::SD::types::substitution_map{}; },
        optimizer_type,
        optimization_flags,
        UpdateFlags::update_default);

    const auto residual_p = determinant(F) - J_tilde;

    const auto residual_J =
      residual_ss_J.template value<SDNumber_t, dim, spacedim>(
        [lqph_q_point](const SDNumber_t &p_tilde, const SDNumber_t &J_tilde)
        {
          const SDNumber_t dPsi_vol_dJ = lqph_q_point->get_dPsi_vol_dJ(J_tilde);
          const SDNumber_t dPsi_vol_dJ_minus_p_tilde = dPsi_vol_dJ - p_tilde;
          return dPsi_vol_dJ_minus_p_tilde;
        },
        [](const SDNumber_t &p_tilde, const SDNumber_t &J_tilde)
        {
          // Due to our shortcut, we've not made the constitutive
          // parameters symbolic.
          return Differentiation::SD::types::substitution_map{};
        },
        [](const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
           const std::vector<SolutionExtractionData<dim, spacedim>>
             &                solution_extraction_data,
           const unsigned int q_point)
        { return Differentiation::SD::types::substitution_map{}; },
        optimizer_type,
        optimization_flags,
        UpdateFlags::update_default);

    // Field variables: External force
    static const double p0 =
      -4.0 / (this->parameters.scale * this->parameters.scale);
    const double time_ramp = (this->time.current() / this->time.end());
    const double pressure  = p0 * this->parameters.p_p0 * time_ramp;

    const auto force_u =
      constant_scalar<spacedim>(pressure, "p_ext", "p^{\\text{ext}}") * N;

    // Boundary conditions
    const dealii::types::boundary_id traction_boundary_id = 6;

    // Assembly
    MatrixBasedAssembler<dim> assembler(ad_sd_cache);
    assembler +=
      residual_form(residual_u).dV() +
      residual_view_form<dim, spacedim>("R", "R", test_p, residual_p).dV() +
      residual_form(residual_J).dV() -
      linear_form(test_u, force_u).dA(traction_boundary_id);

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
      Step44::Step44<dim> solid(SOURCE_DIR
                                "/prm/parameters-step-44-refined_short.prm");
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
