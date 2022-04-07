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
// This test replicates step-44 exactly.
// This test follows the implementation of
// step-44-variant_02-residual_view_ad_02.
// - Optimizer type: Lambda
// - Optimization method: All
// - AD/SD Cache
// - Parameter file: parameters-step-44.prm

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
      , assembler(ad_sd_cache)
    {}

  protected:
    WeakForms::AD_SD_Functor_Cache       ad_sd_cache;
    WeakForms::MatrixBasedAssembler<dim> assembler;

    void
    build_assembler();

    void
    assemble_system(const BlockVector<double> &solution_delta) override;
  };

  template <int dim>
  void
  Step44<dim>::build_assembler()
  {
    using namespace WeakForms;
    using namespace Differentiation;

    constexpr int spacedim = dim;
    using SDNumber_t       = typename Differentiation::SD::Expression;

    constexpr Differentiation::SD::OptimizerType optimizer_type =
      Differentiation::SD::OptimizerType::lambda;
    constexpr Differentiation::SD::OptimizationFlags optimization_flags =
      Differentiation::SD::OptimizationFlags::optimize_all;

    this->timer.enter_subsection("Construct assembler");

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

    // Residual
    const auto residual_func_u = residual_functor("R", "R", Grad_u, p_tilde);
    const auto residual_func_p = residual_functor("R", "R", Grad_u, J_tilde);
    const auto residual_func_J = residual_functor("R", "R", p_tilde, J_tilde);
    const auto residual_ss_u   = residual_func_u[Grad_test_u];
    const auto residual_ss_p   = residual_func_p[test_p];
    const auto residual_ss_J   = residual_func_J[test_J];

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

    const auto residual_p =
      residual_ss_p.template value<SDNumber_t, dim, spacedim>(
        [&spacedim](const Tensor<2, spacedim, SDNumber_t> &Grad_u,
                    const SDNumber_t &                     J_tilde)
        {
          const Tensor<2, spacedim, SDNumber_t> F =
            Grad_u + Physics::Elasticity::StandardTensors<dim>::I;
          const SDNumber_t det_F_minus_J_tilde = determinant(F) - J_tilde;
          return det_F_minus_J_tilde;
        },
        [&spacedim](const Tensor<2, spacedim, SDNumber_t> &Grad_u,
                    const SDNumber_t &                     J_tilde)
        {
          // Due to our shortcut, we've not made the constitutive
          // parameters symbolic.
          return Differentiation::SD::types::substitution_map{};
        },
        [&spacedim](const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                    const std::vector<SolutionExtractionData<dim, spacedim>>
                      &                solution_extraction_data,
                    const unsigned int q_point)
        { return Differentiation::SD::types::substitution_map{}; },
        optimizer_type,
        optimization_flags,
        UpdateFlags::update_default);

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
    const auto force_func_u = residual_functor("F", "F", u);
    const auto force_ss_u   = force_func_u[test_u];

    const SDNumber_t symb_pressure = Differentiation::SD::make_symbol("p");
    const Tensor<1, spacedim, SDNumber_t> symb_N =
      Differentiation::SD::make_vector_of_symbols<spacedim>("N");
    const auto force_u = force_ss_u.template value<SDNumber_t, dim, spacedim>(
      [symb_pressure, symb_N, &spacedim](
        const Tensor<1, spacedim, SDNumber_t> &u)
      { return symb_pressure * symb_N; },
      [symb_pressure, symb_N, &spacedim](
        const Tensor<1, spacedim, SDNumber_t> &u)
      { return Differentiation::SD::make_symbol_map(symb_pressure, symb_N); },
      [this, symb_pressure, symb_N, &spacedim](
        const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
        const std::vector<SolutionExtractionData<dim, spacedim>>
          &                solution_extraction_data,
        const unsigned int q_point)
      {
        static const double p0 =
          -4.0 / (this->parameters.scale * this->parameters.scale);
        const double time_ramp = (this->time.current() / this->time.end());
        const double pressure  = p0 * this->parameters.p_p0 * time_ramp;
        const Tensor<1, spacedim> &N =
          scratch_data.get_normal_vectors()[q_point];

        return Differentiation::SD::make_substitution_map(
          std::make_pair(symb_pressure, pressure), std::make_pair(symb_N, N));
      },
      optimizer_type,
      optimization_flags,
      UpdateFlags::update_normal_vectors);

    // Boundary conditions
    const dealii::types::boundary_id traction_boundary_id = 6;

    // Assembly
    assembler += residual_form(residual_u).dV() +
                 residual_form(residual_p).dV() +
                 residual_form(residual_J).dV() -
                 residual_form(force_u).dA(traction_boundary_id);

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

    this->timer.leave_subsection();
  }

  template <int dim>
  void
  Step44<dim>::assemble_system(const BlockVector<double> &solution_delta)
  {
    // Initialise the assembler.
    // We need to do it here because the need to have the grid built
    // first (we fetch the lqph for a cell).
    {
      static bool assembler_initialised = false;
      if (!assembler_initialised)
        {
          build_assembler();
          assembler_initialised = true;
        }
    }

    this->timer.enter_subsection("Assemble system");
    std::cout << " ASM_SYS " << std::flush;
    this->tangent_matrix = 0.0;
    this->system_rhs     = 0.0;
    const BlockVector<double> solution_total(
      this->get_total_solution(solution_delta));

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
