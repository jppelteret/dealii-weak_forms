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

// Finite strain elasticity problem: Assembly using self-linearizing energy
// functional weak form in conjunction with symbolic differentiation. The
// internal energy is calculated by hand (not retrieved from LQPH), and the
// external energy is also supplied. This test replicates step-44 exactly.
// - Optimizer type: LLVM
// - Optimization method: All
// - AD/SD Cache

#include <deal.II/differentiation/sd.h>

#include <weak_forms/weak_forms.h>

#include "../../tests/weak_forms/wf_common_tests/step-44.h"
#include "../../tests/weak_forms_tests.h"

namespace Step44
{
  template <int dim>
  class Step44 : public Step44_Base<dim>
  {
  public:
    Step44(const std::string &input_file)
      : Step44_Base<dim>(input_file, true /*timer_output*/)
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
      Differentiation::SD::OptimizerType::llvm;
    constexpr Differentiation::SD::OptimizationFlags optimization_flags =
      Differentiation::SD::OptimizationFlags::optimize_all;

    this->timer.enter_subsection("Construct assembler");

    // Symbolic types for test function, and the field solution.
    const FieldSolution<dim, spacedim> field_solution;
    const SubSpaceExtractors::Vector   subspace_extractor_u(
      this->u_dof, this->first_u_component, "u", "\\mathbf{u}");
    const SubSpaceExtractors::Scalar subspace_extractor_p(this->p_dof,
                                                          this->p_component,
                                                          "p_tilde",
                                                          "\\tilde{p}");
    const SubSpaceExtractors::Scalar subspace_extractor_J(this->J_dof,
                                                          this->J_component,
                                                          "J_tilde",
                                                          "\\tilde{J}");

    // Field solution (subspaces)
    const auto u       = field_solution[subspace_extractor_u].value();
    const auto Grad_u  = field_solution[subspace_extractor_u].gradient();
    const auto p_tilde = field_solution[subspace_extractor_p].value();
    const auto J_tilde = field_solution[subspace_extractor_J].value();

    // Field variables: Internal energy
    const auto internal_energy_func =
      energy_functor("e^{int}", "\\Psi^{int}", Grad_u, p_tilde, J_tilde);

    const SDNumber_t symb_c_1   = Differentiation::SD::make_symbol("c1");
    const SDNumber_t symb_kappa = Differentiation::SD::make_symbol("kappa");
    const auto       internal_energy =
      internal_energy_func.template value<SDNumber_t, dim, spacedim>(
        [symb_c_1,
         symb_kappa,
         &spacedim](const Tensor<2, spacedim, SDNumber_t> &Grad_u,
                    const SDNumber_t &                     p_tilde,
                    const SDNumber_t &                     J_tilde)
        {
          const Tensor<2, spacedim, SDNumber_t> F =
            unit_symmetric_tensor<spacedim>() + Grad_u;
          const SymmetricTensor<2, spacedim, SDNumber_t> C =
            symmetrize(transpose(F) * F);
          const SDNumber_t                         det_F = determinant(F);
          SymmetricTensor<2, spacedim, SDNumber_t> C_bar(C);
          C_bar *= std::pow(det_F, -2.0 / dim);

          // Isochoric part
          SDNumber_t psi_CpJ = symb_c_1 * (trace(C_bar) - SDNumber_t(spacedim));
          // Volumetric part
          psi_CpJ += (symb_kappa / 4.0) *
                     (J_tilde * J_tilde - SDNumber_t(1.0) - 2.0 * log(J_tilde));
          // Penalisation term
          psi_CpJ += p_tilde * (det_F - J_tilde);

          return psi_CpJ;
        },
        [symb_c_1, symb_kappa](const Tensor<2, spacedim, SDNumber_t> &Grad_u,
                               const SDNumber_t &                     p_tilde,
                               const SDNumber_t &                     J_tilde)
        { return Differentiation::SD::make_symbol_map(symb_c_1, symb_kappa); },
        [this,
         symb_c_1,
         symb_kappa](const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                     const std::vector<std::string> &solution_names,
                     const unsigned int              q_point)
        {
          const double mu = this->parameters.mu;
          const double nu = this->parameters.nu;
          const double kappa =
            (2.0 * mu * (1.0 + nu)) / (3.0 * (1.0 - 2.0 * nu));
          const double c_1 = mu / 2.0;

          return Differentiation::SD::make_substitution_map(
            std::make_pair(symb_c_1, c_1), std::make_pair(symb_kappa, kappa));
        },
        optimizer_type,
        optimization_flags);


    // Field variables: External energy
    const auto external_energy_func =
      energy_functor("e^{ext}", "\\Psi^{ext}", u);

    const SDNumber_t symb_pressure = Differentiation::SD::make_symbol("p");
    const Tensor<1, spacedim, SDNumber_t> symb_N =
      Differentiation::SD::make_vector_of_symbols<spacedim>("N");
    const auto external_energy =
      external_energy_func.template value<SDNumber_t, dim, spacedim>(
        [symb_pressure, symb_N, &spacedim](
          const Tensor<1, spacedim, SDNumber_t> &u)
        { return -u * (symb_pressure * symb_N); },
        [symb_pressure, symb_N](const Tensor<1, spacedim, SDNumber_t> &u)
        { return Differentiation::SD::make_symbol_map(symb_pressure, symb_N); },
        [this, symb_pressure, symb_N, &spacedim](
          const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
          const std::vector<std::string> &              solution_names,
          const unsigned int                            q_point)
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
    assembler +=
      energy_functional_form(internal_energy).dV() +
      energy_functional_form(external_energy).dA(traction_boundary_id);
    assembler.symmetrize();

    // // Look at what we're going to compute
    // const SymbolicDecorations decorator;
    // static bool               output = true;
    // if (output)
    //   {
    //     deallog << "\n" << std::endl;
    //     deallog << "Weak form (ascii):\n"
    //             << assembler.as_ascii(decorator) << std::endl;
    //     deallog << "Weak form (LaTeX):\n"
    //             << assembler.as_latex(decorator) << std::endl;
    //     deallog << "\n" << std::endl;
    //     output = false;
    //   }

    this->timer.leave_subsection();
  }

  template <int dim>
  void
  Step44<dim>::assemble_system(const BlockVector<double> &solution_delta)
  {
    // Initialise the assembler. This is done once up front to
    // any impact of the overhead of creating the differential forms.
    // We need to do it here because the need to have the grid built
    // first (we fetch the LQPH for a cell).
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
    argc, argv, numbers::invalid_unsigned_int);

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
