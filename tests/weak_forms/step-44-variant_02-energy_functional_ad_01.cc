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
// functional weak form in conjunction with automatic differentiation. The
// internal energy is calculated by hand (not retrieved from LQPH), and the
// external energy is also supplied. This test replicates step-44 exactly.

#include <deal.II/differentiation/ad.h>

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
    using namespace Differentiation;

    constexpr int  spacedim = dim;
    constexpr auto ad_typecode =
      Differentiation::AD::NumberTypes::sacado_dfad_dfad;
    using ADNumber_t =
      typename Differentiation::AD::NumberTraits<double, ad_typecode>::ad_type;

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

    // Field variables: Internal energy
    const auto internal_energy_func =
      energy_functor("e^{int}", "\\Psi^{int}", Grad_u, p_tilde, J_tilde);
    using EnergyADNumber_t = typename decltype(
      internal_energy_func)::template ad_type<double, ad_typecode>;
    static_assert(std::is_same<ADNumber_t, EnergyADNumber_t>::value,
                  "Expected identical AD number types");

    const auto internal_energy =
      internal_energy_func.template value<ADNumber_t, dim, spacedim>(
        [this,
         &spacedim](const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                    const std::vector<SolutionExtractionData<dim, spacedim>>
                      &                solution_extraction_data,
                    const unsigned int q_point,
                    const Tensor<2, spacedim, ADNumber_t> &Grad_u,
                    const ADNumber_t &                     p_tilde,
                    const ADNumber_t &                     J_tilde)
        {
          const double mu = this->parameters.mu;
          const double nu = this->parameters.nu;
          const double kappa =
            (2.0 * mu * (1.0 + nu)) / (3.0 * (1.0 - 2.0 * nu));
          const double c_1 = mu / 2.0;

          const Tensor<2, spacedim, ADNumber_t> F =
            unit_symmetric_tensor<spacedim>() + Grad_u;
          const SymmetricTensor<2, spacedim, ADNumber_t> C =
            symmetrize(transpose(F) * F);
          const ADNumber_t                         det_F = determinant(F);
          SymmetricTensor<2, spacedim, ADNumber_t> C_bar(C);
          C_bar *= std::pow(det_F, -2.0 / dim);

          // Isochoric part
          ADNumber_t psi_CpJ = c_1 * (trace(C_bar) - ADNumber_t(spacedim));
          // Volumetric part
          psi_CpJ += (kappa / 4.0) *
                     (J_tilde * J_tilde - ADNumber_t(1.0) - 2.0 * log(J_tilde));
          // Penalisation term
          psi_CpJ += p_tilde * (det_F - J_tilde);

          return psi_CpJ;
        });


    // Field variables: External energy
    const auto external_energy_func =
      energy_functor("e^{ext}", "\\Psi^{ext}", u);
    using EnergyADNumber_t = typename decltype(
      external_energy_func)::template ad_type<double, ad_typecode>;
    static_assert(std::is_same<ADNumber_t, EnergyADNumber_t>::value,
                  "Expected identical AD number types");

    const auto external_energy =
      external_energy_func.template value<ADNumber_t, dim, spacedim>(
        [this,
         &spacedim](const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                    const std::vector<SolutionExtractionData<dim, spacedim>>
                      &                solution_extraction_data,
                    const unsigned int q_point,
                    const Tensor<1, spacedim, ADNumber_t> &u)
        {
          static const double p0 =
            -4.0 / (this->parameters.scale * this->parameters.scale);
          const double time_ramp = (this->time.current() / this->time.end());
          const double pressure  = p0 * this->parameters.p_p0 * time_ramp;
          const Tensor<1, spacedim> &N =
            scratch_data.get_normal_vectors()[q_point];

          return -u * (pressure * N);
        },
        UpdateFlags::update_normal_vectors);

    // Boundary conditions
    const dealii::types::boundary_id traction_boundary_id = 6;

    // Assembly
    MatrixBasedAssembler<dim> assembler;
    assembler +=
      energy_functional_form(internal_energy).dV() +
      energy_functional_form(external_energy).dA(traction_boundary_id);
    // assembler.symmetrize(); // Check this one without symmetrisation

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
