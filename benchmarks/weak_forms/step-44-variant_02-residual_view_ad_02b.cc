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
// weak form in conjunction with automatic differentiation.
// This test replicates step-44 exactly.
//
// This variant of the test uses ADOL-C tapeless as the AD type.

#include <deal.II/differentiation/ad.h>

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
      Differentiation::AD::NumberTypes::adolc_tapeless;
    using ADNumber_t =
      typename Differentiation::AD::NumberTraits<double, ad_typecode>::ad_type;

    this->timer.enter_subsection("Assemble system");
    std::cout << " ASM_SYS " << std::flush;
    this->tangent_matrix = 0.0;
    this->system_rhs     = 0.0;
    const BlockVector<double> solution_total(
      this->get_total_solution(solution_delta));

    // Symbolic types for test function, and the field solution.
    const TestFunction<dim, spacedim>  test;
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
    // ADOL-C does not support the number of directional derivatives changing,
    // so we have to parameterise all of the AD functors identically.
    // (Not even using AD::HelperBase::configure_tapeless_mode() allows us to
    // simplify this.)
    const auto residual_func_u =
      residual_functor("R", "R", u, Grad_u, p_tilde, J_tilde);
    const auto residual_func_p =
      residual_functor("R", "R", u, Grad_u, p_tilde, J_tilde);
    const auto residual_func_J =
      residual_functor("R", "R", u, Grad_u, p_tilde, J_tilde);
    const auto residual_ss_u = residual_func_u[Grad_test_u];
    const auto residual_ss_p = residual_func_p[test_p];
    const auto residual_ss_J = residual_func_J[test_J];

    using ResidualADNumber_t =
      typename decltype(residual_ss_u)::template ad_type<double, ad_typecode>;
    static_assert(std::is_same<ADNumber_t, ResidualADNumber_t>::value,
                  "Expected identical AD number types");
    using Result_t_u =
      typename decltype(residual_ss_u)::template value_type<ADNumber_t>;
    using Result_t_p =
      typename decltype(residual_ss_p)::template value_type<ADNumber_t>;
    using Result_t_J =
      typename decltype(residual_ss_J)::template value_type<ADNumber_t>;

    const auto residual_u =
      residual_ss_u.template value<ADNumber_t, dim, spacedim>(
        [this,
         &spacedim](const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                    const std::vector<SolutionExtractionData<dim, spacedim>>
                      &                solution_extraction_data,
                    const unsigned int q_point,
                    const Tensor<1, spacedim, ADNumber_t> &u,
                    const Tensor<2, spacedim, ADNumber_t> &Grad_u,
                    const ADNumber_t &                     p_tilde,
                    const ADNumber_t &                     J_tilde)
        {
          (void)u;
          (void)J_tilde;

          const auto &cell = scratch_data.get_current_fe_values().get_cell();
          const auto &qph  = this->quadrature_point_history;
          const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
            qph.get_data(cell);
          const Tensor<2, spacedim, ADNumber_t> F =
            Grad_u + Physics::Elasticity::StandardTensors<dim>::I;
          return lqph[q_point]->get_P(F, p_tilde);
        },
        UpdateFlags::update_default);

    const auto residual_p =
      residual_ss_p.template value<ADNumber_t, dim, spacedim>(
        [&spacedim](const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                    const std::vector<SolutionExtractionData<dim, spacedim>>
                      &                solution_extraction_data,
                    const unsigned int q_point,
                    const Tensor<1, spacedim, ADNumber_t> &u,
                    const Tensor<2, spacedim, ADNumber_t> &Grad_u,
                    const ADNumber_t &                     p_tilde,
                    const ADNumber_t &                     J_tilde)
        {
          (void)u;
          (void)p_tilde;

          const Tensor<2, spacedim, ADNumber_t> F =
            Grad_u + Physics::Elasticity::StandardTensors<dim>::I;
          return determinant(F) - J_tilde;
        },
        UpdateFlags::update_default);

    const auto residual_J =
      residual_ss_J.template value<ADNumber_t, dim, spacedim>(
        [this](const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
               const std::vector<SolutionExtractionData<dim, spacedim>>
                 &                                    solution_extraction_data,
               const unsigned int                     q_point,
               const Tensor<1, spacedim, ADNumber_t> &u,
               const Tensor<2, spacedim, ADNumber_t> &Grad_u,
               const ADNumber_t &                     p_tilde,
               const ADNumber_t &                     J_tilde)
        {
          (void)u;
          (void)Grad_u;

          const auto &cell = scratch_data.get_current_fe_values().get_cell();
          const auto &qph  = this->quadrature_point_history;
          const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
            qph.get_data(cell);
          const ADNumber_t dPsi_vol_dJ =
            lqph[q_point]->get_dPsi_vol_dJ(J_tilde);
          return dPsi_vol_dJ - p_tilde;
        },
        UpdateFlags::update_default);

    // Field variables: External force
    const auto force_func_u =
      residual_functor("F", "F", u, Grad_u, p_tilde, J_tilde);
    const auto force_ss_u = force_func_u[test_u];

    using ForceADNumber_t =
      typename decltype(force_ss_u)::template ad_type<double, ad_typecode>;
    static_assert(std::is_same<ADNumber_t, ForceADNumber_t>::value,
                  "Expected identical AD number types");

    const auto force_u = force_ss_u.template value<ADNumber_t, dim, spacedim>(
      [this,
       &spacedim](const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                  const std::vector<SolutionExtractionData<dim, spacedim>>
                    &                solution_extraction_data,
                  const unsigned int q_point,
                  const Tensor<1, spacedim, ADNumber_t> &u,
                  const Tensor<2, spacedim, ADNumber_t> &Grad_u,
                  const ADNumber_t &                     p_tilde,
                  const ADNumber_t &                     J_tilde)
      {
        (void)Grad_u;
        (void)p_tilde;
        (void)J_tilde;

        static const double p0 =
          -4.0 / (this->parameters.scale * this->parameters.scale);
        const double time_ramp = (this->time.current() / this->time.end());
        const double pressure  = p0 * this->parameters.p_p0 * time_ramp;
        const Tensor<1, spacedim> &N =
          scratch_data.get_normal_vectors()[q_point];

        return pressure * N;
      },
      UpdateFlags::update_normal_vectors);

    // Boundary conditions
    const dealii::types::boundary_id traction_boundary_id = 6;

    // Assembly
    MatrixBasedAssembler<dim> assembler;
    assembler += residual_form(residual_u).dV() +
                 residual_form(residual_p).dV() +
                 residual_form(residual_J).dV() -
                 residual_form(force_u).dA(traction_boundary_id);
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

  // ADOL-C doesn't support multithreading
  constexpr unsigned int n_threads = 1;
  // constexpr unsigned int n_threads = numbers::invalid_unsigned_int;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, n_threads);

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
