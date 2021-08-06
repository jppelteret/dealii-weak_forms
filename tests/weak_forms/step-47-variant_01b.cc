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

    // Test function (subspaces)
    const auto test_value       = test.value();
    const auto test_gradient    = test.gradient();
    const auto test_hessian     = test.hessian();
    const auto test_ave_hessian = test.average_of_hessians();
    const auto test_jump_gradient   = test.jump_in_gradients();

    // Trial solution (subspaces)
    const auto trial_gradient     = trial.gradient();
    const auto trial_hessian     = trial.hessian();
    const auto trial_ave_hessian = trial.average_of_hessians();
    const auto trial_jump_gradient   = trial.jump_in_gradients();

    // Boundaries and interfaces
    const Normal<spacedim> normal{};
    const auto             N = normal.value();

    // Functions
    const ExactSolution::RightHandSide<dim> right_hand_side;
    const ScalarFunctionFunctor<spacedim>   rhs_function(
      "f(x)", "f\\left(\\mathbf{X}\\right)");

    const ExactSolution::Solution<dim>    exact_solution;
    const VectorFunctionFunctor<spacedim> exact_solution_gradients_function(
      "grad_u(x)", "\\Nabla \\mathbf{U}\\left(\\mathbf{X}\\right)");
    // exact_solution.gradient_list(q_points, exact_gradients);

    const ScalarFunctor gamma_over_h_functor("gamma/h", "\\frac{\\gamma}{h}");
    const auto          gamma_over_h =
      gamma_over_h_functor.template value<double, dim, spacedim>(
        [this](const FEValuesBase<dim, spacedim> &fe_values,
               const unsigned int                 q_point) {
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
               const unsigned int                      q_point) {
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
      bilinear_form(N * test_ave_hessian * N, 1.0, trial_jump_gradient * N).dI() -
      bilinear_form(test_jump_gradient * N, 1.0, N * trial_ave_hessian * N).dI()
    + bilinear_form(test_jump_gradient * N, gamma_over_h, trial_jump_gradient * N).dI() -
      bilinear_form(N * test_hessian * N, 0.5, trial_gradient * N).dA() -
      bilinear_form(test_gradient * N, 0.5, N * trial_hessian * N).dA()
    + bilinear_form(test_gradient * N, gamma_over_h, trial_gradient * N).dA();

    // Cell RHS to assemble:
    //   (phi_i(x) * f(x)).dV
    // - ({grad^2 v n n} * (grad u_exact . n)).dA
    // + (gamma/h [grad v n] * (grad u_exact . n)).dA
    assembler += linear_form(test_value, rhs_function(right_hand_side)).dV();

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


    //   auto face_worker = [&](const Iterator &    cell,
    //                          const unsigned int &f,
    //                          const unsigned int &sf,
    //                          const Iterator &    ncell,
    //                          const unsigned int &nf,
    //                          const unsigned int &nsf,
    //                          ScratchData<dim> &  scratch_data,
    //                          CopyData &          copy_data) {
    //     FEInterfaceValues<dim> &fe_interface_values =
    //       scratch_data.fe_interface_values;
    //     fe_interface_values.reinit(cell, f, sf, ncell, nf, nsf);

    //     copy_data.face_data.emplace_back();
    //     CopyData::FaceData &copy_data_face = copy_data.face_data.back();

    //     copy_data_face.joint_dof_indices =
    //       fe_interface_values.get_interface_dof_indices();

    //     const unsigned int n_interface_dofs =
    //       fe_interface_values.n_current_interface_dofs();
    //     copy_data_face.cell_matrix.reinit(n_interface_dofs,
    //     n_interface_dofs);

    //     const unsigned int p = this->fe.degree;
    //     const double       gamma_over_h =
    //       std::max((1.0 * p * (p + 1) /
    //                 cell->extent_in_direction(
    //                   GeometryInfo<dim>::unit_normal_direction[f])),
    //                (1.0 * p * (p + 1) /
    //                 ncell->extent_in_direction(
    //                   GeometryInfo<dim>::unit_normal_direction[nf])));

    //     for (unsigned int qpoint = 0;
    //          qpoint < fe_interface_values.n_quadrature_points;
    //          ++qpoint)
    //       {
    //         const auto &n = fe_interface_values.normal(qpoint);

    //         for (unsigned int i = 0; i < n_interface_dofs; ++i)
    //           {
    //             const double av_hessian_i_dot_n_dot_n =
    //               (fe_interface_values.average_hessian(i, qpoint) * n * n);
    //             const double jump_gradient_i_dot_n =
    //               (fe_interface_values.jump_gradientient(i, qpoint) * n);

    //             for (unsigned int j = 0; j < n_interface_dofs; ++j)
    //               {
    //                 const double av_hessian_j_dot_n_dot_n =
    //                   (fe_interface_values.average_hessian(j, qpoint) * n *
    //                   n);
    //                 const double jump_gradient_j_dot_n =
    //                   (fe_interface_values.jump_gradientient(j, qpoint) * n);

    //                 copy_data_face.cell_matrix(i, j) +=
    //                   (-av_hessian_i_dot_n_dot_n       // - {grad^2 v n n }
    //                      * jump_gradient_j_dot_n           // [grad u n]
    //                    - av_hessian_j_dot_n_dot_n      // - {grad^2 u n n }
    //                        * jump_gradient_i_dot_n         // [grad v n]
    //                    +                               // +
    //                    gamma_over_h *                  // gamma/h
    //                      jump_gradient_i_dot_n *           // [grad v n]
    //                      jump_gradient_j_dot_n) *          // [grad u n]
    //                   fe_interface_values.JxW(qpoint); // dx
    //               }
    //           }
    //       }
    //   };


    //   auto boundary_worker = [&](const Iterator &    cell,
    //                              const unsigned int &face_no,
    //                              ScratchData<dim> &  scratch_data,
    //                              CopyData &          copy_data) {
    //     FEInterfaceValues<dim> &fe_interface_values =
    //       scratch_data.fe_interface_values;
    //     fe_interface_values.reinit(cell, face_no);
    //     const auto &q_points = fe_interface_values.get_quadrature_points();

    //     copy_data.face_data.emplace_back();
    //     CopyData::FaceData &copy_data_face = copy_data.face_data.back();

    //     const unsigned int n_dofs =
    //       fe_interface_values.n_current_interface_dofs();
    //     copy_data_face.joint_dof_indices =
    //       fe_interface_values.get_interface_dof_indices();

    //     copy_data_face.cell_matrix.reinit(n_dofs, n_dofs);

    //     const std::vector<double> &JxW =
    //     fe_interface_values.get_JxW_values(); const std::vector<Tensor<1,
    //     dim>> &normals =
    //       fe_interface_values.get_normal_vectors();


    //     const ExactSolution::Solution<dim> exact_solution;
    //     std::vector<Tensor<1, dim>>        exact_gradients(q_points.size());
    //     exact_solution.gradient_list(q_points, exact_gradients);


    //     const unsigned int p = this->fe.degree;
    //     const double       gamma_over_h =
    //       (1.0 * p * (p + 1) /
    //        cell->extent_in_direction(
    //          GeometryInfo<dim>::unit_normal_direction[face_no]));

    //     for (unsigned int qpoint = 0; qpoint < q_points.size(); ++qpoint)
    //       {
    //         const auto &n = normals[qpoint];

    //         for (unsigned int i = 0; i < n_dofs; ++i)
    //           {
    //             const double av_hessian_i_dot_n_dot_n =
    //               (fe_interface_values.average_hessian(i, qpoint) * n * n);
    //             const double jump_gradient_i_dot_n =
    //               (fe_interface_values.jump_gradientient(i, qpoint) * n);

    //             for (unsigned int j = 0; j < n_dofs; ++j)
    //               {
    //                 const double av_hessian_j_dot_n_dot_n =
    //                   (fe_interface_values.average_hessian(j, qpoint) * n *
    //                   n);
    //                 const double jump_gradient_j_dot_n =
    //                   (fe_interface_values.jump_gradientient(j, qpoint) * n);

    //                 copy_data_face.cell_matrix(i, j) +=
    //                   (-av_hessian_i_dot_n_dot_n  // - {grad^2 v n n}
    //                      * jump_gradient_j_dot_n      //   [grad u n]
    //                    - av_hessian_j_dot_n_dot_n // - {grad^2 u n n}
    //                        * jump_gradient_i_dot_n    //   [grad v n]
    //                    + gamma_over_h             //  gamma/h
    //                        * jump_gradient_i_dot_n    // [grad v n]
    //                        * jump_gradient_j_dot_n    // [grad u n]
    //                    ) *
    //                   JxW[qpoint]; // dx
    //               }

    //             copy_data.cell_rhs(i) +=
    //               (-av_hessian_i_dot_n_dot_n *       // - {grad^2 v n n }
    //                  (exact_gradients[qpoint] * n)   //   (grad u_exact . n)
    //                +                                 // +
    //                gamma_over_h                      //  gamma/h
    //                  * jump_gradient_i_dot_n             // [grad v n]
    //                  * (exact_gradients[qpoint] * n) // (grad u_exact . n)
    //                ) *
    //               JxW[qpoint]; // dx
    //           }
    //       }
    //   };

    //   auto copier = [&](const CopyData &copy_data) {
    //     this->constraints.distribute_local_to_global(copy_data.cell_matrix,
    //                                                  copy_data.cell_rhs,
    //                                                  copy_data.local_dof_indices,
    //                                                  this->system_matrix,
    //                                                  this->system_rhs);

    //     for (auto &cdf : copy_data.face_data)
    //       {
    //         this->constraints.distribute_local_to_global(cdf.cell_matrix,
    //                                                      cdf.joint_dof_indices,
    //                                                      this->system_matrix);
    //       }
    //   };


    //   const unsigned int n_gauss_points = this->dof_handler.get_fe().degree +
    //   1; ScratchData<dim>   scratch_data(this->mapping,
    //                                 this->fe,
    //                                 n_gauss_points,
    //                                 update_values | update_gradients |
    //                                   update_hessians | update_quadrature_points |
    //                                   update_JxW_values,
    //                                 update_values | update_gradients |
    //                                   update_hessians | update_quadrature_points |
    //                                   update_JxW_values | update_normal_vectors);
    //   CopyData copy_data(this->dof_handler.get_fe().n_dofs_per_cell());
    //   MeshWorker::mesh_loop(this->dof_handler.begin_active(),
    //                         this->dof_handler.end(),
    //                         cell_worker,
    //                         copier,
    //                         scratch_data,
    //                         copy_data,
    //                         MeshWorker::assemble_own_cells |
    //                           MeshWorker::assemble_boundary_faces |
    //                           MeshWorker::assemble_own_interior_faces_once,
    //                         boundary_worker,
    //                         face_worker);
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
