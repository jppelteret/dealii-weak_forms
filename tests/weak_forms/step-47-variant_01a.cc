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

// This test replicates step-47.
// It is used as a baseline for the weak form tests.

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
    using Iterator = typename DoFHandler<dim>::active_cell_iterator;

    auto cell_worker = [&](const Iterator &  cell,
                           ScratchData<dim> &scratch_data,
                           CopyData &        copy_data) {
      copy_data.cell_matrix = 0;
      copy_data.cell_rhs    = 0;

      FEValues<dim> &fe_values = scratch_data.fe_values;
      fe_values.reinit(cell);

      cell->get_dof_indices(copy_data.local_dof_indices);

      const ExactSolution::RightHandSide<dim> right_hand_side;

      const unsigned int dofs_per_cell =
        scratch_data.fe_values.get_fe().n_dofs_per_cell();

      for (unsigned int qpoint = 0; qpoint < fe_values.n_quadrature_points;
           ++qpoint)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const Tensor<2, dim> hessian_i =
                fe_values.shape_hessian(i, qpoint);

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const Tensor<2, dim> hessian_j =
                    fe_values.shape_hessian(j, qpoint);

                  copy_data.cell_matrix(i, j) +=
                    scalar_product(hessian_i,   // nabla^2 phi_i(x)
                                   hessian_j) * // nabla^2 phi_j(x)
                    fe_values.JxW(qpoint);      // dx
                }

              copy_data.cell_rhs(i) +=
                fe_values.shape_value(i, qpoint) * // phi_i(x)
                right_hand_side.value(
                  fe_values.quadrature_point(qpoint)) * // f(x)
                fe_values.JxW(qpoint);                  // dx
            }
        }
    };


    auto face_worker = [&](const Iterator &    cell,
                           const unsigned int &f,
                           const unsigned int &sf,
                           const Iterator &    ncell,
                           const unsigned int &nf,
                           const unsigned int &nsf,
                           ScratchData<dim> &  scratch_data,
                           CopyData &          copy_data) {
      FEInterfaceValues<dim> &fe_interface_values =
        scratch_data.fe_interface_values;
      fe_interface_values.reinit(cell, f, sf, ncell, nf, nsf);

      copy_data.face_data.emplace_back();
      CopyData::FaceData &copy_data_face = copy_data.face_data.back();

      copy_data_face.joint_dof_indices =
        fe_interface_values.get_interface_dof_indices();

      const unsigned int n_interface_dofs =
        fe_interface_values.n_current_interface_dofs();
      copy_data_face.cell_matrix.reinit(n_interface_dofs, n_interface_dofs);

      const unsigned int p = this->fe.degree;
      const double       gamma_over_h =
        std::max((1.0 * p * (p + 1) /
                  cell->extent_in_direction(
                    GeometryInfo<dim>::unit_normal_direction[f])),
                 (1.0 * p * (p + 1) /
                  ncell->extent_in_direction(
                    GeometryInfo<dim>::unit_normal_direction[nf])));

      for (unsigned int qpoint = 0;
           qpoint < fe_interface_values.n_quadrature_points;
           ++qpoint)
        {
          const auto &n = fe_interface_values.normal(qpoint);

          for (unsigned int i = 0; i < n_interface_dofs; ++i)
            {
              const double av_hessian_i_dot_n_dot_n =
                (fe_interface_values.average_hessian(i, qpoint) * n * n);
              const double jump_grad_i_dot_n =
                (fe_interface_values.jump_gradient(i, qpoint) * n);

              for (unsigned int j = 0; j < n_interface_dofs; ++j)
                {
                  const double av_hessian_j_dot_n_dot_n =
                    (fe_interface_values.average_hessian(j, qpoint) * n * n);
                  const double jump_grad_j_dot_n =
                    (fe_interface_values.jump_gradient(j, qpoint) * n);

                  copy_data_face.cell_matrix(i, j) +=
                    (-av_hessian_i_dot_n_dot_n       // - {grad^2 v n n }
                       * jump_grad_j_dot_n           // [grad u n]
                     - av_hessian_j_dot_n_dot_n      // - {grad^2 u n n }
                         * jump_grad_i_dot_n         // [grad v n]
                     +                               // +
                     gamma_over_h *                  // gamma/h
                       jump_grad_i_dot_n *           // [grad v n]
                       jump_grad_j_dot_n) *          // [grad u n]
                    fe_interface_values.JxW(qpoint); // dx
                }
            }
        }
    };


    auto boundary_worker = [&](const Iterator &    cell,
                               const unsigned int &face_no,
                               ScratchData<dim> &  scratch_data,
                               CopyData &          copy_data) {
      FEInterfaceValues<dim> &fe_interface_values =
        scratch_data.fe_interface_values;
      fe_interface_values.reinit(cell, face_no);
      const auto &q_points = fe_interface_values.get_quadrature_points();

      copy_data.face_data.emplace_back();
      CopyData::FaceData &copy_data_face = copy_data.face_data.back();

      const unsigned int n_dofs =
        fe_interface_values.n_current_interface_dofs();
      copy_data_face.joint_dof_indices =
        fe_interface_values.get_interface_dof_indices();

      copy_data_face.cell_matrix.reinit(n_dofs, n_dofs);

      const std::vector<double> &JxW = fe_interface_values.get_JxW_values();
      const std::vector<Tensor<1, dim>> &normals =
        fe_interface_values.get_normal_vectors();


      const ExactSolution::Solution<dim> exact_solution;
      std::vector<Tensor<1, dim>>        exact_gradients(q_points.size());
      exact_solution.gradient_list(q_points, exact_gradients);


      const unsigned int p = this->fe.degree;
      const double       gamma_over_h =
        (1.0 * p * (p + 1) /
         cell->extent_in_direction(
           GeometryInfo<dim>::unit_normal_direction[face_no]));

      for (unsigned int qpoint = 0; qpoint < q_points.size(); ++qpoint)
        {
          const auto &n = normals[qpoint];

          for (unsigned int i = 0; i < n_dofs; ++i)
            {
              const double av_hessian_i_dot_n_dot_n =
                (fe_interface_values.average_hessian(i, qpoint) * n * n);
              const double jump_grad_i_dot_n =
                (fe_interface_values.jump_gradient(i, qpoint) * n);

              for (unsigned int j = 0; j < n_dofs; ++j)
                {
                  const double av_hessian_j_dot_n_dot_n =
                    (fe_interface_values.average_hessian(j, qpoint) * n * n);
                  const double jump_grad_j_dot_n =
                    (fe_interface_values.jump_gradient(j, qpoint) * n);

                  copy_data_face.cell_matrix(i, j) +=
                    (-av_hessian_i_dot_n_dot_n  // - {grad^2 v n n}
                       * jump_grad_j_dot_n      //   [grad u n]
                     - av_hessian_j_dot_n_dot_n // - {grad^2 u n n}
                         * jump_grad_i_dot_n    //   [grad v n]
                     + gamma_over_h             //  gamma/h
                         * jump_grad_i_dot_n    // [grad v n]
                         * jump_grad_j_dot_n    // [grad u n]
                     ) *
                    JxW[qpoint]; // dx
                }

              copy_data.cell_rhs(i) +=
                (-av_hessian_i_dot_n_dot_n *       // - {grad^2 v n n }
                   (exact_gradients[qpoint] * n)   //   (grad u_exact . n)
                 +                                 // +
                 gamma_over_h                      //  gamma/h
                   * jump_grad_i_dot_n             // [grad v n]
                   * (exact_gradients[qpoint] * n) // (grad u_exact . n)
                 ) *
                JxW[qpoint]; // dx
            }
        }
    };

    auto copier = [&](const CopyData &copy_data) {
      this->constraints.distribute_local_to_global(copy_data.cell_matrix,
                                                   copy_data.cell_rhs,
                                                   copy_data.local_dof_indices,
                                                   this->system_matrix,
                                                   this->system_rhs);

      for (auto &cdf : copy_data.face_data)
        {
          this->constraints.distribute_local_to_global(cdf.cell_matrix,
                                                       cdf.joint_dof_indices,
                                                       this->system_matrix);
        }
    };


    const unsigned int n_gauss_points = this->dof_handler.get_fe().degree + 1;
    ScratchData<dim>   scratch_data(this->mapping,
                                  this->fe,
                                  n_gauss_points,
                                  update_values | update_gradients |
                                    update_hessians | update_quadrature_points |
                                    update_JxW_values,
                                  update_values | update_gradients |
                                    update_hessians | update_quadrature_points |
                                    update_JxW_values | update_normal_vectors);
    CopyData           copy_data(this->dof_handler.get_fe().n_dofs_per_cell());
    MeshWorker::mesh_loop(this->dof_handler.begin_active(),
                          this->dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells |
                            MeshWorker::assemble_boundary_faces |
                            MeshWorker::assemble_own_interior_faces_once,
                          boundary_worker,
                          face_worker);
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
