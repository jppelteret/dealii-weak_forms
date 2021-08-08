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

// This test replicates step-74.
// It is used as a baseline for the weak form tests.

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
    const auto cell_worker =
      [this](const auto &cell, auto &scratch_data, auto &copy_data)
    {
      const FEValues<dim> &fe_v          = scratch_data.reinit(cell);
      const unsigned int   dofs_per_cell = fe_v.dofs_per_cell;
      copy_data.reinit(cell, dofs_per_cell);

      const auto &       q_points    = scratch_data.get_quadrature_points();
      const unsigned int n_q_points  = q_points.size();
      const std::vector<double> &JxW = scratch_data.get_JxW_values();

      std::vector<double> rhs(n_q_points);
      this->rhs_function->value_list(q_points, rhs);

      for (unsigned int point = 0; point < n_q_points; ++point)
        for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < fe_v.dofs_per_cell; ++j)
              copy_data.cell_matrix(i, j) +=
                this->diffusion_coefficient * // nu
                fe_v.shape_grad(i, point) *   // grad v_h
                fe_v.shape_grad(j, point) *   // grad u_h
                JxW[point];                   // dx

            copy_data.cell_rhs(i) += fe_v.shape_value(i, point) * // v_h
                                     rhs[point] *                 // f
                                     JxW[point];                  // dx
          }
    };

    const auto boundary_worker = [this](const auto &        cell,
                                        const unsigned int &face_no,
                                        auto &              scratch_data,
                                        auto &              copy_data)
    {
      const FEFaceValuesBase<dim> &fe_fv = scratch_data.reinit(cell, face_no);

      const auto &       q_points      = scratch_data.get_quadrature_points();
      const unsigned int n_q_points    = q_points.size();
      const unsigned int dofs_per_cell = fe_fv.dofs_per_cell;

      const std::vector<double> &        JxW = scratch_data.get_JxW_values();
      const std::vector<Tensor<1, dim>> &normals =
        scratch_data.get_normal_vectors();

      std::vector<double> g(n_q_points);
      this->exact_solution->value_list(q_points, g);

      const double extent1 = cell->measure() / cell->face(face_no)->measure();
      const double penalty = get_penalty_factor(this->degree, extent1, extent1);

      for (unsigned int point = 0; point < n_q_points; ++point)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              copy_data.cell_matrix(i, j) +=
                (-this->diffusion_coefficient *  // - nu
                   fe_fv.shape_value(i, point) * // v_h
                   (fe_fv.shape_grad(j, point) * // (grad u_h .
                    normals[point])              //  n)

                 - this->diffusion_coefficient *   // - nu
                     (fe_fv.shape_grad(i, point) * // (grad v_h .
                      normals[point]) *            //  n)
                     fe_fv.shape_value(j, point)   // u_h

                 + this->diffusion_coefficient * penalty * // + nu sigma
                     fe_fv.shape_value(i, point) *         // v_h
                     fe_fv.shape_value(j, point)           // u_h

                 ) *
                JxW[point]; // dx

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            copy_data.cell_rhs(i) +=
              (-this->diffusion_coefficient *  // - nu
                 (fe_fv.shape_grad(i, point) * // (grad v_h .
                  normals[point]) *            //  n)
                 g[point]                      // g


               + this->diffusion_coefficient * penalty *  // + nu sigma
                   fe_fv.shape_value(i, point) * g[point] // v_h g

               ) *
              JxW[point]; // dx
        }
    };

    const auto face_worker = [this](const auto &        cell,
                                    const unsigned int &f,
                                    const unsigned int &sf,
                                    const auto &        ncell,
                                    const unsigned int &nf,
                                    const unsigned int &nsf,
                                    auto &              scratch_data,
                                    auto &              copy_data)
    {
      const FEInterfaceValues<dim> &fe_iv =
        scratch_data.reinit(cell, f, sf, ncell, nf, nsf);

      copy_data.face_data.emplace_back();
      CopyDataFace &     copy_data_face = copy_data.face_data.back();
      const unsigned int n_dofs_face    = fe_iv.n_current_interface_dofs();
      copy_data_face.joint_dof_indices  = fe_iv.get_interface_dof_indices();
      copy_data_face.cell_matrix.reinit(n_dofs_face, n_dofs_face);

      const std::vector<double> &        JxW     = fe_iv.get_JxW_values();
      const std::vector<Tensor<1, dim>> &normals = fe_iv.get_normal_vectors();

      const double extent1 = cell->measure() / cell->face(f)->measure();
      const double extent2 = ncell->measure() / ncell->face(nf)->measure();
      const double penalty = get_penalty_factor(this->degree, extent1, extent2);

      for (const unsigned int point : fe_iv.quadrature_point_indices())
        {
          for (const unsigned int i : fe_iv.dof_indices())
            for (const unsigned int j : fe_iv.dof_indices())
              copy_data_face.cell_matrix(i, j) +=
                (-this->diffusion_coefficient *               // - nu
                   fe_iv.jump_in_shape_values(i, point) *     // [v_h]
                   (fe_iv.average_of_shape_gradients(j,       //
                                                     point) * // ({grad u_h} .
                    normals[point])                           //  n)

                 -
                 this->diffusion_coefficient *                   // - nu
                   (fe_iv.average_of_shape_gradients(i, point) * // (grad v_h .
                    normals[point]) *                            //  n)
                   fe_iv.jump_in_shape_values(j, point)          // [u_h]

                 + this->diffusion_coefficient * penalty *  // + nu sigma
                     fe_iv.jump_in_shape_values(i, point) * // [v_h]
                     fe_iv.jump_in_shape_values(j, point)   // [u_h]

                 ) *
                JxW[point]; // dx
        }
    };

    AffineConstraints<double> constraints;
    constraints.close();
    const auto copier = [this, &constraints](const auto &c)
    {
      constraints.distribute_local_to_global(c.cell_matrix,
                                             c.cell_rhs,
                                             c.local_dof_indices,
                                             this->system_matrix,
                                             this->system_rhs);

      for (auto &cdf : c.face_data)
        {
          constraints.distribute_local_to_global(cdf.cell_matrix,
                                                 cdf.joint_dof_indices,
                                                 this->system_matrix);
        }
    };


    const UpdateFlags cell_flags = update_values | update_gradients |
                                   update_quadrature_points | update_JxW_values;
    const UpdateFlags face_flags = update_values | update_gradients |
                                   update_quadrature_points |
                                   update_normal_vectors | update_JxW_values;

    ScratchData scratch_data(this->mapping,
                             this->fe,
                             this->quadrature,
                             cell_flags,
                             this->face_quadrature,
                             face_flags);
    CopyData    copy_data;

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
