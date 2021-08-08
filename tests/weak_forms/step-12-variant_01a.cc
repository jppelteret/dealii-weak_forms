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

// This test replicates step-12.
// It is used as a baseline for the weak form tests.

#include "../weak_forms_tests.h"
#include "wf_common_tests/step-12.h"

namespace Step12
{
  template <int dim>
  class Step12 : public AdvectionProblem<dim>
  {
  public:
    Step12()
      : AdvectionProblem<dim>()
    {}

  protected:
    void
    assemble_system() override;
  };


  template <int dim>
  void
  Step12<dim>::assemble_system()
  {
    using Iterator = typename DoFHandler<dim>::active_cell_iterator;
    const BoundaryValues<dim> boundary_function;

    const auto cell_worker = [&](const Iterator &  cell,
                                 ScratchData<dim> &scratch_data,
                                 CopyData &        copy_data)
    {
      const unsigned int n_dofs =
        scratch_data.fe_values.get_fe().n_dofs_per_cell();
      copy_data.reinit(cell, n_dofs);
      scratch_data.fe_values.reinit(cell);

      const auto &q_points = scratch_data.fe_values.get_quadrature_points();

      const FEValues<dim> &      fe_v = scratch_data.fe_values;
      const std::vector<double> &JxW  = fe_v.get_JxW_values();

      for (unsigned int point = 0; point < fe_v.n_quadrature_points; ++point)
        {
          auto beta_q = beta(q_points[point]);
          for (unsigned int i = 0; i < n_dofs; ++i)
            for (unsigned int j = 0; j < n_dofs; ++j)
              {
                copy_data.cell_matrix(i, j) +=
                  -beta_q                      // -\beta
                  * fe_v.shape_grad(i, point)  // \nabla \phi_i
                  * fe_v.shape_value(j, point) // \phi_j
                  * JxW[point];                // dx
              }
        }
    };

    const auto boundary_worker = [&](const Iterator &    cell,
                                     const unsigned int &face_no,
                                     ScratchData<dim> &  scratch_data,
                                     CopyData &          copy_data)
    {
      scratch_data.fe_interface_values.reinit(cell, face_no);
      const FEFaceValuesBase<dim> &fe_face =
        scratch_data.fe_interface_values.get_fe_face_values(0);

      const auto &q_points = fe_face.get_quadrature_points();

      const unsigned int n_facet_dofs = fe_face.get_fe().n_dofs_per_cell();
      const std::vector<double> &        JxW     = fe_face.get_JxW_values();
      const std::vector<Tensor<1, dim>> &normals = fe_face.get_normal_vectors();

      std::vector<double> g(q_points.size());
      boundary_function.value_list(q_points, g);

      for (unsigned int point = 0; point < q_points.size(); ++point)
        {
          const double beta_dot_n = beta(q_points[point]) * normals[point];

          if (beta_dot_n > 0)
            {
              for (unsigned int i = 0; i < n_facet_dofs; ++i)
                for (unsigned int j = 0; j < n_facet_dofs; ++j)
                  copy_data.cell_matrix(i, j) +=
                    fe_face.shape_value(i, point)   // \phi_i
                    * fe_face.shape_value(j, point) // \phi_j
                    * beta_dot_n                    // \beta . n
                    * JxW[point];                   // dx
            }
          else
            for (unsigned int i = 0; i < n_facet_dofs; ++i)
              copy_data.cell_rhs(i) += -fe_face.shape_value(i, point) // \phi_i
                                       * g[point]                     // g
                                       * beta_dot_n  // \beta . n
                                       * JxW[point]; // dx
        }
    };

    const auto face_worker = [&](const Iterator &    cell,
                                 const unsigned int &f,
                                 const unsigned int &sf,
                                 const Iterator &    ncell,
                                 const unsigned int &nf,
                                 const unsigned int &nsf,
                                 ScratchData<dim> &  scratch_data,
                                 CopyData &          copy_data)
    {
      FEInterfaceValues<dim> &fe_iv = scratch_data.fe_interface_values;
      fe_iv.reinit(cell, f, sf, ncell, nf, nsf);
      const auto &q_points = fe_iv.get_quadrature_points();

      copy_data.face_data.emplace_back();
      CopyDataFace &copy_data_face = copy_data.face_data.back();

      const unsigned int n_dofs        = fe_iv.n_current_interface_dofs();
      copy_data_face.joint_dof_indices = fe_iv.get_interface_dof_indices();

      copy_data_face.cell_matrix.reinit(n_dofs, n_dofs);

      const std::vector<double> &        JxW     = fe_iv.get_JxW_values();
      const std::vector<Tensor<1, dim>> &normals = fe_iv.get_normal_vectors();

      for (unsigned int qpoint = 0; qpoint < q_points.size(); ++qpoint)
        {
          const double beta_dot_n = beta(q_points[qpoint]) * normals[qpoint];
          for (unsigned int i = 0; i < n_dofs; ++i)
            for (unsigned int j = 0; j < n_dofs; ++j)
              copy_data_face.cell_matrix(i, j) +=
                fe_iv.jump(i, qpoint) // [\phi_i]
                *
                fe_iv.shape_value((beta_dot_n > 0), j, qpoint) // phi_j^{upwind}
                * beta_dot_n                                   // (\beta . n)
                * JxW[qpoint];                                 // dx
        }
    };

    const AffineConstraints<double> constraints;

    const auto copier = [&](const CopyData &c)
    {
      constraints.distribute_local_to_global(c.cell_matrix,
                                             c.cell_rhs,
                                             c.local_dof_indices,
                                             this->system_matrix,
                                             this->right_hand_side);

      for (auto &cdf : c.face_data)
        {
          constraints.distribute_local_to_global(cdf.cell_matrix,
                                                 cdf.joint_dof_indices,
                                                 this->system_matrix);
        }
    };

    const unsigned int n_gauss_points = this->dof_handler.get_fe().degree + 1;

    ScratchData<dim> scratch_data(this->mapping, this->fe, n_gauss_points);
    CopyData         copy_data;

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

} // namespace Step12

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
      const unsigned int n_local_refinement_levels = 6;

      Step12::Step12<dim> dgmethod;
      dgmethod.run(n_local_refinement_levels);
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
