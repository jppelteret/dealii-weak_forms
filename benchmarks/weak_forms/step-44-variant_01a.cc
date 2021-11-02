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

// This test replicates step-44.
// It is used as a baseline for the weak form tests.

#include "../../tests/weak_forms/wf_common_tests/step-44.h"
#include "../../tests/weak_forms_tests.h"

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
    struct PerTaskData_ASM;
    struct ScratchData_ASM;
    void
    assemble_system(const BlockVector<double> &solution_delta) override;
    void
    assemble_system_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM &                                     scratch,
      PerTaskData_ASM &                                     data) const;
    void
    copy_local_to_global_system(const PerTaskData_ASM &data);
  };
  template <int dim>
  struct Step44<dim>::PerTaskData_ASM
  {
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    PerTaskData_ASM(const unsigned int dofs_per_cell)
      : cell_matrix(dofs_per_cell, dofs_per_cell)
      , cell_rhs(dofs_per_cell)
      , local_dof_indices(dofs_per_cell)
    {}
    void
    reset()
    {
      cell_matrix = 0.0;
      cell_rhs    = 0.0;
    }
  };
  template <int dim>
  struct Step44<dim>::ScratchData_ASM
  {
    const BlockVector<double> &                       solution_total;
    FEValues<dim>                                     fe_values_ref;
    FEFaceValues<dim>                                 fe_face_values_ref;
    std::vector<std::vector<double>>                  Nx;
    std::vector<std::vector<Tensor<2, dim>>>          grad_Nx;
    std::vector<std::vector<SymmetricTensor<2, dim>>> symm_grad_Nx;
    ScratchData_ASM(const FiniteElement<dim> & fe_cell,
                    const QGauss<dim> &        qf_cell,
                    const UpdateFlags          uf_cell,
                    const QGauss<dim - 1> &    qf_face,
                    const UpdateFlags          uf_face,
                    const BlockVector<double> &solution_total)
      : solution_total(solution_total)
      , fe_values_ref(fe_cell, qf_cell, uf_cell)
      , fe_face_values_ref(fe_cell, qf_face, uf_face)
      , Nx(qf_cell.size(), std::vector<double>(fe_cell.dofs_per_cell))
      , grad_Nx(qf_cell.size(),
                std::vector<Tensor<2, dim>>(fe_cell.dofs_per_cell))
      , symm_grad_Nx(qf_cell.size(),
                     std::vector<SymmetricTensor<2, dim>>(
                       fe_cell.dofs_per_cell))
    {}
    ScratchData_ASM(const ScratchData_ASM &rhs)
      : solution_total(rhs.solution_total)
      , fe_values_ref(rhs.fe_values_ref.get_fe(),
                      rhs.fe_values_ref.get_quadrature(),
                      rhs.fe_values_ref.get_update_flags())
      , fe_face_values_ref(rhs.fe_face_values_ref.get_fe(),
                           rhs.fe_face_values_ref.get_quadrature(),
                           rhs.fe_face_values_ref.get_update_flags())
      , Nx(rhs.Nx)
      , grad_Nx(rhs.grad_Nx)
      , symm_grad_Nx(rhs.symm_grad_Nx)
    {}
    void
    reset()
    {
      const unsigned int n_q_points      = Nx.size();
      const unsigned int n_dofs_per_cell = Nx[0].size();
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          Assert(Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());
          Assert(grad_Nx[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());
          Assert(symm_grad_Nx[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());
          for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
            {
              Nx[q_point][k]           = 0.0;
              grad_Nx[q_point][k]      = 0.0;
              symm_grad_Nx[q_point][k] = 0.0;
            }
        }
    }
  };
  template <int dim>
  void
  Step44<dim>::assemble_system(const BlockVector<double> &solution_delta)
  {
    this->timer.enter_subsection("Assemble system");
    std::cout << " ASM_SYS " << std::flush;
    this->tangent_matrix = 0.0;
    this->system_rhs     = 0.0;
    const BlockVector<double> solution_total(
      this->get_total_solution(solution_delta));
    const UpdateFlags uf_cell(update_values | update_gradients |
                              update_JxW_values);
    const UpdateFlags uf_face(update_values | update_normal_vectors |
                              update_JxW_values);
    PerTaskData_ASM   per_task_data(this->dofs_per_cell);
    ScratchData_ASM   scratch_data(
      this->fe, this->qf_cell, uf_cell, this->qf_face, uf_face, solution_total);
    WorkStream::run(this->dof_handler_ref.begin_active(),
                    this->dof_handler_ref.end(),
                    std::bind(&Step44<dim>::assemble_system_one_cell,
                              this,
                              std::placeholders::_1,
                              std::placeholders::_2,
                              std::placeholders::_3),
                    std::bind(&Step44<dim>::copy_local_to_global_system,
                              this,
                              std::placeholders::_1),
                    scratch_data,
                    per_task_data);
    this->timer.leave_subsection();
  }
  template <int dim>
  void
  Step44<dim>::copy_local_to_global_system(const PerTaskData_ASM &data)
  {
    this->constraints.distribute_local_to_global(data.cell_matrix,
                                                 data.cell_rhs,
                                                 data.local_dof_indices,
                                                 this->tangent_matrix,
                                                 this->system_rhs);
  }
  template <int dim>
  void
  Step44<dim>::assemble_system_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData_ASM &                                     scratch,
    PerTaskData_ASM &                                     data) const
  {
    data.reset();
    scratch.reset();
    scratch.fe_values_ref.reinit(cell);
    cell->get_dof_indices(data.local_dof_indices);
    const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
      this->quadrature_point_history.get_data(cell);
    Assert(lqph.size() == this->n_q_points, ExcInternalError());
    for (unsigned int q_point = 0; q_point < this->n_q_points; ++q_point)
      {
        const Tensor<2, dim> F_inv = lqph[q_point]->get_F_inv();
        for (unsigned int k = 0; k < this->dofs_per_cell; ++k)
          {
            const unsigned int k_group =
              this->fe.system_to_base_index(k).first.first;
            if (k_group == this->u_dof)
              {
                scratch.grad_Nx[q_point][k] =
                  scratch.fe_values_ref[this->u_fe].gradient(k, q_point) *
                  F_inv;
                scratch.symm_grad_Nx[q_point][k] =
                  symmetrize(scratch.grad_Nx[q_point][k]);
              }
            else if (k_group == this->p_dof)
              scratch.Nx[q_point][k] =
                scratch.fe_values_ref[this->p_fe].value(k, q_point);
            else if (k_group == this->J_dof)
              scratch.Nx[q_point][k] =
                scratch.fe_values_ref[this->J_fe].value(k, q_point);
            else
              Assert(k_group <= this->J_dof, ExcInternalError());
          }
      }
    for (unsigned int q_point = 0; q_point < this->n_q_points; ++q_point)
      {
        const SymmetricTensor<2, dim> tau     = lqph[q_point]->get_tau();
        const Tensor<2, dim>          tau_ns  = lqph[q_point]->get_tau();
        const double                  J_tilde = lqph[q_point]->get_J_tilde();
        const double                  p_tilde = lqph[q_point]->get_p_tilde();
        const SymmetricTensor<4, dim> Jc      = lqph[q_point]->get_Jc();
        const double dPsi_vol_dJ   = lqph[q_point]->get_dPsi_vol_dJ();
        const double d2Psi_vol_dJ2 = lqph[q_point]->get_d2Psi_vol_dJ2();
        const double det_F         = lqph[q_point]->get_det_F();
        const std::vector<double> &                 N = scratch.Nx[q_point];
        const std::vector<SymmetricTensor<2, dim>> &symm_grad_Nx =
          scratch.symm_grad_Nx[q_point];
        const std::vector<Tensor<2, dim>> &grad_Nx = scratch.grad_Nx[q_point];
        const double JxW = scratch.fe_values_ref.JxW(q_point);
        for (unsigned int i = 0; i < this->dofs_per_cell; ++i)
          {
            const unsigned int component_i =
              this->fe.system_to_component_index(i).first;
            const unsigned int i_group =
              this->fe.system_to_base_index(i).first.first;

            if (i_group == this->u_dof)
              data.cell_rhs(i) -= (symm_grad_Nx[i] * tau) * JxW;
            else if (i_group == this->p_dof)
              data.cell_rhs(i) -= N[i] * (det_F - J_tilde) * JxW;
            else if (i_group == this->J_dof)
              data.cell_rhs(i) -= N[i] * (dPsi_vol_dJ - p_tilde) * JxW;
            else
              Assert(i_group <= this->J_dof, ExcInternalError());

            for (unsigned int j = 0; j <= i; ++j)
              {
                const unsigned int component_j =
                  this->fe.system_to_component_index(j).first;
                const unsigned int j_group =
                  this->fe.system_to_base_index(j).first.first;
                if ((i_group == j_group) && (i_group == this->u_dof))
                  {
                    data.cell_matrix(i, j) += symm_grad_Nx[i] *
                                              Jc // The material contribution:
                                              * symm_grad_Nx[j] * JxW;
                    if (component_i ==
                        component_j) // geometrical stress contribution
                      data.cell_matrix(i, j) += grad_Nx[i][component_i] *
                                                tau_ns *
                                                grad_Nx[j][component_j] * JxW;
                  }
                else if ((i_group == this->p_dof) && (j_group == this->u_dof))
                  {
                    data.cell_matrix(i, j) +=
                      N[i] * det_F *
                      (symm_grad_Nx[j] * StandardTensors<dim>::I) * JxW;
                  }
                else if ((i_group == this->J_dof) && (j_group == this->p_dof))
                  data.cell_matrix(i, j) -= N[i] * N[j] * JxW;
                else if ((i_group == j_group) && (i_group == this->J_dof))
                  data.cell_matrix(i, j) += N[i] * d2Psi_vol_dJ2 * N[j] * JxW;
                else
                  Assert((i_group <= this->J_dof) && (j_group <= this->J_dof),
                         ExcInternalError());
              }
          }
      }
    for (const unsigned int face : GeometryInfo<dim>::face_indices())
      if (cell->face(face)->at_boundary() == true &&
          cell->face(face)->boundary_id() == 6)
        {
          scratch.fe_face_values_ref.reinit(cell, face);
          for (unsigned int f_q_point = 0; f_q_point < this->n_q_points_f;
               ++f_q_point)
            {
              const Tensor<1, dim> &N =
                scratch.fe_face_values_ref.normal_vector(f_q_point);
              static const double p0 =
                -4.0 / (this->parameters.scale * this->parameters.scale);
              const double time_ramp =
                (this->time.current() / this->time.end());
              const double pressure = p0 * this->parameters.p_p0 * time_ramp;
              const Tensor<1, dim> traction = pressure * N;
              for (unsigned int i = 0; i < this->dofs_per_cell; ++i)
                {
                  const unsigned int i_group =
                    this->fe.system_to_base_index(i).first.first;
                  if (i_group == this->u_dof)
                    {
                      const unsigned int component_i =
                        this->fe.system_to_component_index(i).first;
                      const double Ni =
                        scratch.fe_face_values_ref.shape_value(i, f_q_point);
                      const double JxW =
                        scratch.fe_face_values_ref.JxW(f_q_point);
                      data.cell_rhs(i) += (Ni * traction[component_i]) * JxW;
                    }
                }
            }
        }

    for (unsigned int i = 0; i < this->dofs_per_cell; ++i)
      for (unsigned int j = i + 1; j < this->dofs_per_cell; ++j)
        data.cell_matrix(i, j) = data.cell_matrix(j, i);
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
