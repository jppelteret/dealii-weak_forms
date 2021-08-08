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

// This test replicates step-72 (unassisted formulation).
// It is used as a baseline for the weak form tests.

#include "../weak_forms_tests.h"
#include "wf_common_tests/step-72.h"

namespace Step72
{
  template <int dim>
  class Step72 : public Step72_Base<dim>
  {
  public:
    Step72()
      : Step72_Base<dim>()
    {}

  protected:
    void
    assemble_system() override;
  };


  template <int dim>
  void
  Step72<dim>::assemble_system()
  {
    this->system_matrix = 0;
    this->system_rhs    = 0;

    const unsigned int dofs_per_cell = this->fe.n_dofs_per_cell();

    using ScratchData = MeshWorker::ScratchData<dim>;
    using CopyData    = MeshWorker::CopyData<1, 1, 1>;

    using CellIteratorType = decltype(this->dof_handler.begin_active());

    const ScratchData sample_scratch_data(this->fe,
                                          this->quadrature_formula,
                                          update_gradients |
                                            update_quadrature_points |
                                            update_JxW_values);
    const CopyData    sample_copy_data(dofs_per_cell);

    const auto cell_worker = [this](const CellIteratorType &cell,
                                    ScratchData &           scratch_data,
                                    CopyData &              copy_data)
    {
      const auto &fe_values = scratch_data.reinit(cell);

      FullMatrix<double> &                  cell_matrix = copy_data.matrices[0];
      Vector<double> &                      cell_rhs    = copy_data.vectors[0];
      std::vector<types::global_dof_index> &local_dof_indices =
        copy_data.local_dof_indices[0];
      cell->get_dof_indices(local_dof_indices);

      std::vector<Tensor<1, dim>> old_solution_gradients(
        fe_values.n_quadrature_points);
      fe_values.get_function_gradients(this->current_solution,
                                       old_solution_gradients);

      for (const unsigned int q : fe_values.quadrature_point_indices())
        {
          const double coeff =
            1.0 / std::sqrt(1.0 + old_solution_gradients[q] *
                                    old_solution_gradients[q]);

          for (const unsigned int i : fe_values.dof_indices())
            {
              for (const unsigned int j : fe_values.dof_indices())
                cell_matrix(i, j) +=
                  (((fe_values.shape_grad(i, q)      // ((\nabla \phi_i
                     * coeff                         //   * a_n
                     * fe_values.shape_grad(j, q))   //   * \nabla \phi_j)
                    -                                //  -
                    (fe_values.shape_grad(i, q)      //  (\nabla \phi_i
                     * coeff * coeff * coeff         //   * a_n^3
                     * (fe_values.shape_grad(j, q)   //   * (\nabla \phi_j
                        * old_solution_gradients[q]) //      * \nabla u_n)
                     * old_solution_gradients[q]))   //   * \nabla u_n)))
                   * fe_values.JxW(q));              // * dx

              cell_rhs(i) -= (fe_values.shape_grad(i, q)  // \nabla \phi_i
                              * coeff                     // * a_n
                              * old_solution_gradients[q] // * u_n
                              * fe_values.JxW(q));        // * dx
            }
        }
    };

    const auto copier = [dofs_per_cell, this](const CopyData &copy_data)
    {
      const FullMatrix<double> &cell_matrix = copy_data.matrices[0];
      const Vector<double> &    cell_rhs    = copy_data.vectors[0];
      const std::vector<types::global_dof_index> &local_dof_indices =
        copy_data.local_dof_indices[0];

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            this->system_matrix.add(local_dof_indices[i],
                                    local_dof_indices[j],
                                    cell_matrix(i, j));

          this->system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    };

    MeshWorker::mesh_loop(this->dof_handler.active_cell_iterators(),
                          cell_worker,
                          copier,
                          sample_scratch_data,
                          sample_copy_data,
                          MeshWorker::assemble_own_cells);

    this->hanging_node_constraints.condense(this->system_matrix);
    this->hanging_node_constraints.condense(this->system_rhs);

    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(this->dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             boundary_values);
    MatrixTools::apply_boundary_values(boundary_values,
                                       this->system_matrix,
                                       this->newton_update,
                                       this->system_rhs);
  }
} // namespace Step72

int
main(int argc, char *argv[])
{
  initlog();
  deallog << std::setprecision(9);

  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

      std::string prm_file;
      if (argc > 1)
        prm_file = argv[1];
      else
        prm_file = "parameters.prm";

      const Step72::Step72_Parameters parameters;

      Step72::Step72<2> minimal_surface_problem_2d;
      minimal_surface_problem_2d.run(parameters.tolerance);
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
