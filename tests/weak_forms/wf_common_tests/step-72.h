/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2021 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 */


#include <deal.II/base/function.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/differentiation/ad.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/meshworker/copy_data.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/meshworker/scratch_data.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

namespace Step72
{
  using namespace dealii;


  class Step72_Parameters : public ParameterAcceptor
  {
  public:
    Step72_Parameters() = default;

    const double tolerance = 1e-3;
  };



  template <int dim>
  class Step72_Base
  {
  public:
    Step72_Base();

    void
    run(const double tolerance);

  protected:
    void
    setup_system(const bool initial_step);
    virtual void
    assemble_system() = 0;
    void
    solve();
    void
    refine_mesh();
    void
    set_boundary_values();
    double
    compute_residual(const double alpha) const;
    double
    determine_step_length() const;
    void
    output_results(const unsigned int refinement_cycle) const;

    Triangulation<dim> triangulation;

    DoFHandler<dim> dof_handler;
    FE_Q<dim>       fe;
    QGauss<dim>     quadrature_formula;

    AffineConstraints<double> hanging_node_constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> current_solution;
    Vector<double> newton_update;
    Vector<double> system_rhs;
  };


  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;
  };


  template <int dim>
  double
  BoundaryValues<dim>::value(const Point<dim> &p,
                             const unsigned int /*component*/) const
  {
    return std::sin(2 * numbers::PI * (p[0] + p[1]));
  }



  template <int dim>
  Step72_Base<dim>::Step72_Base()
    : dof_handler(triangulation)
    , fe(2)
    , quadrature_formula(fe.degree + 1)
  {}



  template <int dim>
  void
  Step72_Base<dim>::setup_system(const bool initial_step)
  {
    if (initial_step)
      {
        dof_handler.distribute_dofs(fe);
        current_solution.reinit(dof_handler.n_dofs());

        hanging_node_constraints.clear();
        DoFTools::make_hanging_node_constraints(dof_handler,
                                                hanging_node_constraints);
        hanging_node_constraints.close();
      }

    newton_update.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);

    hanging_node_constraints.condense(dsp);

    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
  }



  // template <int dim>
  // void Step72_Base<dim>::assemble_system_unassisted()
  // {
  //   system_matrix = 0;
  //   system_rhs    = 0;

  //   const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  //   using ScratchData = MeshWorker::ScratchData<dim>;
  //   using CopyData    = MeshWorker::CopyData<1, 1, 1>;

  //   using CellIteratorType = decltype(dof_handler.begin_active());

  //   const ScratchData sample_scratch_data(fe,
  //                                         quadrature_formula,
  //                                         update_gradients |
  //                                           update_quadrature_points |
  //                                           update_JxW_values);
  //   const CopyData    sample_copy_data(dofs_per_cell);

  //   const auto cell_worker = [this](const CellIteratorType &cell,
  //                                   ScratchData &           scratch_data,
  //                                   CopyData &              copy_data) {
  //     const auto &fe_values = scratch_data.reinit(cell);

  //     FullMatrix<double> &                  cell_matrix =
  //     copy_data.matrices[0]; Vector<double> &                      cell_rhs
  //     = copy_data.vectors[0]; std::vector<types::global_dof_index>
  //     &local_dof_indices =
  //       copy_data.local_dof_indices[0];
  //     cell->get_dof_indices(local_dof_indices);

  //     std::vector<Tensor<1, dim>> old_solution_gradients(
  //       fe_values.n_quadrature_points);
  //     fe_values.get_function_gradients(current_solution,
  //                                      old_solution_gradients);

  //     for (const unsigned int q : fe_values.quadrature_point_indices())
  //       {
  //         const double coeff =
  //           1.0 / std::sqrt(1.0 + old_solution_gradients[q] *
  //                                   old_solution_gradients[q]);

  //         for (const unsigned int i : fe_values.dof_indices())
  //           {
  //             for (const unsigned int j : fe_values.dof_indices())
  //               cell_matrix(i, j) +=
  //                 (((fe_values.shape_grad(i, q)      // ((\nabla \phi_i
  //                    * coeff                         //   * a_n
  //                    * fe_values.shape_grad(j, q))   //   * \nabla \phi_j)
  //                   -                                //  -
  //                   (fe_values.shape_grad(i, q)      //  (\nabla \phi_i
  //                    * coeff * coeff * coeff         //   * a_n^3
  //                    * (fe_values.shape_grad(j, q)   //   * (\nabla \phi_j
  //                       * old_solution_gradients[q]) //      * \nabla u_n)
  //                    * old_solution_gradients[q]))   //   * \nabla u_n)))
  //                  * fe_values.JxW(q));              // * dx

  //             cell_rhs(i) -= (fe_values.shape_grad(i, q)  // \nabla \phi_i
  //                             * coeff                     // * a_n
  //                             * old_solution_gradients[q] // * u_n
  //                             * fe_values.JxW(q));        // * dx
  //           }
  //       }
  //   };

  //   const auto copier = [dofs_per_cell, this](const CopyData &copy_data) {
  //     const FullMatrix<double> &cell_matrix = copy_data.matrices[0];
  //     const Vector<double> &    cell_rhs    = copy_data.vectors[0];
  //     const std::vector<types::global_dof_index> &local_dof_indices =
  //       copy_data.local_dof_indices[0];

  //     for (unsigned int i = 0; i < dofs_per_cell; ++i)
  //       {
  //         for (unsigned int j = 0; j < dofs_per_cell; ++j)
  //           system_matrix.add(local_dof_indices[i],
  //                             local_dof_indices[j],
  //                             cell_matrix(i, j));

  //         system_rhs(local_dof_indices[i]) += cell_rhs(i);
  //       }
  //   };

  //   MeshWorker::mesh_loop(dof_handler.active_cell_iterators(),
  //                         cell_worker,
  //                         copier,
  //                         sample_scratch_data,
  //                         sample_copy_data,
  //                         MeshWorker::assemble_own_cells);

  //   hanging_node_constraints.condense(system_matrix);
  //   hanging_node_constraints.condense(system_rhs);

  //   std::map<types::global_dof_index, double> boundary_values;
  //   VectorTools::interpolate_boundary_values(dof_handler,
  //                                            0,
  //                                            Functions::ZeroFunction<dim>(),
  //                                            boundary_values);
  //   MatrixTools::apply_boundary_values(boundary_values,
  //                                      system_matrix,
  //                                      newton_update,
  //                                      system_rhs);
  // }


  // template <int dim>
  // void Step72_Base<dim>::assemble_system_with_residual_linearization()
  // {
  //   system_matrix = 0;
  //   system_rhs    = 0;

  //   const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  //   using ScratchData      = MeshWorker::ScratchData<dim>;
  //   using CopyData         = MeshWorker::CopyData<1, 1, 1>;
  //   using CellIteratorType = decltype(dof_handler.begin_active());

  //   const ScratchData sample_scratch_data(fe,
  //                                         quadrature_formula,
  //                                         update_gradients |
  //                                           update_quadrature_points |
  //                                           update_JxW_values);
  //   const CopyData    sample_copy_data(dofs_per_cell);

  //   using ADHelper = Differentiation::AD::ResidualLinearization<
  //     Differentiation::AD::NumberTypes::sacado_dfad,
  //     double>;
  //   using ADNumberType = typename ADHelper::ad_type;

  //   const FEValuesExtractors::Scalar u_fe(0);

  //   const auto cell_worker = [&u_fe, this](const CellIteratorType &cell,
  //                                          ScratchData & scratch_data,
  //                                          CopyData &              copy_data)
  //                                          {
  //     const auto &       fe_values     = scratch_data.reinit(cell);
  //     const unsigned int dofs_per_cell =
  //     fe_values.get_fe().n_dofs_per_cell();

  //     FullMatrix<double> &                  cell_matrix =
  //     copy_data.matrices[0]; Vector<double> &                      cell_rhs
  //     = copy_data.vectors[0]; std::vector<types::global_dof_index>
  //     &local_dof_indices =
  //       copy_data.local_dof_indices[0];
  //     cell->get_dof_indices(local_dof_indices);

  //     const unsigned int n_independent_variables = local_dof_indices.size();
  //     const unsigned int n_dependent_variables   = dofs_per_cell;
  //     ADHelper ad_helper(n_independent_variables, n_dependent_variables);

  //     ad_helper.register_dof_values(current_solution, local_dof_indices);

  //     const std::vector<ADNumberType> &dof_values_ad =
  //       ad_helper.get_sensitive_dof_values();

  //     std::vector<Tensor<1, dim, ADNumberType>> old_solution_gradients(
  //       fe_values.n_quadrature_points);
  //     fe_values[u_fe].get_function_gradients_from_local_dof_values(
  //       dof_values_ad, old_solution_gradients);

  //     std::vector<ADNumberType> residual_ad(n_dependent_variables,
  //                                           ADNumberType(0.0));
  //     for (const unsigned int q : fe_values.quadrature_point_indices())
  //       {
  //         const ADNumberType coeff =
  //           1.0 / std::sqrt(1.0 + old_solution_gradients[q] *
  //                                   old_solution_gradients[q]);

  //         for (const unsigned int i : fe_values.dof_indices())
  //           {
  //             residual_ad[i] += (fe_values.shape_grad(i, q)   // \nabla
  //             \phi_i
  //                                * coeff                      // * a_n
  //                                * old_solution_gradients[q]) // * u_n
  //                               * fe_values.JxW(q);           // * dx
  //           }
  //       }

  //     ad_helper.register_residual_vector(residual_ad);

  //     ad_helper.compute_residual(cell_rhs);
  //     cell_rhs *= -1.0;

  //     ad_helper.compute_linearization(cell_matrix);
  //   };

  //   const auto copier = [dofs_per_cell, this](const CopyData &copy_data) {
  //     const FullMatrix<double> &cell_matrix = copy_data.matrices[0];
  //     const Vector<double> &    cell_rhs    = copy_data.vectors[0];
  //     const std::vector<types::global_dof_index> &local_dof_indices =
  //       copy_data.local_dof_indices[0];

  //     for (unsigned int i = 0; i < dofs_per_cell; ++i)
  //       {
  //         for (unsigned int j = 0; j < dofs_per_cell; ++j)
  //           system_matrix.add(local_dof_indices[i],
  //                             local_dof_indices[j],
  //                             cell_matrix(i, j));

  //         system_rhs(local_dof_indices[i]) += cell_rhs(i);
  //       }
  //   };

  //   MeshWorker::mesh_loop(dof_handler.active_cell_iterators(),
  //                         cell_worker,
  //                         copier,
  //                         sample_scratch_data,
  //                         sample_copy_data,
  //                         MeshWorker::assemble_own_cells);

  //   hanging_node_constraints.condense(system_matrix);
  //   hanging_node_constraints.condense(system_rhs);

  //   std::map<types::global_dof_index, double> boundary_values;
  //   VectorTools::interpolate_boundary_values(dof_handler,
  //                                            0,
  //                                            Functions::ZeroFunction<dim>(),
  //                                            boundary_values);
  //   MatrixTools::apply_boundary_values(boundary_values,
  //                                      system_matrix,
  //                                      newton_update,
  //                                      system_rhs);
  // }


  // template <int dim>
  // void Step72_Base<dim>::assemble_system_using_energy_functional()
  // {
  //   system_matrix = 0;
  //   system_rhs    = 0;

  //   const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  //   using ScratchData      = MeshWorker::ScratchData<dim>;
  //   using CopyData         = MeshWorker::CopyData<1, 1, 1>;
  //   using CellIteratorType = decltype(dof_handler.begin_active());

  //   const ScratchData sample_scratch_data(fe,
  //                                         quadrature_formula,
  //                                         update_gradients |
  //                                           update_quadrature_points |
  //                                           update_JxW_values);
  //   const CopyData    sample_copy_data(dofs_per_cell);

  //   using ADHelper = Differentiation::AD::EnergyFunctional<
  //     Differentiation::AD::NumberTypes::sacado_dfad_dfad,
  //     double>;
  //   using ADNumberType = typename ADHelper::ad_type;

  //   const FEValuesExtractors::Scalar u_fe(0);

  //   const auto cell_worker = [&u_fe, this](const CellIteratorType &cell,
  //                                          ScratchData & scratch_data,
  //                                          CopyData &              copy_data)
  //                                          {
  //     const auto &fe_values = scratch_data.reinit(cell);

  //     FullMatrix<double> &                  cell_matrix =
  //     copy_data.matrices[0]; Vector<double> &                      cell_rhs
  //     = copy_data.vectors[0]; std::vector<types::global_dof_index>
  //     &local_dof_indices =
  //       copy_data.local_dof_indices[0];
  //     cell->get_dof_indices(local_dof_indices);

  //     const unsigned int n_independent_variables = local_dof_indices.size();
  //     ADHelper           ad_helper(n_independent_variables);

  //     ad_helper.register_dof_values(current_solution, local_dof_indices);

  //     const std::vector<ADNumberType> &dof_values_ad =
  //       ad_helper.get_sensitive_dof_values();

  //     std::vector<Tensor<1, dim, ADNumberType>> old_solution_gradients(
  //       fe_values.n_quadrature_points);
  //     fe_values[u_fe].get_function_gradients_from_local_dof_values(
  //       dof_values_ad, old_solution_gradients);

  //     ADNumberType energy_ad = ADNumberType(0.0);
  //     for (const unsigned int q : fe_values.quadrature_point_indices())
  //       {
  //         const ADNumberType psi = std::sqrt(1.0 + old_solution_gradients[q]
  //         *
  //                                                    old_solution_gradients[q]);

  //         energy_ad += psi * fe_values.JxW(q);
  //       }

  //     ad_helper.register_energy_functional(energy_ad);

  //     ad_helper.compute_residual(cell_rhs);
  //     cell_rhs *= -1.0;

  //     ad_helper.compute_linearization(cell_matrix);
  //   };

  //   const auto copier = [dofs_per_cell, this](const CopyData &copy_data) {
  //     const FullMatrix<double> &cell_matrix = copy_data.matrices[0];
  //     const Vector<double> &    cell_rhs    = copy_data.vectors[0];
  //     const std::vector<types::global_dof_index> &local_dof_indices =
  //       copy_data.local_dof_indices[0];

  //     for (unsigned int i = 0; i < dofs_per_cell; ++i)
  //       {
  //         for (unsigned int j = 0; j < dofs_per_cell; ++j)
  //           system_matrix.add(local_dof_indices[i],
  //                             local_dof_indices[j],
  //                             cell_matrix(i, j));

  //         system_rhs(local_dof_indices[i]) += cell_rhs(i);
  //       }
  //   };

  //   MeshWorker::mesh_loop(dof_handler.active_cell_iterators(),
  //                         cell_worker,
  //                         copier,
  //                         sample_scratch_data,
  //                         sample_copy_data,
  //                         MeshWorker::assemble_own_cells);

  //   hanging_node_constraints.condense(system_matrix);
  //   hanging_node_constraints.condense(system_rhs);

  //   std::map<types::global_dof_index, double> boundary_values;
  //   VectorTools::interpolate_boundary_values(dof_handler,
  //                                            0,
  //                                            Functions::ZeroFunction<dim>(),
  //                                            boundary_values);
  //   MatrixTools::apply_boundary_values(boundary_values,
  //                                      system_matrix,
  //                                      newton_update,
  //                                      system_rhs);
  // }



  template <int dim>
  void
  Step72_Base<dim>::solve()
  {
    SolverControl            solver_control(system_rhs.size(),
                                 system_rhs.l2_norm() * 1e-6,
                                 false,
                                 false);
    SolverCG<Vector<double>> solver(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    solver.solve(system_matrix, newton_update, system_rhs, preconditioner);

    hanging_node_constraints.distribute(newton_update);

    const double alpha = determine_step_length();
    current_solution.add(alpha, newton_update);
  }



  template <int dim>
  void
  Step72_Base<dim>::refine_mesh()
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      QGauss<dim - 1>(fe.degree + 1),
      std::map<dealii::types::boundary_id, const Function<dim> *>(),
      current_solution,
      estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    estimated_error_per_cell,
                                                    0.3,
                                                    0.03);

    triangulation.prepare_coarsening_and_refinement();
    SolutionTransfer<dim> solution_transfer(dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(current_solution);
    triangulation.execute_coarsening_and_refinement();

    dof_handler.distribute_dofs(fe);

    Vector<double> tmp(dof_handler.n_dofs());
    solution_transfer.interpolate(current_solution, tmp);
    current_solution = tmp;

    hanging_node_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            hanging_node_constraints);
    hanging_node_constraints.close();

    set_boundary_values();

    setup_system(false);
  }



  template <int dim>
  void
  Step72_Base<dim>::set_boundary_values()
  {
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             BoundaryValues<dim>(),
                                             boundary_values);
    for (auto &boundary_value : boundary_values)
      current_solution(boundary_value.first) = boundary_value.second;

    hanging_node_constraints.distribute(current_solution);
  }



  template <int dim>
  double
  Step72_Base<dim>::compute_residual(const double alpha) const
  {
    Vector<double> residual(dof_handler.n_dofs());

    Vector<double> evaluation_point(dof_handler.n_dofs());
    evaluation_point = current_solution;
    evaluation_point.add(alpha, newton_update);

    const QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim>     fe_values(fe,
                            quadrature_formula,
                            update_gradients | update_quadrature_points |
                              update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    Vector<double>              cell_residual(dofs_per_cell);
    std::vector<Tensor<1, dim>> gradients(n_q_points);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_residual = 0;
        fe_values.reinit(cell);

        fe_values.get_function_gradients(evaluation_point, gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const double coeff =
              1.0 / std::sqrt(1.0 + gradients[q] * gradients[q]);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              cell_residual(i) -= (fe_values.shape_grad(i, q) // \nabla \phi_i
                                   * coeff                    // * a_n
                                   * gradients[q]             // * u_n
                                   * fe_values.JxW(q));       // * dx
          }

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          residual(local_dof_indices[i]) += cell_residual(i);
      }

    hanging_node_constraints.condense(residual);

    for (types::global_dof_index i :
         DoFTools::extract_boundary_dofs(dof_handler))
      residual(i) = 0;

    return residual.l2_norm();
  }



  template <int dim>
  double
  Step72_Base<dim>::determine_step_length() const
  {
    return 0.1;
  }



  template <int dim>
  void
  Step72_Base<dim>::output_results(const unsigned int refinement_cycle) const
  {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(current_solution, "solution");
    data_out.add_data_vector(newton_update, "update");
    data_out.build_patches();

    const std::string filename =
      "solution-" + Utilities::int_to_string(refinement_cycle, 2) + ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);
  }



  template <int dim>
  void
  Step72_Base<dim>::run(const double tolerance)
  {
    TimerOutput timer(std::cout, TimerOutput::summary, TimerOutput::wall_times);

    GridGenerator::hyper_ball(triangulation);
    triangulation.refine_global(2);

    setup_system(/*first time=*/true);
    set_boundary_values();

    double        last_residual_norm  = std::numeric_limits<double>::max();
    unsigned int  refinement_cycle    = 0;
    constexpr int n_refinement_cycles = 3;
    do
      {
        deallog << "Mesh refinement step " << refinement_cycle << std::endl;

        if (refinement_cycle != 0)
          refine_mesh();

        deallog << "  Initial residual: " << compute_residual(0) << std::endl;

        for (unsigned int inner_iteration = 0; inner_iteration < 5;
             ++inner_iteration)
          {
            {
              TimerOutput::Scope t(timer, "Assemble");
              assemble_system();
            }

            last_residual_norm = system_rhs.l2_norm();

            {
              TimerOutput::Scope t(timer, "Solve");
              solve();
            }


            deallog << "  Residual: " << compute_residual(0) << std::endl;
          }

        constexpr bool print_vtk = false;
        if (print_vtk)
          output_results(refinement_cycle);

        ++refinement_cycle;
        deallog << std::endl;
    } while (last_residual_norm > tolerance &&
             refinement_cycle < n_refinement_cycles);
  }
} // namespace Step72


// int main(int argc, char *argv[])
// {
//   try
//     {
//       using namespace Step72;

//       Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

//       std::string prm_file;
//       if (argc > 1)
//         prm_file = argv[1];
//       else
//         prm_file = "parameters.prm";

//       const Step72_Parameters parameters;

//       Step72_Base<2> minimal_surface_problem_2d;
//       minimal_surface_problem_2d.run(parameters.tolerance);
//     }
//   catch (std::exception &exc)
//     {
//       std::cerr << std::endl
//                 << std::endl
//                 << "----------------------------------------------------"
//                 << std::endl;
//       std::cerr << "Exception on processing: " << std::endl
//                 << exc.what() << std::endl
//                 << "Aborting!" << std::endl
//                 << "----------------------------------------------------"
//                 << std::endl;

//       return 1;
//     }
//   catch (...)
//     {
//       std::cerr << std::endl
//                 << std::endl
//                 << "----------------------------------------------------"
//                 << std::endl;
//       std::cerr << "Unknown exception!" << std::endl
//                 << "Aborting!" << std::endl
//                 << "----------------------------------------------------"
//                 << std::endl;
//       return 1;
//     }
//   return 0;
// }
