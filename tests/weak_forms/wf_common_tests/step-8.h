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

// This header replicates step-8, leaves some aspects of its implementation
// out so that they may be modified.
// It is used as a baseline for the weak form tests.


#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>



using namespace dealii;


template <int dim>
class Step8_Base
{
public:
  Step8_Base();
  void
  run();

protected:
  void
  setup_system();
  virtual void
  assemble_system() = 0;
  void
  solve();
  void
  refine_grid();
  void
  output_results(const unsigned int cycle) const;

  Triangulation<dim> triangulation;
  DoFHandler<dim>    dof_handler;

  FESystem<dim> fe;

  AffineConstraints<double> constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
};



template <int dim>
void
right_hand_side(const std::vector<Point<dim>> &points,
                std::vector<Tensor<1, dim>> &  values)
{
  Assert(values.size() == points.size(),
         ExcDimensionMismatch(values.size(), points.size()));
  Assert(dim >= 2, ExcNotImplemented());

  Point<dim> point_1, point_2;
  point_1(0) = 0.5;
  point_2(0) = -0.5;

  for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
    {
      if (((points[point_n] - point_1).norm_square() < 0.2 * 0.2) ||
          ((points[point_n] - point_2).norm_square() < 0.2 * 0.2))
        values[point_n][0] = 1.0;
      else
        values[point_n][0] = 0.0;

      if (points[point_n].norm_square() < 0.2 * 0.2)
        values[point_n][1] = 1.0;
      else
        values[point_n][1] = 0.0;
    }
}


template <int dim>
class RightHandSide : public TensorFunction<1, dim, double>
{
public:
  virtual Tensor<1, dim, double>
  value(const Point<dim> &p) const override
  {
    Point<dim> point_1, point_2;
    point_1(0) = 0.5;
    point_2(0) = -0.5;

    Tensor<1, dim, double> out;

    if (((p - point_1).norm_square() < 0.2 * 0.2) ||
        ((p - point_2).norm_square() < 0.2 * 0.2))
      out[0] = 1.0;
    else
      out[0] = 0.0;

    if (p.norm_square() < 0.2 * 0.2)
      out[1] = 1.0;
    else
      out[1] = 0.0;

    return out;
  }
};


template <int dim>
class Coefficient : public TensorFunction<4, dim, double>
{
public:
  Coefficient(const double lambda = 1.0, const double mu = 1.0)
    : lambda(lambda)
    , mu(mu)
  {}

  virtual Tensor<4, dim, double>
  value(const Point<dim> &p) const override
  {
    Tensor<4, dim, double>        C;
    const SymmetricTensor<2, dim> I = unit_symmetric_tensor<dim>();

    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        for (unsigned int k = 0; k < dim; ++k)
          for (unsigned int l = 0; l < dim; ++l)
            C[i][j][k][l] = lambda * I[i][j] * I[k][l] +
                            mu * (I[i][k] * I[j][l] + I[i][l] * I[j][k]);

    return C;
  }

private:
  const double lambda;
  const double mu;
};


// template <int dim>
// class Coefficient : public TensorFunction<4,dim>
// {
// public:
//   virtual double
//   value(const Point<dim> & p,
//         const unsigned int component = 0) const override
//   {
//      if (p.square() < 0.5 * 0.5)
//       return 20;
//     else
//       return 1;
//   }
// };


template <int dim>
Step8_Base<dim>::Step8_Base()
  : dof_handler(triangulation)
  , fe(FE_Q<dim>(1), dim)
{}



template <int dim>
void
Step8_Base<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(dim),
                                           constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
}



template <int dim>
void
Step8_Base<dim>::solve()
{
  SolverControl            solver_control(1000, 1e-12, false, false);
  SolverCG<Vector<double>> cg(solver_control);

  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  cg.solve(system_matrix, solution, system_rhs, preconditioner);

  constraints.distribute(solution);
}



template <int dim>
void
Step8_Base<dim>::refine_grid()
{
  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

  KellyErrorEstimator<dim>::estimate(dof_handler,
                                     QGauss<dim - 1>(fe.degree + 1),
                                     {},
                                     solution,
                                     estimated_error_per_cell);

  GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                  estimated_error_per_cell,
                                                  0.3,
                                                  0.03);

  triangulation.execute_coarsening_and_refinement();
}



template <int dim>
void
Step8_Base<dim>::output_results(const unsigned int cycle) const
{
  constexpr bool output_vtu = false;
  if (output_vtu)
    {
      DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler);

      std::vector<std::string> solution_names;
      switch (dim)
        {
          case 1:
            solution_names.emplace_back("displacement");
            break;
          case 2:
            solution_names.emplace_back("x_displacement");
            solution_names.emplace_back("y_displacement");
            break;
          case 3:
            solution_names.emplace_back("x_displacement");
            solution_names.emplace_back("y_displacement");
            solution_names.emplace_back("z_displacement");
            break;
          default:
            Assert(false, ExcNotImplemented());
        }

      data_out.add_data_vector(solution, solution_names);
      data_out.build_patches();

      std::ofstream output("solution-" + std::to_string(cycle) + ".vtk");
      data_out.write_vtk(output);
    }

  {
    const Point<dim>                      soln_pt;
    const Functions::FEFieldFunction<dim> fe_field_function(dof_handler,
                                                            solution);
    deallog << "Cycle " << cycle << ": " << fe_field_function.value(soln_pt)
            << std::endl;
  }
}



template <int dim>
void
Step8_Base<dim>::run()
{
  for (unsigned int cycle = 0; cycle < 4; ++cycle)
    {
      std::cout << "Cycle " << cycle << ':' << std::endl;

      if (cycle == 0)
        {
          GridGenerator::hyper_cube(triangulation, -1, 1);
          triangulation.refine_global(2);
        }
      else
        refine_grid();

      std::cout << "   Number of active cells:       "
                << triangulation.n_active_cells() << std::endl;

      setup_system();

      std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
                << std::endl;

      assemble_system();
      solve();
      output_results(cycle);
    }
}
