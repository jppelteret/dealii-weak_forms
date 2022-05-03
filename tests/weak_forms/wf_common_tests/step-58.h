/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2018 - 2021 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Wolfgang Bangerth, Colorado State University
 *         Yong-Yong Cai, Beijing Computational Science Research Center
 */

// ---------------------------------------------------------------------
//
// Copyright (C) 2022 by Jean-Paul Pelteret
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

// This header replicates step-58, leaves some aspects of its implementation
// out so that they may be modified.
// It is used as a baseline for the weak form tests.


#include <deal.II/base/logstream.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
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
class Step58_Base
{
public:
  Step58_Base();
  void
  run();

protected:
  void
  setup_system();
  virtual void
  assemble_matrices() = 0;
  void
  do_half_phase_step();
  void
  do_full_spatial_step();
  void
  output_results() const;


  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  AffineConstraints<std::complex<double>> constraints;

  SparsityPattern                    sparsity_pattern;
  SparseMatrix<std::complex<double>> system_matrix;
  SparseMatrix<std::complex<double>> rhs_matrix;

  Vector<std::complex<double>> solution;
  Vector<std::complex<double>> system_rhs;

  double       time;
  double       time_step;
  unsigned int timestep_number;

  double kappa;
};



template <int dim>
class InitialValues : public Function<dim, std::complex<double>>
{
public:
  InitialValues()
    : Function<dim, std::complex<double>>(1)
  {}

  virtual std::complex<double>
  value(const Point<dim> &p, const unsigned int component = 0) const override;
};



template <int dim>
std::complex<double>
InitialValues<dim>::value(const Point<dim> & p,
                          const unsigned int component) const
{
  static_assert(dim == 2, "This initial condition only works in 2d.");

  (void)component;
  Assert(component == 0, ExcIndexRange(component, 0, 1));

  const std::vector<Point<dim>> vortex_centers = {{0, -0.3},
                                                  {0, +0.3},
                                                  {+0.3, 0},
                                                  {-0.3, 0}};

  const double R = 0.1;
  const double alpha =
    1. / (std::pow(R, dim) * std::pow(numbers::PI, dim / 2.));

  double sum = 0;
  for (const auto &vortex_center : vortex_centers)
    {
      const Tensor<1, dim> distance = p - vortex_center;
      const double         r        = distance.norm();

      sum += alpha * std::exp(-(r * r) / (R * R));
    }

  return {std::sqrt(sum), 0.};
}



template <int dim>
class Potential : public Function<dim>
{
public:
  Potential() = default;
  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;
};



template <int dim>
double
Potential<dim>::value(const Point<dim> &p, const unsigned int component) const
{
  (void)component;
  Assert(component == 0, ExcIndexRange(component, 0, 1));

  return (Point<dim>().distance(p) > 0.7 ? 1000 : 0);
}



template <int dim>
Step58_Base<dim>::Step58_Base()
  : fe(2)
  , dof_handler(triangulation)
  , time(0)
  , time_step(1. / 32)
  , timestep_number(0)
  , kappa(1)
{}



template <int dim>
void
Step58_Base<dim>::setup_system()
{
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(3);

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;

  dof_handler.distribute_dofs(fe);

  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl
            << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
  rhs_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  constraints.close();
}



template <int dim>
void
Step58_Base<dim>::do_half_phase_step()
{
  for (auto &value : solution)
    {
      const std::complex<double> i         = {0, 1};
      const double               magnitude = std::abs(value);

      value =
        std::exp(-i * kappa * magnitude * magnitude * (time_step / 2)) * value;
    }
}



template <int dim>
void
Step58_Base<dim>::do_full_spatial_step()
{
  rhs_matrix.vmult(system_rhs, solution);

  SparseDirectUMFPACK direct_solver;
  direct_solver.solve(system_matrix, system_rhs);

  solution = system_rhs;
}



namespace DataPostprocessors
{
  template <int dim>
  class ComplexAmplitude : public DataPostprocessorScalar<dim>
  {
  public:
    ComplexAmplitude();

    virtual void
    evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &inputs,
      std::vector<Vector<double>> &computed_quantities) const override;
  };


  template <int dim>
  ComplexAmplitude<dim>::ComplexAmplitude()
    : DataPostprocessorScalar<dim>("Amplitude", update_values)
  {}


  template <int dim>
  void
  ComplexAmplitude<dim>::evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &inputs,
    std::vector<Vector<double>> &               computed_quantities) const
  {
    AssertDimension(computed_quantities.size(), inputs.solution_values.size());

    for (unsigned int q = 0; q < computed_quantities.size(); ++q)
      {
        AssertDimension(computed_quantities[q].size(), 1);
        AssertDimension(inputs.solution_values[q].size(), 2);

        const std::complex<double> psi(inputs.solution_values[q](0),
                                       inputs.solution_values[q](1));
        computed_quantities[q](0) = std::norm(psi);
      }
  }



  template <int dim>
  class ComplexPhase : public DataPostprocessorScalar<dim>
  {
  public:
    ComplexPhase();

    virtual void
    evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &inputs,
      std::vector<Vector<double>> &computed_quantities) const override;
  };


  template <int dim>
  ComplexPhase<dim>::ComplexPhase()
    : DataPostprocessorScalar<dim>("Phase", update_values)
  {}


  template <int dim>
  void
  ComplexPhase<dim>::evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &inputs,
    std::vector<Vector<double>> &               computed_quantities) const
  {
    AssertDimension(computed_quantities.size(), inputs.solution_values.size());

    double max_phase = -numbers::PI;
    for (unsigned int q = 0; q < computed_quantities.size(); ++q)
      {
        AssertDimension(computed_quantities[q].size(), 1);
        AssertDimension(inputs.solution_values[q].size(), 2);

        max_phase =
          std::max(max_phase,
                   std::arg(
                     std::complex<double>(inputs.solution_values[q](0),
                                          inputs.solution_values[q](1))));
      }

    for (auto &output : computed_quantities)
      output(0) = max_phase;
  }

} // namespace DataPostprocessors


template <int dim>
void
Step58_Base<dim>::output_results() const
{
  constexpr bool output_vtk = false;
  if (output_vtk == false)
    return;

  const DataPostprocessors::ComplexAmplitude<dim> complex_magnitude;
  const DataPostprocessors::ComplexPhase<dim>     complex_phase;

  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "Psi");
  data_out.add_data_vector(solution, complex_magnitude);
  data_out.add_data_vector(solution, complex_phase);
  data_out.build_patches();

  data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));

  const std::string filename =
    "solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtu";
  std::ofstream output(filename);
  data_out.write_vtu(output);
}



template <int dim>
void
Step58_Base<dim>::run()
{
  setup_system();
  assemble_matrices();

  time = 0;
  VectorTools::interpolate(dof_handler, InitialValues<dim>(), solution);
  output_results();

  const double end_time = 1;
  for (; time <= end_time; time += time_step)
    {
      ++timestep_number;

      deallog << "Time step " << timestep_number << " at t=" << time
              << std::flush;

      do_half_phase_step();
      do_full_spatial_step();
      do_half_phase_step();

      Functions::FEFieldFunction<dim, Vector<std::complex<double>>> fe_ff(
        dof_handler, solution);

      const Point<dim> pt{};
      deallog << "  ;  Solution at origin: " << fe_ff.value(pt) << std::endl;

      if (timestep_number % 1 == 0)
        output_results();
    }
}
