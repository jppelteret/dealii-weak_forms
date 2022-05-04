/* $Id: NavierStokes-Beltrami.cc 2008-04-15 10:54:52CET martinkr $ */
/* Author: Martin Kronbichler, Uppsala University, 2008 */
/*    $Id: NavierStokes-Beltrami.cc 2008-04-15 10:54:52CET martinkr $ */
/*    Version: $Name$                                             */
/*                                                                */
/*    Copyright (C) 2008 by the author                            */
/*                                                                */
/*    This file is subject to QPL and may not be  distributed     */
/*    without copyright and license information. Please refer     */
/*    to the file deal.II/doc/license.html for the  text  and     */
/*    further information on this license.                        */

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

// @sect3{The Navier-Stokes problem}

// This program implements an algorithm for the
// incompressible Navier-Stokes equations solving
// the whole system at once, i.e., without any
// projection equation. This approach is also called
// "monolithic" in the literature.
//
// The program is build mainly on step-20 for the
// solution of the mixed Laplace problem and the
// step-22 tutorial program for the stationary
// Stokes problem - with the relevant changes for
// Navier-Stokes, though. The program compares the
// solution of the Navier-Stokes equations with the
// exact velocities and pressures for the so-called
// 3D Beltrami flow, a laminar flow field with an
// analytic solution expression.  The Beltrami
// parameters $a$ and $d$ are set to $a = \pi/4$ and
// $d = a \sqrt{2}$, respectively.  For details on
// the Beltrami flow, see, e.g., V. Gravemeier, The
// Variational Multiscale Method for Laminar and
// Turbulent Flow, <i>Arch. Comput. Meth. Engng.</i>
// 13(2):249-324, section 5.1. The 2D version of
// this flow is referred to as Taylor flow in the
// literature.

// @sect4{Technical remarks on this file}

// You compile and run this program by just
// typing <code>make run</code> in the
// terminal window of the directory where
// this file and the appropriate <code>
// Makefile</code> sits.
// It is assumed that
// you have a fully compiled deal.II program
// on your computer (obtained by first
// performing the <code>./configure</code>
// in the main deal.II directory and then
// running <code>make all</code> (what usually
// takes some hour).
// The directory of this
// file is supposed to sit two levels above
// the deal.II main directory, i.e.,
// <code>../../</code> refers to deal's main
// directory and <code>../../lib/</code> is
// the directory of the deal.II library
// files.

// @sect3{Include files}

// In the beginning, we have to include the relevant
// deal.II libraries that contain the main FEM info.
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_selector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_vanka.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

// Next, import all dealii names
// into global namespace.
using namespace dealii;

namespace StepNavierStokesBeltrami
{
  // @sect3{The <code>NavierStokesProblemBase</code> class template}
  template <int dim>
  class NavierStokesProblemBase
  {
    // This class is basically an extension of the step-20 and
    // step-21 tutorial programs of deal.II. The
    // constructor of the Navier-Stokes class is
    // hence build up in the same way as there.

    // @sect4{Member functions}

    // The member functions that are called from
    // outside are the constructor itself and
    // the run function.
  public:
    NavierStokesProblemBase();
    void
    run();

    // The <code>private</code> part is only
    // accessible to member functions of
    // <code>NavierStokesProblemBase</code>.
    // The first few functions do, in order,
    // first generate the grid and dof
    // structure, assemble the linear system
    // and right hand side (where the first
    // of these functions acutally only distributes
    // the work to possibly several threads
    // that run the next function in parallel).
    // The <code>solve</code> function solves
    // the linear system to a given right hand
    // sides (including a few options
    // for the preconditioner).
    // The last functions calculate the
    // L2 error, write the information to
    // files and print the computing
    // times of individual code aspects.
  private:
    void
    make_grid_and_dofs();

    void
    assemble_system();

    void
    assemble_system_interval(
      const typename DoFHandler<dim>::active_cell_iterator &beginc,
      const typename DoFHandler<dim>::active_cell_iterator &endc);

    unsigned int
    solve();

    void
    compute_errors() const;

    void
    output_results(const unsigned int timestep_number) const;

    void
    print_computing_times() const;

    // @sect4{Problem variables}

    // The first four declared variables
    // regard the triangulation object
    // and the finite element discretization.
    // Then define the variable to contain
    // information on the degrees of
    // freedom and (possibly) constraints
    // for adaptive grid
    // refinement as well as for the pressure
    // that might need to be
    // fixed by a zero mean constraint.
    const unsigned int degree;
    FESystem<dim>      fe;
    Triangulation<dim> triangulation;
    const unsigned int n_global_refinements;

    DoFHandler<dim>           dof_handler;
    AffineConstraints<double> hanging_node_and_pressure_constraints;

    // In contrast to the step-20 tutorial
    // program, we do not use block vectors
    // and matrices here, but build up the
    // system at once. A more efficient
    // implementation may require changes
    // at this point, though.
    //
    // As opposed to the linear and time
    // independent systems in step-20 and
    // step-22, we need some vectors to
    // store old solution values from
    // time stepping.
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solution_update;
    Vector<double> solution, solution_old, solution_old_scaled;
    Vector<double> system_rhs, exact;

    // We set the variables for the time integration,
    // i.e., time, final time, time step, previous time
    // step (for BDF-2), time step number and a weight
    // that is used in assembly for the contribution of
    // the time derivative.
    //
    // The variable <code>nu</code> is used to define
    // the fluid kinematic viscosity, the only parameter
    // in the incompressible Navier-Stokes equations,
    // where density is constant.
    //
    // The subsequent parameters are used to control how
    // many nonlinear iterations we want to perform at
    // most, the tolerance level at which we want to
    // stop that iteration, and how often we write
    // results to an output file.
    //
    // The last parameter is a switch to disable/enable
    // the stabilization.  There are a few options that
    // can be set, but it is recommended to either use
    // the setting 0 (which uses plain finite elements
    // without additional stabilization) or setting 5,
    // which uses all stabilization options, which are
    // SUPG and LSIC for velocity test functions, and
    // PSPG for the pressure test function.
    // More details:
    // <ul>
    // <li>0: no stabilization at all
    // <li>1: PSPG stabilization only
    // <li>2: LSIC only
    // <li>3: PSPG and LSIC
    // <li>4: SUPG and LSIC
    // <li>5: SUPG, PSPG, and LSIC </ul>
    const enum TimeStepping { EulerBackw, BDF2 } time_stepping;
    double       time;
    const double time_final;
    double       time_step, time_step_old, time_step_weight;
    int          time_step_number;

    const double nu;

    const unsigned int max_nl_iteration;
    const double       tol_nl_iteration;

    const unsigned int output_timestep_skip;

    const unsigned int stabilization;

    // Assembly and linear solver options.
    //
    // The variable <code>pressure_constraint</code>
    // determines whether to constraint the pressure to
    // zero mean value on the boundary or to set it the
    // standard Dirichlet conditions onto it.
    //
    // The variable <code>assembly_type</code> specifies
    // whether the assembly should be done in parallel
    // (with as many threads as there are CPUs detected
    // on the system) or in serial.
    //
    // The variable <code>solver_type</code> specifies
    // the linear solver to be used.  One can use
    // iterative solvers or direct solve with UMFPACK,
    // or one can disable the solve process (e.g. when
    // testing something else).
    //
    // Last, the variable
    // <code>preconditioner_type</code> specifies the
    // preconditioning to be used for the iterative
    // solver.
    const bool pressure_constraint;
    const enum AssemblyType { Parallel, Serial } assembly_type;
    const enum SolverType { None, GMRES, BiCGStab, Direct } solver_type;
    const enum PreconditionerType { Id, SSOR, ILU, Vanka } preconditioner_type;

    // We shall use a vector that measures the times in
    // individual steps for the solution, namely setting
    // up the system like making the grid, assembling
    // the linear systems, manipulating the variables
    // during the steps and solving the linear systems.
    Vector<double> comptimes;
    Vector<int>    compcounter;
  };



  // @sect3{Equation data}

  // The task is now to define - in analogy to the
  // step-20 and step-22 tutorial programs, the right
  // hand side of the problem (i.e.  the forcing term
  // f in the momentum equation of the Navier-Stokes
  // system), boundary values for both the velocity
  // and the pressure, and a function that describes
  // both velocity and pressure in the exact solution.
  // The functions are of dimension <code>dim</code>
  // in the velocity and one for the pressure.
  //
  // @sect4{Right hand side}
  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide(const unsigned int n_components = dim, const double time = 0.)
      : Function<dim>(n_components, time)
    {}

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const;
  };

  template <int dim>
  void
  RightHandSide<dim>::vector_value(const Point<dim> &p,
                                   Vector<double> &  values) const
  {
    Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));

    // double time = this->get_time ();


    for (unsigned int i = 0; i < dim; ++i)
      values(i) = 0. + 0 * p[i];
  }



  // @sect4{Exact Solution}

  // Before specifying the initial condition and the
  // boundary conditions, we implement the exact
  // solution.  From this, we shall get the necessary
  // boundary and initial data afterwards.  The
  // function implemented here is the so-called Taylor
  // flow in 2D (see Kim and Moin, JCP 59, pp. 308-323
  // (1985)) and the Beltrami flow in 3D (Ethier and
  // Steiman, Int. J. Num.  Meth. Fluids 19, 369-375,
  // 1994).
  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution(const unsigned int n_components = dim + 1,
                  const double       time         = 0.,
                  const double       viscosity    = 1.)
      : Function<dim>(n_components, time)
      , nu(viscosity)
    {}

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const;

  private:
    const double nu;
  };

  template <int dim>
  void
  ExactSolution<dim>::vector_value(const Point<dim> &p,
                                   Vector<double> &  values) const
  {
    Assert(values.size() == dim + 1,
           ExcDimensionMismatch(values.size(), dim + 1));

    const double time = this->get_time();

    const double a = 0.25 * numbers::PI;
    const double d = std::sqrt(2.) * a;

    switch (dim)
      {
        case 3:
          values(0) = -a *
                      (std::exp(a * p[0]) * std::sin(a * p[1] + d * p[2]) +
                       std::exp(a * p[2]) * std::cos(a * p[0] + d * p[1])) *
                      std::exp(-nu * d * d * time);
          values(1) = -a *
                      (std::exp(a * p[1]) * std::sin(a * p[2] + d * p[0]) +
                       std::exp(a * p[0]) * std::cos(a * p[1] + d * p[2])) *
                      std::exp(-nu * d * d * time);
          values(2) = -a *
                      (std::exp(a * p[2]) * std::sin(a * p[0] + d * p[1]) +
                       std::exp(a * p[1]) * std::cos(a * p[2] + d * p[0])) *
                      std::exp(-nu * d * d * time);
          values(3) =
            -a * a * 0.5 *
            (std::exp(2 * a * p[0]) + std::exp(2 * a * p[1]) +
             std::exp(2 * a * p[2]) +
             2 * std::sin(a * p[0] + d * p[1]) * std::cos(a * p[2] + d * p[0]) *
               std::exp(a * (p[1] + p[2])) +
             2 * std::sin(a * p[1] + d * p[2]) * std::cos(a * p[0] + d * p[1]) *
               std::exp(a * (p[2] + p[0])) +
             2 * std::sin(a * p[2] + d * p[0]) * std::cos(a * p[1] + d * p[2]) *
               std::exp(a * (p[0] + p[1]))) *
            std::exp(-2 * nu * d * d * time);
          break;
        case 2:
          values(0) = -a * std::cos(a * p[0]) * std::sin(a * p[1]) *
                      std::exp(-nu * d * d * time);
          values(1) = a * std::sin(a * p[0]) * std::cos(a * p[1]) *
                      std::exp(-nu * d * d * time);
          values(2) = -a * a * 0.25 *
                      (std::cos(2 * a * p[0]) + std::cos(2 * a * p[1])) *
                      std::exp(-2 * nu * d * d * time);
          break;
        default:
          Assert(false, ExcNotImplemented());
      }
  }



  // @sect4{Initial values at time 0}
  //
  // Note that we only refer to the exact solution at
  // this point (and we'll do so for the boundary
  // condition as well). This has the advantage that
  // changes would be needed only at one point in the
  // code.
  template <int dim>
  class InitialValues : public Function<dim>
  {
  public:
    InitialValues(const unsigned int n_components = dim + 1,
                  const double       time         = 0.,
                  const double       viscosity    = 1.)
      : Function<dim>(n_components, time)
      , nu(viscosity)
    {}

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const;

  private:
    const double nu;
  };

  template <int dim>
  void
  InitialValues<dim>::vector_value(const Point<dim> &p,
                                   Vector<double> &  values) const
  {
    ExactSolution<dim>(dim + 1, this->get_time(), nu).vector_value(p, values);
  }



  // @sect4{Boundary values}
  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues(const unsigned int n_components = dim + 1,
                   const double       time         = 0.,
                   const double       viscosity    = 1.)
      : Function<dim>(n_components, time)
      , nu(viscosity)
    {}

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const;

  private:
    const double nu;
  };


  template <int dim>
  void
  BoundaryValues<dim>::vector_value(const Point<dim> &p,
                                    Vector<double> &  values) const
  {
    ExactSolution<dim>(dim + 1, this->get_time(), nu).vector_value(p, values);
  }



  // @sect3{Calculation of NS-relevant basis functions}

  // This function calculates the required basis
  // functions for the assembly process, including
  // stabilization terms as given in Y. Bazilevs,
  // V.M. Calo, J.A. Cottrell, T.J.R. Hughes,
  // A. Reali, G. Scovazzi: Variational multiscale
  // residual-based turbulence modeling for large eddy
  // simulation of incompressible flows;
  // <i>Comput. Methods Appl. Mech. Engrg.</i> 197
  // (2007), 173--201. These parameters are based on
  // local Green's functions representing the
  // Navier--Stokes differential operator (and its
  // inverse).
  //
  // The reason why we do this in an extra function
  // has mainly cosmetic reasons, since this code is
  // relatively lengthy.
  template <int dim, class SolutionVector>
  inline void
  get_fe(const FEValuesBase<dim> &                 fe_values,
         const SolutionVector &                    solution,
         const SolutionVector &                    solution_old_scaled,
         const std::vector<unsigned int> &         local_dof_indices,
         const double                              nu,
         const std::vector<Vector<double>> &       rhs_values,
         const double                              time_step_weight,
         const unsigned int                        stabilization,
         const double                              h,
         std::vector<std::vector<double>> &        phi_u,
         std::vector<std::vector<double>> &        phi_u_weight,
         std::vector<std::vector<Tensor<1, dim>>> &grad_phi_u,
         std::vector<std::vector<double>> &        gradT_phi_u,
         std::vector<std::vector<double>> &        div_phi_u_p,
         std::vector<std::vector<double>> &        residual_phi,
         std::vector<std::vector<Tensor<1, dim>>> &stab_grad_phi,
         std::vector<Tensor<1, dim>> &             func_rhs,
         std::vector<Tensor<1, dim>> &             stab_rhs)
  {
    const FEValuesExtractors::Vector velocities(0);

    const unsigned int dofs_per_cell = fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = fe_values.get_JxW_values().size();

    const double nu_sqrt = std::sqrt(nu);

    // We now do something terribly complicated:
    // we want to extract the degree of the
    // velocity interpolation, but do not want
    // to use that as an input parameter.
    // Hence, we extract the FE from
    // FEValuesBase<dim>, use the velocity
    // block and get the name. The result will be
    // something like FE_Q<2>(3) (for dim=2,
    // degree = 3). Then we do some C++ tricks
    // to extract the ninth character and finally
    // convert it to an integer.
    std::string element_name  = fe_values.get_fe().base_element(0).get_name();
    const unsigned int degree = atoi(&(element_name[8]));

    const double constant_inverse_estimate =
      (degree == 1) ? 24. : (244. + std::sqrt(9136.)) / 3.;

    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        // First calculate some square roots
        // of numbers we are going to use.
        const double weight      = fe_values.JxW(q);
        const double weight_sqrt = std::sqrt(weight);

        // @sect4{Evaluate solution at old time steps}
        //
        // Now we evaluate the velocity at
        // the old time step, the current one
        // and calculate the current residual.
        // This evaluation is similar to what
        // is done in the function
        // fe_values.get_function_values (and
        // the corresponding derivative
        // functions), but we do it manually
        // here since we're going to need a few
        // function evaluations and not
        // everything that is included by
        // these functions.
        //
        // The first thing to do is to
        // create some temporary variables
        // that will hold these function
        // evaluations. Then we proceed
        // by the actual procedure of the
        // function evaluations, which
        // is basically just a weighted sum over
        // all basis functions or its derivatives,
        // respectively.
        Tensor<1, dim> velocity;
        Tensor<1, dim> velocity_olds;
        Tensor<2, dim> velocity_grad;
        Tensor<1, dim> velocity_lapl;
        Tensor<1, dim> pressure_grad;
        double         velocity_div = 0.;

        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            const unsigned int component_k =
              fe_values.get_fe().system_to_component_index(k).first;
            const double sol_k = solution(local_dof_indices[k]);

            if (component_k < dim)
              {
                velocity[component_k] += fe_values.shape_value(k, q) * sol_k;
                velocity_olds[component_k] +=
                  fe_values.shape_value(k, q) *
                  solution_old_scaled(local_dof_indices[k]);
                velocity_grad[component_k] +=
                  fe_values.shape_grad(k, q) * sol_k;

                double tmp = 0.;
                for (unsigned int d = 0; d < dim; ++d)
                  tmp += fe_values.shape_hessian(k, q)[d][d];
                velocity_lapl[component_k] += tmp * sol_k;

                velocity_div += fe_values.shape_grad(k, q)[component_k] * sol_k;
              }
            else if (component_k == dim)
              pressure_grad += fe_values.shape_grad(k, q) * sol_k;
          }

        // Now we reshape the right hand side
        // function as a tensor (which will
        // be used further down).
        Tensor<1, dim> rhs;
        for (unsigned int d = 0; d < dim; ++d)
          rhs[d] = rhs_values[q](d);

        // @sect4{Calculation of stabilization parameters}
        //
        // The next step is to calculate the
        // stabilization parameters for SUPG/PSPG/LISC
        // stabilization, following the
        // definitions in Bazilevs et al.
        // These calculations involve the
        // inverse of the Jacobian matrix of
        // the transformation from unit to
        // real cell and, at their heart,
        // are based on an asymtotic
        // expansion of the Green's function
        // representation of the inverse of
        // the Navier-Stokes velocity
        // operator.
        double tau_supg, tau_lsic;
        {
          Tensor<1, dim> ones;
          for (unsigned int d = 0; d < dim; ++d)
            ones[d] = 1.;

          const Tensor<2, dim> inverse_jacobian = fe_values.inverse_jacobian(q);
          const Tensor<2, dim> g_matrix =
            inverse_jacobian * transpose(inverse_jacobian);
          const Tensor<1, dim> g_vector = transpose(inverse_jacobian) * ones;

          double uGu = 0., GG = 0., gg = 0.;
          for (unsigned int d = 0; d < dim; d++)
            {
              gg += g_vector[d] * g_vector[d];
              for (unsigned int e = 0; e < dim; e++)
                {
                  uGu += velocity[d] * g_matrix[d][e] * velocity[e];
                  GG += g_matrix[d][e] * g_matrix[d][e];
                }
            }

          tau_supg =
            1. / std::sqrt(4. * time_step_weight * time_step_weight + uGu +
                           constant_inverse_estimate * nu * nu * GG);

          if (tau_supg > 1e-8 && gg > 1e-8)
            tau_lsic = 1. / (tau_supg * gg);
          else
            tau_lsic = 1.;
        }
        const double tau_supg_sqrt = std::sqrt(tau_supg);
        const double tau_lsic_sqrt = std::sqrt(tau_lsic);


        // From the evaluated functions, we can construct
        // the residual of the current solution function.
        Tensor<1, dim> residual;
        if (stabilization > 3)
          {
            residual = (velocity * time_step_weight - velocity_olds +
                        velocity * velocity_grad - rhs) +
                       pressure_grad - nu * velocity_lapl;
            residual *= tau_supg * weight_sqrt;
          }

        // @sect4{Generation of NS basis functions}
        //
        // Now comes the actual loop over all cell dofs that
        // constructs the basis functions that are going to
        // be used in the calculations. Note that we already
        // incorporate most of the products at this points,
        // since this saves us from having to do that at the
        // inner loops of assembly where it would be more
        // expensive.
        //
        // Next comes a short description of the information
        // that the whole bunch of target arrays are filled
        // with. All numbers are multiplied by the square
        // root of the integration weight (JxW), since all
        // terms appear in products. $\phi$ denotes basis
        // functions for velocity and $\psi$ for pressure.
        //
        // <ul>
        // <li><code>phi_u</code>: represents the velocity
        // test function $\phi$
        //
        // <li><code>phi_u_weight</code>:
        // $\phi / dt + u \cdot \nabla \phi +
        // (\nabla \cdot u) \phi $
        //
        // <li><code>grad_phi_u</code>:
        // $ \sqrt{\nu} \nabla \phi $
        //
        // <li><code>gradT_phi_u</code>:
        // $ \sqrt{\nu} (\nabla \phi)^T $
        //
        // <li><code>div_phi_u_p</code> has the velocity part
        // $ \sqrt{\tau_{LSIC}} \nabla \cdot \phi$
        // and the pressure part
        // $ \psi / \sqrt{\tau_{LSIC}} $
        //
        // <li><code>residual_phi</code> contains the
        // residual part in the velocity dofs, i.e.
        // $ (-\nu \Delta \phi + u \cdot \nabla \phi + \phi/dt)
        // \sqrt{\tau_{SUPG}}$
        //
        // <li><code>stab_grad_phi</code> stabilization
        // (SUPG type) test function, for velocity as
        // $ (u \cdot \nabla \phi - res_u \tau_{SUPG} \phi
        // + u \cdot (\nabla \phi)^T) \sqrt{\tau_{SUPG}} $
        // (with $res_u$ denoting the residual in the
        // Navier-Stokes equations using the current
        // velocity)
        // and pressure as
        // $ \sqrt{\tau_{SUPG}}\nabla \psi $
        //
        //  </ul>
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            const unsigned int component_k =
              fe_values.get_fe().system_to_component_index(k).first;

            if (component_k < dim)
              {
                const double div_phi_u_k =
                  fe_values.shape_grad(k, q)[component_k] * weight_sqrt;
                const double u_grad_phi_k =
                  velocity * fe_values.shape_grad(k, q) * weight_sqrt;

                phi_u[k][q]        = fe_values.shape_value(k, q) * weight_sqrt;
                phi_u_weight[k][q] = phi_u[k][q] * time_step_weight;
                phi_u_weight[k][q] +=
                  u_grad_phi_k +
                  weight_sqrt * velocity_div * fe_values.shape_value(k, q);

                grad_phi_u[k][q] =
                  fe_values.shape_grad(k, q) * (weight_sqrt * nu_sqrt);
                gradT_phi_u[k][q] = div_phi_u_k * nu_sqrt;

                div_phi_u_p[k][q] = div_phi_u_k * tau_lsic_sqrt;

                residual_phi[k][q]  = 0;
                stab_grad_phi[k][q] = 0;
                if (stabilization > 3 || stabilization % 2 != 0)
                  {
                    for (unsigned int l = 0; l < dim; ++l)
                      residual_phi[k][q] -= fe_values.shape_hessian(k, q)[l][l];
                    residual_phi[k][q] *= nu * weight_sqrt;
                    residual_phi[k][q] +=
                      u_grad_phi_k + phi_u[k][q] * time_step_weight;
                    residual_phi[k][q] *= tau_supg_sqrt;

                    if (stabilization > 3)
                      {
                        stab_grad_phi[k][q][component_k] +=
                          u_grad_phi_k - residual * fe_values.shape_grad(k, q);
                        for (unsigned int d = 0; d < dim; ++d)
                          stab_grad_phi[k][q][d] +=
                            fe_values.shape_grad(k, q) * velocity * weight_sqrt;
                        stab_grad_phi[k][q] *= tau_supg_sqrt;
                      }
                  }
              }
            else if (component_k == dim)
              {
                div_phi_u_p[k][q] =
                  fe_values.shape_value(k, q) * weight_sqrt / tau_lsic_sqrt;
                if (stabilization % 2 != 0)
                  stab_grad_phi[k][q] =
                    fe_values.shape_grad(k, q) * weight_sqrt * tau_supg_sqrt;
              }
          }

        // Finally, construct the right hand side
        // function values, where the arrays are
        // given as follows:
        // <ul>
        // <li><code>func_rhs</code> contains
        // $u_{old}/dt + f_{rhs}$
        // <li><code>stab_rhs</code> contains
        // $(u_{old}/dt + f_{rhs}) \sqrt{\tau_{SUPG}}$
        // </ul>
        func_rhs[q] = (rhs + velocity_olds) * (weight_sqrt);
        stab_rhs[q] = func_rhs[q] * tau_supg_sqrt;
      }
  }



  // @sect3{NavierStokesProblemBase class implementation}

  // @sect4{NavierStokesProblemBase::NavierStokesProblemBase}

  // In the constructor of this class, we first store
  // the value that was passed in concerning the
  // degree of the finite elements we shall use (a
  // degree of one, for example, means to use Q2 and
  // Q1 elements for velocity and pressure, resp.),
  // and then construct the vector valued element
  // belonging to the space X_h, described in the
  // introduction of step-20. We set the desired time
  // stepping scheme, the time interval, viscosity,
  // output options and the stabilization.
  template <int dim>
  NavierStokesProblemBase<dim>::NavierStokesProblemBase()
    : degree(1)
    , fe(FE_Q<dim>(degree + 1), dim, FE_Q<dim>(degree), 1)
    , n_global_refinements(5)
    , dof_handler(triangulation)
    , time_stepping(BDF2)
    , time(0.)
    , time_final(0.5)
    , time_step(0.04)
    , nu(.1)
    , max_nl_iteration(10)
    , tol_nl_iteration(1e-5)
    , output_timestep_skip(1)
    , stabilization(0)
    , pressure_constraint(false)
    , assembly_type(Serial)
    , solver_type(GMRES)
    , preconditioner_type(ILU)
  {}



  // @sect4{NavierStokesProblemBase::make_grid_and_dofs}
  template <int dim>
  void
  NavierStokesProblemBase<dim>::make_grid_and_dofs()
  {
    // This next function starts out with functions
    // calls that create and refine a mesh, and then
    // associate degrees of freedom with it. Since our
    // program includes the measurement of computing
    // times of important steps of the program, the very
    // first command in this function starts the timer.
    //
    // Next, we generate a (-1,1)^dim cube and
    // refine it <code>n_global_refinements</code>
    // times, and associate the finite element
    // with it.
    Timer computing_timer;

    GridGenerator::hyper_cube(triangulation, -1, 1);
    /*GridGenerator::hyper_rectangle (triangulation, Point<dim>(-1,-1),
            Point<dim>(1,1));*/
    triangulation.refine_global(n_global_refinements);

    dof_handler.distribute_dofs(fe);

    // The degrees of freedom are renumbered using
    // King reordering from the boost library
    // in case we use an ILU preconditioner. The
    // deal.II documentation of the DoFRenumbering
    // class compares several choices of
    // renumbering strategies, where King ordering
    // turns out to be the best choice for
    // the stationary Stokes equations. The discretized
    // time-dependent Navier-Stokes equations result
    // in a similar system of equations, so it should
    // perform well in this case, too. However,
    // we need to take care regarding the high
    // memory consumption of the <code>boost</code>
    // functions, so we switch to a simpler
    // Cuthill_McKee algorithm (from the deal
    // library) in case we have a lot of
    // degrees of freedom.
    if (preconditioner_type == ILU && solver_type == GMRES)
      {
        if (dof_handler.n_dofs() < 600000)
          DoFRenumbering::boost::king_ordering(dof_handler);
        else
          DoFRenumbering::Cuthill_McKee(dof_handler);
      }

    // The next step is to construct a mean value
    // constraint on the pressure. This is only
    // necessary when there are no boundary conditions
    // set on the pressure, which is the case when
    // Dirichlet conditions on the velocity are set on
    // the whole domain boundary.
    //
    // Hence, we build a constraint that we set the
    // first pressure variable to the sum of all the
    // others on the boundary in analogy to what is done
    // in the step-11 tutorial program. We choose to
    // enforce only the mean pressure on the boundary
    // because this is much (!!)  faster in deal.II -
    // note that the standard constraint on all pressure
    // dofs, which would be another possibility from a
    // mathematical point of view, causes the
    // pressure-pressure matrix to be full!  We
    // emphasize that the current algorithm assumes a
    // grid that is refined uniformly.  For non-uniform
    // grids, additional work is necessary!
    //
    // The actual work done here first clears old
    // contents in the constraints and then proceeds by
    // extracting the pressure degrees of freedom, which
    // are in the end of the degrees of freedom list due
    // to the component-wise numbering.  This
    // construction is similar to what is done in the
    // deal.II tutorial step-11.
    Timer computing_timer2;
    hanging_node_and_pressure_constraints.clear();

    if (pressure_constraint)
      {
        std::vector<bool> pressure_dofs(dof_handler.n_dofs(), false);
        std::vector<bool> pressure_mask(dim + 1, false);
        pressure_mask[dim] = true;
        DoFTools::extract_boundary_dofs(static_cast<DoFHandler<dim> &>(
                                          dof_handler),
                                        pressure_mask,
                                        pressure_dofs);

        const unsigned int first_pressure_dof = std::distance(
          pressure_dofs.begin(),
          std::find(pressure_dofs.begin(), pressure_dofs.end(), true));

        // We want to check for hanging nodes as well when
        // adaptive grid refinement is used. Additionally,
        // we now impose the zero mean on the pressure by
        // setting the first variable in the pressure to the
        // sum of all the others. First we add a constraint
        // on the first pressure dof, and then we create
        // entries to all other pressure dofs on the
        // boundary.
        //
        // Then, finally, the constraint matrix for hanging
        // nodes and pressure constraints is set up.
        hanging_node_and_pressure_constraints.add_line(first_pressure_dof);

        for (unsigned int i = first_pressure_dof + 1; i < dof_handler.n_dofs();
             ++i)
          if (pressure_dofs[i] == true)
            {
              hanging_node_and_pressure_constraints.add_entry(
                first_pressure_dof, i, -1);
            }
      }

    DoFTools::make_hanging_node_constraints(
      dof_handler, hanging_node_and_pressure_constraints);
    hanging_node_and_pressure_constraints.close();
    comptimes(4) += computing_timer2.wall_time();
    computing_timer2.reset();

    // In order to figure out the sizes of the
    // respective blocks, we call the
    // <code>DoFTools::count_dofs_per_fe_block</code>
    // function that counts how many shape functions are
    // non-zero for a particular vector component. With
    // this done, we write some information about the
    // dofs and some equation info to the screen. Note
    // that the decision on which stabilization to use
    // is rather involved. As mentioned in the
    // declaration of the class members, a value of 0
    // and 5 for stabilization are the two reasonable
    // choices for no stabilization or full
    // stabilization.
    {
      std::vector<unsigned int> block_component(dim + 1, 0);
      block_component[dim] = 1;
      DoFRenumbering::component_wise(dof_handler, block_component);

      std::vector<unsigned int> dofs_per_block(2);
      dofs_per_block =
        DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

      std::cout << std::endl
                << " Calculation of laminar " << dim
                << "D Taylor/Beltrami flow." << std::endl
                << " Number of active cells: " << triangulation.n_active_cells()
                << "." << std::endl
                << " Number of degrees of freedom: " << dof_handler.n_dofs()
                << " (" << dofs_per_block[0] << " + " << dofs_per_block[1]
                << ")." << std::endl
                << " Enabled stabilizations SUPG/PSPG/LSIC: ";
      std::cout.precision(2);
      const bool tau_supg = (stabilization > 3) ? true : false;
      const bool tau_pspg = (stabilization % 2 == 0) ? false : true;
      const bool tau_lsic = (stabilization > 1) ? true : false;
      std::cout << tau_supg << " / " << tau_pspg << " / " << tau_lsic << "."
                << std::endl;
    }

    // The next step is to generate the actual sparsity
    // pattern. The idea is to rely on the class
    // CompressedSparsityPattern, which directly builds
    // up the sparsity pattern of the matrix without any
    // intermediate condensing. See the documentation of
    // step-22 and step-31 for more details on this
    // process. With that generation completed, the
    // system matrix can be derived from the sparsity
    // pattern.
    DynamicSparsityPattern csp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    csp,
                                    hanging_node_and_pressure_constraints,
                                    false);

    sparsity_pattern.copy_from(csp);

    system_matrix.reinit(sparsity_pattern);

    // Here, one can have a look at the newly generated
    // sparsity pattern.
    /*std::ofstream out ("sparsity_pattern.gpl");
    sparsity_pattern.print_gnuplot (out);*/

    // Eventually, we have to resize the solution and
    // right hand side vectors to the number of degrees
    // of freedom in the problem.
    solution_update.reinit(dof_handler.n_dofs());
    solution.reinit(dof_handler.n_dofs());
    solution_old.reinit(dof_handler.n_dofs());
    solution_old_scaled.reinit(dof_handler.n_dofs());
    exact.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    // Now we are done for this function, so we add the
    // time to the timer variable.
    comptimes(0) += computing_timer.wall_time();
  }



  // @sect4{NavierStokesProblemBase::assemble_system}
  template <int dim>
  void
  NavierStokesProblemBase<dim>::assemble_system()
  {
    // This function assembles the stiffness matrix in
    // the Navier-Stokes equations. Since we want to do
    // the assembly in parallel, this function is merely
    // a wrapper that distributes all work to the
    // function
    // <code>NavierStokesProblemBase::assemble_system_interval</code>
    // as soon as a start and end cell for the
    // individual chunks are determined.
    //
    // In the lines of the deal.II tutorial programs, we
    // first clear the matrix and right hand sides from
    // previous time steps, and then enter the
    // respective option of serial or parallel assembly
    // using a switch command.
    Timer computing_timer;

    system_matrix = 0;
    system_rhs    = 0;

    switch (assembly_type)
      {
        case Parallel:
          {
            AssertThrow(false, ExcNotImplemented());
            break;
          }
        case Serial:
          {
            // Serial assembly just runs the process from the
            // beginning to the end.
            NavierStokesProblemBase::assemble_system_interval(
              dof_handler.begin_active(), dof_handler.end());
            break;
          }
        default:
          {
          }
      }

    // When the matrix has been generated, we set the
    // constraints for hanging nodes (in the case of an
    // adaptively refined grid), see tutorial step-8.
    //
    // Then, we implement the Dirichlet boundary
    // conditions. Nodes that are subject to Dirichlet
    // boundary conditions get their node values
    // imposed. The process how this is done is quite
    // simple. First, the diagonal element in the matrix
    // corresponding to the current degree of freedom
    // (fixed by Dirichlet b.c.) will get a nonzero
    // value (the one that is already there or some
    // other value that is equally large as other
    // diagonal entries). By Gaussian elimination, all
    // nonzeros in this column are eliminated (giving
    // the resp. boundary influence to the right hand
    // side <code>system_rhs</code>), and then even all
    // entries in the row corresponding to that degree
    // of freedom are deleted.  See the deal.II
    // documentation on the boundary interpolation in
    // the VectorTools class and
    // <code>MatrixTools::apply_boundary_values</code>
    // for details.
    //
    // We build a component mask so that we can impose
    // Dirichlet boundary conditions only on the
    // velocity but not on the pressure. The pressure
    // component will simply be disabled in the
    // <code>bool</code> vector in case the constraint
    // is set.  For the Beltrami flow considered in this
    // program, this is actually not the case, so there
    // will still be Dirichlet conditions on pressure.
    Timer computing_timer2;
    hanging_node_and_pressure_constraints.condense(system_matrix);
    hanging_node_and_pressure_constraints.condense(system_rhs);


    std::map<unsigned int, double> boundary_values;
    std::vector<bool>              velocity_mask(dim + 1, true);

    if (pressure_constraint)
      velocity_mask[dim] = false;

    // Now we write the respective boundary values to
    // the variable <code>boundary_values</code> and
    // then impose these values to the global matrix and
    // right hand side.
    //
    // Then we are done and record the computing times.
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             BoundaryValues<dim>(dim + 1,
                                                                 time,
                                                                 nu),
                                             boundary_values,
                                             velocity_mask);
    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       solution_update,
                                       system_rhs);
    comptimes(5) += computing_timer2.wall_time();

    comptimes(1) += computing_timer.wall_time();
    compcounter(1) += 1;
  }


  // @sect4{NavierStokesProblemBase::assemble_system_interval}
  template <int dim>
  void
  NavierStokesProblemBase<dim>::assemble_system_interval(
    const typename DoFHandler<dim>::active_cell_iterator &beginc,
    const typename DoFHandler<dim>::active_cell_iterator &endc)
  {
    // The function that actually
    // assembles the linear system is
    // similar to the one in step-22
    // of the deal.II tutorial.
    // We allocate some variables
    // and build the structure
    // for the cell stiffness matrix
    // <code>local_matrix</code> and the
    // local right hand side vector
    // <code>local_rhs</code>.
    // As usual, <code>local_dof_indices</code>
    // contains the information on
    // the position of the cell
    // degrees of freedom in the
    // global system.
    QGauss<dim> quadrature_formula(degree + 2);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients | update_hessians |
                              update_inverse_jacobians |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);

    std::vector<unsigned int> local_dof_indices(dofs_per_cell);

    // We declare the objects that contain
    // the information on the right hand
    // side of the Navier-Stokes equations and
    // the boudary values on both velocity and pressure.
    // We need both a representation for the continuous
    // functions as well as their values
    // at the quadrature points in the cells and
    // on the faces, respectively.
    const RightHandSide<dim>    right_hand_side(dim, time);
    std::vector<Vector<double>> rhs_values(n_q_points, Vector<double>(dim));

    // Vectors to hold the evaluations of
    // the FE basis functions and derived
    // quantities in the Navier-Stokes
    // equations assembly. The names should
    // mostly be self-explaining. The terms
    // usually include also weights from
    // the quadrature formula, so one needs
    // to be careful when applying the
    // terms in other orders as the ones
    // used in the code.
    //
    // Next follow similar arrays for
    // the evalutions of the right hand
    // side at the quadrature points,
    // i.e. forcing function and information
    // from the old time step.
    std::vector<std::vector<double>> phi_u(dofs_per_cell,
                                           std::vector<double>(n_q_points));
    std::vector<std::vector<double>> phi_u_weight(
      dofs_per_cell, std::vector<double>(n_q_points));
    std::vector<std::vector<Tensor<1, dim>>> grad_phi_u(
      dofs_per_cell, std::vector<Tensor<1, dim>>(n_q_points));
    std::vector<std::vector<double>> gradT_phi_u(
      dofs_per_cell, std::vector<double>(n_q_points));
    std::vector<std::vector<double>> div_phi_u_p(
      dofs_per_cell, std::vector<double>(n_q_points));
    std::vector<std::vector<double>> residual_phi(
      dofs_per_cell, std::vector<double>(n_q_points));
    std::vector<std::vector<Tensor<1, dim>>> stab_grad_phi(
      dofs_per_cell, std::vector<Tensor<1, dim>>(n_q_points));

    std::vector<Tensor<1, dim>> func_rhs(n_q_points);
    std::vector<Tensor<1, dim>> stab_rhs(n_q_points);

    // With all this in place, we can
    // go on with the loop over all
    // cells and add the local contributions.
    //
    // The first thing to do is to
    // evaluate the FE basis functions
    // at the quadrature
    // points of the cell, as well as
    // derivatives and the other
    // quantities specified above.
    // Moreover, we need to reset
    // the local matrices and
    // right hand side before
    // filling them with new information
    // from the current cell.
    typename DoFHandler<dim>::active_cell_iterator cell;
    for (cell = beginc; cell != endc; ++cell)
      {
        fe_values.reinit(cell);
        local_matrix = 0;
        local_rhs    = 0;
        cell->get_dof_indices(local_dof_indices);
        const double h = cell->diameter();

        // Evalulate right hand side function,
        // density and viscosity at the
        // integration points. Then, we use
        // the finite element basis
        // functions to get the interpolated
        // velocity value at the quadrature
        // points. We do this for both
        // the current velocity and
        // the (weighted) old velocity.
        right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                          rhs_values);

        // For building the total matrix,
        // start by translating the general
        // basis functions into the relevant
        // information for the Navier-Stokes
        // system.
        get_fe(fe_values,
               solution,
               solution_old_scaled,
               local_dof_indices,
               nu,
               rhs_values,
               time_step_weight,
               stabilization,
               h,
               phi_u,
               phi_u_weight,
               grad_phi_u,
               gradT_phi_u,
               div_phi_u_p,
               residual_phi,
               stab_grad_phi,
               func_rhs,
               stab_rhs);

        // Loop over the cell dofs in
        // order to fill the local matrix
        // with information. Note that we
        // loop first over the dofs and
        // only at the innermost position
        // over the quadrature points. This
        // accelerates the code for this
        // example, since we need less
        // <code>if</code> statements in
        // order to determine which terms
        // we need to calculate for the
        // current result.
        //
        // We also include different assembly
        // options, depending on whether we
        // have stabilization turned on
        // or off. This gives us a code that
        // is more difficult to read, but it
        // avoids a performance penalty we'd
        // face otherwise when no stabilization
        // would be used.
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            const unsigned int component_i =
              fe.system_to_component_index(i).first;
            if (component_i < dim && stabilization < 2)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    const unsigned int component_j =
                      fe.system_to_component_index(j).first;

                    if (component_j < dim && component_i == component_j)
                      for (unsigned int q = 0; q < n_q_points; ++q)
                        local_matrix(i, j) +=
                          phi_u[i][q] * phi_u_weight[j][q] +
                          grad_phi_u[i][q] * grad_phi_u[j][q] +
                          gradT_phi_u[i][q] * gradT_phi_u[j][q];

                    else if (component_j < dim)
                      for (unsigned int q = 0; q < n_q_points; ++q)
                        local_matrix(i, j) +=
                          gradT_phi_u[i][q] * gradT_phi_u[j][q];

                    else if (component_j == dim)
                      for (unsigned int q = 0; q < n_q_points; ++q)
                        local_matrix(i, j) -=
                          div_phi_u_p[i][q] * div_phi_u_p[j][q];
                  }

                for (unsigned int q = 0; q < n_q_points; ++q)
                  local_rhs(i) += phi_u[i][q] * func_rhs[q][component_i];
              } /* end case for velocity dofs w/o stabilization */
            else if (component_i < dim)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    const unsigned int component_j =
                      fe.system_to_component_index(j).first;

                    if (component_j < dim && component_j == component_i)
                      for (unsigned int q = 0; q < n_q_points; ++q)
                        local_matrix(i, j) +=
                          phi_u[i][q] * phi_u_weight[j][q] +
                          grad_phi_u[i][q] * grad_phi_u[j][q] +
                          gradT_phi_u[i][q] * gradT_phi_u[j][q] +
                          div_phi_u_p[i][q] * div_phi_u_p[j][q] +
                          stab_grad_phi[i][q][component_j] * residual_phi[j][q];

                    else if (component_j < dim)
                      for (unsigned int q = 0; q < n_q_points; ++q)
                        local_matrix(i, j) +=
                          gradT_phi_u[i][q] * gradT_phi_u[j][q] +
                          div_phi_u_p[i][q] * div_phi_u_p[j][q] +
                          stab_grad_phi[i][q][component_j] * residual_phi[j][q];

                    else if (component_j == dim)
                      for (unsigned int q = 0; q < n_q_points; ++q)
                        local_matrix(i, j) +=
                          -div_phi_u_p[i][q] * div_phi_u_p[j][q] +
                          stab_grad_phi[i][q] * stab_grad_phi[j][q];
                  }

                for (unsigned int q = 0; q < n_q_points; ++q)
                  local_rhs(i) += phi_u[i][q] * func_rhs[q][component_i] +
                                  stab_grad_phi[i][q] * stab_rhs[q];
              } /* end case for velocity dofs w/ stabilization */
            else if (component_i == dim && stabilization % 2 == 0)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    const unsigned int component_j =
                      fe.system_to_component_index(j).first;
                    if (component_j < dim)
                      for (unsigned int q = 0; q < n_q_points; ++q)
                        local_matrix(i, j) +=
                          div_phi_u_p[i][q] * div_phi_u_p[j][q];
                  }
              } /* end case for pressure dofs w/o stabilization */
            else if (component_i == dim)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    const unsigned int component_j =
                      fe.system_to_component_index(j).first;
                    if (component_j < dim)
                      for (unsigned int q = 0; q < n_q_points; ++q)
                        local_matrix(i, j) +=
                          div_phi_u_p[i][q] * div_phi_u_p[j][q] +
                          stab_grad_phi[i][q][component_j] * residual_phi[j][q];
                    else if (component_j == dim)
                      for (unsigned int q = 0; q < n_q_points; ++q)
                        local_matrix(i, j) +=
                          stab_grad_phi[i][q] * stab_grad_phi[j][q];
                  }

                for (unsigned int q = 0; q < n_q_points; ++q)
                  local_rhs(i) += stab_grad_phi[i][q] * stab_rhs[q];
              } /* end case for pressure dofs w/ stabilization */
          }     /* end loop over i in dofs_per_cell */

        // The final stage in the loop over all cells is to
        // insert the local contributions into the global
        // stiffness matrix <code>system_matrix</code> and
        // the global right hand side
        // <code>system_rhs</code>. Since there may be
        // several processes running in parallel, we have to
        // aquire permission to write into the global matrix
        // by the <code>assembler_lock</code> variable (in
        // order to not access the global matrix by two
        // processes simulaneously and possibly lose data).
        hanging_node_and_pressure_constraints.distribute_local_to_global(
          local_matrix, local_dof_indices, system_matrix);
        hanging_node_and_pressure_constraints.distribute_local_to_global(
          local_rhs, local_dof_indices, system_rhs);
      }
  }



  // @sect4{NavierStokesProblemBase::solve}
  template <int dim>
  unsigned int
  NavierStokesProblemBase<dim>::solve()
  {
    unsigned int n_iterations = 0;
    Timer        computing_timer;
    switch (solver_type)
      {
          // For the iterative solvers, we
          // choose to use a unified interface
          // to access the solver. We first
          // extract the solver information
          // and then call an object of type
          // SolverSelector, which automatically
          // assigns the correct solver.
          // In case of GMRES, we also want
          // to use more than the standard
          // of 30 basis vector, so we need
          // to interfere once again at this
          // point. This is done by the function
          // <code>set_data</code>.
        case GMRES:
        case BiCGStab:
          {
            std::string solver_string;
            if (solver_type == GMRES)
              solver_string = "gmres";
            else
              solver_string = "bicgstab";

            SolverControl                  solver_control(system_matrix.m(),
                                         system_rhs.l2_norm() * 1e-10);
            SolverSelector<Vector<double>> solver_selector(solver_string,
                                                           solver_control);
            if (solver_type == GMRES)
              solver_selector.set_data(
                SolverGMRES<Vector<double>>::AdditionalData(120));

            switch (preconditioner_type)
              {
                case SSOR: /* SSOR preconditioner */
                  {
                    Timer              computing_timer2;
                    PreconditionSSOR<> preconditioner;
                    preconditioner.initialize(system_matrix, 1.33);
                    comptimes(6) += computing_timer2.wall_time();

                    solver_selector.solve(system_matrix,
                                          solution_update,
                                          system_rhs,
                                          preconditioner);
                    break;
                  }
                case ILU: /* Sparse ILU preconditioner */
                  {
                    Timer                             computing_timer2;
                    SparseILU<double>                 preconditioner;
                    SparseILU<double>::AdditionalData ilu_data;
                    preconditioner.initialize(system_matrix, ilu_data);
                    comptimes(6) += computing_timer2.wall_time();

                    solver_selector.solve(system_matrix,
                                          solution_update,
                                          system_rhs,
                                          preconditioner);
                    break;
                  }
                case Vanka: /* Vanka preconditioner: expensive, bad quality */
                  {
                    Timer         computing_timer2;
                    ComponentMask signature(3, false);
                    signature.set(dim, true);
                    const IndexSet selected_indices =
                      DoFTools::extract_dofs(dof_handler, signature);
                    std::vector<bool> selected_dofs(dof_handler.n_dofs(),
                                                    false);
                    for (const auto index : selected_indices)
                      selected_dofs[index] = true;
                    SparseVanka<double> preconditioner(system_matrix,
                                                       selected_dofs);
                    comptimes(6) += computing_timer2.wall_time();

                    solver_selector.solve(system_matrix,
                                          solution_update,
                                          system_rhs,
                                          preconditioner);
                    break;
                  }
                default: /* no preconditioner */
                  solver_selector.solve(system_matrix,
                                        solution_update,
                                        system_rhs,
                                        PreconditionIdentity());
              }

            n_iterations = solver_control.last_step();
            break;
          }
        case Direct: /* direct solve */
          {
            Timer               computing_timer2;
            SparseDirectUMFPACK A_direct;
            A_direct.initialize(system_matrix);
            comptimes(6) += computing_timer2.wall_time();
            A_direct.vmult(solution_update, system_rhs);
            n_iterations = 1;
            break;
          }
        default:
          {
            Assert(false, ExcNotImplemented());
          }
      }

    comptimes(2) += computing_timer.wall_time();
    compcounter(2) += 1;
    return n_iterations;
  }


  // @sect4{NavierStokesProblemBase::compute_errors}
  template <int dim>
  void
  NavierStokesProblemBase<dim>::compute_errors() const
  {
    // After we have dealt with the
    // linear solver and preconditioners,
    // we continue with the
    // implementation of our main
    // class. In particular, the next
    // task is to compute the errors in
    // our numerical solution, in both
    // the pressures as well as
    // velocities, in analogy to
    // step-20.
    //
    // To compute errors in the solution,
    // we have already introduced the
    // <code>VectorTools::integrate_difference</code>
    // function in step-7 and
    // step-11. However, there we only
    // dealt with scalar solutions,
    // whereas here we have a
    // vector-valued solution with
    // components that even denote
    // different quantities and may have
    // different orders of convergence,
    // a case that is quite usual for
    // mixed finite element programs.
    // What we therefore
    // have to do is to `mask' the
    // components that we are interested
    // in. This is easily done: the
    // <code>VectorTools::integrate_difference</code>
    // function takes as its last
    // argument a pointer to a weight
    // function (the parameter defaults
    // to the null pointer, meaning unit
    // weights). What we simply have to
    // do is to pass a function object
    // that equals one in the components
    // we are interested in, and zero in
    // the other ones. For example, to
    // compute the pressure error, we
    // should pass a function that
    // represents the constant vector
    // with a unit value in component
    // <code>dim</code>, whereas for the velocity
    // the constant vector should be one
    // in the first <code>dim</code> components,
    // and zero in the location of the
    // pressure.
    //
    // In deal.II, the
    // <code>ComponentSelectFunction</code> does
    // exactly this: it wants to know how
    // many vector components the
    // function it is to represent should
    // have (in our case this would be
    // <code>dim+1</code>, for the joint
    // velocity-pressure space) and which
    // individual or range of components
    // should be equal to one. We
    // therefore define two such masks at
    // the beginning of the function,
    // following by an object
    // representing the exact solution
    // and a vector in which we will
    // store the cellwise errors as
    // computed by
    // <code>integrate_difference</code>:
    const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);
    const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim),
                                                     dim + 1);

    Vector<double> cellwise_errors(triangulation.n_active_cells());

    // As already discussed in step-7,
    // we have to realize that it is
    // impossible to integrate the
    // errors exactly. All we can do is
    // approximate this integral using
    // quadrature. This actually
    // presents a slight twist here: if
    // we naively chose an object of
    // type <code>QGauss@<dim@>(degree+1)</code>
    // as one may be inclined to do
    // (this is what we used for
    // integrating the linear system),
    // one realizes that the error is
    // very small and does not follow
    // the expected convergence curves
    // at all. What is happening is
    // that for the mixed finite
    // elements used here, the Gauss
    // points happen to be
    // superconvergence points in which
    // the pointwise error is much
    // smaller (and converges with
    // higher order) than anywhere
    // else. These are therefore not
    // particularly good points for
    // ingration. To avoid this
    // problem, we simply use a
    // trapezoidal rule and iterate it
    // <code>degree+2</code> times in each
    // coordinate direction (again as
    // explained in step-7):
    QTrapez<1>     q_trapez;
    QIterated<dim> quadrature(q_trapez, degree + 3);

    // With this, we can then let the
    // library compute the errors and
    // output them to the screen:
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      ExactSolution<dim>(dim + 1, time, nu),
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &pressure_mask);
    const double p_l2_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      ExactSolution<dim>(dim + 1, time, nu),
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &velocity_mask);
    const double u_l2_error = cellwise_errors.l2_norm();

    std::cout.precision(4);
    std::cout << "  L2-Errors: ||e_p||_L2 = " << p_l2_error
              << ",   ||e_u||_L2 = " << u_l2_error << std::endl;
  }


  // @sect4{NavierStokesProblemBase::output_results}
  template <int dim>
  void
  NavierStokesProblemBase<dim>::output_results(
    const unsigned int timestep_number) const
  {
    const bool output_vtk = false;
    if (output_vtk == false)
      return;

    // The last interesting function is the one
    // in which we generate graphical
    // output. Everything here looks obvious
    // and familiar. Note how we construct
    // unique names for all the solution
    // variables at the beginning, like we did
    // in step-8 and other programs later
    // on. The only thing worth mentioning is
    // that for higher order elements, in seems
    // inappropriate to only show a single
    // bilinear quadrilateral per cell in the
    // graphical output. We therefore generate
    // patches of size (degree+1)x(degree+1) to
    // capture the full information content of
    // the solution. See the step-7 tutorial
    // program for more information on this.
    //
    // Some of the code in this function is
    // re-used from step-22, especially
    // regarding to output format of a
    // <code>vtk</code> file.
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.push_back("p");

    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim + 1, DataComponentInterpretation::component_is_scalar);
    for (unsigned int i = 0; i < dim; ++i)
      data_component_interpretation[i] =
        DataComponentInterpretation::component_is_part_of_vector;

    // First write out the numerical
    // solution.
    {
      data_out.add_data_vector(solution,
                               solution_names,
                               DataOut<dim>::type_dof_data,
                               data_component_interpretation);

      data_out.build_patches(degree + 1);

      std::ostringstream filename;
      filename << "numerical-" << Utilities::int_to_string(timestep_number, 3)
               << ".vtk";
      std::ofstream output(filename.str().c_str());
      data_out.write_vtk(output);
    }

    // Now, we write out the error
    // between the exact solution
    // and the numerical solution,
    // which sometimes can be interesting
    // to study.
    {
      data_out.add_data_vector(solution_update,
                               solution_names,
                               DataOut<dim>::type_dof_data,
                               data_component_interpretation);

      data_out.build_patches(degree + 1);

      std::ostringstream filename;
      filename << "difference-" << Utilities::int_to_string(timestep_number, 3)
               << ".vtk";
      std::ofstream output(filename.str().c_str());
      data_out.write_vtk(output);
    }
  }


  // @sect4{NavierStokesProblemBase::print_computing_times}
  template <int dim>
  void
  NavierStokesProblemBase<dim>::print_computing_times() const
  {
    // This function prints the various computing
    // times to the screen. To keep it short,
    // it generates a (nice) table that includes
    // assembly times and times for the solution
    // of the linear systems.
    const double total_time =
      comptimes(0) + comptimes(1) + comptimes(2) + comptimes(3);
    std::cout << "\n\n+-------------------------------------+--------------"
              << "+---------+\n"
              << "| Computing timer (CPU times)         |";
    std::cout.width(10);
    std::cout.precision(3);
    std::cout << total_time << " s  |    100% |\n";
    std::cout << "+-------------------------------------+--------------"
              << "+---------+\n"
              << "|    Setup system                     |";
    std::cout.width(10);
    std::cout.precision(3);
    std::cout << comptimes(0) << " s  | ";
    std::cout.precision(2);
    std::cout.width(6);
    std::cout << comptimes(0) / total_time * 100
              << "% |\n|     -> thereof pressure constraint  |";
    std::cout.width(10);
    std::cout.precision(3);
    std::cout << comptimes(4) << " s  | ";
    std::cout.width(6);
    std::cout.precision(1);
    std::cout << comptimes(4) / total_time * 100 << "% |\n|    Assembly of ";
    std::cout.width(7);
    std::cout << compcounter(1) << " matrices     |";
    std::cout.width(10);
    std::cout.precision(3);
    std::cout << comptimes(1) << " s  | ";
    std::cout.width(6);
    std::cout.precision(2);
    std::cout << comptimes(1) / total_time * 100
              << "% |\n|     -> thereof bc, pressure constr. |";
    std::cout.width(10);
    std::cout.precision(3);
    std::cout << comptimes(5) << " s  | ";
    std::cout.width(6);
    std::cout.precision(2);
    std::cout << comptimes(5) / total_time * 100
              << "% |\n|      average time per assembly:     |";
    std::cout.width(10);
    std::cout.precision(3);
    std::cout << comptimes(1) / compcounter(1)
              << " s  |     --- |\n|    Solving ";
    std::cout.width(7);
    std::cout << compcounter(2) << " linear systems   |";
    std::cout.width(10);
    std::cout.precision(3);
    std::cout << comptimes(2) << " s  | ";
    std::cout.width(6);
    std::cout.precision(2);
    std::cout << comptimes(2) / total_time * 100
              << "% |\n|     -> thereof preconditioner build |";
    std::cout.width(10);
    std::cout.precision(3);
    std::cout << comptimes(6) << " s  | ";
    std::cout.width(6);
    std::cout.precision(2);
    std::cout << comptimes(6) / total_time * 100
              << "% |\n|      average time per solve:        |";
    std::cout.width(10);
    std::cout.precision(3);
    std::cout << comptimes(2) / compcounter(2)
              << " s  |     --- |\n|    Manipulations, output, error cal |";
    std::cout.width(10);
    std::cout.precision(3);
    std::cout << comptimes(3) << " s  | ";
    std::cout.width(6);
    std::cout.precision(2);
    std::cout << comptimes(3) / total_time * 100
              << "% |\n+-------------------------------------+"
              << "--------------+---------+\n"
              << std::endl;
  }


  // @sect4{NavierStokesProblemBase::run}
  template <int dim>
  void
  NavierStokesProblemBase<dim>::run()
  {
    // This is the final function of our
    // main class. In this function, the actual
    // time integration and nonlinear iteration
    // is performed. Of course, we start by
    // by building up the grid and set the
    // initial values, and then proceed with the
    // time loop. During the calculations,
    // some information is written to file and
    // $L_2$-errors are computed.

    comptimes.reinit(7);
    compcounter.reinit(4);

    Timer computing_timer;

    make_grid_and_dofs();

    {
      VectorTools::interpolate(dof_handler,
                               InitialValues<dim>(dim + 1, time, nu),
                               solution);
      VectorTools::interpolate(dof_handler,
                               InitialValues<dim>(dim + 1, time, nu),
                               exact);
    }

    output_results(0);

    // Some variables for the time integration
    time_step_number                   = 0;
    bool         done                  = false;
    const double time_step_actual_size = time_step;

    // @sect5{Time loop}
    while (!done)
      {
        // We possibly use BDF-2 as time integration,
        // so in the first step, we have to do something
        // in a different way (we need two old solution
        // values for BDF-2, but we only have the
        // initial value). We use the backward Euler
        // with a very small time step then. Of course
        // we have to account for this modification
        // in the next step and 'reset' the actual time
        // step size. This will be done once the
        // work for the first time step is completed.
        if (time_stepping == BDF2 && time_step_number == 0)
          {
            time_step =
              std::min(0.25 * time_step_actual_size,
                       4. * time_step_actual_size * time_step_actual_size);
          }

        // Check whether we can already reach the final time.
        // If so, adjust the time step so that we hit the
        // target time exactly and set the variable
        // <code>done</code> to true, which will end
        // the <code>while</code> body once we're
        // through to the bottom.
        if (time + time_step > time_final - 1e-1 * time_step)
          {
            time_step = time_final - time;
            done      = true;
          }

        // Now that we have determined the correct time step,
        // we advance in time and increase the step counter.
        // Additionally, we update the values from old time
        // steps. In case of BDF2 integration, we take the
        // values from two previous time steps, weight them
        // accordingly and save it in
        // <code>solution_old_scaled</code>. In case of
        // backward Euler time integration, this is just the
        // previous solution divided by the time step size.
        // Additionally, we calculate an initial guess for
        // the nonlinear iteration that is based on an
        // extrapolation of the old values. Again, this is
        // somewhat more involved for BDF-2 than implicit
        // Euler. However, the quality of the initial
        // guess for the nonlinear iteration is
        // considerably improved from this
        // extrapolation than a naive use of the old
        // solution value.
        time += time_step;
        time_step_number += 1;
        if (time_stepping == BDF2 && time_step_number > 1)
          {
            // Calculate extrapolated solution
            // (= initial guess for nonlinear
            // iteration) at time n+1
            solution_update = solution;
            solution_update.sadd((time_step + time_step_old) / time_step_old,
                                 -time_step / time_step_old,
                                 solution_old);

            // Calculate scaled old solutions at n and n-1
            solution_old_scaled = solution;
            solution_old_scaled.sadd(
              (time_step + time_step_old) / (time_step * time_step_old),
              -time_step / (time_step_old * (time_step + time_step_old)),
              solution_old);

            // Calculate scaling for time n+1
            time_step_weight = (2. * time_step + time_step_old) /
                               (time_step * (time_step + time_step_old));
          }
        else
          {
            solution_old_scaled = solution;
            solution_old_scaled /= time_step;
            solution_update  = solution;
            time_step_weight = 1. / time_step;
          }

        // Now we got all necessary information from
        // the old time step, and are able to update the
        // solution vectors for the next step.
        solution_old = solution;
        solution     = solution_update;


        // Print some information about
        // the current time step to screen.
        std::cout << std::endl << "Time step #" << time_step_number << ", ";
        std::cout.precision(3);
        std::cout << "advancing to t = " << time << " "
                  << "(dt = " << time_step;
        std::cout << "). " << std::endl;


        // Now we prepare temporary variables for
        // the nonlinear iteration.
        // <code>solution_norm</code> will be used to
        // measure the initial size of the defect, and
        // <code>iteration_count</code> counts the nonlinear
        // iterations.
        double       solution_norm   = solution.l2_norm();
        unsigned int iteration_count = 0;
        std::cout << "  Nonlinear iteration [nl error / lin. its]:   "
                  << std::flush;


        // @sect5{Nonlinear iteration}
        //
        // We start the nonlinear iteration with
        // the assembly of the current matrix
        // system, which is then solved.
        // Hanging node constraints are, if present,
        // distributed after the solution process.
        do
          {
            iteration_count += 1;

            assemble_system();

            const unsigned int n_iterations = solve();

            hanging_node_and_pressure_constraints.distribute(solution_update);

            // We want to compute the difference between the
            // new solution and the one from the previous
            // iteration. We save the difference in the
            // variable system_rhs, which is not longer used
            // at this point anyway. Moreover, we use
            // a scaling when writing the difference between
            // the old and the new solution to the screen.
            system_rhs = solution_update;
            system_rhs.add(-1., solution);

            solution = solution_update;

            std::cout << '[';
            std::cout.precision(2);
            std::cout << system_rhs.l2_norm() / solution_norm << " / ";
            std::cout << n_iterations << "] " << std::flush;
          }
        // We have to iterate as long as the relative error is
        // larger than the prescribed tolerance and the maximum
        // number of allowed iterations is not exceeded.
        while (system_rhs.l2_norm() > tol_nl_iteration * solution_norm &&
               iteration_count < max_nl_iteration);

        std::cout << std::endl;

        // Now we are done with the nonlinear
        // iteration and finalize the time
        // step. First, we get the exact
        // solution at the current time, which
        // will then be used in the calculation
        // of the absolute L2-errors in
        // the velocity and the pressure.
        {
          VectorTools::interpolate(dof_handler,
                                   ExactSolution<dim>(dim + 1, time, nu),
                                   exact);
        }

        compute_errors();

        // Now we 'reset' to the actual time step size in case
        // we're doing the first time step in BDF-2.
        time_step_old = time_step;
        if (time_stepping == BDF2 && time_step_number == 1)
          {
            time_step = time_step_actual_size;
          }

        // We check whether we are at
        // a time step where to save
        // the current solution to a file.
        if (time_step_number % output_timestep_skip == 0)
          {
            solution_update -= exact;
            output_results(time_step_number / output_timestep_skip);
          }
      } /* End of the time loop */

    comptimes(3) =
      computing_timer.wall_time() - comptimes(0) - comptimes(1) - comptimes(2);

    print_computing_times();
  }

} // namespace StepNavierStokesBeltrami
