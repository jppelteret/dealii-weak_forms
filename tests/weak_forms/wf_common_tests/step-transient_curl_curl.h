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

// This header implements the transient version of a curl-curl problem.
// Only the magnetic vector potential is used (so, not coupled with an
// electric scalar potential). There is an option to use a gradient field
// to supply (div-free) source terms to the RHS of the curl-curl equation.
// It is used as a baseline for the weak form tests.
//
// Reference: Zaglmayr2006


#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nedelec_sz.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <mpi.h>

#include <algorithm>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <map>
#include <vector>

using namespace dealii;

namespace StepTransientCurlCurl
{
  // @sect3{Run-time parameters}
  //
  // There are several parameters that can be set in the code so we set up a
  // ParameterHandler object to read in the choices at run-time.
  namespace Parameters
  {
    // @sect4{Boundary conditions}

    struct BoundaryConditions
    {
      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };


    void
    BoundaryConditions::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary conditions");
      {}
      prm.leave_subsection();
    }

    void
    BoundaryConditions::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary conditions");
      {}
      prm.leave_subsection();
    }

    // @sect4{Time}

    struct TimeDiscretisation
    {
      double end_time;
      double delta_t;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    TimeDiscretisation::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time Discretisation");
      {
        prm.declare_entry("End time", "1.0", Patterns::Double(), "End time");

        prm.declare_entry("Time step size",
                          "0.1",
                          Patterns::Double(1e-9),
                          "Time step size");
      }
      prm.leave_subsection();
    }

    void
    TimeDiscretisation::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time Discretisation");
      {
        end_time = prm.get_double("End time");
        delta_t  = prm.get_double("Time step size");
      }
      prm.leave_subsection();
    }

    // @sect4{Source terms}

    struct SourceTerms
    {
      std::string wire_excitation;
      double      wire_current;
      double      wire_voltage;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    SourceTerms::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Source terms");
      {
        prm.declare_entry("Excitation method",
                          "Current",
                          Patterns::Selection("Current|Voltage"),
                          "Specification for how the wire is excited");

        prm.declare_entry("Wire current",
                          "1.0",
                          Patterns::Double(),
                          "Total current through the wire CSA (A/m2)");

        prm.declare_entry("Wire voltage drop",
                          "1.0",
                          Patterns::Double(),
                          "Total voltage drop across the wire length (V)");
      }
      prm.leave_subsection();
    }

    void
    SourceTerms::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Source terms");
      {
        wire_excitation = prm.get("Excitation method");
        wire_current    = prm.get_double("Wire current");
        wire_voltage    = prm.get_double("Wire voltage drop");
      }
      prm.leave_subsection();
    }

    // @sect4{Finite Element system}

    // As mentioned in the introduction, a different order interpolation should
    // be used for the displacement $\mathbf{u}$ than for the pressure
    // $\widetilde{p}$ and the dilatation $\widetilde{J}$.  Choosing
    // $\widetilde{p}$ and $\widetilde{J}$ as discontinuous (constant) functions
    // at the element level leads to the mean-dilatation method. The
    // discontinuous approximation allows $\widetilde{p}$ and $\widetilde{J}$ to
    // be condensed out and a classical displacement based method is recovered.
    // Here we specify the polynomial order used to approximate the
    // solution. The quadrature order should be adjusted accordingly, but
    // this is done at a later stage.
    struct FESystem
    {
      unsigned int poly_degree_min;
      unsigned int poly_degree_max;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };


    void
    FESystem::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        prm.declare_entry("Minimum polynomial degree",
                          "0",
                          Patterns::Integer(0),
                          "Magnetic vector potential system polynomial order");

        prm.declare_entry("Maximum polynomial degree",
                          "1",
                          Patterns::Integer(0),
                          "Magnetic vector potential system polynomial order");
      }
      prm.leave_subsection();
    }

    void
    FESystem::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        poly_degree_min = prm.get_integer("Minimum polynomial degree");
        poly_degree_max = prm.get_integer("Maximum polynomial degree");
      }
      prm.leave_subsection();
    }

    // @sect4{Geometry}

    struct Geometry
    {
      std::string  discretisation_type;
      double       radius_wire;
      double       side_length_surroundings;
      double       grid_scale;
      unsigned int n_divisions_radial;
      unsigned int n_divisions_longitudinal;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    Geometry::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        prm.declare_entry(
          "Discretisation type",
          "Exact",
          Patterns::Selection("Cartesian|Exact"),
          "Build the mesh with a Cartesian grid or with a smooth surface capturing the matrix-particle material interface");

        prm.declare_entry("Wire radius",
                          "0.5",
                          Patterns::Double(0.0),
                          "Radius of the current-carrying wire");

        prm.declare_entry(
          "Surrounding side length",
          "2.0",
          Patterns::Double(0.0),
          "Overall length of the bounding box for the material surrounding the wire");

        prm.declare_entry("Grid scale",
                          "1.0",
                          Patterns::Double(0.0),
                          "Global grid scaling factor");

        prm.declare_entry("Number of radial divisions",
                          "3",
                          Patterns::Integer(1),
                          "Discretisation: Elements along wire radius");

        // Must have at least one set of active edges in the domain centre
        prm.declare_entry("Number of longitudinal divisions",
                          "3",
                          Patterns::Integer(2),
                          "Discretisation: Elements along wire length");
      }
      prm.leave_subsection();
    }

    void
    Geometry::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        discretisation_type      = prm.get("Discretisation type");
        radius_wire              = prm.get_double("Wire radius");
        side_length_surroundings = prm.get_double("Surrounding side length");
        grid_scale               = prm.get_double("Grid scale");
        n_divisions_radial = prm.get_integer("Number of radial divisions");
        n_divisions_longitudinal =
          prm.get_integer("Number of longitudinal divisions");
      }
      prm.leave_subsection();
    }

    // @sect4{Refinement}

    struct Refinement
    {
      std::string  refinement_strategy;
      unsigned int n_global_refinements;
      unsigned int n_cycles_max;
      unsigned int n_levels_max;
      double       frac_refine;
      double       frac_coarsen;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };


    void
    Refinement::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Refinement");
      {
        prm.declare_entry("Refinement strategy",
                          "h-AMR",
                          Patterns::Selection(
                            "h-GMR|p-GMR|h-AMR|p-AMR"), // hp-AMR
                          "Strategy used to perform hp refinement");

        prm.declare_entry("Initial global refinements",
                          "1",
                          Patterns::Integer(0),
                          "Initial global refinement level");

        prm.declare_entry("Maximum cycles",
                          "10",
                          Patterns::Integer(0),
                          "Maximum number of h-refinement cycles");

        prm.declare_entry(
          "Maximum h-level",
          "6",
          Patterns::Integer(0, 10),
          "Number of h-refinement levels in the discretisation");

        prm.declare_entry("Refinement fraction",
                          "0.3",
                          Patterns::Double(0.0, 1.0),
                          "Fraction of cells to refine");

        prm.declare_entry("Coarsening fraction",
                          "0.03",
                          Patterns::Double(0.0, 1.0),
                          "Fraction of cells to coarsen");
      }
      prm.leave_subsection();
    }

    void
    Refinement::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Refinement");
      {
        refinement_strategy  = prm.get("Refinement strategy");
        n_global_refinements = prm.get_integer("Initial global refinements");
        n_cycles_max         = prm.get_integer("Maximum cycles");
        n_levels_max         = prm.get_integer("Maximum h-level");
        frac_refine          = prm.get_double("Refinement fraction");
        frac_coarsen         = prm.get_double("Coarsening fraction");
      }
      prm.leave_subsection();
    }

    // @sect4{Materials}

    struct Materials
    {
      double mu_r_surroundings;
      double mu_r_wire;
      double sigma_surroundings;
      double sigma_wire;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    Materials::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material properties");
      {
        prm.declare_entry("Wire relative permeability",
                          "1.0",
                          Patterns::Double(1e-9),
                          "Relative permeability of wire");

        prm.declare_entry("Wire electrical conductivity",
                          "1.0e6",
                          Patterns::Double(1e-9),
                          "Electrical conductivity of wire");

        prm.declare_entry("Surrounding relative permeability",
                          "1.0",
                          Patterns::Double(1e-9),
                          "Relative permeability of surrounding material");

        prm.declare_entry("Surrounding electrical conductivity",
                          "1.0",
                          Patterns::Double(1e-9),
                          "Electrical conductivity of surrounding material");
      }
      prm.leave_subsection();
    }

    void
    Materials::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material properties");
      {
        mu_r_wire         = prm.get_double("Wire relative permeability");
        mu_r_surroundings = prm.get_double("Surrounding relative permeability");
        sigma_wire        = prm.get_double("Wire electrical conductivity");
        sigma_surroundings =
          prm.get_double("Surrounding electrical conductivity");
      }
      prm.leave_subsection();
    }

    // @sect4{Linear solver}

    // Next, we choose both solver and preconditioner settings.  The use of an
    // effective preconditioner is critical to ensure convergence when a large
    // nonlinear motion occurs within a Newton increment.
    struct LinearSolver
    {
      std::string lin_slvr_type;
      double      lin_slvr_tol;
      double      lin_slvr_max_it;
      std::string preconditioner_type;
      double      preconditioner_relaxation;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    LinearSolver::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Linear solver");
      {
        prm.declare_entry("Solver type",
                          "Iterative",
                          Patterns::Selection("Iterative|Direct"),
                          "Type of solver used to solve the linear system");

        prm.declare_entry("Residual",
                          "1e-6",
                          Patterns::Double(0.0),
                          "Linear solver residual (scaled by residual norm)");

        prm.declare_entry(
          "Max iteration multiplier",
          "1",
          Patterns::Double(0.0),
          "Linear solver iterations (multiples of the system matrix size)");

        prm.declare_entry("Preconditioner type",
                          "ssor",
                          Patterns::Selection("jacobi|ssor|AMG"),
                          "Type of preconditioner");

        prm.declare_entry("Preconditioner relaxation",
                          "0.65",
                          Patterns::Double(0.0),
                          "Preconditioner relaxation value");
      }
      prm.leave_subsection();
    }

    void
    LinearSolver::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Linear solver");
      {
        lin_slvr_type             = prm.get("Solver type");
        lin_slvr_tol              = prm.get_double("Residual");
        lin_slvr_max_it           = prm.get_double("Max iteration multiplier");
        preconditioner_type       = prm.get("Preconditioner type");
        preconditioner_relaxation = prm.get_double("Preconditioner relaxation");
      }
      prm.leave_subsection();
    }

    // @sect4{All parameters}

    // Finally we consolidate all of the above structures into a single
    // container that holds all of our run-time selections.
    struct AllParameters : public BoundaryConditions,
                           public TimeDiscretisation,
                           public SourceTerms,
                           public FESystem,
                           public Geometry,
                           public Refinement,
                           public Materials,
                           public LinearSolver

    {
      AllParameters(const std::string &input_file);

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);

      bool
      use_voltage_excitation() const;

      bool
      use_current_excitation() const;
    };

    AllParameters::AllParameters(const std::string &input_file)
    {
      ParameterHandler prm;
      declare_parameters(prm);
      // prm.read_input(input_file);
      prm.parse_input(input_file);
      parse_parameters(prm);
    }

    void
    AllParameters::declare_parameters(ParameterHandler &prm)
    {
      BoundaryConditions::declare_parameters(prm);
      TimeDiscretisation::declare_parameters(prm);
      SourceTerms::declare_parameters(prm);
      FESystem::declare_parameters(prm);
      Geometry::declare_parameters(prm);
      Refinement::declare_parameters(prm);
      Materials::declare_parameters(prm);
      LinearSolver::declare_parameters(prm);
    }

    void
    AllParameters::parse_parameters(ParameterHandler &prm)
    {
      BoundaryConditions::parse_parameters(prm);
      TimeDiscretisation::parse_parameters(prm);
      SourceTerms::parse_parameters(prm);
      FESystem::parse_parameters(prm);
      Geometry::parse_parameters(prm);
      Refinement::parse_parameters(prm);
      Materials::parse_parameters(prm);
      LinearSolver::parse_parameters(prm);
    }

    bool
    AllParameters::use_voltage_excitation() const
    {
      if (wire_excitation == "Voltage")
        return true;

      Assert(wire_excitation == "Current",
             ExcMessage("Unknown wire excitation option selected."));
      return false;
    }

    bool
    AllParameters::use_current_excitation() const
    {
      return !use_voltage_excitation();
    }
  } // namespace Parameters

  // @sect3{Geometry}
  template <int dim>
  struct Geometry
  {
    Geometry(const double &    radius_wire,
             const double &    side_length,
             const Point<dim> &direction,
             const Point<dim> &point_on_axis)
      : radius_wire(radius_wire)
      , side_length(side_length)
      , direction(direction)
      , point_on_axis(point_on_axis)
    {
      AssertThrow(std::abs(direction.norm() - 1.0) < 1e-6,
                  ExcMessage("Not a unit vector"));
    }

    virtual ~Geometry()
    {}

    bool
    within_wire(const Point<dim> &p, const double &tol = 1e-6) const
    {
      const Point<dim - 1> p1(p[0], p[1]);
      return p1.square() <= (radius_wire * radius_wire - tol);
    }

    double
    wire_CSA() const
    {
      return M_PI * radius_wire * radius_wire;
    }

    const double     radius_wire;
    const double     side_length;
    const Point<dim> direction;
    const Point<dim> point_on_axis;
  };

  // @sect3{Nonconstant coefficients}

  template <int dim>
  class PermeabilityCoefficient : public Function<dim>
  {
  public:
    PermeabilityCoefficient(const Geometry<dim> &geometry,
                            const double &       mu_r_surroundings,
                            const double &       mu_r_wire)
      : Function<dim>()
      , geometry(geometry)
      , mu_0(4 * M_PI * 1e-7)
      , mu_r_surroundings(mu_r_surroundings)
      , mu_r_wire(mu_r_wire)
      , mu_surroundings(mu_r_surroundings * mu_0)
      , mu_wire(mu_r_wire * mu_0)
    {}

    virtual ~PermeabilityCoefficient()
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const;

    virtual void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<double> &          values,
               const unsigned int             component = 0) const;

    const Geometry<dim> &geometry;
    const double         mu_0;
    const double         mu_r_surroundings;
    const double         mu_r_wire;
    const double         mu_surroundings;
    const double         mu_wire;
  };



  template <int dim>
  double
  PermeabilityCoefficient<dim>::value(const Point<dim> &p,
                                      const unsigned int) const
  {
    if (geometry.within_wire(p) == true)
      return mu_wire;
    else
      return mu_surroundings;
  }



  template <int dim>
  void
  PermeabilityCoefficient<dim>::value_list(
    const std::vector<Point<dim>> &points,
    std::vector<double> &          values,
    const unsigned int             component) const
  {
    const unsigned int n_points = points.size();

    Assert(values.size() == n_points,
           ExcDimensionMismatch(values.size(), n_points));

    Assert(component == 0, ExcIndexRange(component, 0, 1));

    for (unsigned int i = 0; i < n_points; ++i)
      {
        if (geometry.within_wire(points[i]) == true)
          values[i] = mu_wire;
        else
          values[i] = mu_surroundings;
      }
  }

  template <int dim>
  class ConductivityCoefficient : public Function<dim>
  {
  public:
    ConductivityCoefficient(const Geometry<dim> &geometry,
                            const double &       sigma_surroundings,
                            const double &       sigma_wire)
      : Function<dim>()
      , geometry(geometry)
      , sigma_surroundings(sigma_surroundings)
      , sigma_wire(sigma_wire)
    {}

    virtual ~ConductivityCoefficient()
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const;

    virtual void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<double> &          values,
               const unsigned int             component = 0) const;

    const Geometry<dim> &geometry;
    const double         sigma_surroundings;
    const double         sigma_wire;
  };



  template <int dim>
  double
  ConductivityCoefficient<dim>::value(const Point<dim> &p,
                                      const unsigned int) const
  {
    if (geometry.within_wire(p) == true)
      return sigma_wire;
    else
      return sigma_surroundings;
  }



  template <int dim>
  void
  ConductivityCoefficient<dim>::value_list(
    const std::vector<Point<dim>> &points,
    std::vector<double> &          values,
    const unsigned int             component) const
  {
    const unsigned int n_points = points.size();

    Assert(values.size() == n_points,
           ExcDimensionMismatch(values.size(), n_points));

    Assert(component == 0, ExcIndexRange(component, 0, 1));

    for (unsigned int i = 0; i < n_points; ++i)
      {
        if (geometry.within_wire(points[i]) == true)
          values[i] = sigma_wire;
        else
          values[i] = sigma_surroundings;
      }
  }


  // @sect3{Source terms}
  // Note: J_f must be divergence free!

  template <int dim>
  class SourceFreeCurrentDensity : public TensorFunction<1, dim>
  {
  public:
    SourceFreeCurrentDensity(const Geometry<dim> &geometry,
                             const double &       wire_current)
      : TensorFunction<1, dim>()
      , geometry(geometry)
      , wire_current(wire_current)
      , free_current_density(wire_current / geometry.wire_CSA())
    {}

    virtual ~SourceFreeCurrentDensity()
    {}

    virtual Tensor<1, dim>
    value(const Point<dim> &p) const override;

    const Geometry<dim> &geometry;
    const double         wire_current;         // I
    const double         free_current_density; // J
  };

  template <>
  Tensor<1, 3>
  SourceFreeCurrentDensity<3>::value(const Point<3> &p) const
  {
    if (geometry.within_wire(p) == true)
      {
        return free_current_density * geometry.direction;
      }
    else
      {
        return Tensor<1, 3>();
      }
  }

  // @sect3{Analytical solution}
  // Analytical solution to a linearly magnetisable
  // straight wire, aligned in the z-direction,
  // immersed in a medium of infinite size.
  // The fundamentals required to derive the solution
  // can be found in:
  // [Griffiths1999a] Griffiths, D. J. & College, R.
  // Introduction to electrodynamics. Prentice Hall, 1999
  // The analytical solution is the solution to questions
  // 5.25 on page 239. Note that q5.22 is loosely related...

  template <int dim>
  class AnalyticalSolution : public Function<dim>
  {
  public:
    enum e_Field
    {
      MVP, // Magnetic vector potential
      MI   // Magnetic induction
    };

    AnalyticalSolution(const Geometry<dim> &                geometry,
                       const PermeabilityCoefficient<dim> & coefficient,
                       const SourceFreeCurrentDensity<dim> &source_free_current,
                       const e_Field &                      field)
      : Function<dim>(dim)
      , geometry(geometry)
      , coefficient(coefficient)
      , source_free_current(source_free_current)
      , field(field)
    {}

    virtual ~AnalyticalSolution()
    {}

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const;

    //  virtual void
    //  vector_gradient (const Point<dim> & p,
    //                   std::vector< Tensor<1,dim> > &gradients) const;

    const Geometry<dim> &                geometry;
    const PermeabilityCoefficient<dim> & coefficient;
    const SourceFreeCurrentDensity<dim> &source_free_current;
    const e_Field                        field;
  };

  template <int dim>
  void
  AnalyticalSolution<dim>::vector_value(const Point<dim> &,
                                        Vector<double> &) const
  {
    AssertThrow(false, ExcInternalError());
  }

  template <>
  void
  AnalyticalSolution<3>::vector_value(const Point<3> &p,
                                      Vector<double> &values) const
  {
    const unsigned int dim = 3;
    Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));
    values = 0.0;

    // Compute the radial distance from the wire
    const Tensor<1, dim> diff = p - geometry.point_on_axis;
    const Point<dim>     proj_axis =
      geometry.point_on_axis + (diff * geometry.direction) * geometry.direction;
    const Tensor<1, dim> r_vec = p - proj_axis;
    const double         r     = r_vec.norm();

    const Tensor<1, dim> &M = geometry.direction;
    const double &        R = geometry.radius_wire;
    const double &        I = source_free_current.wire_current;

    if (field == e_Field::MVP)
      {
        const double c =
          (geometry.within_wire(p) == true ?
             -coefficient.mu_wire * I / (4.0 * M_PI * R * R) * (r * r - R * R) :
             -coefficient.mu_surroundings * I / (2.0 * M_PI) * std::log(r / R));

        // A = A(r) = A(x,y) = c(r).M
        for (unsigned int d = 0; d < dim; ++d)
          values[d] = c * M[d];
      }
    else // e_Field::MI
      {
        // B = grad x A(x,y)
        //   = (dA/dy - dA/dz).e_i + (dA/dz - dA/dx).e_j + (dA/dx - dA/dy).e_k
        //   = (dA/dy).e_i + (-dA/dx).e_j + (dA/dx - dA/dy).e_k

        const double &x = p[0];
        const double &y = p[1];
        //    const double yz = y;
        //    const double theta = std::atan2(yz,x);
        //    const double cos_theta = std::cos(theta);
        //    const double sin_theta = std::sin(theta);

        const double dc_dr =
          (geometry.within_wire(p) == true ?
             -coefficient.mu_wire * I / (4.0 * M_PI * R * R) * (2.0 * r) :
             -coefficient.mu_surroundings * I / (2.0 * M_PI) * (1.0 / r));

        const double dr_dx = x / r;
        const double dr_dy = y / r;
        //    const double dtheta_dx = 1.0/(1.0+(y*y)/(x*x))*(-y/(x*x));
        //    const double dtheta_dy = 1.0/(1.0+(y*y)/(x*x))*(1.0/x);

        values[0] = dc_dr * dr_dy;  //*M[0];
        values[1] = -dc_dr * dr_dx; //*M[1];
        values[2] = 0.0;            // dc_dr*(dr_dx - dr_dy); //*M[2];
      }
  }

  template <int dim>
  class RefinementStrategy
  {
  public:
    RefinementStrategy(const std::string &refinement_strategy)
      : _use_h_refinement(refinement_strategy == "h-GMR" ||
                          refinement_strategy == "h-AMR" ||
                          refinement_strategy == "hp-AMR")
      , _use_p_refinement(refinement_strategy == "p-GMR" ||
                          refinement_strategy == "p-AMR" ||
                          refinement_strategy == "hp-AMR")
      , _use_AMR(refinement_strategy == "h-AMR" ||
                 refinement_strategy == "p-AMR")
    {}

    bool
    use_h_refinement(void) const
    {
      return _use_h_refinement;
    }
    bool
    use_p_refinement(void) const
    {
      return _use_p_refinement;
    }
    bool
    use_hp_refinement(void) const
    {
      Assert(!(use_h_refinement() && use_p_refinement()),
             ExcMessage("hp refinement not implemented."));
      return false;
    }
    bool
    use_AMR(void) const
    {
      return _use_AMR;
    }
    bool
    use_GR(void) const
    {
      return !use_AMR();
    }
    //  unsigned int n_global_refinements (void);
    unsigned int
    n_cycles_max(void);

  private:
    const bool _use_h_refinement;
    const bool _use_p_refinement;
    const bool _use_AMR;
  };

  //// Two dimensions
  // template <>
  // unsigned int
  // GridRefinementStrategy<2>::n_global_refinements (void)
  //{return 4;}

  template <>
  unsigned int
  RefinementStrategy<2>::n_cycles_max(void)
  {
    return use_GR() == true ? 5 : 8;
  }
  // { return use_GR() == true ?  8 : 8; }

  //// Three dimensions
  // template <>
  // unsigned int
  // GridRefinementStrategy<3>::n_global_refinements (void)
  //{return 2;}

  template <>
  unsigned int
  RefinementStrategy<3>::n_cycles_max(void)
  {
    return use_GR() == true ? 5 : 8;
  }
  // { return use_GR() == true ?  4 : 6; }
  // { return use_GR() == true ?  6 : 6; }


  // @sect3{The <code>StepTransientCurlCurl_Base</code> class template}

  template <int dim>
  class StepTransientCurlCurl_Base
  {
  public:
    StepTransientCurlCurl_Base(const std::string &input_file);
    ~StepTransientCurlCurl_Base();

    void
    run();

  protected:
    void
    setup_system();

    void
    verify_source_terms() const;

    virtual void
    assemble_system_esp(TrilinosWrappers::SparseMatrix & system_matrix_esp,
                        TrilinosWrappers::MPI::Vector &  system_rhs_esp,
                        const AffineConstraints<double> &constraints_esp) = 0;

    void
    solve_esp(const TrilinosWrappers::SparseMatrix &system_matrix_esp,
              TrilinosWrappers::MPI::Vector &       solution_esp,
              const TrilinosWrappers::MPI::Vector & system_rhs_esp,
              const AffineConstraints<double> &     constraints_esp,
              const IndexSet &                      locally_owned_dofs_esp,
              const DoFHandler<dim> &               dof_handler_esp);

    virtual void
    assemble_system_mvp(TrilinosWrappers::SparseMatrix & system_matrix_mvp,
                        TrilinosWrappers::MPI::Vector &  system_rhs_mvp,
                        const AffineConstraints<double> &constraints_mvp) = 0;

    void
    solve_mvp(const TrilinosWrappers::SparseMatrix &system_matrix_mvp,
              TrilinosWrappers::MPI::Vector &       solution_mvp,
              const TrilinosWrappers::MPI::Vector & system_rhs_mvp,
              const AffineConstraints<double> &     constraints_mvp,
              const IndexSet &                      locally_owned_dofs_mvp,
              const DoFHandler<dim> &               dof_handler_mvp);

    void
    solve_linear_system(const TrilinosWrappers::SparseMatrix &system_matrix,
                        TrilinosWrappers::MPI::Vector &       solution,
                        const TrilinosWrappers::MPI::Vector & system_rhs,
                        const AffineConstraints<double> &     constraints,
                        const IndexSet &       locally_owned_dofs,
                        const DoFHandler<dim> &dof_handler);

    void
    verify_solution_mvp() const;

    void
    make_grid();

    void
    set_material_and_boundary_ids();

    void
    refine_grid();

    void
    output_grid(const unsigned int timestep, const unsigned int cycle) const;

    void
    output_results(const unsigned int timestep, const unsigned int cycle) const;

    void
    output_point_solution_mvp(const unsigned int timestep,
                              const unsigned int cycle) const;

    MPI_Comm                   mpi_communicator;
    const unsigned int         n_mpi_processes;
    const unsigned int         this_mpi_process;
    mutable ConditionalOStream pcout;
    mutable TimerOutput        computing_timer;

    Parameters::AllParameters parameters;

    const types::manifold_id refinement_manifold_id;
    CylindricalManifold<dim> surface_description;

    Triangulation<dim>      triangulation;
    RefinementStrategy<dim> refinement_strategy;

    const unsigned int  degree;
    const unsigned int  degree_offset;
    const MappingQ<dim> mapping;

    // MVP system
    const FE_NedelecSZ<dim> fe_mvp;
    const QGauss<dim>       qf_cell_mvp;
    const QGauss<dim - 1>   qf_face_mvp;
    DoFHandler<dim>         dof_handler_mvp;

    const unsigned int               mvp_group;
    const unsigned int               first_mvp_dof;
    const FEValuesExtractors::Vector mvp_extractor;

    IndexSet                  locally_owned_dofs_mvp;
    IndexSet                  locally_relevant_dofs_mvp;
    AffineConstraints<double> constraints_mvp;

    TrilinosWrappers::SparseMatrix system_matrix_mvp;
    TrilinosWrappers::MPI::Vector  system_rhs_mvp;
    TrilinosWrappers::MPI::Vector  solution_mvp;
    TrilinosWrappers::MPI::Vector  solution_mvp_t1;
    TrilinosWrappers::MPI::Vector  d_solution_mvp_dt;

    // ESP system
    const FE_Q<dim>   fe_esp;
    const QGauss<dim> qf_cell_esp;
    DoFHandler<dim>   dof_handler_esp;

    const unsigned int               esp_group;
    const unsigned int               first_esp_dof;
    const FEValuesExtractors::Scalar esp_extractor;

    IndexSet                  locally_owned_dofs_esp;
    IndexSet                  locally_relevant_dofs_esp;
    AffineConstraints<double> constraints_esp;

    TrilinosWrappers::SparseMatrix system_matrix_esp;
    TrilinosWrappers::MPI::Vector  system_rhs_esp;
    TrilinosWrappers::MPI::Vector  solution_esp;

    // Other
    const Geometry<dim>           geometry;
    PermeabilityCoefficient<dim>  function_material_permeability_coefficients;
    ConductivityCoefficient<dim>  function_material_conductivity_coefficients;
    SourceFreeCurrentDensity<dim> function_free_current_density;
    AnalyticalSolution<dim>       function_analytical_solution_mvp_MI;
  };


  // @sect3{The <code>StepTransientCurlCurl_Base</code> class implementation}

  // @sect4{StepTransientCurlCurl_Base::StepTransientCurlCurl_Base}

  template <int dim>
  StepTransientCurlCurl_Base<dim>::StepTransientCurlCurl_Base(
    const std::string &input_file)
    : mpi_communicator(MPI_COMM_WORLD)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
    , pcout(std::cout, this_mpi_process == 0)
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::never,
                      TimerOutput::wall_times)
    , parameters(input_file)
    , refinement_manifold_id(1)
    , surface_description(Point<dim>(0, 0, 1) /*direction*/,
                          Point<dim>(0, 0, 0) /*point_on_axis*/)
    , triangulation(Triangulation<dim>::maximum_smoothing)
    , refinement_strategy(parameters.refinement_strategy)
    , degree(parameters.poly_degree_min)
    , degree_offset(
        1) // Edge elements are weird... Account for Nedelec degree order
    , mapping(degree + degree_offset, /*use_mapping_q_on_all_cells=*/false)
    // Magnetic vector potential
    , fe_mvp(degree)
    , qf_cell_mvp(degree + 1 + degree_offset)
    , qf_face_mvp(degree + 1 + degree_offset)
    , dof_handler_mvp(triangulation)
    , mvp_group(0)
    , first_mvp_dof(0)
    , mvp_extractor(first_mvp_dof)
    // Electric scalar potential
    , fe_esp(degree + degree_offset)
    , qf_cell_esp(degree + 1 + degree_offset)
    , dof_handler_esp(triangulation)
    , esp_group(0)
    , first_esp_dof(0)
    , esp_extractor(first_esp_dof)
    // Other
    , geometry(parameters.radius_wire * parameters.grid_scale,
               parameters.side_length_surroundings * parameters.grid_scale,
               Point<dim>(0, 0, 1) /*direction*/,
               Point<dim>(0, 0, 0) /*point_on_axis*/)
    , function_material_permeability_coefficients(geometry,
                                                  parameters.mu_r_surroundings,
                                                  parameters.mu_r_wire)
    , function_material_conductivity_coefficients(geometry,
                                                  parameters.sigma_surroundings,
                                                  parameters.sigma_wire)
    , function_free_current_density(geometry, parameters.wire_current)
    , function_analytical_solution_mvp_MI(
        geometry,
        function_material_permeability_coefficients,
        function_free_current_density,
        AnalyticalSolution<dim>::MI)
  {
    if (parameters.discretisation_type != "Cartesian")
      AssertThrow(parameters.radius_wire <
                    (parameters.side_length_surroundings / 2.0),
                  ExcInternalError());
    AssertThrow(parameters.poly_degree_max >= parameters.poly_degree_min,
                ExcInternalError());
  }

  // @sect4{StepTransientCurlCurl_Base::~StepTransientCurlCurl_Base}

  template <int dim>
  StepTransientCurlCurl_Base<dim>::~StepTransientCurlCurl_Base()
  {
    dof_handler_mvp.clear();
    dof_handler_esp.clear();
  }

  // @sect4{StepTransientCurlCurl_Base::setup_system}

  template <int dim>
  void
  StepTransientCurlCurl_Base<dim>::setup_system()
  {
    std::vector<IndexSet> all_locally_owned_dofs_mvp;
    std::vector<IndexSet> all_locally_owned_dofs_esp;

    {
      TimerOutput::Scope timer_scope(computing_timer, "Setup: distribute DoFs");

      // Partition triangulation
      GridTools::partition_triangulation(n_mpi_processes, triangulation);

      // Distribute DoFs
      dof_handler_mvp.distribute_dofs(fe_mvp);
      DoFRenumbering::subdomain_wise(dof_handler_mvp);

      // Construct DoF indices
      locally_owned_dofs_mvp.clear();
      locally_relevant_dofs_mvp.clear();
      all_locally_owned_dofs_mvp =
        DoFTools::locally_owned_dofs_per_subdomain(dof_handler_mvp);
      locally_owned_dofs_mvp    = all_locally_owned_dofs_mvp[this_mpi_process];
      locally_relevant_dofs_mvp = DoFTools::locally_relevant_dofs_per_subdomain(
        dof_handler_mvp)[this_mpi_process];

      if (parameters.use_voltage_excitation())
        {
          dof_handler_esp.distribute_dofs(fe_esp);
          DoFRenumbering::subdomain_wise(dof_handler_esp);

          locally_owned_dofs_esp.clear();
          locally_relevant_dofs_esp.clear();
          all_locally_owned_dofs_esp =
            DoFTools::locally_owned_dofs_per_subdomain(dof_handler_esp);
          locally_owned_dofs_esp = all_locally_owned_dofs_esp[this_mpi_process];
          locally_relevant_dofs_esp =
            DoFTools::locally_relevant_dofs_per_subdomain(
              dof_handler_esp)[this_mpi_process];
        }
    }

    {
      TimerOutput::Scope timer_scope(computing_timer, "Setup: constraints");

      constraints_mvp.clear();
      DoFTools::make_hanging_node_constraints(dof_handler_mvp, constraints_mvp);

      // "Perfect electrical conductor" on top/bottom surfaces
      // See Zaglmayr 2006, p 10, eq. 2.14
      for (const auto b_id : {2, 3})
        {
          VectorTools::project_boundary_values_curl_conforming_l2(
            dof_handler_mvp,
            first_mvp_dof,
            Functions::ZeroFunction<dim>(dim),
            b_id,
            constraints_mvp,
            mapping);
        }

      constraints_mvp.close();

      if (parameters.use_voltage_excitation())
        {
          constraints_esp.clear();
          DoFTools::make_hanging_node_constraints(dof_handler_esp,
                                                  constraints_esp);

          // Top face
          VectorTools::interpolate_boundary_values(
            mapping,
            dof_handler_esp,
            2,
            Functions::ConstantFunction<dim>(+parameters.wire_voltage / 2.0),
            constraints_esp);

          // Bottom face
          VectorTools::interpolate_boundary_values(
            mapping,
            dof_handler_esp,
            3,
            Functions::ConstantFunction<dim>(-parameters.wire_voltage / 2.0),
            constraints_esp);

          // Everywhere else
          VectorTools::interpolate_boundary_values(
            mapping,
            dof_handler_esp,
            1,
            Functions::ZeroFunction<dim>(),
            constraints_esp);

          // Fix all DoF's that are not in the wire
          IndexSet wire_dofs(dof_handler_esp.n_dofs());

          typename DoFHandler<dim>::active_cell_iterator
            cell = this->dof_handler_esp.begin_active(),
            endc = this->dof_handler_esp.end();
          for (; cell != endc; ++cell)
            {
              if (this->geometry.within_wire(cell->center()))
                {
                  const unsigned int &n_dofs_per_cell = fe_esp.dofs_per_cell;
                  std::vector<types::global_dof_index> local_dof_indices(
                    n_dofs_per_cell);
                  cell->get_dof_indices(local_dof_indices);
                  wire_dofs.add_indices(local_dof_indices.begin(),
                                        local_dof_indices.end());
                }
            }

          for (unsigned int dof = 0; dof < dof_handler_esp.n_dofs(); ++dof)
            if (wire_dofs.is_element(dof) == false)
              constraints_esp.add_line(dof);

          constraints_esp.close();
        }
    }

    {
      TimerOutput::Scope timer_scope(computing_timer, "Setup: matrix, vectors");

      std::vector<dealii::types::global_dof_index>
        n_locally_owned_dofs_mvp_per_processor(n_mpi_processes);
      {
        AssertThrow(all_locally_owned_dofs_mvp.size() ==
                      n_locally_owned_dofs_mvp_per_processor.size(),
                    ExcInternalError());
        for (unsigned int i = 0;
             i < n_locally_owned_dofs_mvp_per_processor.size();
             ++i)
          n_locally_owned_dofs_mvp_per_processor[i] =
            all_locally_owned_dofs_mvp[i].n_elements();
      }

      DynamicSparsityPattern dsp(locally_relevant_dofs_mvp);
      DoFTools::make_sparsity_pattern(dof_handler_mvp,
                                      dsp,
                                      constraints_mvp,
                                      /* keep constrained dofs */ false,
                                      Utilities::MPI::this_mpi_process(
                                        mpi_communicator));
      dealii::SparsityTools::distribute_sparsity_pattern(
        dsp,
        n_locally_owned_dofs_mvp_per_processor,
        mpi_communicator,
        locally_relevant_dofs_mvp);

      system_matrix_mvp.reinit(locally_owned_dofs_mvp,
                               locally_owned_dofs_mvp,
                               dsp,
                               mpi_communicator);
      system_rhs_mvp.reinit(locally_owned_dofs_mvp, mpi_communicator);
      solution_mvp.reinit(locally_owned_dofs_mvp,
                          locally_relevant_dofs_mvp,
                          mpi_communicator);
      solution_mvp_t1.reinit(locally_owned_dofs_mvp,
                             locally_relevant_dofs_mvp,
                             mpi_communicator);
      d_solution_mvp_dt.reinit(locally_owned_dofs_mvp,
                               locally_relevant_dofs_mvp,
                               mpi_communicator);



      if (parameters.use_voltage_excitation())
        {
          std::vector<dealii::types::global_dof_index>
            n_locally_owned_dofs_esp_per_processor(n_mpi_processes);
          {
            AssertThrow(all_locally_owned_dofs_esp.size() ==
                          n_locally_owned_dofs_esp_per_processor.size(),
                        ExcInternalError());
            for (unsigned int i = 0;
                 i < n_locally_owned_dofs_esp_per_processor.size();
                 ++i)
              n_locally_owned_dofs_esp_per_processor[i] =
                all_locally_owned_dofs_esp[i].n_elements();
          }

          DynamicSparsityPattern dsp(locally_relevant_dofs_esp);
          DoFTools::make_sparsity_pattern(dof_handler_esp,
                                          dsp,
                                          constraints_esp,
                                          /* keep constrained dofs */ false,
                                          Utilities::MPI::this_mpi_process(
                                            mpi_communicator));
          dealii::SparsityTools::distribute_sparsity_pattern(
            dsp,
            n_locally_owned_dofs_esp_per_processor,
            mpi_communicator,
            locally_relevant_dofs_esp);

          system_matrix_esp.reinit(locally_owned_dofs_esp,
                                   locally_owned_dofs_esp,
                                   dsp,
                                   mpi_communicator);
          system_rhs_esp.reinit(locally_owned_dofs_esp, mpi_communicator);
          solution_esp.reinit(locally_owned_dofs_esp,
                              locally_relevant_dofs_esp,
                              mpi_communicator);
        }
    }
  }

  // @sect4{StepTransientCurlCurl_Base::verify_source_terms}

  template <int dim>
  void
  StepTransientCurlCurl_Base<dim>::verify_source_terms() const
  {
    TimerOutput::Scope timer_scope(computing_timer, "Verify source terms");

    FEFaceValues<dim> fe_face_values(mapping,
                                     fe_mvp,
                                     qf_face_mvp,
                                     update_quadrature_points |
                                       update_normal_vectors |
                                       update_JxW_values);

    // The source term must be divergence free in the strong
    // sense (point-wise), which implies that for the weak
    // formulation of the problem that the flux on an element
    // should be zero.
    // So to verify that the source terms are divergence free,
    // we make use of the divergence theorem:
    // 0 = \int_{\Omega_{e}} div (J_f)
    //   = \int_{\partial\Omega_{e}} J_f . n

    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_mvp
                                                            .begin_active(),
                                                   endc = dof_handler_mvp.end();
    for (; cell != endc; ++cell)
      {
        //    if (cell->is_locally_owned() == false) continue;
        if (cell->subdomain_id() != this_mpi_process)
          continue;

        double div_J_f = 0.0;
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             ++face)
          {
            fe_face_values.reinit(cell, face);
            const unsigned int &n_fq_points =
              fe_face_values.n_quadrature_points;

            std::vector<Tensor<1, dim>> source_values(n_fq_points);
            function_free_current_density.value_list(
              fe_face_values.get_quadrature_points(), source_values);

            for (unsigned int fq_point = 0; fq_point < n_fq_points; ++fq_point)
              {
                // Note: J_f must be divergence free!
                const Tensor<1, dim> &J_f = source_values[fq_point];
                const double          JxW = fe_face_values.JxW(fq_point);
                const Tensor<1, dim> &N =
                  fe_face_values.normal_vector(fq_point);

                div_J_f += (J_f * N) * JxW;
              }
          }

        AssertThrow(std::abs(div_J_f) < 1e-9,
                    ExcMessage("Source term is not divergence free!"));
      }
  }


  // @sect4{StepTransientCurlCurl_Base::verify_solution_mvp}

  template <int dim>
  void
  StepTransientCurlCurl_Base<dim>::verify_solution_mvp() const
  {
    TimerOutput::Scope timer_scope(computing_timer, "Verify solution_mvp");

    FEFaceValues<dim> fe_face_values(mapping,
                                     fe_mvp,
                                     qf_face_mvp,
                                     update_gradients |
                                       update_quadrature_points |
                                       update_normal_vectors |
                                       update_JxW_values);

    // The magnetic induction must be divergence free in the strong
    // sense (point-wise), which implies that for the weak
    // formulation of the problem that its flux on an element
    // should be zero.
    // So to verify that the induction field is divergence free,
    // we make use of the divergence theorem:
    // 0 = \int_{\Omega_{e}} div (b)
    //   = \int_{\Omega_{e}} div (curl(A)) // (**)
    //   = \int_{\partial\Omega_{e}} b . n
    //
    // (**) Should be identically zero, but if we're post-processed the
    //      the gauged degrees-of-freedom incorrectly then we might produce
    //      a magnetic induction field that is not divergence free

    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_mvp
                                                            .begin_active(),
                                                   endc = dof_handler_mvp.end();
    for (; cell != endc; ++cell)
      {
        //    if (cell->is_locally_owned() == false) continue;
        if (cell->subdomain_id() != this_mpi_process)
          continue;

        double div_B = 0.0;
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             ++face)
          {
            fe_face_values.reinit(cell, face);
            const unsigned int &n_fq_points =
              fe_face_values.n_quadrature_points;

            // Pre-compute solution_mvp at QPs
            std::vector<Tensor<1, dim>> qp_curl_A(n_fq_points);
            fe_face_values[mvp_extractor].get_function_curls(solution_mvp,
                                                             qp_curl_A);

            for (unsigned int fq_point = 0; fq_point < n_fq_points; ++fq_point)
              {
                const Tensor<1, dim>  B   = qp_curl_A[fq_point];
                const double          JxW = fe_face_values.JxW(fq_point);
                const Tensor<1, dim> &N =
                  fe_face_values.normal_vector(fq_point);

                div_B += (B * N) * JxW;
              }
          }

        AssertThrow(std::abs(div_B) < 1e-9,
                    ExcMessage(
                      "Magnetic induction field is not divergence free!"));
      }
  }


  // @sect4{StepTransientCurlCurl_Base::solve}

  template <int dim>
  void
  StepTransientCurlCurl_Base<dim>::solve_esp(
    const TrilinosWrappers::SparseMatrix &system_matrix_esp,
    TrilinosWrappers::MPI::Vector &       solution_esp,
    const TrilinosWrappers::MPI::Vector & system_rhs_esp,
    const AffineConstraints<double> &     constraints_esp,
    const IndexSet &                      locally_owned_dofs_esp,
    const DoFHandler<dim> &               dof_handler_esp)
  {
    TimerOutput::Scope timer_scope(computing_timer, "Solve linear system: ESP");
    solve_linear_system(system_matrix_esp,
                        solution_esp,
                        system_rhs_esp,
                        constraints_esp,
                        locally_owned_dofs_esp,
                        dof_handler_esp);
  }

  template <int dim>
  void
  StepTransientCurlCurl_Base<dim>::solve_mvp(
    const TrilinosWrappers::SparseMatrix &system_matrix_mvp,
    TrilinosWrappers::MPI::Vector &       solution_mvp,
    const TrilinosWrappers::MPI::Vector & system_rhs_mvp,
    const AffineConstraints<double> &     constraints_mvp,
    const IndexSet &                      locally_owned_dofs_mvp,
    const DoFHandler<dim> &               dof_handler_mvp)
  {
    TimerOutput::Scope timer_scope(computing_timer, "Solve linear system: MVP");
    solve_linear_system(system_matrix_mvp,
                        solution_mvp,
                        system_rhs_mvp,
                        constraints_mvp,
                        locally_owned_dofs_mvp,
                        dof_handler_mvp);
  }

  template <int dim>
  void
  StepTransientCurlCurl_Base<dim>::solve_linear_system(
    const TrilinosWrappers::SparseMatrix &system_matrix,
    TrilinosWrappers::MPI::Vector &       solution,
    const TrilinosWrappers::MPI::Vector & system_rhs,
    const AffineConstraints<double> &     constraints,
    const IndexSet &                      locally_owned_dofs,
    const DoFHandler<dim> &               dof_handler)
  {
    TrilinosWrappers::MPI::Vector distributed_solution_increment(
      locally_owned_dofs, mpi_communicator);

    SolverControl solver_control(
      static_cast<unsigned int>(parameters.lin_slvr_max_it * system_matrix.m()),
      std::min(parameters.lin_slvr_tol,
               parameters.lin_slvr_tol * system_rhs.l2_norm()),
      false,
      false);

    if (parameters.lin_slvr_type == "Iterative")
      {
        std::unique_ptr<TrilinosWrappers::PreconditionBase> preconditioner;
        if (parameters.preconditioner_type == "jacobi")
          {
            TrilinosWrappers::PreconditionJacobi *ptr_prec =
              new TrilinosWrappers::PreconditionJacobi();

            TrilinosWrappers::PreconditionJacobi::AdditionalData
              additional_data(parameters.preconditioner_relaxation);

            ptr_prec->initialize(system_matrix, additional_data);
            preconditioner.reset(ptr_prec);
          }
        else if (parameters.preconditioner_type == "ssor")
          {
            TrilinosWrappers::PreconditionSSOR *ptr_prec =
              new TrilinosWrappers::PreconditionSSOR();

            TrilinosWrappers::PreconditionSSOR::AdditionalData additional_data(
              parameters.preconditioner_relaxation);

            ptr_prec->initialize(system_matrix, additional_data);
            preconditioner.reset(ptr_prec);
          }
        else // AMG
          {
            // Default settings for AMG preconditioner are
            // good for a Laplace problem
            TrilinosWrappers::PreconditionAMG *ptr_prec =
              new TrilinosWrappers::PreconditionAMG();

            TrilinosWrappers::PreconditionAMG::AdditionalData additional_data;

            typename DoFHandler<dim>::active_cell_iterator
              cell = dof_handler.begin_active(),
              endc = dof_handler.end();
            for (; cell != endc; ++cell)
              {
                //    if (cell->is_locally_owned() == false) continue;
                if (cell->subdomain_id() != this_mpi_process)
                  continue;

                const unsigned int cell_fe_idx = cell->active_fe_index();
                const unsigned int cell_poly   = cell_fe_idx + 1;
                if (cell_poly > 1)
                  {
                    additional_data.higher_order_elements = true;
                    break;
                  }
              }
            {
              const int hoe = additional_data.higher_order_elements;
              additional_data.higher_order_elements =
                Utilities::MPI::max(hoe, mpi_communicator);
            }
            ptr_prec->initialize(system_matrix, additional_data);
            preconditioner.reset(ptr_prec);
          }

        const bool use_deal_II_solver = false;
        if (use_deal_II_solver == true)
          {
            SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
            solver.solve(system_matrix,
                         distributed_solution_increment,
                         system_rhs,
                         *preconditioner);
          }
        else
          {
            TrilinosWrappers::SolverCG solver(solver_control);

            solver.solve(system_matrix,
                         distributed_solution_increment,
                         system_rhs,
                         *preconditioner);
          }
      }
    else // Direct
      {
        TrilinosWrappers::SolverDirect solver(solver_control);
        solver.solve(system_matrix, distributed_solution_increment, system_rhs);
      }

    constraints.distribute(distributed_solution_increment);

    // The linear problem is still not incremental, due to the linear RHS
    // contribution.
    solution = distributed_solution_increment;

    // The nonlinear problem would be an incremental problem in time.
    // solution += distributed_solution_increment;

    pcout << "   Solver: " << parameters.lin_slvr_type
          << "  Iterations: " << solver_control.last_step()
          << "  Residual: " << solver_control.last_value() << std::endl;
  }


  // @sect4{StepTransientCurlCurl_Base::refine_grid}

  template <int dim>
  void
  StepTransientCurlCurl_Base<dim>::refine_grid()
  {
    TimerOutput::Scope timer_scope(computing_timer, "Grid refinement");

    // Global refinement
    if (refinement_strategy.use_GR() == true)
      {
        if (refinement_strategy.use_h_refinement() == true)
          {
            AssertThrow(triangulation.n_global_levels() <
                          parameters.n_levels_max,
                        ExcInternalError());
            triangulation.refine_global(1);
          }
        else // p-refinement
          {
            AssertThrow(false, ExcNotImplemented());
          }
      }
    else // Adaptive mesh refinement
      {
        const QGauss<dim - 1> EE_qf_face_mvp_QGauss(degree + 2);

        // Refine based on eddy current solution_mvp
        Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
        using FunctionMap_type =
          std::map<types::boundary_id,
                   const Function<dim, typename Vector<double>::value_type> *>;
        estimated_error_per_cell = 0.0;
        KellyErrorEstimator<dim>::estimate(
          dof_handler_mvp,
          EE_qf_face_mvp_QGauss,
          FunctionMap_type{},
          d_solution_mvp_dt,
          estimated_error_per_cell,
          ComponentMask(),
          /*coefficients = */ 0,
          /*n_threads = */ numbers::invalid_unsigned_int,
          /*subdomain_id = */ this_mpi_process);

        // Mark cells for adaptive mesh refinement...
        GridRefinement::refine_and_coarsen_fixed_number(
          triangulation,
          estimated_error_per_cell,
          parameters.frac_refine,
          parameters.frac_coarsen);

        // Check that there are no violations on maximum cell level
        // If so, then remove the marking
        //      if (triangulation.n_global_levels() >
        //      static_cast<int>(parameters.n_levels_max))
        if (triangulation.n_levels() >= parameters.n_levels_max)
          for (typename Triangulation<dim>::active_cell_iterator cell =
                 triangulation.begin_active(parameters.n_levels_max - 1);
               cell != triangulation.end();
               ++cell)
            {
              cell->clear_refine_flag();
            }

        if (refinement_strategy.use_h_refinement() == true &&
            refinement_strategy.use_p_refinement() == false) // h-refinement
          {
            triangulation.execute_coarsening_and_refinement();
          }
        else if (refinement_strategy.use_h_refinement() == false &&
                 refinement_strategy.use_p_refinement() == true) // p-refinement
          {
            typename DoFHandler<dim>::active_cell_iterator
              cell = dof_handler_mvp.begin_active(),
              endc = dof_handler_mvp.end();
            for (; cell != endc; ++cell)
              {
                //    if (cell->is_locally_owned() == false) continue;
                if (cell->subdomain_id() != this_mpi_process)
                  {
                    // Clear flags on non-owned cell that would
                    // be cleared on the owner processor anyway...
                    cell->clear_refine_flag();
                    cell->clear_coarsen_flag();
                    continue;
                  }

                const unsigned int cell_fe_idx = cell->active_fe_index();
                const unsigned int cell_poly   = cell_fe_idx + 1;

                if (cell->refine_flag_set())
                  {
                    if (cell_poly < parameters.poly_degree_max)
                      cell->set_active_fe_index(cell_fe_idx + 1);
                    cell->clear_refine_flag();
                  }

                if (cell->coarsen_flag_set())
                  {
                    if (cell_poly > parameters.poly_degree_min)
                      cell->set_active_fe_index(cell_fe_idx - 1);
                    cell->clear_coarsen_flag();
                  }

                AssertThrow(!(cell->refine_flag_set()), ExcInternalError());
                AssertThrow(!(cell->coarsen_flag_set()), ExcInternalError());
              }
          }
        else // hp-refinement
          {
            AssertThrow(refinement_strategy.use_hp_refinement() == true,
                        ExcInternalError());
            AssertThrow(false, ExcNotImplemented());
          }
      }
  }


  // @sect4{StepTransientCurlCurl_Base::output_results}

  template <int dim>
  struct Filename
  {
    static std::string
    get_filename_vtu(const unsigned int process,
                     const unsigned int timestep,
                     const unsigned int cycle,
                     const std::string  name     = "solution_mvp",
                     const unsigned int n_digits = 4)
    {
      std::ostringstream filename_vtu;
      filename_vtu << name << "-" << (std::to_string(dim) + "d") << "."
                   << Utilities::int_to_string(process, n_digits) << "."
                   << Utilities::int_to_string(timestep, n_digits) << "."
                   << Utilities::int_to_string(cycle, n_digits) << ".vtu";
      return filename_vtu.str();
    }

    static std::string
    get_filename_pvtu(const unsigned int timestep,
                      const unsigned int cycle,
                      const std::string  name     = "solution_mvp",
                      const unsigned int n_digits = 4)
    {
      std::ostringstream filename_vtu;
      filename_vtu << name << "-" << (std::to_string(dim) + "d") << "."
                   << Utilities::int_to_string(timestep, n_digits) << "."
                   << Utilities::int_to_string(cycle, n_digits) << ".pvtu";
      return filename_vtu.str();
    }

    static std::string
    get_filename_pvd(const std::string name = "solution_mvp")
    {
      std::ostringstream filename_vtu;
      filename_vtu << name << "-" << (std::to_string(dim) + "d") << ".pvd";
      return filename_vtu.str();
    }
  };


  template <int dim>
  void
  StepTransientCurlCurl_Base<dim>::output_grid(const unsigned int timestep,
                                               const unsigned int cycle) const
  {
    const bool output_vtk = false;
    if (output_vtk == false)
      return;

    TimerOutput::Scope timer_scope(computing_timer, "Grid output");

    // Write out main data file
    const std::string filename_vtu = Filename<dim>::get_filename_vtu(
      this_mpi_process, timestep, cycle, "grid");
    std::ofstream output(filename_vtu.c_str());
    if (this_mpi_process == 0)
      {
        GridOut().write_vtu(triangulation, output);
      }
  }


  template <int dim>
  class MagneticInductionPostprocessor : public DataPostprocessorVector<dim>
  {
  public:
    MagneticInductionPostprocessor()
      : DataPostprocessorVector<dim>("B", update_gradients)
    {}

    virtual void
    evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const override
    {
      Assert(dim == 3, ExcNotImplemented());
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());

      const std::vector<std::vector<Tensor<1, dim>>> &grad_A =
        input_data.solution_gradients;

      const unsigned int n_evaluation_points =
        input_data.solution_gradients.size();

      for (unsigned int p = 0; p < n_evaluation_points; ++p)
        {
          AssertDimension(computed_quantities[p].size(), dim);

          // B = curl(A)
          computed_quantities[p][0] = grad_A[p][2][1] - grad_A[p][1][2];
          computed_quantities[p][1] = grad_A[p][0][2] - grad_A[p][2][0];
          computed_quantities[p][2] = grad_A[p][1][0] - grad_A[p][0][1];
        }
    }
  };


  template <int dim>
  class EddyCurrentPostprocessor : public DataPostprocessorVector<dim>
  {
  public:
    EddyCurrentPostprocessor(const Geometry<dim> &geometry,
                             const double &       sigma_surroundings,
                             const double &       sigma_wire)
      : DataPostprocessorVector<dim>("J_eddy",
                                     update_values | update_quadrature_points)
      , function_material_conductivity_coefficients(geometry,
                                                    sigma_surroundings,
                                                    sigma_wire)
    {}

    virtual void
    evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.evaluation_points.size(),
                      computed_quantities.size());
      AssertDimension(input_data.solution_values.size(),
                      computed_quantities.size());

      const std::vector<Point<dim>> &evaluation_points =
        input_data.evaluation_points;
      const std::vector<Vector<double>> &dA_dt = input_data.solution_values;

      const unsigned int n_evaluation_points = evaluation_points.size();

      std::vector<double> conductivity_coefficient_values(n_evaluation_points);
      function_material_conductivity_coefficients.value_list(
        evaluation_points, conductivity_coefficient_values);

      for (unsigned int p = 0; p < n_evaluation_points; ++p)
        {
          AssertDimension(computed_quantities[p].size(), dim);
          const double &sigma = conductivity_coefficient_values[p];

          for (unsigned int d = 0; d < dim; ++d)
            computed_quantities[p][d] = -sigma * dA_dt[p][d];
        }
    }

  private:
    ConductivityCoefficient<dim> function_material_conductivity_coefficients;
  };


  template <int dim>
  class SourceFreeCurrentPostprocessor : public DataPostprocessorVector<dim>
  {
  public:
    SourceFreeCurrentPostprocessor(const Geometry<dim> &geometry,
                                   const double &       wire_current)
      : DataPostprocessorVector<dim>("J_free", update_quadrature_points)
      , source_free_current(geometry, wire_current)
    {}

    virtual void
    evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.evaluation_points.size(),
                      computed_quantities.size());

      const std::vector<Point<dim>> &evaluation_points =
        input_data.evaluation_points;

      const unsigned int n_evaluation_points = evaluation_points.size();

      for (unsigned int p = 0; p < n_evaluation_points; ++p)
        {
          AssertDimension(computed_quantities[p].size(), dim);
          const Tensor<1, dim> J_f =
            source_free_current.value(evaluation_points[p]);
          for (unsigned int d = 0; d < dim; ++d)
            computed_quantities[p][d] = J_f[d];
        }
    }

  private:
    const SourceFreeCurrentDensity<dim> source_free_current;
  };


  template <int dim>
  class SourceFEQFreeCurrentDensity : public DataPostprocessorVector<dim>
  {
  public:
    SourceFEQFreeCurrentDensity(const Geometry<dim> &geometry,
                                const double &       sigma_surroundings,
                                const double &       sigma_wire)
      : DataPostprocessorVector<dim>("J_free_grad_field",
                                     update_gradients |
                                       update_quadrature_points)
      , function_material_conductivity_coefficients(geometry,
                                                    sigma_surroundings,
                                                    sigma_wire)
    {}

    virtual void
    evaluate_scalar_field(
      const DataPostprocessorInputs::Scalar<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.evaluation_points.size(),
                      computed_quantities.size());
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());

      const std::vector<Point<dim>> &evaluation_points =
        input_data.evaluation_points;
      const std::vector<Tensor<1, dim>> &grad_esp =
        input_data.solution_gradients;

      const unsigned int n_evaluation_points = evaluation_points.size();

      std::vector<double> conductivity_coefficient_values(n_evaluation_points);
      function_material_conductivity_coefficients.value_list(
        evaluation_points, conductivity_coefficient_values);

      for (unsigned int p = 0; p < n_evaluation_points; ++p)
        {
          AssertDimension(computed_quantities[p].size(), dim);
          const double &sigma = conductivity_coefficient_values[p];

          for (unsigned int d = 0; d < dim; ++d)
            computed_quantities[p][d] = -sigma * grad_esp[p][d];
        }
    }

  private:
    ConductivityCoefficient<dim> function_material_conductivity_coefficients;
  };


  template <int dim>
  void
  StepTransientCurlCurl_Base<dim>::output_results(
    const unsigned int timestep,
    const unsigned int cycle) const
  {
    const bool output_vtk = false;
    if (output_vtk == false)
      return;

    TimerOutput::Scope timer_scope(computing_timer, "Output results");

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_mvp);

    const std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation_sclr(
        1, DataComponentInterpretation::component_is_scalar);
    const std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation_vec(
        dim, DataComponentInterpretation::component_is_part_of_vector);

    // solution_mvp
    const std::vector<std::string> name_MVP_solution_mvp(dim, "A");
    data_out.add_data_vector(solution_mvp,
                             name_MVP_solution_mvp,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation_vec);
    // Rate of change of solution_mvp
    const std::vector<std::string> name_d_MVP_dt_solution_mvp(dim, "dA_dt");
    data_out.add_data_vector(d_solution_mvp_dt,
                             name_d_MVP_dt_solution_mvp,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation_vec);

    // Magnetic induction
    const MagneticInductionPostprocessor<dim> magnetic_induction_postprocessor;
    data_out.add_data_vector(solution_mvp, magnetic_induction_postprocessor);

    // Eddy currents
    const EddyCurrentPostprocessor<dim> eddy_current_postprocessor(
      geometry, parameters.sigma_surroundings, parameters.sigma_wire);
    data_out.add_data_vector(d_solution_mvp_dt, eddy_current_postprocessor);

    // Current source
    const SourceFreeCurrentPostprocessor<dim> source_free_current_postprocessor(
      geometry, parameters.wire_current);
    data_out.add_data_vector(solution_mvp, source_free_current_postprocessor);

    const SourceFEQFreeCurrentDensity<dim>
      source_feq_free_current_postprocessor(geometry,
                                            parameters.sigma_surroundings,
                                            parameters.sigma_wire);
    if (parameters.use_voltage_excitation())
      {
        const std::vector<std::string> name_ESP_solution_esp(1, "phi");
        data_out.add_data_vector(dof_handler_esp,
                                 solution_esp,
                                 name_ESP_solution_esp,
                                 data_component_interpretation_sclr);

        data_out.add_data_vector(dof_handler_esp,
                                 solution_esp,
                                 source_feq_free_current_postprocessor);
      }

    // const std::vector<std::string> name_J_solution_mvp(dim, "J_free");
    // data_out.add_data_vector(postprocessing.J.dof_handler_mvp,
    //                          postprocessing.J.solution_mvp,
    //                          name_J_solution_mvp,
    //                          data_component_interpretation_vec);

    // // Current source: Verification of interpolation
    // const bool interpolate_source = false;
    // TrilinosWrappers::MPI::Vector source_terms_int_proj;
    // if (interpolate_source == true)
    // {
    //   // Compute source current by interpolation
    //   source_terms_int_proj.reinit(postprocessing.J.locally_owned_dofs_mvp,
    //                                postprocessing.J.locally_relevant_dofs_mvp,
    //                                mpi_communicator);
    //   VectorTools::interpolate(/*mapping_collection,*/
    //                            postprocessing.J.dof_handler_mvp,
    //                            function_free_current_density,
    //                            source_terms_int_proj);
    //   const std::vector<std::string> name_source_free_current (dim,
    //   "source_free_current_int"); data_out.add_data_vector
    //   (postprocessing.J.dof_handler_mvp,
    //                             source_terms_int_proj,
    //                             name_source_free_current,
    //                             data_component_interpretation_vec);
    // }
    // else
    // {
    //   // Compute source current by projection
    //   source_terms_int_proj.reinit(postprocessing.J.locally_owned_dofs_mvp,
    //                                postprocessing.J.locally_relevant_dofs_mvp,
    //                                mpi_communicator);
    //   TrilinosWrappers::MPI::Vector distributed_source_terms;
    //   distributed_source_terms.reinit(postprocessing.J.locally_owned_dofs_mvp,
    //                                   mpi_communicator);
    //   AffineConstraints<double> constraints_mvp_source;
    //   constraints_mvp_source.close();
    //   hp::QCollection<dim> qf_collection_cell_QGauss;
    //   for (unsigned int degree = parameters.poly_degree_min;
    //       degree <= parameters.poly_degree_max; ++degree)
    //   {
    //     qf_collection_cell_QGauss.push_back(QGauss<dim> (degree + 2));
    //   }
    //   VectorTools::project (postprocessing.J.dof_handler_mvp,
    //                         constraints_mvp_source,
    //                         qf_collection_cell_QGauss,
    //                         function_free_current_density,
    //                         distributed_source_terms);
    //   source_terms_int_proj = distributed_source_terms;
    //   const std::vector<std::string> name_source_free_current (dim,
    //   "source_free_current_proj"); data_out.add_data_vector
    //   (postprocessing.J.dof_handler_mvp,
    //                             source_terms_int_proj,
    //                             name_source_free_current,
    //                             data_component_interpretation_vec);
    // }

    // --- Additional data ---
    // Material coefficients
    Vector<double> material_id;
    Vector<double> material_permeability_coefficients;
    Vector<double> material_conductivity_coefficients;
    material_id.reinit(triangulation.n_active_cells());
    material_permeability_coefficients.reinit(triangulation.n_active_cells());
    material_conductivity_coefficients.reinit(triangulation.n_active_cells());
    {
      unsigned int c = 0;
      typename Triangulation<dim>::active_cell_iterator
        cell = triangulation.begin_active(),
        endc = triangulation.end();
      for (; cell != endc; ++cell, ++c)
        {
          //      if (cell->is_locally_owned() == false) continue;
          if (cell->subdomain_id() != this_mpi_process)
            continue;

          material_id(c) = cell->material_id();
          material_permeability_coefficients(c) =
            function_material_permeability_coefficients.value(cell->center());
          material_conductivity_coefficients(c) =
            function_material_conductivity_coefficients.value(cell->center());
        }
    }

    data_out.add_data_vector(material_id, "material_id");
    data_out.add_data_vector(material_permeability_coefficients, "mu");
    data_out.add_data_vector(material_conductivity_coefficients, "sigma");

    std::vector<types::subdomain_id> partition_int(
      triangulation.n_active_cells());
    GridTools::get_subdomain_association(triangulation, partition_int);
    const Vector<double> partitioning(partition_int.begin(),
                                      partition_int.end());
    data_out.add_data_vector(partitioning, "partitioning");

    data_out.build_patches(mapping, degree + degree_offset);
    //  data_out.build_patches(max_used_poly_degree);

    // Write out main data file
    const std::string filename_vtu =
      Filename<dim>::get_filename_vtu(this_mpi_process, timestep, cycle);
    std::ofstream output(filename_vtu.c_str());
    data_out.write_vtu(output);

    // Collection of files written in parallel
    // This next set of steps should only be performed
    // by master process
    if (this_mpi_process == 0)
      {
        // List of all files written out at this timestep by all processors
        std::vector<std::string> parallel_filenames_vtu;
        for (unsigned int p = 0; p < n_mpi_processes; ++p)
          {
            parallel_filenames_vtu.push_back(
              Filename<dim>::get_filename_vtu(p, timestep, cycle));
          }

        const std::string filename_pvtu(
          Filename<dim>::get_filename_pvtu(timestep, cycle));
        std::ofstream pvtu_master(filename_pvtu.c_str());
        data_out.write_pvtu_record(pvtu_master, parallel_filenames_vtu);

        // Time dependent data master file
        static std::vector<std::pair<double, std::string>>
          time_and_name_history;
        time_and_name_history.push_back(
          std::make_pair(timestep, filename_pvtu));
        const std::string filename_pvd(Filename<dim>::get_filename_pvd());
        std::ofstream     pvd_output(filename_pvd.c_str());
        DataOutBase::write_pvd_record(pvd_output, time_and_name_history);
      }
  }


  template <int dim>
  void
  StepTransientCurlCurl_Base<dim>::output_point_solution_mvp(
    const unsigned int timestep,
    const unsigned int cycle) const
  {
    const QMidpoint<dim> soln_qrule;

    FEValues<dim> fe_values(mapping,
                            fe_mvp,
                            soln_qrule,
                            update_values | update_gradients |
                              update_quadrature_points);

    std::vector<Point<dim>> points_of_interest({
      Point<dim>(0.9 * parameters.radius_wire * parameters.grid_scale, 0, 0),
      Point<dim>(1.1 * parameters.radius_wire * parameters.grid_scale, 0, 0),
    });

    deallog << "Timestep: " << timestep << " ; Cycle: " << cycle << std::endl;
    for (const auto &p : points_of_interest)
      {
        const auto cell_iterator_and_point =
          GridTools::find_active_cell_around_point(mapping, dof_handler_mvp, p);
        const auto &cell = cell_iterator_and_point.first;

        const auto output_result = [this](const Point<dim> &    point,
                                          const Tensor<1, dim> &B,
                                          const Tensor<1, dim> &J_eddy,
                                          const Tensor<1, dim> &J_free)
        {
          deallog << " ; Point: " << point
                  << " ; Within wire: " << geometry.within_wire(point)
                  << " ; B: " << B << " ; J_eddy (axial): " << J_eddy[dim - 1]
                  << " ; J_free (axial): " << J_free[dim - 1] << std::endl;
        };

        if (cell != dof_handler_mvp.end() &&
            cell->subdomain_id() == this_mpi_process)
          {
            std::cout << "FOUND IN SUBDOMAIN " << cell->subdomain_id()
                      << std::endl;
            fe_values.reinit(cell);

            const unsigned int &n_q_points = fe_values.n_quadrature_points;
            const unsigned int  q_point    = 0;
            const Point<dim> point = fe_values.get_quadrature_points()[q_point];

            std::vector<double> conductivity_coefficient_values(n_q_points);
            std::vector<Tensor<1, dim>> source_values(n_q_points);

            this->function_material_conductivity_coefficients.value_list(
              fe_values.get_quadrature_points(),
              conductivity_coefficient_values);
            this->function_free_current_density.value_list(
              fe_values.get_quadrature_points(), source_values);

            std::vector<Tensor<1, dim>> solution_mvp_curls(n_q_points);
            std::vector<Tensor<1, dim>> d_solution_mvp_dt_values(n_q_points);
            fe_values[mvp_extractor].get_function_curls(solution_mvp,
                                                        solution_mvp_curls);
            fe_values[mvp_extractor].get_function_values(
              d_solution_mvp_dt, d_solution_mvp_dt_values);

            const double sigma      = conductivity_coefficient_values[q_point];
            const Tensor<1, dim>  B = solution_mvp_curls[q_point];
            const Tensor<1, dim>  dA_dt  = d_solution_mvp_dt_values[q_point];
            const Tensor<1, dim>  J_eddy = -sigma * dA_dt;
            const Tensor<1, dim> &J_free = source_values[q_point];

            if (n_mpi_processes > 1)
              {
                Utilities::MPI::broadcast(mpi_communicator,
                                          point,
                                          cell->subdomain_id());
                Utilities::MPI::broadcast(mpi_communicator,
                                          B,
                                          cell->subdomain_id());
                Utilities::MPI::broadcast(mpi_communicator,
                                          J_eddy,
                                          cell->subdomain_id());
                Utilities::MPI::broadcast(mpi_communicator,
                                          J_free,
                                          cell->subdomain_id());
              }

            output_result(point, B, J_eddy, J_free);
          }
        else
          {
            Point<dim>     point;
            Tensor<1, dim> B;
            Tensor<1, dim> J_eddy;
            Tensor<1, dim> J_free;

            if (n_mpi_processes > 1)
              {
                point  = Utilities::MPI::broadcast(mpi_communicator,
                                                  point,
                                                  cell->subdomain_id());
                B      = Utilities::MPI::broadcast(mpi_communicator,
                                              B,
                                              cell->subdomain_id());
                J_eddy = Utilities::MPI::broadcast(mpi_communicator,
                                                   J_eddy,
                                                   cell->subdomain_id());
                J_free = Utilities::MPI::broadcast(mpi_communicator,
                                                   J_free,
                                                   cell->subdomain_id());
              }

            output_result(point, B, J_eddy, J_free);
          }
      }
  }

  // @sect4{StepTransientCurlCurl_Base::make_grid}
  template <int dim>
  void
  StepTransientCurlCurl_Base<dim>::make_grid()
  {
    TimerOutput::Scope timer_scope(computing_timer, "Make grid");

    const double half_length =
      parameters.side_length_surroundings * parameters.grid_scale / 2.0;
    Point<dim> corner;
    for (unsigned int i = 0; i < dim; ++i)
      corner(i) = half_length;

    if (parameters.discretisation_type == "Cartesian")
      {
        const unsigned int n_el_per_edge_tan = static_cast<unsigned int>(
          std::ceil(parameters.side_length_surroundings /
                    (2.0 * parameters.radius_wire /
                     double(parameters.n_divisions_radial))));
        const std::vector<unsigned int> divisions(
          {n_el_per_edge_tan,
           n_el_per_edge_tan,
           parameters.n_divisions_longitudinal});
        GridGenerator::subdivided_hyper_rectangle(
          triangulation, divisions, corner, -corner, true);

        triangulation.refine_global(parameters.n_global_refinements);
      }
    else
      {
        Triangulation<dim> tria_surroundings;
        {
          const double inner_radius =
            parameters.radius_wire * parameters.grid_scale;
          const double length =
            parameters.side_length_surroundings * parameters.grid_scale;
          const double outer_radius =
            std::min(2 * inner_radius, 0.5 * (length + inner_radius));
          const double             pad               = length - outer_radius;
          const double             pad_bottom        = pad;
          const double             pad_top           = pad;
          const double             pad_left          = pad;
          const double             pad_right         = pad;
          const Point<dim> &       center            = Point<dim>();
          const types::manifold_id polar_manifold_id = 0;
          const types::manifold_id tfi_manifold_id   = 1;
          // The wire gets refined once, so account for that here.
          Assert(
            parameters.n_divisions_longitudinal % 2 == 0,
            ExcMessage(
              "The number of initial divisions must be a multiple of two."));
          const unsigned int n_slices = parameters.n_divisions_longitudinal + 1;
          GridGenerator::plate_with_a_hole(tria_surroundings,
                                           inner_radius,
                                           outer_radius,
                                           pad_bottom,
                                           pad_top,
                                           pad_left,
                                           pad_right,
                                           center,
                                           polar_manifold_id,
                                           tfi_manifold_id,
                                           length,
                                           n_slices);
        }

        Triangulation<dim> tria_wire;
        {
          const double radius = parameters.radius_wire * parameters.grid_scale;
          const double length =
            parameters.side_length_surroundings * parameters.grid_scale;
          const unsigned int n_divisions = parameters.n_divisions_longitudinal;

          Triangulation<dim> tria_wire_tmp;
          GridGenerator::subdivided_cylinder(tria_wire_tmp,
                                             n_divisions / 2,
                                             radius,
                                             length / 2.0);

          // Refine before rotation, due to attached manifolds
          tria_wire_tmp.refine_global(1);

          // Align with axis of hole
          GridTools::rotate(Tensor<1, dim>({0, 1, 0}),
                            90.0 * (numbers::PI / 180.0),
                            tria_wire_tmp);

          GridGenerator::flatten_triangulation(tria_wire_tmp, tria_wire);
        }

        // Merge triangulations
        GridGenerator::merge_triangulations(tria_surroundings,
                                            tria_wire,
                                            triangulation);

        // Remove all boundary and manifold IDs
        typename Triangulation<dim>::active_cell_iterator
          cell = triangulation.begin_active(),
          endc = triangulation.end();
        for (; cell != endc; ++cell)
          {
            for (unsigned int face = 0;
                 face < GeometryInfo<dim>::faces_per_cell;
                 ++face)
              {
                if (cell->at_boundary(face) == true)
                  {
                    cell->face(face)->set_all_manifold_ids(
                      numbers::flat_manifold_id - 1);
                    cell->face(face)->set_all_boundary_ids(
                      numbers::flat_manifold_id - 1);
                  }
                else
                  {
                    cell->face(face)->set_all_manifold_ids(
                      numbers::flat_manifold_id - 2);
                  }
              }
          }

        // Set refinement manifold to keep geometry of wire
        // as exact as possible
        {
          const double bdr = 1.0 / std::sqrt(2.0) * parameters.radius_wire *
                             parameters.grid_scale;

          typename Triangulation<dim>::active_cell_iterator
            cell = triangulation.begin_active(),
            endc = triangulation.end();
          for (; cell != endc; ++cell)
            {
              for (unsigned int face = 0;
                   face < GeometryInfo<dim>::faces_per_cell;
                   ++face)
                {
                  if (cell->at_boundary(face) == false)
                    {
                      const Point<dim> fctr = cell->face(face)->center();

                      if ((std::abs(std::abs(fctr[0]) - bdr) < 1e-6) ||
                          (std::abs(std::abs(fctr[1]) - bdr) < 1e-6))
                        {
                          cell->face(face)->set_all_manifold_ids(
                            refinement_manifold_id);
                        }
                    }
                }
            }
        }
        triangulation.set_manifold(refinement_manifold_id, surface_description);

        // // Force one refinement to get material interface to match up
        // triangulation.refine_global(1);

        // // Refine, keeping in mind the extra subdivisions
        // // already in place
        // if (parameters.n_global_refinements > 0)
        //   parameters.n_global_refinements -= 1;
        // triangulation.refine_global(parameters.n_global_refinements);

        // Reset the maximum number of levels if our original refinement
        // scheme violates it
        parameters.n_levels_max =
          std::max(parameters.n_levels_max, triangulation.n_global_levels());
      }
  }


  template <int dim>
  void
  StepTransientCurlCurl_Base<dim>::set_material_and_boundary_ids()
  {
    TimerOutput::Scope timer_scope(computing_timer, "Set cell data id");

    const double half_length =
      parameters.side_length_surroundings * parameters.grid_scale / 2.0;
    const double tol_b_id = 1e-6;

    // Set material and boundary IDs
    typename Triangulation<dim>::active_cell_iterator cell = triangulation
                                                               .begin_active(),
                                                      endc =
                                                        triangulation.end();
    for (; cell != endc; ++cell)
      {
        // Material ID
        if (parameters.discretisation_type == "Cartesian")
          {
            int n_pts_in_wire = 0;
            if (geometry.within_wire(cell->center()) == true)
              ++n_pts_in_wire;
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
                 ++v)
              if (geometry.within_wire(cell->vertex(v)) == true)
                ++n_pts_in_wire;

            if (n_pts_in_wire == 0)
              cell->set_material_id(1);
            else if (n_pts_in_wire == GeometryInfo<dim>::vertices_per_cell + 1)
              cell->set_material_id(2);
            else
              cell->set_material_id(3);
          }
        else
          {
            if (geometry.within_wire(cell->center()) == true)
              cell->set_material_id(2);
            else
              cell->set_material_id(1);
          }

        // Boundary ID
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             ++face)
          {
            if (cell->at_boundary(face) == true)
              {
                // Here it is required that the upper/lower boundaries of the
                // wire (in/out of which the current flows) have the magnetic
                // induction tangential to them
                // --> Set b.n = n_x_a = 0
                // This is the "perfect electric conductor" condition.
                // On the lateral surfaces , the boundary is traction free.
                // This is the "perfect magnetic conductor" condition.
                // See Zaglmayr 2006, p 10.
                // ID = 1:   Lateral faces (perpendicular to wire)
                // ID = 2/3: Top/bottom faces (parallel to wire)
                const Point<dim> pt = cell->face(face)->center();
                if (geometry.within_wire(pt) == true)
                  {
                    if (std::abs(pt[2] - half_length) < tol_b_id)
                      cell->face(face)->set_boundary_id(2);
                    else if (std::abs(pt[2] + half_length) < tol_b_id)
                      cell->face(face)->set_boundary_id(3);
                  }
                else
                  cell->face(face)->set_boundary_id(1);
              }
          }
      }

    // Print material and boundary data
    std::map<types::material_id, unsigned int> n_cells_with_material_id;
    std::map<types::boundary_id, unsigned int> n_faces_with_boundary_id;
    cell = triangulation.begin_active();
    for (; cell != endc; ++cell)
      {
        n_cells_with_material_id[cell->material_id()] += 1;
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             ++face)
          {
            if (cell->at_boundary(face) == true)
              {
                n_faces_with_boundary_id[cell->face(face)->boundary_id()] += 1;
              }
          }
      }

    pcout << "Material information:" << std::endl;
    for (const auto &m : n_cells_with_material_id)
      pcout << "ID: " << int(m.first) << "  Count: " << m.second << std::endl;

    pcout << "Boundary information:" << std::endl;
    for (const auto &b : n_faces_with_boundary_id)
      pcout << "ID: " << int(b.first) << "  Count: " << b.second << std::endl;
  }

  // @sect4{StepTransientCurlCurl_Base::run}

  template <int dim>
  void
  StepTransientCurlCurl_Base<dim>::run()
  {
    computing_timer.reset();
    const unsigned int n_cycles =
      std::min(refinement_strategy.n_cycles_max(), parameters.n_cycles_max);

    unsigned int cycle        = 0;
    unsigned int time_step    = 0;
    double       current_time = 0;

    make_grid();
    set_material_and_boundary_ids();
    output_grid(time_step, cycle);

    for (; current_time <= parameters.end_time;)
      {
        pcout << "Time step " << time_step << " @ " << current_time << "s."
              << std::endl;

        if (time_step == 0)
          {
            pcout << "Cycle " << cycle << ':' << std::endl;

            if (cycle > 0)
              {
                refine_grid();
                set_material_and_boundary_ids();
              }

            pcout << "   Number of active cells:       "
                  << triangulation.n_active_cells() << " (on "
                  << triangulation.n_levels() << " levels)" << std::endl;

            setup_system();

            pcout << "   Number of degrees of freedom: "
                  << dof_handler_mvp.n_dofs() << std::endl;
          }

#ifdef DEBUG
        // Check that the source term is truly
        // divergence free on this grid. This
        // is a strict requirement of the magnetic
        // vector potential formulation
        verify_source_terms();
#endif

        pcout << "   Assembling and solving... " << std::endl;
        if (parameters.use_voltage_excitation())
          {
            assemble_system_esp(system_matrix_esp,
                                system_rhs_esp,
                                constraints_esp);
            solve_esp(system_matrix_esp,
                      solution_esp,
                      system_rhs_esp,
                      constraints_esp,
                      locally_owned_dofs_esp,
                      dof_handler_esp);
          }

        assemble_system_mvp(system_matrix_mvp, system_rhs_mvp, constraints_mvp);
        solve_mvp(system_matrix_mvp,
                  solution_mvp,
                  system_rhs_mvp,
                  constraints_mvp,
                  locally_owned_dofs_mvp,
                  dof_handler_mvp);

        // Update solution_mvp rate / time derivative
        d_solution_mvp_dt = solution_mvp;
        d_solution_mvp_dt -= solution_mvp_t1;
        d_solution_mvp_dt /= parameters.delta_t;

        pcout << "   Postprocessing... " << std::endl;

        output_results(time_step, cycle);
        output_point_solution_mvp(time_step, cycle);

        if (time_step == 0 && cycle < n_cycles)
          {
            ++cycle;
          }
        else
          {
            pcout << "   Update at end of timestep... " << std::endl;
            // Update old solution_mvp (history variable)
            solution_mvp_t1 = solution_mvp;

            current_time += parameters.delta_t;
            ++time_step;
            cycle = 0;
          }
      }

    deallog << "OK" << std::endl;
  }

} // namespace StepTransientCurlCurl
