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

// This program implements an algorithm for the incompressible Navier-Stokes
// equations solving the whole system at once, i.e., without any projection
// equation.
// It is used as a baseline for the weak form tests.

#include "../weak_forms_tests.h"
#include "wf_common_tests/step-navier_stokes-beltrami.h"

namespace StepNavierStokesBeltrami
{
  template <int dim>
  class NavierStokesProblem : public NavierStokesProblemBase<dim>
  {
    //   using ScratchData = typename SIPGLaplace<dim>::ScratchData;

  public:
    NavierStokesProblem()
      : NavierStokesProblemBase<dim>()
    {}

    // protected:
    //   void
    //   assemble_system() override;
  };


  // template <int dim>
  // void
  // NavierStokesProblem<dim>::assemble_system()
  // {
  // }

} // namespace StepNavierStokesBeltrami

int
main(int argc, char **argv)
{
  initlog();
  deallog << std::setprecision(9);

  deallog.depth_file(1);

  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  using namespace dealii;
  try
    {
      const int                                          dim = 2;
      StepNavierStokesBeltrami::NavierStokesProblem<dim> navier_stokes_problem;
      navier_stokes_problem.run();
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
