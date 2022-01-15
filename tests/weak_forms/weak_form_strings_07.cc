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


// Check functor form stringization and printing
// - Cache functors

#include <weak_forms/cache_functors.h>
#include <weak_forms/symbolic_decorations.h>
#include <weak_forms/symbolic_operators.h>

#include "../weak_forms_tests.h"


template <int dim, int spacedim = dim, typename NumberType = double>
void
run()
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;

  // Customise the naming convensions, if we wish to.
  const SymbolicDecorations decorator;

  const ScalarCacheFunctor                  scalar("s", "s");
  const VectorCacheFunctor<dim>             vector("v", "v");
  const TensorCacheFunctor<2, dim>          tensor2("T2", "T");
  const TensorCacheFunctor<3, dim>          tensor3("T3", "P");
  const TensorCacheFunctor<4, dim>          tensor4("T4", "K");
  const SymmetricTensorCacheFunctor<2, dim> symm_tensor2("S2", "T");
  const SymmetricTensorCacheFunctor<4, dim> symm_tensor4("S4", "K");

  // Test strings
  {
    LogStream::Prefix prefix("string");

    deallog << "Scalar: " << scalar.as_ascii(decorator) << std::endl;
    deallog << "Vector: " << vector.as_ascii(decorator) << std::endl;
    deallog << "Tensor (rank 2): " << tensor2.as_ascii(decorator) << std::endl;
    deallog << "Tensor (rank 3): " << tensor3.as_ascii(decorator) << std::endl;
    deallog << "Tensor (rank 4): " << tensor4.as_ascii(decorator) << std::endl;
    deallog << "SymmetricTensor (rank 2): " << symm_tensor2.as_ascii(decorator)
            << std::endl;
    deallog << "SymmetricTensor (rank 4): " << symm_tensor4.as_ascii(decorator)
            << std::endl;

    deallog << std::endl;
  }

  // Test LaTeX
  {
    LogStream::Prefix prefix("LaTeX");

    deallog << "Scalar: " << scalar.as_latex(decorator) << std::endl;
    deallog << "Vector: " << vector.as_latex(decorator) << std::endl;
    deallog << "Tensor (rank 2): " << tensor2.as_latex(decorator) << std::endl;
    deallog << "Tensor (rank 3): " << tensor3.as_latex(decorator) << std::endl;
    deallog << "Tensor (rank 4): " << tensor4.as_latex(decorator) << std::endl;
    deallog << "SymmetricTensor (rank 2): " << symm_tensor2.as_latex(decorator)
            << std::endl;
    deallog << "SymmetricTensor (rank 4): " << symm_tensor4.as_latex(decorator)
            << std::endl;

    deallog << std::endl;
  }

  deallog << "OK" << std::endl << std::endl;
}


int
main()
{
  initlog();

  run<2>();
  run<3>();

  deallog << "OK" << std::endl;
}
