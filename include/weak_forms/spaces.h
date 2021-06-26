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

#ifndef dealii_weakforms_spaces_h
#define dealii_weakforms_spaces_h

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/meshworker/scratch_data.h>

#include <weak_forms/config.h>
#include <weak_forms/numbers.h>
#include <weak_forms/solution_storage.h>
#include <weak_forms/subspace_extractors.h>
#include <weak_forms/symbolic_decorations.h>
#include <weak_forms/symbolic_operators.h>
#include <weak_forms/type_traits.h>
#include <weak_forms/types.h>


WEAK_FORMS_NAMESPACE_OPEN


#ifndef DOXYGEN

// Forward declarations
namespace WeakForms
{
  template <int dim, int spacedim = dim>
  class TestFunction;
  template <int dim, int spacedim = dim>
  class TrialSolution;
  template <int dim, int spacedim = dim>
  class FieldSolution;

  namespace SubSpaceViews
  {
    template <typename SpaceType>
    class Scalar;
    template <typename SpaceType>
    class Vector;
    template <int rank, typename SpaceType>
    class Tensor;
    template <int rank, typename SpaceType>
    class SymmetricTensor;
  } // namespace SubSpaceViews


  namespace SelfLinearization
  {
    namespace internal
    {
      struct ConvertTo;
    } // namespace internal
  }   // namespace SelfLinearization

  /* --------------- Finite element spaces: Test functions --------------- */


  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<WeakForms::TestFunction<dim, spacedim>,
                                   WeakForms::Operators::SymbolicOpCodes::value>
  value(const WeakForms::TestFunction<dim, spacedim> &operand);



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TestFunction<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::gradient>
  gradient(const WeakForms::TestFunction<dim, spacedim> &operand);



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TestFunction<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::laplacian>
  laplacian(const WeakForms::TestFunction<dim, spacedim> &operand);



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TestFunction<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::hessian>
  hessian(const WeakForms::TestFunction<dim, spacedim> &operand);



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TestFunction<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::third_derivative>
  third_derivative(const WeakForms::TestFunction<dim, spacedim> &operand);



  /* ---------- Finite element spaces: Test functions (interface) ---------- */



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TestFunction<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::jump_in_values>
  jump_in_values(const WeakForms::TestFunction<dim, spacedim> &operand);



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TestFunction<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::jump_in_gradients>
  jump_in_gradients(const WeakForms::TestFunction<dim, spacedim> &operand);



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TestFunction<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::jump_in_hessians>
  jump_in_hessians(const WeakForms::TestFunction<dim, spacedim> &operand);



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TestFunction<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::jump_in_third_derivatives>
  jump_in_third_derivatives(
    const WeakForms::TestFunction<dim, spacedim> &operand);


  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TestFunction<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::average_of_values>
  average_of_values(const WeakForms::TestFunction<dim, spacedim> &operand);


  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TestFunction<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::average_of_gradients>
  average_of_gradients(const WeakForms::TestFunction<dim, spacedim> &operand);


  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TestFunction<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::average_of_hessians>
  average_of_hessians(const WeakForms::TestFunction<dim, spacedim> &operand);



  /* --------------- Finite element spaces: Trial solutions --------------- */



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<WeakForms::TrialSolution<dim, spacedim>,
                                   WeakForms::Operators::SymbolicOpCodes::value>
  value(const WeakForms::TrialSolution<dim, spacedim> &operand);



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TrialSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::gradient>
  gradient(const WeakForms::TrialSolution<dim, spacedim> &operand);



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TrialSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::laplacian>
  laplacian(const WeakForms::TrialSolution<dim, spacedim> &operand);



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TrialSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::hessian>
  hessian(const WeakForms::TrialSolution<dim, spacedim> &operand);



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TrialSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::third_derivative>
  third_derivative(const WeakForms::TrialSolution<dim, spacedim> &operand);



  /* --------- Finite element spaces: Trial solutions (interface) --------- */



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TrialSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::jump_in_values>
  jump_in_values(const WeakForms::TrialSolution<dim, spacedim> &operand);



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TrialSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::jump_in_gradients>
  jump_in_gradients(const WeakForms::TrialSolution<dim, spacedim> &operand);



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TrialSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::jump_in_hessians>
  jump_in_hessians(const WeakForms::TrialSolution<dim, spacedim> &operand);



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TrialSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::jump_in_third_derivatives>
  jump_in_third_derivatives(
    const WeakForms::TrialSolution<dim, spacedim> &operand);



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TrialSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::average_of_values>
  average_of_values(const WeakForms::TrialSolution<dim, spacedim> &operand);



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TrialSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::average_of_gradients>
  average_of_gradients(const WeakForms::TrialSolution<dim, spacedim> &operand);



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TrialSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::average_of_hessians>
  average_of_hessians(const WeakForms::TrialSolution<dim, spacedim> &operand);



  /* --------------- Finite element spaces: Solution fields --------------- */

  namespace internal
  {
    // Used to work around the restriction that template arguments
    // for template type parameter must be a type
    template <types::solution_index solution_index_>
    struct SolutionIndex
    {
      static const types::solution_index solution_index = solution_index_;
    };
  } // namespace internal



  template <types::solution_index solution_index = 0, int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::FieldSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::value,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  value(const WeakForms::FieldSolution<dim, spacedim> &operand);



  template <types::solution_index solution_index = 0, int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::FieldSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::gradient,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  gradient(const WeakForms::FieldSolution<dim, spacedim> &operand);



  template <types::solution_index solution_index = 0, int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::FieldSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::laplacian,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  laplacian(const WeakForms::FieldSolution<dim, spacedim> &operand);



  template <types::solution_index solution_index = 0, int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::FieldSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::hessian,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  hessian(const WeakForms::FieldSolution<dim, spacedim> &operand);



  template <types::solution_index solution_index = 0, int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::FieldSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::third_derivative,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  third_derivative(const WeakForms::FieldSolution<dim, spacedim> &operand);

} // namespace WeakForms

#endif // DOXYGEN


namespace WeakForms
{
  template <int dim, int spacedim>
  class Space
  {
  public:
    /**
     * Dimension in which this object operates.
     */
    static const unsigned int dimension = dim;

    /**
     * Dimension of the space in which this object operates.
     */
    static const unsigned int space_dimension = spacedim;

    /**
     * Rank of continuous space
     */
    static const int rank = 0;

    using FEValuesViewsType = FEValuesViews::Scalar<dimension, space_dimension>;

    template <typename ScalarType>
    using value_type =
      typename FEValuesViewsType::template solution_value_type<ScalarType>;

    template <typename ScalarType>
    using gradient_type =
      typename FEValuesViewsType::template solution_gradient_type<ScalarType>;

    template <typename ScalarType>
    using hessian_type =
      typename FEValuesViewsType::template solution_hessian_type<ScalarType>;

    template <typename ScalarType>
    using laplacian_type =
      typename FEValuesViewsType::template solution_laplacian_type<ScalarType>;

    template <typename ScalarType>
    using third_derivative_type =
      typename FEValuesViewsType::template solution_third_derivative_type<
        ScalarType>;

    // using FEInterfaceValuesViewsType = FEInterfaceViews::Scalar<dimension,
    // space_dimension>;

    // template <typename ScalarType>
    // using interface_value_type =
    //   typename FEValuesViewsType::template solution_value_type<ScalarType>;

    // template <typename ScalarType>
    // using interface_gradient_type =
    //   typename FEValuesViewsType::template
    //   solution_gradient_type<ScalarType>;

    // template <typename ScalarType>
    // using interface_hessian_type =
    //   typename FEValuesViewsType::template solution_hessian_type<ScalarType>;

    // template <typename ScalarType>
    // using interface_third_derivative_type =
    //   typename FEValuesViewsType::template solution_third_derivative_type<
    //     ScalarType>;

    virtual ~Space() = default;

    virtual Space *
    clone() const = 0;

    types::field_index
    get_field_index() const
    {
      return field_index;
    }

    // ----  Ascii ----

    std::string
    as_ascii(const SymbolicDecorations &decorator) const
    {
      return decorator.symbolic_op_operand_as_ascii(*this);
    }

    virtual std::string
    get_field_ascii(const SymbolicDecorations &decorator) const
    {
      (void)decorator;
      return field_ascii;
    }

    virtual std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const = 0;

    // ---- LaTeX ----

    std::string
    as_latex(const SymbolicDecorations &decorator) const
    {
      return decorator.symbolic_op_operand_as_latex(*this);
    }

    virtual std::string
    get_field_latex(const SymbolicDecorations &decorator) const
    {
      (void)decorator;
      return field_latex;
    }

    virtual std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const = 0;

  protected:
    // Allow access to get_field_ascii_raw() and get_field_latex_raw()
    friend WeakForms::SelfLinearization::internal::ConvertTo;

    // Create a subspace
    Space(const types::field_index field_index,
          const std::string &      field_ascii,
          const std::string &      field_latex)
      : field_index(field_index)
      , field_ascii(field_ascii)
      , field_latex(field_latex != "" ? field_latex : field_ascii)
    {}

    Space(const Space &) = default;

    const std::string &
    get_field_ascii_raw() const
    {
      return field_ascii;
    }

    const std::string &
    get_field_latex_raw() const
    {
      return field_latex;
    }

  private:
    const types::field_index field_index;
    const std::string        field_ascii;
    const std::string        field_latex;
  };



  template <int dim, int spacedim>
  class TestFunction : public Space<dim, spacedim>
  {
  public:
    // Full space
    TestFunction()
      : TestFunction(numbers::invalid_field_index, "", "")
    {}

    TestFunction(const TestFunction &) = default;

    virtual TestFunction *
    clone() const override
    {
      return new TestFunction(*this);
    }

    std::string
    get_field_ascii(const SymbolicDecorations &decorator) const override
    {
      if (this->get_field_ascii_raw().empty())
        {
          const auto &naming = decorator.get_naming_ascii().discretization;
          return naming.solution_field;
        }
      else
        return this->get_field_ascii_raw();
    }

    std::string
    get_field_latex(const SymbolicDecorations &decorator) const override
    {
      if (this->get_field_latex_raw().empty())
        {
          const auto &naming = decorator.get_naming_latex().discretization;
          return naming.solution_field;
        }
      else
        return this->get_field_latex_raw();
    }

    std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const override
    {
      const auto &naming = decorator.get_naming_ascii().discretization;
      return naming.test_function;
    }

    std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const override
    {
      const auto &naming = decorator.get_naming_latex().discretization;
      return naming.test_function;
    }

    // Cell / face

    auto
    value() const
    {
      return WeakForms::value(*this);
    }

    auto
    gradient() const
    {
      return WeakForms::gradient(*this);
    }

    auto
    laplacian() const
    {
      return WeakForms::laplacian(*this);
    }

    auto
    hessian() const
    {
      return WeakForms::hessian(*this);
    }

    auto
    third_derivative() const
    {
      return WeakForms::third_derivative(*this);
    }

    SubSpaceViews::Scalar<TestFunction>
    operator[](const SubSpaceExtractors::Scalar &extractor) const
    {
      const TestFunction subspace(extractor.field_index,
                                  extractor.field_ascii,
                                  extractor.field_latex);
      return SubSpaceViews::Scalar<TestFunction>(subspace, extractor.extractor);
    }

    SubSpaceViews::Vector<TestFunction>
    operator[](const SubSpaceExtractors::Vector &extractor) const
    {
      const TestFunction subspace(extractor.field_index,
                                  extractor.field_ascii,
                                  extractor.field_latex);
      return SubSpaceViews::Vector<TestFunction>(subspace, extractor.extractor);
    }

    template <int rank>
    SubSpaceViews::Tensor<rank, TestFunction>
    operator[](const SubSpaceExtractors::Tensor<rank> &extractor) const
    {
      const TestFunction subspace(extractor.field_index,
                                  extractor.field_ascii,
                                  extractor.field_latex);
      return SubSpaceViews::Tensor<rank, TestFunction>(subspace,
                                                       extractor.extractor);
    }

    template <int rank>
    SubSpaceViews::SymmetricTensor<rank, TestFunction>
    operator[](const SubSpaceExtractors::SymmetricTensor<rank> &extractor) const
    {
      const TestFunction subspace(extractor.field_index,
                                  extractor.field_ascii,
                                  extractor.field_latex);
      return SubSpaceViews::SymmetricTensor<rank, TestFunction>(
        subspace, extractor.extractor);
    }

    // Interface

    auto
    jump_in_values() const
    {
      return WeakForms::jump_in_values(*this);
    }

    auto
    jump_in_gradients() const
    {
      return WeakForms::jump_in_gradients(*this);
    }

    auto
    jump_in_hessians() const
    {
      return WeakForms::jump_in_hessians(*this);
    }

    auto
    jump_in_third_derivatives() const
    {
      return WeakForms::jump_in_third_derivatives(*this);
    }

    auto
    average_of_values() const
    {
      return WeakForms::average_of_values(*this);
    }

    auto
    average_of_gradients() const
    {
      return WeakForms::average_of_gradients(*this);
    }

    auto
    average_of_hessians() const
    {
      return WeakForms::average_of_hessians(*this);
    }

  protected:
    // Subspace
    TestFunction(const types::field_index field_index,
                 const std::string &      field_ascii,
                 const std::string &      field_latex)
      : Space<dim, spacedim>(field_index, field_ascii, field_latex)
    {}
  };



  template <int dim, int spacedim>
  class TrialSolution : public Space<dim, spacedim>
  {
  public:
    // Full space
    TrialSolution()
      : TrialSolution(numbers::invalid_field_index, "", "")
    {}

    TrialSolution(const TrialSolution &) = default;

    virtual TrialSolution *
    clone() const override
    {
      return new TrialSolution(*this);
    }

    std::string
    get_field_ascii(const SymbolicDecorations &decorator) const override
    {
      if (this->get_field_ascii_raw().empty())
        {
          const auto &naming = decorator.get_naming_ascii().discretization;
          return naming.solution_field;
        }
      else
        return this->get_field_ascii_raw();
    }

    std::string
    get_field_latex(const SymbolicDecorations &decorator) const override
    {
      if (this->get_field_latex_raw().empty())
        {
          const auto &naming = decorator.get_naming_latex().discretization;
          return naming.solution_field;
        }
      else
        return this->get_field_latex_raw();
    }

    std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const override
    {
      const auto &naming = decorator.get_naming_ascii().discretization;
      return naming.trial_solution;
    }

    std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const override
    {
      const auto &naming = decorator.get_naming_latex().discretization;
      return naming.trial_solution;
    }

    // Cell / face

    auto
    value() const
    {
      return WeakForms::value(*this);
    }

    auto
    gradient() const
    {
      return WeakForms::gradient(*this);
    }

    auto
    laplacian() const
    {
      return WeakForms::laplacian(*this);
    }

    auto
    hessian() const
    {
      return WeakForms::hessian(*this);
    }

    auto
    third_derivative() const
    {
      return WeakForms::third_derivative(*this);
    }

    SubSpaceViews::Scalar<TrialSolution>
    operator[](const SubSpaceExtractors::Scalar &extractor) const
    {
      const TrialSolution subspace(extractor.field_index,
                                   extractor.field_ascii,
                                   extractor.field_latex);
      return SubSpaceViews::Scalar<TrialSolution>(subspace,
                                                  extractor.extractor);
    }

    SubSpaceViews::Vector<TrialSolution>
    operator[](const SubSpaceExtractors::Vector &extractor) const
    {
      const TrialSolution subspace(extractor.field_index,
                                   extractor.field_ascii,
                                   extractor.field_latex);
      return SubSpaceViews::Vector<TrialSolution>(subspace,
                                                  extractor.extractor);
    }

    template <int rank>
    SubSpaceViews::Tensor<rank, TrialSolution>
    operator[](const SubSpaceExtractors::Tensor<rank> &extractor) const
    {
      const TrialSolution subspace(extractor.field_index,
                                   extractor.field_ascii,
                                   extractor.field_latex);
      return SubSpaceViews::Tensor<rank, TrialSolution>(subspace,
                                                        extractor.extractor);
    }

    template <int rank>
    SubSpaceViews::SymmetricTensor<rank, TrialSolution>
    operator[](const SubSpaceExtractors::SymmetricTensor<rank> &extractor) const
    {
      const TrialSolution subspace(extractor.field_index,
                                   extractor.field_ascii,
                                   extractor.field_latex);
      return SubSpaceViews::SymmetricTensor<rank, TrialSolution>(
        subspace, extractor.extractor);
    }

    // Interface

    auto
    jump_in_values() const
    {
      return WeakForms::jump_in_values(*this);
    }

    auto
    jump_in_gradients() const
    {
      return WeakForms::jump_in_gradients(*this);
    }

    auto
    jump_in_hessians() const
    {
      return WeakForms::jump_in_hessians(*this);
    }

    auto
    jump_in_third_derivatives() const
    {
      return WeakForms::jump_in_third_derivatives(*this);
    }

    auto
    average_of_values() const
    {
      return WeakForms::average_of_values(*this);
    }

    auto
    average_of_gradients() const
    {
      return WeakForms::average_of_gradients(*this);
    }

    auto
    average_of_hessians() const
    {
      return WeakForms::average_of_hessians(*this);
    }

  protected:
    // Subspace
    TrialSolution(const types::field_index field_index,
                  const std::string &      field_ascii,
                  const std::string &      field_latex)
      : Space<dim, spacedim>(field_index, field_ascii, field_latex)
    {}
  };



  template <int dim, int spacedim>
  class FieldSolution : public Space<dim, spacedim>
  {
  public:
    // Full space
    FieldSolution()
      : FieldSolution(numbers::invalid_field_index, "", "")
    {}

    FieldSolution(const FieldSolution &) = default;

    virtual FieldSolution *
    clone() const override
    {
      return new FieldSolution(*this);
    }

    std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const override
    {
      // Don't double-print names for fields
      if (this->get_field_ascii(decorator) == "")
        {
          const auto &naming = decorator.get_naming_ascii().discretization;
          return naming.solution_field;
        }
      else
        return "";
    }

    std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const override
    {
      // Don't double-print names for fields
      if (this->get_field_latex(decorator) == "")
        {
          const auto &naming = decorator.get_naming_latex().discretization;
          return naming.solution_field;
        }
      else
        return "";
    }

    // Cell / face

    template <types::solution_index solution_index = 0>
    auto
    value() const
    {
      return WeakForms::value<solution_index>(*this);
    }

    template <types::solution_index solution_index = 0>
    auto
    gradient() const
    {
      return WeakForms::gradient<solution_index>(*this);
    }

    template <types::solution_index solution_index = 0>
    auto
    laplacian() const
    {
      return WeakForms::laplacian<solution_index>(*this);
    }

    template <types::solution_index solution_index = 0>
    auto
    hessian() const
    {
      return WeakForms::hessian<solution_index>(*this);
    }

    template <types::solution_index solution_index = 0>
    auto
    third_derivative() const
    {
      return WeakForms::third_derivative<solution_index>(*this);
    }

    SubSpaceViews::Scalar<FieldSolution>
    operator[](const SubSpaceExtractors::Scalar &extractor) const
    {
      const FieldSolution subspace(extractor.field_index,
                                   extractor.field_ascii,
                                   extractor.field_latex);
      return SubSpaceViews::Scalar<FieldSolution>(subspace,
                                                  extractor.extractor);
    }

    SubSpaceViews::Vector<FieldSolution>
    operator[](const SubSpaceExtractors::Vector &extractor) const
    {
      const FieldSolution subspace(extractor.field_index,
                                   extractor.field_ascii,
                                   extractor.field_latex);
      return SubSpaceViews::Vector<FieldSolution>(subspace,
                                                  extractor.extractor);
    }

    template <int rank>
    SubSpaceViews::Tensor<rank, FieldSolution>
    operator[](const SubSpaceExtractors::Tensor<rank> &extractor) const
    {
      const FieldSolution subspace(extractor.field_index,
                                   extractor.field_ascii,
                                   extractor.field_latex);
      return SubSpaceViews::Tensor<rank, FieldSolution>(subspace,
                                                        extractor.extractor);
    }

    template <int rank>
    SubSpaceViews::SymmetricTensor<rank, FieldSolution>
    operator[](const SubSpaceExtractors::SymmetricTensor<rank> &extractor) const
    {
      const FieldSolution subspace(extractor.field_index,
                                   extractor.field_ascii,
                                   extractor.field_latex);
      return SubSpaceViews::SymmetricTensor<rank, FieldSolution>(
        subspace, extractor.extractor);
    }

    // Interface

  protected:
    // Subspace
    FieldSolution(const types::field_index field_index,
                  const std::string &      field_ascii,
                  const std::string &      field_latex)
      : Space<dim, spacedim>(field_index, field_ascii, field_latex)
    {}
  };



  namespace NonLinear
  {
    template <int dim, int spacedim = dim>
    using Variation = WeakForms::TestFunction<dim, spacedim>;

    template <int dim, int spacedim = dim>
    using Linearization = WeakForms::TrialSolution<dim, spacedim>;

    template <int dim, int spacedim = dim>
    using FieldSolution = WeakForms::FieldSolution<dim, spacedim>;
  } // namespace NonLinear

} // namespace WeakForms



/* ================== Specialization of unary operators ================== */

namespace WeakForms
{
  namespace Operators
  {
    /**
     *
     * @note Add at end due to reliance on @p value_type
     */
#define DEAL_II_SPACE_SYMBOLIC_OP_COMMON_IMPL(ClassName, Op_, solution_index_) \
public:                                                                        \
  /**                                                                          \
   *                                                                           \
   */                                                                          \
  using Op = Op_;                                                              \
  /**                                                                          \
   * Value at all quadrature points                                            \
   */                                                                          \
  template <typename ScalarType>                                               \
  using qp_value_type = std::vector<value_type<ScalarType>>;                   \
  /**                                                                          \
   * Value for all DoFs at all quadrature points                               \
   */                                                                          \
  template <typename ScalarType>                                               \
  using dof_value_type = std::vector<qp_value_type<ScalarType>>;               \
                                                                               \
  template <typename ScalarType, std::size_t width>                            \
  using vectorized_qp_value_type = typename numbers::VectorizedValue<          \
    value_type<ScalarType>>::template type<width>;                             \
                                                                               \
  template <typename ScalarType, std::size_t width>                            \
  using vectorized_dof_value_type =                                            \
    std::vector<vectorized_qp_value_type<ScalarType, width>>;                  \
  /**                                                                          \
   * The index in the solution history that this field solution                \
   * corresponds to. The default value (0) indicates that it relates           \
   * to the current solution.                                                  \
   */                                                                          \
  static const types::solution_index solution_index = solution_index_;         \
                                                                               \
  types::field_index get_field_index() const                                   \
  {                                                                            \
    return get_operand().get_field_index();                                    \
  }                                                                            \
                                                                               \
protected:                                                                     \
  /**                                                                          \
   * Allow access to get_operand()                                             \
   */                                                                          \
  friend WeakForms::SelfLinearization::internal::ConvertTo;                    \
  /**                                                                          \
   * Only want this to be a base class                                         \
   */                                                                          \
  explicit ClassName(const Op &operand)                                        \
    : operand(operand.clone())                                                 \
  {}                                                                           \
                                                                               \
  const Op &get_operand() const                                                \
  {                                                                            \
    Assert(operand, ExcNotInitialized());                                      \
    return *operand;                                                           \
  }                                                                            \
                                                                               \
private:                                                                       \
  const std::shared_ptr<Op> operand;


    /* ---- Mix-in classes ---- */

    // Cell / face

    template <typename Op_, types::solution_index solution_index_ = 0>
    class SymbolicOpValueBase
    {
    public:
      static const enum SymbolicOpCodes op_code = SymbolicOpCodes::value;

      // The value_type<> might be a scalar or tensor, so we can't fetch the
      // rank from it.
      static const int rank = Op_::rank;

      template <typename ScalarType>
      using value_type = typename Op_::template value_type<ScalarType>;

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_ascii().differential_operators;
        return decorator.decorate_with_operator_ascii(
          naming.value,
          decorator.make_time_indexed_symbol_ascii(
            get_operand().as_ascii(decorator), solution_index));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_latex().differential_operators;
        return decorator.decorate_with_operator_latex(
          naming.value,
          decorator.make_time_indexed_symbol_latex(
            get_operand().as_latex(decorator), solution_index));
      }

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_values;
      }

      DEAL_II_SPACE_SYMBOLIC_OP_COMMON_IMPL(SymbolicOpValueBase,
                                            Op_,
                                            solution_index_)
    };


    template <typename Op_, types::solution_index solution_index_ = 0>
    class SymbolicOpGradientBase
    {
    public:
      static const enum SymbolicOpCodes op_code = SymbolicOpCodes::gradient;

      template <typename ScalarType>
      using value_type = typename Op_::template gradient_type<ScalarType>;

      static const int rank = value_type<double>::rank;

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_ascii().differential_operators;
        return decorator.decorate_with_operator_ascii(
          naming.gradient,
          decorator.make_time_indexed_symbol_ascii(
            get_operand().as_ascii(decorator), solution_index));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_latex().differential_operators;
        return decorator.decorate_with_operator_latex(
          naming.gradient,
          decorator.make_time_indexed_symbol_latex(
            get_operand().as_latex(decorator), solution_index));
      }

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_gradients;
      }

      DEAL_II_SPACE_SYMBOLIC_OP_COMMON_IMPL(SymbolicOpGradientBase,
                                            Op_,
                                            solution_index_)
    };


    template <typename Op_, types::solution_index solution_index_ = 0>
    class SymbolicOpSymmetricGradientBase
    {
    public:
      static const enum SymbolicOpCodes op_code =
        SymbolicOpCodes::symmetric_gradient;

      template <typename ScalarType>
      using value_type =
        typename Op_::template symmetric_gradient_type<ScalarType>;

      static const int rank = value_type<double>::rank;

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_ascii().differential_operators;
        return decorator.decorate_with_operator_ascii(
          naming.symmetric_gradient,
          decorator.make_time_indexed_symbol_ascii(
            get_operand().as_ascii(decorator), solution_index));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_latex().differential_operators;
        return decorator.decorate_with_operator_latex(
          naming.symmetric_gradient,
          decorator.make_time_indexed_symbol_latex(
            get_operand().as_latex(decorator), solution_index));
      }

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_gradients;
      }

      DEAL_II_SPACE_SYMBOLIC_OP_COMMON_IMPL(SymbolicOpSymmetricGradientBase,
                                            Op_,
                                            solution_index_)
    };


    template <typename Op_, types::solution_index solution_index_ = 0>
    class SymbolicOpDivergenceBase
    {
    public:
      static const enum SymbolicOpCodes op_code = SymbolicOpCodes::divergence;

      // The value_type<> might be a scalar or tensor, so we can't fetch the
      // rank from it.
      static const int rank = Op_::rank;

      template <typename ScalarType>
      using value_type = typename Op_::template divergence_type<ScalarType>;

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_ascii().differential_operators;
        return decorator.decorate_with_operator_ascii(
          naming.divergence,
          decorator.make_time_indexed_symbol_ascii(
            get_operand().as_ascii(decorator), solution_index));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_latex().differential_operators;
        return decorator.decorate_with_operator_latex(
          naming.divergence,
          decorator.make_time_indexed_symbol_latex(
            get_operand().as_latex(decorator), solution_index));
      }

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_gradients;
      }

      DEAL_II_SPACE_SYMBOLIC_OP_COMMON_IMPL(SymbolicOpDivergenceBase,
                                            Op_,
                                            solution_index_)
    };


    template <typename Op_, types::solution_index solution_index_ = 0>
    class SymbolicOpCurlBase
    {
    public:
      static const enum SymbolicOpCodes op_code = SymbolicOpCodes::curl;

      template <typename ScalarType>
      using value_type = typename Op_::template curl_type<ScalarType>;

      static const int rank = value_type<double>::rank;
      static_assert(rank == 1, "Invalid rank for curl operation");

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_ascii().differential_operators;
        return decorator.decorate_with_operator_ascii(
          naming.curl,
          decorator.make_time_indexed_symbol_ascii(
            get_operand().as_ascii(decorator), solution_index));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_latex().differential_operators;
        return decorator.decorate_with_operator_latex(
          naming.curl,
          decorator.make_time_indexed_symbol_latex(
            get_operand().as_latex(decorator), solution_index));
      }

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_gradients;
      }

      DEAL_II_SPACE_SYMBOLIC_OP_COMMON_IMPL(SymbolicOpCurlBase,
                                            Op_,
                                            solution_index_)
    };


    template <typename Op_, types::solution_index solution_index_ = 0>
    class SymbolicOpLaplacianBase
    {
    public:
      static const enum SymbolicOpCodes op_code = SymbolicOpCodes::laplacian;

      // The value_type<> might be a scalar or tensor, so we can't fetch the
      // rank from it.
      static const int rank = Op_::rank;

      template <typename ScalarType>
      using value_type = typename Op_::template laplacian_type<ScalarType>;

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_ascii().differential_operators;
        return decorator.decorate_with_operator_ascii(
          naming.laplacian,
          decorator.make_time_indexed_symbol_ascii(
            get_operand().as_ascii(decorator), solution_index));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_latex().differential_operators;
        return decorator.decorate_with_operator_latex(
          naming.laplacian,
          decorator.make_time_indexed_symbol_latex(
            get_operand().as_latex(decorator), solution_index));
      }

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_hessians;
      }

      DEAL_II_SPACE_SYMBOLIC_OP_COMMON_IMPL(SymbolicOpLaplacianBase,
                                            Op_,
                                            solution_index_)
    };


    template <typename Op_, types::solution_index solution_index_ = 0>
    class SymbolicOpHessianBase
    {
    public:
      static const enum SymbolicOpCodes op_code = SymbolicOpCodes::hessian;

      template <typename ScalarType>
      using value_type = typename Op_::template hessian_type<ScalarType>;

      static const int rank = value_type<double>::rank;

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_ascii().differential_operators;
        return decorator.decorate_with_operator_ascii(
          naming.hessian,
          decorator.make_time_indexed_symbol_ascii(
            get_operand().as_ascii(decorator), solution_index));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_latex().differential_operators;
        return decorator.decorate_with_operator_latex(
          naming.hessian,
          decorator.make_time_indexed_symbol_latex(
            get_operand().as_latex(decorator), solution_index));
      }

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_hessians;
      }

      DEAL_II_SPACE_SYMBOLIC_OP_COMMON_IMPL(SymbolicOpHessianBase,
                                            Op_,
                                            solution_index_)
    };


    template <typename Op_, types::solution_index solution_index_ = 0>
    class SymbolicOpThirdDerivativeBase
    {
    public:
      static const enum SymbolicOpCodes op_code =
        SymbolicOpCodes::third_derivative;

      template <typename ScalarType>
      using value_type =
        typename Op_::template third_derivative_type<ScalarType>;

      static const int rank = value_type<double>::rank;

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_ascii().differential_operators;
        return decorator.decorate_with_operator_ascii(
          naming.third_derivative,
          decorator.make_time_indexed_symbol_ascii(
            get_operand().as_ascii(decorator), solution_index));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_latex().differential_operators;
        return decorator.decorate_with_operator_latex(
          naming.third_derivative,
          decorator.make_time_indexed_symbol_latex(
            get_operand().as_latex(decorator), solution_index));
      }

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_3rd_derivatives;
      }

      DEAL_II_SPACE_SYMBOLIC_OP_COMMON_IMPL(SymbolicOpThirdDerivativeBase,
                                            Op_,
                                            solution_index_)
    };


    // Interface


    template <typename Op_, types::solution_index solution_index_ = 0>
    class SymbolicOpJumpValueBase
    {
    public:
      static const enum SymbolicOpCodes op_code =
        SymbolicOpCodes::jump_in_values;

      // The value_type<> might be a scalar or tensor, so we can't fetch the
      // rank from it.
      static const int rank = Op_::rank;

      template <typename ScalarType>
      using value_type = typename Op_::template value_type<ScalarType>;

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_ascii().differential_operators;
        return decorator.make_jump_symbol_ascii(
          decorator.decorate_with_operator_ascii(
            naming.value,
            decorator.make_time_indexed_symbol_ascii(
              get_operand().as_ascii(decorator), solution_index)));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_latex().differential_operators;
        return decorator.make_jump_symbol_latex(
          decorator.decorate_with_operator_latex(
            naming.value,
            decorator.make_time_indexed_symbol_latex(
              get_operand().as_latex(decorator), solution_index)));
      }

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_values;
      }

      DEAL_II_SPACE_SYMBOLIC_OP_COMMON_IMPL(SymbolicOpJumpValueBase,
                                            Op_,
                                            solution_index_)
    };


    template <typename Op_, types::solution_index solution_index_ = 0>
    class SymbolicOpJumpGradientBase
    {
    public:
      static const enum SymbolicOpCodes op_code =
        SymbolicOpCodes::jump_in_gradients;

      template <typename ScalarType>
      using value_type = typename Op_::template gradient_type<ScalarType>;

      static const int rank = value_type<double>::rank;

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_ascii().differential_operators;
        return decorator.make_jump_symbol_ascii(
          decorator.decorate_with_operator_ascii(
            naming.gradient,
            decorator.make_time_indexed_symbol_ascii(
              get_operand().as_ascii(decorator), solution_index)));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_latex().differential_operators;
        return decorator.make_jump_symbol_latex(
          decorator.decorate_with_operator_latex(
            naming.gradient,
            decorator.make_time_indexed_symbol_latex(
              get_operand().as_latex(decorator), solution_index)));
      }

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_gradients;
      }

      DEAL_II_SPACE_SYMBOLIC_OP_COMMON_IMPL(SymbolicOpJumpGradientBase,
                                            Op_,
                                            solution_index_)
    };


    template <typename Op_, types::solution_index solution_index_ = 0>
    class SymbolicOpJumpHessianBase
    {
    public:
      static const enum SymbolicOpCodes op_code =
        SymbolicOpCodes::jump_in_hessians;

      template <typename ScalarType>
      using value_type = typename Op_::template hessian_type<ScalarType>;

      static const int rank = value_type<double>::rank;

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_ascii().differential_operators;
        return decorator.make_jump_symbol_ascii(
          decorator.decorate_with_operator_ascii(
            naming.hessian,
            decorator.make_time_indexed_symbol_ascii(
              get_operand().as_ascii(decorator), solution_index)));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_latex().differential_operators;
        return decorator.make_jump_symbol_latex(
          decorator.decorate_with_operator_latex(
            naming.hessian,
            decorator.make_time_indexed_symbol_latex(
              get_operand().as_latex(decorator), solution_index)));
      }

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_hessians;
      }

      DEAL_II_SPACE_SYMBOLIC_OP_COMMON_IMPL(SymbolicOpJumpHessianBase,
                                            Op_,
                                            solution_index_)
    };


    template <typename Op_, types::solution_index solution_index_ = 0>
    class SymbolicOpJumpThirdDerivativeBase
    {
    public:
      static const enum SymbolicOpCodes op_code =
        SymbolicOpCodes::jump_in_third_derivatives;

      template <typename ScalarType>
      using value_type =
        typename Op_::template third_derivative_type<ScalarType>;

      static const int rank = value_type<double>::rank;

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_ascii().differential_operators;
        return decorator.make_jump_symbol_ascii(
          decorator.decorate_with_operator_ascii(
            naming.third_derivative,
            decorator.make_time_indexed_symbol_ascii(
              get_operand().as_ascii(decorator), solution_index)));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_latex().differential_operators;
        return decorator.make_jump_symbol_latex(
          decorator.decorate_with_operator_latex(
            naming.third_derivative,
            decorator.make_time_indexed_symbol_latex(
              get_operand().as_latex(decorator), solution_index)));
      }

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_3rd_derivatives;
      }

      DEAL_II_SPACE_SYMBOLIC_OP_COMMON_IMPL(SymbolicOpJumpThirdDerivativeBase,
                                            Op_,
                                            solution_index_)
    };


    template <typename Op_, types::solution_index solution_index_ = 0>
    class SymbolicOpAverageValueBase
    {
    public:
      static const enum SymbolicOpCodes op_code =
        SymbolicOpCodes::average_of_values;

      // The value_type<> might be a scalar or tensor, so we can't fetch the
      // rank from it.
      static const int rank = Op_::rank;

      template <typename ScalarType>
      using value_type = typename Op_::template value_type<ScalarType>;

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_ascii().differential_operators;
        return decorator.make_average_symbol_ascii(
          decorator.decorate_with_operator_ascii(
            naming.value,
            decorator.make_time_indexed_symbol_ascii(
              get_operand().as_ascii(decorator), solution_index)));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_latex().differential_operators;
        return decorator.make_average_symbol_latex(
          decorator.decorate_with_operator_latex(
            naming.value,
            decorator.make_time_indexed_symbol_latex(
              get_operand().as_latex(decorator), solution_index)));
      }

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_values;
      }

      DEAL_II_SPACE_SYMBOLIC_OP_COMMON_IMPL(SymbolicOpAverageValueBase,
                                            Op_,
                                            solution_index_)
    };


    template <typename Op_, types::solution_index solution_index_ = 0>
    class SymbolicOpAverageGradientBase
    {
    public:
      static const enum SymbolicOpCodes op_code =
        SymbolicOpCodes::average_of_gradients;

      template <typename ScalarType>
      using value_type = typename Op_::template gradient_type<ScalarType>;

      static const int rank = value_type<double>::rank;

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_ascii().differential_operators;
        return decorator.make_average_symbol_ascii(
          decorator.decorate_with_operator_ascii(
            naming.gradient,
            decorator.make_time_indexed_symbol_ascii(
              get_operand().as_ascii(decorator), solution_index)));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_latex().differential_operators;
        return decorator.make_average_symbol_latex(
          decorator.decorate_with_operator_latex(
            naming.gradient,
            decorator.make_time_indexed_symbol_latex(
              get_operand().as_latex(decorator), solution_index)));
      }

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_gradients;
      }

      DEAL_II_SPACE_SYMBOLIC_OP_COMMON_IMPL(SymbolicOpAverageGradientBase,
                                            Op_,
                                            solution_index_)
    };


    template <typename Op_, types::solution_index solution_index_ = 0>
    class SymbolicOpAverageHessianBase
    {
    public:
      static const enum SymbolicOpCodes op_code =
        SymbolicOpCodes::average_of_hessians;

      template <typename ScalarType>
      using value_type = typename Op_::template hessian_type<ScalarType>;

      static const int rank = value_type<double>::rank;

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_ascii().differential_operators;
        return decorator.make_average_symbol_ascii(
          decorator.decorate_with_operator_ascii(
            naming.hessian,
            decorator.make_time_indexed_symbol_ascii(
              get_operand().as_ascii(decorator), solution_index)));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming =
          decorator.get_naming_latex().differential_operators;
        return decorator.make_average_symbol_latex(
          decorator.decorate_with_operator_latex(
            naming.hessian,
            decorator.make_time_indexed_symbol_latex(
              get_operand().as_latex(decorator), solution_index)));
      }

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_hessians;
      }

      DEAL_II_SPACE_SYMBOLIC_OP_COMMON_IMPL(SymbolicOpAverageHessianBase,
                                            Op_,
                                            solution_index_)
    };


    // template <typename Op_, types::solution_index solution_index_ = 0>
    // class SymbolicOpAverageThirdDerivativeBase
    // {
    // public:
    //   static const enum SymbolicOpCodes op_code =
    //     SymbolicOpCodes::average_of_third_derivatives;

    //   template <typename ScalarType>
    //   using value_type =
    //     typename Op_::template third_derivative_type<ScalarType>;

    //   static const int rank = value_type<double>::rank;

    //   std::string
    //   as_ascii(const SymbolicDecorations &decorator) const
    //   {
    //     const auto &naming =
    //       decorator.get_naming_ascii().differential_operators;
    //     return
    //     decorator.make_average_symbol_ascii(decorator.decorate_with_operator_ascii(
    //       naming.third_derivative,
    //       decorator.make_time_indexed_symbol_ascii(
    //         get_operand().as_ascii(decorator), solution_index)));
    //   }

    //   std::string
    //   as_latex(const SymbolicDecorations &decorator) const
    //   {
    //     const auto &naming =
    //       decorator.get_naming_latex().differential_operators;
    //     return
    //     decorator.make_average_symbol_latex(decorator.decorate_with_operator_latex(
    //       naming.third_derivative,
    //       decorator.make_time_indexed_symbol_latex(
    //         get_operand().as_latex(decorator), solution_index)));
    //   }

    //   UpdateFlags
    //   get_update_flags() const
    //   {
    //     return UpdateFlags::update_3rd_derivatives;
    //   }

    //   DEAL_II_SPACE_SYMBOLIC_OP_COMMON_IMPL(SymbolicOpAverageThirdDerivativeBase,
    //                                         Op_,
    //                                         solution_index_)
    // };



#undef DEAL_II_SPACE_SYMBOLIC_OP_COMMON_IMPL


    /* ---- Finite element spaces: Test functions and trial solutions ---- */

/**
 * A macro to implement the common parts of a symbolic op class
 * for test functions and trial solution spaces.
 * It is expected that the unary op derives from a
 * SymbolicOp[TYPE]Base<Space<dim, spacedim>> .
 *
 * @note It is intended that this should used immediately after class
 * definition is opened.
 */
#define DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SPACE_COMMON_IMPL(SymbolicOpBaseType, \
                                                         dim,                \
                                                         spacedim)           \
private:                                                                     \
  using Base_t = SymbolicOpBaseType<Space<dim, spacedim>>;                   \
  using typename Base_t::Op;                                                 \
                                                                             \
public:                                                                      \
  /**                                                                        \
   * Dimension in which this object operates.                                \
   */                                                                        \
  static const unsigned int dimension = dim;                                 \
                                                                             \
  /**                                                                        \
   * Dimension of the space in which this object operates.                   \
   */                                                                        \
  static const unsigned int space_dimension = spacedim;                      \
                                                                             \
  template <typename ScalarType>                                             \
  using value_type = typename Base_t::template value_type<ScalarType>;       \
                                                                             \
  template <typename ScalarType>                                             \
  using qp_value_type = typename Base_t::template qp_value_type<ScalarType>; \
                                                                             \
  template <typename ScalarType>                                             \
  using return_type = typename Base_t::template dof_value_type<ScalarType>;  \
                                                                             \
  template <typename ScalarType, std::size_t width>                          \
  using vectorized_return_type =                                             \
    typename Base_t::template vectorized_dof_value_type<ScalarType, width>;  \
                                                                             \
  /**                                                                        \
   * Return all shape function values all quadrature points.                 \
   *                                                                         \
   * The outer index is the shape function, and the inner index              \
   * is the quadrature point.                                                \
   *                                                                         \
   * @tparam ScalarType                                                      \
   * @param fe_values_dofs                                                   \
   * @param fe_values_op                                                     \
   * @return return_type<ScalarType>                                         \
   */                                                                        \
  template <typename ScalarType>                                             \
  return_type<ScalarType> operator()(                                        \
    const FEValuesBase<dim, spacedim> &fe_values_dofs,                       \
    const FEValuesBase<dim, spacedim> &fe_values_op) const                   \
  {                                                                          \
    return_type<ScalarType> out(fe_values_dofs.dofs_per_cell);               \
                                                                             \
    for (const auto dof_index : fe_values_dofs.dof_indices())                \
      {                                                                      \
        out[dof_index].reserve(fe_values_op.n_quadrature_points);            \
                                                                             \
        for (const auto q_point : fe_values_op.quadrature_point_indices())   \
          out[dof_index].emplace_back(this->template operator()<ScalarType>( \
            fe_values_op, dof_index, q_point));                              \
      }                                                                      \
                                                                             \
    return out;                                                              \
  }                                                                          \
                                                                             \
  template <typename ScalarType, std::size_t width>                          \
  vectorized_return_type<ScalarType, width> operator()(                      \
    const FEValuesBase<dim, spacedim> & fe_values_dofs,                      \
    const FEValuesBase<dim, spacedim> & fe_values_op,                        \
    const types::vectorized_qp_range_t &q_point_range) const                 \
  {                                                                          \
    vectorized_return_type<ScalarType, width> out(                           \
      fe_values_dofs.dofs_per_cell);                                         \
                                                                             \
    Assert(q_point_range.size() <= width,                                    \
           ExcIndexRange(q_point_range.size(), 0, width));                   \
                                                                             \
    for (const auto dof_index : fe_values_dofs.dof_indices())                \
      {                                                                      \
        DEAL_II_OPENMP_SIMD_PRAGMA                                           \
        for (unsigned int i = 0; i < q_point_range.size(); ++i)              \
          numbers::set_vectorized_values(                                    \
            out[dof_index],                                                  \
            i,                                                               \
            this->template operator()<ScalarType>(fe_values_op,              \
                                                  dof_index,                 \
                                                  q_point_range[i]));        \
      }                                                                      \
                                                                             \
    return out;                                                              \
  }                                                                          \
                                                                             \
protected:                                                                   \
  /**                                                                        \
   * Only want this to be a base class providing common implementation       \
   * for test functions / trial solutions.                                   \
   */                                                                        \
  explicit SymbolicOp(const Op &operand)                                     \
    : Base_t(operand)                                                        \
  {}



    /**
     * Extract the shape function values from a finite element space.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <int dim, int spacedim>
    class SymbolicOp<Space<dim, spacedim>, SymbolicOpCodes::value>
      : public SymbolicOpValueBase<Space<dim, spacedim>>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SPACE_COMMON_IMPL(SymbolicOpValueBase,
                                                       dim,
                                                       spacedim)

    protected:
      // Return single entry
      template <typename ScalarType>
      const value_type<ScalarType> &
      operator()(const FEValuesBase<dim, spacedim> &fe_values,
                 const unsigned int                 dof_index,
                 const unsigned int                 q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values.shape_value(dof_index, q_point);
      }
    };



    /**
     * Extract the shape function gradients from a finite element space.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <int dim, int spacedim>
    class SymbolicOp<Space<dim, spacedim>, SymbolicOpCodes::gradient>
      : public SymbolicOpGradientBase<Space<dim, spacedim>>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SPACE_COMMON_IMPL(SymbolicOpGradientBase,
                                                       dim,
                                                       spacedim)

    protected:
      // Return single entry
      template <typename ScalarType>
      const value_type<ScalarType> &
      operator()(const FEValuesBase<dim, spacedim> &fe_values,
                 const unsigned int                 dof_index,
                 const unsigned int                 q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values.shape_grad(dof_index, q_point);
      }
    };



    /**
     * Extract the shape function Laplacians from a finite element space.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <int dim, int spacedim>
    class SymbolicOp<Space<dim, spacedim>, SymbolicOpCodes::laplacian>
      : public SymbolicOpLaplacianBase<Space<dim, spacedim>>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SPACE_COMMON_IMPL(SymbolicOpLaplacianBase,
                                                       dim,
                                                       spacedim)

    protected:
      // Return single entry
      template <typename ScalarType>
      value_type<ScalarType>
      operator()(const FEValuesBase<dim, spacedim> &fe_values,
                 const unsigned int                 dof_index,
                 const unsigned int                 q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return trace(fe_values.shape_hessian(dof_index, q_point));
      }
    };



    /**
     * Extract the shape function Hessians from a finite element space.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <int dim, int spacedim>
    class SymbolicOp<Space<dim, spacedim>, SymbolicOpCodes::hessian>
      : public SymbolicOpHessianBase<Space<dim, spacedim>>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SPACE_COMMON_IMPL(SymbolicOpHessianBase,
                                                       dim,
                                                       spacedim)

    protected:
      // Return single entry
      template <typename ScalarType>
      const value_type<ScalarType> &
      operator()(const FEValuesBase<dim, spacedim> &fe_values,
                 const unsigned int                 dof_index,
                 const unsigned int                 q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values.shape_hessian(dof_index, q_point);
      }
    };



    /**
     * Extract the shape function third derivatives from a finite element
     * space.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <int dim, int spacedim>
    class SymbolicOp<Space<dim, spacedim>, SymbolicOpCodes::third_derivative>
      : public SymbolicOpThirdDerivativeBase<Space<dim, spacedim>>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SPACE_COMMON_IMPL(
        SymbolicOpThirdDerivativeBase,
        dim,
        spacedim)

    protected:
      // Return single entry
      template <typename ScalarType>
      const value_type<ScalarType> &
      operator()(const FEValuesBase<dim, spacedim> &fe_values,
                 const unsigned int                 dof_index,
                 const unsigned int                 q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values.shape_3rd_derivative(dof_index, q_point);
      }
    };


    // All test functions have the same operations as the FE space itself
    template <int dim, int spacedim, enum SymbolicOpCodes OpCode>
    class SymbolicOp<TestFunction<dim, spacedim>, OpCode>
      : public SymbolicOp<Space<dim, spacedim>, OpCode> {
        using Op     = TestFunction<dim, spacedim>;
        using Base_t = SymbolicOp<Space<dim, spacedim>, OpCode>;
        public:

          explicit SymbolicOp(const Op &operand): Base_t(operand){}
      };


    // All trial solution have the same operations as the FE space itself
    template <int dim, int spacedim, enum SymbolicOpCodes OpCode>
    class SymbolicOp<TrialSolution<dim, spacedim>, OpCode>
      : public SymbolicOp<Space<dim, spacedim>, OpCode> {
        using Op     = TrialSolution<dim, spacedim>;
        using Base_t = SymbolicOp<Space<dim, spacedim>, OpCode>;
        public:

          explicit SymbolicOp(const Op &operand): Base_t(operand){}
      };



#undef DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SPACE_COMMON_IMPL



    /* -- Finite element spaces: Test functions and trial solutions (interface)
     * -- */

/**
 * A macro to implement the common parts of a symbolic op class
 * for test functions and trial solution spaces.
 * It is expected that the unary op derives from a
 * SymbolicOp[TYPE]Base<Space<dim, spacedim>> .
 *
 * @note It is intended that this should used immediately after class
 * definition is opened.
 */
#define DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SPACE_INTERFACE_COMMON_IMPL(          \
  SymbolicOpBaseType, dim, spacedim)                                         \
private:                                                                     \
  using Base_t = SymbolicOpBaseType<Space<dim, spacedim>>;                   \
  using typename Base_t::Op;                                                 \
                                                                             \
public:                                                                      \
  /**                                                                        \
   * Dimension in which this object operates.                                \
   */                                                                        \
  static const unsigned int dimension = dim;                                 \
                                                                             \
  /**                                                                        \
   * Dimension of the space in which this object operates.                   \
   */                                                                        \
  static const unsigned int space_dimension = spacedim;                      \
                                                                             \
  template <typename ScalarType>                                             \
  using value_type = typename Base_t::template value_type<ScalarType>;       \
                                                                             \
  template <typename ScalarType>                                             \
  using qp_value_type = typename Base_t::template qp_value_type<ScalarType>; \
                                                                             \
  template <typename ScalarType>                                             \
  using return_type = typename Base_t::template dof_value_type<ScalarType>;  \
                                                                             \
  /**                                                                        \
   * Return all shape function values all quadrature points.                 \
   *                                                                         \
   * The outer index is the shape function, and the inner index              \
   * is the quadrature point.                                                \
   *                                                                         \
   * @tparam ScalarType                                                      \
   * @param fe_values_dofs                                                   \
   * @param fe_values_op                                                     \
   * @return return_type<ScalarType>                                         \
   */                                                                        \
  template <typename ScalarType>                                             \
  return_type<ScalarType> operator()(                                        \
    const FEInterfaceValues<dim, spacedim> &fe_interface_values) const       \
  {                                                                          \
    return_type<ScalarType> out(                                             \
      fe_interface_values.n_current_interface_dofs());                       \
                                                                             \
    for (const auto interface_dof_index :                                    \
         fe_interface_values.get_interface_dof_indices())                    \
      {                                                                      \
        out[interface_dof_index].reserve(                                    \
          fe_interface_values.n_quadrature_points);                          \
                                                                             \
        for (const auto q_point :                                            \
             fe_interface_values.quadrature_point_indices())                 \
          out[interface_dof_index].emplace_back(                             \
            this->template operator()<ScalarType>(fe_interface_values,       \
                                                  interface_dof_index,       \
                                                  q_point));                 \
      }                                                                      \
                                                                             \
    return out;                                                              \
  }                                                                          \
                                                                             \
protected:                                                                   \
  /**                                                                        \
   * Only want this to be a base class providing common implementation       \
   * for test functions / trial solutions.                                   \
   */                                                                        \
  explicit SymbolicOp(const Op &operand)                                     \
    : Base_t(operand)                                                        \
  {}



    /**
     * Extract the jump in shape function values from a finite element space.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <int dim, int spacedim>
    class SymbolicOp<Space<dim, spacedim>, SymbolicOpCodes::jump_in_values>
      : public SymbolicOpJumpValueBase<Space<dim, spacedim>>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SPACE_INTERFACE_COMMON_IMPL(
        SymbolicOpJumpValueBase,
        dim,
        spacedim)

    protected:
      // Return single entry
      template <typename ScalarType>
      value_type<ScalarType>
      operator()(const FEInterfaceValues<dim, spacedim> &fe_interface_values,
                 const unsigned int                      interface_dof_index,
                 const unsigned int                      q_point) const
      {
        Assert(interface_dof_index <
                 fe_interface_values.n_current_interface_dofs(),
               ExcIndexRange(interface_dof_index,
                             0,
                             fe_interface_values.n_current_interface_dofs()));
        Assert(q_point < fe_interface_values.n_quadrature_points,
               ExcIndexRange(q_point,
                             0,
                             fe_interface_values.n_quadrature_points));

        return fe_interface_values.jump_shape_value(interface_dof_index,
                                                    q_point);
      }
    };



    /**
     * Extract the jump in shape function gradients from a finite element space.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <int dim, int spacedim>
    class SymbolicOp<Space<dim, spacedim>, SymbolicOpCodes::jump_in_gradients>
      : public SymbolicOpJumpGradientBase<Space<dim, spacedim>>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SPACE_INTERFACE_COMMON_IMPL(
        SymbolicOpJumpGradientBase,
        dim,
        spacedim)

    protected:
      // Return single entry
      template <typename ScalarType>
      value_type<ScalarType>
      operator()(const FEInterfaceValues<dim, spacedim> &fe_interface_values,
                 const unsigned int                      interface_dof_index,
                 const unsigned int                      q_point) const
      {
        Assert(interface_dof_index <
                 fe_interface_values.n_current_interface_dofs(),
               ExcIndexRange(interface_dof_index,
                             0,
                             fe_interface_values.n_current_interface_dofs()));
        Assert(q_point < fe_interface_values.n_quadrature_points,
               ExcIndexRange(q_point,
                             0,
                             fe_interface_values.n_quadrature_points));

        return fe_interface_values.jump_shape_gradient(interface_dof_index,
                                                       q_point);
      }
    };



    /**
     * Extract the jump in shape function Hessians from a finite element space.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <int dim, int spacedim>
    class SymbolicOp<Space<dim, spacedim>, SymbolicOpCodes::jump_in_hessians>
      : public SymbolicOpJumpHessianBase<Space<dim, spacedim>>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SPACE_INTERFACE_COMMON_IMPL(
        SymbolicOpJumpHessianBase,
        dim,
        spacedim)

    protected:
      // Return single entry
      template <typename ScalarType>
      value_type<ScalarType>
      operator()(const FEInterfaceValues<dim, spacedim> &fe_interface_values,
                 const unsigned int                      interface_dof_index,
                 const unsigned int                      q_point) const
      {
        Assert(interface_dof_index <
                 fe_interface_values.n_current_interface_dofs(),
               ExcIndexRange(interface_dof_index,
                             0,
                             fe_interface_values.n_current_interface_dofs()));
        Assert(q_point < fe_interface_values.n_quadrature_points,
               ExcIndexRange(q_point,
                             0,
                             fe_interface_values.n_quadrature_points));

        return fe_interface_values.jump_shape_hessian(interface_dof_index,
                                                      q_point);
      }
    };



    /**
     * Extract the jump in shape function third derivatives from a finite
     * element space.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <int dim, int spacedim>
    class SymbolicOp<Space<dim, spacedim>,
                     SymbolicOpCodes::jump_in_third_derivatives>
      : public SymbolicOpJumpThirdDerivativeBase<Space<dim, spacedim>>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SPACE_INTERFACE_COMMON_IMPL(
        SymbolicOpJumpThirdDerivativeBase,
        dim,
        spacedim)

    protected:
      // Return single entry
      template <typename ScalarType>
      value_type<ScalarType>
      operator()(const FEInterfaceValues<dim, spacedim> &fe_interface_values,
                 const unsigned int                      interface_dof_index,
                 const unsigned int                      q_point) const
      {
        Assert(interface_dof_index <
                 fe_interface_values.n_current_interface_dofs(),
               ExcIndexRange(interface_dof_index,
                             0,
                             fe_interface_values.n_current_interface_dofs()));
        Assert(q_point < fe_interface_values.n_quadrature_points,
               ExcIndexRange(q_point,
                             0,
                             fe_interface_values.n_quadrature_points));

        return fe_interface_values.jump_shape_3rd_derivative(
          interface_dof_index, q_point);
      }
    };



    /**
     * Extract the average of shape function values from a finite element space.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <int dim, int spacedim>
    class SymbolicOp<Space<dim, spacedim>, SymbolicOpCodes::average_of_values>
      : public SymbolicOpAverageValueBase<Space<dim, spacedim>>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SPACE_INTERFACE_COMMON_IMPL(
        SymbolicOpAverageValueBase,
        dim,
        spacedim)

    protected:
      // Return single entry
      template <typename ScalarType>
      value_type<ScalarType>
      operator()(const FEInterfaceValues<dim, spacedim> &fe_interface_values,
                 const unsigned int                      interface_dof_index,
                 const unsigned int                      q_point) const
      {
        Assert(interface_dof_index <
                 fe_interface_values.n_current_interface_dofs(),
               ExcIndexRange(interface_dof_index,
                             0,
                             fe_interface_values.n_current_interface_dofs()));
        Assert(q_point < fe_interface_values.n_quadrature_points,
               ExcIndexRange(q_point,
                             0,
                             fe_interface_values.n_quadrature_points));

        return fe_interface_values.average_shape_value(interface_dof_index,
                                                       q_point);
      }
    };



    /**
     * Extract the average of shape function gradients from a finite element
     * space.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <int dim, int spacedim>
    class SymbolicOp<Space<dim, spacedim>,
                     SymbolicOpCodes::average_of_gradients>
      : public SymbolicOpAverageGradientBase<Space<dim, spacedim>>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SPACE_INTERFACE_COMMON_IMPL(
        SymbolicOpAverageGradientBase,
        dim,
        spacedim)

    protected:
      // Return single entry
      template <typename ScalarType>
      value_type<ScalarType>
      operator()(const FEInterfaceValues<dim, spacedim> &fe_interface_values,
                 const unsigned int                      interface_dof_index,
                 const unsigned int                      q_point) const
      {
        Assert(interface_dof_index <
                 fe_interface_values.n_current_interface_dofs(),
               ExcIndexRange(interface_dof_index,
                             0,
                             fe_interface_values.n_current_interface_dofs()));
        Assert(q_point < fe_interface_values.n_quadrature_points,
               ExcIndexRange(q_point,
                             0,
                             fe_interface_values.n_quadrature_points));

        return fe_interface_values.average_shape_gradient(interface_dof_index,
                                                          q_point);
      }
    };



    /**
     * Extract the average of shape function gradients from a finite element
     * space.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <int dim, int spacedim>
    class SymbolicOp<Space<dim, spacedim>, SymbolicOpCodes::average_of_hessians>
      : public SymbolicOpAverageHessianBase<Space<dim, spacedim>>
    {
      DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SPACE_INTERFACE_COMMON_IMPL(
        SymbolicOpAverageHessianBase,
        dim,
        spacedim)

    protected:
      // Return single entry
      template <typename ScalarType>
      value_type<ScalarType>
      operator()(const FEInterfaceValues<dim, spacedim> &fe_interface_values,
                 const unsigned int                      interface_dof_index,
                 const unsigned int                      q_point) const
      {
        Assert(interface_dof_index <
                 fe_interface_values.n_current_interface_dofs(),
               ExcIndexRange(interface_dof_index,
                             0,
                             fe_interface_values.n_current_interface_dofs()));
        Assert(q_point < fe_interface_values.n_quadrature_points,
               ExcIndexRange(q_point,
                             0,
                             fe_interface_values.n_quadrature_points));

        return fe_interface_values.average_shape_hessian(interface_dof_index,
                                                         q_point);
      }
    };



    // /**
    //  * Extract the average of shape function third derivatives from a finite
    //  element space.
    //  *
    //  * @tparam dim
    //  * @tparam spacedim
    //  */
    // template <int dim, int spacedim>
    // class SymbolicOp<Space<dim, spacedim>,
    // SymbolicOpCodes::average_of_third_derivatives>
    //   : public SymbolicOpAverageThirdDerivativeBase<Space<dim, spacedim>>
    // {
    //   DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SPACE_INTERFACE_COMMON_IMPL(
    //     SymbolicOpAverageThirdDerivativeBase,
    //     dim,
    //     spacedim)

    // protected:
    //   // Return single entry
    //   template <typename ScalarType>
    //   value_type<ScalarType>
    //   operator()(const FEInterfaceValues<dim, spacedim> &fe_interface_values,
    //              const unsigned int                      interface_dof_index,
    //              const unsigned int                      q_point) const
    //   {
    //     Assert(interface_dof_index <
    //              fe_interface_values.n_current_interface_dofs(),
    //            ExcIndexRange(interface_dof_index,
    //                          0,
    //                          fe_interface_values.n_current_interface_dofs()));
    //     Assert(q_point < fe_interface_values.n_quadrature_points,
    //            ExcIndexRange(q_point,
    //                          0,
    //                          fe_interface_values.n_quadrature_points));

    //     return
    //     fe_interface_values.average_shape_third_derivative(interface_dof_index,
    //                                                    q_point);
    //   }
    // };



#undef DEAL_II_SYMBOLIC_OP_TEST_TRIAL_SPACE_INTERFACE_COMMON_IMPL



    /* ------------ Finite element spaces: Solution fields ------------ */

/**
 * A macro to implement the common parts of a symbolic op class
 * for field solutions.
 * It is expected that the unary op derives from a
 * SymbolicOp[TYPE]Base<FieldSolution<dim, spacedim>, solution_index> .
 *
 * @note It is intended that this should used immediately after class
 * definition is opened.
 */
#define DEAL_II_SYMBOLIC_OP_FIELD_SOLUTION_COMMON_IMPL(SymbolicOpBaseType, \
                                                       dim,                \
                                                       spacedim,           \
                                                       solution_index)     \
private:                                                                   \
  using Base_t =                                                           \
    SymbolicOpBaseType<FieldSolution<dim, spacedim>, solution_index>;      \
  using typename Base_t::Op;                                               \
                                                                           \
public:                                                                    \
  /**                                                                      \
   * Dimension in which this object operates.                              \
   */                                                                      \
  static const unsigned int dimension = dim;                               \
                                                                           \
  /**                                                                      \
   * Dimension of the space in which this object operates.                 \
   */                                                                      \
  static const unsigned int space_dimension = spacedim;                    \
                                                                           \
  template <typename ScalarType>                                           \
  using value_type = typename Base_t::template value_type<ScalarType>;     \
                                                                           \
  template <typename ScalarType>                                           \
  using return_type = typename Base_t::template qp_value_type<ScalarType>; \
                                                                           \
  explicit SymbolicOp(const Op &operand)                                   \
    : Base_t(operand)                                                      \
  {}


    /**
     * Extract the solution values from the discretized solution field.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <types::solution_index solution_index, int dim, int spacedim>
    class SymbolicOp<FieldSolution<dim, spacedim>,
                     SymbolicOpCodes::value,
                     void,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpValueBase<FieldSolution<dim, spacedim>, solution_index>
    {
      DEAL_II_SYMBOLIC_OP_FIELD_SOLUTION_COMMON_IMPL(SymbolicOpValueBase,
                                                     dim,
                                                     spacedim,
                                                     solution_index)

    public:
      // Return solution values at all quadrature points
      template <typename ScalarType>
      return_type<ScalarType>
      operator()(MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                 const std::vector<std::string> &        solution_names) const
      {
        (void)scratch_data;
        (void)solution_names;

        AssertThrow(
          false,
          ExcMessage(
            "Solution field value extraction for has not been implemented for the global solution space. "
            "Use a weak form subspace extractor to isolate a component of the field solution before trying "
            "to retrieve its value."));

        return return_type<ScalarType>();
      }
    };



    /**
     * Extract the solution gradients from the discretized solution field.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <types::solution_index solution_index, int dim, int spacedim>
    class SymbolicOp<FieldSolution<dim, spacedim>,
                     SymbolicOpCodes::gradient,
                     void,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpGradientBase<FieldSolution<dim, spacedim>,
                                      solution_index>
    {
      DEAL_II_SYMBOLIC_OP_FIELD_SOLUTION_COMMON_IMPL(SymbolicOpGradientBase,
                                                     dim,
                                                     spacedim,
                                                     solution_index)

    public:
      // Return solution gradients at all quadrature points
      template <typename ScalarType>
      return_type<ScalarType>
      operator()(MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                 const std::vector<std::string> &        solution_names) const
      {
        (void)scratch_data;
        (void)solution_names;

        AssertThrow(
          false,
          ExcMessage(
            "Solution field gradient extraction for has not been implemented for the global solution space. "
            "Use a weak form subspace extractor to isolate a component of the field solution before trying "
            "to retrieve its gradient."));

        return return_type<ScalarType>();
      }
    };



    /**
     * Extract the solution Laplacians from the discretized solution field.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <types::solution_index solution_index, int dim, int spacedim>
    class SymbolicOp<FieldSolution<dim, spacedim>,
                     SymbolicOpCodes::laplacian,
                     void,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpLaplacianBase<FieldSolution<dim, spacedim>,
                                       solution_index>
    {
      DEAL_II_SYMBOLIC_OP_FIELD_SOLUTION_COMMON_IMPL(SymbolicOpLaplacianBase,
                                                     dim,
                                                     spacedim,
                                                     solution_index)

    public:
      // Return solution Laplacians at all quadrature points
      template <typename ScalarType>
      return_type<ScalarType>
      operator()(MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                 const std::vector<std::string> &        solution_names) const
      {
        (void)scratch_data;
        (void)solution_names;

        AssertThrow(
          false,
          ExcMessage(
            "Solution field Laplacian extraction for has not been implemented for the global solution space. "
            "Use a weak form subspace extractor to isolate a component of the field solution before trying "
            "to retrieve its Laplacian."));

        return return_type<ScalarType>();
      }
    };



    /**
     * Extract the solution Hessians from the discretized solution field.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <types::solution_index solution_index, int dim, int spacedim>
    class SymbolicOp<FieldSolution<dim, spacedim>,
                     SymbolicOpCodes::hessian,
                     void,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpHessianBase<FieldSolution<dim, spacedim>,
                                     solution_index>
    {
      DEAL_II_SYMBOLIC_OP_FIELD_SOLUTION_COMMON_IMPL(SymbolicOpHessianBase,
                                                     dim,
                                                     spacedim,
                                                     solution_index)

    public:
      // Return solution Hessians at all quadrature points
      template <typename ScalarType>
      return_type<ScalarType>
      operator()(MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                 const std::vector<std::string> &        solution_names) const
      {
        (void)scratch_data;
        (void)solution_names;

        AssertThrow(
          false,
          ExcMessage(
            "Solution field Hessian extraction for has not been implemented for the global solution space. "
            "Use a weak form subspace extractor to isolate a component of the field solution before trying "
            "to retrieve its Hessian."));

        return return_type<ScalarType>();
      }
    };



    /**
     * Extract the solution third derivatives from the discretized solution
     * field.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <types::solution_index solution_index, int dim, int spacedim>
    class SymbolicOp<FieldSolution<dim, spacedim>,
                     SymbolicOpCodes::third_derivative,
                     void,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpThirdDerivativeBase<FieldSolution<dim, spacedim>,
                                             solution_index>
    {
      DEAL_II_SYMBOLIC_OP_FIELD_SOLUTION_COMMON_IMPL(
        SymbolicOpThirdDerivativeBase,
        dim,
        spacedim,
        solution_index)

    public:
      // Return solution third derivatives at all quadrature points
      template <typename ScalarType>
      return_type<ScalarType>
      operator()(MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                 const std::vector<std::string> &        solution_names) const
      {
        (void)scratch_data;
        (void)solution_names;

        AssertThrow(
          false,
          ExcMessage(
            "Solution field third derivative extraction for has not been implemented for the global solution space. "
            "Use a weak form subspace extractor to isolate a component of the field solution before trying "
            "to retrieve its third derivative."));

        return return_type<ScalarType>();
      }
    };

#undef DEAL_II_SYMBOLIC_OP_FIELD_SOLUTION_COMMON_IMPL

  } // namespace Operators
} // namespace WeakForms



/* ======================== Convenience functions ======================== */



namespace WeakForms
{
  /* --------------- Finite element spaces: Test functions --------------- */


  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<WeakForms::TestFunction<dim, spacedim>,
                                   WeakForms::Operators::SymbolicOpCodes::value>
  value(const WeakForms::TestFunction<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TestFunction<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::value>;

    return OpType(operand);
  }



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TestFunction<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::gradient>
  gradient(const WeakForms::TestFunction<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TestFunction<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::gradient>;

    return OpType(operand);
  }



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TestFunction<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::laplacian>
  laplacian(const WeakForms::TestFunction<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TestFunction<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::laplacian>;

    return OpType(operand);
  }



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TestFunction<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::hessian>
  hessian(const WeakForms::TestFunction<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TestFunction<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::hessian>;

    return OpType(operand);
  }



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TestFunction<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::third_derivative>
  third_derivative(const WeakForms::TestFunction<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TestFunction<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::third_derivative>;

    return OpType(operand);
  }



  /* ---------- Finite element spaces: Test functions (interface) ---------- */


  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TestFunction<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::jump_in_values>
  jump_in_values(const WeakForms::TestFunction<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TestFunction<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::jump_in_values>;

    return OpType(operand);
  }


  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TestFunction<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::jump_in_gradients>
  jump_in_gradients(const WeakForms::TestFunction<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TestFunction<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::jump_in_gradients>;

    return OpType(operand);
  }


  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TestFunction<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::jump_in_hessians>
  jump_in_hessians(const WeakForms::TestFunction<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TestFunction<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::jump_in_hessians>;

    return OpType(operand);
  }


  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TestFunction<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::jump_in_third_derivatives>
  jump_in_third_derivatives(
    const WeakForms::TestFunction<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TestFunction<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::jump_in_third_derivatives>;

    return OpType(operand);
  }


  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TestFunction<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::average_of_values>
  average_of_values(const WeakForms::TestFunction<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TestFunction<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::average_of_values>;

    return OpType(operand);
  }


  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TestFunction<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::average_of_gradients>
  average_of_gradients(const WeakForms::TestFunction<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TestFunction<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::average_of_gradients>;

    return OpType(operand);
  }


  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TestFunction<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::average_of_hessians>
  average_of_hessians(const WeakForms::TestFunction<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TestFunction<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::average_of_hessians>;

    return OpType(operand);
  }



  /* --------------- Finite element spaces: Trial solutions --------------- */



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<WeakForms::TrialSolution<dim, spacedim>,
                                   WeakForms::Operators::SymbolicOpCodes::value>
  value(const WeakForms::TrialSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TrialSolution<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::value>;

    return OpType(operand);
  }



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TrialSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::gradient>
  gradient(const WeakForms::TrialSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TrialSolution<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::gradient>;

    return OpType(operand);
  }



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TrialSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::laplacian>
  laplacian(const WeakForms::TrialSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TrialSolution<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::laplacian>;

    return OpType(operand);
  }



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TrialSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::hessian>
  hessian(const WeakForms::TrialSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TrialSolution<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::hessian>;

    return OpType(operand);
  }



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TrialSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::third_derivative>
  third_derivative(const WeakForms::TrialSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TrialSolution<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::third_derivative>;

    return OpType(operand);
  }



  /* --------- Finite element spaces: Trial solutions (interface) --------- */



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TrialSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::jump_in_values>
  jump_in_values(const WeakForms::TrialSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TrialSolution<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::jump_in_values>;

    return OpType(operand);
  }



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TrialSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::jump_in_gradients>
  jump_in_gradients(const WeakForms::TrialSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TrialSolution<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::jump_in_gradients>;

    return OpType(operand);
  }



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TrialSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::jump_in_hessians>
  jump_in_hessians(const WeakForms::TrialSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TrialSolution<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::jump_in_hessians>;

    return OpType(operand);
  }



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TrialSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::jump_in_third_derivatives>
  jump_in_third_derivatives(
    const WeakForms::TrialSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TrialSolution<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::jump_in_third_derivatives>;

    return OpType(operand);
  }



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TrialSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::average_of_values>
  average_of_values(const WeakForms::TrialSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TrialSolution<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::average_of_values>;

    return OpType(operand);
  }



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TrialSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::average_of_gradients>
  average_of_gradients(const WeakForms::TrialSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TrialSolution<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::average_of_gradients>;

    return OpType(operand);
  }



  template <int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TrialSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::average_of_hessians>
  average_of_hessians(const WeakForms::TrialSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TrialSolution<dim, spacedim>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::average_of_hessians>;

    return OpType(operand);
  }



  /* --------------- Finite element spaces: Solution fields --------------- */



  template <types::solution_index solution_index, int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::FieldSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::value,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  value(const WeakForms::FieldSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op = FieldSolution<dim, spacedim>;
    using OpType =
      SymbolicOp<Op,
                 SymbolicOpCodes::value,
                 void,
                 WeakForms::internal::SolutionIndex<solution_index>>;

    return OpType(operand);
  }



  template <types::solution_index solution_index, int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::FieldSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::gradient,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  gradient(const WeakForms::FieldSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op = FieldSolution<dim, spacedim>;
    using OpType =
      SymbolicOp<Op,
                 SymbolicOpCodes::gradient,
                 void,
                 WeakForms::internal::SolutionIndex<solution_index>>;

    return OpType(operand);
  }



  template <types::solution_index solution_index, int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::FieldSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::laplacian,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  laplacian(const WeakForms::FieldSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op = FieldSolution<dim, spacedim>;
    using OpType =
      SymbolicOp<Op,
                 SymbolicOpCodes::laplacian,
                 void,
                 WeakForms::internal::SolutionIndex<solution_index>>;

    return OpType(operand);
  }



  template <types::solution_index solution_index, int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::FieldSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::hessian,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  hessian(const WeakForms::FieldSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op = FieldSolution<dim, spacedim>;
    using OpType =
      SymbolicOp<Op,
                 SymbolicOpCodes::hessian,
                 void,
                 WeakForms::internal::SolutionIndex<solution_index>>;

    return OpType(operand);
  }



  template <types::solution_index solution_index, int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::FieldSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::third_derivative,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  third_derivative(const WeakForms::FieldSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op = FieldSolution<dim, spacedim>;
    using OpType =
      SymbolicOp<Op,
                 SymbolicOpCodes::third_derivative,
                 void,
                 WeakForms::internal::SolutionIndex<solution_index>>;

    return OpType(operand);
  }

} // namespace WeakForms



/* ==================== Specialization of type traits ==================== */



#ifndef DOXYGEN


namespace WeakForms
{
  // Decorator classes

  template <int dim, int spacedim>
  struct is_test_function<TestFunction<dim, spacedim>> : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_trial_solution<TrialSolution<dim, spacedim>> : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_field_solution<FieldSolution<dim, spacedim>> : std::true_type
  {};



  // Symbolic operations

  template <int dim, int spacedim, enum Operators::SymbolicOpCodes OpCode>
  struct is_test_function_op<
    Operators::SymbolicOp<TestFunction<dim, spacedim>, OpCode>> : std::true_type
  {};

  template <int dim, int spacedim, enum Operators::SymbolicOpCodes OpCode>
  struct is_trial_solution_op<
    Operators::SymbolicOp<TrialSolution<dim, spacedim>, OpCode>>
    : std::true_type
  {};

  template <std::size_t                     solution_index,
            int                             dim,
            int                             spacedim,
            enum Operators::SymbolicOpCodes OpCode>
  struct is_field_solution_op<
    Operators::SymbolicOp<FieldSolution<dim, spacedim>,
                          OpCode,
                          void,
                          WeakForms::internal::SolutionIndex<solution_index>>>
    : std::true_type
  {};



  // Interface operations

  template <typename Op>
  struct is_interface_op<
    Operators::SymbolicOp<Op, Operators::SymbolicOpCodes::jump_in_values>>
    : std::true_type
  {};

  template <typename Op>
  struct is_interface_op<
    Operators::SymbolicOp<Op, Operators::SymbolicOpCodes::jump_in_gradients>>
    : std::true_type
  {};

  template <typename Op>
  struct is_interface_op<
    Operators::SymbolicOp<Op, Operators::SymbolicOpCodes::jump_in_hessians>>
    : std::true_type
  {};

  template <typename Op>
  struct is_interface_op<Operators::SymbolicOp<
    Op,
    Operators::SymbolicOpCodes::jump_in_third_derivatives>> : std::true_type
  {};

  template <typename Op>
  struct is_interface_op<
    Operators::SymbolicOp<Op, Operators::SymbolicOpCodes::average_of_values>>
    : std::true_type
  {};

  template <typename Op>
  struct is_interface_op<
    Operators::SymbolicOp<Op, Operators::SymbolicOpCodes::average_of_gradients>>
    : std::true_type
  {};

  template <typename Op>
  struct is_interface_op<
    Operators::SymbolicOp<Op, Operators::SymbolicOpCodes::average_of_hessians>>
    : std::true_type
  {};

  // template <typename Op>
  // struct is_interface_op<
  //   Operators::SymbolicOp<Op,
  //   Operators::SymbolicOpCodes::average_of_third_derivatives>> :
  //   std::true_type
  // {};

} // namespace WeakForms


#endif // DOXYGEN


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_spaces_h
