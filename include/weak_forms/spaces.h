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

#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/meshworker/scratch_data.h>

#include <weak_forms/config.h>
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



  /* --------------- Finite element spaces: Solution fields --------------- */

  namespace internal
  {
    // Used to work around the restriction that template arguments
    // for template type parameter must be a type
    template <std::size_t solution_index_>
    struct SolutionIndex
    {
      static const std::size_t solution_index = solution_index_;
    };
  } // namespace internal



  template <std::size_t solution_index = 0, int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::FieldSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::value,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  value(const WeakForms::FieldSolution<dim, spacedim> &operand);



  template <std::size_t solution_index = 0, int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::FieldSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::gradient,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  gradient(const WeakForms::FieldSolution<dim, spacedim> &operand);



  template <std::size_t solution_index = 0, int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::FieldSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::laplacian,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  laplacian(const WeakForms::FieldSolution<dim, spacedim> &operand);



  template <std::size_t solution_index = 0, int dim, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::FieldSolution<dim, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::hessian,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  hessian(const WeakForms::FieldSolution<dim, spacedim> &operand);



  template <std::size_t solution_index = 0, int dim, int spacedim>
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
    using OutputType =
      typename FEValuesViewsType::template OutputType<ScalarType>;

    template <typename ScalarType>
    using value_type = typename OutputType<ScalarType>::value_type;

    template <typename ScalarType>
    using gradient_type = typename OutputType<ScalarType>::gradient_type;

    template <typename ScalarType>
    using hessian_type = typename OutputType<ScalarType>::hessian_type;

    template <typename ScalarType>
    using laplacian_type = typename OutputType<ScalarType>::laplacian_type;

    template <typename ScalarType>
    using third_derivative_type =
      typename OutputType<ScalarType>::third_derivative_type;

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

    template <std::size_t solution_index = 0>
    auto
    value() const
    {
      return WeakForms::value<solution_index>(*this);
    }

    template <std::size_t solution_index = 0>
    auto
    gradient() const
    {
      return WeakForms::gradient<solution_index>(*this);
    }

    template <std::size_t solution_index = 0>
    auto
    laplacian() const
    {
      return WeakForms::laplacian<solution_index>(*this);
    }

    template <std::size_t solution_index = 0>
    auto
    hessian() const
    {
      return WeakForms::hessian<solution_index>(*this);
    }

    template <std::size_t solution_index = 0>
    auto
    third_derivative() const
    {
      return WeakForms::third_derivative<solution_index>(*this);
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

  protected:
    // Subspace
    FieldSolution(const types::field_index field_index,
                  const std::string &      field_ascii,
                  const std::string &      field_latex)
      : Space<dim, spacedim>(field_index, field_ascii, field_latex)
    {}
  };



  // namespace Linear
  // {
  //   template <int dim, int spacedim = dim>
  //   using TestFunction = WeakForms::TestFunction<dim, spacedim>;

  //   template <int dim, int spacedim = dim>
  //   using TrialSolution = WeakForms::TrialSolution<dim, spacedim>;

  //   template <int dim, int spacedim = dim>
  //   using Solution = WeakForms::Solution<dim, spacedim>;
  // } // namespace Linear



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
    /* ---- Mix-in classes ---- */
    // TODO[JPP]: Use CRTP here?
    template <typename Op_, std::size_t solution_index_ = 0>
    class SymbolicOpValueBase
    {
    public:
      using Op = Op_;

      template <typename ScalarType>
      using value_type = typename Op::template value_type<ScalarType>;

      // Value at all quadrature points
      template <typename ScalarType>
      using qp_value_type = std::vector<value_type<ScalarType>>;

      // Value for all DoFs at all quadrature points
      template <typename ScalarType>
      using dof_value_type = std::vector<qp_value_type<ScalarType>>;

      // The index in the solution history that this field solution
      // corresponds to. The default value (0) indicates that it relates
      // to the current solution.
      static const std::size_t solution_index = solution_index_;

      static const int rank = Op::rank;

      static const enum SymbolicOpCodes op_code = SymbolicOpCodes::value;

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

    protected:
      // Allow access to get_operand()
      friend WeakForms::SelfLinearization::internal::ConvertTo;

      // Only want this to be a base class
      explicit SymbolicOpValueBase(const Op &operand)
        : operand(operand.clone())
      {}

      const Op &
      get_operand() const
      {
        Assert(operand, ExcNotInitialized());
        return *operand;
      }

    private:
      const std::shared_ptr<Op> operand;
    };


    template <typename Op_, std::size_t solution_index_ = 0>
    class SymbolicOpGradientBase
    {
    public:
      using Op = Op_;

      template <typename ScalarType>
      using value_type = typename Op::template gradient_type<ScalarType>;

      // Value at all quadrature points
      template <typename ScalarType>
      using qp_value_type = std::vector<value_type<ScalarType>>;

      // Value for all DoFs at all quadrature points
      template <typename ScalarType>
      using dof_value_type = std::vector<qp_value_type<ScalarType>>;

      // The index in the solution history that this field solution
      // corresponds to. The default value (0) indicates that it relates
      // to the current solution.
      static const std::size_t solution_index = solution_index_;

      static const int rank = value_type<double>::rank;

      static const enum SymbolicOpCodes op_code = SymbolicOpCodes::gradient;

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

    protected:
      // Allow access to get_operand()
      friend WeakForms::SelfLinearization::internal::ConvertTo;

      // Only want this to be a base class
      explicit SymbolicOpGradientBase(const Op &operand)
        : operand(operand.clone())
      {}

      const Op &
      get_operand() const
      {
        Assert(operand, ExcNotInitialized());
        return *operand;
      }

    private:
      const std::shared_ptr<Op> operand;
    };


    template <typename Op_, std::size_t solution_index_ = 0>
    class SymbolicOpSymmetricGradientBase
    {
    public:
      using Op = Op_;

      template <typename ScalarType>
      using value_type =
        typename Op::template symmetric_gradient_type<ScalarType>;

      // Value at all quadrature points
      template <typename ScalarType>
      using qp_value_type = std::vector<value_type<ScalarType>>;

      // Value for all DoFs at all quadrature points
      template <typename ScalarType>
      using dof_value_type = std::vector<qp_value_type<ScalarType>>;

      // The index in the solution history that this field solution
      // corresponds to. The default value (0) indicates that it relates
      // to the current solution.
      static const std::size_t solution_index = solution_index_;

      static const int rank = value_type<double>::rank;

      static const enum SymbolicOpCodes op_code =
        SymbolicOpCodes::symmetric_gradient;

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

    protected:
      // Allow access to get_operand()
      friend WeakForms::SelfLinearization::internal::ConvertTo;

      // Only want this to be a base class
      explicit SymbolicOpSymmetricGradientBase(const Op &operand)
        : operand(operand.clone())
      {}

      const Op &
      get_operand() const
      {
        Assert(operand, ExcNotInitialized());
        return *operand;
      }

    private:
      const std::shared_ptr<Op> operand;
    };


    template <typename Op_, std::size_t solution_index_ = 0>
    class SymbolicOpDivergenceBase
    {
    public:
      using Op = Op_;

      template <typename ScalarType>
      using value_type = typename Op::template divergence_type<ScalarType>;

      // Value at all quadrature points
      template <typename ScalarType>
      using qp_value_type = std::vector<value_type<ScalarType>>;

      // Value for all DoFs at all quadrature points
      template <typename ScalarType>
      using dof_value_type = std::vector<qp_value_type<ScalarType>>;

      // The index in the solution history that this field solution
      // corresponds to. The default value (0) indicates that it relates
      // to the current solution.
      static const std::size_t solution_index = solution_index_;

      // static const int rank = value_type<double>::rank;
      static const int rank =
        Op_::rank; // The value_type<> might be a scalar or tensor, so we
                   // can't fetch the rank from it.

      static const enum SymbolicOpCodes op_code = SymbolicOpCodes::divergence;

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

    protected:
      // Allow access to get_operand()
      friend WeakForms::SelfLinearization::internal::ConvertTo;

      // Only want this to be a base class
      explicit SymbolicOpDivergenceBase(const Op &operand)
        : operand(operand.clone())
      {}

      const Op &
      get_operand() const
      {
        Assert(operand, ExcNotInitialized());
        return *operand;
      }

    private:
      const std::shared_ptr<Op> operand;
    };


    template <typename Op_, std::size_t solution_index_ = 0>
    class SymbolicOpCurlBase
    {
    public:
      using Op = Op_;

      template <typename ScalarType>
      using value_type = typename Op::template curl_type<ScalarType>;

      // Value at all quadrature points
      template <typename ScalarType>
      using qp_value_type = std::vector<value_type<ScalarType>>;

      // Value for all DoFs at all quadrature points
      template <typename ScalarType>
      using dof_value_type = std::vector<qp_value_type<ScalarType>>;

      // The index in the solution history that this field solution
      // corresponds to. The default value (0) indicates that it relates
      // to the current solution.
      static const std::size_t solution_index = solution_index_;

      static const int rank = value_type<double>::rank;

      static const enum SymbolicOpCodes op_code = SymbolicOpCodes::curl;

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

    protected:
      // Allow access to get_operand()
      friend WeakForms::SelfLinearization::internal::ConvertTo;

      // Only want this to be a base class
      explicit SymbolicOpCurlBase(const Op &operand)
        : operand(operand.clone())
      {}

      const Op &
      get_operand() const
      {
        Assert(operand, ExcNotInitialized());
        return *operand;
      }

    private:
      const std::shared_ptr<Op> operand;
    };


    template <typename Op_, std::size_t solution_index_ = 0>
    class SymbolicOpLaplacianBase
    {
    public:
      using Op = Op_;

      template <typename ScalarType>
      using value_type = typename Op::template laplacian_type<ScalarType>;

      // Value at all quadrature points
      template <typename ScalarType>
      using qp_value_type = std::vector<value_type<ScalarType>>;

      // Value for all DoFs at all quadrature points
      template <typename ScalarType>
      using dof_value_type = std::vector<qp_value_type<ScalarType>>;

      // The index in the solution history that this field solution
      // corresponds to. The default value (0) indicates that it relates
      // to the current solution.
      static const std::size_t solution_index = solution_index_;

      // static const int rank = value_type<double>::rank;
      static const int rank =
        Op_::rank; // The value_type<> might be a scalar or tensor, so we
                   // can't fetch the rank from it.

      static const enum SymbolicOpCodes op_code = SymbolicOpCodes::laplacian;

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

    protected:
      // Allow access to get_operand()
      friend WeakForms::SelfLinearization::internal::ConvertTo;

      // Only want this to be a base class
      explicit SymbolicOpLaplacianBase(const Op &operand)
        : operand(operand.clone())
      {}

      const Op &
      get_operand() const
      {
        Assert(operand, ExcNotInitialized());
        return *operand;
      }

    private:
      const std::shared_ptr<Op> operand;
    };


    template <typename Op_, std::size_t solution_index_ = 0>
    class SymbolicOpHessianBase
    {
    public:
      using Op = Op_;

      template <typename ScalarType>
      using value_type = typename Op::template hessian_type<ScalarType>;

      // Value at all quadrature points
      template <typename ScalarType>
      using qp_value_type = std::vector<value_type<ScalarType>>;

      // Value for all DoFs at all quadrature points
      template <typename ScalarType>
      using dof_value_type = std::vector<qp_value_type<ScalarType>>;

      // The index in the solution history that this field solution
      // corresponds to. The default value (0) indicates that it relates
      // to the current solution.
      static const std::size_t solution_index = solution_index_;

      static const int rank = value_type<double>::rank;
      // static const int rank = Op_::rank; // The value_type<> might be a
      // scalar or tensor, so we can't fetch the rank from it.

      static const enum SymbolicOpCodes op_code = SymbolicOpCodes::hessian;

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

    protected:
      // Allow access to get_operand()
      friend WeakForms::SelfLinearization::internal::ConvertTo;

      // Only want this to be a base class
      explicit SymbolicOpHessianBase(const Op &operand)
        : operand(operand.clone())
      {}

      const Op &
      get_operand() const
      {
        Assert(operand, ExcNotInitialized());
        return *operand;
      }

    private:
      const std::shared_ptr<Op> operand;
    };


    template <typename Op_, std::size_t solution_index_ = 0>
    class SymbolicOpThirdDerivativeBase
    {
    public:
      using Op = Op_;

      template <typename ScalarType>
      using value_type =
        typename Op::template third_derivative_type<ScalarType>;

      // Value at all quadrature points
      template <typename ScalarType>
      using qp_value_type = std::vector<value_type<ScalarType>>;

      // Value for all DoFs at all quadrature points
      template <typename ScalarType>
      using dof_value_type = std::vector<qp_value_type<ScalarType>>;

      // The index in the solution history that this field solution
      // corresponds to. The default value (0) indicates that it relates
      // to the current solution.
      static const std::size_t solution_index = solution_index_;

      static const int rank = value_type<double>::rank;
      // static const int rank = Op_::rank; // The value_type<> might be a
      // scalar or tensor, so we can't fetch the rank from it.

      static const enum SymbolicOpCodes op_code =
        SymbolicOpCodes::third_derivative;

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

    protected:
      // Allow access to get_operand()
      friend WeakForms::SelfLinearization::internal::ConvertTo;

      // Only want this to be a base class
      explicit SymbolicOpThirdDerivativeBase(const Op &operand)
        : operand(operand.clone())
      {}

      const Op &
      get_operand() const
      {
        Assert(operand, ExcNotInitialized());
        return *operand;
      }

    private:
      const std::shared_ptr<Op> operand;
    };


    /* ---- Finite element spaces: Test functions and trial solutions ---- */

    // // A little bit of CRTP, with a workaround to deal with templates
    // // in the derived class.
    // // See https://stackoverflow.com/a/45801893
    // template <typename Derived>
    // struct SymbolicOpShapeFunctionTraits;

    // template <typename Derived>
    // class SymbolicOpShapeFunctionBase
    // {
    //   protected:

    //   SymbolicOpShapeFunctionBase(const Derived &derived)
    //     : derived(derived)
    //   {}

    //   class AccessKey
    //   {
    //     friend class SymbolicOpShapeFunctionBase<Derived>;
    //     AccessKey(){}
    //   };

    //   public:
    //   /**
    //    * Dimension in which this object operates.
    //    */
    //   static const unsigned int dimension = Derived::dimension;

    //   /**
    //    * Dimension of the space in which this object operates.
    //    */
    //   static const unsigned int space_dimension = Derived::space_dimension;

    //   template <typename ScalarType>
    //   using value_type = typename Derived::template value_type<ScalarType>;

    //   template <typename ScalarType>
    //   using qp_value_type = typename Derived::template
    //   qp_value_type<ScalarType>;

    //   template <typename ScalarType>
    //   using return_type = typename Derived::template
    //   dof_value_type<ScalarType>;

    //   /**
    //    * Return all shape function values all quadrature points.
    //    *
    //    * The outer index is the shape function, and the inner index
    //    * is the quadrature point.
    //    *
    //    * @tparam ScalarType
    //    * @param fe_values_dofs
    //    * @param fe_values_op
    //    * @return return_type<ScalarType>
    //    */
    //   template <typename ScalarType>
    //   return_type<ScalarType>
    //   operator()(const FEValuesBase<dimension, space_dimension>
    //   &fe_values_dofs,
    //              const FEValuesBase<dimension, space_dimension>
    //              &fe_values_op) const
    //   {
    //     return_type<ScalarType> out(fe_values_dofs.dofs_per_cell);

    //     for (const auto dof_index : fe_values_dofs.dof_indices())
    //     {
    //       out[dof_index].reserve(fe_values_op.n_quadrature_points);

    //       for (const auto q_point : fe_values_op.quadrature_point_indices())
    //         out[dof_index].emplace_back(derived.template
    //         operator()<ScalarType>(fe_values_op,
    //                                                                             dof_index,
    //                                                                             q_point,
    //                                                                             AccessKey()));
    //     }

    //     return out;
    //   }

    // private:
    //   const Derived &derived;

    // TODO[JPP]: Put this in public section of derived classes
    // // Return single entry
    // template <typename ScalarType>
    // const value_type<ScalarType> &
    // operator()(const FEValuesBase<dim, spacedim> &fe_values,
    //            const unsigned int                 dof_index,
    //            const unsigned int                 q_point,
    //            const AccessKey) const
    // {
    //   Assert(dof_index < fe_values.dofs_per_cell,
    //          ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
    //   Assert(q_point < fe_values.n_quadrature_points,
    //          ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

    //   return fe_values.shape_value(dof_index, q_point);
    // }
    // };


    /**
     * Extract the shape function values from a finite element space.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <int dim, int spacedim>
    class SymbolicOp<Space<dim, spacedim>, SymbolicOpCodes::value>
      : public SymbolicOpValueBase<Space<dim, spacedim>>
    // , public SymbolicOpShapeFunctionBase<SymbolicOp<Space<dim, spacedim>,
    // SymbolicOpCodes::value>>
    {
      using Base_t = SymbolicOpValueBase<Space<dim, spacedim>>;
      using typename Base_t::Op;

      // using This = SymbolicOp<Space<dim, spacedim>, SymbolicOpCodes::value>;
      // using ShapeFunctionBase_t = SymbolicOpShapeFunctionBase<This>;
      // using AccessKey = typename ShapeFunctionBase_t::AccessKey;

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = dim;

      /**
       * Dimension of the space in which this object operates.
       */
      static const unsigned int space_dimension = spacedim;

      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;

      template <typename ScalarType>
      using qp_value_type = typename Base_t::template qp_value_type<ScalarType>;

      template <typename ScalarType>
      using return_type = typename Base_t::template dof_value_type<ScalarType>;

      // using ShapeFunctionBase_t::operator();

      /**
       * Return all shape function values all quadrature points.
       *
       * The outer index is the shape function, and the inner index
       * is the quadrature point.
       *
       * @tparam ScalarType
       * @param fe_values_dofs
       * @param fe_values_op
       * @return return_type<ScalarType>
       */
      template <typename ScalarType>
      return_type<ScalarType>
      operator()(const FEValuesBase<dim, spacedim> &fe_values_dofs,
                 const FEValuesBase<dim, spacedim> &fe_values_op) const
      {
        return_type<ScalarType> out(fe_values_dofs.dofs_per_cell);

        for (const auto dof_index : fe_values_dofs.dof_indices())
          {
            out[dof_index].reserve(fe_values_op.n_quadrature_points);

            for (const auto q_point : fe_values_op.quadrature_point_indices())
              out[dof_index].emplace_back(this->template operator()<ScalarType>(
                fe_values_op, dof_index, q_point));
          }

        return out;
      }

    protected:
      // Only want this to be a base class providing common implementation
      // for test functions / trial solutions.
      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      // , ShapeFunctionBase_t(*this)
      {}

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
    // , public SymbolicOpShapeFunctionBase<SymbolicOp<Space<dim, spacedim>,
    // SymbolicOpCodes::gradient>>
    {
      using Base_t = SymbolicOpGradientBase<Space<dim, spacedim>>;
      using typename Base_t::Op;

      // using This = SymbolicOp<Space<dim, spacedim>,
      // SymbolicOpCodes::gradient>; using ShapeFunctionBase_t =
      // SymbolicOpShapeFunctionBase<This>; using AccessKey = typename
      // ShapeFunctionBase_t::AccessKey;

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = dim;

      /**
       * Dimension of the space in which this object operates.
       */
      static const unsigned int space_dimension = spacedim;
      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;

      template <typename ScalarType>
      using qp_value_type = typename Base_t::template qp_value_type<ScalarType>;

      template <typename ScalarType>
      using return_type = typename Base_t::template dof_value_type<ScalarType>;

      // using ShapeFunctionBase_t::operator();

      /**
       * Return all shape function values all quadrature points.
       *
       * The outer index is the shape function, and the inner index
       * is the quadrature point.
       *
       * @tparam ScalarType
       * @param fe_values_dofs
       * @param fe_values_op
       * @return return_type<ScalarType>
       */
      template <typename ScalarType>
      return_type<ScalarType>
      operator()(const FEValuesBase<dim, spacedim> &fe_values_dofs,
                 const FEValuesBase<dim, spacedim> &fe_values_op) const
      {
        return_type<ScalarType> out(fe_values_dofs.dofs_per_cell);

        for (const auto dof_index : fe_values_dofs.dof_indices())
          {
            out[dof_index].reserve(fe_values_op.n_quadrature_points);

            for (const auto q_point : fe_values_op.quadrature_point_indices())
              out[dof_index].emplace_back(this->template operator()<ScalarType>(
                fe_values_op, dof_index, q_point));
          }

        return out;
      }

    protected:
      // Only want this to be a base class providing common implementation
      // for test functions / trial solutions.
      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      // , ShapeFunctionBase_t(*this)
      {}

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
    // , public SymbolicOpShapeFunctionBase<SymbolicOp<Space<dim, spacedim>,
    // SymbolicOpCodes::laplacian>>
    {
      using Base_t = SymbolicOpLaplacianBase<Space<dim, spacedim>>;
      using typename Base_t::Op;

      // using This = SymbolicOp<Space<dim, spacedim>,
      // SymbolicOpCodes::laplacian>; using ShapeFunctionBase_t =
      // SymbolicOpShapeFunctionBase<This>; using AccessKey = typename
      // ShapeFunctionBase_t::AccessKey;

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = dim;

      /**
       * Dimension of the space in which this object operates.
       */
      static const unsigned int space_dimension = spacedim;

      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;

      template <typename ScalarType>
      using qp_value_type = typename Base_t::template qp_value_type<ScalarType>;

      template <typename ScalarType>
      using return_type = typename Base_t::template dof_value_type<ScalarType>;

      // using ShapeFunctionBase_t::operator();

      /**
       * Return all shape function values all quadrature points.
       *
       * The outer index is the shape function, and the inner index
       * is the quadrature point.
       *
       * @tparam ScalarType
       * @param fe_values_dofs
       * @param fe_values_op
       * @return return_type<ScalarType>
       */
      template <typename ScalarType>
      return_type<ScalarType>
      operator()(const FEValuesBase<dim, spacedim> &fe_values_dofs,
                 const FEValuesBase<dim, spacedim> &fe_values_op) const
      {
        return_type<ScalarType> out(fe_values_dofs.dofs_per_cell);

        for (const auto dof_index : fe_values_dofs.dof_indices())
          {
            out[dof_index].reserve(fe_values_op.n_quadrature_points);

            for (const auto q_point : fe_values_op.quadrature_point_indices())
              out[dof_index].emplace_back(this->template operator()<ScalarType>(
                fe_values_op, dof_index, q_point));
          }

        return out;
      }

    protected:
      // Only want this to be a base class providing common implementation
      // for test functions / trial solutions.
      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      // , ShapeFunctionBase_t(*this)
      {}

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
    // , public SymbolicOpShapeFunctionBase<SymbolicOp<Space<dim, spacedim>,
    // SymbolicOpCodes::hessian>>
    {
      using Base_t = SymbolicOpHessianBase<Space<dim, spacedim>>;
      using typename Base_t::Op;

      // using This = SymbolicOp<Space<dim, spacedim>,
      // SymbolicOpCodes::hessian>; using ShapeFunctionBase_t =
      // SymbolicOpShapeFunctionBase<This>; using AccessKey = typename
      // ShapeFunctionBase_t::AccessKey;

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = dim;

      /**
       * Dimension of the space in which this object operates.
       */
      static const unsigned int space_dimension = spacedim;

      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;

      template <typename ScalarType>
      using qp_value_type = typename Base_t::template qp_value_type<ScalarType>;

      template <typename ScalarType>
      using return_type = typename Base_t::template dof_value_type<ScalarType>;

      // using ShapeFunctionBase_t::operator();

      /**
       * Return all shape function values all quadrature points.
       *
       * The outer index is the shape function, and the inner index
       * is the quadrature point.
       *
       * @tparam ScalarType
       * @param fe_values_dofs
       * @param fe_values_op
       * @return return_type<ScalarType>
       */
      template <typename ScalarType>
      return_type<ScalarType>
      operator()(const FEValuesBase<dim, spacedim> &fe_values_dofs,
                 const FEValuesBase<dim, spacedim> &fe_values_op) const
      {
        return_type<ScalarType> out(fe_values_dofs.dofs_per_cell);

        for (const auto dof_index : fe_values_dofs.dof_indices())
          {
            out[dof_index].reserve(fe_values_op.n_quadrature_points);

            for (const auto q_point : fe_values_op.quadrature_point_indices())
              out[dof_index].emplace_back(this->template operator()<ScalarType>(
                fe_values_op, dof_index, q_point));
          }

        return out;
      }

    protected:
      // Only want this to be a base class providing common implementation
      // for test functions / trial solutions.
      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      // , ShapeFunctionBase_t(*this)
      {}

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
    // , public SymbolicOpShapeFunctionBase<SymbolicOp<Space<dim, spacedim>,
    // SymbolicOpCodes::third_derivative>>
    {
      using Base_t = SymbolicOpThirdDerivativeBase<Space<dim, spacedim>>;
      using typename Base_t::Op;

      // using This = SymbolicOp<Space<dim, spacedim>,
      // SymbolicOpCodes::third_derivative>; using ShapeFunctionBase_t =
      // SymbolicOpShapeFunctionBase<This>; using AccessKey = typename
      // ShapeFunctionBase_t::AccessKey;

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = dim;

      /**
       * Dimension of the space in which this object operates.
       */
      static const unsigned int space_dimension = spacedim;

      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;

      template <typename ScalarType>
      using qp_value_type = typename Base_t::template qp_value_type<ScalarType>;

      template <typename ScalarType>
      using return_type = typename Base_t::template dof_value_type<ScalarType>;

      // using ShapeFunctionBase_t::operator();

      /**
       * Return all shape function values all quadrature points.
       *
       * The outer index is the shape function, and the inner index
       * is the quadrature point.
       *
       * @tparam ScalarType
       * @param fe_values_dofs
       * @param fe_values_op
       * @return return_type<ScalarType>
       */
      template <typename ScalarType>
      return_type<ScalarType>
      operator()(const FEValuesBase<dim, spacedim> &fe_values_dofs,
                 const FEValuesBase<dim, spacedim> &fe_values_op) const
      {
        return_type<ScalarType> out(fe_values_dofs.dofs_per_cell);

        for (const auto dof_index : fe_values_dofs.dof_indices())
          {
            out[dof_index].reserve(fe_values_op.n_quadrature_points);

            for (const auto q_point : fe_values_op.quadrature_point_indices())
              out[dof_index].emplace_back(this->template operator()<ScalarType>(
                fe_values_op, dof_index, q_point));
          }

        return out;
      }

    protected:
      // Only want this to be a base class providing common implementation
      // for test functions / trial solutions.
      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      // , ShapeFunctionBase_t(*this)
      {}

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



    /* ------------ Finite element spaces: Solution fields ------------ */


    /**
     * Extract the solution values from the discretized solution field.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <std::size_t solution_index, int dim, int spacedim>
    class SymbolicOp<FieldSolution<dim, spacedim>,
                     SymbolicOpCodes::value,
                     void,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpValueBase<FieldSolution<dim, spacedim>, solution_index>
    {
      using Base_t =
        SymbolicOpValueBase<FieldSolution<dim, spacedim>, solution_index>;
      using typename Base_t::Op;

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = dim;

      /**
       * Dimension of the space in which this object operates.
       */
      static const unsigned int space_dimension = spacedim;

      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;

      template <typename ScalarType>
      using return_type = typename Base_t::template qp_value_type<ScalarType>;

      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      {}

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

        // return_type<ScalarType> out(fe_values.n_quadrature_points);
        // // Need to implement a "get_function_values_from_local_dof_values()"
        // // function fe_values.get_function_values(solution_local_dof_values,
        // // out);
        // return out;
      }
    };



    /**
     * Extract the solution gradients from the discretized solution field.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <std::size_t solution_index, int dim, int spacedim>
    class SymbolicOp<FieldSolution<dim, spacedim>,
                     SymbolicOpCodes::gradient,
                     void,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpGradientBase<FieldSolution<dim, spacedim>,
                                      solution_index>
    {
      using Base_t =
        SymbolicOpGradientBase<FieldSolution<dim, spacedim>, solution_index>;
      using typename Base_t::Op;

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = dim;

      /**
       * Dimension of the space in which this object operates.
       */
      static const unsigned int space_dimension = spacedim;

      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;

      template <typename ScalarType>
      using return_type = typename Base_t::template qp_value_type<ScalarType>;

      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      {}

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

        return_type<ScalarType> out; //(fe_values.n_quadrature_points);
        // Need to implement a
        // "get_function_gradients_from_local_dof_values()" function
        // fe_values.get_function_gradients(solution_local_dof_values, out);
        return out;
      }
    };



    /**
     * Extract the solution Laplacians from the discretized solution field.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <std::size_t solution_index, int dim, int spacedim>
    class SymbolicOp<FieldSolution<dim, spacedim>,
                     SymbolicOpCodes::laplacian,
                     void,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpLaplacianBase<FieldSolution<dim, spacedim>,
                                       solution_index>
    {
      using Base_t =
        SymbolicOpLaplacianBase<FieldSolution<dim, spacedim>, solution_index>;
      using typename Base_t::Op;

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = dim;

      /**
       * Dimension of the space in which this object operates.
       */
      static const unsigned int space_dimension = spacedim;

      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;

      template <typename ScalarType>
      using return_type = typename Base_t::template qp_value_type<ScalarType>;

      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      {}

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

        // return_type<ScalarType> out(fe_values.n_quadrature_points);
        // // Need to implement a
        // // "get_function_laplacians_from_local_dof_values()" function
        // // fe_values.get_function_laplacians(solution_local_dof_values, out);
        // return out;
      }
    };



    /**
     * Extract the solution Hessians from the discretized solution field.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <std::size_t solution_index, int dim, int spacedim>
    class SymbolicOp<FieldSolution<dim, spacedim>,
                     SymbolicOpCodes::hessian,
                     void,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpHessianBase<FieldSolution<dim, spacedim>,
                                     solution_index>
    {
      using Base_t =
        SymbolicOpHessianBase<FieldSolution<dim, spacedim>, solution_index>;
      using typename Base_t::Op;

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = dim;

      /**
       * Dimension of the space in which this object operates.
       */
      static const unsigned int space_dimension = spacedim;

      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;

      template <typename ScalarType>
      using return_type = typename Base_t::template qp_value_type<ScalarType>;

      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      {}

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

        // return_type<ScalarType> out(fe_values.n_quadrature_points);
        // // Need to implement a
        // "get_function_hessians_from_local_dof_values()"
        // // function
        // fe_values.get_function_hessians(solution_local_dof_values,
        // // out);
        // return out;
      }
    };



    /**
     * Extract the solution third derivatives from the discretized solution
     * field.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <std::size_t solution_index, int dim, int spacedim>
    class SymbolicOp<FieldSolution<dim, spacedim>,
                     SymbolicOpCodes::third_derivative,
                     void,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpThirdDerivativeBase<FieldSolution<dim, spacedim>,
                                             solution_index>
    {
      using Base_t = SymbolicOpThirdDerivativeBase<FieldSolution<dim, spacedim>,
                                                   solution_index>;
      using typename Base_t::Op;

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = dim;

      /**
       * Dimension of the space in which this object operates.
       */
      static const unsigned int space_dimension = spacedim;

      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;

      template <typename ScalarType>
      using return_type = typename Base_t::template qp_value_type<ScalarType>;

      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      {}

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

        // return_type<ScalarType> out(fe_values.n_quadrature_points);
        // // Need to implement a
        // // "get_function_third_derivatives_from_local_dof_values()" function
        // //
        // fe_values.get_function_third_derivatives(solution_local_dof_values,
        // // out);
        // return out;
      }
    };

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



  /* --------------- Finite element spaces: Solution fields --------------- */



  template <std::size_t solution_index, int dim, int spacedim>
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



  template <std::size_t solution_index, int dim, int spacedim>
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



  template <std::size_t solution_index, int dim, int spacedim>
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



  template <std::size_t solution_index, int dim, int spacedim>
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



  template <std::size_t solution_index, int dim, int spacedim>
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



  // Unary operations

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

} // namespace WeakForms


#endif // DOXYGEN


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_spaces_h
